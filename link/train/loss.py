"""Loss assembly: the sparsity penalty, and the total objective.

The penalty is a RoPE-pair group LASSO on the NORMALISED key (and, by default,
query) vectors — the tensors ``QKNormStack`` captures from the model's own
``q_norm``/``k_norm``.  Why Q is penalised too (the single most important design
decision here) is argued in the module that used to carry it, and the short form
is: on Llama a per-pair rescaling ``q_p -> lam_p q_p``, ``k_p -> k_p / lam_p``
leaves every attention score bit-identical while scaling ``||k_p||``, so a
penalty on K alone exerts zero first-order force on the relative pair-norm
distribution.  Under native QK-Norm the RMSNorm pins the head norm ("up to the
eps term"), so that particular free direction is closed by construction; the
tied penalty is kept as cheap insurance (see ``spkv/core/qknorm.py`` for the
residual W_k-only escape it does NOT close).

Three conventions that once disagreed across files, now pinned in one place:

* ``norm``   — "raw" (lambda is the physical LASSO weight) vs "step0" (divide
               each layer by its step-0 value).  BOTH are logged every step.
* ``reduce`` — "sum" (lambda weights EACH layer) vs "mean" (divides the force by
               the layer count).  "sum" is the default and the one the
               single-layer sweeps used, so a lambda transfers between paths.
* ``ramp``   — 0 -> 1 over the first ``ramp_frac`` of the run.

Chunked CE lives in ``spkv.core.ce`` (the judge scores with the same function)
and is issued here when CE is active.
"""

from __future__ import annotations

from typing import Mapping, Optional

from torch import Tensor, nn
import torch

from ..core.ce import ChunkedCrossEntropy
from ..core.rope import RopePairs
from .distill import Distiller

__all__ = ["GroupLassoRope", "PenaltyNormalizer", "SpKVLoss", "layer_penalty"]


def layer_penalty(fn: "GroupLassoRope", x: Tensor) -> Tensor:
    """``fn(x)``, computed inside a CHECKPOINTED region when ``x`` carries grad.

    One layer's group-LASSO graph is O(T x hidden) -- the pair norms and their
    intermediates at the full sequence -- and it stays reachable from the running
    total until the joint backward.  Across 36 layers that retention alone was
    measured to saturate a 72 GB card at 32K (a penalty-only step peaked at
    69.9 GB).  Checkpointing keeps one scalar per layer instead; the norm is
    recomputed during backward, which is trivial next to the model's own
    recompute and the map/KD chunks.

    Applied by the TRAINER's per-layer loop (``harness._step``); the scalar it
    returns is identical to ``fn(x)`` -- pinned by test.
    """
    from functools import partial

    from torch.utils.checkpoint import checkpoint

    if torch.is_grad_enabled() and x.requires_grad:
        return checkpoint(partial(_apply, fn), x, use_reentrant=False)
    return fn(x)


def _apply(fn: "GroupLassoRope", x: Tensor) -> Tensor:
    return fn(x)


class GroupLassoRope:
    """Group-LASSO over RoPE pairs: mean over tokens of the sum of pair L2 norms.

    Bound to one head geometry (the K and Q sides have different head counts).
    ``x``: [B, T, num_heads * head_dim] — a raw projection output OR the
    normalised capture; the caller decides which parametrisation is penalised,
    and it must be the same one every consumer (normalizer, diagnostics, mask)
    sees.  Pre-RoPE is exactly equivalent to the post-RoPE cached tensor, since
    RoPE preserves each pair's norm.

    ``mean`` (not ``sum``) makes the penalty invariant to batch size and token
    count, which is what lets a lambda tuned at 4K transfer to 128K.
    """

    def __init__(self, num_heads: int, head_dim: int):
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.pairs = RopePairs(self.num_heads, self.head_dim)

    def __call__(self, x: Tensor) -> Tensor:
        n = self.pairs.norms(x)  # [B, T, H, d/2]
        return n.sum(dim=-1).mean()


class PenaltyNormalizer(nn.Module):
    """Per-layer step-0 reference normalisation for the penalties.

    At step 0 the student equals the teacher, so each layer's penalty value
    ``P^(l),0`` is a natural, layer-specific scale.  Dividing by it makes every
    term O(1) at init: layers contribute comparably (deep layers do not dominate)
    and lambda becomes dimensionless and transferable across layers, models and
    sequence lengths.

    Reference values are measured once (see ``set_reference``) and frozen as
    buffers.  This is preferred over dividing by the tensor's own detached mean,
    which has a perverse gradient (the model could reduce the penalty by
    inflating the denominator rather than by sparsifying).
    """

    def __init__(self, num_layers: int, names: tuple[str, ...] = ("q", "k"), eps: float = 1e-12):
        super().__init__()
        self.names = tuple(names)
        self.eps = float(eps)
        self.num_layers = int(num_layers)
        for name in self.names:
            self.register_buffer(f"{name}_ref", torch.ones(self.num_layers), persistent=True)

    @torch.no_grad()
    def set_reference(self, name: str, layer: int, value: Tensor) -> None:
        buf = getattr(self, f"{name}_ref")
        buf[layer] = float(value.detach())

    def normalize(self, name: str, layer: int, value: Tensor) -> Tensor:
        ref = getattr(self, f"{name}_ref")[layer]
        return value / (ref + self.eps)


class SpKVLoss:
    """Assembles the total objective and records each term for logging.

    ``L = ce_weight * CE + lam_d * attn_output + lam_kd * KD + lam_map * map + pen``

    CE is optional and OFF in the distillation-only configuration.  Dropping it
    is safe *because* every frozen module is byte-identical to the teacher's:
    only q/k differ, and they enter the residual stream solely through
    ``attn_out = o_proj(a V)``.  So matching attention at every layer already
    forces every hidden state, and hence every logit, to match — CE would be a
    second gradient path to a quantity the attention term already pins.
    """

    def __init__(self, penalty_cfg, train_cfg, distiller: Optional[Distiller] = None):
        self.cfg = penalty_cfg.resolved()
        self.train_cfg = train_cfg
        self.distiller = distiller
        self.ce = ChunkedCrossEntropy(train_cfg.ce_chunk)

    @staticmethod
    def _ramp(step: int, max_steps: int, frac: float) -> float:
        """Lambda ramp: 0 -> 1 over the first ``frac`` of training.

        Turning group-LASSO on at step 0, while attention geometry is still
        settling, tends to collapse pairs before the CE has stabilised and is
        hard to undo.
        """
        if frac <= 0:
            return 1.0
        k = max(1, int(max_steps * frac))
        return min(1.0, (step + 1) / k)

    def __call__(
        self,
        *,
        hidden: Tensor | None = None,
        input_ids: Tensor | None = None,
        lm_head=None,
        student_attn: Optional[Mapping[int, Tensor]] = None,
        teacher_attn: Optional[Mapping[int, Tensor]] = None,
        student_hidden: Tensor | None = None,
        teacher_hidden: Tensor | None = None,
        penalties: Mapping[str, Mapping[int, Tensor]] = (),
        normalizer=None,
        step: int = 0,
        total_steps: int | None = None,
        map_distill: Tensor | None = None,
    ) -> tuple[Tensor, dict[str, float]]:
        """Assemble the total loss.

        ``student_hidden``/``teacher_hidden`` feed the KD term: the LM head is
        applied CHUNK-WISE (``Distiller.logit_kd_chunked``), never through a
        materialised full-logits tensor (5 GB per side at 8K, OOM at 40K).

        The per-term gradient probe (harness ``_probe_terms_offline``) builds its
        own short passes rather than reading this graph: over a checkpointed
        model, ``autograd.grad(..., retain_graph=True)`` per term re-runs and
        retains every layer's recomputation, which OOMed a 72 GB card.
        """
        # dtype/device anchor.  ``hidden`` is only present when CE is active, so a
        # distillation-only run anchors on a captured attention tensor instead.
        anchor = hidden
        if anchor is None and student_attn:
            anchor = next(iter(student_attn.values()))
        if anchor is None:
            raise ValueError(
                "SpKVLoss needs at least one live tensor to anchor dtype/device; "
                "got neither `hidden` nor `student_attn`"
            )
        zero = anchor.new_zeros(())

        # A term that is REQUESTED but whose tensors are missing must be loud:
        # silently contributing 0.0 produces a legitimate-looking curve for a run
        # that is not training what its config says.
        if self.train_cfg.attn_distill and (not student_attn or not teacher_attn):
            raise ValueError(
                "attn_distill is on but student/teacher attention outputs are missing — "
                "the term would silently be zero.")
        if self.train_cfg.lam_kd > 0 and (student_hidden is None or teacher_hidden is None):
            raise ValueError(
                "lam_kd > 0 but student/teacher hidden states are missing — the KD "
                "term would silently be zero.")
        if self.train_cfg.attn_map_distill and map_distill is None:
            raise ValueError(
                "attn_map_distill is on but no map term was passed — it would "
                "silently be zero.")

        ce = zero
        if self.train_cfg.ce_weight > 0.0 and hidden is not None:
            ce = self.ce(hidden, input_ids, lm_head)

        dist = zero
        if self.train_cfg.attn_distill:
            dist = self.distiller.attn_output(student_attn, teacher_attn)

        kd = zero
        if self.train_cfg.lam_kd > 0:
            kd = self.distiller.logit_kd_chunked(
                student_hidden, teacher_hidden, lm_head, chunk=self.train_cfg.ce_chunk)

        # Ramp against the number of steps actually being run, not the config's
        # max_steps, or a short run never reaches full lambda.
        horizon = total_steps if total_steps is not None else self.train_cfg.max_steps
        scale = self._ramp(step, horizon, self.cfg.ramp_frac)
        # Built from the anchor-derived ``zero`` (NOT ``hidden``): the documented
        # distillation-only path passes hidden=None.
        pen = zero
        parts: dict[str, float] = {}
        # ``norm`` decides only which of the two values is MULTIPLIED by lambda.
        # Both are logged every step regardless, because the single largest source
        # of inconsistent results in this project was two harnesses silently using
        # different conventions and calling the knob by the same name.
        use_norm = self.cfg.norm == "step0"
        for name, weight in (("q", self.cfg.lam_q), ("k", self.cfg.lam_k)):
            if name not in penalties:
                continue
            raw_acc = zero
            norm_acc = zero
            vals = penalties[name]
            for layer, val in vals.items():
                raw_acc = raw_acc + val
                norm_acc = norm_acc + (
                    normalizer.normalize(name, layer, val) if normalizer is not None else val
                )
            n_layers = max(len(vals), 1)
            # ALWAYS logged as the mean over layers, whatever ``reduce`` does to the
            # loss: otherwise the log's units would shift by the layer count when
            # someone flips `reduce`, and two runs could not be compared.
            per_layer_raw = raw_acc / n_layers
            per_layer_norm = norm_acc / n_layers
            used = per_layer_norm if use_norm else per_layer_raw
            if weight != 0.0:
                # "sum" weights EACH layer by lambda; "mean" divides that force by
                # the layer count. See PenaltyCfg.reduce.
                contribution = used * n_layers if self.cfg.reduce == "sum" else used
                pen = pen + weight * scale * contribution
            parts[f"pen_{name}_raw"] = float(per_layer_raw.detach())
            parts[f"pen_{name}"] = float(used.detach())
        parts["pen_norm_mode"] = 1.0 if use_norm else 0.0

        mdist = map_distill if map_distill is not None else zero
        # Weighted contributions, keyed by the canonical term names.  Built in one
        # place so the total, the logged floats and the gradient probe can never
        # disagree about what "the map term" weighs.
        contrib = {"ce": self.train_cfg.ce_weight * ce,
                   "output": self.cfg.lam_d * dist,
                   "kd": self.train_cfg.lam_kd * kd,
                   "map": self.train_cfg.lam_map * mdist,
                   "pen": pen}
        total = contrib["ce"] + contrib["output"] + contrib["kd"] + contrib["map"] + pen
        parts.update(
            ce=float(ce.detach()),
            distill=float(dist.detach()),
            kd=float(kd.detach()),
            map_distill=float(mdist.detach()),
            total=float(total.detach()),
            penalty=float(pen.detach()),
        )

        return total, parts
