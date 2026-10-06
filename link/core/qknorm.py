"""Native QK-Norm capture (Qwen3) and the late-stage threshold-pruning layer.

**The attention layer itself is NOT modified.**  ``Qwen3Attention`` keeps doing
exactly what upstream transformers does:  ``q_proj -> q_norm (per-head RMSNorm)
-> RoPE``.  What this module does is *attach* two things to the layer's own
``q_norm``/``k_norm`` forwards, by patching ``module.forward`` (not a forward
hook — see :class:`QKNormStack` for why):

* **capture** — read the tensor the RMSNorm produces *before its own gain*
  (call it ``u``); ``u`` is the observation point of the L1/group-LASSO penalty
  and of every sparsity diagnostic;
* **mask** (optional, late-stage) — hard-zero the RoPE pairs of ``u_k`` whose L2
  norm is below a frozen ``tau``, so the remaining training steps run against
  keys that are genuinely sparse.

Why ``u`` and not the post-gain ``g * u``
-----------------------------------------
Two reasons, and they are the same reason:

* ``d(pen)/d(g) == 0`` identically, because the penalty reads ``u`` and ``g``
  only ever multiplies *after* the capture.
* ``sum_j ||u_j||^2 == head_dim`` **up to the eps term** (exactly
  ``head_dim * (1 - eps/(mean(x^2)+eps))``; relative error <= ~1e-4, larger for
  small activations).  So the per-pair rescaling ``u_p -> lam_p u_p`` is not
  merely penalised, it does not exist: any change to one pair's norm is paid for
  by the others.  This is the direction ``penalties.py`` spends its opening
  section closing by hand for Llama.

That is NOT the same as "no free direction exists", and an adversarial derivation
found the gap -- this paragraph used to claim the stronger thing, and it was
wrong.  Under CAUSAL attention a key at position ``s`` is attended by only
``T - s`` queries (times the number of query heads per kv head); once that count
falls below ``head_dim`` the key can be moved inside the null space of its own
attending-query matrix, so ``pen(u_k)`` falls at FIRST order while the logits move
only at second order.  Measured on a faithful small layer (pre-gain RMSNorm, GQA,
half-split RoPE), the fraction of the penalty gradient lying in that null space
is 0.21 with the causal mask and 0.0000 with it removed.  The regime is the
causal TAIL: the window is ``head_dim / (num_q_heads / num_kv_heads)`` positions
wide (32 of 1024 for Qwen3-4B), and it exists at any total length -- hidden_size
does not enter the mechanism.

Two things bound the damage, and they are why this is not the Llama failure: the
direction moves only those last ~32 positions (at most a few percent of the mean
penalty), and those are the positions the fewest future predictions depend on.
The Llama pathology was a collapse to 3% of the initial penalty; this is not that.

Note also that penalising Q -- ``tie_qk`` -- does NOT close it: the direction is
W_k-only and the tied objective decreases too (cosine between the two projected
gradients is about -0.06).  ``tie_qk`` is cheap insurance, not the fix.  The
operator that actually protects the method is the hard threshold applied in the
forward, which needs no property of the gradient at all.

The late-stage threshold-pruning layer
--------------------------------------
``PairThreshold.apply`` (see :mod:`spkv.prune`) is the honest deployment operator:
zero the RoPE pairs whose L2 norm falls below an absolute ``tau``, with ``tau``
anchored to a per-head reference so it means the same thing across models.  Top-k
is deliberately *not* used, because a fixed keep fraction manufactures sparsity
even for a model whose pair norms are flat.

Training with only a soft penalty leaves every pair a little non-zero.  This
module closes that gap: once training passes ``prune_start_frac``, the *same*
operator is applied **inside** ``k_norm.forward``, so the rest of the network
spends the remaining steps adapting to keys that are genuinely sparse.  The mask
is applied to ``u`` (pre-gain) — equivalent to zeroing the cached post-gain key,
since ``g * 0 == 0`` — and because it is the same tensor the penalty reads and
the same tensor the deployment operator thresholds, "what we penalise", "what we
prune in training" and "what we prune at inference" are one quantity, not three.
What is NOT the same between training and judging is the *number*: ``tau`` is
frozen at the transition step, while the judge re-derives it from the final
model's own distribution unless the checkpoint's frozen ``prune_tau`` is used
(the judge now evaluates BOTH: an ``as-trained`` sweep from the frozen tau, and
the alpha sweep with re-derived references).

Three design choices worth stating:

* **K only.**  Deployment sparsifies K; ``q``'s pair is multiplied against a
  zeroed key pair, so its contribution is already zero.  Masking Q as well would
  train a function deployment never runs.
* **``tau`` is frozen, the mask is not.**  ``tau`` is measured once, by
  :meth:`QKNormStack.measure_taus`, from ONE **unmasked** forward, and stored in a
  dict that outlives the per-micro-batch stacks; it is then a constant, so the
  model cannot move the goalposts.  The mask is recomputed every step from the
  current ``u``, so a pair driven back above ``tau`` returns instead of dying
  permanently.  Measuring it lazily, inside the forward that is already masking,
  is wrong twice over: layer ``i > 0`` would read a ``u_k`` computed from
  already-masked upstream keys (disagreeing with what the evaluator sees, which
  is no mask at all), and because the stack is rebuilt per micro-batch the
  "frozen" tau would in fact be re-derived several times per step.
* **The mask is a multiply, not a mutation.**  The captured tensor is a non-leaf
  intermediate with a ``grad_fn``; assigning into it in place would corrupt
  autograd.  ``u * mask.detach()`` gives surviving pairs their gradient and the
  dropped ones exactly zero.

Checkpointing
-------------
``tau`` and the transition step go into ``Checkpoint.save``'s ``extra`` dict; the
judge's ``as-trained`` sweep reads them back to reproduce the exact operator.
The FULL per-head tau vector is stored per layer (not a per-layer scalar),
because the operator is per-head.  NOTE: the repository has no training-resume
path (``Checkpoint.load`` is called only from ``spkv/eval/judge.py``), so the
schedule is meaningful within a single continuous run only.
"""

from __future__ import annotations

import contextlib
from typing import Optional

import torch
from torch import Tensor, nn

__all__ = ["QKNormState", "QKNormStack", "PruneSpec"]


# ------------------------------------------------------------------ spec

class PruneSpec:
    """Late-stage threshold pruning: when to start, and how to set tau.

    ``alpha`` is expressed in units of the per-head median pair norm, i.e. the
    same units as :meth:`spkv.core.prune.PairThreshold.refs` and therefore the
    same units the evaluation's ``eval_alphas`` use.  ``granularity`` is passed
    straight to that method; note its docstring's position that ``"head"`` is
    the honest cross-model operator and ``"token_head"`` is diagnostic only.
    """

    def __init__(self, start_frac: float, alpha: float = 1.0, granularity: str = "head"):
        if not 0.0 <= start_frac < 1.0:
            raise ValueError(
                f"prune.start_frac must be in [0, 1) (it is a fraction of the run), "
                f"got {start_frac}"
            )
        self.start_frac = float(start_frac)
        self.alpha = float(alpha)
        self.granularity = str(granularity)

    def start_step(self, total_steps: int) -> int:
        return max(1, int(total_steps * self.start_frac))

    def __repr__(self) -> str:
        return (f"PruneSpec(start_frac={self.start_frac}, alpha={self.alpha}, "
                f"granularity={self.granularity!r})")


# ------------------------------------------------------------------ state

class QKNormState:
    """Per-layer capture of the pre-gain RMSNorm outputs."""

    __slots__ = ("q", "k", "keep_frac", "tail_mass")

    def __init__(self) -> None:
        self.q: Optional[Tensor] = None      # [B, T, num_q_heads * head_dim]
        self.k: Optional[Tensor] = None      # [B, T, num_kv_heads * head_dim]
        self.keep_frac: Optional[float] = None   # realised, post-mask
        self.tail_mass: Optional[float] = None    # sum_{j not in S} ||u_j||


# ------------------------------------------------------------------ stack

class QKNormStack:
    """Patch every ``q_norm``/``k_norm`` forward to capture ``u`` (and maybe mask it).

    Usage::

        with QKNormStack(model, n_q, n_kv, hd, prune=spec) as stack:
            stack.set_step(step, total_steps)
            model(...)                       # or the manual layer loop
        # stack.state[i].q / .k are the live, grad-carrying tensors

    The patches are installed by replacing ``module.forward``, not by a forward
    hook: a hook that returns a *new* output tensor makes
    ``torch.utils.checkpoint`` save a different number of tensors on
    recomputation and aborts (observed 44 vs 34 elsewhere in this repo).  Patching
    ``forward`` also keeps parameter names — ``...q_norm.weight`` — unchanged, so
    checkpoints stay loadable.
    """

    def __init__(self, model: nn.Module, num_q_heads: int, num_kv_heads: int,
                 head_dim: int, prune: Optional[PruneSpec] = None,
                 fixed_tau: Optional[dict[int, Tensor]] = None,
                 tau_store: Optional[dict[int, Tensor]] = None,
                 measure_only: bool = False,
                 q_capture_stride: int = 1,
                 capture_q: bool = True,
                 channel_scale: Optional[dict[int, Tensor]] = None):
        if int(q_capture_stride) < 1:
            raise ValueError(f"q_capture_stride must be >= 1, got {q_capture_stride}")
        self.model = model
        self.n_q = int(num_q_heads)
        self.n_kv = int(num_kv_heads)
        self.hd = int(head_dim)
        # >1 stores only every q_capture_stride-th QUERY row of the capture.  The
        # consumers are per-token MEANS (the q-side penalty) or already strided
        # row samples (the map distillation), so the smaller capture is an
        # unbiased estimator of the same quantity -- and at 40K it is the
        # difference between 12 GiB and 1.5 GiB of graph-held tensors.  K is
        # never strided: the map needs every key.
        self._q_stride = int(q_capture_stride)
        # Consumers that need only K (the pruning judge) switch the q store off:
        # at 131072 q is 36 x [1, 32, 131072, 128] bf16 ~= 38.6 GiB per forward
        # of dead weight -- the first 128K smoke OOMed on exactly that stacked
        # under the 9.7 GiB of held K captures.  Default True: the trainer's map
        # distillation reads q.
        self._capture_q = bool(capture_q)
        # Per-layer query-saliency weights for the qstat operator: the live
        # mask keeps a pair iff ``pair_norm * channel_scale >= tau`` (see
        # PairThreshold.refs).  Empty -> the plain K-norm operator.
        self._scale: dict[int, Tensor] = (
            {int(k): v for k, v in channel_scale.items()} if channel_scale else {})
        self.prune = prune
        self.state: list[QKNormState] = []
        self._orig: list[tuple[nn.Module, object]] = []
        self._gains: dict[int, tuple[Tensor, Tensor]] = {}
        self._step = 0
        self._total = 0
        self._start_step: Optional[int] = None
        # Storing captures can be switched OFF without uninstalling the patches.
        # With gradient checkpointing the forward is RECOMPUTED during backward,
        # and the patches fire again -- overwriting ``state[i].q/.k`` with tensors
        # from the recompute graph, each of which pins that layer's recompute
        # intermediates until the whole backward ends (measured: ~50 GB at 32K,
        # 36 layers).  The patches must stay installed (a different number of
        # saved tensors aborts the recompute), but nothing reads the captures
        # after the loss is built, so the trainer switches STORING off around
        # backward().
        self._capture = True
        # ``measure_only`` runs the model UNMASKED and only records tau.  It exists
        # because measuring tau lazily -- inside the very forward that is already
        # masking -- is wrong in two distinct ways:
        #   * for layer i > 0 the tensor handed to _measure_tau is computed from
        #     hidden states that already consumed the MASKED keys of layers 0..i-1,
        #     so it is not the model's natural distribution, and it disagrees with
        #     what the evaluator measures (which sees no mask at all);
        #   * the stack is rebuilt per micro-batch, so a lazily-measured tau was
        #     re-derived on every micro-batch of every step -- never frozen, and
        #     never the same number twice.
        # Both are fixed by measuring once, from a caller-supplied tau_store that
        # outlives the per-micro-batch stacks.
        self._measure_only = bool(measure_only)
        # tau[layer] -> Tensor broadcastable onto [B, T, H_kv, P].
        self._tau: dict[int, Tensor] = tau_store if tau_store is not None else {}
        if fixed_tau is not None:
            # Evaluator path: the tau is supplied, so nothing is measured and the
            # mask is live from the first forward.  This is deliberate reuse, not
            # a parallel implementation -- the evaluator prunes with the SAME code
            # the trainer pruned with, so "what we trained against" and "what the
            # judge measures" cannot drift apart.
            self._tau.update({int(k): v for k, v in fixed_tau.items()})
            self._start_step = 0

    # ------------------------------------------------------------ primitives

    @staticmethod
    def pre_gain(hidden_states: Tensor, eps: float) -> Tensor:
        """What ``Qwen3RMSNorm`` computes *before* multiplying by its weight.

        This reproduces ``transformers.models.qwen3.modeling_qwen3.Qwen3RMSNorm.forward``
        op for op — cast to fp32, mean-square, rsqrt, cast back — so that with the
        mask disabled the patched forward is **bit-identical** to the native one.
        The cast back to ``input_dtype`` happens before the caller multiplies by the
        fp32 weight, exactly as upstream does it.
        """
        input_dtype = hidden_states.dtype
        x = hidden_states.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + eps)
        return x.to(input_dtype)

    @staticmethod
    def _as_4d(v: Tensor, head_dim: int) -> Tensor:
        """Accept either [B, T, H, head_dim] or the flattened [B, T, H*head_dim].

        The flattened layout is what every existing consumer expects, so the
        stack *stores* its captures flattened and these helpers take either form.
        """
        if v.dim() == 4:
            return v
        b, t, hd = v.shape
        if hd % head_dim:
            raise ValueError(f"last dim {hd} is not a multiple of head_dim {head_dim}")
        return v.reshape(b, t, hd // head_dim, head_dim)

    @classmethod
    def pair_norms(cls, v: Tensor, head_dim: int) -> Tensor:
        """Pair L2 norms of a per-head tensor.

        ``v``: [B, T, H, head_dim] (or flattened) -> [B, T, H, head_dim//2], with
        the HF half-split pairing ``(j, j + head_dim//2)``.  Delegates to
        :class:`spkv.core.rope.RopePairs` so the pairing is defined in exactly
        one place.
        """
        from .rope import RopePairs

        v = cls._as_4d(v, head_dim)
        b, t, h, d = v.shape
        return RopePairs(h, d).norms(v.reshape(b, t, h * d))

    @classmethod
    def keep_mask(cls, v: Tensor, head_dim: int, tau: Tensor | float,
                  scale: Tensor | None = None) -> Tensor:
        """Boolean pair mask ``||v_j|| * scale >= tau``, broadcast onto the layout.

        Returns [B, T, H, head_dim], the two halves carrying the same pair mask —
        which is what makes it a *pair* mask and not a channel mask (a channel-only
        zero does not survive RoPE; see :mod:`spkv.core.rope`).  ``scale`` is the
        optional per-pair query-saliency weight (the qstat operator); ``None``
        reproduces the plain K-norm predicate exactly.
        """
        v4 = cls._as_4d(v, head_dim)
        n = cls.pair_norms(v4, head_dim).float()
        if scale is not None:
            n = n * scale
        keep = n >= tau                                   # [B, T, H, P]
        return torch.cat([keep, keep], dim=-1)            # [B, T, H, D]

    @classmethod
    def apply_mask(cls, v: Tensor, head_dim: int, tau: Tensor | float,
                   scale: Tensor | None = None) -> Tensor:
        """Hard-zero the pairs of ``v`` below the (scaled) ``tau``. Gradient stops."""
        return v * cls.keep_mask(v, head_dim, tau, scale=scale).to(v.dtype)

    @staticmethod
    def find_qk_norms(model: nn.Module) -> list[tuple[int, nn.Module, nn.Module]]:
        """Locate ``(layer_index, q_norm, k_norm)`` without hard-coding the prefix.

        Raises if the model does not carry per-head q/k norms — this module is for
        Qwen3-style native QK-Norm, and silently doing nothing on a Llama would make
        the penalty read the wrong tensor while looking healthy.
        """
        layers = None
        for name, mod in model.named_modules():
            if name.endswith("layers") and isinstance(mod, nn.ModuleList):
                layers = list(mod)
                break
        if layers is None:
            raise RuntimeError("could not locate the decoder layer ModuleList")

        out: list[tuple[int, nn.Module, nn.Module]] = []
        for i, layer in enumerate(layers):
            attn = getattr(layer, "self_attn", None)
            qn = getattr(attn, "q_norm", None)
            kn = getattr(attn, "k_norm", None)
            if qn is None or kn is None:
                raise RuntimeError(
                    f"layer {i} has no self_attn.q_norm/.k_norm — this module only "
                    f"applies to models with native QK-Norm (Qwen3-style)."
                )
            for mod, which in ((qn, "q_norm"), (kn, "k_norm")):
                cls_name = type(mod).__name__
                if not cls_name.startswith("Qwen3"):
                    raise RuntimeError(
                        f"layer {i}: {which} is {cls_name}, not a Qwen3 RMSNorm. Only the "
                        f"Qwen3 contract is implemented (per-head normalisation over "
                        f"the last dim of a [B,T,H,D] tensor, eps in "
                        f"`variance_epsilon`, output multiplied by `weight`); Gemma3's "
                        f"layout and OLMo2's whole-hidden-size norm are refused "
                        f"rather than silently mis-measured."
                    )
            out.append((i, qn, kn))
        return out

    # ------------------------------------------------------------ schedule

    def set_step(self, step: int, total_steps: int) -> None:
        """Tell the stack where in the run we are.

        Must be called before the forward for the step.  When the transition step
        is first reached, ``tau`` is measured from that step's OWN ``u_k`` — the
        model's natural, pre-mask distribution — and then frozen.

        (In practice the trainer measures tau with :meth:`measure_taus` BEFORE any
        masking has happened, and hands the store in; this path exists for
        schedule bookkeeping and the armed-but-not-started guard.)
        """
        self._step = int(step)
        self._total = int(total_steps)
        if self.prune is not None and self._start_step is None:
            self._start_step = self.prune.start_step(total_steps)

    def _should_mask(self, idx: int) -> bool:
        """Whether this layer's K should be masked on the forward about to run.

        Masking requires a tau that ALREADY EXISTS — it is never conjured lazily
        inside the forward being masked.  See ``measure_only`` for why.

        If the schedule has armed but no tau exists, that is a caller bug, and it
        raises rather than silently training an unmasked model.  A silent no-op is
        the worst outcome here: the run produces a legitimate-looking curve, and
        the only sign that the threshold layer never ran is a keep fraction of 1.0
        that reads as "nothing was prunable".
        """
        if self._measure_only or self._start_step is None or self._step < self._start_step:
            return False
        if idx not in self._tau:
            raise RuntimeError(
                f"layer {idx}: the threshold layer is armed (step {self._step} >= "
                f"{self._start_step}) but no tau exists for it. Call "
                f"QKNormStack.measure_taus for the whole model first, or pass "
                f"fixed_tau/tau_store. Measuring tau here, inside the forward that "
                f"is already masking, would read a u_k built from masked upstream "
                f"keys and would disagree with the evaluator for every layer but "
                f"the first."
            )
        return True

    @property
    def pruning_active(self) -> bool:
        """True once the transition step has been reached (and a tau exists)."""
        return bool(self._tau) and self._step >= (self._start_step or 0)

    def set_capture(self, enabled: bool) -> None:
        """Toggle capture STORING (the patches stay installed either way).

        The trainer switches this off for the backward pass: with gradient
        checkpointing the layer forwards run again during backward, and storing
        those captures pins every layer's recompute graph until the backward
        ends.  Callers must have finished reading the captures (the loss is
        already built) before switching it off.
        """
        self._capture = bool(enabled)

    # ------------------------------------------------------------ capture

    def _stats(self, idx: int, u_pre: Tensor) -> None:
        """Diagnostics for the step, measured on the **pre-mask** tensor.

        The tail mass is by definition the mass the mask THROWS AWAY, so measuring
        it after masking would read a structural zero: the dropped pairs have norm
        exactly 0 there, and ``n * (1 - keep)`` is 0 for the kept ones too.  It has
        to be read from the tensor as the model produced it.
        """
        with torch.no_grad():
            n = self.pair_norms(u_pre.detach(), self.hd).float()      # [B,T,H,P]
            st = self.state[idx]
            if idx in self._tau:
                keep = (n >= self._tau[idx]).float()
                st.keep_frac = float(keep.mean())
                st.tail_mass = float((n * (1.0 - keep)).sum(-1).mean())
            else:
                st.keep_frac = 1.0
                st.tail_mass = 0.0

    def _measure_tau(self, idx: int, u: Tensor) -> None:
        """Freeze tau for this layer from the CURRENT (unmasked) u_k.

        Uses :meth:`spkv.core.prune.PairThreshold.refs` — literally the
        deployment/eval function, not a reimplementation — so the training
        threshold and the judged threshold are the same function by construction.
        """
        from .prune import PairThreshold

        b, t, h, d = u.shape
        flat = u.detach().float().reshape(b, t, h * d)
        ref = PairThreshold(h, d).refs(flat, alpha=self.prune.alpha,
                                       granularity=self.prune.granularity)
        self._tau[idx] = ref.to(u.device)

    def _patch(self, idx: int, module: nn.Module, which: str, attr: str) -> None:
        original = module.forward
        eps = getattr(module, "variance_epsilon", 1e-6)
        gain = module.weight
        state = self.state[idx]
        stack = self

        def forward(x, _orig=original, _eps=eps, _gain=gain, _attr=attr, _i=idx):
            u = stack.pre_gain(x, _eps)
            if _attr == "k":
                if stack._measure_only:
                    stack._measure_tau(_i, u)
                live = stack._should_mask(_i)
                # stats read the tensor as the MODEL produced it, before the mask
                stack._stats(_i, u)
                if live:
                    u = stack.apply_mask(u, stack.hd, stack._tau[_i],
                                         stack._scale.get(_i))
            # Stored FLATTENED, [B, T, H*head_dim]: that is the layout every
            # existing consumer of these captures speaks, so nothing downstream
            # has to learn about the head axis.  It is a view of the tensor
            # attention consumes, so the gradient path is unchanged.
            if stack._capture and (_attr == "k" or stack._capture_q):
                b, t, h, d = u.shape
                stored = u.reshape(b, t, h * d)
                if _attr == "q" and stack._q_stride > 1:
                    # contiguous() matters: a bare strided view would keep the
                    # full-T storage alive through backward, which is the very
                    # memory this stride exists to release.
                    stored = stored[:, :: stack._q_stride].contiguous()
                setattr(stack.state[_i], _attr, stored)
            return _gain * u

        module.forward = forward
        self._orig.append((module, original))

    def deployed_qk(self) -> dict[int, tuple[Tensor, Tensor]]:
        """The tensors attention actually consumes, per layer: ``weight * u``.

        Pre-RoPE (the caller applies RoPE), post-gain, and — when the mask is
        live — post-mask, because the mask is applied to ``u`` before the gain.
        This is exactly ``Qwen3Attention``'s q/k at the point before
        ``apply_rotary_pos_emb``, which is what the attention-MAP distillation
        must be built from: a map built from the RAW ``q_proj``/``k_proj``
        outputs is a proxy map the model never computes.
        """
        out: dict[int, tuple[Tensor, Tensor]] = {}
        for i, st in enumerate(self.state):
            if st.q is None or st.k is None or i not in self._gains:
                continue
            gq, gk = self._gains[i]
            out[i] = (self._apply_gain(st.q, gq), self._apply_gain(st.k, gk))
        return out

    @classmethod
    def _apply_gain(cls, u_flat: Tensor, gain: Tensor) -> Tensor:
        b, t, hd = u_flat.shape
        d = gain.shape[-1]
        return (u_flat.view(b, t, hd // d, d) * gain.to(u_flat.dtype)).reshape(b, t, hd)

    # ------------------------------------------------------------ lifecycle

    def __enter__(self) -> "QKNormStack":
        found = self.find_qk_norms(self.model)
        self.state = [QKNormState() for _ in found]
        self._gains = {idx: (qn.weight, kn.weight) for idx, qn, kn in found}
        for idx, qn, kn in found:
            self._patch(idx, qn, "q_norm", "q")
            self._patch(idx, kn, "k_norm", "k")
        return self

    def __exit__(self, *exc) -> None:
        # Unsubscribe the patched forwards only.  Do NOT clear ``self.state``:
        # the caller builds the penalty from the captured q/k AFTER the with-block,
        # and clearing it would silently make every penalty term zero while the
        # run looked healthy.
        for module, original in self._orig:
            module.forward = original
        self._orig.clear()

    # ------------------------------------------------------------ reporting

    def report(self) -> dict[str, float]:
        """Aggregate diagnostics for the step just run."""
        fr = [s.keep_frac for s in self.state if s.keep_frac is not None]
        tm = [s.tail_mass for s in self.state if s.tail_mass is not None]
        out: dict[str, float] = {}
        if fr:
            out["prune_keep_frac"] = sum(fr) / len(fr)
        if tm:
            out["prune_tail_mass"] = sum(tm) / len(tm)
        if self._tau:
            # Median over ALL per-head values, not head 0: with granularity
            # "head" each entry is already a [1,1,H,1] vector, and collapsing it
            # hid the very per-head structure the operator is about.
            taus = torch.cat([t.reshape(-1).float() for t in self._tau.values()])
            out["prune_tau_median"] = float(taus.median())
        return out

    def checkpoint_extra(self) -> dict:
        """Payload additions so the judge can reproduce the operator."""
        if self.prune is None:
            return {}
        return {
            "prune_start_frac": self.prune.start_frac,
            "prune_start_step": self._start_step,
            "prune_alpha": self.prune.alpha,
            "prune_granularity": self.prune.granularity,
            "prune_tau": {int(k): [float(x) for x in v.reshape(-1)] for k, v in self._tau.items()},
        }

    # ------------------------------------------------------------ measuring

    @classmethod
    @torch.no_grad()
    def measure_taus(cls, model: nn.Module, num_q_heads: int, num_kv_heads: int,
                     head_dim: int, ids: Tensor, spec: PruneSpec,
                     amp=None) -> dict[int, Tensor]:
        """Freeze ``tau`` per layer from ONE **unmasked** forward.

        Every layer's tau is read in the same forward, with no mask applied
        anywhere, so each one sees the model's natural k distribution — the same
        tensor the evaluator's ``capture_k`` sees, measured with the same function.
        That correspondence is the whole point: measuring lazily inside the
        forward that is already masking gives a different answer for every layer
        but the first, because layer i's k is then computed from already-masked
        upstream keys.

        Returns a store to hand to every :class:`QKNormStack` used for the rest of
        the run (via ``tau_store``).  It is a plain dict, so it deliberately
        outlives the per-micro-batch stacks.
        """
        store: dict[int, Tensor] = {}
        stack = cls(model, num_q_heads, num_kv_heads, head_dim,
                    prune=spec, tau_store=store, measure_only=True)
        ctx = amp if amp is not None else contextlib.nullcontext()
        with stack, ctx:
            model.model(input_ids=ids, use_cache=False)
        n_layers = len(stack.state)
        if len(store) != n_layers:
            raise RuntimeError(
                f"measured tau for {len(store)} of {n_layers} layers; every layer must "
                f"have one or the mask would silently skip the rest"
            )
        return store
