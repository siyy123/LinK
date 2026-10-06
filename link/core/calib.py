"""Deployment-time calibration of the pruning threshold.

No checkpoint in this repository carries a frozen ``prune_tau`` -- the save
payloads hold ``model/step/eval_ce/base_ce/k_scale_mode`` only -- so the operator
a deployment runs must be RE-DERIVED from the final model's own pair-norm
distribution.  That derivation is the judge's re-derived path, moved here with
the statements unchanged (same capture, same per-layer fp32 conversion, same
``alpha=1.0`` call into :meth:`PairThreshold.refs`) so the judge and the
inference stack cannot drift apart.  ``tests/test_calib.py`` pins the recipe
against an inline copy of the original form.

Semantics (do NOT "fix" these here): **threshold, not top-k**.  One ``tau`` per
layer (``granularity="layer"``) or per kv-head (``"head"``) is compared against
every (token, head, pair) norm; the kept fraction is an OUTCOME of the model's
own distribution (see :mod:`spkv.core.prune`'s header).  ``"token_head"`` is
diagnostic-only -- its reference is length-coupled and cannot be deployed to
generation.  ``calibrate_to_keep`` bisects alpha so a threshold HITS a target
kept fraction; at no point does anything select pairs by rank.
"""

from __future__ import annotations

import contextlib
import dataclasses
import math
from typing import Optional

import torch
from torch import Tensor

from .prune import PairThreshold
from .qknorm import QKNormStack
from .rope import RopePairs

__all__ = ["CalibReport", "capture_k", "q_pair_scale", "reference_taus",
           "tau_for_alpha", "calibrate", "calibrate_to_keep"]


def _full_attention_mask(ids: Tensor) -> Tensor:
    """Dense all-ones 2D mask: keeps transformers from materialising a causal mask.

    ``Qwen3Model.forward`` (transformers 4.57) always calls ``create_causal_mask``,
    and with ``attention_mask=None`` + non-None position_ids + no cache it takes
    the PACKED-sequence path -- whose ``and_masks`` wrapper forces a dense
    ``[1, 1, T, T]`` bool: 16 GiB at T=131072 (this was the 128K eval OOM), and
    4 GiB plus a 4x KV repeat (``enable_gqa`` is disabled whenever a mask is
    passed) even at 64K.  A dense all-ones mask costs T bytes and disables that
    path: ``_ignore_causal_mask_sdpa`` then recognises the plain-causal case
    (q_len == kv_len, no padding) and returns None, so SDPA runs with
    ``is_causal=True`` and GQA.  Logits equal the no-mask path within fp noise;
    locked by tests/test_eval_causal_mask.py.

    PREFILL ONLY: it is valid exactly when ``q_len == kv_len`` -- never pass it
    to a cached decode step (generation passes the tokenizer's own 2D mask).
    """
    return torch.ones(ids.shape, dtype=torch.bool, device=ids.device)


def capture_k(model, ids: Tensor, n_q: int, n_kv: int, hd: int,
              amp=None, sdpa=None) -> dict[int, Tensor]:
    """Per-layer K captures, in the parametrisation the mask will threshold.

    One UNMASKED prefill forward (``use_cache=False``); ``amp``/``sdpa`` are
    optional caller-supplied context managers (a ``Runtime``'s), so this module
    needs no Runtime of its own.  ``capture_q=False``: only K is read (a q
    capture at 131072 is ~39 GiB of dead weight per forward).
    """
    stack = QKNormStack(model, n_q, n_kv, hd, capture_q=False)
    with contextlib.ExitStack() as es:
        es.enter_context(stack)
        if amp is not None:
            es.enter_context(amp)
        if sdpa is not None:
            es.enter_context(sdpa)
        with torch.no_grad():
            model.model(input_ids=ids, attention_mask=_full_attention_mask(ids),
                        use_cache=False)
    return {i: s.k.detach() for i, s in enumerate(stack.state) if s.k is not None}


def q_pair_scale(model, ids: Tensor, n_q: int, n_kv: int, hd: int,
                 amp=None, sdpa=None) -> dict[int, Tensor]:
    """Per-(layer, kv-head, pair) query-saliency weights (the qstat operator).

    Specification (agreed 2026-10-06): the statistic is the pair's L2 norm on
    the QUERY side -- the same RoPE-pair functional the K side uses -- taken as
    a running (cumulative) mean over the calibration tokens, computed per query
    head and AVERAGED over the query heads sharing a kv head (GQA grouping).
    Returned tensors broadcast against the pair-norm layout ``[B, T, H_kv, P]``;
    the deployment predicate is ``||k_pair|| * scale >= tau``.

    One unmasked prefill forward with the Q capture ON (q at 8K is
    1 x 8192 x 32 x 128 bf16 = 64 MiB per layer; released per layer below).
    """
    reps = n_q // n_kv
    if reps * n_kv != n_q:
        raise ValueError(f"n_q={n_q} is not a multiple of n_kv={n_kv}")
    stack = QKNormStack(model, n_q, n_kv, hd, capture_q=True)
    with contextlib.ExitStack() as es:
        es.enter_context(stack)
        if amp is not None:
            es.enter_context(amp)
        if sdpa is not None:
            es.enter_context(sdpa)
        with torch.no_grad():
            model.model(input_ids=ids, attention_mask=_full_attention_mask(ids),
                        use_cache=False)

    pairs = RopePairs(n_q, hd)
    scales: dict[int, Tensor] = {}
    for i, st in enumerate(stack.state):
        if st.q is None:
            raise RuntimeError(f"q capture missing for layer {i} (capture_q?)")
        n = pairs.norms(st.q).float()             # [B, T, n_q, P]
        s = n.mean(dim=(0, 1))                    # [n_q, P] cumulative mean
        s = s.reshape(n_kv, reps, -1).mean(dim=1)  # [n_kv, P] group q-heads
        scales[i] = s.view(1, 1, n_kv, -1)
        del n
    return scales


def reference_taus(caps: dict[int, Tensor], n_kv: int, hd: int,
                   granularities: tuple[str, ...] = ("layer", "head"),
                   scales: dict[int, Tensor] | None = None,
                   ) -> dict[str, list[Tensor]]:
    """Per-layer references, one list per granularity -- the judge's recipe.

    fp32, one layer at a time (upcasting every layer's capture at once is
    36 x 131072 x 1024 x 4 B = 19.3 GiB at 128K), ``alpha=1.0``: alpha is
    applied later, when the deployment taus are built.  ``scales`` switches the
    statistic to the qstat operator (median of ``norm * scale``).
    """
    pruner = PairThreshold(n_kv, hd)
    out: dict[str, list[Tensor]] = {g: [] for g in granularities}
    for i in sorted(caps):
        c = caps[i].float()
        s = (scales or {}).get(i)
        for g in granularities:
            out[g].append(pruner.refs(c, alpha=1.0, granularity=g, scale=s))
        del c
    return out


def tau_for_alpha(refs: list[Tensor], alpha: float) -> dict[int, Tensor]:
    """``{layer: alpha * ref}`` for ``QKNormStack(fixed_tau=...)``."""
    return {i: r * alpha for i, r in enumerate(refs)}


@dataclasses.dataclass
class CalibReport:
    """What a calibration measured -- the deployed operator's provenance."""

    granularity: str
    alpha: float
    doc_tokens: int
    kept_frac: float                      # pooled over layers, natural capture
    per_layer_kept: dict[int, float]
    # Per-layer reference as a SCALAR for the scale report: the ref itself for
    # granularity="layer", its per-head MEAN for "head" (the per-head vector is
    # summarised by head_ref_spread instead).
    layer_refs: list[float]
    head_ref_spread: Optional[float]      # worst-layer per-head ref max/min

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def _pooled_kept(caps: dict[int, Tensor], refs: list[Tensor],
                 pruner: PairThreshold, alpha: float, stride: int,
                 scales: dict[int, Tensor] | None = None) -> float:
    vals = [pruner.kept_frac(caps[i][:, ::stride], refs[i] * alpha,
                             scale=(scales or {}).get(i))
            for i in sorted(caps)]
    return sum(vals) / max(len(vals), 1)


def _report(caps: dict[int, Tensor], refs: list[Tensor], granularity: str,
            alpha: float, doc_tokens: int, pruner: PairThreshold,
            scales: dict[int, Tensor] | None = None) -> CalibReport:
    per_layer = {i: pruner.kept_frac(caps[i], refs[i] * alpha,
                                     scale=(scales or {}).get(i))
                 for i in sorted(caps)}
    pooled = sum(per_layer.values()) / max(len(per_layer), 1)
    spread = None
    if granularity == "head":
        spreads = [float(r.reshape(-1).max() / r.reshape(-1).min().clamp_min(1e-12))
                   for r in refs]
        spread = max(spreads)
        layer_refs = [float(r.float().mean()) for r in refs]
    else:
        layer_refs = [float(r) for r in refs]
    return CalibReport(granularity=granularity, alpha=float(alpha),
                       doc_tokens=int(doc_tokens), kept_frac=float(pooled),
                       per_layer_kept={i: float(v) for i, v in per_layer.items()},
                       layer_refs=layer_refs,
                       head_ref_spread=spread)


def calibrate(model, ids: Tensor, n_q: int, n_kv: int, hd: int, *,
              granularity: str = "layer", alpha: float = 1.0,
              scales: dict[int, Tensor] | None = None,
              amp=None, sdpa=None) -> tuple[dict[int, Tensor], CalibReport]:
    """Capture + references + ``tau = alpha * ref`` for one granularity."""
    caps = capture_k(model, ids, n_q, n_kv, hd, amp=amp, sdpa=sdpa)
    refs = reference_taus(caps, n_kv, hd, (granularity,), scales=scales)[granularity]
    report = _report(caps, refs, granularity, alpha, ids.shape[1],
                     PairThreshold(n_kv, hd), scales=scales)
    return tau_for_alpha(refs, alpha), report


def calibrate_to_keep(model, ids: Tensor, n_q: int, n_kv: int, hd: int,
                      target_keep: float, *, granularity: str = "layer",
                      alpha_bounds: tuple[float, float] = (1e-3, 16.0),
                      iters: int = 24, tol: float = 0.02,
                      sample_stride: int = 4,
                      scales: dict[int, Tensor] | None = None,
                      amp=None, sdpa=None
                      ) -> tuple[dict[int, Tensor], CalibReport, float]:
    """Bisect alpha so the energy THRESHOLD hits ``target_keep`` (an outcome).

    Every bisection point is still "one tau vs every pair's norm" -- the target
    only chooses WHICH threshold to deploy, never selects pairs by rank.
    Returns ``(taus, report, alpha)``.  ``target_keep >= 1.0`` installs no mask
    at all (``({}, report, 0.0)``): "keep everything" must be exactly the plain
    model, not a threshold that happens to delete nothing.
    """
    if target_keep >= 1.0:
        report = CalibReport(granularity=granularity, alpha=0.0,
                             doc_tokens=int(ids.shape[1]), kept_frac=1.0,
                             per_layer_kept={}, layer_refs=[],
                             head_ref_spread=None)
        return {}, report, 0.0

    caps = capture_k(model, ids, n_q, n_kv, hd, amp=amp, sdpa=sdpa)
    refs = reference_taus(caps, n_kv, hd, (granularity,), scales=scales)[granularity]
    pruner = PairThreshold(n_kv, hd)

    # kept(alpha) is DECREASING in alpha; the bounds span ~4 orders of
    # magnitude, so bisect in log-space.
    lo, hi = (float(alpha_bounds[0]), float(alpha_bounds[1]))
    for _ in range(max(1, iters)):
        mid = math.sqrt(lo * hi)
        if _pooled_kept(caps, refs, pruner, mid, sample_stride,
                        scales=scales) > target_keep:
            lo = mid          # keeping too much -> raise the threshold
        else:
            hi = mid
    alpha = math.sqrt(lo * hi)
    kept = _pooled_kept(caps, refs, pruner, alpha, 1, scales=scales)
    if abs(kept - target_keep) > tol:
        print(f"[calib] warning: target keep {target_keep:.3f} not met: got "
              f"{kept:.4f} at alpha={alpha:.4g} (alpha_bounds={alpha_bounds})")
    report = _report(caps, refs, granularity, alpha, ids.shape[1], pruner,
                     scales=scales)
    return tau_for_alpha(refs, alpha), report, alpha
