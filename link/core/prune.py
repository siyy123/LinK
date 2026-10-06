"""Inference-time dynamic sparsification of the KV cache.

Phase 1 sparsifies K only, so pruning happens at **RoPE-pair** granularity: the
deployment operator (:class:`PairThreshold`) drops a pair when its L2 norm falls
below an absolute threshold, and the achieved sparsity is an OUTCOME of the
model's own pair-norm distribution.  Top-k is deliberately not the deployment
operator — a fixed keep fraction manufactures sparsity even for a model whose
pair norms are flat; it survives only as the test-only reference operator
(:class:`TopKReference`) that pins the pairing / zero-survival contracts.

Pair granularity is not a choice, it is forced: single-channel zeros do NOT
survive RoPE (the rotation mixes the two channels of a pair), so only whole-pair
zeros exist in the cache.  This is also the operator that exactly matches the
training-time penalty, which is why the pair view here is
``rope.RopePairs.pair_view`` — that correspondence is the guarantee that the
unit we train against is the unit we prune.
"""

from __future__ import annotations

import torch
from torch import Tensor

from .rope import RopePairs

__all__ = ["REF_GRANULARITIES", "PairThreshold", "TopKReference"]

# How a layer's pair-norm distribution is reduced to a threshold reference.
#
#   layer       one median pooled over every (token, head, pair) of the layer.
#               The historical behaviour.  It removes only a LAYER-wide scale, so
#               it leaves the per-head scale in place -- and per-head scale is a
#               pure nuisance here.  Thresholding a mixture of heads with one
#               pooled median is what lets a single threshold delete the whole of
#               a low-norm head while never being able to do so to a normalised
#               (native-QK-Norm) model, whose heads all have norm ~sqrt(head_dim).
#               That asymmetry is a property of the OPERATOR, not of the model.
#   head        one reference per kv head (a constant, so it stays a plain
#               offline threshold -- no per-token statistics at inference).
#   token_head  one reference per (token, head); a within-head concentration
#               probe.  It over-normalises: it divides away the per-token
#               magnitude collapse that is the method's own claim, and its keep
#               fraction at alpha=1 is ~0.515 for EVERY model, so it barely
#               discriminates.  Diagnostic only, never the headline operator.
REF_GRANULARITIES = ("layer", "head", "token_head")


def _write_pairs_back(kept: Tensor, x_template: Tensor, num_heads: int,
                      head_dim: int) -> Tensor:
    """[B,T,H,P,2] masked pairs -> [B,T,H*D], without the wrong ``view``.

    ``kept`` is the output of ``RopePairs.pair_view`` times a keep mask.  Undoing the
    stack with ``.view(..., 2, half)`` would silently produce the wrong
    permutation (the stacked axis is not a reshape of contiguous memory), so the
    halves are indexed explicitly.  Both operators below route through this one
    function, so the permutation is written down exactly once.
    """
    b, t, _ = x_template.shape
    half = head_dim // 2
    out = torch.empty_like(x_template).view(b, t, num_heads, head_dim)
    out[..., :half] = kept[..., 0]
    out[..., half:] = kept[..., 1]
    return out.reshape(b, t, num_heads * head_dim)


class PairThreshold:
    """The deployment operator: zero RoPE pairs whose L2 norm is below ``tau``.

    ``tau`` is absolute, so for a fair cross-model comparison it must be scaled by
    each model's own typical pair norm — that is what :meth:`refs` computes, in
    the same units the training-time ``prune_alpha`` uses (same function, called
    from :meth:`spkv.core.qknorm.QKNormStack._measure_tau`), so the operator
    trained against and the operator judged cannot be different implementations.

    ``tau`` may be a scalar or a tensor broadcasting against the pair-norm tensor
    ``[B, T, H, P]`` — :meth:`refs` shapes are chosen so that this comparison is
    always against the HEAD axis and never against the pair axis.
    """

    def __init__(self, num_heads: int, head_dim: int):
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.pairs = RopePairs(self.num_heads, self.head_dim)

    # --------------------------------------------------------------- refs

    def refs(self, x: Tensor, alpha: float = 1.0, granularity: str = "layer",
             reduce: str = "median", scale: Tensor | None = None) -> Tensor:
        """Absolute ``tau`` for a key tensor, relative to its own distribution.

        ``scale`` (optional, broadcastable to ``[B, T, H, P]``): query-saliency
        weights -- the statistic becomes the median of ``pair_norm * scale``,
        and the retained set under a fixed tau follows ``pair_norm * scale >=
        tau``, i.e. an effective per-pair threshold ``tau / scale``.  This is
        the historic query-statistics operator (channels queries exercise less
        get pruned harder); ``scale=None`` is bit-identical to the plain
        K-norm operator this repository trains against.

        It MUST be called on the tensor in the same parametrisation that
        :meth:`apply` will later threshold.  The evaluation path used to measure
        the reference on the RAW ``k_proj`` output and then threshold a
        differently-scaled tensor, so ``alpha`` was expressed in a different unit
        from the quantity actually being cut, by an unknown factor.

        Returns a tensor shaped to broadcast against the pair-norm tensor
        ``[B, T, H, P]``, already multiplied by ``alpha``::

            layer       -> []            (0-dim; identical to the old float result)
            head        -> [1, 1, H, 1]
            token_head  -> [B, T, H, 1]

        The trailing ``1`` on the ``head`` form is load-bearing.  A bare ``[H]``
        broadcasts against the **pair** axis of ``[B, T, H, P]``, so the comparison
        would silently become "is pair j's norm above head j's reference" -- a
        plausible-looking, entirely wrong curve, with no error raised.  It is the
        same class of silent failure as the adjacent-pairing trap in ``rope.py``.
        """
        if granularity not in REF_GRANULARITIES:
            raise ValueError(
                f"unknown granularity {granularity!r}; allowed {REF_GRANULARITIES}")
        if reduce not in ("median", "mean"):
            raise ValueError(f"reduce must be 'median' or 'mean', got {reduce!r}")

        h = self.num_heads
        n = self.pairs.norms(x).float()                                   # [B,T,H,P]
        if scale is not None:
            n = n * scale
        if n.numel() == 0:
            return torch.zeros((), device=x.device)

        if granularity == "layer":
            flat = n.reshape(-1)
            ref = flat.median() if reduce == "median" else flat.mean()
        elif granularity == "head":
            # torch.median takes a SINGLE dim, so a per-head median pooled over
            # (token, pair) has to be built by moving H to the front and flattening
            # the rest.  n.median(dim=(0,1)) is not supported.
            flat = n.permute(2, 0, 1, 3).reshape(h, -1)                   # [H, B*T*P]
            ref = flat.median(dim=-1).values if reduce == "median" else flat.mean(dim=-1)
            ref = ref.view(1, 1, h, 1)
        else:  # token_head
            ref = n.median(dim=-1).values if reduce == "median" else n.mean(dim=-1)
            ref = ref.unsqueeze(-1)                                       # [B,T,H,1]
        return ref * alpha

    # ------------------------------------------------------------- operator

    def apply(self, x: Tensor, tau: float | Tensor,
              scale: Tensor | None = None) -> Tensor:
        """Zero the RoPE pairs of ``x`` whose (scaled) L2 norm is below ``tau``.

        ``x``: [B, T, num_heads * head_dim] — a key tensor in the same
        parametrisation :meth:`refs` measured.  Returns the same shape with the
        dropped pair slots zeroed.  (Zeroing pre-RoPE is exactly equivalent to
        zeroing the cached post-RoPE key: RoPE preserves each pair's norm and
        sends a zeroed pair to zero — see :mod:`spkv.core.rope`.)

        ``scale``: the query-saliency weights (see :meth:`refs`); the predicate
        is ``||pair|| * scale >= tau``.
        """
        h, d = self.num_heads, self.head_dim
        pairs = self.pairs.pair_view(x)                       # [B,T,H,half,2]
        w = pairs.norm(dim=-1)                                # [B,T,H,half]
        if scale is not None:
            w = w * scale
        keep = (w >= tau).unsqueeze(-1)                       # [B,T,H,half,1]
        return _write_pairs_back(pairs * keep, x, h, d)

    # ------------------------------------------------------------- counting

    @staticmethod
    def all_le_zero(tau: float | Tensor) -> bool:
        """True when the threshold keeps everything, for a scalar OR tensor tau.

        ``spkv/eval/judge.py`` used to write ``if taus[i] <= 0:``, which raises
        "Boolean value of Tensor with more than one element is ambiguous" the
        moment tau becomes a per-head tensor.  One helper, used by both
        consumers, so they cannot drift apart.
        """
        if isinstance(tau, (int, float)):
            return tau <= 0
        return bool((tau <= 0).all())

    def kept_frac(self, x: Tensor, tau: float | Tensor,
                  scale: Tensor | None = None) -> float:
        """Fraction of RoPE pairs at or above ``tau``, in ``x``'s own parametrisation."""
        if self.all_le_zero(tau):
            return 1.0
        n = self.pairs.norms(x)
        if scale is not None:
            n = n * scale
        return float((n >= tau).float().mean())

    def kept_frac_per_head(self, x: Tensor, tau: float | Tensor,
                           scale: Tensor | None = None) -> Tensor:
        """Kept fraction per kv head -> ``[num_heads]``.

        Uses the SAME ``>=`` predicate on the SAME ``tau`` object as :meth:`apply`,
        so "count the way you prune" holds by construction;
        ``kept_frac_per_head(...).mean()`` therefore equals :meth:`kept_frac`
        exactly.  The per-head vector is the statistic a scalar cannot show: under
        a layer-pooled threshold a model with uneven head norms loses WHOLE HEADS
        while a normalised one never can, and both report the same 0.50.
        """
        if self.all_le_zero(tau):
            return torch.ones(self.num_heads, device=x.device)
        n = self.pairs.norms(x)                                             # [B,T,H,P]
        if scale is not None:
            n = n * scale
        return (n >= tau).float().mean(dim=(0, 1, 3))                       # [H]


class TopKReference:
    """Tests-only reference: keep the top ``keep_frac`` pairs by norm.

    This is NOT the deployment operator (see the module header).  It exists so
    the tests can pin the half-split pairing, the whole-pair-zero rule and the
    zero-survival-through-RoPE contract against an independent selection rule.
    """

    def __init__(self, num_heads: int, head_dim: int):
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.pairs = RopePairs(self.num_heads, self.head_dim)

    def projection(self, x: Tensor, keep_frac: float) -> Tensor:
        """Zero the lowest-norm RoPE pairs of a raw ``k_proj`` output.

        ``x``: [B, T, num_heads * head_dim] -> same shape, dropped pair slots zeroed.
        """
        if not 0.0 < keep_frac <= 1.0:
            raise ValueError(f"keep_frac must be in (0, 1], got {keep_frac}")
        h, d = self.num_heads, self.head_dim
        half = d // 2
        pairs = self.pairs.pair_view(x)                       # [B,T,H,half,2]
        norms = pairs.norm(dim=-1)
        kk = max(1, min(half, int(round(keep_frac * half))))
        thresh = norms.topk(kk, dim=-1).values[..., -1:]      # [B,T,H,1]
        keep = (norms >= thresh).unsqueeze(-1)                # [B,T,H,half,1]
        return _write_pairs_back(pairs * keep, x, h, d)

    @staticmethod
    def cache_pair_view(k: Tensor, head_dim: int) -> Tensor:
        """[B, H, T, D] cache layout -> [B, H, T, D//2, 2], pair = (j, j + D//2)."""
        return torch.stack([k[..., : head_dim // 2], k[..., head_dim // 2 :]], dim=-1)

    def cache(self, k: Tensor, keep_frac: float) -> tuple[Tensor, Tensor]:
        """Zero the lowest-norm RoPE pairs of a K tensor, per (batch, head, token).

        ``k``: [B, H_kv, T, D].  Returns ``(k_pruned, keep_mask)`` with ``keep_mask``
        of shape [B, H_kv, T, D//2] (bool).  At least one pair is always kept, so
        ``keep_frac`` is a soft floor rather than an exact target.
        """
        if not 0.0 < keep_frac <= 1.0:
            raise ValueError(f"keep_frac must be in (0, 1], got {keep_frac}")
        d = self.head_dim
        pairs = self.cache_pair_view(k, d)               # [B, H, T, d/2, 2]
        norms = pairs.norm(dim=-1)                       # [B, H, T, d/2]
        n_pairs = d // 2
        kk = max(1, min(n_pairs, int(round(keep_frac * n_pairs))))
        thresh = norms.topk(kk, dim=-1).values[..., -1:]     # [B, H, T, 1]
        keep = norms >= thresh                              # [B, H, T, d/2]

        pruned = torch.empty_like(k)
        pruned[..., : d // 2] = pairs[..., 0] * keep
        pruned[..., d // 2 :] = pairs[..., 1] * keep
        return pruned, keep
