"""THE single definition of every training diagnostic.

Every earlier harness carried its own copy of these functions and they did not
agree.  There is now exactly one implementation (:class:`PairStats`), bound to
one (num_heads, head_dim) geometry, and it is the only thing any logger calls.

All statistics are computed on RoPE-PAIR norms ``n[b, t, h, p]``, where ``p``
indexes PAIRS (there are ``head_dim // 2`` of them), not channels:

============  ==============================================================
``cv``        std / mean over the flattened ``(b, t, h, p)`` set
``e_top10``   share of the TOTAL pair-norm sum carried by the largest 10%
``r``         mean over ``(b, t, h)`` of ``sum_p n / ||n||_2``
``pen``       mean over ``(b, t, h)`` of ``sum_p n``
============  ==============================================================

``r`` deserves its own note.  ``sum_p n / ||n||_2`` is the l1/l2 ratio of a
vector, so it is bounded by ``1 <= r <= sqrt(P)``.  A value outside that range
is impossible for a single tensor, so seeing one means the numerator and the
denominator came from DIFFERENT tensors -- e.g. a raw capture divided by a
unit-norm head.  ``r_in_range`` exists so this can never pass silently again.

``pen`` is numerically IDENTICAL to :class:`spkv.train.loss.GroupLassoRope`
(``mean`` over ``(b, t, h)`` of the per-head sum over pairs).  It is recomputed
here from the same pair-norm tensor so that the value the optimiser is given and
the value the log records can never drift apart -- the whole reason this module
exists.

Nothing here is a training signal.  These are measured, never optimised.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from .rope import RopePairs

__all__ = ["PairStats"]


class PairStats:
    """Diagnostics for one (num_heads, head_dim) geometry — measured, never optimised."""

    def __init__(self, num_heads: int, head_dim: int):
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.pairs = RopePairs(self.num_heads, self.head_dim)

    # ------------------------------------------------------------ the penalty

    def pen_value(self, x: Tensor) -> float:
        """The penalty functional itself: mean over (b, t, h) of the pair-norm sum.

        Range for a normalised head with ``P = head_dim // 2`` pairs is
        ``[1, sqrt(P)]`` (``1`` = all mass in one pair, ``sqrt(P)`` = uniform).
        """
        n = self.pairs.norms(x)
        return float(n.sum(dim=-1).mean())

    # ------------------------------------------------------------ distribution

    def concentration(self, k: Tensor) -> dict[str, float]:
        """Distribution statistics of the pair norms of ``k``."""
        n = self.pairs.norms(k.float().detach())                           # [B,T,H,P]
        flat = n.reshape(-1)
        if flat.numel() == 0:
            return {"cv": 0.0, "e_top10": 0.0, "r": 0.0, "pen": 0.0}

        mean = flat.mean()
        srt, _ = flat.sort(descending=True)
        tot = srt.sum().clamp_min(1e-12)

        per_head_sum = n.sum(dim=-1)                                  # [B,T,H]
        per_head_l2 = n.pow(2).sum(dim=-1).sqrt().clamp_min(1e-12)

        return {
            "cv": float(flat.std() / mean.clamp_min(1e-12)),
            "e_top10": float(srt[: max(1, int(0.10 * len(srt)))].sum() / tot),
            # r is the l1/l2 ratio per head, averaged.  Bounded by [1, sqrt(P)].
            "r": float((per_head_sum / per_head_l2).mean()),
            "pen": float(per_head_sum.mean()),
        }

    # ------------------------------------------------------------ lambda

    def quantile_lambda(self, acts, m_survive: int) -> float:
        """lambda such that ~``m_survive`` of the ``head_dim // 2`` pairs survive.

        The group soft-threshold ``k_p <- k_p (1 - lambda/||k_p||)_+`` zeroes a
        pair when ``||k_p|| < lambda``, so a target SURVIVING COUNT maps to the
        ``(1 - m/P)`` quantile of the observed pair norms.  This replaces sweeping
        lambda with a statement of what the run is trying to achieve.

        CAVEAT, and it is not a small one: that closed form is the PROXIMAL operator
        of ``lambda * sum_p ||k_p||``, i.e. one ISTA step with unit step size.  The
        trainers here do NOT take proximal steps -- they call ``loss.backward()`` and
        hand the gradient to Adam/Muon, so the effective threshold is the continuous
        decay rate times the distance travelled, not ``lambda`` itself.  Treat the
        quantile as setting lambda at the right ORDER OF MAGNITUDE for a target
        sparsity, and read the achieved pair count off the diagnostics.

        ``acts`` may be a single tensor or an iterable of per-layer tensors.

        ``torch.quantile`` refuses inputs above 2**24 elements, and concatenating
        every layer's pair norms blows past that at any real sequence length.  So
        each layer's quantile is taken separately and the per-layer values are
        averaged weighted by size.  For equal-sized layers that is CLOSE to, but
        not identical with, the pooled quantile (quantiles do not commute with
        averaging); the difference is absorbed by the order-of-magnitude caveat
        above.
        """
        P = self.head_dim // 2
        m = max(1, min(int(m_survive), P))
        q = 1.0 - m / P
        tensors = [acts] if isinstance(acts, torch.Tensor) else list(acts)

        total, acc = 0, 0.0
        for t in tensors:
            n = self.pairs.norms(t.float()).reshape(-1)
            if n.numel() == 0:
                continue
            cap = 1 << 24
            if n.numel() > cap:
                n = n[:: (n.numel() + cap - 1) // cap]
            acc += float(torch.quantile(n, q)) * n.numel()
            total += n.numel()
        if total == 0:
            raise ValueError("no pair norms to take a quantile of")
        return acc / total

    # ------------------------------------------------------------ single-tensor checks

    @staticmethod
    def r_in_range(r: float, n_pairs: int, tol: float = 1e-3) -> bool:
        """Is ``r`` inside the only range a single tensor can produce, ``[1, sqrt(P)]``?"""
        return (1.0 - tol) <= r <= (math.sqrt(n_pairs) + tol)

    def pair_usage_gini(self, k: Tensor) -> float:
        """Gini of per-pair mean activation, across all pairs.

        0 = every pair used equally; 1 = all usage on one pair.  This detects
        *global channel collapse* (a few pairs used by every token), which is a
        different failure from per-token sparsity and would also make the cache
        effectively low-dimensional for every token at once.
        """
        n = self.pairs.norms(k.detach())                               # [B,T,H,P]
        usage = n.reshape(-1, n.shape[-2] * n.shape[-1]).mean(dim=0).float()
        if usage.numel() < 2 or usage.sum() <= 0:
            return 0.0
        u, _ = torch.sort(usage)
        idx = torch.arange(1, u.numel() + 1, dtype=u.dtype, device=u.device)
        return float((2 * (idx * u).sum() / (u.numel() * u.sum())) - (u.numel() + 1) / u.numel())

    def gain_effective_rank_corr(self, act: Tensor, gain: Tensor,
                                 sample: int = 20000) -> float:
        """Rank correlation between the pair-norm profile the PENALTY trains and the
        one the KV CACHE actually holds.

        The penalty reads the pre-gain normalised vector ``u``; the cache holds
        ``g * u``.  The gain is token-independent, so it can reweight which pairs
        matter to attention without the penalty seeing it.  If the two profiles
        drift apart, the sparsity that was trained is not the sparsity that can be
        pruned -- a train/inference mismatch of exactly the kind this project has
        been bitten by before, and one that is invisible unless measured.

        Returns ~1.0 when the gain is not reshaping anything (the profiles agree),
        and lower as the gain starts to concentrate on dimensions the penalty does
        not know about.  Pairs are subsampled by stride for speed; the statistic is
        a rank correlation, so the subsample is unbiased.
        """
        with torch.no_grad():
            b, t, _ = act.shape
            h, d = self.num_heads, self.head_dim
            a = self.pairs.norms(act.detach().float()).reshape(-1)
            eff = (act.detach().float().view(b, t, h, d)
                   * gain.float().view(1, 1, 1, d)).reshape(b, t, h * d)
            c = self.pairs.norms(eff).reshape(-1)
            if a.numel() > sample:
                # clamp: linspace's last element is computed by accumulation and
                # overshoots to ``numel`` (one past the end) for sizes whose
                # endpoint is not exactly representable -- at T=40960 it handed
                # gather an out-of-bounds index and aborted the whole run with a
                # device-side assert (T=32768, a power of two, was exact, which
                # is why a size-dependent crash appeared only at the longer one).
                idx = torch.linspace(0, a.numel() - 1, sample).long().clamp_(0, a.numel() - 1)
                a, c = a[idx.to(a.device)], c[idx.to(c.device)]
            ra = a.argsort().argsort().float()
            rc = c.argsort().argsort().float()
            ra = ra - ra.mean()
            rc = rc - rc.mean()
            return float((ra * rc).sum() / (ra.norm() * rc.norm()).clamp_min(1e-12))

    @staticmethod
    def qk_head_ratio(q: Tensor, k: Tensor) -> float:
        """``||q|| / ||k||`` averaged over tokens.

        Under native QK-Norm both norms are pinned by the RMSNorm, so a steady
        drift of this ratio would be the fingerprint of the free q/k compensation
        direction (``q -> c q``, ``k -> k/c``) reappearing.

        The two sides are MEANED SEPARATELY before dividing: the q capture may
        be row-strided (``train.distill_q_stride``) while k never is, and pairing
        them elementwise made this ratio a shape error -- or, worse, a silent
        comparison of mismatched positions if the shapes happened to broadcast.
        Mean-of-norms is what the diagnostic wants and it is stride-invariant.
        """
        with torch.no_grad():
            qn = q.detach().float().norm(dim=-1).mean()
            kn = k.detach().float().norm(dim=-1).mean().clamp_min(1e-12)
            return float(qn / kn)
