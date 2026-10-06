"""RoPE pair geometry — the structural unit the whole method depends on.

HF's Llama uses the *half-split* ``rotate_half`` layout::

    inv_freq = 1.0 / (base ** (arange(0, dim, 2) / dim))
    emb      = cat(freqs, freqs)                 # halves DUPLICATED, not interleaved
    cos, sin = emb.cos(), emb.sin()
    rotate_half(x): x1, x2 = x[:d//2], x[d//2:]; return cat((-x2, x1))

so RoPE acts as a 2-D rotation on the pair ``(j, j + head_dim/2)`` for
``j in [0, head_dim/2)``.  Because the same ``cos/sin`` is used for both slots,
the rotation is orthogonal on each pair, hence::

    k[j]**2 + k[j + d/2]**2   is invariant under RoPE            (norm preservation)
    k[j] = k[j + d/2] = 0     stays exactly 0 after RoPE         (zero survives RoPE)

Both facts are load-bearing:
  * norm preservation  => penalizing the pre-RoPE ``k_proj`` output is EXACTLY
                          equivalent to penalizing the cached (post-RoPE) key;
  * zero preservation  => the sparsity pattern learned in training survives the
                          position encoding, and pair-level runtime pruning is
                          the matched operator.

This is NOT the adjacent ``(2j, 2j+1)`` pairing of the original RoFormer / GPT-J
convention (vLLM's ``is_neox=False``).  Using the wrong pairing is a SILENT
failure: shapes stay valid, the loss curve looks normal, and accuracy quietly
degrades — a single-channel zero would not survive RoPE at all.

Note (verified for Llama-3.1-8B): ``rope_scaling = {rope_type: "llama3",
factor: 8.0}`` only rewrites ``inv_freq``; ``attention_scaling`` is exactly
1.0, so the rotation is a *pure* orthogonal rotation and the invariance above
holds exactly (not merely up to a constant).  ``partial_rotary_factor`` is 1.0,
so there are no non-rotating "pass-through" dimensions.
"""

from __future__ import annotations

import torch
from torch import Tensor

__all__ = ["RopePairs"]


class RopePairs:
    """The half-split RoPE pair structure, bound to one head geometry."""

    def __init__(self, num_heads: int, head_dim: int):
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)

    def pair_view(self, x: Tensor) -> Tensor:
        """Group ``x`` into RoPE pairs — the CORRECT half-split pairing.

        ``x`` : [B, T, num_heads * head_dim]  (a raw ``k_proj``/``q_proj`` output
        or a normalised capture)
        returns [B, T, num_heads, head_dim//2, 2] where ``out[..., j, :] =
        (x[..., j], x[..., j + head_dim//2])``.

        Implemented by splitting the head dim in half FIRST and then stacking, so
        the trailing axis of size 2 is exactly the rotation unit.
        """
        h, d = self.num_heads, self.head_dim
        if x.shape[-1] != h * d:
            raise ValueError(
                f"expected last dim {h * d} (num_heads={h} * head_dim={d}), "
                f"got {x.shape[-1]}"
            )
        b, t, _ = x.shape
        xh = x.view(b, t, h, d)
        return torch.stack([xh[..., : d // 2], xh[..., d // 2 :]], dim=-1)

    def norms(self, x: Tensor) -> Tensor:
        """L2 norm of each RoPE pair.  Shape [B, T, num_heads, head_dim//2]."""
        return self.pair_view(x).norm(dim=-1)

    def adjacent_view(self, x: Tensor) -> Tensor:
        """The WRONG pairing: adjacent ``(2j, 2j+1)``.

        Kept only so tests can assert that it differs — this is what a naive
        ``view(..., -1, 2)`` produces, and it is not rotation-invariant on HF
        models.
        """
        b, t, _ = x.shape
        return x.view(b, t, self.num_heads, self.head_dim // 2, 2)

    @staticmethod
    def rotate_half(x: Tensor) -> Tensor:
        """HF's ``rotate_half``: split the last dim in half and negate-swap."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def apply(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        """Reference RoPE application.

        ``x``   : [B, H, T, D]
        ``cos`` : [B, 1, T, D] (or broadcastable)

        Unlike HF's ``apply_rotary_pos_emb`` there is no ``unsqueeze_dim``:
        callers pass the cos/sin layout they need.
        """
        return (x * cos) + (self.rotate_half(x) * sin)

    @torch.no_grad()
    def check_invariance(self, x: Tensor, cos: Tensor, sin: Tensor,
                         atol: float = 1e-6) -> tuple[bool, float]:
        """Numerically verify that RoPE preserves pair norms.

        Returns ``(ok, max_rel_err)``.  ``x`` is [B, T, num_heads*head_dim] (as a
        raw projection output or capture); it is viewed to [B, H, T, D] internally.
        """
        h, d = self.num_heads, self.head_dim
        b, t, _ = x.shape
        xv = x.view(b, t, h, d).transpose(1, 2)  # [B, H, T, D]
        pre = self.norms(x.view(b, t, h * d))
        post_v = self.apply(xv, cos, sin)
        post = self.norms(post_v.transpose(1, 2).reshape(b, t, h * d))
        rel = ((pre - post).abs() / pre.abs().clamp_min(1e-12)).max().item()
        return rel <= atol, rel
