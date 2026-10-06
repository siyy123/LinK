"""Chunked next-token cross-entropy, shared by the trainer and the judge.

Chunked CE is not an optimisation -- it is required.  HF's ``ForCausalLMLoss``
calls ``logits.float()`` unconditionally once labels are present, so at 128K it
would materialise ``[1, 131072, 128256]`` fp32 ~ 67 GB before the softmax even
starts.  Chunking over the time axis keeps the peak at ~|chunk| x vocab.

What chunking alone did NOT do (measured at 32K: a CE-only training step peaked
at 70.4 GB on a 72 GB card): release the per-chunk graphs.  Every chunk's
log-softmax graph stays reachable from the running ``total`` until the joint
backward, so the retention is the same ``T x vocab`` footprint the chunking
meant to avoid, spread across chunks.  Each chunk therefore runs inside a
CHECKPOINTED region -- retained per chunk: one scalar; recomputed during
backward: one head-matmul plus a softmax, the cheapest part of the step.
"""

from __future__ import annotations

from functools import partial

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint

__all__ = ["ChunkedCrossEntropy"]


class ChunkedCrossEntropy:
    """The chunked scorer, bound to its chunk size and ignore index.

    A callable-loss object so the trainer and the judge cannot score with
    different conventions: one instance, one definition, both call sites.
    """

    def __init__(self, chunk: int = 1024, ignore_index: int = -100):
        self.chunk = int(chunk)
        self.ignore_index = int(ignore_index)

    def _chunk_sum(self, h_chunk: Tensor, tgt: Tensor, lm_head) -> Tensor:
        """One chunk's summed CE (the checkpointed unit; see the module docstring)."""
        logits = lm_head(h_chunk).float()
        return F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            tgt.reshape(-1),
            ignore_index=self.ignore_index,
            reduction="sum",
        )

    def __call__(self, hidden: Tensor, input_ids: Tensor, lm_head) -> Tensor:
        """Next-token CE without ever materialising the full ``[B, T, V]`` logits.

        ``hidden``: [B, T, H] (grad-carrying).  ``input_ids``: [B, T].
        ``lm_head`` is frozen, so we only need gradients w.r.t. ``hidden``.
        """
        b, t, _ = hidden.shape
        total = hidden.new_zeros(())
        n = 0
        for s in range(0, t, self.chunk):
            e = min(s + self.chunk, t)
            h_chunk = hidden[:, s:e]
            tgt = input_ids[:, s + 1 : e + 1]                   # target for position i is id[i+1]
            if tgt.shape[1] < h_chunk.shape[1]:
                # Must be integer: cross_entropy requires a Long target. Building
                # this with logits.new_full (float) silently upcasts tgt and
                # raises "expected scalar type Long but found Float".
                pad = torch.full(
                    (h_chunk.shape[0], h_chunk.shape[1] - tgt.shape[1]),
                    self.ignore_index,
                    dtype=tgt.dtype,
                    device=tgt.device,
                )
                tgt = torch.cat([tgt, pad], dim=1)
            if torch.is_grad_enabled() and h_chunk.requires_grad:
                term = checkpoint(partial(self._chunk_sum, tgt=tgt, lm_head=lm_head),
                                  h_chunk, use_reentrant=False)
            else:
                term = self._chunk_sum(h_chunk, tgt, lm_head)
            total = total + term
            n += int((tgt != self.ignore_index).sum())
        return total / max(n, 1)
