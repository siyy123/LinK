"""Checkpoint I/O -- the interface between the training stack and the judge.

A checkpoint carries the weights, the step, and whatever the judge needs to
reproduce the training-time operator (the frozen per-head ``prune_tau`` and its
spec, written by ``QKNormStack.checkpoint_extra``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch

__all__ = ["Checkpoint"]


class Checkpoint:
    """One checkpoint file: save into it, load out of it, or write a rotating one."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def save(self, model, step: int, extra: Optional[dict] = None) -> "Checkpoint":
        """Save weights, step, and the operator payload (``extra``)."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"model": model.state_dict(), "step": int(step)}
        if extra:
            payload.update(extra)
        torch.save(payload, self.path)
        return self

    def load(self, model) -> dict:
        """Load ``self.path`` into ``model`` and return the full payload."""
        payload = torch.load(self.path, map_location="cpu", weights_only=False)
        missing, unexpected = model.load_state_dict(payload["model"], strict=False)
        n_par = sum(v.numel() for v in payload["model"].values() if v.dtype.is_floating_point)
        print(f"[ckpt] {self.path}: missing={len(missing)} unexpected={len(unexpected)} "
              f"({n_par/1e6:.1f}M float params)")
        return payload

    @classmethod
    def write_rotating(cls, out_dir: Path, prefix: str, tag: int, model, keep: int) -> "Checkpoint":
        """Write ``<prefix>_<tag>.pt`` and delete all but the most recent ``keep``.

        A long run that only writes at the end loses everything to a crash near
        the end, and the judge needs a checkpoint that actually exists.
        """
        ckpt = cls(out_dir / f"{prefix}_{tag:04d}.pt").save(model, tag)
        print(f"[ckpt] wrote {ckpt.path.name}", flush=True)
        saved = sorted(out_dir.glob(f"{prefix}_*.pt"),
                       key=lambda p: int(p.stem.rsplit("_", 1)[1]))
        for old in saved[:-max(1, keep)]:
            old.unlink()
            print(f"[ckpt] pruned {old.name}")
        return ckpt
