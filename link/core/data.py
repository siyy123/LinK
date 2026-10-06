"""Corpus loading at NATIVE length — no packing, no concatenation, no mask.

User-decided policy.  It is also what the memory budget forces: at 128K any
``attention_mask`` is a ``[B, 1, T, T]`` tensor (~34 GB), and SDPA only skips it
when there is no padding and ``kv_length == query_length``.  So each sequence is
a single document, at its own length, and ``batch_size`` is 1 (variable-length
batching would need padding, hence a mask).

Measured per-example token lengths (real Llama-3 tokenizer):
  emozilla/pg19                      median 71,240   p90 196,176   max 1.47M
  open-r1/Mixture-of-Thoughts        median  3,457   max  30,982
  SWE-bench/SWE-smith-trajectories   median 29,396   max 100,948
Only PG19 reaches 128K, and about 25% of its books do — those produce the clean
single-document 128K windows the method is aimed at.
"""

from __future__ import annotations

import os
from typing import Any, Optional

import torch
from torch.utils.data import Dataset

__all__ = ["NativeLengthDocs", "Corpus"]

# hf-mirror serves datasets; huggingface.co does not resolve on this machine.
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


class NativeLengthDocs(Dataset):
    """Documents tokenised and truncated to ``max_len``, kept at native length.

    No padding, no packing, no concatenation.  ``start``/``limit`` select a
    CONTIGUOUS BLOCK of the split by global row index, so a training set and an
    evaluation set can be made disjoint by construction and the disjointness can
    be ASSERTED (see :class:`Corpus`) rather than trusted.

    ``min_len`` switches the block for a SELECTION: scan forward from ``start``
    keeping only documents whose tokenised length reaches ``min_len``, until
    ``limit`` are collected (or the scan budget ``limit * scan_cap_mult`` rows is
    exhausted, which raises).  This is what a long-context run needs — at 40K
    roughly a quarter of pg19 books are too short, and an "eval at 128K" that
    silently scores whatever the first validation books happen to be is worse
    than no eval at all.  ``scan_end`` records where the scan stopped, so an
    evaluation block can start AFTER every row the training scan touched.

    Tokenisation is cached per document (as an int64 tensor; consumers only
    slice/copy it), because the selection scan tokenises exactly the documents
    training then reads — and at 40K a re-tokenisation per pass is on the
    critical path.
    """

    def __init__(
        self,
        hf_id: str,
        tokenizer,
        split: str = "train",
        max_len: int = 131_072,
        start: int = 0,
        limit: Optional[int] = None,
        text_field: str = "text",
        min_len: Optional[int] = None,
        scan_cap_mult: int = 10,
    ):
        from datasets import load_dataset

        self.tokenizer = tokenizer
        self.max_len = max_len
        self.text_field = text_field
        self.split = split
        self.hf_id = hf_id
        self._cache: dict[int, "torch.Tensor"] = {}

        raw = load_dataset(hf_id, split=split)
        if not 0 <= start <= len(raw):
            raise ValueError(f"bad start={start} (split {split!r} has {len(raw)})")
        if min_len is None:
            stop = len(raw) if limit is None else min(start + limit, len(raw))
            self.rows = list(range(start, stop))
        else:
            cap = len(raw) if limit is None else min(len(raw), start + max(limit * scan_cap_mult, limit))
            self.rows = []
            for i in range(start, cap):
                if int(self._length_of(raw, i)) >= int(min_len):
                    self.rows.append(i)
                    if limit is not None and len(self.rows) >= limit:
                        break
            if limit is not None and len(self.rows) < limit:
                raise ValueError(
                    f"only {len(self.rows)} of the requested {limit} documents in "
                    f"split {split!r} from row {start} (scanned {cap - start}) reach "
                    f"{min_len} tokens; raise data.max_len, lower the minimum, or "
                    f"start from a different row."
                )
        self.start = start
        # First row NOT consumed by this block: for a selection it is where the
        # scan stopped, so a sibling block can start there and provably not
        # re-read a row this one took.
        self.scan_end = self.rows[-1] + 1 if self.rows else start
        self.ds = raw.select(self.rows)

    @property
    def index_range(self) -> list[int]:
        """Global row indices of this block, within its split (possibly strided)."""
        return list(self.rows)

    def __len__(self) -> int:
        return len(self.rows)

    def _length_of(self, raw, global_row: int) -> int:
        """Tokenised length of a raw row, via the same cache ``__getitem__`` uses."""
        ids = self._cache.get(global_row)
        if ids is None:
            ids = self._tokenise(raw[global_row][self.text_field])
            self._cache[global_row] = ids
        return int(ids.numel())

    def _tokenise(self, text: str) -> "torch.Tensor":
        # Truncate DURING tokenisation. Tokenising a whole PG19 book (up to 1.4M
        # tokens) and then slicing is dramatically slower than letting the
        # tokenizer stop at max_len.
        return torch.tensor(
            self.tokenizer(text, add_special_tokens=False, truncation=True,
                           max_length=self.max_len)["input_ids"],
            dtype=torch.long,
        )

    def __getitem__(self, idx: int) -> dict[str, Any]:
        global_row = self.rows[idx]
        ids = self._cache.get(global_row)
        if ids is None:
            # ``self.ds`` is the SELECTED table, so its local row is ``idx``; the
            # cache is keyed by the global row so the selection scan and the
            # training pass share entries.
            ids = self._tokenise(self.ds[idx][self.text_field])
            self._cache[global_row] = ids
        # The cached tensor is SHARED: consumers must slice/copy, never mutate
        # (the trainer slices [:seq_len] and copies with .to(device); the
        # collator only unsqueezes, which is a view).
        return {"input_ids": ids, "n": int(ids.numel())}

    @staticmethod
    def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
        """Collator for batch_size == 1: no padding, therefore no attention mask."""
        if len(batch) != 1:
            raise ValueError(
                "batch_size must be 1: variable-length batches would require padding, "
                "which requires an attention mask, which is ~34 GB at 128K and defeats "
                "the whole memory budget. Use gradient accumulation for a larger "
                "effective batch."
            )
        return {"input_ids": batch[0]["input_ids"].unsqueeze(0), "n": batch[0]["n"]}


class Corpus:
    """The training/evaluation document pair for one run, disjoint by construction.

    The joint trainer used to build its "held-out" set as ``ds[len-3:]`` of the
    very dataset it cycled through, so every evaluation document was also a
    training document.  The reported held-out CE fell to memorisation (a sharp
    cliff exactly at the micro-batch where the eval documents entered the
    training cycle) and was mistaken for the model recovering.  Both sides are
    now built from ONE place, with the disjointness asserted rather than trusted.
    """

    def __init__(self, cfg, tokenizer):
        self.cfg = cfg
        self.tokenizer = tokenizer

    def train_eval(self) -> tuple[NativeLengthDocs, NativeLengthDocs]:
        """Training and evaluation documents.

        ``eval_split: validation``   -- a different split entirely (preferred)
        ``eval_split: train_tail``   -- same split, the block AFTER the training block

        ``data.require_full_len`` makes the TRAINING block a selection of
        documents that reach ``seq_len`` tokens; ``data.eval_min_tokens`` does the
        same for the evaluation block (both scan forward, and the tail block then
        starts after every row the training scan looked at).
        """
        cfg = self.cfg
        hf_id = cfg.data.sources.get("pg19", "emozilla/pg19")
        train_min = cfg.data.seq_len if cfg.data.require_full_len else None
        common = dict(hf_id=hf_id, tokenizer=self.tokenizer, max_len=cfg.data.max_len)
        train = NativeLengthDocs(split="train", start=0, limit=cfg.data.n_train_docs,
                                 min_len=train_min, **common)

        if cfg.data.eval_split == "validation":
            ev = NativeLengthDocs(split="validation", start=0, limit=cfg.data.n_eval_docs,
                                  min_len=cfg.data.eval_min_tokens, **common)
            if train.split == ev.split:
                raise ValueError("eval_split=validation must load a different split")
            return train, ev

        if cfg.data.eval_split == "train_tail":
            ev = NativeLengthDocs(split="train", start=max(train.scan_end, cfg.data.n_train_docs),
                                  limit=cfg.data.n_eval_docs,
                                  min_len=cfg.data.eval_min_tokens, **common)
            Corpus.assert_disjoint(train, ev)
            return train, ev

        raise ValueError(
            f"data.eval_split must be 'train_tail' or 'validation', got {cfg.data.eval_split!r}"
        )

    @staticmethod
    def assert_disjoint(train: NativeLengthDocs, ev: NativeLengthDocs) -> None:
        """Raise if any evaluation row is also a training row."""
        if train.split != ev.split:
            return                       # different splits cannot collide
        overlap = set(train.index_range) & set(ev.index_range)
        if overlap:
            raise ValueError(
                f"train/eval index ranges overlap on split {train.split!r}: "
                f"{sorted(overlap)[:10]}{' ...' if len(overlap) > 10 else ''} "
                f"(train={train.index_range}, eval={ev.index_range})"
            )
