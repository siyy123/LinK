"""Downstream task loading + prompt assembly -- LongBench protocol replication.

* **Data**: the OFFICIAL THUDM/LongBench (now ``zai-org/LongBench``) release,
  distributed as ``data.zip`` on the Hub.  The task files used here are
  extracted under ``data/longbench/data/`` on first use (hf-mirror; the
  endpoint is set by importing :mod:`spkv.core.data`).
* **Prompts**: vendored VERBATIM from the official ``config/dataset2prompt.json``
  (fetched 2026-10-05).  Punctuation is load-bearing for protocol fidelity --
  do NOT "improve" these strings.
* **Truncation**: the official rule (``pred.py``): when a prompt exceeds the
  budget, keep ``budget // 2`` tokens from EACH end, decode both halves with
  ``skip_special_tokens=True`` and re-encode -- see :func:`fit_prompt`.
* **max_new_tokens**: official ``config/dataset2maxlen.json``.
* **Anchors**: ``gsm8k`` (cached ``openai/gsm8k``, deterministic 8-shot CoT) and
  ``niah`` (synthetic needle-in-a-haystack from pg19, built at the TOKEN level
  so the length/depth grid is exact; no download).

The chat template is deliberately NOT applied: the checkpoints are base-LM
distillation targets trained on raw text, and the official scripts only wrap
chat models.  This (and every other choice here) must be listed as a protocol
note wherever the numbers are compared to published ones.
"""

from __future__ import annotations

import dataclasses
import json
import shutil
import zipfile
from pathlib import Path
from typing import Optional

from ..core import data as _data  # noqa: F401  (sets HF_ENDPOINT via setdefault)

__all__ = ["Item", "TaskSpec", "TASKS", "resolve_task", "load_task", "fit_prompt"]

_LB_DIR = Path(__file__).resolve().parents[2] / "data" / "longbench" / "data"

_LB_QA_PROMPT = (
    "Answer the question based on the given passages. Only give me the answer and do "
    "not output any other words.\n\nThe following are given passages.\n{context}\n\n"
    "Answer the question based on the given passages. Only give me the answer and do "
    "not output any other words.\n\nQuestion: {input}\nAnswer:"
)

# Vendored verbatim from THUDM/LongBench config/dataset2prompt.json (2026-10-05).
_LB_PROMPT = {
    "hotpotqa": _LB_QA_PROMPT,
    "2wikimqa": _LB_QA_PROMPT,
    "musique": _LB_QA_PROMPT,
    "lcc": "Please complete the code given below. \n{context}Next line of code:\n",
    "repobench-p": "Please complete the code given below. \n{context}{input}Next line of code:\n",
}
# config/dataset2maxlen.json (2026-10-05).
_LB_MAX_NEW = {"hotpotqa": 32, "2wikimqa": 32, "musique": 32,
               "lcc": 64, "repobench-p": 64}

NIAH_NEEDLE = " The magic number is 92417."
NIAH_QUESTION = "\n\nQuestion: What is the magic number?\nAnswer:"


@dataclasses.dataclass(frozen=True)
class TaskSpec:
    name: str
    kind: str            # "longbench" | "gsm8k" | "niah"
    metric: str          # spkv.eval.score metric
    post: str            # spkv.eval.score post_process mode
    max_new_tokens: int


TASKS: dict[str, TaskSpec] = {
    **{t: TaskSpec(t, "longbench", "f1", "raw", _LB_MAX_NEW[t])
       for t in ("hotpotqa", "2wikimqa", "musique")},
    **{t: TaskSpec(t, "longbench", "code_sim", "raw", _LB_MAX_NEW[t])
       for t in ("lcc", "repobench-p")},
    "gsm8k": TaskSpec("gsm8k", "gsm8k", "em", "number", 256),
    "niah": TaskSpec("niah", "niah", "contains", "raw", 32),
}

# RULER (NVIDIA) via the simonjegou/ruler parquet mirror: per-length configs
# 4096/8192/16384, 13 subtasks x 500 rows, each row prompt-ready
# (context + question + answer_prefix) with its own max_new_tokens.  The subset
# keeps one retrieval task per flavour plus tracing and aggregation; ids are
# interleaved across subtasks (robin) so a generic `limit` stays balanced.
_RULER_SUBTASKS = ("niah_single_1", "niah_multikey_2", "niah_multivalue",
                   "vt", "cwe")
for _L in (4096, 8192, 16384):
    TASKS[f"ruler{_L // 1024}k"] = TaskSpec(
        f"ruler{_L // 1024}k", "ruler", "contains_frac", "raw", 128)


@dataclasses.dataclass(frozen=True)
class Item:
    id: str
    prompt: str
    golds: tuple[str, ...]
    max_new_tokens: int
    # Pre-tokenised prompts (NIAH builds at the token level so the length x
    # depth grid is exact): the runner uses these ids verbatim, skipping both
    # tokenisation and truncation.
    prompt_ids: Optional[tuple[int, ...]] = None


def resolve_task(name: str) -> TaskSpec:
    if name not in TASKS:
        raise KeyError(f"unknown task {name!r}; allowed {sorted(TASKS)}")
    return TASKS[name]


def _ensure_longbench_file(name: str) -> Path:
    """Extract ``data/<name>.jsonl`` from the official ``data.zip`` on first use."""
    path = _LB_DIR / f"{name}.jsonl"
    if path.exists():
        return path
    from huggingface_hub import hf_hub_download

    zip_path = hf_hub_download("zai-org/LongBench", "data.zip", repo_type="dataset")
    _LB_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z, z.open(f"data/{name}.jsonl") as src, \
            open(path, "wb") as dst:
        shutil.copyfileobj(src, dst)
    return path


def _load_longbench(spec: TaskSpec) -> list[Item]:
    path = _ensure_longbench_file(spec.name)
    items: list[Item] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            prompt = _LB_PROMPT[spec.name].format(context=row["context"],
                                                  input=row["input"])
            items.append(Item(id=str(row["_id"]), prompt=prompt,
                              golds=tuple(row["answers"]),
                              max_new_tokens=spec.max_new_tokens))
    return items


def _load_gsm8k(limit: int) -> list[Item]:
    """Deterministic 8-shot CoT from the CACHED openai/gsm8k (no download)."""
    from datasets import load_dataset

    train = load_dataset("openai/gsm8k", "main", split="train")
    test = load_dataset("openai/gsm8k", "main", split="test")
    shots = "\n\n".join(
        f"Q: {train[i]['question']}\nA: {train[i]['answer']}" for i in range(8))
    items = []
    for i in range(min(limit, len(test))):
        row = test[i]
        gold = row["answer"].rsplit("####", 1)[-1].strip().replace(",", "")
        items.append(Item(id=f"gsm8k-{i}", prompt=f"{shots}\n\nQ: {row['question']}\nA: ",
                          golds=(gold,), max_new_tokens=TASKS["gsm8k"].max_new_tokens))
    return items


def _load_ruler(spec: TaskSpec) -> list[Item]:
    """RULER rows for one length config, round-robin across the subtask subset.

    Prompts are prompt-ready in the dataset (``context + question +
    answer_prefix``); golds are the row's ``answer`` list; ``max_new_tokens`` is
    the row's own value (128 for retrieval/tracing, 50 for cwe, ...).  The
    round-robin order keeps a generic ``limit`` balanced across subtasks.
    """
    from datasets import load_dataset

    length = int(spec.name[5:-1]) * 1024          # "ruler8k" -> 8192
    ds = load_dataset("simonjegou/ruler", str(length), split="test")
    per_task: dict[str, list[Item]] = {t: [] for t in _RULER_SUBTASKS}
    for i, row in enumerate(ds):
        if row["task"] not in per_task:
            continue
        per_task[row["task"]].append(Item(
            id=f"ruler{length}-{row['task']}-{i}",
            prompt=row["context"] + row["question"] + row["answer_prefix"],
            golds=tuple(row["answer"]),
            max_new_tokens=int(row["max_new_tokens"])))
    items: list[Item] = []
    for j in range(max((len(v) for v in per_task.values()), default=0)):
        for t in _RULER_SUBTASKS:
            if j < len(per_task[t]):
                items.append(per_task[t][j])
    return items


def _build_niah(tok, lengths=(8192, 16384, 32768),
                depths=(0.0, 0.25, 0.5, 0.75, 1.0), n_docs=3,
                min_len: int = 32768) -> list[Item]:
    """Synthetic needle-in-a-haystack, exact at the token level.

    The haystack is a pg19 validation document; the needle is inserted at a
    fraction of the 0-based token length; the runner sends ``prompt_ids``
    verbatim (no re-tokenisation, no truncation), so the (length, depth) grid
    is exactly what the item says it is.
    """
    from ..core.data import NativeLengthDocs

    docs = NativeLengthDocs("emozilla/pg19", tok, split="validation",
                            max_len=max(lengths), limit=n_docs, min_len=min_len)
    needle_ids = tok(NIAH_NEEDLE, add_special_tokens=False)["input_ids"]
    q_ids = tok(NIAH_QUESTION, add_special_tokens=False)["input_ids"]
    items = []
    for di, doc in enumerate(docs):
        ids = doc["input_ids"].tolist()      # NativeLengthDocs yields tensors
        for L in lengths:
            hay = ids[: max(1, L - len(needle_ids) - len(q_ids))]
            for depth in depths:
                pos = int(len(hay) * depth)
                full = hay[:pos] + needle_ids + hay[pos:] + q_ids
                items.append(Item(id=f"niah-d{di}-L{L}-p{int(depth * 100):03d}",
                                  prompt="", golds=("92417",),
                                  max_new_tokens=TASKS["niah"].max_new_tokens,
                                  prompt_ids=tuple(full)))
    return items


def load_task(spec: TaskSpec, tok, *, limit: Optional[int] = None, start: int = 0,
              shard: tuple[int, int] = (0, 1)) -> list[Item]:
    """Items for one task: ``[start : start+limit]``, THEN strided by ``shard``.

    The limit-then-shard order keeps each shard's item set identical across
    GPUs for a given (start, limit) -- shards partition, never overlap.
    """
    if spec.kind == "longbench":
        items = _load_longbench(spec)
    elif spec.kind == "ruler":
        items = _load_ruler(spec)
    elif spec.kind == "gsm8k":
        items = _load_gsm8k(limit if limit is not None else 1319)
    elif spec.kind == "niah":
        items = _build_niah(tok)
    else:
        raise ValueError(f"unknown task kind {spec.kind!r}")

    stop = None if limit is None else start + limit
    items = items[start:stop]
    i, n = shard
    return items[i::n] if n > 1 else items


def fit_prompt(ids: list[int], tok, max_tokens: int, mode: str) -> tuple[list[int], bool]:
    """Official truncation rule; returns ``(ids, was_truncated)``.

    ``head_tail`` is pred.py's rule verbatim: decode ``budget//2`` tokens from
    each end (``skip_special_tokens=True``) and re-encode the concatenation --
    the middle of the prompt is what gets dropped.  ``tail`` keeps the last
    ``budget`` tokens; ``none`` never truncates.
    """
    if mode == "none" or len(ids) <= max_tokens:
        return list(ids), False
    if mode == "tail":
        return list(ids[-max_tokens:]), True
    if mode == "head_tail":
        half = max_tokens // 2
        text = (tok.decode(ids[:half], skip_special_tokens=True)
                + tok.decode(ids[-half:], skip_special_tokens=True))
        return tok(text, add_special_tokens=False)["input_ids"], True
    raise ValueError(f"unknown truncate mode {mode!r}")
