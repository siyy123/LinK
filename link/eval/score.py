"""Downstream-task scorers -- LongBench's official metrics, replicated.

Sources fetched 2026-10-05 from THUDM/LongBench (now ``zai-org/LongBench``):
``LongBench/metrics.py`` and ``LongBench/eval.py``.  The replication rules:

* QA tasks (hotpotqa/2wikimqa/musique): SQuAD-style token F1 over
  ``normalize_answer``; the generation is scored RAW -- official eval.py applies
  first-line truncation ONLY to trec/triviaqa/samsum/lsht, not to these.
* Code tasks (lcc/repobench-p): ``code_sim_score`` -- take the FIRST line of the
  generation that contains no backtick, no ``#`` and no ``//``, then
  ``fuzz.ratio(line, gold) / 100``.  fuzzywuzzy's ``ratio`` is difflib
  ``SequenceMatcher.ratio()`` scaled to a rounded integer percent, which is what
  the stdlib implementation below reproduces.
* Final scores are reported by the official scripts as mean x 100 (percent);
  here per-item scores are kept in [0, 1] and ``bench_summary`` scales.

No new dependencies (stdlib only).  Any deviation from the official scripts must
be listed in the experiment report -- protocol fidelity is what makes the
numbers comparable to published methods.
"""

from __future__ import annotations

import difflib
import re
import string
from collections import Counter

__all__ = ["normalize_answer", "qa_f1", "exact_match", "code_sim",
           "post_process", "score"]


def normalize_answer(s: str) -> str:
    """Official ``metrics.normalize_answer``: lower, drop punctuation/articles, fix ws."""

    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        return "".join(ch for ch in text if ch not in set(string.punctuation))

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def qa_f1(pred: str, gold: str) -> float:
    """Official ``qa_f1_score``: token-bag F1 after normalisation.  No special
    casing for empty inputs -- ``num_same == 0`` scores 0, exactly as official."""
    p = normalize_answer(pred).split()
    g = normalize_answer(gold).split()
    same = sum((Counter(p) & Counter(g)).values())
    if same == 0:
        return 0.0
    precision = same / len(p)
    recall = same / len(g)
    return 2 * precision * recall / (precision + recall)


def exact_match(pred: str, gold: str) -> float:
    return float(normalize_answer(pred) == normalize_answer(gold))


def code_sim(pred: str, gold: str) -> float:
    """Official ``code_sim_score`` (fuzzywuzzy ``fuzz.ratio`` semantics)."""
    line = ""
    for candidate in pred.lstrip("\n").split("\n"):
        if "`" not in candidate and "#" not in candidate and "//" not in candidate:
            line = candidate
            break
    ratio = difflib.SequenceMatcher(None, line, gold).ratio()
    # fuzzywuzzy returns int(round(ratio * 100)); the official caller divides by
    # 100, so the reported score is round(ratio*100)/100.
    return round(ratio * 100) / 100


def post_process(text: str, mode: str) -> str:
    """Answer cleanup.  ``first_line`` is official for trec/triviaqa/samsum/lsht;
    QA and code tasks are scored raw (code filtering lives in :func:`code_sim`)."""
    if mode == "first_line":
        return text.lstrip("\n").split("\n")[0]
    if mode == "raw":
        return text
    if mode == "number":
        # GSM8K-style: the number after '####', else the LAST number seen.
        m = re.findall(r"####\s*([-+]?\d[\d,]*\.?\d*)", text)
        if m:
            return m[-1].replace(",", "").rstrip(".")
        nums = re.findall(r"[-+]?\d[\d,]*\.?\d*", text)
        return nums[-1].replace(",", "").rstrip(".") if nums else ""
    raise ValueError(f"unknown post_process mode {mode!r}")


def score(pred: str, golds: list[str], metric: str, post: str = "raw") -> float:
    """Best score over the gold list; ``metric`` in {f1, em, code_sim, contains,
    contains_frac}.

    ``contains`` / ``contains_frac`` are RULER-style recall checks (substring,
    case-insensitive): any-gold-present, and the FRACTION of golds present
    (RULER's partial-credit recall for multi-answer tasks such as cwe/fwe and
    the multikey variants; it degenerates to 0/1 for single-answer tasks).
    """
    pred = post_process(pred, post)
    if metric == "f1":
        return max((qa_f1(pred, g) for g in golds), default=0.0)
    if metric == "em":
        return max((exact_match(pred, g) for g in golds), default=0.0)
    if metric == "code_sim":
        return max((code_sim(pred, g) for g in golds), default=0.0)
    if metric == "contains":
        p = pred.lower()
        return float(any(g.lower() in p for g in golds))
    if metric == "contains_frac":
        if not golds:
            return 0.0
        p = pred.lower()
        return sum(1.0 for g in golds if g.lower() in p) / len(golds)
    raise ValueError(f"unknown metric {metric!r}")
