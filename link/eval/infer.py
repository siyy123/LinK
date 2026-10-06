"""Deployment operator + generation, and the ``run.mode: infer`` benchmark entry.

The operator is the SAME one the judge measures: a per-layer (or per-head)
ENERGY THRESHOLD -- ``tau = alpha * the model's own reference`` -- applied to
each (token, head, RoPE-pair) norm of K, pre-gain, pre-RoPE (see
``spkv/core/calib.py``).  Never top-k; the kept fraction is an outcome.

Generation contract, each point pinned by tests/test_infer_generate.py:

* the mask runs under ``QKNormStack(fixed_tau=...)`` around ``model.generate``
  -- patches stay installed across prefill AND every decode step, and a masked
  pair is exactly zero in the KV cache (zero survives RoPE);
* greedy decoding REQUIRES both mutating ``model.generation_config`` and
  passing the kwargs -- see :func:`greedy_generation` for the trap;
* the tokenizer's 2D all-ones mask is passed; ``calib._full_attention_mask``
  must NOT be (it is prefill-only, q_len == kv_len).

Every generated item is appended to a per-(arm, task, keep) jsonl with its
generation text, score, timing and the REALIZED cache-zero fraction, so a run
is resumable and every number is auditable.
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import time
from pathlib import Path

import torch
from torch import Tensor

from ..config import parse_shard, SpKVConfig
from ..core import calib
from ..core.checkpoints import Checkpoint
from ..core.data import NativeLengthDocs
from ..core.qknorm import QKNormStack
from ..core.runtime import Runtime
from . import score as score_mod
from . import tasks as tasks_mod

__all__ = ["pruned_forward", "greedy_generation", "cache_keys", "cache_zero_stats",
           "GenResult", "BenchRunner", "run_infer"]


@contextlib.contextmanager
def pruned_forward(model, taus, n_q: int, n_kv: int, hd: int,
                   channel_scale: dict | None = None):
    """``QKNormStack`` with supplied taus and capture storing OFF.

    Empty/None ``taus`` -> the PLAIN model (nullcontext): the unpruned arm must
    be exactly the base model, not a stack that happens to mask nothing.
    ``channel_scale`` switches to the qstat operator (see ``calib.q_pair_scale``).
    """
    if not taus:
        yield model
        return
    with QKNormStack(model, n_q, n_kv, hd, fixed_tau=taus, capture_q=False,
                     channel_scale=channel_scale) as stack:
        stack.set_capture(False)
        yield stack


def greedy_generation(model) -> dict:
    """Greedy settings, with the silent-sampling TRAP neutralised.

    transformers 4.57's ``_prepare_generation_config`` restores any
    generation-config value equal to the global default back to the MODEL's own
    ``generation_config.json`` for models saved with transformers >= 4.50 -- and
    Qwen3-4B ships ``do_sample: true, temperature: 0.6, top_k: 20, top_p: 0.95``.
    A pre-built ``GenerationConfig(do_sample=False)`` OBJECT is therefore
    silently reverted (only a ``warning_once``).  Mutating the model's config
    AND passing the same values as kwargs is the combination that survives;
    ``tests/test_infer_generate.py`` pins determinism across seeds.
    """
    gc = model.generation_config
    gc.do_sample = False
    gc.temperature = 1.0
    gc.top_p = 1.0
    gc.top_k = 0
    gc.num_beams = 1
    return {"do_sample": False, "num_beams": 1, "temperature": 1.0}


def cache_keys(past_key_values) -> list:
    """Per-layer K tensors ``[B, H, T, D]`` from a transformers Cache.

    transformers 4.57's ``DynamicCache`` exposes ``.layers`` (key_cache is the
    older layout); support both so a version bump cannot silently break the
    deployment-side statistic.
    """
    keys = getattr(past_key_values, "key_cache", None)
    if keys is None:
        keys = [layer.keys for layer in past_key_values.layers]
    return list(keys)


def cache_zero_stats(past_key_values, n_kv: int, hd: int) -> dict:
    """Realized sparsity of the DEPLOYED cache: pairs with BOTH halves == 0.

    Zero survives RoPE exactly (spkv/core/rope.py) and nothing else produces
    exact zeros, so this is the deployment-side counterpart of the judge's
    kept-fraction ("count the way you prune", after the fact).  Half-split
    pairing ``(j, j + hd/2)`` -- the repo's one convention.
    """
    per_layer = []
    for k in cache_keys(past_key_values):    # [B, H, T, D]
        a, b = k[..., : hd // 2], k[..., hd // 2:]
        per_layer.append(float(((a == 0) & (b == 0)).float().mean()))
    return {"cache_zero_frac": sum(per_layer) / max(len(per_layer), 1),
            "per_layer_zero": per_layer}


@dataclasses.dataclass
class GenResult:
    text: str
    n_prompt: int
    n_gen: int
    stopped: str                     # "eos" | "length"
    seconds: float
    cache_zero_frac: float


@dataclasses.dataclass
class _Point:
    """One operating point: a keep target and the threshold that hits it."""

    tag: str
    target_keep: float
    alpha: float
    taus: dict
    report: calib.CalibReport
    scales: dict | None = None      # qstat query-saliency weights (per layer)


class BenchRunner:
    """Scores every (arm x task x operating point); resumable, auditable."""

    def __init__(self, cfg: SpKVConfig):
        self.cfg = cfg
        self.device = torch.device(
            cfg.run.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
        self.rt = Runtime(cfg, self.device)
        self.tok = self.rt.tokenizer()
        self.out_dir = Path(cfg.run.out_dir)
        self.bench_dir = self.out_dir / "bench"
        self.shard = parse_shard(cfg.bench.shard)
        self.budget = (cfg.bench.max_prompt_tokens
                       if cfg.bench.max_prompt_tokens is not None else cfg.data.seq_len)
        self.arms = [a for a in cfg.run.resolve_arms()
                     if a["ckpt"] is not None or cfg.bench.include_original]

    # ------------------------------------------------------------- entry

    def run(self) -> Path:
        cfg = self.cfg
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.bench_dir.mkdir(parents=True, exist_ok=True)
        cfg.to_yaml(self.out_dir / "config.yaml")
        specs = [tasks_mod.resolve_task(t) for t in cfg.bench.tasks]
        print(f"[bench] device={self.device} shard={cfg.bench.shard} "
              f"arms={[a['tag'] for a in self.arms]}")
        print(f"[bench] tasks={[s.name for s in specs]} keeps={cfg.bench.keeps} "
              f"budget={self.budget} truncate={cfg.bench.truncate}")

        summary: dict = {"schema": 1, "shard": cfg.bench.shard, "arms": {}}
        for arm in self.arms:
            summary["arms"][arm["tag"]] = self._run_arm(arm, specs)
        (self.out_dir / "bench_summary.json").write_text(
            json.dumps(summary, indent=2, default=str))
        self._write_table(summary)
        return self.out_dir / "bench_summary.json"

    # ------------------------------------------------------------- one arm

    def _calib_ids(self) -> Tensor:
        b = self.cfg.bench
        # Mirrors the judge: the pg19 repo id comes from data.sources, so a
        # bare "pg19" can never reach datasets' script-loading refusal.
        source = b.calib_source or self.cfg.data.sources.get("pg19", "emozilla/pg19")
        ds = NativeLengthDocs(source, self.tok, split=b.calib_split,
                              max_len=b.calib_tokens or self.cfg.data.seq_len,
                              start=b.calib_doc, limit=1)
        return ds[0]["input_ids"].unsqueeze(0).to(self.device)

    def _points(self, model, n_q: int, n_kv: int, hd: int) -> list[_Point]:
        b = self.cfg.bench
        ids = self._calib_ids()
        scales = None
        if b.prune_operator == "qstat":
            scales = calib.q_pair_scale(model, ids, n_q, n_kv, hd,
                                        amp=self.rt.autocast(),
                                        sdpa=self.rt.sdpa(ids.shape[1]))
            print(f"[bench] qstat operator: query scales over {len(scales)} layers")
        points: list[_Point] = []
        for keep in b.keeps:
            taus, report, alpha = calib.calibrate_to_keep(
                model, ids, n_q, n_kv, hd, keep,
                granularity=b.prune_granularity, tol=b.keep_tol,
                sample_stride=b.calib_sample_stride, scales=scales,
                amp=self.rt.autocast(), sdpa=self.rt.sdpa(ids.shape[1]))
            tag = f"k{round(keep * 100):03d}"
            print(f"[bench] {tag}: alpha={alpha:.4g} kept={report.kept_frac:.4f}")
            points.append(_Point(tag, keep, alpha, taus, report, scales))
        for alpha in b.alphas:
            taus, report = calib.calibrate(model, ids, n_q, n_kv, hd,
                                           granularity=b.prune_granularity,
                                           alpha=alpha, scales=scales,
                                           amp=self.rt.autocast(),
                                           sdpa=self.rt.sdpa(ids.shape[1]))
            points.append(_Point(f"a{alpha:g}", float("nan"), alpha, taus, report,
                                 scales))
        return points

    def _run_arm(self, arm: dict, specs: list[tasks_mod.TaskSpec]) -> dict:
        tag, ckpt = arm["tag"], arm["ckpt"]
        print(f"\n[bench] arm {tag}: building model ({ckpt or 'original base'})")
        model = self.rt.build_model()
        if ckpt is not None:
            payload = Checkpoint(ckpt).load(model)
            ck_mode = payload.get("k_scale_mode")
            if ck_mode and ck_mode != "qk_norm":
                raise ValueError(
                    f"[{tag}] checkpoint was trained with k_scale_mode={ck_mode!r}; "
                    f"this runner only knows the native-QK-Norm parametrisation."
                )
        model.eval()
        n_q, n_kv, hd = Runtime.head_dims(model)
        points = self._points(model, n_q, n_kv, hd)
        (self.bench_dir / f"{tag}.meta.json").write_text(json.dumps({
            "tag": tag, "ckpt": ckpt,
            "granularity": self.cfg.bench.prune_granularity,
            "operator": self.cfg.bench.prune_operator,
            "points": [{"tag": p.tag, "target_keep": p.target_keep,
                        "alpha": p.alpha, "report": p.report.to_dict()}
                       for p in points],
            "budget": self.budget, "truncate": self.cfg.bench.truncate,
            "seed": self.cfg.bench.seed,
        }, indent=2, default=str))

        arm_summary: dict = {}
        for spec in specs:
            limit = self.cfg.bench.limits.get(spec.name, self.cfg.bench.limit)
            items = tasks_mod.load_task(spec, self.tok, limit=limit,
                                        start=self.cfg.bench.start, shard=self.shard)
            arm_summary[spec.name] = {}
            for point in points:
                arm_summary[spec.name][point.tag] = self._run_cell(
                    model, spec, items, point, n_q, n_kv, hd, tag)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return arm_summary

    # ------------------------------------------------------------- one cell

    def _run_cell(self, model, spec, items, point: _Point, n_q, n_kv, hd,
                  arm_tag: str) -> dict:
        i, n = self.shard
        path = self.bench_dir / f"{arm_tag}__{spec.name}__{point.tag}__s{i}of{n}.jsonl"
        done = self._done_ids(path) if self.cfg.bench.resume else set()
        todo = [it for it in items if it.id not in done]
        print(f"[bench] {arm_tag}/{spec.name}/{point.tag}: "
              f"{len(todo)} items ({len(done)} already done)")
        sess_scores: list[float] = []
        with open(path, "a", encoding="utf-8") as f:
            for k, item in enumerate(todo):
                rec = self._generate_one(model, item, spec, point, n_q, n_kv, hd)
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                sess_scores.append(rec["score"])
                if (k + 1) % 20 == 0 or k + 1 == len(todo):
                    print(f"    [{k + 1}/{len(todo)}] session-mean="
                          f"{sum(sess_scores) / len(sess_scores):.4f}")

        # Aggregate over the FULL file, not just this session's items: a resumed
        # run (skipped ids, added arms sharing cells) must report complete cells.
        scores, zeros, times = [], [], []
        truncs = 0
        for line in path.open(encoding="utf-8"):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            scores.append(r["score"])
            zeros.append(r["cache_zero_frac"])
            times.append(r["seconds"])
            truncs += int(bool(r.get("truncated")))
        n_all = len(scores)
        return {
            "items": n_all,
            "score_mean": sum(scores) / n_all if n_all else None,
            "score_pct": 100 * sum(scores) / n_all if n_all else None,
            "cache_zero_frac_mean": sum(zeros) / n_all if n_all else None,
            "truncated_frac": truncs / n_all if n_all else None,
            "seconds_per_item": sum(times) / n_all if n_all else None,
            "alpha": point.alpha,
            "target_keep": point.target_keep,
            "predicted_keep": point.report.kept_frac,
        }

    def _generate_one(self, model, item: tasks_mod.Item, spec, point: _Point,
                      n_q: int, n_kv: int, hd: int) -> dict:
        if item.prompt_ids is not None:      # NIAH: pre-fitted token ids
            ids = list(item.prompt_ids)
            truncated = False
        else:
            ids = self.tok(item.prompt, add_special_tokens=True)["input_ids"]
            ids, truncated = tasks_mod.fit_prompt(ids, self.tok, self.budget,
                                                  self.cfg.bench.truncate)
        max_new = self.cfg.bench.max_new_tokens.get(spec.name, item.max_new_tokens)
        input_ids = torch.tensor([ids], device=self.device)
        attn = torch.ones_like(input_ids)
        t0 = time.perf_counter()
        with torch.no_grad(), self.rt.autocast(), \
                self.rt.sdpa(input_ids.shape[1]), \
                pruned_forward(model, point.taus, n_q, n_kv, hd,
                               channel_scale=point.scales):
            out = model.generate(input_ids=input_ids, attention_mask=attn,
                                 max_new_tokens=max_new,
                                 return_dict_in_generate=True,
                                 **greedy_generation(model))
        seconds = time.perf_counter() - t0
        new_ids = out.sequences[0, input_ids.shape[1]:]
        text = self.tok.decode(new_ids, skip_special_tokens=True)
        eos = model.generation_config.eos_token_id
        eos_set = set(eos if isinstance(eos, (list, tuple)) else [eos])
        stopped = "eos" if len(new_ids) and int(new_ids[-1]) in eos_set else "length"
        cz = cache_zero_stats(out.past_key_values, n_kv, hd)
        return {
            "id": item.id,
            "prompt_tokens": len(ids),
            "truncated": truncated,
            "gold": list(item.golds),
            "pred": text,
            "score": score_mod.score(text, list(item.golds), spec.metric, spec.post),
            "n_gen": int(len(new_ids)),
            "stopped": stopped,
            "seconds": round(seconds, 3),
            "cache_zero_frac": cz["cache_zero_frac"],
        }

    # ------------------------------------------------------------- output

    @staticmethod
    def _done_ids(path: Path) -> set[str]:
        if not path.exists():
            return set()
        done = set()
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(json.loads(line)["id"])
                except (json.JSONDecodeError, KeyError):
                    continue
        return done

    def _write_table(self, summary: dict) -> None:
        arms = list(summary["arms"])
        if not arms:
            return
        tasks_ = list(summary["arms"][arms[0]])
        keeps = sorted({k for t in tasks_ for k in summary["arms"][arms[0]][t]},
                       key=lambda s: (len(s), s))
        lines = ["# bench-table (score %; zero=cache zero-pair fraction)", ""]
        for t in tasks_:
            lines.append(f"## {t}")
            header = "| arm | " + " | ".join(keeps) + " |"
            lines.append(header)
            lines.append("|---" * (len(keeps) + 1) + "|")
            for a in arms:
                cells = []
                for k in keeps:
                    c = summary["arms"][a].get(t, {}).get(k)
                    if not c or c["score_pct"] is None:
                        cells.append("—")
                    else:
                        cells.append(f"{c['score_pct']:.2f} (z={c['cache_zero_frac_mean']:.2f})")
                lines.append(f"| {a} | " + " | ".join(cells) + " |")
            lines.append("")
        (self.out_dir / "bench-table.md").write_text("\n".join(lines))
        print("\n" + "\n".join(lines))


def run_infer(cfg: SpKVConfig) -> Path:
    """``run.mode: infer`` entry -- benchmark every arm x task x keep point."""
    return BenchRunner(cfg).run()
