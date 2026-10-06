"""THE judge: pruned-model quality vs achieved keep fraction, by THRESHOLD.

Why threshold and not top-k
---------------------------
top-k keeps a fixed fraction of RoPE pairs per (token, head) for EVERY model, so
it manufactures sparsity for a model whose K is genuinely flat (the "top 10%" of a
uniform vector is still uniform) and assumes exactly the per-token selection
machinery this method claims not to need.  The method's claim is that training
drives most pair norms toward zero so that a plain magnitude threshold drops them
for free.  The achieved sparsity must therefore be an OUTCOME, not an input:

    for alpha in alphas:  drop pairs with ||pair|| < alpha * (this model's own
                          per-head median pair norm)
    report (achieved keep fraction, held-out CE)

The operator being judged is ``PairThreshold`` — the same class the trainer trains
against — applied through the same ``QKNormStack`` machinery, so implementation
drift between the two sides is impossible.

Three defects of the previous evaluator are fixed here and are worth stating:

1. **The reference was measured in the wrong parametrisation.**  The per-layer
   median was taken from the RAW ``k_proj`` output while the pruning hook fired
   after a normalisation had rescaled the tensor.  The reference is now measured
   under the SAME parametrisation that will be thresholded.
2. **The mask's tau and the judge's tau were different numbers.**  The trainer
   freezes tau at the transition step and records it in the checkpoint; this
   judge used to re-derive the reference from the FINAL model's own distribution
   even for the trained arm, so "alpha == prune_alpha" was not the operator the
   model trained under (measured drift: 3% median, up to 27% per element).  The
   judge now ALSO evaluates an ``as-trained`` sweep built from the checkpoint's
   frozen tau (scaled by alpha / prune_alpha), and prints the frozen-vs-re-derived
   drift so the two can never silently be confused again.
3. **``kept_frac`` was not checked for monotonicity in alpha**, and it was in fact
   non-monotone on a real run.  A rising keep fraction with a rising threshold
   means the pruned model's own activations changed the statistics being counted.
   This is reported loudly rather than passed over.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

import torch
from torch import Tensor

from ..config import SpKVConfig
from ..core import calib
from ..core.ce import ChunkedCrossEntropy
from ..core.calib import _full_attention_mask
from ..core.checkpoints import Checkpoint
from ..core.data import NativeLengthDocs
from ..core.prune import REF_GRANULARITIES, PairThreshold
from ..core.qknorm import QKNormStack
from ..core.runtime import Runtime

__all__ = ["parse_granularities", "run_eval"]

PG19 = "emozilla/pg19"


# ``_full_attention_mask`` now lives in spkv/core/calib.py -- the calibration
# forward is its primary consumer -- and is imported at the top of this module.
# It is re-exported here by name because tests/test_eval_causal_mask.py imports
# it from this module (the guard for the 128K dense-mask fix).


def parse_granularities(spec: str) -> tuple[str, ...]:
    """``"layer+head"`` -> ``("layer", "head")``.

    A ``+``-joined STRING rather than a list field on purpose: ``run.apply_sets``
    coerces ``--set`` values by the field's CURRENT type, so a tuple field would
    be clobbered with a raw string by ``--set run.eval_ref_granularity=head``.
    """
    gs = tuple(p for p in spec.split("+") if p)
    bad = [g for g in gs if g not in REF_GRANULARITIES]
    if not gs or bad:
        raise ValueError(
            f"run.eval_ref_granularity={spec!r}: unknown {bad}; "
            f"allowed {REF_GRANULARITIES} joined by '+'"
        )
    return gs


class PruningJudge:
    """Scores every arm under every pruning operator, and writes the comparison.

    One instance per ``run_eval`` invocation; all state (models, refs, rows)
    lives on ``self`` and is consumed by :meth:`run`.
    """

    def __init__(self, cfg: SpKVConfig):
        self.cfg = cfg
        # run.device, same resolution as the trainer: the 128K arm belongs on the
        # 72GB card (one 131072-token forward is ~100 s and the refs sweep needs
        # headroom), while an 8K sweep fits anywhere.
        self.device = torch.device(
            cfg.run.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
        self.rt = Runtime(cfg, self.device)
        self.ce = ChunkedCrossEntropy(cfg.train.ce_chunk)
        self.granularities = parse_granularities(cfg.run.eval_ref_granularity)
        # alpha=0 (full attention) is ALWAYS evaluated: it is the baseline the
        # "degradation" column subtracts, and a baseline taken at alpha>0 (as a
        # q3-judge config once did) silently reports degradation against an
        # already-pruned model.
        self.alphas = tuple(sorted(set(cfg.run.eval_alphas) | {0.0}))
        self.warnings: list[str] = []
        self.ref_spread: dict[str, float] = {}

    # ------------------------------------------------------------- entry

    def run(self) -> Path:
        cfg = self.cfg
        out_dir = Path(cfg.run.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        tok = self.rt.tokenizer()
        source = cfg.data.sources.get("pg19", PG19)
        if source != PG19:
            raise ValueError(
                f"the judge has a loader only for pg19; data.sources['pg19']={source!r}"
            )
        if cfg.data.eval_split == "validation":
            split, start = "validation", 0
        elif cfg.data.eval_split == "train_tail":
            split, start = "train", cfg.data.n_train_docs
        else:
            raise ValueError(
                f"data.eval_split must be 'train_tail' or 'validation', got "
                f"{cfg.data.eval_split!r}"
            )
        ds = NativeLengthDocs(source, tok, split=split, max_len=cfg.data.seq_len,
                              start=start, limit=cfg.data.n_eval_docs,
                              min_len=cfg.data.eval_min_tokens)
        docs = [ds[i]["input_ids"].unsqueeze(0).to(self.device) for i in range(len(ds))]
        print(f"[eval] {len(docs)} held-out {split} documents at seq_len<={cfg.data.seq_len}"
              + (f" (all >= {cfg.data.eval_min_tokens} tokens)" if cfg.data.eval_min_tokens
                 else ""))
        print(f"[eval] granularities: {self.granularities} | alphas: {self.alphas}")
        if cfg.penalty.prune_start_frac is not None or cfg.penalty.prune_alpha != 1.0:
            print("[eval] note: penalty.prune_* in this config belongs to TRAINING and is "
                  "not used by the judge; the judged operator comes from the checkpoint's "
                  "frozen tau (the 'frozen' rows) and from re-measured references (the "
                  "alpha rows).")

        results: dict[str, dict[str, list[dict]]] = {}
        for spec in self._resolve_specs():
            results[spec["tag"]] = self._judge_arm(spec, docs)

        out = out_dir / "eval_prune.json"
        out.write_text(json.dumps(
            {"schema": 3, "results": results,
             "ref_spread": {k: self.ref_spread.get(k) for k in results},
             "warnings": self.warnings}, indent=2))
        self._print_table(results)
        print(f"\n(saved to {out})")
        return out

    def _resolve_specs(self) -> list[dict]:
        """``[{tag, ckpt}]`` from ``run.ckpts``, else from ``run.ckpt`` (+ original).

        Delegates to :meth:`RunCfg.resolve_arms` so the judge and the inference
        benchmark share ONE arm-resolution contract (tags cannot drift apart).
        """
        return self.cfg.run.resolve_arms()

    # ------------------------------------------------------------- one arm

    def _judge_arm(self, spec: dict, docs: list[Tensor]) -> dict[str, list[dict]]:
        cfg = self.cfg
        tag, ckpt_path = spec["tag"], spec["ckpt"]
        # rope_scaling is applied ONLY here (the 128K arm): a checkpoint trained
        # without scaling and judged at 128K WITH YaRN is an extrapolation
        # extension test, not the operator the model was trained under — say so
        # wherever the numbers are reported.
        model = self.rt.build_model(
            rope_scaling=cfg.run.eval_rope_scaling,
            max_position_embeddings=cfg.data.seq_len if cfg.run.eval_rope_scaling else None,
        ).eval()
        if cfg.run.eval_rope_scaling:
            print(f"[{tag}] eval-time rope_scaling={cfg.run.eval_rope_scaling} "
                  f"(NOT trained with it)")

        payload: dict = {}
        frozen_tau = None
        if ckpt_path is not None:
            peek = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            # Judging a legacy spherical checkpoint here used to be silent and
            # produced a confident, meaningless curve.
            ck_mode = peek.get("k_scale_mode")
            if ck_mode and ck_mode != "qk_norm":
                raise ValueError(
                    f"[{tag}] checkpoint was trained with k_scale_mode={ck_mode!r}; "
                    f"this judge only knows the native-QK-Norm parametrisation."
                )
            payload = Checkpoint(ckpt_path).load(model)
            if payload.get("prune_tau"):
                frozen_tau = {int(k): torch.tensor(v, dtype=torch.float, device=self.device)
                              .view(1, 1, -1, 1)
                              for k, v in payload["prune_tau"].items()}
                g = payload.get("prune_granularity")
                if g and g not in self.granularities:
                    self.warnings.append(
                        f"[{tag}] checkpoint trained with prune_granularity={g!r}, which is "
                        f"not among the requested granularities {self.granularities} -- the "
                        f"alpha sweep re-measures references and does NOT reproduce the "
                        f"training operator; use the 'frozen' rows for that."
                    )
                    print(f"[WARNING] {self.warnings[-1]}")
        n_q, n_kv, hd = Runtime.head_dims(model)
        # References measured in the SAME parametrisation the mask will be applied
        # to (the pre-gain RMSNorm output, via the same stack the trainer uses) --
        # the recipe lives in spkv/core/calib.py so the judge and the inference
        # stack share it verbatim (fp32, one layer at a time: upcasting every
        # layer's capture at once is 36 x 131072 x 1024 x 4 B = 19.3 GiB at 128K).
        with torch.no_grad():
            caps = self._capture_k(model, docs[0], n_q, n_kv, hd)
        refs_by_gran = calib.reference_taus(caps, n_kv, hd, self.granularities)
        med = [float(r) for r in refs_by_gran["layer"]] if "layer" in refs_by_gran else []
        if med:
            print(f"[{tag}] per-layer median pair norm (pre-gain RMSNorm output): "
                  f"[{min(med):.4f}, {max(med):.4f}]")
        if "head" in refs_by_gran:
            # A model-level fact, independent of alpha: how badly a single pooled
            # threshold is mis-scaled across heads (~1 means the layer-pooled and
            # per-head operators coincide).  Computed per layer and then
            # maximised, deliberately: pooling across layers would fold in
            # layer-to-layer scale variation that has nothing to do with heads.
            spreads = [float(r.reshape(-1).max() / r.reshape(-1).min().clamp_min(1e-12))
                       for r in refs_by_gran["head"]]
            self.ref_spread[tag] = max(spreads)
            print(f"[{tag}] per-head reference spread, worst layer = "
                  f"{self.ref_spread[tag]:.2f}x")

        per_gran: dict[str, list[dict]] = {}
        torch.manual_seed(cfg.train.seed)
        for g in self.granularities:
            per_gran[g] = self._sweep(model, docs, refs_by_gran[g], n_q, n_kv, hd, g)
            self.warnings += self._check_monotone(per_gran[g], f"{tag}/{g}")
            self.warnings += self._check_head_invariance(per_gran, g, tag)

        if frozen_tau is not None:
            per_gran["frozen"] = self._frozen_sweep(model, docs, frozen_tau, payload,
                                                    n_q, n_kv, hd)
            self.warnings += self._check_monotone(per_gran["frozen"], f"{tag}/frozen")
            self._report_frozen_drift(tag, frozen_tau, refs_by_gran.get("head"))

        del model
        torch.cuda.empty_cache()
        return per_gran

    # ------------------------------------------------------------- sweeping

    def _sweep(self, model, docs, refs, n_q, n_kv, hd, granularity: str) -> list[dict]:
        rows = []
        for alpha in self.alphas:
            taus = [alpha * r for r in refs]
            r = self._evaluate(model, docs, taus, n_q, n_kv, hd, granularity, alpha)
            rows.append(r)
            print(f"[{granularity}] alpha={alpha:<5} kept={r['kept_frac']*100:5.1f}% "
                  f"minhead={r['per_head_kept_min']*100:5.1f}%  CE={r['ce']:.4f}",
                  flush=True)
        self._add_degradation(rows)
        return rows

    def _frozen_sweep(self, model, docs, frozen_tau: dict, payload: dict,
                      n_q: int, n_kv: int, hd: int) -> list[dict]:
        """The operator the model ACTUALLY trained under, at matching alpha.

        tau(alpha) = tau_frozen * (alpha / prune_alpha).  At alpha == prune_alpha
        this is exactly the frozen operator; other alphas scale it, which is the
        only well-defined extension of a single measured operating point.
        """
        prune_alpha = float(payload.get("prune_alpha", 1.0))
        rows = []
        for alpha in self.alphas:
            scale = (alpha / prune_alpha) if prune_alpha > 0 else alpha
            taus = [t * scale for t in
                    (frozen_tau[i] for i in sorted(frozen_tau))]
            r = self._evaluate(model, docs, taus, n_q, n_kv, hd, "frozen", alpha)
            r["frozen_scale"] = float(scale)
            rows.append(r)
            star = " <- as-trained" if abs(alpha - prune_alpha) < 1e-12 else ""
            print(f"[frozen] alpha={alpha:<5} kept={r['kept_frac']*100:5.1f}% "
                  f"minhead={r['per_head_kept_min']*100:5.1f}%  CE={r['ce']:.4f}{star}",
                  flush=True)
        self._add_degradation(rows)
        return rows

    @staticmethod
    def _add_degradation(rows: list[dict]) -> None:
        """CE delta in nats against the alpha=0 (full-attention) row, in place."""
        base = next((r for r in rows if r["alpha"] == 0.0), rows[0])
        for r in rows:
            r["degradation"] = r["ce"] - base["ce"]

    # --------------------------------------------------------------- core

    @torch.no_grad()
    def _capture_k(self, model, ids: Tensor, n_q: int, n_kv: int, hd: int) -> dict:
        """Per-layer K, captured in the parametrisation the mask will threshold."""
        return calib.capture_k(model, ids, n_q, n_kv, hd,
                               amp=self.rt.autocast(),
                               sdpa=self.rt.sdpa(ids.shape[1]))

    @torch.no_grad()
    def _evaluate(self, model, docs, taus: list, n_q: int, n_kv: int, hd: int,
                  granularity: str, alpha: float) -> dict:
        """Held-out CE and kept-fraction accounting with the given per-layer taus.

        The mask is installed by :class:`QKNormStack` with SUPPLIED taus — the
        same code the trainer masks with, so the operator being judged and the
        operator being trained cannot drift apart.
        """
        pruner = PairThreshold(n_kv, hd)
        stack = QKNormStack(model, n_q, n_kv, hd,
                            fixed_tau={i: t for i, t in enumerate(taus)},
                            capture_q=False)   # only K is counted/pruned here
        total, n_tok = 0.0, 0
        # stats[layer] = list of ([H] kept vector, n_tokens); token-weighted so the
        # reported keep fraction weighs each document the way the CE does.
        stats: dict[int, list[tuple[Tensor, int]]] = {}
        for ids in docs:
            if granularity == "token_head":
                # A per-(token, head) threshold can only be applied to documents
                # of identical length.  Fail loudly rather than broadcast into a
                # silently wrong curve.
                for tau in taus:
                    if torch.is_tensor(tau) and tau.shape[1] != ids.shape[1]:
                        raise ValueError(
                            f"token_head reference has T={tau.shape[1]} but this doc "
                            f"has T={ids.shape[1]}"
                        )
            with stack, self.rt.autocast(), self.rt.sdpa(ids.shape[1]):
                h = model.model(input_ids=ids, attention_mask=_full_attention_mask(ids),
                                use_cache=False).last_hidden_state
            t = ids.shape[1]
            total += float(self.ce(h, ids, model.lm_head)) * t
            n_tok += t
            for i, s in enumerate(stack.state):
                if s.k is None:
                    continue
                # ``s.k`` is the MASKED tensor: a dropped pair has norm exactly 0,
                # so counting ``>= tau`` on it reproduces the kept set without
                # re-deriving the mask ("count the way you prune").
                stats.setdefault(i, []).append(
                    (pruner.kept_frac_per_head(s.k, taus[i]).float(), t))

        per_layer_vec = {}
        for i, entries in stats.items():
            wsum = sum(w for _, w in entries)
            vec = sum(v * w for v, w in entries) / max(wsum, 1)
            per_layer_vec[i] = vec
        if per_layer_vec:
            per_head = torch.stack(list(per_layer_vec.values())).mean(0)
        else:
            per_head = torch.ones(n_kv)
        return {
            "alpha": float(alpha),
            "kept_frac": float(per_head.mean()),
            "per_head_kept_min": float(per_head.min()),
            "per_head_kept_max": float(per_head.max()),
            "per_layer_kept": {i: float(v.mean()) for i, v in per_layer_vec.items()},
            "ce": total / max(n_tok, 1),
        }

    # ------------------------------------------------------------- checks

    def _check_monotone(self, rows: list[dict], tag: str) -> list[str]:
        """A rising threshold must never keep MORE pairs. Report violations, loudly."""
        warnings = []
        for a, b in zip(rows, rows[1:]):
            if b["kept_frac"] > a["kept_frac"] + 1e-6:
                warnings.append(
                    f"{tag}: kept_frac ROSE from {a['kept_frac']:.4f} (alpha={a['alpha']}) "
                    f"to {b['kept_frac']:.4f} (alpha={b['alpha']}) with a larger threshold"
                )
        for w in warnings:
            print(f"[WARNING] non-monotone keep fraction -- {w}")
        if warnings:
            print("[WARNING] this means the pruned model's own activations shifted the "
                  "statistics being counted; the affected alpha range is not trustworthy.")
        return warnings

    def _check_head_invariance(self, per_gran: dict, gran: str, tag: str) -> list[str]:
        """For a model whose heads all have the same norm, layer and head coincide.

        True of a native-QK-Norm model: every head has norm ~sqrt(head_dim), from
        the RMSNorm itself.  A layer-pooled median and a per-head median then
        differ only through within-head concentration.  If they diverge, the
        reference code is wrong -- this is a free self-check, not a result.
        """
        if gran != "head" or "layer" not in per_gran:
            return []
        warnings = []
        for a, b in zip(per_gran["layer"], per_gran["head"]):
            dk = abs(b["kept_frac"] - a["kept_frac"])
            # The condition is on KEPT FRACTION only.  In the aggressive-pruning
            # regime CE is so steep in the keep fraction that a 0.1% difference in
            # which pairs survive moves CE by ~0.2 nats; warning on that would be
            # warning about the slope, not about the operator.
            if dk > 0.02:
                warnings.append(
                    f"{tag}: qk_norm checkpoint but layer and head granularities "
                    f"disagree at alpha={b['alpha']} (dkept={dk:.4f}) -- the two "
                    f"operators should coincide when all heads have equal norm"
                )
        for w in warnings:
            print(f"[WARNING] {w}")
        return warnings

    def _report_frozen_drift(self, tag: str, frozen_tau: dict,
                             head_refs: list | None) -> None:
        """Print how far the judge's re-derived reference sits from the frozen tau.

        Non-zero drift is expected (the model kept training after the transition;
        the reference document differs); the number exists so that no reader can
        mistake the re-derived alpha sweep for the as-trained operator.
        """
        if not head_refs:
            return
        rels = []
        for i in sorted(frozen_tau):
            ref = head_refs[i].reshape(-1).float()
            frz = frozen_tau[i].reshape(-1).float()
            n = min(ref.numel(), frz.numel())
            rels.append(((frz[:n] - ref[:n]).abs() / ref[:n].clamp_min(1e-12)).cpu())
        if not rels:
            return
        rel = torch.cat(rels)
        msg = (f"[{tag}] frozen vs re-derived head references: |rel| "
               f"median {float(rel.median()):.3%}, mean {float(rel.mean()):.3%}, "
               f"max {float(rel.max()):.3%} -- the alpha sweep is NOT the as-trained "
               f"operator; the 'frozen' rows are.")
        print(msg)

    # ------------------------------------------------------------- report

    def _print_table(self, results: dict[str, dict[str, list[dict]]]) -> None:
        """The headline grid: every model under every pruning operator.

        The ``minhead`` column is the point of the whole exercise.  Two models can
        report the same overall keep fraction while one of them has lost ENTIRE
        heads, and only the per-head vector shows it.
        """
        print("\n=== pruning judge: model x operator ===")
        for tag, spread in self.ref_spread.items():
            print(f"[{tag}] per-head reference spread max/min = {spread:.2f}x")
        cols = list(self.granularities) + [g for g in ("frozen",)
                                           if any("frozen" in rows for rows in results.values())]
        for alpha in self.alphas:
            print(f"\nalpha={alpha}")
            head = f"{'model':<20}"
            for g in cols:
                head += f" | {g+' kept%':>12} {'minhead':>8} {'CE':>8}"
            print(head)
            for tag, rows in results.items():
                line = f"{tag:<20}"
                for g in cols:
                    r = next((x for x in rows.get(g, []) if x["alpha"] == alpha), None)
                    if r is None:
                        line += f" | {'-':>12} {'-':>8} {'-':>8}"
                    else:
                        line += (f" | {r['kept_frac']*100:>11.1f}% "
                                 f"{r['per_head_kept_min']*100:>7.1f}% {r['ce']:>8.4f}")
                print(line)

        if "original" not in results:
            return
        for g in cols:
            base = results["original"].get(g)
            if not base:
                continue
            others = [t for t in results if t != "original" and g in results[t]]
            if not others:
                continue
            print(f"\n=== matched keep fraction [{g}] -- delta<0 means the model beats "
                  f"original ===")
            header = f"{'kept%':>7}"
            for t in others:
                header += f" | {t[:18]:>18}"
            print(header)
            xs = sorted({r["kept_frac"] for r in base + sum((results[t][g] for t in others), [])})
            for x in xs:
                line = f"{x*100:>6.1f}%"
                a = self._interp(base, x)
                for t in others:
                    b = self._interp(results[t][g], x)
                    line += f" | {b - a:>+18.4f}"
                print(line)

    @staticmethod
    def _interp(rows: list[dict], x: float) -> float:
        """CE at an arbitrary keep fraction, linearly interpolated across alpha."""
        pts = sorted((r["kept_frac"], r["ce"]) for r in rows)
        if not pts:
            return float("nan")
        if x <= pts[0][0]:
            return pts[0][1]
        if x >= pts[-1][0]:
            return pts[-1][1]
        for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
            if x0 <= x <= x1:
                t = 0.0 if x1 == x0 else (x - x0) / (x1 - x0)
                return y0 + t * (y1 - y0)
        return float("nan")


def run_eval(cfg: SpKVConfig) -> Path:
    """Entry point for ``run.mode: eval``."""
    return PruningJudge(cfg).run()
