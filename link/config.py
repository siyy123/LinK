"""Configuration dataclasses, loadable from YAML.

One CLI reads every config file (``python -m spkv.run --config <path>``); the
``run.mode`` key selects what it does (``train`` or ``eval``).  Nothing that a
run needs is passed on the command line -- if an experiment differs, it differs
in a file that can be diffed and re-run.

Two rules are enforced here rather than documented:

* **Unknown keys are an error.**  ``from_yaml`` raises on any key that is not a
  field, so a typo cannot silently become a default.
* **Enums are validated.**  An unknown value used to fall through to "no scaling
  at all" in a trainer -- a silent behaviour change that would quietly invalidate
  an ablation.  Anything not in the allowed set raises.

There is exactly ONE q/k parametrisation: the model's native QK-Norm
(``k_scale_mode`` used to select between "spherical", "qk_norm" and "none"; the
spherical and raw paths have been deleted, so the knob is gone with them).  The
penalty reads the model's own per-head RMSNorm output, pre-gain — see
``spkv/core/qknorm.py`` for why that is the only correct observation point.

Fields that were declared but never read by any code path have been DELETED
rather than left as decoration.  Add a field back when the code that reads it
exists.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

__all__ = ["RunCfg", "ModelCfg", "PenaltyCfg", "DataCfg", "TrainCfg", "BenchCfg",
           "SpKVConfig", "parse_terms"]

MODES = ("train", "eval", "infer")
# How a benchmark prompt is fitted to a token budget: "tail" keeps the LAST
# ``budget`` tokens (the question survives; LongBench's own rule), "head_tail"
# keeps ``budget//2`` from each end, "none" never truncates (the beyond-window
# probe).  All arms see identical prompts -- that is the comparability contract.
TRUNCATION_MODES = ("tail", "head_tail", "none")
# Distillation terms that can be MONITORED while inactive (computed under no_grad
# on held-out documents every log step) -- this is how the loss-migration
# experiment sees whether a term it stopped training on drifts.  Join with "+".
MONITOR_TERMS = ("map", "output", "kd")
# Granularities allowed for the TRAINING-time pruning threshold.  "token_head"
# is excluded on purpose: prune.REF_GRANULARITIES documents it as diagnostic-only,
# because a per-(token, head) reference divides away the very per-token magnitude
# collapse the method claims, and it pins the keep fraction near 0.5 for every
# model -- a threshold that always keeps half is not a sparsity operator.
PRUNE_GRANULARITIES = ("layer", "head")
# Deployment operators: "knorm" is the plain pair-norm threshold this repo
# trains against; "qstat" is the historic query-statistics variant -- the same
# threshold applied to ``||pair|| * s`` with ``s`` the per-(layer, kv-head,
# pair) running mean of the QUERY's pair L2 norms (channels queries exercise
# less get pruned harder).  Both remain THRESHOLD operators.
PRUNE_OPERATORS = ("knorm", "qstat")
PENALTY_NORMS = ("raw", "step0")
PENALTY_REDUCE = ("sum", "mean")
OPTIMIZERS = ("muon", "block_muon", "adamw", "sgd", "lion")
LR_SCHEDULES = ("cosine", "constant")


def parse_shard(spec: str) -> tuple[int, int]:
    """``"i/n"`` -> ``(i, n)``; raises unless ``0 <= i < n`` with ``n >= 1``."""
    try:
        i_s, n_s = str(spec).split("/")
        i, n = int(i_s), int(n_s)
    except ValueError as e:
        raise ValueError(f"shard must look like 'i/n', got {spec!r}") from e
    if n < 1 or not 0 <= i < n:
        raise ValueError(f"shard {spec!r}: need 0 <= i < n with n >= 1")
    return i, n


def parse_terms(spec: str) -> tuple[str, ...]:
    """``"map+output+kd"`` -> ``("map", "output", "kd")``; ``""`` -> ``()``.

    A ``+``-joined STRING rather than a tuple field for the same reason as
    ``judge.parse_granularities``: ``run.apply_sets`` coerces ``--set`` values by
    the field's CURRENT type, so a tuple field would be clobbered with a raw
    string by ``--set train.monitor_terms=map+kd``.
    """
    terms = tuple(p for p in (spec or "").split("+") if p)
    bad = [t for t in terms if t not in MONITOR_TERMS]
    if bad:
        raise ValueError(
            f"train.monitor_terms={spec!r}: unknown {bad}; "
            f"allowed {MONITOR_TERMS} joined by '+'"
        )
    return terms


@dataclass
class RunCfg:
    """What to do, and where to put it."""

    mode: str = "train"
    out_dir: str = "runs/run"
    # Device the (single-process) student runs on.  None -> "cuda:0" if a GPU is
    # available else "cpu" -- the historical behaviour.  IGNORED when
    # train.dp_size > 1: each process must take cuda:$LOCAL_RANK, or two ranks
    # collide on one GPU.
    device: Optional[str] = None
    # Rope scaling applied ONLY when building the judge's model (the 128K eval
    # arm).  Training never sees it: a checkpoint trained without scaling and
    # evaluated at 128K WITH YaRN is an extrapolation-extension test, not the
    # trained operator -- say so wherever its numbers are reported.
    eval_rope_scaling: Optional[dict] = None
    # total optimiser steps.  None -> train.max_steps.
    max_steps: Optional[int] = None
    # eval only: a single checkpoint, or a LIST scored in ONE process so the arms
    # of a comparison are measured under one code path, one reference rule and
    # one random seed.  Each entry: {tag: str, ckpt: str}.
    ckpt: Optional[str] = None
    ckpts: Optional[tuple[dict, ...]] = None
    eval_alphas: tuple[float, ...] = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0)
    eval_model: str = "both"          # both | trained | original
    # Threshold-reference granularity for the judge.  "layer" pools every
    # (token, head, pair) of a layer into ONE median -- the historical behaviour,
    # and the one that lets a single threshold delete whole low-norm heads of a
    # model with uneven head norms while being unable to do so to a normalised
    # one, whose heads all have norm ~sqrt(head_dim).  That asymmetry is a
    # property of the operator, not of the model.  "head" gives each kv head its
    # own reference (a constant, so it stays a plain offline threshold);
    # "token_head" gives each (token, head) its own.  Join with "+" to sweep
    # several in one run, e.g. "layer+head".  See spkv.core.prune.REF_GRANULARITIES.
    eval_ref_granularity: str = "layer"
    log_every: int = 10
    # Save an intermediate checkpoint every N steps and keep the most recent
    # ``ckpt_keep`` of them.  A long run that only writes at the end loses
    # everything to a crash at step N-1, and the judge needs a checkpoint that
    # exists.
    ckpt_every: int = 500
    ckpt_keep: int = 2

    def resolve_arms(self) -> list[dict]:
        """``[{tag, ckpt}]`` arms: ``run.ckpts`` else ``run.ckpt``; the ORIGINAL
        model is prepended per ``run.eval_model`` ({"tag": "original", "ckpt":
        None}).  ONE arm-resolution contract, shared by the judge and the
        inference benchmark, so their tag sets cannot drift apart."""
        if self.ckpts:
            specs = [dict(s) for s in self.ckpts]
            for s in specs:
                missing = {"tag", "ckpt"} - set(s)
                if missing:
                    raise KeyError(f"run.ckpts entry missing {sorted(missing)}: {s}")
            tags = [s["tag"] for s in specs]
            if len(set(tags)) != len(tags):
                raise ValueError(f"run.ckpts tags must be unique, got {tags}")
            specs = [s for s in specs if s["ckpt"]]
        else:
            specs = [{"tag": "trained",
                      "ckpt": self.ckpt or str(Path(self.out_dir) / "final.pt")}]
        original = {"tag": "original", "ckpt": None}
        if self.eval_model == "original":
            return [original]
        if self.eval_model == "both":
            return [original] + specs
        return specs


@dataclass
class ModelCfg:
    # Local path (downloaded from ModelScope) or a repo id if a hub is reachable.
    name_or_path: str = "models/Qwen3-4B"
    dtype: str = "bfloat16"
    # Must be sdpa or flash_attention_2. NEVER "eager": at 128K an eager
    # attention forward materialises [B, 32, 128K, 128K] ~ 1.1 TB of scores.
    attn_implementation: str = "sdpa"
    # Weight-locked fine-tuning: only these attention projections train (plus the
    # model's own q_norm/k_norm gains, which the penalty provably cannot reach).
    #
    # Phase 1 is Q and K only.  V and O stay frozen:
    #   * phase 1 applies no penalty to V, so a trainable v_proj would drift with
    #     nothing pulling on it;
    #   * a trainable o_proj is exactly the loophole that makes any V-side penalty
    #     evadable for free (v -> v/c with o_proj columns -> c*o_proj leaves
    #     attn_out and the CE bit-identical).
    trainable_projections: tuple[str, ...] = ("q_proj", "k_proj")


@dataclass
class PenaltyCfg:
    lam_q: float = 0.0
    lam_k: float = 0.0
    # tie_qk forces lam_q == lam_k. It closes the ANTISYMMETRIC half of the free
    # q/k direction (q_p up while k_p down). Under native QK-Norm the RMSNorm
    # pins the head norm, so the older symmetric-shrink worry does not apply; the
    # tied term is kept as cheap insurance (it does NOT close the residual
    # W_k-only causal-tail escape -- see qknorm.py).
    tie_qk: bool = True
    lam_d: float = 1.0                # attention-output distillation weight

    # How the per-layer penalty is scaled before multiplying by lambda.
    #   "raw"   -- lambda is the physical LASSO weight (default)
    #   "step0" -- divide each layer by its step-0 value so every layer starts at
    #              O(1). Changes the units of lambda by the size of that reference
    #              (~50-150x for K), which is exactly the discrepancy that made
    #              lambda=0.1 mean two different things in two harnesses.
    # Both RAW and NORMALISED values are logged every step regardless of this
    # setting, so the two can never diverge unnoticed again.
    norm: str = "raw"
    ramp_frac: float = 0.05

    # How the per-layer penalties are combined into the one scalar the optimiser
    # sees.
    #   "sum"  -- pen = lambda * SUM_l P_l.  lambda is then the LASSO weight on
    #             EACH layer's penalty, so a lambda transfers between paths.
    #             This is the default.
    #   "mean" -- pen = lambda * MEAN_l P_l.  Makes the CE/penalty balance
    #             depth-independent, but silently divides the force each layer
    #             feels by the layer count -- 28x at 3B.  Keep it only as an
    #             explicit ablation.
    reduce: str = "sum"

    # Set lambda from a TARGET SURVIVING-PAIR COUNT instead of sweeping it.
    #
    # Group soft-thresholding zeroes pair p when ||k_p|| < lambda, so asking for
    # ``m`` survivors out of P pairs per head fixes lambda at the ``(1 - m/P)``
    # quantile of the pair-norm distribution.  This turns a blind hyper-parameter
    # search into a direct statement of intent, and it is scale-free: the quantile
    # is taken in whatever parametrisation the penalty acts on.
    #
    # When set, these OVERRIDE lam_k / lam_q.  Reported in the log either way.
    lam_k_quantile: Optional[int] = None
    lam_q_quantile: Optional[int] = None

    # ---- late-stage threshold pruning (spkv/core/qknorm.py) ---------------------
    #
    # The group-LASSO shrinks the pair norms it dislikes but leaves every one of
    # them slightly non-zero, while deployment zeroes a pair outright.  After
    # ``prune_start_frac`` of the run, the SAME operator the evaluator uses is
    # applied INSIDE k_norm.forward, so the remaining steps train the network
    # against keys that are genuinely sparse.  None (the default) = off.
    #
    # ``prune_alpha`` is in units of the per-head median pair norm -- the same
    # units as run.eval_alphas and the same function (PairThreshold.refs), so
    # alpha=1.0 here means what alpha=1.0 means in the judge.
    #
    # tau is measured ONCE, at the transition step, from the model's own
    # distribution, and then frozen: the model must not be able to move its own
    # target.  The mask is recomputed every step, so a pair driven back above tau
    # returns rather than dying permanently.  (The filter layer is NOT trainable;
    # an earlier design description said "trainable threshold layer" -- what
    # exists is a frozen measured threshold.  Revisit only if a trainable variant
    # is actually wanted.)
    prune_start_frac: Optional[float] = None
    prune_alpha: float = 1.0
    prune_granularity: str = "head"

    def resolved(self) -> "PenaltyCfg":
        # Validate FIRST. An earlier version returned early on the tie_qk branch,
        # so with the default tie_qk=True a bad enum was never checked.
        if self.norm not in PENALTY_NORMS:
            raise ValueError(f"penalty.norm must be one of {PENALTY_NORMS}, got {self.norm!r}")
        if self.reduce not in PENALTY_REDUCE:
            raise ValueError(f"penalty.reduce must be one of {PENALTY_REDUCE}, got {self.reduce!r}")
        if self.prune_granularity not in PRUNE_GRANULARITIES:
            raise ValueError(
                f"penalty.prune_granularity must be one of {PRUNE_GRANULARITIES}, "
                f"got {self.prune_granularity!r}. 'token_head' is diagnostic-only "
                f"in the judge and is not a usable training threshold."
            )
        if self.prune_start_frac is not None and not 0.0 <= self.prune_start_frac < 1.0:
            raise ValueError(
                f"penalty.prune_start_frac is a fraction of the run and must be in "
                f"[0, 1), got {self.prune_start_frac!r}. Use null to disable."
            )
        if self.tie_qk:
            shared = max(self.lam_q, self.lam_k)
            return dataclasses.replace(self, lam_q=shared, lam_k=shared)
        return self


@dataclass
class DataCfg:
    # Native length, no packing, no concatenation (user-decided).
    seq_len: int = 4096
    max_len: int = 4096
    # Training documents and evaluation documents are drawn from DISJOINT index
    # ranges of the same split, or from different splits entirely.
    n_train_docs: int = 512
    n_eval_docs: int = 8
    eval_split: str = "train_tail"     # train_tail | validation
    # Keep only training documents whose tokenised length reaches seq_len
    # (scanning FORWARD from the block start).  Native-length batching otherwise
    # gives every micro-batch its own T; every loss term is a per-TOKEN mean, so
    # lambda itself is T-invariant -- but a constant T keeps the data-parallel
    # gradient average exactly token-weighted and keeps runs comparable
    # step-for-step.  Selection can take minutes at 40K (one tokenisation per
    # scanned document, cached for training afterwards).
    require_full_len: bool = False
    # Eval-side minimum: keep only documents with at least this many tokens.
    # REQUIRED for the 128K context -- without it an "eval at 128K" silently
    # scores whatever length the first validation books happen to have, truncated
    # at max_len.
    eval_min_tokens: Optional[int] = None
    sources: dict[str, str] = field(default_factory=lambda: {
        "pg19": "emozilla/pg19",
        "cot": "open-r1/Mixture-of-Thoughts",
        "code": "SWE-bench/SWE-smith-trajectories",
    })
    # NOTE: only pg19 has a loader. cot/code are listed here for the phase-2
    # corpus work and are NOT reachable from any code path today.


@dataclass
class TrainCfg:
    grad_accum: int = 4
    lr: float = 5e-5
    # adamw/sgd/lion only: the muon optimisers carry no decay term, and
    # validate() refuses a non-zero value with them rather than dropping it
    # silently.
    weight_decay: float = 0.0
    betas: tuple[float, float] = (0.9, 0.95)
    max_steps: int = 2000
    grad_clip: float = 1.0

    lr_schedule: str = "cosine"
    warmup_frac: float = 0.03
    min_lr_frac: float = 0.05

    optimizer: str = "muon"            # see OPTIMIZERS
    muon_lr: float = 0.02              # Muon's own scale, NOT comparable to Adam
    muon_momentum: float = 0.95
    # Per-optimizer LR for the MATRIX parameter group only.  ``lr`` is the
    # Adam-scale learning rate and ALSO drives the 1-D QK-Norm gain group, which
    # is AdamW in every configuration -- raising ``lr`` to an SGD scale to "make
    # SGD work" would detonate the gains, so SGD/Lion get their own key.  None
    # falls back to ``lr`` (the historical behaviour).
    sgd_lr: Optional[float] = None
    lion_lr: Optional[float] = None

    # Data-parallel degree.  1 = single process (historical behaviour).  >1
    # requires launching under torchrun with the SAME world size; gradients are
    # all-reduced (sum / world) before clipping, and each rank reads a disjoint
    # slice of the document stream.  Deliberately NOT torch DDP: only ~472M
    # params carry grads, and ``fp32_master`` rewrites ``p.data`` -- which would
    # leave a DDP reducer's buckets pointing at stale storages.
    dp_size: int = 1
    # Initialise the TRAINABLE weights (projections + QK-Norm gains) from a
    # previous run's checkpoint; frozen weights always come from the base model.
    # This is how the distillation-migration stages chain (stage B starts from
    # stage A's final.pt).  The optimizer starts fresh -- stages are separate
    # optimisations on purpose.
    init_ckpt: Optional[str] = None
    # "+"-joined names among MONITOR_TERMS: every log step, compute those
    # distillation terms under no_grad on the held-out documents and record them
    # (``mon_<term>``).  This is how a term that training STOPPED optimising is
    # watched for drift across a stage boundary.  Empty = off.
    monitor_terms: str = ""
    # One-shot per-term gradient-norm probe at step 0, reported as
    # ``probe_g_<term>`` for every term whose graph exists (active terms plus
    # the monitored ones, which are computed with grad for this single step).
    # Measures whether a newly activated term and the term it replaced pull with
    # comparable force at the switch.
    grad_probe: bool = False
    # Stride the QUERY capture (student AND teacher) that feeds the map term and
    # the q-side penalty.  At 40K the full q capture is ~336 MB/layer held
    # through backward (12 GiB across 36 layers, plus the same again in teacher
    # targets); at stride 8 it is 1.5 GiB.  Must divide attn_map_row_stride, and
    # when it EQUALS the row stride the map term is numerically unchanged (the
    # same rows are sampled either way); the q penalty becomes an unbiased
    # strided estimate of the same per-token mean.
    distill_q_stride: int = 1

    # Cross-entropy weight.  CE is the default anchor but is optional, because
    # in the attention-distillation configuration it is provably redundant.
    # Every frozen module (embeddings, v_proj, o_proj, both norms, the MLP,
    # lm_head) is byte-identical to the teacher's, and q/k enter the residual
    # stream only through attn_out = o_proj(a V).  So matching attention at every
    # layer already forces every hidden state -- and hence every logit -- to
    # match, and CE is a second gradient path to a quantity the attention term
    # already pins.  Set to 0.0 for the distillation-only run.
    ce_weight: float = 1.0
    attn_distill: bool = False
    lam_kd: float = 0.0                # logit KD weight (loss is a per-TOKEN mean)
    # Attention-MAP distillation (spkv/train/distill.py): KL(teacher map || student map)
    # on strided query rows, built from the DEPLOYED q/k.  This is the objective
    # the sparsity argument rests on -- dropping key pair p moves the logit by
    # q_p.k_p exactly, so matching the MAP forces that dot product to zero for
    # the pairs the LASSO shrinks, while matching only the attention OUTPUT would
    # let the student compensate in a rank-<=n_kv*d subspace and un-compensate as
    # soon as the mask changes.
    attn_map_distill: bool = False
    lam_map: float = 1.0
    # Rows of the [T, T] map computed at once.  The map is NEVER materialised in
    # full: the KL is accumulated per chunk, so the cost is O(chunk * T).
    attn_map_chunk: int = 64
    # Query rows are strided because each row's KL is an independent term, so a
    # strided subset is an unbiased estimator.  At 8 the whole term costs ~3% of
    # a model forward.  Keys are never strided -- that would sample the tail of a
    # distribution whose mass is concentrated in the head.
    attn_map_row_stride: int = 8

    # fp32 master copies of the trainable projections, forward under bf16
    # autocast. Without this an update of ~2.7e-5 lands on bf16 weights whose ULP
    # at |w|~1.5e-2 is 5.9e-5 -- below half an ULP, so it is rounded away or
    # snapped by a whole ULP. Measured: with V/O trainable this destroyed the
    # model (CE 2.48 -> 5.73 in ONE step).  Only applied when autocast is active
    # (CUDA); on CPU it is skipped rather than left in mismatched dtypes.
    fp32_master: bool = True
    # Chunked CE: HF's default loss calls logits.float() unconditionally, which at
    # 128K is ~67 GB of fp32 logits. Chunk over the time axis instead.
    ce_chunk: int = 1024
    # Position subsampling for the attention-OUTPUT distillation targets (teacher
    # side only; the student side follows with the same stride in Distiller).
    distill_sample_stride: int = 1
    gradient_checkpointing: bool = False
    # Only needed for the reentrant checkpoint path; the extra embedding hook it
    # installs breaks use_reentrant=False (saved-tensor mismatch).
    force_input_require_grads: bool = False
    # Where the frozen teacher lives (e.g. cuda:1 for a second GPU).
    teacher_device: str = "cpu"
    seed: int = 0

    # How close to a plateau counts as converged. Used by the trainer's report,
    # not enforced: held-out CE must move less than this over the last window.
    converge_window: int = 200
    converge_tol: float = 0.005


@dataclass
class BenchCfg:
    """What a ``run.mode: infer`` benchmark run needs (see ``spkv/eval/infer.py``).

    ``keeps`` are TARGET KEPT FRACTIONS, realised by bisecting the pruning ALPHA
    so the ENERGY THRESHOLD hits them.  They choose WHICH thresholds to deploy --
    the operator is still "one tau vs every (token, head, pair) norm", never a
    rank-based selection (see ``spkv/core/prune.py``'s header); the achieved
    fraction is measured back and reported per arm.
    """

    # Which tasks (names resolve in spkv/eval/tasks.py, deliberately NOT here:
    # config.py must not import the eval stack).
    tasks: tuple[str, ...] = ("hotpotqa", "2wikimqa", "musique", "lcc", "repobench-p")
    limits: dict[str, int] = field(default_factory=dict)  # per-task item cap
    limit: Optional[int] = None        # default item cap for every task
    start: int = 0
    shard: str = "0/1"                 # "i/n" over each task's items
    keeps: tuple[float, ...] = (1.0, 0.8, 0.5, 0.3, 0.16)
    alphas: tuple[float, ...] = ()     # extra explicit-alpha operating points
    keep_tol: float = 0.02             # warning threshold for missed keeps
    prune_granularity: str = "layer"   # one of PRUNE_GRANULARITIES
    prune_operator: str = "knorm"      # one of PRUNE_OPERATORS
    # None -> data.sources["pg19"] (the same indirection the judge uses);
    # a bare short name like "pg19" would hit datasets' script-loading refusal.
    calib_source: Optional[str] = None
    calib_split: str = "validation"
    calib_doc: int = 0
    calib_tokens: Optional[int] = None     # None -> data.seq_len
    calib_sample_stride: int = 4           # bisection probe stride (tokens)
    # Prompt fitting: budget None -> data.seq_len; "none" never truncates.
    max_prompt_tokens: Optional[int] = None
    truncate: str = "tail"                 # one of TRUNCATION_MODES
    max_new_tokens: dict[str, int] = field(default_factory=dict)  # per-task override
    include_original: bool = True
    seed: int = 0
    resume: bool = True


@dataclass
class SpKVConfig:
    run: RunCfg = field(default_factory=RunCfg)
    model: ModelCfg = field(default_factory=ModelCfg)
    penalty: PenaltyCfg = field(default_factory=PenaltyCfg)
    data: DataCfg = field(default_factory=DataCfg)
    train: TrainCfg = field(default_factory=TrainCfg)
    bench: BenchCfg = field(default_factory=BenchCfg)

    def validate(self) -> "SpKVConfig":
        if self.run.mode not in MODES:
            raise ValueError(f"run.mode must be one of {MODES}, got {self.run.mode!r}")
        if self.train.optimizer not in OPTIMIZERS:
            raise ValueError(f"train.optimizer must be one of {OPTIMIZERS}, got {self.train.optimizer!r}")
        if self.train.lr_schedule not in LR_SCHEDULES:
            raise ValueError(f"train.lr_schedule must be one of {LR_SCHEDULES}, got {self.train.lr_schedule!r}")
        if self.train.optimizer in ("muon", "block_muon") and self.train.weight_decay != 0.0:
            raise ValueError(
                "train.weight_decay is not implemented for muon/block_muon (their "
                "updates carry no decay term, so the value would be silently "
                "dropped); set it to 0.0 or use adamw/sgd/lion."
            )
        # Keep the RESOLVED penalty: tie_qk rewrites lam_q/lam_k, and every
        # consumer (loss, logs, the archived config.yaml) must see the same
        # values -- discarding this return once made the archive disagree with
        # the loss.
        self.penalty = self.penalty.resolved()
        # NOTE: penalty.prune_* values are legal in an eval config too -- pointing
        # the judge at the training config is the normal workflow.  They do not
        # steer the judge (the operator comes from the checkpoint payload); the
        # judge prints a notice to make that explicit rather than silent.
        if self.data.seq_len > self.data.max_len:
            raise ValueError(
                f"data.seq_len ({self.data.seq_len}) exceeds data.max_len "
                f"({self.data.max_len}); documents are truncated at max_len and "
                f"then sliced to seq_len, so seq_len must be the smaller of the two."
            )
        if (self.run.mode == "train" and self.train.ce_weight == 0.0
                and not self.train.attn_distill and not self.train.attn_map_distill
                and self.train.lam_kd == 0.0
                and self.penalty.lam_q == 0.0 and self.penalty.lam_k == 0.0
                and self.penalty.lam_k_quantile is None
                and self.penalty.lam_q_quantile is None):
            raise ValueError(
                "no objective is active: ce_weight=0, attn_distill=False, "
                "attn_map_distill=False, lam_kd=0 and no penalty. The model would "
                "receive no gradient."
            )
        if self.train.dp_size < 1:
            raise ValueError(f"train.dp_size must be >= 1, got {self.train.dp_size}")
        if self.train.distill_q_stride < 1:
            raise ValueError(f"train.distill_q_stride must be >= 1, got {self.train.distill_q_stride}")
        if self.train.distill_q_stride > 1 and self.train.attn_map_row_stride % self.train.distill_q_stride:
            # The map loss samples every (row_stride / q_stride)-th captured row;
            # a non-divisible pair would silently sample rows the capture never
            # kept -- a different estimator than the config states.
            raise ValueError(
                f"train.attn_map_row_stride ({self.train.attn_map_row_stride}) must "
                f"be a multiple of train.distill_q_stride ({self.train.distill_q_stride})"
            )
        parse_terms(self.train.monitor_terms)     # raises on an unknown name
        if (self.data.eval_min_tokens is not None
                and self.data.eval_min_tokens > self.data.max_len):
            raise ValueError(
                f"data.eval_min_tokens ({self.data.eval_min_tokens}) exceeds "
                f"data.max_len ({self.data.max_len}); documents are truncated at "
                f"max_len, so no document could ever qualify."
            )
        if self.run.eval_rope_scaling is not None:
            rs = self.run.eval_rope_scaling
            if not isinstance(rs, dict) or "rope_type" not in rs or "factor" not in rs:
                raise ValueError(
                    f"run.eval_rope_scaling must be a dict with at least 'rope_type' "
                    f"and 'factor' (e.g. {{rope_type: yarn, factor: 4.0, "
                    f"original_max_position_embeddings: 40960}}), got {rs!r}"
                )
        for name in ("sgd_lr", "lion_lr"):
            v = getattr(self.train, name)
            if v is not None and v <= 0.0:
                raise ValueError(f"train.{name} must be > 0 when set, got {v}")
        if self.run.mode == "infer":
            b = self.bench
            if b.truncate not in TRUNCATION_MODES:
                raise ValueError(
                    f"bench.truncate must be one of {TRUNCATION_MODES}, got "
                    f"{b.truncate!r}")
            if not b.tasks:
                raise ValueError("bench.tasks is empty: nothing to run")
            for k in b.keeps:
                if not 0.0 < k <= 1.0:
                    raise ValueError(f"bench.keeps entries must be in (0, 1], got {k}")
            for a in b.alphas:
                if a < 0.0:
                    raise ValueError(f"bench.alphas entries must be >= 0, got {a}")
            parse_shard(b.shard)                       # raises when malformed
            if b.prune_granularity not in PRUNE_GRANULARITIES:
                raise ValueError(
                    f"bench.prune_granularity must be one of {PRUNE_GRANULARITIES}, "
                    f"got {b.prune_granularity!r} (token_head is diagnostic-only "
                    f"and cannot be deployed to generation)")
            if b.prune_operator not in PRUNE_OPERATORS:
                raise ValueError(
                    f"bench.prune_operator must be one of {PRUNE_OPERATORS}, "
                    f"got {b.prune_operator!r}")
            if b.max_prompt_tokens is not None and b.max_prompt_tokens <= 0:
                raise ValueError(
                    f"bench.max_prompt_tokens must be > 0 or null, got "
                    f"{b.max_prompt_tokens}")
            if b.calib_tokens is not None and b.calib_tokens <= 0:
                raise ValueError(
                    f"bench.calib_tokens must be > 0 or null, got {b.calib_tokens}")
            if b.calib_sample_stride < 1:
                raise ValueError(
                    f"bench.calib_sample_stride must be >= 1, got {b.calib_sample_stride}")
        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SpKVConfig":
        raw: dict[str, Any] = yaml.safe_load(Path(path).read_text()) or {}
        cfg = cls()
        for section_name, section in raw.items():
            if not hasattr(cfg, section_name):
                raise KeyError(f"unknown config section: {section_name}")
            target = getattr(cfg, section_name)
            for k, v in (section or {}).items():
                if not hasattr(target, k):
                    raise KeyError(f"unknown key {section_name}.{k}")
                if isinstance(v, list):
                    v = tuple(v)
                setattr(target, k, v)
        return cfg.validate()

    def to_yaml(self, path: str | Path) -> None:
        def enc(o: Any) -> Any:
            if dataclasses.is_dataclass(o):
                return {f.name: enc(getattr(o, f.name)) for f in dataclasses.fields(o)}
            if isinstance(o, tuple):
                return list(o)
            return o

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(yaml.safe_dump(enc(self), sort_keys=False, allow_unicode=True))
