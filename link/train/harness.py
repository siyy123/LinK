"""Training: the Trainer (the single training loop) and everything it assembles.

``run.mode: train`` -> ``Trainer(cfg).train()``.  There is one loop and one
objective assembler; the four training recipes are config switches on the same
graph (``ce_weight`` / ``attn_map_distill`` / ``attn_distill`` / ``lam_kd``),
each orthogonally combinable with the L1 penalty and the late-stage threshold
layer.

``L = ce_weight * CE + lam_d * attn_out + lam_kd * KD + lam_map * map + pen``

Two defects of an earlier generation of this file are fixed here and are worth
stating plainly:

1. **The evaluation set was inside the training cycle.**  ``eval_ids`` used to be
   the last rows of the same dataset the loop cycled through, so those documents
   were trained on.  The reported "held-out" CE fell to ~1.0 at exactly the steps
   where the eval documents entered the cycle -- a cliff, not a trend -- and was
   read as the model recovering.  Eval now comes from
   :class:`spkv.core.data.Corpus`, whose disjointness is asserted at
   construction.
2. **The penalty was normalised against a reference measured in different
   units.**  The step-0 reference used to be measured on the RAW ``k_proj``
   output while the penalty under the normalised parametrisation acts on unit
   vectors, so ``lambda`` in the loop was silently two orders of magnitude
   weaker than the same ``lambda`` elsewhere.  The reference is now measured
   **in the same parametrisation the penalty sees** (the ``QKNormStack`` captures).
"""

from __future__ import annotations

import contextlib
import json
import os
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from torch import nn

from ..config import SpKVConfig, parse_terms
from ..core.ce import ChunkedCrossEntropy
from ..core.checkpoints import Checkpoint
from ..core.data import Corpus, NativeLengthDocs
from ..core.metrics import PairStats
from ..core.qknorm import PruneSpec, QKNormStack
from ..core.runtime import Runtime
from .distill import AttentionCapture, Distiller, Teacher
from .loss import GroupLassoRope, PenaltyNormalizer, SpKVLoss, layer_penalty
from .optim import LRSchedule, OptimizerFactory

__all__ = ["Trainer"]


class Trainer:
    """The training loop.  ``Trainer(cfg).train()`` is the whole entry point.

    State assembled once in :meth:`train` (model, datasets, calibrations,
    optimizer, schedule, criterion) lives on ``self``; every helper below is a
    method so there is exactly one place each decision is made.

    Data parallelism (``train.dp_size > 1``, launched under torchrun) is a
    MANUAL all-reduce of the trainable gradients before clipping -- deliberately
    not ``torch.nn.parallel.DistributedDataParallel``: only ~472M of 4B params
    carry gradients, and ``fp32_master`` rewrites ``p.data`` after the model is
    built, which would leave a DDP reducer's buckets glued to stale storages.
    Rank 0 alone does calibration, logging, evaluation and checkpointing; every
    rank runs the same number of collectives per step.
    """

    def __init__(self, cfg: SpKVConfig):
        self.cfg = cfg
        self.rank, self.world, self.is_main, self._local_rank = self._init_distributed()
        self.device = self._resolve_device()
        self.rt = Runtime(cfg, self.device)
        self.monitor_terms = parse_terms(cfg.train.monitor_terms)
        self._last_grad_norm: float | None = None
        self.ce = ChunkedCrossEntropy(cfg.train.ce_chunk)
        self.model: nn.Module | None = None
        self.tokenizer = None
        self.train_ds = None
        self.eval_ds = None
        self.eval_ids: list[torch.Tensor] = []
        self.n_q = self.n_kv = self.hd = 0
        self.prune_spec: PruneSpec | None = None
        self.prune_tau: dict = {}
        self.normalizer: PenaltyNormalizer | None = None
        self.teacher: Teacher | None = None
        self.distiller: Distiller | None = None
        self.capture: AttentionCapture | None = None
        self.criterion: SpKVLoss | None = None
        self.optimizer = None
        self.history: list[tuple[int, float]] = []
        self.last_eval_ce: float | None = None
        self.schedule = LRSchedule(cfg.train.warmup_frac, cfg.train.lr_schedule,
                                   cfg.train.min_lr_frac)
        self.optim_factory = OptimizerFactory(cfg.train)

    # ------------------------------------------------------------- distributed

    def _init_distributed(self) -> tuple[int, int, bool, int]:
        """Join the process group when launched under torchrun.

        The world size in the environment and ``train.dp_size`` must AGREE: a
        run that meant to be data-parallel but was started without torchrun would
        otherwise silently train on one rank, and one launched under torchrun with
        dp_size=1 would have each rank train the whole thing redundantly.
        """
        import os

        world = int(os.environ.get("WORLD_SIZE", "1"))
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if world != int(self.cfg.train.dp_size):
            raise ValueError(
                f"WORLD_SIZE={world} from the environment does not match "
                f"train.dp_size={self.cfg.train.dp_size}; launch with "
                f"torchrun --nproc_per_node={self.cfg.train.dp_size} or fix the config."
            )
        if world == 1:
            return 0, 1, True, 0
        import torch.distributed as dist

        if not (dist.is_available() and torch.cuda.is_available()):
            raise RuntimeError("train.dp_size > 1 needs torch.distributed with NCCL and CUDA")
        torch.cuda.set_device(local_rank)
        # device_id pins the NCCL communicator to this rank's GPU (torch warns
        # and guesses from the current device otherwise).
        dist.init_process_group("nccl", device_id=torch.device(f"cuda:{local_rank}"))
        rank = dist.get_rank()
        print(f"[dp] rank {rank}/{world} on cuda:{local_rank} (nccl)", flush=True)
        return rank, world, rank == 0, local_rank

    def _shutdown_distributed(self) -> None:
        if self.world == 1:
            return
        import torch.distributed as dist

        if dist.is_initialized():
            dist.destroy_process_group()

    def _resolve_device(self) -> torch.device:
        if self.world > 1:
            # run.device is IGNORED under DP: letting both ranks read one value
            # would put them on the same GPU, which fails late and confusingly.
            return torch.device(f"cuda:{self._local_rank}")
        if self.cfg.run.device:
            return torch.device(self.cfg.run.device)
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------- assembly

    def train(self) -> Path:
        cfg = self.cfg
        torch.manual_seed(cfg.train.seed)

        out_dir = Path(cfg.run.out_dir)
        if self.is_main:
            out_dir.mkdir(parents=True, exist_ok=True)
            cfg.to_yaml(out_dir / "config.yaml")

        self.tokenizer = self.rt.tokenizer()
        self._build_model()
        self._maybe_fp32_masters()
        self._load_init_ckpt()
        self._broadcast_weights()
        self.n_q, self.n_kv, self.hd = Runtime.head_dims(self.model)
        n_layers = len(self.model.model.layers)
        print(f"[model] {n_layers} layers | q_heads={self.n_q} kv_heads={self.n_kv} "
              f"head_dim={self.hd}")

        self.train_ds, self.eval_ds = Corpus(cfg, self.tokenizer).train_eval()
        Corpus.assert_disjoint(self.train_ds, self.eval_ds)
        tr, ev = self.train_ds.index_range, self.eval_ds.index_range
        print(f"[data] train rows {len(tr)} from {tr[0]}..{tr[-1]} | eval rows "
              f"{self.eval_ds.split}:{len(ev)} from {ev[0]}..{ev[-1]} (disjoint)")

        steps = cfg.run.max_steps if cfg.run.max_steps is not None else cfg.train.max_steps
        if cfg.penalty.prune_start_frac is not None:
            self.prune_spec = PruneSpec(cfg.penalty.prune_start_frac,
                                        cfg.penalty.prune_alpha,
                                        cfg.penalty.prune_granularity)
            print(f"[prune] late-stage threshold layer: {self.prune_spec} "
                  f"(arms at step {self.prune_spec.start_step(steps)}/{steps})")

        calib_ids = [self.train_ds[i]["input_ids"][: cfg.data.seq_len].unsqueeze(0).to(self.device)
                     for i in range(min(cfg.train.grad_accum, len(self.train_ds)))]
        if self.is_main:
            # Only rank 0 measures: the references must be ONE set of numbers by
            # construction, not by "the ranks probably computed the same float",
            # and rank 0 is the only writer of config.resolved.yaml.
            self._calibrate_penalties(calib_ids, out_dir)
        self._sync_calibration()

        need_teacher = (cfg.train.attn_distill or cfg.train.lam_kd > 0
                        or cfg.train.attn_map_distill or bool(self.monitor_terms)
                        or cfg.train.grad_probe)
        if need_teacher:
            self.teacher = Teacher(cfg.model.name_or_path, device=cfg.train.teacher_device,
                                   dtype=getattr(torch, cfg.model.dtype),
                                   attn_implementation=cfg.model.attn_implementation)
            print(f"[teacher] frozen copy on {cfg.train.teacher_device}")
        else:
            print("[loss] no distillation term active -> teacher not loaded")
        self.distiller = Distiller(self.n_q, self.n_kv, self.hd,
                                   chunk=cfg.train.attn_map_chunk,
                                   row_stride=cfg.train.attn_map_row_stride,
                                   sample_stride=cfg.train.distill_sample_stride)
        # AttentionCapture observes post-o_proj outputs — the attention-OUTPUT
        # distillation target.  The MAP distillation reads the QKNormStack
        # captures instead (they are the deployed q/k), so the capture is only
        # attached when the output term is on.  grad_probe also needs it: the
        # probe measures the output term's gradient even when it is inactive.
        self.capture = AttentionCapture(self.model, require_grad=True) \
            if cfg.train.attn_distill else None

        print(f"[loss] ce_weight={cfg.train.ce_weight} + attn_distill={cfg.train.attn_distill} "
              f"+ attn_map_distill={cfg.train.attn_map_distill}"
              f"(chunk={cfg.train.attn_map_chunk}, row_stride={cfg.train.attn_map_row_stride}, "
              f"q_stride={cfg.train.distill_q_stride}) "
              f"+ kd={cfg.train.lam_kd} "
              f"+ penalties(lam_q={cfg.penalty.lam_q}, lam_k={cfg.penalty.lam_k}, "
              f"norm={cfg.penalty.norm}, ramp={cfg.penalty.ramp_frac}) "
              f"+ monitors={self.monitor_terms or None}")

        pen = cfg.penalty.resolved()
        self.criterion = SpKVLoss(pen, cfg.train, self.distiller)
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        self.trainable = trainable
        self.optimizer = self.optim_factory.build(trainable, self.hd)

        self.eval_ids = [self.eval_ds[i]["input_ids"][: cfg.data.seq_len].unsqueeze(0).to(self.device)
                         for i in range(len(self.eval_ds))]
        base_ce = None
        if self.is_main:
            base_ce = self._eval_ce(self.eval_ids)
            print(f"[eval] held-out CE at init: {base_ce:.4f}")

        self._run_steps(steps, calib_ids, trainable)

        if self.teacher is not None:
            self.teacher.free()
        if not self.is_main:
            self._shutdown_distributed()
            return None
        prune_extra = self._prune_checkpoint_extra()
        ckpt = Checkpoint(out_dir / "final.pt").save(
            self.model, steps,
            extra={"eval_ce": self.last_eval_ce,
                   "base_ce": base_ce,
                   **prune_extra,
                   # Recorded so the judge can refuse to score a legacy
                   # checkpoint from the removed spherical parametrisation.
                   "k_scale_mode": "qk_norm"}).path
        self._report_convergence(self.history)
        self._shutdown_distributed()
        if self.device.type == "cuda":
            print(f"[mem] peak allocated {torch.cuda.max_memory_allocated(self.device)/2**30:.2f} GiB "
                  f"| reserved {torch.cuda.max_memory_reserved(self.device)/2**30:.2f} GiB")
        print(f"[done] saved {ckpt}")
        return ckpt

    def _build_model(self) -> None:
        cfg = self.cfg
        self.model = self.rt.build_model()
        if cfg.train.gradient_checkpointing:
            # use_reentrant=False is REQUIRED: the reentrant variant runs the
            # forward under no_grad, so our hooks capture detached tensors and
            # the distillation loss contributes exactly zero gradient, silently.
            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs=self._gradient_checkpointing_kwargs(cfg.train))
            if cfg.train.force_input_require_grads and hasattr(self.model, "enable_input_require_grads"):
                self.model.enable_input_require_grads()

        # The model's OWN per-head norms stay trainable: they are the exact
        # quantity the penalty must not be able to pay itself down with (it reads
        # the pre-gain vector, so d(pen)/d(gain) == 0), which is why they can be
        # trained by the CE/distillation terms alone.  Leaving them frozen would
        # not be wrong, but it would make the model strictly weaker than the one
        # being deployed.
        extra = ("q_norm.weight", "k_norm.weight")
        n_train = self._freeze_except_projections(cfg.model.trainable_projections, extra=extra)
        print(f"[model] trainable params: {n_train / 1e9:.3f} B "
              f"({', '.join(cfg.model.trainable_projections)}, {', '.join(extra)})")

    def _maybe_fp32_masters(self) -> None:
        """Cast the trainables to fp32.  Separate from _build_model because the
        init-checkpoint load MUST come after it: a checkpoint stores fp32 masters,
        and loading them into bf16 parameters first would round-trip every weight
        through bf16 before the upcast -- silently lossy, at exactly the scale
        (ULP ~5.9e-5) the master copy exists to protect."""
        cfg = self.cfg
        if not cfg.train.fp32_master:
            return
        if self.rt.use_autocast:
            # fp32 master copies; the forward runs under bf16 autocast so the
            # matmuls stay fast.  This is what keeps an update of ~2.7e-5 from
            # being rounded away by the bf16 ULP (measured: without it, CE
            # 2.48 -> 5.73 in ONE step).
            for p in self.model.parameters():
                if p.requires_grad:
                    p.data = p.data.float()
            print("[model] trainable params kept in fp32 (master weights)")
        else:
            # The cast is only meaningful when bf16 autocast is actually on
            # (CUDA).  Unconditionally casting on CPU left the frozen bf16
            # weights and the fp32 trainables in different dtypes, and the
            # first q_proj matmul raised "m1 and m2 must have the same dtype".
            print("[model] fp32_master requested but autocast is off "
                  "(non-CUDA device) -- skipped")

    def _load_init_ckpt(self) -> None:
        """Stage chaining: re-initialise the TRAINABLE weights from a checkpoint.

        Only the trainable names are copied; frozen weights always come from the
        base model, so a stage cannot smuggle in a different teacher.  The
        optimizer starts fresh -- the stages in the migration experiment are
        separate optimisations on purpose (a fresh optimizer state is part of
        what "switching objective" means here).
        """
        cfg = self.cfg
        if not cfg.train.init_ckpt:
            return
        path = Path(cfg.train.init_ckpt)
        if not path.exists():
            raise FileNotFoundError(f"train.init_ckpt does not exist: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        ck_mode = payload.get("k_scale_mode")
        if ck_mode and ck_mode != "qk_norm":
            raise ValueError(
                f"init_ckpt {path} was trained with k_scale_mode={ck_mode!r}; this "
                f"trainer only knows the native-QK-Norm parametrisation."
            )
        state = payload.get("model", {})
        want = {n for n, p in self.model.named_parameters() if p.requires_grad}
        sub = {n: t for n, t in state.items() if n in want}
        missing = sorted(want - set(sub))
        if missing:
            raise ValueError(
                f"init_ckpt {path} is missing {len(missing)} trainable tensors, "
                f"e.g. {missing[:5]}"
            )
        self.model.load_state_dict(sub, strict=False)
        print(f"[init] loaded {len(sub)} trainable tensors from {path} "
              f"(step {payload.get('step')})")

    def _broadcast_weights(self) -> None:
        """DP: rank 0's trainables are the truth.

        Every rank loads the same files, so the values are identical in practice;
        broadcasting makes identity a construction rather than an assumption (and
        covers the init_ckpt path).  Cheap: 1.9 GB once.
        """
        if self.world == 1:
            return
        import torch.distributed as dist

        n = 0
        with torch.no_grad():
            for p in self.model.parameters():
                if p.requires_grad:
                    dist.broadcast(p.data, src=0)
                    n += 1
        print(f"[dp] broadcast {n} trainable tensors from rank 0")

    def _sync_calibration(self) -> None:
        """DP: share rank 0's step-0 references and the resolved lambdas."""
        if self.world == 1:
            return
        import torch.distributed as dist

        pen = self.cfg.penalty.resolved()
        vals = torch.tensor([pen.lam_q, pen.lam_k], device=self.device, dtype=torch.float64)
        dist.broadcast(vals, src=0)
        self.cfg.penalty.lam_q = float(vals[0])
        self.cfg.penalty.lam_k = float(vals[1])
        if self.normalizer is None:
            self.normalizer = PenaltyNormalizer(
                num_layers=len(self.model.model.layers), names=("q", "k")).to(self.device)
        with torch.no_grad():
            for name in ("q", "k"):
                dist.broadcast(getattr(self.normalizer, f"{name}_ref"), src=0)
        print(f"[dp] calibration synced (lam_q={self.cfg.penalty.lam_q:.6g}, "
              f"lam_k={self.cfg.penalty.lam_k:.6g})")

    @staticmethod
    def _gradient_checkpointing_kwargs(train_cfg) -> dict:
        """Kwargs that do NOT silently kill the distillation gradient."""
        return {"use_reentrant": False} if train_cfg.gradient_checkpointing else {}

    def _freeze_except_projections(self, projections: tuple[str, ...],
                                   extra: tuple[str, ...] = ()) -> int:
        """Freeze everything except the named attention projections (and ``extra``).

        Everything else — embeddings, RMSNorm, MLP, lm_head, rotary — is frozen.
        ``extra`` lists suffix-matched parameters that also stay trainable (the
        model's own per-head norm gains, which are part of the unit being
        trained).
        """
        for name, p in self.model.named_parameters():
            p.requires_grad_(
                any(name.endswith(f".{proj}.weight") for proj in projections)
                or any(name.endswith(f".{e}") for e in extra)
            )
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    @torch.no_grad()
    def _forward_qk(self, ids: torch.Tensor) -> dict[str, dict[int, torch.Tensor]]:
        """One forward; per-layer q/k in the parametrisation the penalty sees.

        That tensor is the model's own RMSNorm output, pre-gain (u), captured by
        :class:`QKNormStack`.  There is exactly one such tensor — a reference
        measured in a different parametrisation from the penalty's is the units
        bug this method exists to prevent.

        The q capture uses ``train.distill_q_stride``: the references are only
        comparable if they are measured on the SAME (possibly strided) tensor the
        penalty will read in the loop.
        """
        stack = QKNormStack(self.model, self.n_q, self.n_kv, self.hd,
                            q_capture_stride=self.cfg.train.distill_q_stride)
        with stack, self.rt.autocast():
            self.model(input_ids=ids, use_cache=False)
        return {"q": {i: s.q for i, s in enumerate(stack.state) if s.q is not None},
                "k": {i: s.k for i, s in enumerate(stack.state) if s.k is not None}}

    @torch.no_grad()
    def _calibrate_penalties(self, calib_ids, out_dir: Path, batches: int = 4) -> None:
        """Step-0 per-layer P_q / P_k, measured IN THE PARAMETRISATION THE PENALTY SEES.

        At step 0 the student is the pre-trained model, so these are its natural
        per-layer scales and dividing by them makes every layer's term comparable
        in size.  Also resolves ``lam_*_quantile`` (a target surviving-pair count
        expressed as the corresponding pair-norm quantile), re-applies ``tie_qk``,
        and archives the resolved config.
        """
        cfg = self.cfg
        q_pen = GroupLassoRope(self.n_q, self.hd)
        k_pen = GroupLassoRope(self.n_kv, self.hd)
        sums: dict[str, dict[int, torch.Tensor]] = {"q": {}, "k": {}}
        for i in range(batches):
            ids = calib_ids[i % len(calib_ids)]
            src = self._forward_qk(ids)
            for name, fn in (("q", q_pen), ("k", k_pen)):
                for layer, act in src[name].items():
                    val = fn(act)
                    prev = sums[name].get(layer)
                    sums[name][layer] = val if prev is None else prev + val

        norm = PenaltyNormalizer(num_layers=len(self.model.model.layers), names=("q", "k"))
        for name in ("q", "k"):
            for layer, total in sums[name].items():
                norm.set_reference(name, layer, total / batches)
        refs_k = [float(v) / batches for v in sums["k"].values()] if sums["k"] else [0.0]
        if refs_k and max(refs_k) > 0:
            print(f"[calib] step-0 refs (qk_norm): P_k in [{min(refs_k):.2f}, {max(refs_k):.2f}]")
        self.normalizer = norm.to(self.device)

        if cfg.penalty.lam_k_quantile or cfg.penalty.lam_q_quantile:
            acts = self._forward_qk(calib_ids[0])
            for name, heads, want in (("k", self.n_kv, cfg.penalty.lam_k_quantile),
                                      ("q", self.n_q, cfg.penalty.lam_q_quantile)):
                if not want:
                    continue
                lam = PairStats(heads, self.hd).quantile_lambda(list(acts[name].values()), want)
                setattr(cfg.penalty, f"lam_{name}", lam)
                print(f"[lambda] lam_{name} = {lam:.6g}  "
                      f"(target: {want} of {self.hd // 2} pairs survive per head)")
            # Re-tie after the quantile override: tie_qk rewrites lam_q/lam_k, and
            # the archived config must show the values the loss actually multiplies.
            cfg.penalty = cfg.penalty.resolved()
            cfg.to_yaml(out_dir / "config.resolved.yaml")
            del acts

    # ------------------------------------------------------------- the loop

    def _run_steps(self, steps: int, calib_ids, trainable) -> None:
        cfg = self.cfg
        out_dir = Path(cfg.run.out_dir)
        self.model.train()
        t0 = time.time()
        log_path = out_dir / "log.jsonl"
        micro = 0
        metrics: dict[str, float] = {}
        prune_start_step = self.prune_spec.start_step(steps) if self.prune_spec else None
        if prune_start_step is not None:
            print(f"[prune] threshold layer arms at step {prune_start_step}/{steps}")

        for step in range(steps):
            lr = self.schedule.apply(self.optimizer, step, steps)
            self.optimizer.zero_grad(set_to_none=True)
            if step == 0 and os.environ.get("SPKV_MEM_SNAPSHOT") and self.device.type == "cuda":
                # Opt-in allocation tracing (see the OOM handler in _step): the
                # long-context budget problems have twice been a term whose
                # per-layer graph nobody counted, and the snapshot names it.
                torch.cuda.memory._record_memory_history(max_entries=200_000)

            if prune_start_step is not None and step >= prune_start_step and not self.prune_tau:
                # Measure tau ONCE, from an UNMASKED forward, BEFORE any masking
                # has happened anywhere.  Measuring lazily inside the masked
                # forward would read layer i's k from already-masked upstream
                # keys, giving a different answer for every layer but the first —
                # and disagreeing with the evaluator, which sees no mask at all.
                # (DP: rank 0 measures, then every rank is handed the same tau —
                # a per-rank measurement would be identical only by luck of
                # floating point, not by construction.)
                if self.is_main:
                    with self.rt.sdpa(calib_ids[0].shape[1]):
                        measured = QKNormStack.measure_taus(
                            self.model, self.n_q, self.n_kv, self.hd, calib_ids[0],
                            self.prune_spec, amp=self.rt.autocast())
                    self.prune_tau.update(measured)
                    med = float(torch.cat([v.reshape(-1) for v in measured.values()]).median())
                    print(f"[prune] step {step}: froze tau for {len(measured)} layers "
                          f"(alpha={self.prune_spec.alpha}, {self.prune_spec.granularity}, "
                          f"median tau={med:.4f})")
                self._sync_taus()

            prof = None
            if os.environ.get("SPKV_PROFILE") and step == 1 and self.is_main:
                # Opt-in: torch.profiler around ONE step, top CUDA ops printed.
                # (8K steps measured >60 s where ~25 s was expected; guessing at
                # the split between teacher / student / captures wastes hours.)
                prof = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA])
                prof.start()

            for _ in range(cfg.train.grad_accum):
                # DP: disjoint document streams -- the local micro counter is
                # rank-synchronous, so the assignment is deterministic.
                doc = self.train_ds[(micro * self.world + self.rank) % len(self.train_ds)]
                ids = NativeLengthDocs.collate([doc])["input_ids"][
                    :, : cfg.data.seq_len
                ].to(self.device)
                micro += 1
                metrics = self._step(ids, step, steps)

            if self.world > 1:
                self._all_reduce_grads(trainable)
            will_log = step % cfg.run.log_every == 0 or step == steps - 1
            snap = [p.detach().clone() for p in trainable] if will_log and self.is_main else None
            if cfg.train.grad_clip:
                # clip_grad_norm_ RETURNS the pre-clip total norm; logging it is
                # the difference between "the LR is what the config says" and
                # "the clip has silently been the effective LR for 2000 steps".
                self._last_grad_norm = float(
                    torch.nn.utils.clip_grad_norm_(trainable, cfg.train.grad_clip))
            else:
                self._last_grad_norm = self._total_norm(trainable)
            self.optimizer.step()
            upd_norm = self._update_norm(snap, trainable) if snap is not None else None

            if cfg.train.grad_probe and step == 0 and self.is_main:
                # BEFORE the log block, or the values would be merged into
                # ``metrics`` only after the step-0 line was already written
                # (the line is built from the metrics dict above).
                metrics.update(self._probe_terms_offline(ids))

            if step % cfg.run.log_every == 0 or step == steps - 1:
                if self.is_main:
                    ce = self._eval_ce(self.eval_ids)
                    self.history.append((step, ce))
                    self.last_eval_ce = ce
                    rec = {"step": step, "lr": round(lr, 8),
                           # Both groups, because train.lr drives the always-AdamW
                           # gain group while the matrix group may run at muon_lr/
                           # sgd_lr/lion_lr -- one number cannot describe both.
                           "lrs": [round(float(g["lr"]), 10) for g in self.optimizer.param_groups],
                           **metrics,
                           "grad_norm": round(self._last_grad_norm, 6),
                           "eval_ce": round(ce, 4),
                           "elapsed": round(time.time() - t0, 1)}
                    if upd_norm is not None:
                        rec["update_norm"] = round(upd_norm, 8)
                    if self.device.type == "cuda":
                        # Peak-so-far, on every logged step: the 40K budget is
                        # tight enough that headroom must be a number in the log,
                        # not an estimate in a plan.
                        rec["peak_gib"] = round(
                            torch.cuda.max_memory_allocated(self.device) / 2**30, 2)
                    rec.update(self._monitor_terms())
                    rec.update(self._step_diagnostics())
                    with log_path.open("a") as fh:
                        fh.write(json.dumps(rec) + "\n")
                    print(f"[step {step}] {json.dumps(rec)}", flush=True)

            if cfg.run.ckpt_every and step > 0 and step % cfg.run.ckpt_every == 0 and self.is_main:
                Checkpoint.write_rotating(out_dir, "step", step, self.model, cfg.run.ckpt_keep)
            if prof is not None:
                prof.stop()
                print(prof.key_averages().table(
                    sort_by="cuda_time_total", row_limit=14), flush=True)
        if self.world > 1:
            import torch.distributed as dist
            dist.barrier()

    def _step(self, ids: torch.Tensor, step: int, total_steps: int) -> dict[str, float]:
        """One micro-batch: forward with the captures, assemble the loss, backward."""
        cfg = self.cfg
        q_stride = cfg.train.distill_q_stride
        if self.capture is not None:
            # RE-ENABLE at the start of every micro-batch.  The capture is
            # switched off around backward (see below) and that flag lives on
            # the trainer-owned capture object, so forgetting this reset made
            # micro-batch 2 read micro-batch 1's captured tensors -- whose graph
            # had already been backwarded and freed -- and torch aborted with
            # "Trying to backward through the graph a second time" (hit first by
            # E3 stage B, where the output term is active).
            self.capture.capture_enabled = True
            self.capture.clear()

        tgt = t_qk = t_hidden = None
        mem: dict[str, float] = {} if step == 0 and self.device.type == "cuda" else None
        if self.teacher is not None:
            t_out = self.teacher.forward_all(
                ids, sample_stride=cfg.train.distill_sample_stride,
                need_attn=cfg.train.attn_distill,
                need_logits=False,
                need_qk=cfg.train.attn_map_distill,
                need_hidden=cfg.train.lam_kd > 0,
                qk_q_stride=q_stride,
            )
            if cfg.train.attn_distill:
                tgt = {k: v.to(self.device, non_blocking=True) for k, v in t_out["attn"].items()}
            if cfg.train.attn_map_distill:
                t_qk = {k: (a.to(self.device, non_blocking=True), b.to(self.device, non_blocking=True))
                        for k, (a, b) in t_out["qk"].items()}
            if cfg.train.lam_kd > 0:
                t_hidden = t_out["hidden"].to(self.device, non_blocking=True)
        if mem is not None:
            mem["targets"] = torch.cuda.memory_allocated(self.device) / 2**30

        cap_ctx = self.capture.attach() if self.capture is not None else nullcontext()
        # set_step must run BEFORE the forward: it is what arms the late-stage
        # mask.  ``prune_tau`` is the frozen store measured once above; the stack
        # never measures tau itself.
        stack = QKNormStack(self.model, self.n_q, self.n_kv, self.hd,
                            prune=self.prune_spec, tau_store=self.prune_tau,
                            q_capture_stride=q_stride)
        stack.set_step(step, total_steps)

        # EVERYTHING that touches the graph must stay inside this context --
        # including backward().  With gradient checkpointing, backward RECOMPUTES
        # each layer's forward; if the patches have already been removed by then,
        # the recomputed graph is the un-patched one and torch aborts with
        # "A different number of tensors was saved during the original forward
        # and recomputation".
        with stack as sph, cap_ctx, self.rt.autocast(), self.rt.sdpa(ids.shape[1]):
            # ALWAYS the backbone.  The CausalLM path materialises the FULL logits
            # tensor unconditionally (131072*151936*2 B = 33.6 GB per micro-batch
            # at 128K, 12.5 GB at 40K) for a tensor neither the chunked CE nor the
            # chunked KD reads -- both only need the hidden states.
            hidden = self.model.model(input_ids=ids, use_cache=False).last_hidden_state
            if mem is not None:
                mem["fwd"] = torch.cuda.memory_allocated(self.device) / 2**30

            q_src = {l: s.q for l, s in enumerate(sph.state) if s.q is not None}
            k_src = {l: s.k for l, s in enumerate(sph.state) if s.k is not None}
            q_pen = GroupLassoRope(self.n_q, self.hd)
            k_pen = GroupLassoRope(self.n_kv, self.hd)
            # Computed unconditionally, even when lambda is 0: the loss skips a
            # zero-weight term anyway, and a control run that reports pen_k = 0
            # tells you nothing about whether the pair-norm distribution drifts
            # on its own. That is exactly the baseline the treated run must beat.
            # ``layer_penalty`` = the same scalar inside a checkpointed region:
            # the per-layer norm graph is O(T x hidden) and 36 of them retained
            # until backward saturated the card at 32K (see its docstring).
            penalties = {"q": {l: layer_penalty(q_pen, a) for l, a in q_src.items()},
                         "k": {l: layer_penalty(k_pen, a) for l, a in k_src.items()}}

            cos = sin = None
            mdist = None
            if cfg.train.attn_map_distill and t_qk:
                cos, sin = Distiller.rope_cos_sin(self.model, ids)
                # The deployed q/k (post-norm-gain, mask included when live): the
                # map is built from the tensors attention actually consumes, NOT
                # from the raw q_proj/k_proj outputs.  Both sides were captured
                # with the SAME query stride.
                mdist = self.distiller.attn_map(sph.deployed_qk(), t_qk, cos, sin,
                                                q_stride=q_stride)

            loss, metrics = self.criterion(
                hidden=hidden, input_ids=ids, lm_head=self.model.lm_head,
                student_attn=dict(self.capture.outputs) if self.capture is not None else None,
                teacher_attn=tgt, student_hidden=hidden, teacher_hidden=t_hidden,
                penalties=penalties, normalizer=self.normalizer, step=step,
                total_steps=total_steps, map_distill=mdist,
            )
            if mem is not None and mdist is not None:
                mem["map"] = torch.cuda.memory_allocated(self.device) / 2**30
            # The loss is fully built; from here the capture STORES must stop.
            # With gradient checkpointing the backward re-runs every layer's
            # forward, and the still-installed patches/hooks would otherwise
            # overwrite the stored tensors with recompute-graph tensors, pinning
            # ~1.4 GiB per layer until the backward ends (measured: 50 GB at 32K).
            # The patches themselves stay installed -- the recompute requires
            # them (a different number of saved tensors aborts it).
            sph.set_capture(False)
            if self.capture is not None:
                self.capture.capture_enabled = False
            try:
                (loss / cfg.train.grad_accum).backward()
            except torch.OutOfMemoryError:
                if os.environ.get("SPKV_MEM_SNAPSHOT"):
                    path = Path(cfg.run.out_dir) / "mem_snapshot.pickle"
                    torch.cuda.memory._dump_snapshot(str(path))
                    print(f"[mem] OOM: snapshot dumped to {path}", flush=True)
                raise
            finally:
                if mem is not None:
                    # finally, so the breakdown survives an OOM in backward --
                    # which is exactly the case it exists to explain.
                    mem["bwd_peak"] = torch.cuda.max_memory_allocated(self.device) / 2**30
                    metrics.update({f"mem_{k}": round(v, 2) for k, v in mem.items()})
                    print(f"[mem] {json.dumps({k: round(v, 2) for k, v in mem.items()})}",
                          flush=True)

        self._last_k = k_src.get(0)
        self._last_q = q_src.get(0)
        self._last_report = sph.report()
        return metrics

    # ------------------------------------------------- distributed / probing

    def _all_reduce_grads(self, trainable) -> None:
        """Sum the per-rank gradients and divide by the world size.

        Deliberately NOT DDP (see the class docstring).  A parameter may
        legitimately carry NO gradient in a given configuration: the penalty
        reads the PRE-gain capture, so ``d(pen)/d(gain) == 0`` by construction
        and a penalty-only run never touches the QK-Norm gains.  What must not
        happen is a parameter that has a gradient on SOME ranks but not others —
        the all-reduce would then average different objectives.  The presence
        bitmask is checked with two tiny collectives, and only the present grads
        are reduced.
        """
        import torch.distributed as dist

        mask = torch.tensor([1 if p.grad is not None else 0 for p in trainable],
                            device=self.device, dtype=torch.int32)
        lo, hi = mask.clone(), mask.clone()
        dist.all_reduce(lo, op=dist.ReduceOp.MIN)
        dist.all_reduce(hi, op=dist.ReduceOp.MAX)
        if not torch.equal(lo, hi):
            bad = [i for i in range(len(trainable)) if int(lo[i]) != int(hi[i])]
            raise RuntimeError(
                f"[dp] {len(bad)} trainable parameters have a gradient on some ranks "
                f"but not others (indices {bad[:8]}); the all-reduce would average "
                f"different objectives."
            )
        for p in trainable:
            if p.grad is None:
                continue
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
            p.grad.div_(self.world)

    def _sync_taus(self) -> None:
        """DP: hand rank 0's frozen tau to every rank, shape included."""
        if self.world == 1 or not self.prune_tau:
            return
        import torch.distributed as dist

        for i in range(len(self.model.model.layers)):
            if self.is_main:
                t = self.prune_tau.get(i)
                meta = torch.tensor([0, 0, 0, 0, 0], device=self.device, dtype=torch.long)
                if t is not None:
                    meta = torch.tensor([t.numel(), *t.shape], device=self.device, dtype=torch.long)
            else:
                meta = torch.zeros(5, device=self.device, dtype=torch.long)
            dist.broadcast(meta, src=0)
            if int(meta[0]) == 0:
                continue
            shape = [int(x) for x in meta[1:]]
            if i not in self.prune_tau:
                self.prune_tau[i] = torch.zeros(shape, device=self.device, dtype=torch.float32)
            dist.broadcast(self.prune_tau[i].reshape(-1), src=0)

    @staticmethod
    def _total_norm(params) -> float:
        sq = 0.0
        for p in params:
            if p.grad is not None:
                sq += float(p.grad.detach().pow(2).sum())
        return sq ** 0.5

    @staticmethod
    def _update_norm(snap, params) -> float:
        sq = 0.0
        for p, s in zip(params, snap):
            sq += float((p.detach() - s).pow(2).sum())
        return sq ** 0.5

    def _probe_terms_offline(self, ids: torch.Tensor, window: int = 2048) -> dict[str, float]:
        """``|| d(lambda_i * term_i) / d theta ||`` per distillation term.

        The "loss matching" measurement at a stage switch: whether the term
        being switched ON pulls with a force comparable to the one being
        switched off.  Each term gets its OWN short forward + backward:

        * in-graph (one forward, several ``autograd.grad`` with retain_graph)
          is a trap here -- over a gradient-checkpointed graph every retained
          pass re-runs and RETAINS all 36 layers' recomputation, and the probe
          OOMed a 72 GB card at four terms;
        * the window is truncated to ``window`` tokens because the quantity
          wanted is a RATIO between terms, which a 2K window estimates at a
          fraction of the long-context cost.  Absolute values are NOT comparable
          to a full-length step and are not used as if they were.
        """
        cfg = self.cfg
        ids = ids[:, : min(window, ids.shape[1])]
        q_stride = cfg.train.distill_q_stride
        t_out = self.teacher.forward_all(
            ids, sample_stride=cfg.train.distill_sample_stride,
            need_attn=True, need_logits=False, need_qk=True, need_hidden=True,
            qk_q_stride=q_stride)
        tgt = {k: v.to(self.device) for k, v in t_out["attn"].items()}
        t_qk = {k: (a.to(self.device), b.to(self.device)) for k, (a, b) in t_out["qk"].items()}
        t_hidden = t_out["hidden"].to(self.device)
        cap = AttentionCapture(self.model, require_grad=True)   # output term needs it

        def build(name):
            from .distill import Distiller as _D
            stack = QKNormStack(self.model, self.n_q, self.n_kv, self.hd,
                                q_capture_stride=q_stride)
            stack.set_step(0, 1)
            with cap.attach(), stack, self.rt.autocast(), self.rt.sdpa(ids.shape[1]):
                hidden = self.model.model(input_ids=ids, use_cache=False).last_hidden_state
                if name == "pen":
                    # RAW (unweighted, unramped) penalty, reduce-consistent: the
                    # probe reports raw gradient norms for every term, so lambda
                    # and the ramp are deliberately NOT folded in.
                    q_pen = GroupLassoRope(self.n_q, self.hd)
                    k_pen = GroupLassoRope(self.n_kv, self.hd)
                    q_src = {l: s.q for l, s in enumerate(stack.state) if s.q is not None}
                    k_src = {l: s.k for l, s in enumerate(stack.state) if s.k is not None}
                    pen = None
                    for fn, src in ((q_pen, q_src), (k_pen, k_src)):
                        vals = [layer_penalty(fn, a) for a in src.values()]
                        if not vals:
                            continue
                        term = sum(vals)
                        if self.criterion.cfg.reduce == "sum":
                            term = term * len(vals)
                        pen = term if pen is None else pen + term
                    return pen
                if name == "map":
                    cos, sin = _D.rope_cos_sin(self.model, ids)
                    return self.distiller.attn_map(stack.deployed_qk(), t_qk, cos, sin,
                                                   q_stride=q_stride)
                if name == "output":
                    return self.distiller.attn_output(dict(cap.outputs), tgt)
                if name == "kd":
                    return self.distiller.logit_kd_chunked(hidden, t_hidden,
                                                           self.model.lm_head, chunk=cfg.train.ce_chunk)
                raise ValueError(name)

        out: dict[str, float] = {}
        active = {"pen": True,
                  "map": bool(cfg.train.attn_map_distill),
                  "output": bool(cfg.train.attn_distill),
                  "kd": cfg.train.lam_kd > 0}
        for name in ("pen", "map", "output", "kd"):
            # A term is probed when it is ACTIVE (what training multiplies) or
            # WATCHED (a stage's retired objective -- the very thing a stage
            # switch wants to compare against).  Skipping watched terms cost the
            # B->C switch its "output" row once; mon_* covers their VALUES but
            # only the probe covers their gradient scale.
            if not (active[name] or name in self.monitor_terms):
                continue
            term = build(name)
            if term is None or not torch.is_tensor(term):
                continue
            grads = torch.autograd.grad(term, self.trainable, allow_unused=True)
            sq = 0.0
            for g in grads:
                if g is not None:
                    sq += float(g.detach().pow(2).sum())
            # RAW gradient norm (no lambda weight) for every term, so the
            # numbers are directly comparable across stages and configurations;
            # the weighted force is this number times the configured weight.
            out[f"probe_g_{name}"] = round(sq ** 0.5, 6)
            del term, grads
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
        return out

    @torch.no_grad()
    def _monitor_terms(self) -> dict[str, float]:
        """Values of the monitored terms on the held-out documents, every log step.

        The training loss only ever contains the ACTIVE terms, so a term whose
        objective was switched away would otherwise be invisible — and "did the
        old objective drift after the switch" is the measurement the migration
        experiment exists for.  Runs on the FIXED eval documents, so the curve is
        comparable across steps and across the chained runs.

        Uses its own ``AttentionCapture(require_grad=False)``: the training-side
        capture asserts a grad_fn and would raise under ``no_grad``.
        """
        if not self.monitor_terms or not self.eval_ids:
            return {}
        if self.teacher is None:
            # The trainer loads the teacher whenever monitor_terms is non-empty;
            # reaching this line means that wiring broke, and a silent {} would
            # look exactly like "the monitored terms are all zero".
            raise RuntimeError(
                "train.monitor_terms is set but no teacher was loaded — the "
                "monitored terms would silently be absent from every log line."
            )
        cfg = self.cfg
        q_stride = cfg.train.distill_q_stride
        cap = AttentionCapture(self.model, require_grad=False)
        sums: dict[str, float] = {t: 0.0 for t in self.monitor_terms}
        n = 0
        for ids in self.eval_ids:
            ids = ids[:, : cfg.data.seq_len]
            t_out = self.teacher.forward_all(
                ids, sample_stride=cfg.train.distill_sample_stride,
                need_attn="output" in self.monitor_terms,
                need_logits=False,
                need_qk="map" in self.monitor_terms,
                need_hidden="kd" in self.monitor_terms,
                qk_q_stride=q_stride,
            )
            stack = QKNormStack(self.model, self.n_q, self.n_kv, self.hd,
                                q_capture_stride=q_stride)
            with cap.attach(), stack, self.rt.autocast(), self.rt.sdpa(ids.shape[1]):
                hidden = self.model.model(input_ids=ids, use_cache=False).last_hidden_state
            if "map" in self.monitor_terms:
                t_qk = {k: (a.to(self.device), b.to(self.device))
                        for k, (a, b) in t_out["qk"].items()}
                cos, sin = Distiller.rope_cos_sin(self.model, ids)
                sums["map"] += float(self.distiller.attn_map(
                    stack.deployed_qk(), t_qk, cos, sin, q_stride=q_stride))
            if "output" in self.monitor_terms:
                tgt = {k: v.to(self.device) for k, v in t_out["attn"].items()}
                sums["output"] += float(self.distiller.attn_output(dict(cap.outputs), tgt))
            if "kd" in self.monitor_terms:
                t_hidden = t_out["hidden"].to(self.device)
                sums["kd"] += float(self.distiller.logit_kd_chunked(
                    hidden, t_hidden, self.model.lm_head, chunk=cfg.train.ce_chunk))
            cap.clear()
            n += 1
        return {f"mon_{k}": round(v / max(n, 1), 6) for k, v in sums.items()}

    # ------------------------------------------------------------ diagnostics

    def _step_diagnostics(self) -> dict:
        """The per-log-step record: pair concentration, mask behaviour, gain drift."""
        rec: dict = {}
        k_src, q_src = getattr(self, "_last_k", None), getattr(self, "_last_q", None)
        stats_k = PairStats(self.n_kv, self.hd)
        if k_src is not None:
            d = stats_k.concentration(k_src)
            if not PairStats.r_in_range(d["r"], self.hd // 2):
                # r = mean_pairs / ||pairs||_2 is an l1/l2 ratio of the SAME
                # tensor: 1 <= r <= sqrt(P).  Outside that range the numerator
                # and denominator came from different tensors, and every
                # layer-0 diagnostic on this line is unusable.
                print(f"[WARNING] r_l0={d['r']:.3f} is outside [1, sqrt(hd/2)] -- "
                      f"the layer-0 diagnostics are broken, not just noisy.")
            rec.update(pair_cv_l0=round(d["cv"], 4), e_top10_l0=round(d["e_top10"], 4),
                       r_l0=round(d["r"], 4),
                       usage_gini_l0=round(stats_k.pair_usage_gini(k_src), 4))
            if q_src is not None:
                rec["qk_ratio_l0"] = round(PairStats.qk_head_ratio(q_src, k_src), 4)
            # The gain is exactly the token-independent reweighting the penalty
            # cannot see (it reads the pre-gain u); if the pre-/post-gain
            # profiles drift apart, the sparsity that was trained is not the
            # sparsity that can be pruned.  Measured every log step so the drift
            # is visible before it matters.
            gain = self.model.model.layers[0].self_attn.k_norm.weight.detach().float()
            rec["gain_min"] = round(float(gain.min()), 4)
            rec["gain_max"] = round(float(gain.max()), 4)
            rec["gain_pen_vs_effective_rank_corr"] = round(
                stats_k.gain_effective_rank_corr(k_src, gain), 4)
        report = getattr(self, "_last_report", None)
        if report and self.prune_tau:
            # Realised post-mask keep fraction, and the tail mass the mask
            # discarded — read from the PRE-mask tensor.  Together they say
            # whether the threshold layer is doing anything and what it costs.
            rec.update({k: round(float(v), 5) for k, v in report.items()})
            # The deployed (masked) model's own held-out CE, next to the unmasked
            # one: the gap is the cost of the sparsity actually being trained.
            rec["eval_ce_pruned"] = round(self._eval_ce(self.eval_ids, masked=True), 4)
        return rec

    # ------------------------------------------------------------- evaluation

    @torch.no_grad()
    def _eval_ce(self, eval_ids, masked: bool = False) -> float:
        """Held-out CE with the SAME chunked scorer the judge uses.

        ``masked=True`` runs the forward with the frozen tau installed in
        k_norm — the CE of the model as it will be deployed, not of the dense
        model it was trained from.
        """
        cfg = self.cfg
        was_training = self.model.training
        self.model.eval()
        if masked and self.prune_tau:
            ctx = QKNormStack(self.model, self.n_q, self.n_kv, self.hd,
                              fixed_tau=self.prune_tau)
        else:
            ctx = contextlib.nullcontext()

        vals = []
        for ids in eval_ids:
            ids = ids[:, : cfg.data.seq_len]
            with ctx, self.rt.autocast(), self.rt.sdpa(ids.shape[1]):
                h = self.model.model(input_ids=ids, use_cache=False).last_hidden_state
            vals.append(float(self.ce(h, ids, self.model.lm_head)))
        self.model.train(was_training)
        if self.device.type == "cuda":
            # At 40K an eval pass is a multi-GB transient whose blocks the
            # training step cannot always reuse; returning them to the driver
            # is a few milliseconds against an OOM with no headroom left.
            torch.cuda.empty_cache()
        return sum(vals) / max(len(vals), 1)

    # ------------------------------------------------------------- checkpoint

    def _prune_checkpoint_extra(self) -> dict:
        """tau and the transition step, so the judge can reproduce the exact operator.

        ``checkpoint_extra`` only reads the spec and the frozen tau store (both
        set at construction time), so no forward is needed here.
        """
        if self.prune_spec is None or not self.prune_tau:
            return {}
        stack = QKNormStack(self.model, self.n_q, self.n_kv, self.hd,
                            prune=self.prune_spec, tau_store=self.prune_tau)
        extra = stack.checkpoint_extra()
        print(f"[prune] frozen tau recorded for {len(extra.get('prune_tau', {}))} layers")
        return extra

    def _report_convergence(self, history: list[tuple[int, float]]) -> None:
        """Say plainly whether the run plateaued, so "converged" is not asserted loosely."""
        if len(history) < 2:
            return
        cfg = self.cfg
        w = cfg.train.converge_window
        tail = [(s, v) for s, v in history if s >= history[-1][0] - w]
        if len(tail) < 2:
            print(f"[converge] not enough eval points in the last {w} steps to judge")
            return
        drift = max(v for _, v in tail) - min(v for _, v in tail)
        ok = drift < cfg.train.converge_tol
        print(f"[converge] held-out CE moved {drift:.4f} over the last {w} steps "
              f"(tol {cfg.train.converge_tol}) -> "
              f"{'CONVERGED' if ok else 'NOT converged; train longer'}")
