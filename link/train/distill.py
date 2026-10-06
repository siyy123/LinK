"""Distillation: the frozen teacher, the captures, and the three objectives.

The three objectives, in decreasing order of how load-bearing they are for the
sparsity claim:

* **attention-MAP** — ``KL(teacher_map || student_map)`` on strided query rows,
  built from the tensors attention actually consumes.  This is the objective the
  sparsity argument rests on: dropping key pair ``p`` moves the logit for the
  ``(i, j)`` entry by exactly ``q_ip . k_jp``, so matching the MAP forces that
  dot product to zero for the pairs the LASSO shrinks, while matching only the
  attention OUTPUT would let the student compensate in a rank-``<= n_kv * d``
  subspace and un-compensate as soon as the mask changes.
* **attention-OUTPUT** — ``cosine + 0.1 * magnitude`` between post-``o_proj``
  attention outputs.  A proxy; structurally at odds with sparsity, hence off by
  default.
* **logit KD** — ``KL(teacher || student)`` on the LM head, mean over TOKENS so
  ``lam_kd`` weighs the same at every sequence length (it used to be a
  ``batchmean``, which at batch 1 is a per-position SUM and silently diluted the
  term by ``T``).

Where the MAP's q/k come from
-----------------------------
From :meth:`spkv.core.qknorm.QKNormStack.deployed_qk` — ``k_norm.weight * u``,
i.e. the tensor ``Qwen3Attention`` uses at the point just before RoPE (mask
included when the late-stage threshold layer is live).  An earlier version hooked
``q_proj``/``k_proj`` directly and distilled from the RAW projection outputs,
which skip the per-head RMSNorm and its gain: that proxy map differs from the
deployed map by whole nats (measured: per-head RMS 0.05 raw vs 1.6 deployed;
``KL(deployed || proxy)`` = 5.6 nats averaged over 36 layers), so the term was
grading a distribution neither model computes.  The teacher side captures through
the same stack, so both sides see one definition of "the map".
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..core.qknorm import QKNormStack
from ..core.runtime import Runtime

__all__ = ["AttentionCapture", "Teacher", "Distiller"]


class AttentionCapture:
    """Forward hook that records each decoder layer's ``self_attn`` output.

    That output is post-``o_proj`` ``[B, T, hidden]`` — the attention-OUTPUT
    distillation target.  (The q/k captures live in
    :class:`spkv.core.qknorm.QKNormStack`, which sees the tensors the model
    actually computes with; this class deliberately only observes.)

    The ``require_grad`` assertion exists because a captured tensor with no
    ``grad_fn`` would make the distillation loss contribute exactly zero
    gradient, silently.  It fires the moment gradient checkpointing is enabled
    with the reentrant variant (which runs the forward under ``no_grad``).
    """

    def __init__(self, model: nn.Module, require_grad: bool = True):
        self.model = model
        self.require_grad = require_grad
        self.outputs: dict[int, Tensor] = {}
        # Storing can be switched off while the hooks stay installed -- the same
        # reason and mechanism as QKNormStack.set_capture: during a
        # gradient-checkpointed backward the layers run AGAIN, and storing the
        # recomputed outputs pins each layer's recompute graph until the
        # backward ends (36 x ~200 MB at 40K).
        self.capture_enabled = True
        self._handles: list[Any] = []
        self._layers = self._find_layers(model)

    @staticmethod
    def _find_layers(model: nn.Module) -> list[nn.Module]:
        """Locate the decoder layer list without hard-coding the prefix."""
        for name, mod in model.named_modules():
            if name.endswith("layers") and isinstance(mod, nn.ModuleList):
                return list(mod)
        raise RuntimeError("could not locate the decoder layer ModuleList")

    def _mk_hook(self, idx: int):
        def hook(_module, _inputs, output):
            # LlamaAttention returns (attn_output, attn_weights); keep [0].
            t = output[0] if isinstance(output, (tuple, list)) else output
            if self.require_grad and not (t.requires_grad and t.grad_fn is not None):
                raise RuntimeError(
                    f"captured attention output at layer {idx} has no grad_fn — the "
                    "distillation loss would silently contribute zero gradient. "
                    "Use gradient_checkpointing_kwargs={'use_reentrant': False} and "
                    "call enable_input_require_grads()."
                )
            if self.capture_enabled:
                self.outputs[idx] = t
        return hook

    def attach(self) -> "AttentionCapture":
        self.detach()
        for i, layer in enumerate(self._layers):
            self._handles.append(layer.self_attn.register_forward_hook(self._mk_hook(i)))
        return self

    def detach(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def clear(self) -> None:
        self.outputs.clear()

    def __enter__(self) -> "AttentionCapture":
        return self.attach()

    def __exit__(self, *exc) -> None:
        self.detach()


class Teacher:
    """Frozen copy of the pre-trained model, used only to produce targets."""

    def __init__(self, name_or_path: str, device: str | torch.device = "cpu",
                 dtype: torch.dtype = torch.bfloat16, attn_implementation: str = "sdpa"):
        from ..config import SpKVConfig

        # A throwaway config keeps ``Runtime.build_model``'s signature uniform.
        cfg = SpKVConfig()
        cfg.model.name_or_path = name_or_path
        cfg.model.attn_implementation = attn_implementation
        self.cfg = cfg
        self.rt = Runtime(cfg, device)
        self.device = self.rt.device
        self.model = self.rt.build_model(dtype=dtype)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.n_q, self.n_kv, self.hd = Runtime.head_dims(self.model)
        self.capture = AttentionCapture(self.model, require_grad=False)
        self._freed = False

    def free(self) -> None:
        """Drop the teacher and hand its device memory back.

        The teacher exists only to produce targets.  The layer-wise loop used to
        compute every target once up front and cache it, after which the teacher
        -- a second full copy of the model -- was dead weight holding a device the
        training itself could be using.
        """
        was = str(self.device)
        self.capture.detach()
        self.model = None
        self._freed = True
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"[teacher] released {was} (no longer needed)")

    def _require(self) -> None:
        if getattr(self, "_freed", False) or self.model is None:
            raise RuntimeError(
                "teacher has been freed -- targets must be computed before Teacher.free()"
            )

    @torch.no_grad()
    def forward_all(
        self,
        input_ids: Tensor,
        sample_stride: int = 1,
        need_attn: bool = False,
        need_logits: bool = False,
        need_qk: bool = False,
        need_hidden: bool = False,
        qk_q_stride: int = 1,
    ) -> dict:
        """One teacher pass returning whichever targets are requested.

        ``need_qk`` returns the per-layer DEPLOYED q/k (post-norm-gain, pre-RoPE,
        ``[B, T', H*D]``) via a :class:`QKNormStack` over the teacher — the map
        distillation applies RoPE itself, so pre-RoPE is the right hand-off point.
        With ``qk_q_stride > 1`` the QUERY rows are captured strided (the K side is
        never strided: the map needs every key), which is what makes the 40K
        target payload 4.5 GB instead of 15 GB — and it is only equivalent to the
        unstrided path when the LOSS uses the same stride (row indices for the
        causal mask are ``captured_index * stride``).

        ``need_hidden`` returns the END hidden state ``[B, T, hidden]`` — 0.2 GB
        at 40K, the cheap shipping unit for logit KD.  The LM head is applied
        chunk-wise by the consumer.

        ``need_logits`` materialises the FULL logits tensor and is legacy
        (short-context smokes only): bf16 logits are 12.5 GB at 40K and the
        consumer upcasts to fp32, which is an instant OOM.  When it is off — the
        only configuration long-context runs use — the backbone alone is run and
        the LM head never executes.
        """
        from contextlib import nullcontext

        self._require()
        self.capture.clear()
        out: dict = {}
        ids = input_ids.to(self.device)
        ctx = self.capture.attach() if need_attn else nullcontext()
        stack = QKNormStack(self.model, self.n_q, self.n_kv, self.hd,
                            q_capture_stride=qk_q_stride) if need_qk else nullcontext()
        with ctx, stack:
            if need_logits:
                out["logits"] = self.model(input_ids=ids, use_cache=False).logits.detach()
            else:
                hidden = self.model.model(input_ids=ids, use_cache=False).last_hidden_state
                if need_hidden:
                    out["hidden"] = hidden.detach()
        if need_attn:
            out["attn"] = {i: t[:, ::sample_stride].detach()
                           for i, t in self.capture.outputs.items()}
        if need_qk:
            out["qk"] = {i: (q.detach(), k.detach())
                         for i, (q, k) in stack.deployed_qk().items()}
        return out

    @torch.no_grad()
    def attention_outputs(self, input_ids: Tensor, sample_stride: int = 1) -> dict[int, Tensor]:
        """Per-layer attention outputs (post-``o_proj``), subsampled on positions.

        ``sample_stride`` must match whatever the student side uses, or the cosine
        will compare mismatched positions.
        """
        self._require()
        self.capture.clear()
        with self.capture.attach():
            self.model(input_ids=input_ids.to(self.device), use_cache=False)
        return {i: t[:, ::sample_stride].detach() for i, t in self.capture.outputs.items()}


class Distiller:
    """The three distillation objectives, bound to one model geometry.

    Holds only knobs (chunk size, row stride, position stride, magnitude weight)
    and the q/k head geometry; the teacher lifecycle belongs to :class:`Teacher`.
    """

    def __init__(self, num_q_heads: int, num_kv_heads: int, head_dim: int, *,
                 chunk: int = 64, row_stride: int = 8, sample_stride: int = 1,
                 mag_weight: float = 0.1):
        self.n_q = int(num_q_heads)
        self.n_kv = int(num_kv_heads)
        self.hd = int(head_dim)
        # Rows of the [T, T] map computed at once.  The map is NEVER materialised
        # in full: cost is O(chunk * T) instead of O(T^2), so 64 rows at T=8192 is
        # 50 MB per layer regardless of length.
        self.chunk = int(chunk)
        # Query rows are strided because each row's KL is an independent term, so a
        # strided subset is an unbiased estimator.  Keys are never strided -- that
        # would sample the tail of a distribution whose mass is in the head.
        self.row_stride = int(row_stride)
        self.sample_stride = int(sample_stride)
        self.mag_weight = float(mag_weight)

    # ------------------------------------------------------------- map distill

    @staticmethod
    def reshape_heads(x: Tensor, num_heads: int, head_dim: int) -> Tensor:
        """``[B, T, H*D] -> [B, H, T, D]``."""
        b, t, _ = x.shape
        return x.view(b, t, num_heads, head_dim).transpose(1, 2)

    @staticmethod
    def _apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

        return apply_rotary_pos_emb(x, x, cos, sin, unsqueeze_dim=1)[0]

    @staticmethod
    def rope_cos_sin(model, ids: Tensor) -> tuple[Tensor, Tensor]:
        """``(cos, sin)`` for ``ids``'s positions, from the model's own rotary module.

        Uses a zeros tensor rather than ``embed_tokens(ids)``: the rotary module
        only reads shape/dtype/device from its argument, and the embedding lookup
        at 128K is a real cost for no benefit.
        """
        b, t = ids.shape
        position_ids = torch.arange(t, device=ids.device).unsqueeze(0).expand(b, t)
        x = torch.zeros(b, t, model.config.hidden_size, device=ids.device,
                        dtype=model.model.embed_tokens.weight.dtype)
        return model.model.rotary_emb(x, position_ids)

    @staticmethod
    def _expand_gqa(k: Tensor, n_rep: int) -> Tensor:
        """kv head j serves the CONTIGUOUS q block [j*n_rep, (j+1)*n_rep).

        ``repeat_interleave`` gives exactly that ordering; a plain repeat would
        not, and the mismatch would be silent -- shapes stay valid either way.
        """
        return k.repeat_interleave(n_rep, dim=1) if n_rep > 1 else k

    @classmethod
    def _chunk_logits(cls, q_rot: Tensor, k_rot: Tensor, scale: float,
                      rows_block: Tensor, keys: Tensor,
                      orig_rows: Tensor | None = None) -> Tensor:
        """Masked logits for ``rows_block`` queries against every key -> ``[B,H,c,T]``.

        The causal mask is applied by index rather than by an additive bias
        tensor, so no [T, T] mask is ever allocated.  Half precision is upcast
        BEFORE the softmax: bf16 softmax over T terms loses the small
        probabilities that carry a long-context retrieval signal.  (Done
        unconditionally it would silently downcast a caller's float64.)

        ``rows_block`` indexes into ``q_rot``'s position axis; ``orig_rows`` is
        where those positions sit in the ORIGINAL sequence, when the query axis
        was captured strided (``orig = captured_index * stride``).  Masking on
        the captured index instead would let a row attend to keys that come after
        it — a silently wrong causal structure, not an error.
        """
        logits = (q_rot[:, :, rows_block] @ k_rot.transpose(-1, -2)) * scale
        mask_rows = rows_block if orig_rows is None else orig_rows
        blocked = keys.unsqueeze(0) > mask_rows.unsqueeze(1)        # [c,T]
        logits = logits.masked_fill(blocked, torch.finfo(logits.dtype).min)
        if logits.dtype in (torch.bfloat16, torch.float16):
            logits = logits.float()
        return logits

    @staticmethod
    def _group_kl_sum(q_g: Tensor, k_g: Tensor, tq_g: Tensor, tk_g: Tensor,
                      cos_q: Tensor, sin_q: Tensor, cos_k: Tensor, sin_k: Tensor,
                      scale: float, rows: Tensor, keys: Tensor, q_stride: int,
                      chunk: int, eps: float) -> Tensor:
        """One kv-head group's KL sum over ALL sampled row chunks (checkpointed unit).

        EVERYTHING rank-``T`` happens here -- RoPE included -- and it is all
        recomputed at backward time.  Both halves of that are load-bearing:

        * a checkpointed region keeps its inputs alive, so anything computed
          outside and passed in is retained for ALL layers until the joint
          backward (measured, in order: per-chunk teacher rows -> 1.07 GiB/layer
          at 8K; per-layer roped q/k for 36 layers -> 9 GiB at 40K).  The
          caller therefore passes VIEWS of tensors that are alive anyway -- the
          q/k captures and the shared rotary tables;
        * the unit is the WHOLE GROUP, not one chunk.  With per-chunk units the
          rope had to be re-applied per chunk, which is 16x redundant on the
          full-length tensors: profiling an 8K step showed ~18k rope calls and
          ~700k kernels per step, 53 s of CPU launch time against 76 s of GPU
          work.  One checkpoint per group ropes once (4 ropes per group) and
          loops the chunks INSIDE the region, where their graphs are discarded
          after the forward and rebuilt during backward exactly as before.
        """
        q_rot = Distiller._apply_rope(q_g, cos_q, sin_q)
        k_rot = Distiller._apply_rope(k_g, cos_k, sin_k)
        tq_rot = Distiller._apply_rope(tq_g, cos_q, sin_q)
        tk_rot = Distiller._apply_rope(tk_g, cos_k, sin_k)
        total = None
        for s in range(0, rows.numel(), chunk):
            blk = rows[s: s + chunk]
            orig = blk * q_stride
            s_rows = torch.softmax(
                Distiller._chunk_logits(q_rot, k_rot, scale, blk, keys, orig_rows=orig), dim=-1)
            with torch.no_grad():
                t_rows = torch.softmax(
                    Distiller._chunk_logits(tq_rot, tk_rot, scale, blk, keys, orig_rows=orig), dim=-1)
            # kl = sum t * (log t - log s); 0*log0 is taken as 0 by clamping t.
            kl = t_rows * (torch.log(t_rows.clamp_min(eps)) - torch.log(s_rows.clamp_min(eps)))
            term = kl.sum()
            total = term if total is None else total + term
        return total

    @staticmethod
    def _stride_rotary(cos: Tensor, sin: Tensor, stride: int) -> tuple[Tensor, Tensor]:
        """``(cos, sin)`` subsampled on the POSITION axis, for a strided q capture.

        Position ``j`` of the captured tensor is original position ``j * stride``;
        feeding the unstrided tables raises (broadcast mismatch between [B,1,T,D]
        and [B,H,T',D]) for T' != T, but would silently BROADCAST for T' == 1 --
        so this is not left to the caller's discipline.
        """
        if stride <= 1:
            return cos, sin
        return cos[:, ::stride], sin[:, ::stride]

    @classmethod
    def _rows_map(cls, q: Tensor, k: Tensor, cos: Tensor, sin: Tensor, scale: float,
                  chunk: int, row_stride: int) -> Tensor:
        """Attention probability rows for a strided subset of queries (dense form).

        ``q``: [B, n_q, T, D] PRE-RoPE.  ``k``: [B, n_kv, T, D] PRE-RoPE.
        Returns ``[B, n_q, r, T]`` with ``r = ceil(T / row_stride)``.

        This materialises the strided rows — it is the TEST path, pinning the row
        computation against a dense reference.  The loss path below uses the same
        :meth:`_chunk_logits` but accumulates KL per chunk and never holds more
        than one chunk.
        """
        q = cls._apply_rope(q, cos, sin)
        k = cls._expand_gqa(cls._apply_rope(k, cos, sin), q.shape[1] // k.shape[1])
        t = q.shape[2]
        rows = torch.arange(0, t, row_stride, device=q.device)
        keys = torch.arange(t, device=q.device)
        out = []
        for s in range(0, rows.numel(), chunk):
            out.append(torch.softmax(
                cls._chunk_logits(q, k, scale, rows[s: s + chunk], keys), dim=-1))
        return torch.cat(out, dim=2)

    def attn_map(self, student_qk: Mapping[int, tuple[Tensor, Tensor]],
                 teacher_qk: Mapping[int, tuple[Tensor, Tensor]],
                 cos: Tensor, sin: Tensor, eps: float = 1e-8,
                 q_stride: int = 1) -> Tensor:
        """``mean_layers KL(teacher_map || student_map)`` over strided query rows.

        ``student_qk``/``teacher_qk``: ``{layer: (q, k)}`` with q of shape
        ``[B, T/q_stride, H*D]`` (the QUERY axis may be captured strided — see
        ``QKNormStack(q_capture_stride=...)``) and k of shape ``[B, T, H*D]``
        (keys are never strided), both in the PRE-RoPE, DEPLOYED parametrisation
        (post-norm-gain, mask included when live).  The student tensors carry
        grad; the teacher's are computed under ``no_grad``.

        Two sampling levels, and the equivalence between them is the point:
        captured row ``j`` is original row ``j * q_stride``, and this loss takes
        every ``row_stride / q_stride``-th captured row, so the SAME original
        rows are sampled whether the capture was strided or not — with
        ``q_stride == row_stride`` the two configurations are numerically
        identical (pinned by test), which is what lets the 40K runs trade 8x of
        capture memory for nothing.  The causal mask uses the ORIGINAL row index.

        Forward KL (teacher || student), the usual distillation direction: it is
        mass-covering, so rows where the teacher concentrates must be matched
        rather than ignored.  A reverse KL would happily let the student drop
        exactly the sharp rows that long-context retrieval depends on.

        Memory, measured rather than assumed (the earlier version of this
        docstring claimed the graph was bounded -- it was not: at 36 layers and
        T=8192 the per-chunk softmax/log graphs summed to O(T^2/stride) retained
        until backward and OOMed a 47 GB card; at T=40960 the same happened
        inside the training probe):

        * each chunk is computed inside a CHECKPOINTED region -- RoPE included --
          so what survives the forward is one scalar per chunk, not the chunk's
          rank-``T`` intermediates, and the only tensors passed in are views of
          the q/k captures and the rotary tables that are alive anyway (roping
          outside and passing the result cost 9 GiB of retained activations at
          40K, enough to OOM a 47 GB card on its own); backward recomputes one
          chunk at a time (~3% of a model forward in total);
        * the key side is consumed HEAD-GROUP BY HEAD-GROUP against its own kv
          head, never expanded to ``n_q`` heads -- the expansion alone was 12 GiB
          of retained activations at 40K.

        The result is unchanged: the KL is still accumulated as
        ``sum over chunks / (rows x heads)`` per layer, meaned over layers.
        """
        from functools import partial

        from torch.utils.checkpoint import checkpoint

        if q_stride < 1 or self.row_stride % q_stride:
            raise ValueError(
                f"attn_map q_stride={q_stride} must be >= 1 and divide "
                f"row_stride={self.row_stride}"
            )
        step = self.row_stride // q_stride
        scale = self.hd ** -0.5
        cos_s, sin_s = self._stride_rotary(cos, sin, q_stride)
        n_rep = self.n_q // self.n_kv
        total, n = None, 0
        for layer, (q_s, k_s) in student_qk.items():
            if layer not in teacher_qk:
                continue
            q_t, k_t = teacher_qk[layer]
            if q_s.shape[1] != q_t.shape[1]:
                raise ValueError(
                    f"layer {layer}: student q has {q_s.shape[1]} captured rows, "
                    f"teacher has {q_t.shape[1]} — both sides must use the same "
                    f"query stride (train.distill_q_stride)."
                )
            # Views only — no rope, no copies.  The head slices below index these
            # directly; RoPE happens inside the checkpointed region.
            q_s4 = self.reshape_heads(q_s, self.n_q, self.hd)
            k_s4 = self.reshape_heads(k_s, self.n_kv, self.hd)
            tq4 = self.reshape_heads(q_t, self.n_q, self.hd)
            tk4 = self.reshape_heads(k_t, self.n_kv, self.hd)

            t_len = k_s4.shape[2]                      # original length, from the keys
            rows = torch.arange(0, q_s4.shape[2], step, device=q_s4.device)
            keys = torch.arange(t_len, device=q_s4.device)
            kl_sum = None
            for h in range(self.n_kv):
                sl = slice(h * n_rep, (h + 1) * n_rep)
                args = dict(cos_q=cos_s, sin_q=sin_s, cos_k=cos, sin_k=sin,
                            scale=scale, rows=rows, keys=keys, q_stride=q_stride,
                            chunk=self.chunk, eps=eps)
                # Views of the captures / targets / rotary tables: nothing new is
                # allocated, so the checkpoint's retained-input cost is zero.
                q_g, k_g = q_s4[:, sl], k_s4[:, h: h + 1]
                tq_g, tk_g = tq4[:, sl], tk4[:, h: h + 1]
                if torch.is_grad_enabled() and q_g.requires_grad:
                    piece = checkpoint(partial(self._group_kl_sum, tq_g=tq_g,
                                               tk_g=tk_g, **args),
                                       q_g, k_g, use_reentrant=False)
                else:                              # no_grad callers (the monitor)
                    piece = self._group_kl_sum(q_g, k_g, tq_g, tk_g, **args)
                kl_sum = piece if kl_sum is None else kl_sum + piece
            row_count = int(rows.numel()) * self.n_q * q_s4.shape[0]
            layer_kl = kl_sum / row_count
            total = layer_kl if total is None else total + layer_kl
            n += 1
        if total is None:
            raise ValueError("attn_map got no common layers between student and teacher")
        return total / n

    # ---------------------------------------------------------- output distill

    def attn_output(self, student: Mapping[int, Tensor],
                    teacher: Mapping[int, Tensor]) -> Tensor:
        """Per-layer attention-OUTPUT distillation: cosine + magnitude term.

        Cosine similarity is used rather than MSE because attention-output
        magnitude varies strongly with depth; a plain MSE summed over layers would
        be dominated by the deepest layers.  A small magnitude term restores the
        information cosine discards (direction can match while scale is wrong,
        which matters for KV pruning).

        Position subsampling has exactly ONE owner, and it is the TEACHER side:
        :meth:`Teacher.attention_outputs` already applies ``sample_stride`` before
        returning.  This method therefore subsamples ONLY the student, with the
        same stride, and hard-asserts the two length sets agree.  Applying the
        stride twice makes the teacher's effective stride ``stride**2``, which
        misaligns the targets (and at stride>=2 silently broadcasts rather than
        erroring); the explicit length check exists to turn that class of bug
        into a crash.
        """
        total, n = None, 0
        for layer in sorted(student):
            if layer not in teacher:
                continue
            t = teacher[layer].detach()          # already subsampled by the caller
            s = student[layer]
            if self.sample_stride > 1:
                s = s[:, :: self.sample_stride]
            if s.shape[1] != t.shape[1]:
                raise ValueError(
                    f"distillation position mismatch at layer {layer}: student has "
                    f"{s.shape[1]} positions, teacher has {t.shape[1]}. The stride is "
                    f"applied on both sides (teacher effective stride becomes "
                    f"stride**2) — the teacher must be subsampled exactly once, in "
                    f"Teacher.attention_outputs."
                )
            cos = (1.0 - F.cosine_similarity(s, t, dim=-1)).mean()
            sn, tn = s.norm(dim=-1), t.norm(dim=-1)
            mag = ((sn - tn) / (tn + 1e-6)).pow(2).mean()
            term = cos + self.mag_weight * mag
            total = term if total is None else total + term
            n += 1
        if total is None:
            raise ValueError("no overlapping layers between student and teacher captures")
        return total / n

    # ------------------------------------------------------------------- KD

    @staticmethod
    def logit_kd(student_logits: Tensor, teacher_logits: Tensor) -> Tensor:
        """Output-level distillation: ``KL(teacher || student)``, mean over TOKENS.

        NOT ``batchmean``: at batch size 1 that reduction is a per-position SUM,
        so the effective per-token weight of ``lam_kd`` was ``lam_kd / T`` —
        the same config meant something 32x weaker at T=8192 than at T=256, with
        nothing in the log to say so.  A per-token mean makes ``lam_kd``
        length-independent.
        """
        s = F.log_softmax(student_logits.float(), dim=-1)
        t = F.softmax(teacher_logits.float(), dim=-1)
        n_tokens = student_logits.shape[0] * student_logits.shape[1]
        return F.kl_div(s, t, reduction="sum") / n_tokens

    @classmethod
    def logit_kd_chunked(cls, student_hidden: Tensor, teacher_hidden: Tensor,
                         lm_head: nn.Module, chunk: int = 1024) -> Tensor:
        """Logit KD from HIDDEN STATES, applying the LM head chunk-wise.

        The full-logits path costs ``T x vocab`` twice (fp32 in
        :meth:`logit_kd`): ~5 GB per side at 8K, an instant OOM at 40K.  One LM
        head is used for both sides on purpose: the head is TIED to the
        embeddings, the embeddings are frozen and byte-identical to the teacher's
        (they are loaded from the same files), so the teacher's head would be the
        same function — and using the student's keeps the teacher off the
        logits path entirely.

        Chunks are meaned with their TOKEN COUNTS, so the result is the exact
        per-token mean of the unchunked computation (pinned by test to 1e-6).

        Each chunk runs inside a CHECKPOINTED region for the same reason as the
        map term: the chunk's ``[chunk, vocab]`` log-softmax graph would
        otherwise be retained until the joint backward, and at 40K that is
        tens of GB.  Retained per chunk: one scalar.
        """
        from functools import partial

        from torch.utils.checkpoint import checkpoint

        if student_hidden.shape[:2] != teacher_hidden.shape[:2]:
            raise ValueError(
                f"logit_kd_chunked: student hidden {tuple(student_hidden.shape[:2])} "
                f"vs teacher {tuple(teacher_hidden.shape[:2])}"
            )
        _, t, _ = student_hidden.shape
        total, n_tok = None, 0
        for s in range(0, t, chunk):
            s_h = student_hidden[:, s: s + chunk]
            t_h = teacher_hidden[:, s: s + chunk]
            n = s_h.shape[0] * s_h.shape[1]
            if torch.is_grad_enabled() and s_h.requires_grad:
                mean = checkpoint(partial(cls._kd_chunk_mean, lm_head=lm_head),
                                  s_h, t_h, use_reentrant=False)
            else:
                mean = cls._kd_chunk_mean(s_h, t_h, lm_head)
            term = mean * n
            total = term if total is None else total + term
            n_tok += n
        return total / max(n_tok, 1)

    @staticmethod
    def _kd_chunk_mean(s_h: Tensor, t_h: Tensor, lm_head: nn.Module) -> Tensor:
        """One chunk's per-token KL mean (the checkpointed unit of the KD term)."""
        return Distiller.logit_kd(lm_head(s_h), lm_head(t_h))
