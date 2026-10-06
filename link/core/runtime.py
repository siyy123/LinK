"""Runtime context shared by the training and inference/eval stacks.

One object, :class:`Runtime`, bound to (config, device): it builds the tokenizer
and the model, reports head dimensions, and hands out the autocast / SDPA
contexts.  Both stacks (and the teacher) construct one, so there is exactly one
place each of these decisions is made.  This module imports neither stack.
"""

from __future__ import annotations

import contextlib

import torch
from torch import nn

from ..config import SpKVConfig

__all__ = ["Runtime"]


class Runtime:
    """Device-bound plumbing for one run: tokenizer, model, contexts."""

    def __init__(self, cfg: SpKVConfig, device: torch.device | str):
        self.cfg = cfg
        self.device = torch.device(device)
        if self.device.type == "cuda" and self.device.index is not None:
            # "invalid device ordinal" from deep inside a .to() call does not
            # say which knob is wrong; this does.
            n = torch.cuda.device_count()
            if self.device.index >= n:
                raise ValueError(
                    f"device {device!r} is not visible to this process: "
                    f"{n} CUDA device(s) are (check CUDA_VISIBLE_DEVICES -- e.g. "
                    f"teacher_device cuda:2 needs the 72GB card included)."
                )

    # ------------------------------------------------------------ introspection

    @staticmethod
    def head_dims(model: nn.Module) -> tuple[int, int, int]:
        """(num_attention_heads, num_key_value_heads, head_dim) for the wrapped model."""
        cfg = model.config
        head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
        return cfg.num_attention_heads, cfg.num_key_value_heads, int(head_dim)

    # ------------------------------------------------------------ construction

    def tokenizer(self):
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(self.cfg.model.name_or_path)

    def build_model(self, dtype: torch.dtype | None = None,
                    rope_scaling: dict | None = None,
                    max_position_embeddings: int | None = None) -> nn.Module:
        """Load the base causal LM in ``dtype`` (default from config) on ``device``.

        One factory for all three callers (trainer, teacher, judge); they
        differed before only in the ``.to()/.eval()`` that follows the load.

        ``rope_scaling``/``max_position_embeddings`` are OVERRIDES on top of the
        checkpoint's own config, applied only when given (the 128K judge arm
        passes YaRN; training never does).  With both None the load is
        byte-identical to before -- the config object is not even constructed.
        """
        from transformers import AutoConfig, AutoModelForCausalLM

        if dtype is None:
            dtype = getattr(torch, self.cfg.model.dtype)
        config = None
        if rope_scaling is not None or max_position_embeddings is not None:
            config = AutoConfig.from_pretrained(self.cfg.model.name_or_path)
            if rope_scaling is not None:
                config.rope_scaling = dict(rope_scaling)
            if max_position_embeddings is not None:
                config.max_position_embeddings = int(max_position_embeddings)
        model = AutoModelForCausalLM.from_pretrained(
            self.cfg.model.name_or_path, dtype=dtype, config=config,
            attn_implementation=self.cfg.model.attn_implementation,
        )
        return model.to(self.device)

    # ------------------------------------------------------------ contexts

    @property
    def use_autocast(self) -> bool:
        """bf16 autocast is CUDA-only; it is also what the fp32 master weights
        compensate for, so the two toggles are the same condition."""
        return bool(self.cfg.train.fp32_master and self.device.type == "cuda")

    def autocast(self):
        return torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.use_autocast)

    @contextlib.contextmanager
    def sdpa(self, seq_len: int, threshold: int = 8192):
        """Restrict SDPA to memory-efficient kernels at long sequence lengths.

        Left alone, SDPA may select the MATH backend, which materialises the full
        ``[B, H, T, T]`` score matrix -- at T=128K that is 131072^2 * 24 heads,
        tens of terabytes, and the symptom is a bare OOM with nothing pointing at
        the kernel choice.  ``flash_attn`` is not installed in this environment,
        so the cuDNN and memory-efficient kernels are the only safe options.

        Below the threshold the default selection is kept: the fused kernels have
        different numerical behaviour, and there is no reason to change results
        for short runs that already fit.
        """
        if seq_len <= threshold:
            yield
            return
        from torch.nn.attention import SDPBackend, sdpa_kernel

        with sdpa_kernel([SDPBackend.CUDNN_ATTENTION, SDPBackend.FLASH_ATTENTION,
                          SDPBackend.EFFICIENT_ATTENTION]):
            yield
