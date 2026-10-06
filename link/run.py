"""Single entry point for every SpKV experiment.

    python -m spkv.run --config configs/<name>.yaml

What runs is decided by ``run.mode`` in the config file, and every knob lives in
that file too.  There is no second way to start a run: previously there were three
trainers and five scripts with overlapping code and -- the damaging part --
different penalty-normalisation conventions, so ``lambda = 0.1`` meant two
different things depending on which file was executed.

Command-line flags exist ONLY to override a config for a quick smoke test.  If an
experiment needs a different setting, put it in a file that can be diffed.

Modes
-----
``train``  the training loop                        (spkv/train/harness.py)
``eval``   the pruning judge                        (spkv/eval/judge.py)
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from .config import MODES, SpKVConfig

# Set BEFORE torch initialises CUDA (torch is imported lazily, further down):
# at 40K the training budget is tight enough that allocator fragmentation turns
# a fitting step into an OOM with the "reserved but unallocated" tell.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

__all__ = ["main", "load_config"]


def load_config(path: str | Path, **overrides) -> SpKVConfig:
    """Read a config and apply the CLI overrides that are allowed to exist."""
    cfg = SpKVConfig.from_yaml(path)
    for key, value in overrides.items():
        if value is None:
            continue
        section, _, field = key.partition(".")
        if not hasattr(cfg, section) or not hasattr(getattr(cfg, section), field):
            raise KeyError(f"unknown override {key}")
        setattr(getattr(cfg, section), field, value)
    return cfg.validate()


def _declared_type(section_obj: object, field: str) -> object | None:
    """The dataclass field's declared type, unwrapping ``Optional[...]``.

    Needed because the coercer below used to key off the field's CURRENT VALUE,
    and a field whose current value is ``None`` has no type to key off -- so
    ``--set penalty.prune_start_frac=0.75`` silently became the STRING "0.75" and
    blew up far away from the CLI, in a comparison.  Every Optional numeric knob
    had the same hole.
    """
    import dataclasses
    import typing

    if not dataclasses.is_dataclass(section_obj):
        return None
    try:
        hints = typing.get_type_hints(type(section_obj))
    except Exception:
        return None
    hint = hints.get(field)
    if hint is None:
        return None
    args = [a for a in typing.get_args(hint) if a is not type(None)]
    return args[0] if len(args) == 1 else hint


def apply_sets(cfg: SpKVConfig, assignments: list[str] | None) -> SpKVConfig:
    """Apply ``--set section.field=value``, coercing to the field's existing type.

    For SHORT VALIDATION SWEEPS only (sweeping one knob over a handful of
    one-config runs).  A real experiment belongs in a file that can be diffed,
    re-read and re-run -- which is the whole reason this CLI is config-driven.
    """
    for item in assignments or []:
        if "=" not in item:
            raise ValueError(f"--set expects section.field=value, got {item!r}")
        key, _, raw = item.partition("=")
        section, _, field = key.partition(".")
        if not hasattr(cfg, section) or not hasattr(getattr(cfg, section), field):
            raise KeyError(f"unknown override {key}")
        section_obj = getattr(cfg, section)
        current = getattr(section_obj, field)
        # The none/null literal is checked FIRST: an Optional[int] currently
        # holding 8 would otherwise be coerced with int("none") and raise.
        if raw.strip().lower() in ("none", "null", ""):
            value: object = None
        else:
            import typing

            # Fall back to the DECLARED type when the current value is None
            # (see _declared_type); otherwise the current value's type.
            declared = _declared_type(section_obj, field) if current is None else None
            kind = type(current) if current is not None else declared
            # Tuple fields (run.eval_alphas, train.betas, ...): comma-separated.
            # Without this branch the raw STRING was stored, e.g.
            # `--set run.eval_alphas=1.0,2.0` became '1.0,2.0' and blew up far
            # from the CLI.
            origin = typing.get_origin(kind)
            if isinstance(current, tuple) or origin is tuple:
                declared_t = _declared_type(section_obj, field)
                dargs = typing.get_args(declared_t) if typing.get_origin(declared_t) is tuple else ()
                el = dargs[0] if dargs else (type(current[0]) if current else None)
                if el not in (int, float, str):
                    raise ValueError(
                        f"--set {key}: cannot infer the tuple element type; put "
                        f"this value in a config file instead."
                    )
                value = tuple(el(p.strip()) for p in raw.split(",") if p.strip())
            elif isinstance(current, bool):
                value = raw.strip().lower() in ("1", "true", "yes", "on")
            elif kind is bool:
                value = raw.strip().lower() in ("1", "true", "yes", "on")
            elif isinstance(current, int) or kind is int:
                value = int(raw)
            elif isinstance(current, float) or kind is float:
                value = float(raw)
            else:
                value = raw
        setattr(section_obj, field, value)
    return cfg.validate()


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m spkv.run")
    ap.add_argument("--config", required=True, help="path to a YAML config file")
    ap.add_argument("--mode", choices=MODES, default=None,
                    help="override run.mode (smoke tests only)")
    ap.add_argument("--out-dir", default=None, help="override run.out_dir")
    ap.add_argument("--ckpt", default=None, help="override run.ckpt (eval mode)")
    ap.add_argument("--max-steps", type=int, default=None, help="override run.max_steps")
    ap.add_argument("--set", dest="sets", action="append", metavar="KEY=VALUE",
                    help="override any config field, e.g. --set penalty.lam_k_quantile=4. "
                         "For short validation sweeps only; real experiments go in a file.")
    args = ap.parse_args(argv)

    cfg = load_config(
        args.config,
        **{"run.mode": args.mode, "run.out_dir": args.out_dir,
           "run.ckpt": args.ckpt, "run.max_steps": args.max_steps},
    )
    cfg = apply_sets(cfg, args.sets)

    print(f"[run] mode={cfg.run.mode} config={args.config} out_dir={cfg.run.out_dir}")

    if cfg.run.mode == "train":
        from .train.harness import Trainer

        Trainer(cfg).train()
    elif cfg.run.mode == "eval":
        from .eval.judge import run_eval

        run_eval(cfg)
    elif cfg.run.mode == "infer":
        from .eval.infer import run_infer

        run_infer(cfg)
    else:  # pragma: no cover - validate() already rejects this
        raise ValueError(f"unhandled mode {cfg.run.mode!r}")


if __name__ == "__main__":
    main()
