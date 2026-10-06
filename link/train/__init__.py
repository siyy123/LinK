"""Training stack: the Trainer, its objectives, optimisers and teacher.

Shares primitives with the inference/eval stack through :mod:`spkv.core` only --
``spkv.train`` must never import ``spkv.eval`` (enforced by
``tests/test_layout.py``).
"""
