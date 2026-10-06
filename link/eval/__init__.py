"""Inference/evaluation stack: the pruning judge.

Imports only :mod:`spkv.config` and :mod:`spkv.core` -- never :mod:`spkv.train`
(enforced by ``tests/test_layout.py``).  A checkpoint is the only thing it
receives from training.
"""
