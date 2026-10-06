"""Shared, stack-neutral machinery: primitives, operators, parametrisations.

``spkv.core`` must never import from ``spkv.train`` or ``spkv.eval`` -- both
stacks sit on top of it (enforced by ``tests/test_layout.py``).  The pruning
operators and the qk parametrisation stacks live here deliberately: the operator
trained against and the operator judged must be the same code.
"""
