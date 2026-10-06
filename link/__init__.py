"""SpKV — training-time LASSO penalties that make the KV cache natively sparse.

Phase 1 scope: RoPE-pair group LASSO on the model's normalised Q and K (V
deferred).  The penalty, the training loop and the pruning judge all read the
model's own QK-Norm output — see ``spkv/core/qknorm.py`` for why that is the only
correct observation point.
"""
