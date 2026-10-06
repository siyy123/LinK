"""Optimizers and the LR schedule — one module for everything the loop optimises with.

Contains, in order: the Newton-Schulz kernel and the three custom optimizers
(:class:`Muon`, :class:`BlockMuon`, :class:`Lion`), :class:`CompositeOptimizer`
(several optimizers driven as one), :class:`OptimizerFactory` (config -> optimizer,
incl. the >=2-D / 1-D routing), and :class:`LRSchedule` (warmup + cosine/constant).

Why Muon is the default (measured, single layer, 200 steps): the penalty is a
GROUP penalty — it wants small RoPE pairs driven to zero while large ones
survive, i.e. *differentiated* shrinkage along the output dimension of ``k_proj``.
Adam's per-coordinate normalisation equalises the step size of every coordinate
regardless of which pair it belongs to, which FLATTENS the pair-norm distribution
instead of differentiating it.  Muon's semi-orthogonal update acts on the row
space as a whole and couples rows within a head, which preserves the pair
structure Adam flattens:

    optimizer        e_top10   cv     layer distill   full-model CE (ref 1.8629)
    AdamW  lr=5e-5   0.4046    1.285  0.4418          1.8981
    Muon   lr=0.02   0.6580    3.240  0.3002          1.8962
    Muon   lr=0.05   0.2556    0.975  0.5179          1.9100

(AdamW at lr=1e-3 is the cautionary case: it drove the penalty down 87% while
pushing e_top10 BELOW the untrained baseline — penalty value is NOT a proxy for
prunability.)  Lion is kept for comparison: its sign update moves every
coordinate exactly ``lr``, the most aggressive equalisation of the three, and at
lr=1e-3 it drove e_top10 to 0.2262, below the untrained 0.2933.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.optim import Optimizer

__all__ = ["Muon", "BlockMuon", "Lion", "CompositeOptimizer", "OptimizerFactory",
           "LRSchedule"]


# ============================================================== optimizers

class Muon(Optimizer):
    """Muon (Newton-Schulz orthogonalised momentum).  Intended for the 2-D
    projection matrices (q_proj, k_proj).

    ``lr`` is in Muon's own scale (~0.02 for these shapes), NOT Adam's — it is
    not comparable to an Adam learning rate.  Caveat: like Adam, the
    orthogonalisation removes the gradient magnitude, so lambda does not control
    the *rate* of sparsification — it only sets the direction; progress is
    governed by lr x steps.
    """

    def __init__(self, params, lr: float = 0.02, momentum: float = 0.95,
                 nesterov: bool = True, ns_steps: int = 5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @staticmethod
    def _orthogonalise(g: Tensor, steps: int = 5, eps: float = 1e-7) -> Tensor:
        """Approximate the nearest semi-orthogonal matrix via Newton-Schulz iteration.

        Coefficients from Keller Jordan's Muon; the quintic iteration converges
        quickly and is safe to run in bf16.
        """
        a, b, c = 3.4445, -4.7750, 2.0315
        x = g.bfloat16()
        transpose = x.size(0) > x.size(1)
        if transpose:
            x = x.T
        x = x / (x.norm() + eps)
        for _ in range(steps):
            gram = x @ x.T
            x = a * x + (b * gram + c * (gram @ gram)) @ x
        return x.T if transpose else x

    def _momentum_update(self, group, p: Tensor) -> Tensor:
        g = p.grad
        state = self.state[p]
        buf = state.get("momentum_buffer")
        if buf is None:
            buf = state["momentum_buffer"] = torch.zeros_like(g)
        buf.mul_(group["momentum"]).add_(g)
        return g.add(buf, alpha=group["momentum"]) if group["nesterov"] else buf

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.ndim != 2:
                    raise ValueError(f"Muon expects 2-D parameters, got shape {tuple(p.shape)}")
                update = self._momentum_update(group, p)
                ortho = self._orthogonalise(update, steps=group["ns_steps"])
                # scale so the update is comparable across non-square shapes
                scale = max(1.0, p.size(0) / p.size(1)) ** 0.5
                p.add_(ortho, alpha=-group["lr"] * scale)
        return loss


class BlockMuon(Muon):
    """Muon with the orthogonalisation applied PER HEAD BLOCK instead of on the
    whole matrix.

    Plain Muon orthogonalises the entire ``k_proj`` matrix, which couples all kv
    heads together — but our sparsity structure is strictly WITHIN a head (the
    RoPE pairs of one head's ``head_dim``).  Blocking the orthogonaliser keeps the
    coupling inside the unit we actually care about and stops heads from
    competing for the same update budget.

    ``block_size`` is the number of output rows per block (head_dim = 128 here).
    """

    def __init__(self, params, lr: float = 0.02, momentum: float = 0.95,
                 nesterov: bool = True, ns_steps: int = 5, block_size: int = 128):
        super().__init__(params, lr=lr, momentum=momentum, nesterov=nesterov,
                         ns_steps=ns_steps)
        for group in self.param_groups:
            group["block_size"] = block_size

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            block, ns_steps = group["block_size"], group["ns_steps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.ndim != 2:
                    raise ValueError(f"BlockMuon expects 2-D parameters, got {tuple(p.shape)}")
                update = self._momentum_update(group, p)
                out_rows, in_cols = p.shape
                if out_rows % block != 0:
                    # fall back to whole-matrix orthogonalisation
                    ortho = self._orthogonalise(update, steps=ns_steps)
                else:
                    nb = out_rows // block
                    u = update.view(nb, block, in_cols)
                    ortho = torch.stack(
                        [self._orthogonalise(u[i], steps=ns_steps) for i in range(nb)]
                    ).view(out_rows, in_cols)
                scale = max(1.0, out_rows / in_cols) ** 0.5
                p.add_(ortho, alpha=-group["lr"] * scale)
        return loss


class Lion(Optimizer):
    """Lion (Chen et al. 2023). Update is ``sign``-based, so every coordinate moves
    exactly ``lr`` — the most aggressive per-coordinate equalisation of the three.

        c_t   = beta1 * m_{t-1} + (1 - beta1) * g_t
        theta = theta - lr * (sign(c_t) + weight_decay * theta)
        m_t   = beta2 * m_{t-1} + (1 - beta2) * g_t

    Note the update uses the PREVIOUS momentum (beta1 interpolation) while the
    momentum is updated with beta2 — getting this wrong silently degrades it.
    """

    def __init__(self, params, lr: float = 1e-4, betas: tuple[float, float] = (0.9, 0.99),
                 weight_decay: float = 0.0):
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"invalid betas: {betas}")
        super().__init__(params, dict(lr=lr, betas=betas, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            wd = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                m = state.get("exp_avg")
                if m is None:
                    m = state["exp_avg"] = torch.zeros_like(p)

                update = m.mul(beta1).add_(g, alpha=1.0 - beta1).sign_()
                if wd:
                    p.add_(p, alpha=-lr * wd)
                p.add_(update, alpha=-lr)
                m.mul_(beta2).add_(g, alpha=1.0 - beta2)
        return loss


class CompositeOptimizer:
    """Several optimizers driven as one.

    Muon (and BlockMuon) orthogonalise 2-D matrices and RAISE on anything else, so
    the 1-D RMSNorm gains need their own AdamW.  This is the same split every Muon
    reference implementation makes, and here it is load-bearing rather than
    cosmetic: the gains are part of the trained unit but must not be
    orthogonalised.
    """

    def __init__(self, *opts):
        self.opts = [o for o in opts if o is not None]
        if not self.opts:
            raise ValueError("CompositeOptimizer needs at least one optimizer")

    @property
    def param_groups(self):
        groups = []
        for o in self.opts:
            groups.extend(o.param_groups)
        return groups

    def zero_grad(self, set_to_none: bool = True):
        for o in self.opts:
            o.zero_grad(set_to_none=set_to_none)

    def step(self, closure=None):
        loss = None
        for o in self.opts:
            out = o.step(closure)
            if out is not None:
                loss = out
        return loss


# ============================================================== factory

class OptimizerFactory:
    """Builds the configured optimizer from ``TrainCfg``, with parameter routing.

    Routing rule: parameters with ``ndim >= 2`` (the projection matrices) go to
    the configured optimizer; 1-D parameters (the QK-Norm gains) always go to
    AdamW.  The gains get NO weight decay — decaying them toward zero is a
    shrinkage the penalty provably cannot see (that is the whole point of
    penalising the pre-gain vector), so nothing would counteract it.

    LR semantics, stated once: ``train.lr`` is an ADAM-scale LR and drives BOTH
    the adamw matrix group AND the always-AdamW gain group.  ``muon_lr``,
    ``sgd_lr`` and ``lion_lr`` each set only the matrix group of their own
    optimizer (falling back to ``train.lr`` when None).  A sweep that raises
    ``lr`` to an SGD-scale value to make SGD effective would simultaneously
    detonate the gains — that is why the per-optimizer keys exist.

    Lion's betas stay at its class defaults (0.9, 0.99): ``train.betas`` is an
    Adam/AdamW setting and is deliberately NOT forwarded (Lion's published
    defaults differ, and silently reusing Adam's would be a retuning).
    """

    VALID = ("muon", "block_muon", "adamw", "sgd", "lion")

    def __init__(self, train_cfg):
        self.name = str(train_cfg.optimizer).lower()
        if self.name not in self.VALID:
            raise ValueError(f"unknown optimizer {self.name!r}; expected one of {self.VALID}")
        self.lr = train_cfg.lr
        self.betas = tuple(train_cfg.betas)
        self.weight_decay = train_cfg.weight_decay
        self.muon_lr = train_cfg.muon_lr
        self.muon_momentum = train_cfg.muon_momentum
        self.sgd_lr = train_cfg.sgd_lr if train_cfg.sgd_lr is not None else train_cfg.lr
        self.lion_lr = train_cfg.lion_lr if train_cfg.lion_lr is not None else train_cfg.lr

    def build(self, params, head_dim: int) -> Optimizer:
        params = list(params)
        matrix = [p for p in params if p.ndim >= 2]
        vector = [p for p in params if p.ndim < 2]
        opts = []
        if matrix:
            opts.append(self._single(self.name, matrix, head_dim))
        if vector:
            opts.append(torch.optim.AdamW(vector, lr=self.lr, betas=self.betas,
                                          weight_decay=0.0))
        if not opts:
            raise ValueError("OptimizerFactory.build got no parameters")
        return opts[0] if len(opts) == 1 else CompositeOptimizer(*opts)

    def _single(self, name: str, params, head_dim: int) -> Optimizer:
        """Construct one optimizer by name.  ``lr`` is Adam's scale; ``muon_lr``
        is Muon's own scale (NOT comparable)."""
        if name == "muon":
            return Muon(params, lr=self.muon_lr, momentum=self.muon_momentum)
        if name == "block_muon":
            return BlockMuon(params, lr=self.muon_lr, momentum=self.muon_momentum,
                             block_size=head_dim)
        if name == "adamw":
            return torch.optim.AdamW(params, lr=self.lr, betas=self.betas,
                                     weight_decay=self.weight_decay)
        if name == "sgd":
            return torch.optim.SGD(params, lr=self.sgd_lr, momentum=0.9,
                                   weight_decay=self.weight_decay)
        if name == "lion":
            return Lion(params, lr=self.lion_lr, weight_decay=self.weight_decay)
        raise ValueError(f"unknown optimizer {name!r}; expected one of {self.VALID}")


# ============================================================== schedule

class LRSchedule:
    """Warmup + cosine (or constant), taken to ``lr * min_lr_frac`` (not to 0, so
    the last steps still make measurable progress).

    There was previously NO LR schedule at all: ``TrainCfg.warmup_frac`` was
    declared and never read, and every run held a constant LR from step 0 to the
    last step.  A constant LR cannot be said to have "converged" -- the loss is
    still moving with the full step size at the final step.
    """

    def __init__(self, warmup_frac: float = 0.03, kind: str = "cosine",
                 min_lr_frac: float = 0.0):
        self.warmup_frac = float(warmup_frac)
        self.kind = str(kind)
        self.min_lr_frac = float(min_lr_frac)

    def lr_at(self, step: int, total_steps: int, base_lr: float) -> float:
        """Learning rate at ``step`` of a ``total_steps`` run."""
        if total_steps <= 0:
            return base_lr
        floor = base_lr * self.min_lr_frac
        if self.kind == "constant":
            return base_lr

        warm = max(1, int(total_steps * self.warmup_frac))
        if step < warm:
            # +1 so that step 0 is a small but non-zero rate rather than a dead step
            return base_lr * (step + 1) / warm

        progress = min(1.0, (step - warm) / max(1, total_steps - warm))
        cos = 0.5 * (1.0 + math.cos(math.pi * progress))
        return floor + (base_lr - floor) * cos

    def apply(self, optimizer, step: int, total_steps: int) -> float:
        """Set ``optimizer``'s LR for ``step`` and return it.

        Call BEFORE ``optimizer.step()`` each iteration.  Each param group keeps
        its own base LR under ``base_lr`` (optimizers such as Muon carry a
        different scale from Adam), captured on first call so the schedule is
        always relative to the configured value rather than to a previously
        decayed one.
        """
        for g in optimizer.param_groups:
            if "base_lr" not in g:
                g["base_lr"] = g["lr"]
            g["lr"] = self.lr_at(step, total_steps, g["base_lr"])
        return float(optimizer.param_groups[0]["lr"])
