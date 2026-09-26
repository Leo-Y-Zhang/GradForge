"""Numerically careful functions composed from Tensor primitives.

Nothing here has a hand-written backward pass: each function is built from
the primitive ops in tensor.py, so its gradient is derived by the engine.
The tests still gradcheck these compositions end to end.
"""
from __future__ import annotations

import math

import numpy as np

from .tensor import Tensor

__all__ = ["softmax", "log_softmax", "gelu", "cross_entropy"]


def softmax(x: Tensor, axis: int = -1) -> Tensor:
    """Softmax with the max subtracted as a detached constant.

    Softmax is shift-invariant, so subtracting the row max changes neither
    the value nor the gradient -- but it keeps exp() out of overflow. The
    shift is a plain constant (no grad), which is exactly right: d/dx of
    softmax(x - c) equals d/dx of softmax(x) for any constant c.
    """
    shift = Tensor(x.data.max(axis=axis, keepdims=True))
    e = (x - shift).exp()
    return e / e.sum(axis=axis, keepdims=True)


def log_softmax(x: Tensor, axis: int = -1) -> Tensor:
    """log(softmax(x)) via the shifted log-sum-exp, never forming softmax."""
    shift = Tensor(x.data.max(axis=axis, keepdims=True))
    z = x - shift
    return z - z.exp().sum(axis=axis, keepdims=True).log()


def gelu(x: Tensor) -> Tensor:
    """GELU, tanh approximation (Hendrycks & Gimpel 2016)."""
    c = math.sqrt(2.0 / math.pi)
    return x * 0.5 * (((x + (x ** 3.0) * 0.044715) * c).tanh() + 1.0)


def cross_entropy(logits: Tensor, targets) -> Tensor:
    """Mean negative log-likelihood of integer targets under logits.

    Accepts (N, V) logits with (N,) targets, or (B, T, V) with (B, T).
    """
    targets = np.asarray(targets)
    # Checked here because a mismatch does not fail by itself: the indexing
    # below still picks one entry per target and returns a plausible loss.
    if logits.ndim not in (2, 3) or targets.shape != logits.shape[:-1]:
        raise ValueError(f"logits of shape {logits.shape} need targets of "
                         f"shape {logits.shape[:-1]}, got {targets.shape}")
    v = logits.shape[-1]
    if targets.size and (targets.min() < 0 or targets.max() >= v):
        raise ValueError(f"targets must be class indices in [0, {v})")
    if logits.ndim == 3:
        b, t, v = logits.shape
        logits = logits.reshape(b * t, v)
        targets = targets.reshape(-1)
    n = targets.shape[0]
    ls = log_softmax(logits, axis=-1)
    picked = ls[np.arange(n), targets]
    return -picked.mean()
