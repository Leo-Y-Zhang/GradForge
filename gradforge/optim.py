"""Optimizers. State updates are in-place numpy; nothing here needs autograd."""
from __future__ import annotations

import numpy as np

from .tensor import Tensor

__all__ = ["SGD", "Adam"]


class SGD:
    """SGD with classical momentum: v = mu*v + g; p -= lr*v."""

    def __init__(self, params, lr: float = 0.1, momentum: float = 0.0):
        self.params: list[Tensor] = list(params)
        self.lr = lr
        self.momentum = momentum
        self.v = [np.zeros_like(p.data) for p in self.params]

    def step(self) -> None:
        for p, v in zip(self.params, self.v):
            if p.grad is None:
                continue
            v *= self.momentum
            v += p.grad
            p.data -= self.lr * v

    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = None


class Adam:
    """Adam (Kingma & Ba 2015) with bias correction."""

    def __init__(self, params, lr: float = 1e-3, betas=(0.9, 0.999),
                 eps: float = 1e-8):
        self.params: list[Tensor] = list(params)
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.t = 0
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]

    def step(self) -> None:
        self.t += 1
        bc1 = 1.0 - self.b1 ** self.t
        bc2 = 1.0 - self.b2 ** self.t
        for p, m, v in zip(self.params, self.m, self.v):
            if p.grad is None:
                continue
            g = p.grad
            m *= self.b1
            m += (1.0 - self.b1) * g
            v *= self.b2
            v += (1.0 - self.b2) * (g * g)
            p.data -= self.lr * (m / bc1) / (np.sqrt(v / bc2) + self.eps)

    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = None
