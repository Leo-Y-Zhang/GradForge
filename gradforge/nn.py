"""Neural-net layers built only from the autograd engine's ops.

No layer computes anything numpy-side that needs a gradient: every
differentiable step goes through Tensor ops, so backward is derived.
Initialization draws from an explicit numpy Generator for determinism.
"""
from __future__ import annotations

import numpy as np

from .functional import cross_entropy, gelu, softmax
from .tensor import Tensor

__all__ = [
    "Module", "Linear", "LayerNorm", "Embedding",
    "CausalSelfAttention", "MLP", "Block", "GPT",
]


class Module:
    def parameters(self) -> list[Tensor]:
        params: list[Tensor] = []
        seen: set[int] = set()

        def collect(obj):
            if isinstance(obj, Tensor):
                if obj.requires_grad and id(obj) not in seen:
                    seen.add(id(obj))
                    params.append(obj)
            elif isinstance(obj, Module):
                for v in vars(obj).values():
                    collect(v)
            elif isinstance(obj, (list, tuple)):
                for v in obj:
                    collect(v)

        collect(self)
        return params

    def zero_grad(self) -> None:
        for p in self.parameters():
            p.grad = None

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Linear(Module):
    def __init__(self, n_in: int, n_out: int, rng: np.random.Generator,
                 bias: bool = True):
        self.weight = Tensor(rng.normal(0.0, n_in ** -0.5, (n_in, n_out)),
                             requires_grad=True)
        self.bias = Tensor(np.zeros(n_out), requires_grad=True) if bias else None

    def forward(self, x: Tensor) -> Tensor:
        out = x @ self.weight
        if self.bias is not None:
            out = out + self.bias
        return out


class LayerNorm(Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        self.gamma = Tensor(np.ones(dim), requires_grad=True)
        self.beta = Tensor(np.zeros(dim), requires_grad=True)
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        mu = x.mean(axis=-1, keepdims=True)
        xc = x - mu
        var = (xc * xc).mean(axis=-1, keepdims=True)
        xhat = xc * ((var + self.eps) ** -0.5)
        return xhat * self.gamma + self.beta


class Embedding(Module):
    def __init__(self, num: int, dim: int, rng: np.random.Generator,
                 scale: float = 0.02):
        self.weight = Tensor(rng.normal(0.0, scale, (num, dim)),
                             requires_grad=True)

    def forward(self, idx) -> Tensor:
        # integer-array indexing; backward is a scatter-add so repeated
        # tokens accumulate gradient correctly
        return self.weight[np.asarray(idx)]


class CausalSelfAttention(Module):
    def __init__(self, d_model: int, n_head: int, rng: np.random.Generator):
        if d_model % n_head:
            raise ValueError("d_model must be divisible by n_head")
        self.n_head = n_head
        self.qkv = Linear(d_model, 3 * d_model, rng)
        self.proj = Linear(d_model, d_model, rng)

    def forward(self, x: Tensor) -> Tensor:
        b, t, c = x.shape
        h = self.n_head
        hd = c // h
        qkv = self.qkv(x)
        q = qkv[:, :, 0:c].reshape(b, t, h, hd).transpose(0, 2, 1, 3)
        k = qkv[:, :, c:2 * c].reshape(b, t, h, hd).transpose(0, 2, 1, 3)
        v = qkv[:, :, 2 * c:3 * c].reshape(b, t, h, hd).transpose(0, 2, 1, 3)
        att = (q @ k.transpose(0, 1, 3, 2)) * (hd ** -0.5)
        future = np.triu(np.ones((t, t), dtype=bool), k=1)
        att = att.masked_fill(future, -1e9)
        att = softmax(att, axis=-1)
        y = (att @ v).transpose(0, 2, 1, 3).reshape(b, t, c)
        return self.proj(y)


class MLP(Module):
    def __init__(self, d_model: int, rng: np.random.Generator):
        self.fc = Linear(d_model, 4 * d_model, rng)
        self.proj = Linear(4 * d_model, d_model, rng)

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(gelu(self.fc(x)))


class Block(Module):
    def __init__(self, d_model: int, n_head: int, rng: np.random.Generator):
        self.ln1 = LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_head, rng)
        self.ln2 = LayerNorm(d_model)
        self.mlp = MLP(d_model, rng)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(Module):
    """Character-level decoder-only transformer, pre-norm."""

    def __init__(self, vocab_size: int, block_size: int, d_model: int = 64,
                 n_layer: int = 2, n_head: int = 4, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.config = dict(vocab_size=vocab_size, block_size=block_size,
                           d_model=d_model, n_layer=n_layer, n_head=n_head)
        self.block_size = block_size
        self.tok = Embedding(vocab_size, d_model, rng)
        self.pos = Tensor(rng.normal(0.0, 0.02, (block_size, d_model)),
                          requires_grad=True)
        self.blocks = [Block(d_model, n_head, rng) for _ in range(n_layer)]
        self.ln_f = LayerNorm(d_model)
        self.head = Linear(d_model, vocab_size, rng, bias=False)

    def forward(self, idx) -> Tensor:
        idx = np.asarray(idx)
        _, t = idx.shape
        if t > self.block_size:
            raise ValueError("sequence longer than block_size")
        x = self.tok(idx) + self.pos[0:t]
        for block in self.blocks:
            x = block(x)
        return self.head(self.ln_f(x))

    def loss(self, idx, targets) -> Tensor:
        return cross_entropy(self.forward(idx), targets)
