"""Reverse-mode autograd over numpy arrays.

A Tensor wraps a float64 ndarray and records, for every operation, a closure
that knows how to push gradient back to the operation's inputs. backward()
topologically sorts the recorded graph and runs the closures in reverse.

numpy is used only for array arithmetic. Every derivative rule here is
hand-written and verified by central-difference tests in tests/.
"""
from __future__ import annotations

import numpy as np

__all__ = ["Tensor"]


def _unbroadcast(grad: np.ndarray, shape: tuple) -> np.ndarray:
    """Sum `grad` down to `shape`, undoing numpy broadcasting.

    If a forward op broadcast an input from `shape` up to grad.shape, the
    input's gradient is the sum of grad over every broadcast axis.
    """
    if grad.shape == shape:
        return grad
    extra = grad.ndim - len(shape)
    if extra > 0:
        grad = grad.sum(axis=tuple(range(extra)))
    axes = tuple(i for i, s in enumerate(shape) if s == 1 and grad.shape[i] != 1)
    if axes:
        grad = grad.sum(axis=axes, keepdims=True)
    return grad.reshape(shape)


def _is_basic_index(idx) -> bool:
    """True if idx uses only ints/slices/Ellipsis/None (no duplicate risk)."""
    items = idx if isinstance(idx, tuple) else (idx,)
    return all(
        isinstance(i, (int, np.integer, slice)) or i is Ellipsis or i is None
        for i in items
    )


class Tensor:
    __slots__ = ("data", "grad", "requires_grad", "_backward", "_prev")

    def __init__(self, data, requires_grad: bool = False):
        if isinstance(data, Tensor):
            data = data.data
        self.data = np.asarray(data, dtype=np.float64)
        self.grad: np.ndarray | None = None
        self.requires_grad = requires_grad
        self._backward = None
        self._prev: tuple = ()

    # ------------------------------------------------------------------ infra

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    def __repr__(self):
        return f"Tensor(shape={self.data.shape}, requires_grad={self.requires_grad})"

    @staticmethod
    def _wrap(other) -> "Tensor":
        return other if isinstance(other, Tensor) else Tensor(other)

    def _accum(self, g: np.ndarray) -> None:
        self.grad = np.array(g) if self.grad is None else self.grad + g

    def _make(self, data, prev, backward) -> "Tensor":
        out = Tensor(data)
        if any(p.requires_grad for p in prev):
            out.requires_grad = True
            out._prev = tuple(prev)
            out._backward = backward
        return out

    def detach(self) -> "Tensor":
        return Tensor(self.data)

    def backward(self, grad=None) -> None:
        if grad is None:
            if self.data.size != 1:
                raise ValueError("backward() without an explicit grad needs a scalar")
            grad = np.ones_like(self.data)
        # iterative topological sort (no recursion-depth surprises)
        topo, visited, stack = [], set(), [(self, False)]
        while stack:
            node, done = stack.pop()
            if done:
                topo.append(node)
                continue
            if id(node) in visited:
                continue
            visited.add(id(node))
            stack.append((node, True))
            for child in node._prev:
                if id(child) not in visited:
                    stack.append((child, False))
        self.grad = np.asarray(grad, dtype=np.float64).reshape(self.data.shape)
        for node in reversed(topo):
            if node._backward is not None and node.grad is not None:
                node._backward()

    # ------------------------------------------------------------- arithmetic

    def __add__(self, other):
        other = self._wrap(other)
        out = self._make(self.data + other.data, (self, other), None)

        def backward():
            if self.requires_grad:
                self._accum(_unbroadcast(out.grad, self.data.shape))
            if other.requires_grad:
                other._accum(_unbroadcast(out.grad, other.data.shape))

        out._backward = backward if out.requires_grad else None
        return out

    def __mul__(self, other):
        other = self._wrap(other)
        out = self._make(self.data * other.data, (self, other), None)

        def backward():
            if self.requires_grad:
                self._accum(_unbroadcast(out.grad * other.data, self.data.shape))
            if other.requires_grad:
                other._accum(_unbroadcast(out.grad * self.data, other.data.shape))

        out._backward = backward if out.requires_grad else None
        return out

    def __neg__(self):
        out = self._make(-self.data, (self,), None)

        def backward():
            self._accum(-out.grad)

        out._backward = backward if out.requires_grad else None
        return out

    def __sub__(self, other):
        return self + (-self._wrap(other))

    def __rsub__(self, other):
        return self._wrap(other) + (-self)

    def __truediv__(self, other):
        if isinstance(other, Tensor):
            return self * (other ** -1.0)
        return self * (1.0 / other)

    def __rtruediv__(self, other):
        return (self ** -1.0) * other

    def __pow__(self, p):
        if not isinstance(p, (int, float)):
            raise TypeError("Tensor ** exponent must be a Python scalar")
        out = self._make(self.data ** p, (self,), None)

        def backward():
            self._accum(out.grad * p * self.data ** (p - 1))

        out._backward = backward if out.requires_grad else None
        return out

    __radd__ = __add__
    __rmul__ = __mul__

    def __matmul__(self, other):
        other = self._wrap(other)
        if self.data.ndim < 2 or other.data.ndim < 2:
            raise ValueError("matmul needs tensors with ndim >= 2")
        out = self._make(self.data @ other.data, (self, other), None)

        def backward():
            if self.requires_grad:
                ga = out.grad @ other.data.swapaxes(-1, -2)
                self._accum(_unbroadcast(ga, self.data.shape))
            if other.requires_grad:
                gb = self.data.swapaxes(-1, -2) @ out.grad
                other._accum(_unbroadcast(gb, other.data.shape))

        out._backward = backward if out.requires_grad else None
        return out

    # ------------------------------------------------------------ elementwise

    def exp(self):
        out = self._make(np.exp(self.data), (self,), None)

        def backward():
            self._accum(out.grad * out.data)

        out._backward = backward if out.requires_grad else None
        return out

    def log(self):
        out = self._make(np.log(self.data), (self,), None)

        def backward():
            self._accum(out.grad / self.data)

        out._backward = backward if out.requires_grad else None
        return out

    def tanh(self):
        out = self._make(np.tanh(self.data), (self,), None)

        def backward():
            self._accum(out.grad * (1.0 - out.data * out.data))

        out._backward = backward if out.requires_grad else None
        return out

    def relu(self):
        out = self._make(np.maximum(self.data, 0.0), (self,), None)

        def backward():
            self._accum(out.grad * (self.data > 0.0))

        out._backward = backward if out.requires_grad else None
        return out

    # -------------------------------------------------------------- reduction

    def sum(self, axis=None, keepdims=False):
        out = self._make(self.data.sum(axis=axis, keepdims=keepdims), (self,), None)

        def backward():
            g = out.grad
            if axis is not None and not keepdims:
                g = np.expand_dims(g, axis)
            self._accum(np.broadcast_to(g, self.data.shape))

        out._backward = backward if out.requires_grad else None
        return out

    def mean(self, axis=None, keepdims=False):
        s = self.sum(axis=axis, keepdims=keepdims)
        n = self.data.size // max(s.data.size, 1)
        return s * (1.0 / n)

    # ------------------------------------------------------------------ shape

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        out = self._make(self.data.reshape(shape), (self,), None)

        def backward():
            self._accum(out.grad.reshape(self.data.shape))

        out._backward = backward if out.requires_grad else None
        return out

    def transpose(self, *axes):
        if len(axes) == 1 and isinstance(axes[0], (tuple, list)):
            axes = tuple(axes[0])
        perm = axes if axes else tuple(range(self.data.ndim))[::-1]
        out = self._make(self.data.transpose(perm), (self,), None)
        inv = tuple(np.argsort(perm))

        def backward():
            self._accum(out.grad.transpose(inv))

        out._backward = backward if out.requires_grad else None
        return out

    # --------------------------------------------------------------- indexing

    def __getitem__(self, idx):
        out = self._make(self.data[idx], (self,), None)

        def backward():
            g = np.zeros_like(self.data)
            if _is_basic_index(idx):
                g[idx] += out.grad
            else:
                # scatter-add: duplicate indices must ACCUMULATE gradient,
                # plain assignment would silently keep only the last one
                np.add.at(g, idx, out.grad)
            self._accum(g)

        out._backward = backward if out.requires_grad else None
        return out

    def masked_fill(self, mask, value: float):
        mask = np.asarray(mask, dtype=bool)
        out = self._make(np.where(mask, value, self.data), (self,), None)

        def backward():
            self._accum(np.where(mask, 0.0, out.grad))

        out._backward = backward if out.requires_grad else None
        return out
