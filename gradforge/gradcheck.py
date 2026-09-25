"""Central-difference gradient checking.

This is the library's core honesty mechanism: every analytic gradient the
engine produces is compared against (f(x+h) - f(x-h)) / 2h. In float64 with
h = 1e-5, a correct derivative agrees to ~1e-9 relative error; a wrong one
misses by orders of magnitude. The test suite runs this over every op and
every layer.
"""
from __future__ import annotations

import numpy as np

from .tensor import Tensor

__all__ = ["gradcheck"]


def gradcheck(fn, wrt, h: float = 1e-5, tol: float = 1e-6,
              rng: np.random.Generator | None = None,
              max_elems: int | None = None) -> float:
    """Verify autograd gradients of a scalar-valued computation.

    fn: zero-argument callable returning a scalar Tensor; it must rebuild
        the computation from the tensors in `wrt` each call.
    wrt: tensors (requires_grad=True) whose gradients are checked.
    max_elems: if set, check a seeded random sample of this many elements
        per tensor instead of all of them (for large parameters).

    Returns the max relative error seen; raises AssertionError on breach.
    """
    wrt = list(wrt)
    for p in wrt:
        p.grad = None
    loss = fn()
    if loss.data.size != 1:
        raise ValueError("gradcheck needs a scalar loss")
    loss.backward()
    analytic = []
    for p in wrt:
        if p.grad is None:
            raise AssertionError("no gradient reached a checked tensor")
        # Elements are compared by flat index, so a gradient of the wrong shape
        # would be read as if it were the right one -- and passes whenever its
        # leading entries happen to hold the right numbers.
        if p.grad.shape != p.data.shape:
            raise AssertionError(
                f"gradient has shape {p.grad.shape}, tensor has {p.data.shape}")
        analytic.append(p.grad.copy())

    max_rel = 0.0
    for p, ana in zip(wrt, analytic):
        # Perturb through a multi-index, not a flattened view: reshape(-1) of
        # a non-contiguous array (a transposed or strided one) is a COPY, and
        # writing to it would leave the real tensor untouched -- a checker
        # that reports a numeric gradient of exactly zero for everything.
        ana_flat = ana.reshape(-1)
        n = p.data.size
        if max_elems is not None and n > max_elems:
            if rng is None:
                raise ValueError("sampling requires an rng")
            idxs = rng.choice(n, size=max_elems, replace=False)
        else:
            idxs = np.arange(n)
        for i in idxs:
            ix = np.unravel_index(i, p.data.shape)
            orig = p.data[ix]
            p.data[ix] = orig + h
            fp = float(fn().data)
            p.data[ix] = orig - h
            fm = float(fn().data)
            p.data[ix] = orig
            num = (fp - fm) / (2.0 * h)
            a = float(ana_flat[i])
            rel = abs(num - a) / max(1.0, abs(num), abs(a))
            max_rel = max(max_rel, rel)
            if rel >= tol:
                raise AssertionError(
                    f"gradcheck failed at element {i}: "
                    f"analytic {a:.10g} vs numeric {num:.10g} (rel {rel:.3g})"
                )
    return max_rel
