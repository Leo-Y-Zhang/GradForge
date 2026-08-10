"""Shared test helpers."""
import numpy as np

from gradforge.tensor import Tensor


def randt(rng, shape, scale=1.0, positive=False, requires_grad=True):
    data = rng.normal(size=shape) * scale
    if positive:
        data = np.abs(data) + 0.5
    return Tensor(data, requires_grad=requires_grad)


def const(rng, shape):
    """A fixed random weighting tensor: makes .sum() losses sensitive to
    which element went where, so permuted/misrouted gradients can't pass."""
    return Tensor(rng.normal(size=shape))
