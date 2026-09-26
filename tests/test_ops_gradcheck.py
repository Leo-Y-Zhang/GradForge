"""Central-difference gradient checks for every primitive op.

Each loss multiplies by a fixed random constant before summing, so a
gradient that reaches the right tensor but the wrong elements cannot pass.
"""
import numpy as np

from gradforge.gradcheck import gradcheck
from gradforge.tensor import Tensor
from helpers import const, randt

TOL = 1e-6


def test_add_broadcast():
    rng = np.random.default_rng(10)
    a, b = randt(rng, (3, 4)), randt(rng, (4,))
    c = const(rng, (3, 4))
    assert gradcheck(lambda: ((a + b) * c).sum(), [a, b], tol=TOL) < TOL


def test_mul_broadcast_both_dims():
    rng = np.random.default_rng(11)
    a, b = randt(rng, (2, 3, 4)), randt(rng, (3, 1))
    c = const(rng, (2, 3, 4))
    assert gradcheck(lambda: ((a * b) * c).sum(), [a, b], tol=TOL) < TOL


def test_sub_and_neg():
    rng = np.random.default_rng(12)
    a, b = randt(rng, (3, 3)), randt(rng, (3,))
    c = const(rng, (3, 3))
    gradcheck(lambda: ((a - b) * c).sum(), [a, b], tol=TOL)
    gradcheck(lambda: ((-a) * c).sum(), [a], tol=TOL)


def test_div():
    rng = np.random.default_rng(13)
    a = randt(rng, (3, 4))
    b = randt(rng, (4,), positive=True)  # bounded away from zero
    c = const(rng, (3, 4))
    c2 = const(rng, (4,))
    gradcheck(lambda: ((a / b) * c).sum(), [a, b], tol=TOL)
    gradcheck(lambda: ((2.0 / b) * c2).sum(), [b], tol=TOL)


def test_pow():
    rng = np.random.default_rng(14)
    a = randt(rng, (3, 3), positive=True)
    c = const(rng, (3, 3))
    gradcheck(lambda: ((a ** 1.7) * c).sum(), [a], tol=TOL)
    gradcheck(lambda: ((a ** -0.5) * c).sum(), [a], tol=TOL)


def test_matmul_plain():
    rng = np.random.default_rng(15)
    a, b = randt(rng, (3, 4)), randt(rng, (4, 5))
    c = const(rng, (3, 5))
    assert gradcheck(lambda: ((a @ b) * c).sum(), [a, b], tol=TOL) < TOL


def test_matmul_batched():
    rng = np.random.default_rng(16)
    a, b = randt(rng, (2, 3, 4)), randt(rng, (2, 4, 5))
    c = const(rng, (2, 3, 5))
    gradcheck(lambda: ((a @ b) * c).sum(), [a, b], tol=TOL)


def test_matmul_broadcast_batch():
    # (2,3,4) @ (4,5): the right operand's grad must sum over the batch
    rng = np.random.default_rng(17)
    a, b = randt(rng, (2, 3, 4)), randt(rng, (4, 5))
    c = const(rng, (2, 3, 5))
    gradcheck(lambda: ((a @ b) * c).sum(), [a, b], tol=TOL)


def test_exp_log_tanh_relu():
    rng = np.random.default_rng(18)
    a = randt(rng, (4, 4))
    p = randt(rng, (4, 4), positive=True)
    c = const(rng, (4, 4))
    gradcheck(lambda: (a.exp() * c).sum(), [a], tol=TOL)
    gradcheck(lambda: (p.log() * c).sum(), [p], tol=TOL)
    gradcheck(lambda: (a.tanh() * c).sum(), [a], tol=TOL)
    gradcheck(lambda: (a.relu() * c).sum(), [a], tol=TOL)


def test_sum_axes():
    rng = np.random.default_rng(19)
    a = randt(rng, (2, 3, 4))
    gradcheck(lambda: (a.sum(axis=1) * const(np.random.default_rng(1), (2, 4))).sum(),
              [a], tol=TOL)
    gradcheck(lambda: (a.sum(axis=(0, 2)) * const(np.random.default_rng(2), (3,))).sum(),
              [a], tol=TOL)
    gradcheck(lambda: (a.sum(axis=1, keepdims=True)
                       * const(np.random.default_rng(3), (2, 1, 4))).sum(),
              [a], tol=TOL)


def test_mean():
    rng = np.random.default_rng(20)
    a = randt(rng, (3, 5))
    gradcheck(lambda: (a.mean(axis=-1, keepdims=True)
                       * const(np.random.default_rng(4), (3, 1))).sum(),
              [a], tol=TOL)
    gradcheck(lambda: a.mean(), [a], tol=TOL)


def test_reshape_transpose():
    rng = np.random.default_rng(21)
    a = randt(rng, (2, 3, 4))
    c = const(rng, (4, 6))
    gradcheck(lambda: (a.transpose(2, 0, 1).reshape(4, 6) * c).sum(), [a], tol=TOL)


def test_getitem_slices():
    rng = np.random.default_rng(22)
    a = randt(rng, (5, 6))
    c = const(rng, (2, 3))
    gradcheck(lambda: (a[1:3, ::2] * c).sum(), [a], tol=TOL)


def test_getitem_duplicate_fancy_indices():
    # the classic scatter-add trap: row 0 appears twice, so its gradient is
    # the SUM of both contributions; assignment instead of add.at fails here
    rng = np.random.default_rng(23)
    a = randt(rng, (4, 3))
    idx = np.array([0, 2, 0, 1])
    c = const(rng, (4, 3))
    assert gradcheck(lambda: (a[idx] * c).sum(), [a], tol=TOL) < TOL


def test_masked_fill():
    rng = np.random.default_rng(24)
    a = randt(rng, (4, 4))
    mask = np.random.default_rng(5).random((4, 4)) < 0.4
    c = const(rng, (4, 4))
    gradcheck(lambda: (a.masked_fill(mask, -3.0) * c).sum(), [a], tol=TOL)


def test_masked_fill_with_a_mask_wider_than_the_tensor():
    # np.where broadcasts the tensor up to the mask's shape, so the output is
    # bigger than the input and the gradient has to be summed back down, like
    # every other broadcasting op. Returned unsummed, a (3,) leaf ends up
    # holding a (2, 3) gradient, which an optimizer then fails to apply.
    rng = np.random.default_rng(25)
    a = randt(rng, (3,))
    mask = np.array([[True, False, False], [False, False, True]])
    c = const(rng, (2, 3))
    gradcheck(lambda: (a.masked_fill(mask, 5.0) * c).sum(), [a], tol=TOL)
    assert a.grad.shape == a.shape


def test_transpose_with_negative_axes():
    # numpy's transpose accepts negative axes, and so does the forward pass.
    # The backward pass has to invert the permutation over the normalised
    # axes: argsort of the raw ones orders -1 before 0 and hands back a
    # wrongly permuted gradient. On a cube it even has the right shape, so
    # only the values give it away.
    rng = np.random.default_rng(26)
    a = randt(rng, (3, 3, 3))
    c = const(rng, (3, 3, 3))
    gradcheck(lambda: (a.transpose(0, -1, -2) * c).sum(), [a], tol=TOL)
    b = randt(rng, (2, 3, 4, 5))
    c2 = const(rng, (5, 2, 4, 3))
    gradcheck(lambda: (b.transpose(-1, 0, 2, -3) * c2).sum(), [b], tol=TOL)
