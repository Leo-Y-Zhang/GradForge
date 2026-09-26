"""Gradient checks and numerical-stability proofs for the composed functions."""
import numpy as np
import pytest

from gradforge.functional import cross_entropy, gelu, log_softmax, softmax
from gradforge.gradcheck import gradcheck
from gradforge.tensor import Tensor
from helpers import const, randt

TOL = 1e-6


def test_softmax_gradcheck():
    rng = np.random.default_rng(30)
    x = randt(rng, (3, 5), scale=2.0)
    c = const(rng, (3, 5))
    assert gradcheck(lambda: (softmax(x) * c).sum(), [x], tol=TOL) < TOL


def test_log_softmax_gradcheck():
    rng = np.random.default_rng(31)
    x = randt(rng, (3, 5), scale=2.0)
    c = const(rng, (3, 5))
    gradcheck(lambda: (log_softmax(x) * c).sum(), [x], tol=TOL)


def test_gelu_gradcheck():
    rng = np.random.default_rng(32)
    x = randt(rng, (4, 4), scale=2.0)
    c = const(rng, (4, 4))
    gradcheck(lambda: (gelu(x) * c).sum(), [x], tol=TOL)


def test_cross_entropy_gradcheck():
    rng = np.random.default_rng(33)
    x = randt(rng, (6, 5), scale=2.0)
    targets = np.array([0, 3, 1, 3, 4, 0])  # duplicates on purpose
    assert gradcheck(lambda: cross_entropy(x, targets), [x], tol=TOL) < TOL


def test_cross_entropy_known_values():
    # two equal logits, one target: loss = ln 2
    x = Tensor(np.zeros((1, 2)))
    assert np.isclose(float(cross_entropy(x, [0]).data), np.log(2.0))
    # general case vs an independent numpy computation
    logits = np.array([[1.0, 2.0, 3.0]])
    expected = -(3.0 - np.log(np.exp(logits).sum()))
    got = float(cross_entropy(Tensor(logits), [2]).data)
    assert np.isclose(got, expected)


def test_cross_entropy_refuses_targets_that_do_not_fit_the_logits():
    # Indexing picks one log-probability per target, so a mismatched call
    # does not fail by itself: it returns a finite, plausible loss. Too few
    # targets average over the first rows only, a column of targets
    # broadcasts to every (row, target) pair, targets laid out (T, B) for
    # (B, T) logits pair each position with another's target, and a negative
    # target quietly reads a class from the end of the row.
    x = Tensor(np.zeros((4, 3)))
    for bad in ([0, 1], [[0], [1], [2], [0]], [0, 1, 2, -1]):
        with pytest.raises(ValueError):
            cross_entropy(x, bad)
    with pytest.raises(ValueError):
        cross_entropy(Tensor(np.zeros((2, 3, 5))), np.zeros((3, 2), dtype=int))


def test_softmax_rows_sum_to_one():
    rng = np.random.default_rng(34)
    x = Tensor(rng.normal(size=(4, 7)) * 3.0)
    s = softmax(x).data
    assert np.allclose(s.sum(axis=-1), 1.0)
    assert (s >= 0).all()


def test_stability_extreme_logits():
    # the naive formula overflows at 1e4; ours must not
    big = np.array([[1e4, 0.0, -1e4], [-1e4, 1e4, 0.0]])
    with np.errstate(over="ignore", invalid="ignore"):
        naive = np.exp(big) / np.exp(big).sum(axis=-1, keepdims=True)
    assert not np.isfinite(naive).all()  # documents why the shift exists

    x = Tensor(big, requires_grad=True)
    s = softmax(x)
    assert np.isfinite(s.data).all()
    assert np.allclose(s.data.sum(axis=-1), 1.0)

    ls = log_softmax(Tensor(big))
    assert np.isfinite(ls.data).all()

    # cross-entropy on the dominant class of a huge-margin row is ~0,
    # and its gradient is finite
    loss = cross_entropy(x, [0, 1])
    assert np.isfinite(float(loss.data))
    assert float(loss.data) < 1e-6
    loss.backward()
    assert np.isfinite(x.grad).all()
