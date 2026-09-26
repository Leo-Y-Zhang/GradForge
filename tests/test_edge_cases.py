"""Gradient checks on the shapes and values where autograd engines go wrong.

Broadcast operands reduced over negative and tuple axes, one node feeding
several consumers, tensors that are views with odd strides, tensors with a
zero-length axis, extreme values for the numerically delicate functions, and
inputs of other dtypes. Each loss is weighted by a fixed random constant, as
in test_ops_gradcheck, so a gradient routed to the wrong element cannot pass.
"""
import numpy as np
import pytest

from gradforge.functional import cross_entropy, log_softmax, softmax
from gradforge.gradcheck import gradcheck
from gradforge.tensor import Tensor
from helpers import const, randt

TOL = 1e-6


# ------------------------------------------------- broadcasting reductions

def test_reductions_of_broadcast_operands():
    # The broadcast operand's gradient is summed by _unbroadcast and the
    # reduction's is expanded again by sum's backward: both have to agree on
    # which axes are which when the axes are negative or given as a tuple.
    rng = np.random.default_rng(80)
    a, b = randt(rng, (2, 3, 4)), randt(rng, (3, 1))
    cases = [
        (lambda: (a * b).sum(axis=-1), (2, 3)),
        (lambda: (a + b).sum(axis=(0, -1)), (3,)),
        (lambda: (a * b).sum(axis=-2, keepdims=True), (2, 1, 4)),
        (lambda: (a - b).mean(axis=(-3, -1), keepdims=True), (1, 3, 1)),
        (lambda: (a / (b * b + 1.0)).mean(axis=1), (2, 4)),
    ]
    for fn, shape in cases:
        c = const(rng, shape)
        assert fn().shape == shape
        gradcheck(lambda: (fn() * c).sum(), [a, b], tol=TOL)
        assert b.grad.shape == (3, 1)


# ------------------------------------------------------------ reused nodes

def test_gradient_accumulates_through_a_reused_intermediate():
    # h feeds four consumers, one of them twice over (h * h); x reaches the
    # loss by two routes. Every path has to add, none may overwrite.
    rng = np.random.default_rng(81)
    x, w = randt(rng, (3, 4)), randt(rng, (4, 4))
    c = const(rng, (3, 4))

    def fn():
        h = (x @ w).tanh()
        return ((h * h + h.exp() - h * x) * c).sum() + h.sum()

    gradcheck(fn, [x, w], tol=TOL)


def test_same_tensor_as_both_matmul_operands():
    rng = np.random.default_rng(82)
    x = randt(rng, (3, 3))
    c = const(rng, (3, 3))
    gradcheck(lambda: ((x @ x) * c).sum(), [x], tol=TOL)
    gradcheck(lambda: ((x @ x.transpose()) * c).sum(), [x], tol=TOL)


# ----------------------------------------------------- non-contiguous input

@pytest.mark.parametrize("view", [
    lambda base: base.T,                   # transposed
    lambda base: base[::2, ::-1],          # strided and reversed
    lambda base: base[:, 1:5].T[::-1],     # a slice of a reversed transpose
])
def test_gradients_of_tensors_built_on_non_contiguous_views(view):
    rng = np.random.default_rng(83)
    base = rng.normal(size=(6, 6))
    x = Tensor(view(base), requires_grad=True)
    assert not x.data.flags.c_contiguous
    m, n = x.shape
    w = randt(rng, (n, 2))
    c = const(rng, (m, 2))
    gradcheck(lambda: ((x.reshape(n, m).transpose() @ w) * c).sum()
              + (x[::-1, 0] * x[:, -1]).sum() + x.exp().mean(), [x, w], tol=TOL)
    assert x.grad.shape == x.shape


# ----------------------------------------------------------- zero-size axes

def test_zero_size_shapes_flow_through_forward_and_backward():
    # Every op must accept an empty axis and hand back a gradient of the
    # input's own (empty or zero) shape, not fail or come back wrong-sized.
    rng = np.random.default_rng(84)
    e = Tensor(np.zeros((0, 3)), requires_grad=True)
    b = randt(rng, (3,))
    w = randt(rng, (3, 2))
    p = Tensor(np.zeros((2, 0)), requires_grad=True)
    q = Tensor(np.zeros((0, 4)), requires_grad=True)
    loss = (((e + b) @ w).sum() + (e * b).sum(axis=0).sum()
            + (p @ q).sum() + e.reshape(3, 0).transpose().exp().sum()
            + e[[], :].sum() + e.masked_fill(np.zeros((0, 3), bool), 1.0).sum())
    assert float(loss.data) == 0.0
    loss.backward()
    assert e.grad.shape == (0, 3)
    assert p.grad.shape == (2, 0) and q.grad.shape == (0, 4)
    # every route from b and w to the loss passes through an empty tensor
    assert np.array_equal(b.grad, np.zeros(3))
    assert np.array_equal(w.grad, np.zeros((3, 2)))
    # the checker itself accepts an empty tensor (there is nothing to perturb)
    assert gradcheck(lambda: (e * b).sum() + (b * b).sum(), [e, b], tol=TOL) < TOL


# -------------------------------------------------------------- indexing

def test_getitem_boolean_mask_reversed_slice_and_mixed_indices():
    rng = np.random.default_rng(85)
    a = randt(rng, (4, 5))
    mask = rng.random((4, 5)) < 0.5
    cm = const(rng, (int(mask.sum()),))
    gradcheck(lambda: (a[mask] * cm).sum(), [a], tol=TOL)
    cr = const(rng, (4, 3))
    gradcheck(lambda: (a[::-1, 4:1:-1] * cr).sum(), [a], tol=TOL)
    cx = const(rng, (2, 3))
    cols = np.array([3, 0, 3])  # column 3 twice, alongside a slice
    gradcheck(lambda: (a[1:3, cols] * cx).sum(), [a], tol=TOL)


# ------------------------------------------------------- numerical stability

def test_log_softmax_gradient_is_finite_and_exact_at_extreme_logits():
    # d/dx sum(c * log_softmax(x)) = c - softmax(x) * sum(c), per row. At
    # logits of 1e4 the naive log(sum(exp)) is inf; the shifted one is exact.
    big = np.array([[1e4, 0.0, -1e4], [-1e4, 1e4, 1e4 - 1.0]])
    x = Tensor(big, requires_grad=True)
    c = np.array([[0.5, -1.0, 2.0], [1.0, 0.3, -0.2]])
    (log_softmax(x) * Tensor(c)).sum().backward()
    s = np.exp(big - big.max(-1, keepdims=True))
    s /= s.sum(-1, keepdims=True)
    assert np.isfinite(x.grad).all()
    assert np.allclose(x.grad, c - s * c.sum(-1, keepdims=True), rtol=0, atol=1e-12)


def test_softmax_is_shift_invariant_in_value_and_gradient():
    # The detached max shift must not change the derivative: the gradient at
    # x + 1000 equals the gradient at x, and both pass a gradient check.
    rng = np.random.default_rng(86)
    x = randt(rng, (3, 5), scale=2.0)
    y = Tensor(x.data + 1000.0, requires_grad=True)
    c = const(rng, (3, 5))
    gradcheck(lambda: (softmax(x) * c).sum(), [x], tol=TOL)
    gradcheck(lambda: (softmax(y) * c).sum(), [y], tol=TOL)
    assert np.allclose(softmax(y).data, softmax(x).data, rtol=0, atol=1e-12)
    assert np.allclose(y.grad, x.grad, rtol=0, atol=1e-12)


def test_masked_minus_infinity_logits_get_zero_gradient():
    # A logit of -inf (a fully masked position) has probability 0: finite
    # outputs elsewhere, and exactly zero gradient into the masked entry.
    x = Tensor(np.array([[0.5, -np.inf, 1.5], [-np.inf, 2.0, 0.0]]),
               requires_grad=True)
    c = Tensor(np.array([[1.0, 2.0, 3.0], [-1.0, 0.5, 2.0]]))
    s = softmax(x)
    assert np.isfinite(s.data).all() and np.allclose(s.data.sum(-1), 1.0)
    (s * c).sum().backward()
    assert np.isfinite(x.grad).all()
    assert x.grad[0, 1] == 0.0 and x.grad[1, 0] == 0.0
    x.grad = None
    loss = cross_entropy(x, np.array([2, 1]))
    assert np.isfinite(float(loss.data))
    loss.backward()
    assert np.isfinite(x.grad).all() and x.grad[0, 1] == 0.0


def test_saturated_tanh_and_underflowing_exp_have_finite_gradients():
    # A sigmoid-shaped curve from the primitives (the model has no sigmoid
    # op): (tanh(x/2) + 1) / 2 must saturate to exactly 0 and 1 with a zero,
    # not NaN, gradient, and exp must underflow to 0 without trouble.
    x = Tensor(np.array([-1e3, -30.0, 0.0, 30.0, 1e3]), requires_grad=True)
    sig = ((x * 0.5).tanh() + 1.0) * 0.5
    assert sig.data[0] == 0.0 and sig.data[-1] == 1.0
    sig.sum().backward()
    assert np.isfinite(x.grad).all()
    assert x.grad[0] == 0.0 and x.grad[-1] == 0.0 and x.grad[2] == 0.25
    y = Tensor(np.array([-1e3, 0.0]), requires_grad=True)
    y.exp().sum().backward()
    assert np.array_equal(y.grad, [0.0, 1.0])


def test_division_by_small_denominators_matches_the_closed_form():
    # Near zero a central difference is too coarse to trust, so compare the
    # analytic gradient of a / b with d/da = 1/b and d/db = -a/b^2 directly.
    a = Tensor(np.array([1.0, -2.0, 3.0]), requires_grad=True)
    b = Tensor(np.array([1e-4, -1e-8, 1e-12]), requires_grad=True)
    (a / b).sum().backward()
    assert np.allclose(a.grad, 1.0 / b.data, rtol=1e-15, atol=0)
    assert np.allclose(b.grad, -a.data / b.data ** 2, rtol=1e-15, atol=0)


# --------------------------------------------------------------------- dtype

def test_every_input_dtype_becomes_float64_and_so_does_every_gradient():
    # Integer, boolean and float32 data would each round or truncate a
    # central-difference perturbation of 1e-5; the engine is float64 only.
    for data in (np.arange(6).reshape(2, 3), np.eye(2, 3, dtype=bool),
                 np.ones((2, 3), np.float32), [[1, 2, 3], [4, 5, 6]]):
        x = Tensor(data, requires_grad=True)
        assert x.data.dtype == np.float64
        y = x * np.float32(0.5) + np.arange(3)  # float32 / int constants
        assert y.data.dtype == np.float64
        y.sum().backward(np.int64(2))           # an integer seed gradient
        assert x.grad.dtype == np.float64
        assert np.array_equal(x.grad, np.full((2, 3), 1.0))
    x = randt(np.random.default_rng(87), (2, 3))
    assert gradcheck(lambda: (x * np.float32(0.1)).sum(), [x], tol=TOL) < TOL
