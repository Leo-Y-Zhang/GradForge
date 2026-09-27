"""The checker itself gets checked.

Every other proof in this suite is only as good as gradcheck, so the one
thing it must never do is report success without having actually perturbed
the tensor it claims to have verified.
"""
import numpy as np
import pytest

from gradforge.gradcheck import gradcheck
from gradforge.tensor import Tensor
from helpers import const

TOL = 1e-6


def test_gradcheck_perturbs_non_contiguous_tensors():
    # A Tensor built from a transposed view shares its base's buffer, and no
    # flattening reshape of it can be a view, so reshape(-1) returns a COPY.
    # Perturbing that copy never reaches the computation: the numeric
    # gradient comes out identically zero, which either raises a bogus
    # failure or -- where the true gradient is small -- passes while having
    # verified nothing at all.
    rng = np.random.default_rng(60)
    base = rng.normal(size=(3, 4))
    x = Tensor(base.T, requires_grad=True)
    assert not np.shares_memory(x.data.reshape(-1), x.data)  # the trap
    c = const(rng, (4, 3))
    assert gradcheck(lambda: (x * c).sum(), [x], tol=TOL) < TOL


def _scaled_exp(x, factor):
    """exp(x), with a backward rule that is off by `factor`."""
    out = x.exp()

    def backward():
        x._accum(out.grad * out.data * factor)

    out._backward = backward
    return out


def test_gradcheck_rejects_a_wrong_derivative():
    # The other half of the contract: every passing test in this suite means
    # something only if a wrong backward rule makes gradcheck raise. A
    # 1e-4 relative error is far below anything a sign or indexing bug
    # produces, and still two orders of magnitude over the tolerance.
    rng = np.random.default_rng(61)
    x = Tensor(rng.normal(size=(2, 3)), requires_grad=True)
    c = const(rng, (2, 3))
    assert gradcheck(lambda: (_scaled_exp(x, 1.0) * c).sum(), [x], tol=TOL) < TOL
    with pytest.raises(AssertionError, match="gradcheck failed"):
        gradcheck(lambda: (_scaled_exp(x, 1.0 + 1e-4) * c).sum(), [x], tol=TOL)


def test_gradcheck_refuses_a_gradient_of_the_wrong_shape():
    # A backward rule that forgets to sum over a broadcast axis leaves a (3,)
    # tensor holding a (2, 3) gradient. Here the first row is exactly the true
    # gradient and the second is zero, so a comparison by flat index -- which
    # only ever reads the first three entries -- would report a pass.
    x = Tensor(np.array([0.5, -1.0, 2.0]), requires_grad=True)

    def fn():
        out = x * 1.0

        def backward():
            g = np.zeros((2, 3))
            g[0] = out.grad
            x._accum(g)

        out._backward = backward
        return out.sum()

    with pytest.raises(AssertionError, match="shape"):
        gradcheck(fn, [x], tol=TOL)


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_gradcheck_rejects_a_non_finite_gradient(bad):
    # NaN compares false with everything, so a check written as
    # `error >= tol` lets a NaN gradient through as a pass -- and an infinite
    # one too, since inf/inf is NaN. The loss here is finite everywhere.
    x = Tensor(np.array([0.5, -1.0, 2.0]), requires_grad=True)

    def fn():
        out = x * 1.0

        def backward():
            x._accum(out.grad * bad)

        out._backward = backward
        return out.sum()

    with pytest.raises(AssertionError, match="gradcheck failed"):
        gradcheck(fn, [x], tol=TOL)
