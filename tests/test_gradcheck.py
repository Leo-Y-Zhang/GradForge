"""The checker itself gets checked.

Every other proof in this suite is only as good as gradcheck, so the one
thing it must never do is report success without having actually perturbed
the tensor it claims to have verified.
"""
import numpy as np

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
