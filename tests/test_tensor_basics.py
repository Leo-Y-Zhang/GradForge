import numpy as np
import pytest

from gradforge.tensor import Tensor


def test_requires_grad_propagates():
    a = Tensor([1.0, 2.0], requires_grad=True)
    b = Tensor([3.0, 4.0])
    assert (a + b).requires_grad
    assert (a * 2.0).requires_grad
    assert not (b + b).requires_grad


def test_reused_tensor_accumulates_grad():
    x = Tensor([2.0], requires_grad=True)
    y = x * x + x  # dy/dx = 2x + 1 = 5
    y.backward()
    assert np.allclose(x.grad, [5.0])


def test_second_backward_over_same_graph_repeats_the_same_gradient():
    # Intermediate nodes carry only transient gradient. If a second backward
    # over the same graph finds them still holding the first pass's values it
    # accumulates on top of those and pushes an inflated gradient down to the
    # leaf -- silently, with no error and no shape mismatch to give it away.
    x = Tensor([2.0], requires_grad=True)
    y = (x * x).sum()  # dy/dx = 2x = 4
    y.backward()
    assert np.allclose(x.grad, [4.0])
    x.grad = None
    y.backward()
    assert np.allclose(x.grad, [4.0])


def test_backward_needs_scalar():
    x = Tensor([1.0, 2.0], requires_grad=True)
    with pytest.raises(ValueError):
        (x * 2.0).backward()


def test_detach_cuts_graph():
    x = Tensor([3.0], requires_grad=True)
    y = (x * 2.0).detach() * x  # gradient flows only through the second x
    y.backward()
    assert np.allclose(x.grad, [6.0])


def test_constant_gets_no_grad():
    x = Tensor([1.0], requires_grad=True)
    c = Tensor([5.0])
    (x * c).backward()
    assert c.grad is None


def test_broadcast_shapes():
    a = Tensor(np.ones((2, 3, 4)), requires_grad=True)
    b = Tensor(np.ones((3, 1)), requires_grad=True)
    out = a * b
    assert out.shape == (2, 3, 4)
    out.sum().backward()
    assert a.grad.shape == (2, 3, 4)
    assert b.grad.shape == (3, 1)
    assert np.allclose(b.grad, 8.0)  # 2*4 broadcast copies each


def test_ndarray_on_the_left_defers_to_tensor():
    # Without Tensor opting out of numpy's ufunc machinery, `ndarray * Tensor`
    # never reaches Tensor.__rmul__: numpy broadcasts the Tensor as an opaque
    # object and returns an object array of per-element Tensors, detached from
    # any graph. No error is raised until something much later fails.
    x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    w = np.array([2.0, -1.0, 0.5])
    for out in (w * x, w + x, w - x, w / x):
        assert isinstance(out, Tensor)
        assert out.shape == (3,)
    x.grad = None
    (w * x).sum().backward()
    assert np.allclose(x.grad, w)
    x.grad = None
    (w / x).sum().backward()
    assert np.allclose(x.grad, -w / x.data ** 2)
