"""Optimizer proofs: SGD+momentum against a hand-computed trace, and Adam
convergence on a seeded regression with a known optimum."""
import numpy as np

from gradforge.optim import SGD, Adam
from gradforge.tensor import Tensor


def test_sgd_momentum_matches_hand_computed_trace():
    # f(p) = (p - 3)^2, p0 = 0, lr = 0.1, momentum = 0.9
    # g0 = -6            -> v1 = -6,                p1 = 0 - 0.1*(-6)  = 0.6
    # g1 = 2(0.6-3)=-4.8 -> v2 = 0.9*(-6) - 4.8 = -10.2
    #                                              p2 = 0.6 + 1.02 = 1.62
    p = Tensor([0.0], requires_grad=True)
    opt = SGD([p], lr=0.1, momentum=0.9)
    for expected in (0.6, 1.62):
        loss = ((p - 3.0) * (p - 3.0)).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
        assert np.allclose(p.data, [expected], atol=1e-12)


def test_adam_solves_seeded_regression_to_known_optimum():
    # noiseless linear regression: the global optimum is exactly (w*, b*)
    # with zero loss; Adam must find it
    rng = np.random.default_rng(50)
    X = rng.normal(size=(32, 3))
    w_true = np.array([[1.5], [-2.0], [0.5]])
    b_true = 0.7
    y = X @ w_true + b_true

    Xt, yt = Tensor(X), Tensor(y)
    w = Tensor(np.zeros((3, 1)), requires_grad=True)
    b = Tensor(np.zeros(1), requires_grad=True)
    opt = Adam([w, b], lr=0.05)
    loss = None
    for _ in range(400):
        pred = Xt @ w + b
        diff = pred - yt
        loss = (diff * diff).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert float(loss.data) < 1e-6
    assert np.allclose(w.data, w_true, atol=1e-3)
    assert np.allclose(b.data, [b_true], atol=1e-3)


def test_adam_first_step_is_lr_sized():
    # with bias correction, the very first Adam step is ~lr * sign(g);
    # without correction it would be ~40x smaller at these betas
    p = Tensor([0.0], requires_grad=True)
    opt = Adam([p], lr=0.01)
    loss = (p * 3.0).sum()  # constant gradient 3
    loss.backward()
    opt.step()
    assert np.isclose(p.data[0], -0.01, rtol=1e-3)


def test_adam_bias_correction_counts_each_parameters_own_steps():
    # step() skips a parameter whose grad is None, so a parameter can join
    # training late. Its moment estimates then hold one update, and bias
    # correction has to divide by 1 - beta^1, not by 1 - beta^t for however
    # many steps the others have taken -- otherwise its first move is about
    # 0.74 lr here instead of lr.
    p = Tensor([0.0], requires_grad=True)
    q = Tensor([0.0], requires_grad=True)
    opt = Adam([p, q], lr=0.01)
    (p * 3.0).sum().backward()          # step 1: q takes no part
    opt.step()
    assert q.data[0] == 0.0
    opt.zero_grad()
    (p * 3.0 + q * 5.0).sum().backward()
    opt.step()
    assert np.isclose(q.data[0], -0.01, rtol=1e-3)
