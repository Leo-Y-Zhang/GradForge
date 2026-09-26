"""The one-command proof has to fail when a gradient is wrong.

`gradforge gradcheck` is what the README points a sceptical reader at, so its
exit status is the claim: 0 only when every analytic gradient agrees with the
numerical one, non-zero as soon as one does not.
"""
import numpy as np

from gradforge.__main__ import main
from gradforge.tensor import Tensor


def test_gradcheck_command_passes_on_the_real_engine(capsys):
    assert main(["gradcheck"]) == 0
    assert "FAILED" not in capsys.readouterr().out


def test_gradcheck_command_exits_nonzero_on_a_wrong_gradient(monkeypatch, capsys):
    def tanh_with_a_sign_error(self):
        out = self._make(np.tanh(self.data), (self,), None)

        def backward():
            self._accum(out.grad * (1.0 + out.data * out.data))

        out._backward = backward if out.requires_grad else None
        return out

    monkeypatch.setattr(Tensor, "tanh", tanh_with_a_sign_error)
    assert main(["gradcheck", "--skip-model"]) == 1
    out = capsys.readouterr().out
    assert "tanh" in out and "FAILED" in out


def test_gradcheck_command_checks_relu_on_both_sides_of_the_kink(monkeypatch, capsys):
    # relu checked only on positive inputs exercises the identity half, and
    # the model uses gelu, so a backward that never zeroes negative inputs
    # would pass the whole command.
    def relu_that_never_zeroes(self):
        out = self._make(np.maximum(self.data, 0.0), (self,), None)

        def backward():
            self._accum(out.grad)

        out._backward = backward if out.requires_grad else None
        return out

    monkeypatch.setattr(Tensor, "relu", relu_that_never_zeroes)
    assert main(["gradcheck", "--skip-model"]) == 1
    out = capsys.readouterr().out
    assert "relu" in out and "FAILED" in out
