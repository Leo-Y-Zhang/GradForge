"""Same seed => bit-identical training. Exact float equality, not approx."""
from gradforge.data import load_text
from gradforge.train import train

_TEXT = load_text()[:5000]


def _run(seed: int):
    return train(steps=15, seed=seed, d_model=32, n_layer=1, n_head=2,
                 block_size=32, batch_size=8, text=_TEXT, quiet=True)


def test_same_seed_bit_identical_loss_curve():
    a = _run(seed=7)
    b = _run(seed=7)
    assert a == b  # exact equality of every float, all 15 steps


def test_different_seed_differs():
    # guards the test above against being vacuously true
    assert _run(seed=7) != _run(seed=8)
