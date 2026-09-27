"""The training pipeline around the engine: batches in, checkpoints out.

Gradient checks cannot see either. A batch whose targets are not the inputs
shifted by one trains a model on the wrong task with perfectly correct
gradients, and a checkpoint that does not restore its weights makes the
README's sample irreproducible while every other test stays green.
"""
import numpy as np

from gradforge.data import get_batch
from gradforge.nn import GPT
from gradforge.train import load_checkpoint, save_checkpoint


def test_targets_are_the_inputs_shifted_one_character_ahead():
    ids = np.arange(200)  # id == position, so a window's contents name its offset
    x, y = get_batch(np.random.default_rng(0), ids, block_size=16, batch_size=64)
    assert x.shape == y.shape == (64, 16)
    assert (np.diff(x, axis=1) == 1).all()  # each row is one contiguous window
    assert (y == x + 1).all()               # and each target is the next id


def test_checkpoint_round_trip_restores_every_parameter(tmp_path):
    chars = list("abcdefg")
    model = GPT(vocab_size=7, block_size=8, d_model=16, n_layer=2, n_head=2, seed=5)
    rng = np.random.default_rng(0)
    for p in model.parameters():
        # values no freshly initialised GPT(seed=5) could have, so a loader
        # that skipped the weights would be caught
        p.data[...] = rng.normal(size=p.data.shape)
    path = str(tmp_path / "model.npz")
    save_checkpoint(path, model, chars)

    loaded, loaded_chars = load_checkpoint(path)
    assert loaded_chars == chars
    assert loaded.config == model.config
    for a, b in zip(model.parameters(), loaded.parameters(), strict=True):
        assert np.array_equal(a.data, b.data)
    idx = rng.integers(0, 7, (2, 8))
    assert np.array_equal(loaded.forward(idx).data, model.forward(idx).data)
