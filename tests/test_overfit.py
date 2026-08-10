"""A small GPT must be able to memorize a tiny fixed batch.

If any gradient in the stack is subtly wrong, the model plateaus far above
the threshold; correct gradients drive the loss toward zero.
"""
import numpy as np

from gradforge.data import CharCodec, load_text
from gradforge.nn import GPT
from gradforge.optim import Adam


def test_gpt_memorizes_tiny_batch():
    text = load_text()[:2000]
    codec = CharCodec(text)
    ids = codec.encode(text)
    x = np.stack([ids[0:32], ids[100:132]])
    y = np.stack([ids[1:33], ids[101:133]])

    model = GPT(codec.vocab_size, block_size=32, d_model=32, n_layer=1,
                n_head=2, seed=9)
    opt = Adam(model.parameters(), lr=3e-3)
    first = None
    loss = None
    for _ in range(400):
        loss = model.loss(x, y)
        if first is None:
            first = float(loss.data)
        opt.zero_grad()
        loss.backward()
        opt.step()
    final = float(loss.data)
    assert first > 3.0  # started near ln(vocab): genuinely untrained
    assert final < 0.1, f"failed to memorize: loss {final:.4f}"
