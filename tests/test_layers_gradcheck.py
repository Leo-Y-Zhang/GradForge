"""Central-difference gradient checks for every layer, and for the full GPT.

Layers are checked with respect to their parameters AND their input, so both
the parameter gradients and the pass-through gradients are verified.
"""
import numpy as np

from gradforge.functional import cross_entropy
from gradforge.gradcheck import gradcheck
from gradforge.nn import GPT, MLP, Block, CausalSelfAttention, Embedding, LayerNorm, Linear
from helpers import const, randt

TOL = 1e-6


def test_linear():
    rng = np.random.default_rng(40)
    lin = Linear(4, 3, rng)
    x = randt(rng, (2, 4))
    c = const(rng, (2, 3))
    wrt = [lin.weight, lin.bias, x]
    assert gradcheck(lambda: (lin(x) * c).sum(), wrt, tol=TOL) < TOL


def test_layernorm():
    rng = np.random.default_rng(41)
    ln = LayerNorm(6)
    x = randt(rng, (3, 6), scale=2.0)
    c = const(rng, (3, 6))
    wrt = [ln.gamma, ln.beta, x]
    assert gradcheck(lambda: (ln(x) * c).sum(), wrt, tol=TOL) < TOL


def test_embedding_with_repeated_tokens():
    # repeated token: its row's gradient must accumulate across occurrences
    rng = np.random.default_rng(42)
    emb = Embedding(5, 4, rng, scale=1.0)
    idx = np.array([[0, 2, 0], [1, 0, 1]])
    c = const(rng, (2, 3, 4))
    assert gradcheck(lambda: (emb(idx) * c).sum(), [emb.weight], tol=TOL) < TOL


def test_causal_self_attention():
    rng = np.random.default_rng(43)
    attn = CausalSelfAttention(8, 2, rng)
    x = randt(rng, (2, 4, 8))
    c = const(rng, (2, 4, 8))
    wrt = attn.parameters() + [x]
    assert gradcheck(lambda: (attn(x) * c).sum(), wrt, tol=TOL) < TOL


def test_attention_is_causal():
    # position i's output must not depend on position j > i
    rng = np.random.default_rng(44)
    attn = CausalSelfAttention(8, 2, rng)
    x1 = randt(rng, (1, 4, 8), requires_grad=False)
    x2_data = x1.data.copy()
    x2_data[0, 3, :] += 10.0  # perturb only the LAST position
    from gradforge.tensor import Tensor
    y1 = attn(x1).data
    y2 = attn(Tensor(x2_data)).data
    assert np.allclose(y1[0, :3], y2[0, :3])  # earlier positions unchanged
    assert not np.allclose(y1[0, 3], y2[0, 3])  # last position did change


def test_mlp():
    rng = np.random.default_rng(45)
    mlp = MLP(6, rng)
    x = randt(rng, (2, 6))
    c = const(rng, (2, 6))
    wrt = mlp.parameters() + [x]
    gradcheck(lambda: (mlp(x) * c).sum(), wrt, tol=TOL)


def test_block():
    rng = np.random.default_rng(46)
    block = Block(8, 2, rng)
    x = randt(rng, (2, 3, 8))
    c = const(rng, (2, 3, 8))
    wrt = block.parameters() + [x]
    gradcheck(lambda: (block(x) * c).sum(), wrt, tol=TOL)


def test_gpt_end_to_end():
    # the whole model against central differences: every parameter tensor,
    # sampled elements (seeded), through embedding, attention, layernorm,
    # gelu, softmax and cross-entropy at once
    rng = np.random.default_rng(47)
    model = GPT(vocab_size=11, block_size=8, d_model=16, n_layer=1,
                n_head=2, seed=3)
    x = rng.integers(0, 11, (2, 8))
    y = rng.integers(0, 11, (2, 8))
    max_rel = gradcheck(lambda: cross_entropy(model.forward(x), y),
                        model.parameters(), tol=TOL, rng=rng, max_elems=20)
    assert max_rel < TOL
