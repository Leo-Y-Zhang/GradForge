"""Forward values against plain numpy references.

A gradient check proves that backward is the derivative of forward. It cannot
tell whether forward is the intended function: attention scaled by the model
width instead of the head width, or a LayerNorm off by a constant factor, has
perfectly consistent gradients and passes every check in the suite.
"""
import numpy as np

from gradforge.nn import CausalSelfAttention, LayerNorm
from gradforge.tensor import Tensor


def test_layernorm_matches_numpy():
    rng = np.random.default_rng(70)
    ln = LayerNorm(6)
    ln.gamma.data[...] = rng.normal(size=6)
    ln.beta.data[...] = rng.normal(size=6)
    x = rng.normal(size=(3, 6)) * 4.0 + 2.0
    xhat = (x - x.mean(-1, keepdims=True)) / np.sqrt(x.var(-1, keepdims=True) + 1e-5)
    ref = xhat * ln.gamma.data + ln.beta.data
    assert np.allclose(ln(Tensor(x)).data, ref, rtol=0.0, atol=1e-12)


def test_causal_attention_matches_numpy():
    rng = np.random.default_rng(71)
    b, t, c, h = 2, 5, 8, 2
    hd = c // h
    attn = CausalSelfAttention(c, h, rng)
    x = rng.normal(size=(b, t, c))

    qkv = x @ attn.qkv.weight.data + attn.qkv.bias.data
    q, k, v = (a.reshape(b, t, h, hd).transpose(0, 2, 1, 3)
               for a in np.split(qkv, 3, axis=-1))
    scores = q @ k.transpose(0, 1, 3, 2) / np.sqrt(hd)
    scores = np.where(np.triu(np.ones((t, t), dtype=bool), k=1), -np.inf, scores)
    w = np.exp(scores - scores.max(-1, keepdims=True))
    w /= w.sum(-1, keepdims=True)
    y = (w @ v).transpose(0, 2, 1, 3).reshape(b, t, c)
    ref = y @ attn.proj.weight.data + attn.proj.bias.data

    assert np.allclose(attn(Tensor(x)).data, ref, rtol=0.0, atol=1e-12)
