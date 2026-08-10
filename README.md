# GradForge

A reverse-mode autograd engine written from scratch, a tiny GPT built on top
of it, and a test suite that proves the mathematics instead of asserting it.

The only runtime dependency is numpy, used strictly as array arithmetic --
every derivative rule is hand-written and every one of them is verified
against central-difference numerical differentiation. (Why numpy at all?
Nested-list tensors were measured at 62.6 Mflop/s on the development machine
versus numpy's 15.1 Gflop/s; training the demo model would take eleven hours
instead of two minutes. The measurement is in [docs/DESIGN.md](docs/DESIGN.md).)

## What proves what

Frameworks ask you to trust `.backward()`. This repo doesn't ask:

| Claim | Proof |
|---|---|
| Every op's gradient is correct | Central-difference check on each primitive op (rel. tol 1e-6, float64), including broadcast operands, batched matmul, duplicate fancy indices |
| Every layer's gradient is correct | Same check through Linear, LayerNorm, Embedding (with repeated tokens), causal self-attention, MLP, a full block, and the whole GPT end to end |
| Attention is actually causal | Perturb the last position, assert earlier outputs are bit-unchanged |
| Softmax/cross-entropy are stable | Logits of +-1e4 must give finite outputs and finite gradients; the test also shows the naive formula overflowing there |
| Training is deterministic | Same seed, two independent runs, exact float equality of every loss value |
| The gradients can actually learn | A 1-layer GPT must memorize a fixed batch to loss < 0.1 |
| Adam is implemented right | Solves a seeded regression to its known optimum; a separate test pins the bias-corrected first-step size, which uncorrected Adam fails |
| SGD+momentum is implemented right | Two steps match a hand-computed trace to 1e-12 |

Each key test was watched failing first: break the op (flip a sign in matmul's
backward, swap scatter-add for assignment, drop the softmax shift, remove the
causal mask, un-correct Adam), see red, restore, see green. A test that has
never failed is decoration.

## The demo model

A character-level GPT (2 layers, 4 heads, d_model 64, block size 64, ~113k
parameters) trained for 1500 Adam steps on a bundled 47 KB public-domain
excerpt of *Alice's Adventures in Wonderland* (Project Gutenberg ebook #11).
Training takes about two minutes on an ordinary CPU:

```
python -m gradforge.train --steps 1500 --out gradforge_model.npz
python -m gradforge.sample --checkpoint gradforge_model.npz --prompt "Alice " --tokens 320 --temperature 0.8 --seed 0
```

The run behind the sample below went from loss 4.8625 at step 1 to
**1.4036 at step 1500** (cross-entropy, nats/char; uniform over the 66-char
vocabulary would be 4.19) in 172 seconds of CPU. The command above
regenerates the sample below character for character (shown re-wrapped;
the raw output is one long line after the first break). This is what ~113k
honest parameters buy you at that loss -- word-shaped Carroll pastiche,
not sense, and the README won't pretend otherwise:

```
Alice of anything
those had she world ahad time nother the birds and largilling in
words the thist tor think the was and it the was veryes, and the
sair! Hold hard yessotily, down ther. No monturnemed sation, and
and had splily a snotice it she watered the hurry_ shill had
natice, shut there that rememement thinky all the co
```

(One more honest note: prompted with a bare newline instead, the model often
emits the excerpt's `*  *  *  *` scene-divider rows rather than prose -- it
memorized the typography of its corpus along with the words.)

## Layout

```
gradforge/tensor.py      the engine: Tensor, backward graph, every gradient rule
gradforge/functional.py  softmax, log_softmax, gelu, cross_entropy (composed)
gradforge/nn.py          Linear, LayerNorm, Embedding, attention, MLP, GPT
gradforge/optim.py       SGD with momentum, Adam
gradforge/gradcheck.py   the central-difference checker the tests are built on
gradforge/data.py        char codec + batching over the bundled excerpt
gradforge/train.py       bounded demo training (python -m gradforge.train)
gradforge/sample.py      text generation      (python -m gradforge.sample)
tests/                   the proof obligations (42 tests)
docs/DESIGN.md           architecture and the measured numpy decision
```

## Install and run

```
pip install -e .[dev]
python -m pytest
```

Python 3.10+. Tests take a few seconds.

## What this deliberately is not

A real framework has a float32/mixed-precision path, GPU kernels, graph
memory management, operator fusion, a stable public API, and years of edge
cases. This has none of that: float64 on CPU, basic-plus-integer-array
indexing only, scalar exponents only, and a Tensor API just big enough for
the model it trains. The point is a complete, verified core that one person
can read end to end: 301 lines for the engine, 57 for the composed math,
161 for the layers, 62 for the optimizers, 71 for the gradient checker --
652 lines of source for a working, gradient-checked transformer.

## License

MIT. The bundled Alice excerpt is public domain (typography normalized,
Project Gutenberg boilerplate removed).
