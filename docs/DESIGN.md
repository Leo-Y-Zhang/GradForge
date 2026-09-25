# GradForge design

One sentence: a reverse-mode autograd engine over numpy tensors, a small set of
neural-net layers built only from that engine's ops, and a character-level GPT
trained on a public-domain text excerpt -- with every mathematical claim backed
by a test.

## Why this exists

Deep-learning frameworks are easy to use and hard to trust from the inside:
you call `.backward()` and believe. GradForge is the opposite trade. It is
small enough to read in an afternoon, and it does not ask for belief -- every
gradient the engine produces is checked against central-difference numerical
differentiation, and the training claims (determinism, convergence,
memorization) are pytest assertions, not README prose.

## The pure-Python question, measured

The first design decision was whether tensors could be nested Python lists
(zero dependencies) or numpy arrays (one dependency). Measured on the target
machine (Python 3.13.14, numpy 2.2.1, Windows 11):

```
matmul 64x64 @ 64x64 (524,288 flops):
  pure Python:     8.38 ms  (   62.6 Mflop/s)
  numpy      :     0.03 ms  (15147.8 Mflop/s)
  speedup    : 242x
```

Extrapolating to the demo GPT (~200k params, batch 16 x block 64, fwd+bwd
~ 6 * params * tokens flops/step, 2000 steps):

```
  pure Python: est  19.64 s/step ->  654.6 min total
  numpy      : est   0.08 s/step ->    2.7 min total
```

Eleven hours versus three minutes. Pure Python cannot train the demo in
reasonable time, so numpy is the single runtime dependency. (That estimate was
made for the planned configuration. The demo as shipped is smaller -- 112,640
parameters, 1500 steps -- which puts the same pure-Python estimate at about
4.6 hours, against a measured 172 seconds with numpy; the conclusion stands.)
The autograd logic itself -- graph construction, topological sort, every
backward rule -- is still written from scratch; numpy is used only as the
array arithmetic substrate, never `numpy.gradient` or any autodiff shortcut.

All tensors are float64. That costs ~2x speed versus float32 but makes
central-difference gradient checks sharp (relative errors around 1e-9 instead
of 1e-3), and at this model scale speed is not the constraint.

## Module map

```
gradforge/
  tensor.py      Tensor: data + grad + backward closure; all primitive ops
  functional.py  softmax, log_softmax, gelu, cross_entropy (composed from ops)
  nn.py          Module, Linear, LayerNorm, Embedding, CausalSelfAttention,
                 MLP, Block, GPT
  optim.py       SGD (with momentum), Adam
  gradcheck.py   central-difference checker used by the test suite
  __main__.py    the gradforge command: gradcheck, train, sample, bench
  data.py        char codec + batch sampling over the bundled excerpt
  train.py       python -m gradforge.train  (bounded demo training)
  sample.py      python -m gradforge.sample (generate text from a checkpoint)
  data/alice_excerpt.txt   public-domain training text (Carroll, via
                           Project Gutenberg ebook #11, boilerplate stripped)
tests/           the proof obligations, see below
```

## Autograd design

- `Tensor` wraps a float64 ndarray, carries `grad`, `requires_grad`, a
  `_backward` closure and its parent nodes. `backward()` does an iterative
  topological sort (no recursion limit surprises) and runs closures in
  reverse order. This is the micrograd architecture generalized to tensors.
- Broadcasting: every binary op's backward funnels through one
  `_unbroadcast(grad, shape)` helper that sums gradient over broadcast axes.
  One helper, one set of tests, instead of per-op ad-hoc logic.
- Primitive ops: add, sub, mul, div, neg, pow (scalar exponent), matmul
  (batched), exp, log, tanh, relu, sum, mean, reshape, transpose, getitem
  (basic and integer-array indexing), masked_fill. Everything else --
  softmax, gelu, layernorm, attention, cross-entropy -- is composed from
  these, so their backward passes are *derived*, not hand-written, and a
  gradcheck of the primitives plus a gradcheck of the compositions covers
  the whole model.
- Indexing backward uses `np.add.at` (scatter-add), not assignment. This is
  the classic silent-wrongness spot: with duplicate indices (an embedding
  looking up the same token twice), assignment keeps only the last gradient.
  There is a test whose whole job is to catch that bug.
- Softmax/log-softmax subtract the row max *as a detached constant* before
  exponentiating. Softmax is shift-invariant so the value and the gradient
  are unchanged, but exp never sees large arguments. The stability test
  feeds logits of 1e4 and asserts finite outputs; the naive formula
  overflows there.

## The model

Character-level GPT: token embedding + learned positional embedding, 2
pre-norm transformer blocks (LayerNorm -> causal self-attention -> residual,
LayerNorm -> MLP with gelu -> residual), final LayerNorm, linear head.
Default demo config: d_model 64, 4 heads, block size 64, batch 16, Adam.
Training data is a ~47 KB excerpt of Alice's Adventures in Wonderland
(public domain), bundled in the package. Training is a bounded script --
minutes of CPU, not an open-ended run -- and the README quotes real output
with its real loss.

## Proof obligations (the test suite)

Every claim below is a pytest test, and each key test was watched failing
against a deliberately broken implementation before being trusted (the break,
the red output, and the fix are recorded in the repo history / PR notes).

1. Gradient checks, ops: central difference vs autograd for every primitive
   op, seeded inputs, relative error < 1e-6 (float64, h=1e-5). Includes the
   nasty cases: broadcasting in both operands, batched matmul, duplicate
   fancy indices, masked positions.
2. Gradient checks, layers: Linear, LayerNorm, Embedding, attention, MLP,
   Block, and a miniature end-to-end GPT loss (sampled elements for the
   larger tensors, seeded).
3. Determinism: same seed => two independent 15-step training runs produce
   bit-identical loss sequences (exact float equality, not approx).
4. Overfit: a 1-layer model memorizes a tiny fixed batch to loss < 0.1.
   If gradients or the optimizer are subtly wrong, this is where it shows.
5. Optimizer convergence: Adam drives a seeded linear regression to its
   known least-squares optimum; SGD+momentum steps match a hand-computed
   trace.
6. Stability: softmax/log-softmax/cross-entropy finite and correct at
   logits of +-1e4, softmax rows sum to 1, cross-entropy matches an
   analytically computed value on a small known case.
7. The checker checked: gradcheck must reject a derivative off by one part
   in 10^4 and a gradient of the wrong shape, and `gradforge gradcheck` must
   exit non-zero when an op's backward is broken.
8. What gradient checks cannot see: attention and LayerNorm forward values
   against plain numpy, the one-character shift between inputs and targets,
   and an exact checkpoint save/load round trip.

## Failure modes and rollback

- This is a greenfield library with no consumers and no deployment surface:
  rollback for any bad commit is `git revert`; nothing external depends on it.
- The known sharp edges are documented rather than hidden: no GPU, no
  float32 path, no graph freeing during backward (fine at this scale, would
  be a memory problem at real scale), basic-plus-integer-array indexing only,
  scalar exponents only in pow.
- CI runs the full suite on every push, so a regression in any proof
  obligation is a red X, not a silent decay.

## Out of scope, deliberately

Kernel fusion, GPU, mixed precision, graph optimization, distributed
anything, model formats, a Tensor type covering all of numpy's API. The
point is a complete, verified core, not a small PyTorch.
