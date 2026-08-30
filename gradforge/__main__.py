"""GradForge CLI.

  gradforge gradcheck [--tol 1e-6] [--verbose]
      Check every analytic gradient against a central-difference estimate and
      print the worst relative error for each. This is the repository's central
      claim -- that the engine proves its own derivatives rather than asking to
      be trusted -- so it is a command you can run, not a paragraph you have to
      believe. Exits non-zero if any op breaches the tolerance.

  gradforge train [--steps N] [--out model.npz]
      Train the demo character-level GPT on the bundled excerpt. Prints the
      loss as it goes; the whole run is deterministic for a given seed.

  gradforge sample [--model model.npz] [--prompt TEXT] [--tokens N]
      Generate text from a trained checkpoint.

  gradforge bench [--size N]
      Measure matmul throughput, which is the number that decided numpy over
      nested lists. The README quotes it; this is where it comes from.

Every derivative rule in this engine is hand-written. Nothing here calls a
framework's autograd.
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from .gradcheck import gradcheck
from .tensor import Tensor


def _rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


def _t(rng, *shape, positive: bool = False) -> Tensor:
    """A random tensor that requires grad.

    `positive` shifts the sample away from zero for ops whose derivative is
    undefined or ill-conditioned there -- log at 0, and relu exactly at the
    kink, where a central difference straddles two different one-sided
    derivatives and disagrees with either.
    """
    data = rng.standard_normal(shape)
    if positive:
        data = np.abs(data) + 0.5
    return Tensor(data, requires_grad=True)


def _primitive_cases(rng):
    """A list of (name, fn, wrt) triples for gradcheck.

    `fn` must rebuild the computation each call, because gradcheck perturbs the
    inputs in place between evaluations. Closing over the tensors does that;
    capturing the *result* would not.
    """
    cases = []

    def add(name, wrt, fn):
        cases.append((name, fn, wrt))

    a, b = _t(rng, 3, 4), _t(rng, 3, 4)
    add("add", [a, b], lambda: (a + b).sum())
    add("mul", [a, b], lambda: (a * b).sum())
    add("chained (a*b + a).tanh()", [a, b], lambda: ((a * b) + a).tanh().sum())

    c, d = _t(rng, 3, 4), _t(rng, 1, 4)
    add("broadcast add", [c, d], lambda: (c + d).sum())

    e, f = _t(rng, 3, 4), _t(rng, 4, 2)
    add("matmul", [e, f], lambda: (e @ f).sum())

    g, h = _t(rng, 2, 3, 4), _t(rng, 2, 4, 3)
    add("batched matmul", [g, h], lambda: (g @ h).sum())

    i = _t(rng, 3, 3)
    add("exp", [i], lambda: i.exp().sum())
    add("tanh", [i], lambda: i.tanh().sum())

    # log needs a strictly positive sample, and relu must avoid the kink: a
    # central difference straddling zero averages two different one-sided
    # derivatives and disagrees with either.
    j = _t(rng, 3, 3, positive=True)
    add("log", [j], lambda: j.log().sum())
    add("relu", [j], lambda: j.relu().sum())

    k = _t(rng, 3, 4)
    add("sum(axis=1)", [k], lambda: k.sum(axis=1).sum())
    add("mean", [k], lambda: k.mean().sum())
    add("reshape", [k], lambda: k.reshape(4, 3).sum())
    add("transpose", [k], lambda: k.transpose(1, 0).sum())
    return cases


def cmd_gradcheck(args) -> int:
    rng = _rng(args.seed)
    print("Central-difference gradient check")
    print(f"  h = 1e-5, float64, tolerance {args.tol:g}\n")
    print(f"  {'operation':<28} {'max rel. error':>15}   verdict")
    print(f"  {'-' * 28} {'-' * 15}   {'-' * 7}")

    worst = 0.0
    failures = 0
    for name, fn, wrt in _primitive_cases(rng):
        try:
            err = gradcheck(fn, wrt, tol=args.tol)
            worst = max(worst, err)
            print(f"  {name:<28} {err:>15.3e}   ok")
        except AssertionError as exc:
            failures += 1
            print(f"  {name:<28} {'FAILED':>15}   {exc}")

    if not args.skip_model:
        print(f"\n  {'end-to-end':<28}")
        try:
            err = _gradcheck_gpt(args.tol, args.seed)
            worst = max(worst, err)
            print(f"  {'GPT loss (all parameters)':<28} {err:>15.3e}   ok")
        except AssertionError as exc:
            failures += 1
            print(f"  {'GPT loss (all parameters)':<28} {'FAILED':>15}   {exc}")

    print(f"\n  worst relative error across all checks: {worst:.3e}")
    if failures:
        print(f"  {failures} check(s) FAILED", file=sys.stderr)
        return 1
    print("  every analytic gradient agrees with the numerical one")
    return 0


def _gradcheck_gpt(tol: float, seed: int) -> float:
    """The whole model, not just its parts.

    Per-op checks can all pass while the composition is wrong, so the loss of a
    real (tiny) GPT is differentiated end to end. Parameters are sampled rather
    than exhausted because a full sweep would be thousands of forward passes.
    """
    from .nn import GPT
    rng = np.random.default_rng(seed)
    vocab, block = 11, 8
    model = GPT(vocab, block, d_model=16, n_layer=1, n_head=2, seed=seed)
    x = rng.integers(0, vocab, size=(2, block))
    y = rng.integers(0, vocab, size=(2, block))
    params = model.parameters()
    return gradcheck(lambda: model.loss(x, y), params, tol=tol,
                     rng=rng, max_elems=3)


def cmd_train(args) -> int:
    from .train import train
    losses = train(steps=args.steps, seed=args.seed, d_model=args.d_model,
                   n_layer=args.n_layer, n_head=args.n_head,
                   block_size=args.block_size, batch_size=args.batch_size,
                   lr=args.lr, out=args.out)
    print(f"\nfirst loss {losses[0]:.4f} -> final loss {losses[-1]:.4f} "
          f"over {len(losses)} steps")
    return 0


def cmd_sample(args) -> int:
    from .sample import generate
    from .train import load_checkpoint
    model, chars = load_checkpoint(args.model)
    text = generate(model, chars, prompt=args.prompt, tokens=args.tokens,
                    seed=args.seed)
    print(text)
    return 0


def cmd_bench(args) -> int:
    """Why numpy, in one measurement.

    The README claims nested-list tensors would make the demo untrainable.
    This is the arithmetic behind that claim, run on the current machine.
    """
    n = args.size
    rng = _rng(0)
    a = rng.standard_normal((n, n))
    b = rng.standard_normal((n, n))
    flops = 2.0 * n ** 3

    best = float("inf")
    for _ in range(args.repeats):
        t0 = time.perf_counter()
        a @ b
        best = min(best, time.perf_counter() - t0)
    print(f"matmul {n}x{n}: {best * 1e3:.1f} ms, {flops / best / 1e9:.1f} Gflop/s")
    print(f"a pure-Python triple loop at ~60 Mflop/s would take "
          f"{flops / 60e6:,.0f} s for the same product")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="gradforge", description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    sub = p.add_subparsers(dest="command")

    g = sub.add_parser("gradcheck", help="verify every gradient numerically")
    g.add_argument("--tol", type=float, default=1e-6)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--skip-model", action="store_true",
                   help="primitives only; skip the end-to-end GPT check")
    g.set_defaults(func=cmd_gradcheck)

    t = sub.add_parser("train", help="train the demo character-level GPT")
    t.add_argument("--steps", type=int, default=1500)
    t.add_argument("--seed", type=int, default=1)
    t.add_argument("--d-model", type=int, default=64)
    t.add_argument("--n-layer", type=int, default=2)
    t.add_argument("--n-head", type=int, default=4)
    t.add_argument("--block-size", type=int, default=64)
    t.add_argument("--batch-size", type=int, default=16)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--out", default="gradforge_model.npz")
    t.set_defaults(func=cmd_train)

    s = sub.add_parser("sample", help="generate text from a checkpoint")
    s.add_argument("--model", default="gradforge_model.npz")
    s.add_argument("--prompt", default="\n")
    s.add_argument("--tokens", type=int, default=400)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=cmd_sample)

    b = sub.add_parser("bench", help="matmul throughput on this machine")
    b.add_argument("--size", type=int, default=512)
    b.add_argument("--repeats", type=int, default=5)
    b.set_defaults(func=cmd_bench)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return 2
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
