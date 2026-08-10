"""Train the demo character-level GPT on the bundled excerpt.

Bounded by design: the default run is a few minutes of CPU. Usage:

    python -m gradforge.train --steps 1500 --out gradforge_model.npz
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from .data import CharCodec, get_batch, load_text
from .nn import GPT
from .optim import Adam

__all__ = ["train"]


def save_checkpoint(path: str, model: GPT, chars: list[str]) -> None:
    arrays = {f"p{i}": p.data for i, p in enumerate(model.parameters())}
    meta = json.dumps({"config": model.config, "chars": "".join(chars)})
    np.savez(path, meta=np.frombuffer(meta.encode("utf-8"), dtype=np.uint8),
             **arrays)


def load_checkpoint(path: str):
    with np.load(path) as z:
        meta = json.loads(bytes(z["meta"]).decode("utf-8"))
        model = GPT(**meta["config"])
        params = model.parameters()
        for i, p in enumerate(params):
            p.data[...] = z[f"p{i}"]
    return model, list(meta["chars"])


def train(steps: int = 1500, seed: int = 1, d_model: int = 64,
          n_layer: int = 2, n_head: int = 4, block_size: int = 64,
          batch_size: int = 16, lr: float = 1e-3, log_every: int = 50,
          out: str | None = None, text: str | None = None,
          quiet: bool = False) -> list[float]:
    """Run training, return the per-step loss list."""
    if text is None:
        text = load_text()
    codec = CharCodec(text)
    ids = codec.encode(text)
    model = GPT(codec.vocab_size, block_size, d_model=d_model,
                n_layer=n_layer, n_head=n_head, seed=seed)
    opt = Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    n_params = sum(p.data.size for p in model.parameters())
    if not quiet:
        print(f"chars {len(text):,} | vocab {codec.vocab_size} | "
              f"params {n_params:,} | steps {steps}")
    losses: list[float] = []
    t0 = time.perf_counter()
    for step in range(1, steps + 1):
        x, y = get_batch(rng, ids, block_size, batch_size)
        loss = model.loss(x, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.data))
        if not quiet and (step % log_every == 0 or step == 1):
            dt = time.perf_counter() - t0
            print(f"step {step:5d} | loss {losses[-1]:.4f} | {dt:6.1f}s")
    if out is not None:
        save_checkpoint(out, model, codec.chars)
        if not quiet:
            print(f"saved checkpoint -> {out}")
    return losses


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--n-layer", type=int, default=2)
    ap.add_argument("--n-head", type=int, default=4)
    ap.add_argument("--block-size", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", type=str, default="gradforge_model.npz")
    args = ap.parse_args()
    train(steps=args.steps, seed=args.seed, d_model=args.d_model,
          n_layer=args.n_layer, n_head=args.n_head,
          block_size=args.block_size, batch_size=args.batch_size,
          lr=args.lr, out=args.out)


if __name__ == "__main__":
    main()
