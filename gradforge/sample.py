"""Generate text from a trained checkpoint.

    python -m gradforge.sample --checkpoint gradforge_model.npz --tokens 400
"""
from __future__ import annotations

import argparse

import numpy as np

from .functional import softmax
from .tensor import Tensor
from .train import load_checkpoint

__all__ = ["generate"]


def generate(model, chars: list[str], prompt: str = "\n", tokens: int = 400,
             temperature: float = 0.8, seed: int = 0) -> str:
    stoi = {c: i for i, c in enumerate(chars)}
    rng = np.random.default_rng(seed)
    ctx = [stoi[c] for c in prompt]
    out = list(ctx)
    for _ in range(tokens):
        window = np.array([out[-model.block_size:]], dtype=np.int64)
        logits = model.forward(window)          # (1, t, vocab)
        last = Tensor(logits.data[0, -1] / temperature)
        probs = softmax(last, axis=-1).data
        out.append(int(rng.choice(len(chars), p=probs)))
    return "".join(chars[i] for i in out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=str, default="gradforge_model.npz")
    ap.add_argument("--prompt", type=str, default="\n")
    ap.add_argument("--tokens", type=int, default=400)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    model, chars = load_checkpoint(args.checkpoint)
    print(generate(model, chars, prompt=args.prompt, tokens=args.tokens,
                   temperature=args.temperature, seed=args.seed))


if __name__ == "__main__":
    main()
