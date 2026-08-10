"""Character-level codec and batch sampling over the bundled excerpt."""
from __future__ import annotations

from importlib import resources

import numpy as np

__all__ = ["load_text", "CharCodec", "get_batch"]


def load_text() -> str:
    """The bundled public-domain training text (Alice's Adventures in
    Wonderland, Lewis Carroll; Project Gutenberg ebook #11, boilerplate
    stripped, typography normalized to ASCII)."""
    return (resources.files("gradforge") / "data" / "alice_excerpt.txt") \
        .read_text(encoding="utf-8")


class CharCodec:
    def __init__(self, text: str):
        self.chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(self.chars)}

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def encode(self, s: str) -> np.ndarray:
        return np.array([self.stoi[c] for c in s], dtype=np.int64)

    def decode(self, ids) -> str:
        return "".join(self.chars[int(i)] for i in ids)


def get_batch(rng: np.random.Generator, ids: np.ndarray,
              block_size: int, batch_size: int):
    """Sample (x, y) where y is x shifted one character ahead."""
    starts = rng.integers(0, len(ids) - block_size - 1, size=batch_size)
    x = np.stack([ids[s:s + block_size] for s in starts])
    y = np.stack([ids[s + 1:s + 1 + block_size] for s in starts])
    return x, y
