"""Deterministic dependency-free tokenizer for the housing demonstration.

Example:
    ``ids, mask = StableHashTokenizer(1024, 16).encode("state=CA")``
"""

from __future__ import annotations

import hashlib
import re


class StableHashTokenizer:
    """Dependency-free, deterministic demo tokenizer with fixed special IDs."""

    PAD, BOS, EOS, UNK = 0, 1, 2, 3
    _pattern = re.compile(r"[A-Za-z]+|\d+(?:\.\d+)?|[^\w\s]")

    def __init__(self, vocab_size: int, max_length: int) -> None:
        """Configure the fixed vocabulary range and padded sequence length."""
        if vocab_size < 128:
            raise ValueError("vocab_size must be at least 128")
        self.vocab_size, self.max_length = vocab_size, max_length

    def _id(self, token: str) -> int:
        """Map a token to a stable non-special vocabulary ID."""
        digest = hashlib.blake2b(token.lower().encode(), digest_size=8).digest()
        return 4 + int.from_bytes(digest, "big") % (self.vocab_size - 4)

    def encode(self, text: str) -> tuple[list[int], list[int]]:
        """Tokenize text and return fixed-length IDs and attention mask."""
        ids = [self.BOS] + [self._id(t) for t in self._pattern.findall(text)]
        ids = (ids[: self.max_length - 1] + [self.EOS])[: self.max_length]
        mask = [1] * len(ids)
        padding = self.max_length - len(ids)
        return ids + [self.PAD] * padding, mask + [0] * padding
