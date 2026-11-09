"""Configurable public/synthetic housing data for the demo model.

Example:
    ``load_rows({"type": "synthetic"}, 100, 42)`` creates deterministic
    non-banking examples without external data access.
"""

from __future__ import annotations

import csv
import math
import random
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from .tokenizer import StableHashTokenizer


def synthetic_rows(count: int, seed: int) -> list[dict[str, Any]]:
    """Generate deterministic Census-style housing records."""
    rng = random.Random(seed)
    states = ["CA", "TX", "NY", "NC", "OH", "WA"]
    rows = []
    for index in range(count):
        state = rng.choice(states)
        bedrooms, bathrooms = rng.randint(1, 6), rng.randint(1, 4)
        area, income = rng.randint(550, 4200), rng.randint(30_000, 180_000)
        population, year = rng.randint(8_000, 1_500_000), rng.randint(1920, 2024)
        state_adjustment = {"CA": 190_000, "NY": 130_000, "WA": 100_000}.get(state, 20_000)
        value = 35_000 + area * 145 + income * 0.9 + bathrooms * 12_000 + state_adjustment + rng.gauss(0, 25_000)
        rows.append({"state": state, "county": f"public-county-{index % 50}", "bedrooms": bedrooms,
                     "bathrooms": bathrooms, "living_area_sqft": area, "median_income": income,
                     "population": population, "year_built": year, "home_value": max(25_000, round(value, 2))})
    return rows


def load_rows(source: dict[str, Any], count: int, seed: int) -> list[dict[str, Any]]:
    """Load rows from synthetic data, a local CSV, or a cached HTTPS CSV."""
    kind = source.get("type", "synthetic")
    if kind == "synthetic":
        return synthetic_rows(count, seed)
    location = source.get("path")
    if not location:
        raise ValueError(f"data.source.path is required for {kind}")
    if kind == "csv":
        path = Path(location).expanduser().resolve()
    elif kind == "url_csv":
        parsed = urllib.parse.urlparse(location)
        if parsed.scheme != "https":
            raise ValueError("url_csv accepts HTTPS URLs only")
        cache = Path(source.get("cache_dir", ".cache/public-data"))
        cache.mkdir(parents=True, exist_ok=True)
        path = cache / (hashlib_name(location) + ".csv")
        if not path.exists():
            urllib.request.urlretrieve(location, path)
    else:
        raise ValueError(f"unsupported data source type: {kind}")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))[:count]


def hashlib_name(text: str) -> str:
    """Return a short stable SHA-256 name suitable for cache files."""
    import hashlib
    return hashlib.sha256(text.encode()).hexdigest()[:16]


class HousingDataset(Dataset[dict[str, torch.Tensor]]):
    """Convert housing rows into tokenized trainer-compatible examples.

    Example:
        ``HousingDataset(rows, fields, "home_value", tokenizer)[0]`` returns
        token IDs, attention mask, language labels, and regression target.
    """

    def __init__(self, rows: list[dict[str, Any]], feature_fields: list[str], target_field: str, tokenizer: StableHashTokenizer) -> None:
        """Validate the schema and retain rows and tokenization settings."""
        required = set(feature_fields) | {target_field}
        if rows and (missing := required - set(rows[0])):
            raise ValueError(f"dataset is missing columns: {sorted(missing)}")
        self.rows, self.fields, self.target, self.tokenizer = rows, feature_fields, target_field, tokenizer

    def __len__(self) -> int:
        """Return the number of housing records."""
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Tokenize one row and return language and regression targets."""
        row = self.rows[index]
        target = float(row[self.target])
        if not math.isfinite(target):
            raise ValueError(f"non-finite target at row {index}")
        prompt = "Estimate median home value. " + "; ".join(f"{name}={row[name]}" for name in self.fields)
        ids, mask = self.tokenizer.encode(prompt)
        input_ids = torch.tensor(ids, dtype=torch.long)
        attention_mask = torch.tensor(mask, dtype=torch.long)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels, "targets": torch.tensor(target)}
