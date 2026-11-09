from __future__ import annotations

from typing import Any

from .data import HousingDataset, load_rows
from .model import HousingTransformer, HousingTransformerConfig
from .tokenizer import StableHashTokenizer


def build_model(config: dict[str, Any]) -> HousingTransformer:
    fields = HousingTransformerConfig.__dataclass_fields__
    return HousingTransformer(HousingTransformerConfig(**{k: v for k, v in config.items() if k in fields}))


def build_datasets(config: dict[str, Any], seed: int = 42) -> tuple[HousingDataset, HousingDataset]:
    count = int(config["num_examples"])
    rows = load_rows(config["source"], count, seed)
    if len(rows) < 2:
        raise ValueError("at least two examples are required")
    split = max(1, min(len(rows) - 1, round(len(rows) * (1 - float(config["validation_fraction"])))))
    tok_cfg = config["tokenizer"]
    tokenizer = StableHashTokenizer(int(tok_cfg["vocab_size"]), int(tok_cfg["max_length"]))
    args = (config["feature_fields"], config["target_field"], tokenizer)
    return HousingDataset(rows[:split], *args), HousingDataset(rows[split:], *args)
