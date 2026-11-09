"""Small configuration adapters at the model/trainer boundary."""

from __future__ import annotations

from typing import Any

from .data import ToyDataset
from .helloworld import ToyModel


def build_model(config: dict[str, Any]) -> ToyModel:
    return ToyModel(
        input_size=int(config.get("input_size", 10)),
        hidden_size=int(config.get("hidden_size", 10)),
        output_size=int(config.get("output_size", 5)),
    )


def build_datasets(config: dict[str, Any], seed: int = 42) -> tuple[ToyDataset, ToyDataset]:
    size = int(config["num_examples"])
    validation_fraction = float(config.get("validation_fraction", 0.2))
    validation_size = max(1, round(size * validation_fraction))
    train_size = size - validation_size
    dimensions = {"input_size": int(config.get("input_size", 10)), "output_size": int(config.get("output_size", 5))}
    return ToyDataset(train_size, seed=seed, **dimensions), ToyDataset(validation_size, seed=seed + 1, **dimensions)
