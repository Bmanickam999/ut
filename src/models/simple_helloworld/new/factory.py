"""Configuration adapters for the lightweight model and datasets."""

from __future__ import annotations

from typing import Any

from .data import ToyDataset
from .helloworld import ToyModel


def build_model(config: dict[str, Any]) -> ToyModel:
    """Build the lightweight model from YAML values."""
    return ToyModel(
        input_size=int(config.get("input_size", 10)),
        hidden_size=int(config.get("hidden_size", 10)),
        output_size=int(config.get("output_size", 5)),
    )


def build_datasets(config: dict[str, Any], seed: int = 42) -> tuple[ToyDataset, ToyDataset]:
    """Split the configured example count into deterministic train and validation sets."""
    size = int(config["num_examples"])
    validation_size = max(1, round(size * float(config.get("validation_fraction", 0.2))))
    dimensions = {"input_size": int(config.get("input_size", 10)), "output_size": int(config.get("output_size", 5))}
    return ToyDataset(size - validation_size, seed=seed, **dimensions), ToyDataset(validation_size, seed=seed + 1, **dimensions)
