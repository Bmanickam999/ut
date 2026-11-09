"""Small model-owned data contract without trainer or device logic."""

from __future__ import annotations

import torch
from torch.utils.data import Dataset


class ToyDataset(Dataset[dict[str, torch.Tensor]]):
    """Deterministic synthetic data for the 128-example demonstration."""

    def __init__(self, size: int, input_size: int = 10, output_size: int = 5, seed: int = 42) -> None:
        """Generate inputs and labels from the shared fixed projection."""
        inputs_generator = torch.Generator().manual_seed(seed)
        projection_generator = torch.Generator().manual_seed(10_000)
        noise_generator = torch.Generator().manual_seed(seed + 20_000)
        self.inputs = torch.randn(size, input_size, generator=inputs_generator)
        projection = torch.randn(input_size, output_size, generator=projection_generator)
        self.labels = self.inputs @ projection + 0.05 * torch.randn(size, output_size, generator=noise_generator)

    def __len__(self) -> int:
        """Return the number of examples."""
        return len(self.inputs)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Return one trainer-compatible batch dictionary."""
        return {"inputs": self.inputs[index], "labels": self.labels[index]}
