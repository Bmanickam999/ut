"""Domain-owned example data contract; no trainer or device logic."""

from __future__ import annotations

import torch
from torch.utils.data import Dataset


class ToyDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, size: int, input_size: int = 10, output_size: int = 5, seed: int = 42) -> None:
        generator = torch.Generator().manual_seed(seed)
        self.inputs = torch.randn(size, input_size, generator=generator)
        projection = torch.randn(input_size, output_size, generator=generator)
        self.labels = self.inputs @ projection + 0.05 * torch.randn(size, output_size, generator=generator)

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {"inputs": self.inputs[index], "labels": self.labels[index]}
