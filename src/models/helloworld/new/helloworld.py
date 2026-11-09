"""AFTER: only model/domain code—no training infrastructure imports."""

from __future__ import annotations

import torch
from torch import nn


class ToyModel(nn.Module):
    """A traditional feed-forward model compatible with the generic trainer.

    Example:
        ``ToyModel()(torch.randn(4, 10), torch.randn(4, 5))["loss"]``
        produces the scalar loss required by the trainer contract.
    """

    def __init__(self, input_size: int = 10, hidden_size: int = 10, output_size: int = 5) -> None:
        """Build a two-layer regression network."""
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, inputs: torch.Tensor, labels: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        """Return predictions and, when labels are supplied, scalar MSE loss."""
        predictions = self.network(inputs)
        output = {"predictions": predictions}
        if labels is not None:
            output["loss"] = nn.functional.mse_loss(predictions, labels)
        return output
