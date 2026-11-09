"""AFTER: lightweight model code without training infrastructure."""

from __future__ import annotations

import torch
from torch import nn


class ToyModel(nn.Module):
    """Two-layer regression model compatible with the Generic Trainer."""

    def __init__(self, input_size: int = 10, hidden_size: int = 10, output_size: int = 5) -> None:
        """Build the same network used by the coupled example."""
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, inputs: torch.Tensor, labels: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        """Return predictions and optional scalar mean-squared error."""
        predictions = self.network(inputs)
        output = {"predictions": predictions}
        if labels is not None:
            output["loss"] = nn.functional.mse_loss(predictions, labels)
        return output
