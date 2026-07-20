"""Temporal Convolutional Network implementation"""

from __future__ import annotations

import torch


class TCN(torch.nn.Module):
    """Temporal Convolutional Network (TCN)"""

    def __init__(self, context_size: int, prediction_horizon: int, num_features: int) -> None:
        """Create the placeholder one-parameter model."""
        super().__init__()
        self.bias = torch.nn.Parameter(torch.zeros(()))

    def forward(self, x: torch.Tensor, x_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Return the input shifted by the learned scalar bias."""
        # TODO implement tcn
        return x + self.bias, x_features + self.bias
