"""Code for creating models"""
from __future__ import annotations

from typing import Any

from torch import nn

from .linear import LinearModel


def create_model(model_name: str, context_size: int, prediction_horizon: int,
                 model_config: dict[str, Any] = {}) -> tuple[nn.Module, bool]:
    """Create a model by given name and parameters.

    Args:
        model_name: name of the model to create
        context_size: number of past values to use for the input
        prediction_horizon: number of future values to predict
        model_config: additional model-specific configuration parameters
    Returns:
        A tuple (model, is_shifted_output), where is_shifted_output indicates if the output of the model
        only contains the future values, or if contains the input shifted by the prediction horizon.
    """
    # TODO add cnn and possibly other models
    if model_name == "linear":
        model = LinearModel(context_size, prediction_horizon)
        is_shifted_output = False
        return model, is_shifted_output
    else:
        raise ValueError(f"Unknown model name {model_name}")
