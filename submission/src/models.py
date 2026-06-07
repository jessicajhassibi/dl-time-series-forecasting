"""Code for creating models"""
from __future__ import annotations

from torch import nn

from .linear import LinearModel


def create_model(model_name: str, context_size: int, prediction_horizon: int) -> tuple[nn.Module, bool]:
    """Create model by given name and parameters.

    Args:
        model_name: name of the model to create
        context_size: number of past values to use for the input
        prediction_horizon: number of future values to predict
    Returns:
        A tuple (model, is_shifted_output), where is_shifted_output indicates if the output of the model
        only contain the future values, or if contains the input shifted by the prediction horizon.
    """
    # TODO add cnn and possibly other models
    if model_name == "linear":
        model = LinearModel(context_size, prediction_horizon)
        is_shifted_output = False
        return model, is_shifted_output
    else:
        raise ValueError(f"Unknown model name {model_name}")
