"""Code for creating models"""
from __future__ import annotations

from typing import Any

from torch import nn

from .datasets import Schema
from .linear import LinearModel, LinearModelWithFeatures


def create_model(model_name: str, context_size: int, prediction_horizon: int,
                 schema: Schema, model_config: dict[str, Any] = {}) -> nn.Module:
    """Create a model by given name and parameters.

    Args:
        model_name: name of the model to create
        context_size: number of past values to use for the input
        prediction_horizon: number of future values to predict
        schema: dataset Schema
        model_config: additional model-specific configuration parameters
    Returns:
        Model
    """
    # TODO add cnn and possibly other models
    if model_name == "linear":
        return LinearModel(context_size, prediction_horizon)
    elif model_name == "linear_features":
        return LinearModelWithFeatures(context_size, prediction_horizon, len(schema.feature_columns))
    else:
        raise ValueError(f"Unknown model name {model_name}")


def is_shifted_output(model_name: str) -> bool:
    """Indicates if the output of the model only contains the future values,
       or if contains the input shifted by the prediction horizon"""
    # TODO add cnn and possibly other models
    if model_name in ("linear", "linear_features"):
        return False
    raise ValueError(f"Unknown model name {model_name}")
