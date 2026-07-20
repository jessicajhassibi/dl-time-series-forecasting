"""Code for creating models"""
from __future__ import annotations

from torch import nn

from .config import Config
from .datasets import Schema
from .models.linear import LinearModel, LinearModelWithFeatures
from .models.tcn import TCN


def create_model(config: Config, schema: Schema) -> nn.Module:
    """Create a model for a given configuration and dataset schema.

    Args:
        config: configuration for creating the model
        schema: dataset Schema
    Returns:
        Model
    """
    if config.model_name == "linear":
        return LinearModel(config.context_size, config.prediction_horizon)
    elif config.model_name == "linear_features":
        return LinearModelWithFeatures(config.context_size, config.prediction_horizon, len(schema.feature_columns))
    elif config.model_name == "tcn":
        return TCN(config.context_size, config.prediction_horizon, len(schema.feature_columns))
    else:
        raise ValueError(f"Unknown model name {config.model_name}")


def is_shifted_output(model_name: str) -> bool:
    """Indicates if the output of the model only contains the future values,
       or if contains the input shifted by the prediction horizon"""
    if model_name in ("linear", "linear_features"):
        return False
    if model_name == "tcn":
        return True
    raise ValueError(f"Unknown model name {model_name}")
