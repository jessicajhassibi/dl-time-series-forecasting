"""Code for creating models"""
from __future__ import annotations

from torch import nn

from .config import Config
from .datasets import Schema
from .models.linear import LinearModel, LinearModelWithFeatures


def create_model(config: Config, schema: Schema) -> nn.Module:
    """Create a model for a given configuration and dataset schema.

    Args:
        config: configuration for creating the model
        schema: dataset Schema
    Returns:
        Model
    """
    # TODO add cnn and possibly other models
    if config.model_name == "linear":
        return LinearModel(config.context_size, config.prediction_horizon)
    elif config.model_name == "linear_features":
        return LinearModelWithFeatures(config.context_size, config.prediction_horizon, len(schema.feature_columns))
    else:
        raise ValueError(f"Unknown model name {config.model_name}")


def is_shifted_output(model_name: str) -> bool:
    """Indicates if the output of the model only contains the future values,
       or if contains the input shifted by the prediction horizon"""
    # TODO add cnn and possibly other models
    if model_name in ("linear", "linear_features"):
        return False
    raise ValueError(f"Unknown model name {model_name}")
