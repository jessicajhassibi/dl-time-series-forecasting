"""Code for creating models"""
from __future__ import annotations

from pathlib import Path

from torch import nn

from .config import Config, get_config_if_exists
from .datasets import Schema
from .models.linear import LinearModel, LinearModelWithFeatures
from .models.tcn_deep import TCNDeep

MODELS = ("linear", "linear_features", "tcn")


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
        return TCNDeep(config.prediction_horizon, len(schema.feature_columns), **config.model_config)
    else:
        raise ValueError(f"Unknown model name {config.model_name}")


def is_shifted_output(model_name: str) -> bool:
    """Indicates if the output of the model only contains the future values,
       or if contains the input shifted by the prediction horizon"""
    if model_name in ("linear", "linear_features"):
        return False
    if model_name == "tcn":
        return False
    raise ValueError(f"Unknown model name {model_name}")


def get_default_config(model_name: str = "tcn") -> Config:
    if model_name in ("linear", "linear_features"):
        return Config(model_name=model_name, context_size=1024, prediction_horizon=2 * 336)
    if model_name == "tcn":
        kernel_size = 3
        levels = 8
        context_size = get_tcn_receptive_field(kernel_size, levels)
        return Config(model_name=model_name, context_size=context_size, prediction_horizon=2 * 336,
                      model_config=dict(hidden=64,
                                        levels=levels,
                                        kernel_size=kernel_size,
                                        dropout=0.1))
    raise ValueError(f"Unknown model name {model_name}")


def get_tcn_receptive_field(kernel_size: int, levels: int) -> int:
    return 1 + 2 * (kernel_size - 1) * (2 ** levels - 1)


def get_config(checkpoint_path: str | Path) -> Config:
    """Load the configuration file for the given checkpoint.
    Configuration file is expected to be located in the same folder as the checkpoint."""
    config = get_config_if_exists(checkpoint_path)
    if config is not None:
        return config[0]
    # TODO read config from checkpoint
    print(f"Could not find config file for checkpoint {checkpoint_path}, using default config")
    return get_default_config()
