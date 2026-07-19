"""Helper functions for reading and writing model and training configuration.

Used for keeping track of hyperparameters to run trained models and to analyze and reproduce results."""
import os
from pathlib import Path
from typing import Any

import yaml

CONFIG_FILE_NAME = "config.yml"


def get_config_if_exists(checkpoint_path: str | Path) -> dict[str, Any] | None:
    """Load the configuration file for the given checkpoint or return None if not found.
    Configuration file is expected to be located in the same folder as the checkpoint."""
    checkpoint_dir = os.path.dirname(checkpoint_path)
    config_path = os.path.join(checkpoint_dir, CONFIG_FILE_NAME)
    if not os.path.exists(config_path):
        return None
    return yaml.safe_load(open(config_path, "r"))


def get_config(checkpoint_path: str | Path) -> dict[str, Any]:
    """Load the configuration file for the given checkpoint.
    Configuration file is expected to be located in the same folder as the checkpoint."""
    config = get_config_if_exists(checkpoint_path)
    if config is not None:
        return config
    # TODO use default config if config file is not found
    # Default config should be the one we submit
    raise ValueError(f"Could not find config file for checkpoint {checkpoint_path}")


def write_config(log_dir: str, model_name: str, context_size: int, prediction_horizon: int,
                 model_config: dict[str, Any] = {}, train_config: dict[str, Any] = {}):
    """Dump model and training configuration to file."""
    with open(os.path.join(log_dir, CONFIG_FILE_NAME), "w") as config_file:
        yaml.dump(dict(model_name=model_name,
                       context_size=context_size,
                       prediction_horizon=prediction_horizon,
                       model_config=model_config,
                       train_config=train_config), config_file)


def get_configuration_id(config: dict) -> str:
    """Identifier to use for this model configuration (for labeling plots, outputting predictions, etc.)"""
    return f"{config["model_name"]}_{config["context_size"]}_{config["prediction_horizon"]}"
