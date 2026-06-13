import os
from pathlib import Path
from typing import Any

import yaml

CONFIG_FILE_NAME = "config.yml"


def get_config(checkpoint_path: str | Path) -> dict[str, Any] | None:
    checkpoint_dir = os.path.dirname(checkpoint_path)
    config_path = os.path.join(checkpoint_dir, CONFIG_FILE_NAME)
    if not os.path.exists(config_path):
        return None
    return yaml.safe_load(open(config_path, "r"))


def write_config(log_dir: str, model_name: str, context_size: int, prediction_horizon: int,
                 model_config: dict[str, Any] = {}):
    with open(os.path.join(log_dir, CONFIG_FILE_NAME), "w") as config_file:
        yaml.dump(dict(model_name=model_name,
                       context_size=context_size,
                       prediction_horizon=prediction_horizon,
                       model_config=model_config), config_file)


def get_configuration_id(config: dict) -> str:
    """Identifier to use for this model configuration (for labeling plots, outputting predictions, etc.)"""
    return f"{config["model_name"]}_{config["context_size"]}_{config["prediction_horizon"]}"
