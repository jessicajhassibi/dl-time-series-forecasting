"""Helper functions for reading and writing model and training configuration.

Used for keeping track of hyperparameters to run trained models and to analyze and reproduce results."""
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


@dataclass
class Config:
    """Model configuration"""
    model_name: str
    """Name of the model (e.g. linear, tcn, etc.)"""
    context_size: int
    """Input size of the model"""
    prediction_horizon: int
    """How many values in the future the model predicts"""
    model_config: dict[str, Any] = field(default_factory=dict)
    """Model-specific parameters"""

    def get_id(self) -> str:
        """Identifier to use for this model configuration (for labeling plots, outputting predictions, etc.)"""
        return f"{self.model_name}_{self.context_size}_{self.prediction_horizon}"


@dataclass
class TrainConfig:
    """Training configuration"""

    dataset_stride: int = 1
    """Shift between consecutive dataset samples"""
    num_epochs: int = 4
    """Number of training epochs"""
    seed: int = 42
    """Random seed"""
    batch_size: int = 256
    """Batch size"""
    lr: float = 1e-3
    """Learning rate"""
    weight_decay: float = 1e-2
    """Weight decay for the optimizer"""
    huber_delta: float = 1.0
    """Delta value for the Huber loss"""

    def set_seed(self):
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

    def create_random_generator(self) -> torch.Generator:
        return torch.Generator().manual_seed(self.seed)


CONFIG_FILE_NAME = "config.yml"


def get_config_if_exists(checkpoint_path: str | Path) -> tuple[Config, TrainConfig] | None:
    """Load the configuration file for the given checkpoint or return None if not found.
    Configuration file is expected to be located in the same folder as the checkpoint."""
    checkpoint_dir = os.path.dirname(checkpoint_path)
    config_path = os.path.join(checkpoint_dir, CONFIG_FILE_NAME)
    if not os.path.exists(config_path):
        return None
    config_dict = yaml.safe_load(open(config_path, "r"))
    return Config(**config_dict["config"]), TrainConfig(**config_dict["train_config"])


def write_config(log_dir: str, config: Config, train_config: TrainConfig):
    """Dump model and training configuration to file."""
    with open(os.path.join(log_dir, CONFIG_FILE_NAME), "w") as config_file:
        yaml.dump(dict(config=asdict(config),
                       train_config=asdict(train_config)), config_file)
