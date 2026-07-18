"""Training entry point.
Runs training for the model and parameters specified in the command line.
Execute `python run_training.py --help` to see available parameters.

To view training logs, run `tensorboard --logdir logs` where "logs" is the name of the directory with log files.
"""

import os.path
from datetime import datetime

import numpy as np
import torch
import tyro

from src.config import write_config
from src.datasets import load_dataset, ForecastDataset, load_schema
from src.models import create_model
from src.train import train_model


def run_training(model_name: str = "linear", context_size: int = 336 * 3, prediction_horizon: int = 336,
                 model_config: dict[str, int | float | str] = {},
                 num_epochs: int = 1, log_dir_name="logs", seed: int = 42):
    """Run training for the given model

    Args:
        model_name: name of the model to train
        context_size: size of the context for the model to use
        prediction_horizon: size of the model prediction
        model_config: additional model-specific configuration parameters
        num_epochs: number of epochs to train the model
        log_dir_name: name of the directory to save log files and model checkpoints
        seed: random seed to use
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    train_df = load_dataset("train")
    schema = load_schema()

    model, is_shifted_output = create_model(model_name, context_size, prediction_horizon, schema, model_config)

    train_dataset = ForecastDataset(train_df, schema, context_size=context_size,
                                    prediction_horizon=prediction_horizon,
                                    is_shifted_output=is_shifted_output)
    print(f"Loaded training dataset of length {len(train_dataset)}")

    log_dir = os.path.join(log_dir_name, model_name, datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
    os.makedirs(log_dir, exist_ok=True)
    print(f"Writing experiment data to {log_dir}")

    write_config(log_dir, model_name, context_size, prediction_horizon, model_config)

    train_model(model, train_dataset, log_dir, num_epochs=num_epochs)


if __name__ == "__main__":
    tyro.cli(run_training)
