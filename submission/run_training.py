"""Training entry point.
Runs training for the model and parameters specified in the command line.
Execute `python run_training.py --help` to see available parameters.

To view training logs, run `tensorboard --logdir logs` where "logs" is the name of the directory with log files.
"""
import os
from datetime import datetime
from pathlib import Path

import torch
import tyro

from src.config import write_config, get_config_if_exists, Config
from src.datasets import ForecastDataset
from src.datasets import load_dataset, load_schema
from src.models import create_model, is_shifted_output
from src.train import train_model
from src.config import TrainConfig


def run_training(model_name: str = "linear_features", context_size: int = 336 * 3, prediction_horizon: int = 336,
                 model_config: dict[str, int | float | str] = {},
                 num_epochs: int = 1, log_dir_name="logs", seed: int = 42,
                 checkpoint: Path | None = None):
    """Run training for the given model

    Args:
        model_name: name of the model to train
        context_size: size of the context for the model to use
        prediction_horizon: size of the model prediction
        model_config: additional model-specific configuration parameters
        num_epochs: number of epochs to train the model
        log_dir_name: name of the directory to save log files and model checkpoints
        seed: random seed to use
        checkpoint: path to checkpoint to resume training from.
                    If a checkpoint is given, its configuration overrides the parameters provided in the command line.
    """
    config = Config(model_name, context_size, prediction_horizon, model_config)
    train_config = TrainConfig(seed=seed)

    if checkpoint is not None:
        checkpoint_config = get_config_if_exists(checkpoint)

        if checkpoint_config is not None:
            print(f"Using configuration found for checkpoint {config}")

            config = checkpoint_config[0]
            train_config = checkpoint_config[1]

    print(f"Running training for the {model_name} model.")

    train_config.set_seed()

    dataframe = load_dataset("train")
    schema = load_schema()

    model = create_model(config, schema)

    dataset = ForecastDataset(dataframe, schema, context_size=context_size,
                              prediction_horizon=prediction_horizon,
                              is_shifted_output=is_shifted_output(model_name))
    print(f"Loaded training dataset of length {len(dataset)}")

    run_dir = datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + f"_{context_size}_{prediction_horizon}_{seed}"
    log_dir = os.path.join(log_dir_name, model_name, run_dir)
    os.makedirs(log_dir, exist_ok=True)
    print(f"Writing experiment data to {log_dir}")

    write_config(log_dir, config, train_config)

    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [0.9, 0.1])
    train_model(model, train_dataset, val_dataset, log_dir, num_epochs=num_epochs,
                checkpoint=checkpoint)


if __name__ == "__main__":
    tyro.cli(run_training)
