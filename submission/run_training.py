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

from src.config import TrainConfig, write_config, get_config_if_exists, Config
from src.datasets import ForecastDataset, load_dataset, load_schema
from src.model_registry import create_model, is_shifted_output, get_default_config
from src.train import train_model
from src.util.util import pick_device


def run_training(config: Config = get_default_config("tcn"),
                 train_config: TrainConfig = TrainConfig(),
                 log_dir_name="logs", checkpoint: Path | None = None):
    """Run training for the given model

    Args:
        config: model parameters
        train_config: training parameters
        log_dir_name: name of the directory to save log files and model checkpoints
        checkpoint: path to checkpoint to resume training from.
                    If a checkpoint is given, its configuration overrides the parameters provided in the command line.
    """
    if checkpoint is not None:
        checkpoint_config = get_config_if_exists(checkpoint)

        if checkpoint_config is not None:
            print(f"Using configuration found for checkpoint {checkpoint_config}")

            config = checkpoint_config[0]
            train_config = checkpoint_config[1]

    print(f"Running training for the {config.model_name} model.")

    train_config.set_seed()

    device = pick_device()
    print(f"Using device: {device}")

    dataframe = load_dataset("train")
    schema = load_schema()

    model = create_model(config, schema).to(device=device)

    dataset = ForecastDataset(dataframe, schema, context_size=config.context_size,
                              prediction_horizon=config.prediction_horizon,
                              stride=train_config.dataset_stride,
                              is_shifted_output=is_shifted_output(config.model_name),
                              device=device)
    print(f"Loaded training dataset of length {len(dataset)}")

    run_dir = (datetime.now().strftime("%Y_%m_%d_%H_%M_%S") +
               f"_{config.context_size}_{config.prediction_horizon}_{train_config.seed}")
    log_dir = os.path.join(log_dir_name, config.model_name, run_dir)
    os.makedirs(log_dir, exist_ok=True)
    print(f"Writing experiment data to {log_dir}")

    write_config(log_dir, config, train_config)

    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [0.9, 0.1],
                                                               generator=train_config.create_random_generator())
    train_model(model, train_dataset, val_dataset, train_config, log_dir,
                checkpoint=checkpoint, device=device)


if __name__ == "__main__":
    # TODO refactor model config, maybe switch to multiple commands (one command for each model)
    tyro.cli(run_training, use_underscores=True)
