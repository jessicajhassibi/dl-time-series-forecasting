"""Training entry point.
Runs training for the model and parameters specified in the command line.
Execute `python run_training.py --help` to see available commands
and `python run_training.py <COMMAND> --help` to see parameters for a command.

To view training logs, run `tensorboard --logdir logs` where "logs" is the name of the directory with log files.
"""

import os
from datetime import datetime
from pathlib import Path

import torch
import tyro

from src.config import TrainConfig, write_config, get_config_if_exists, Config
from src.datasets import ForecastDataset, load_dataset, load_schema
from src.model_registry import (
    create_model,
    is_shifted_output,
    get_default_config,
    get_tcn_receptive_field,
)
from src.train import train_model
from src.util.util import pick_device


def run_linear(
    context_size: int = get_default_config("linear_features").context_size,
    prediction_horizon: int = get_default_config("linear_features").prediction_horizon,
    train_config: TrainConfig = TrainConfig(),
    log_dir_name="logs",
):
    """Train the linear model

    Args:
        context_size: number of past values to use
        prediction_horizon: number of values in the future to predict
        train_config: training parameters
        log_dir_name: name of the directory to save log files and model checkpoints
    """
    run_training(
        Config(
            model_name="linear_features",
            context_size=context_size,
            prediction_horizon=prediction_horizon,
        ),
        train_config=train_config,
        log_dir_name=log_dir_name,
        checkpoint=None,
    )


def run_tcn(
    prediction_horizon: int = get_default_config("tcn").prediction_horizon,
    hidden=get_default_config("tcn").model_config["hidden"],
    levels=get_default_config("tcn").model_config["levels"],
    kernel_size=get_default_config("tcn").model_config["kernel_size"],
    dropout=get_default_config("tcn").model_config["dropout"],
    train_config: TrainConfig = TrainConfig(),
    log_dir_name="logs",
):
    """Train the temporal convolution model

    Args:
        prediction_horizon: number of values in the future to predict
        hidden: hidden layer dimension
        levels: number of residual blocks
        kernel_size: size of the convolution kernel
        dropout: dropout rate
        train_config: training parameters
        log_dir_name: name of the directory to save log files and model checkpoints
    """
    run_training(
        Config(
            model_name="tcn",
            context_size=get_tcn_receptive_field(kernel_size, levels),
            prediction_horizon=prediction_horizon,
            model_config=dict(
                hidden=hidden, levels=levels, dropout=dropout, kernel_size=kernel_size
            ),
        ),
        train_config=train_config,
        log_dir_name=log_dir_name,
        checkpoint=None,
    )


def run_lstm(
    context_size: int = get_default_config("lstm").context_size,
    prediction_horizon: int = get_default_config("lstm").prediction_horizon,
    train_config: TrainConfig = TrainConfig(),
    log_dir_name="logs",
):
    run_training(
        Config(
            model_name="lstm",
            context_size=context_size,
            prediction_horizon=prediction_horizon,
            model_config=dict(
                
            )
        ),
        train_config=train_config,
        log_dir_name=log_dir_name,
        checkpoint=None,
    )


def run_checkpoint(checkpoint: Path, log_dir_name: str):
    """Continue training from the checkpoint

    Args:
        checkpoint: checkpoint path
        log_dir_name: name of the directory to save log files and model checkpoints
    """
    checkpoint_config = get_config_if_exists(checkpoint)
    if checkpoint_config is None:
        raise ValueError(f"Could not find config for checkpoint {checkpoint}")

    config = checkpoint_config[0]
    train_config = checkpoint_config[1]

    print(f"Using configuration:\n{config}\n{train_config}")

    run_training(config, train_config, log_dir_name, checkpoint)


def run_training(
    config: Config,
    train_config: TrainConfig,
    log_dir_name: str,
    checkpoint: Path | None,
):
    print(f"Running training for the {config.model_name} model.")

    train_config.set_seed()

    device = pick_device()
    print(f"Using device: {device}")

    dataframe = load_dataset("train")
    schema = load_schema()

    model = create_model(config, schema).to(device=device)

    dataset = ForecastDataset(
        dataframe,
        schema,
        context_size=config.context_size,
        prediction_horizon=config.prediction_horizon,
        stride=train_config.dataset_stride,
        is_shifted_output=is_shifted_output(config.model_name),
        device=device,
    )
    print(f"Loaded training dataset of length {len(dataset)}")

    run_dir = (
        datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        + f"_{config.context_size}_{config.prediction_horizon}_{train_config.seed}"
    )
    log_dir = os.path.join(log_dir_name, config.model_name, run_dir)
    os.makedirs(log_dir, exist_ok=True)
    print(f"Writing experiment data to {log_dir}")

    write_config(log_dir, config, train_config)

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [0.9, 0.1], generator=train_config.create_random_generator()
    )
    train_model(
        model,
        train_dataset,
        val_dataset,
        train_config,
        log_dir,
        checkpoint=checkpoint,
        device=device,
    )


if __name__ == "__main__":
    tyro.extras.subcommand_cli_from_dict(
        dict(linear=run_linear, tcn=run_tcn, checkpoint=run_checkpoint, lstm=run_lstm),
        use_underscores=True,
        description="Run training for a chosen model type or checkpoint",
    )
