import os.path
from datetime import datetime

import tyro

from src.config import write_config
from src.datasets import load_dataset, ForecastDataset, load_schema
from src.models import create_model
from src.train import train_model


def run_training(model_name: str = "linear", context_size: int = 336 * 3, prediction_horizon: int = 336,
                 model_config: dict[str, int | float | str] = {},
                 num_epochs: int = 1):
    """Run training for the given model

    Args:
        model_name: name of the model to train
        context_size: size of the context for the model to use
        prediction_horizon: size of the model prediction
        model_config: additional model-specific configuration parameters
        num_epochs: number of epochs to train the model
    """
    model, is_shifted_output = create_model(model_name, context_size, prediction_horizon, model_config)
    train_df = load_dataset("train")
    schema = load_schema()
    train_dataset = ForecastDataset(train_df, schema, context_size=context_size,
                                    prediction_horizon=prediction_horizon,
                                    is_shifted_output=is_shifted_output)
    print(f"Loaded training dataset of length {len(train_dataset)}")

    log_dir = os.path.join("logs", model_name, datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
    os.makedirs(log_dir, exist_ok=True)
    print(f"Writing experiment data to {log_dir}")

    write_config(log_dir, model_name, context_size, prediction_horizon, model_config)

    train_model(model, train_dataset, log_dir, num_epochs=num_epochs)


if __name__ == "__main__":
    tyro.cli(run_training)
