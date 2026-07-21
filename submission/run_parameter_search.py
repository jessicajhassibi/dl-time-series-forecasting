import os
from copy import replace
from datetime import datetime

import numpy as np
import tyro

from src.config import Config, TrainConfig, write_config
from src.datasets import ForecastDataset, load_dataset, load_schema, preprocess_dataset
from src.model_registry import create_model, is_shifted_output
from src.train import train_model
from src.validation import get_long_horizon_validation_metrics


def run_search(model_name: str = "linear_features", context_sizes: list[int] = [7 * 24, 2 * 7 * 24, 30 * 24],
               prediction_horizons: list[int] = [7 * 24, 2 * 7 * 24, 4 * 7 * 24],
               model_config: dict[str, int | float | str] = {},
               num_epochs: int = 1,
               n_samples_per_series: int = 2000, n_predictions: int = 2 * 7 * 24, log_dir_name="logs",
               seeds: list[int] = [0, 42, 239], metric_name: str = "wape"):
    """
    Run training on a smaller block of the dataset for several random seeds,
    predict a block of values into the future, compute the WAPE metric,
     and select context size and prediction horizon values with the lowest score.

     Args:
         model_name: name of the model to run
         context_sizes: context size values to check
         prediction_horizons: prediction horizon values to check
         model_config: additional model-specific parameters
         num_epochs: number of training epochs
         n_samples_per_series: number of samples per series to use for training
         n_predictions: how many future values to predict for validation
         log_dir_name: name of the log directory
         seeds: random seeds to use
         metric_name: metric to use for comparison
    """

    print(f"Running hyperparameter search for the {model_name} model.")

    dataframe = load_dataset("train")
    schema = load_schema()

    preprocess_dataset(dataframe, schema)

    result = np.ones((len(context_sizes), len(prediction_horizons))) * float('inf')
    for ci, context_size in enumerate(context_sizes):
        for pi, prediction_horizon in enumerate(prediction_horizons):
            if prediction_horizon > context_size:
                continue

            n_train = n_samples_per_series + context_size + prediction_horizon - 1

            print(f"Running training for context size {context_size} and prediction horizon {prediction_horizon}"
                  f" with series length {n_train} and prediction size {n_predictions}.")

            train_df = schema.get_series_groups(dataframe).head(n_train).copy()
            train_schema = replace(schema, n_training_steps=n_train)

            dataset = ForecastDataset(train_df, train_schema, context_size=context_size,
                                      prediction_horizon=prediction_horizon,
                                      is_shifted_output=is_shifted_output(model_name))
            config = Config(model_name, context_size, prediction_horizon, model_config)

            total_score = 0.0
            for seed in seeds:
                train_config = TrainConfig(seed=seed)
                train_config.set_seed()

                model = create_model(config, schema)

                run_name = datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + f"_{context_size}_{prediction_horizon}_{seed}"
                log_dir = os.path.join(log_dir_name, model_name, "parameter_search", run_name)
                os.makedirs(log_dir, exist_ok=True)

                write_config(log_dir, config, train_config)

                train_model(model, dataset, None, log_dir, num_epochs, 128, -1)

                metrics = get_long_horizon_validation_metrics(model, context_size, prediction_horizon, dataframe,
                                                              schema, n_train, n_predictions)
                if not metric_name in metrics:
                    raise ValueError(f"Metric {metric_name} is not found. Available metrics: {metrics.keys()}.")
                total_score = total_score + metrics[metric_name]
            total_score = total_score / len(seeds)
            print(f"Score for context size {context_size} and prediction horizon {prediction_horizon}: {total_score}")
            result[ci, pi] = total_score

    print(f"Metric values:\n{result}.")

    ci, pi = np.unravel_index(result.argmin(), result.shape)
    context_size = context_sizes[ci]
    prediction_horizon = prediction_horizons[pi]
    print(f"Best context size and prediction horizon: {context_size}, {prediction_horizon}.")


if __name__ == "__main__":
    tyro.cli(run_search)
