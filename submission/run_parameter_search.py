import os
from datetime import datetime

import numpy as np
import torch
import tyro
from torch.utils.data import Subset

from src.config import write_config
from src.datasets import ForecastDataset
from src.datasets import load_dataset, load_schema
from src.models import create_model, is_shifted_output
from src.train import train_model, get_validation_metrics


def run_search(model_name: str = "linear", context_sizes: list[int] = [24, 7 * 24, 336],
               prediction_horizons: list[int] = [24, 3 * 24, 7 * 24],
               model_config: dict[str, int | float | str] = {},
               num_epochs: int = 1, subset_size: int = 10000, log_dir_name="logs",
               seeds: list[int] = [0, 42, 239], metric_name: str = "wape"):
    """
    Run training on a smaller subset of the dataset for several random seeds, compute the specified metric,
     and select context size and prediction horizon values with the lowest score.

     Args:
         model_name: name of the model to run
         context_sizes: context size values to check
         prediction_horizons: prediction horizon values to check
         model_config: additional model-specific parameters
         num_epochs: number of training epochs
         subset_size: size of the subset to train on
         log_dir_name: name of the log directory
         seeds: random seeds to use
         metric_name: metric to use for comparison
    """

    print(f"Running hyperparameter search for the {model_name} model.")

    dataframe = load_dataset("train")
    schema = load_schema()

    result = np.ones((len(context_sizes), len(prediction_horizons))) * float('inf')
    for ci, context_size in enumerate(context_sizes):
        for pi, prediction_horizon in enumerate(prediction_horizons):
            if prediction_horizon > context_size:
                continue

            print(f"Running training for context size {context_size} and prediction horizon {prediction_horizon}.")
            dataset = ForecastDataset(dataframe, schema, context_size=context_size,
                                      prediction_horizon=prediction_horizon,
                                      is_shifted_output=is_shifted_output(model_name))
            value = 0.0
            for seed in seeds:
                np.random.seed(seed)
                torch.manual_seed(seed)

                model = create_model(model_name, context_size, prediction_horizon, schema, model_config)

                run_name = datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + f"_{context_size}_{prediction_horizon}_{seed}"
                log_dir = os.path.join(log_dir_name, model_name, "parameter_search", run_name)
                os.makedirs(log_dir, exist_ok=True)

                write_config(log_dir, model_name, context_size, prediction_horizon,
                             model_config=model_config,
                             train_config=dict(seed=seed))

                random_indices = torch.randperm(len(dataset))[:min(subset_size, len(dataset))].tolist()
                random_subset = Subset(dataset, random_indices)
                train_dataset, val_dataset = torch.utils.data.random_split(random_subset, [0.9, 0.1])
                train_model(model, train_dataset, None, log_dir, num_epochs, 128, -1)
                metrics = get_validation_metrics(model, val_dataset)
                if not metric_name in metrics:
                    raise ValueError(f"Metric {metric_name} is not found. Available metrics: {metrics.keys()}.")
                value += metrics[metric_name]

            result[ci, pi] = value / len(seeds)

    print(f"Metric values:\n{result}.")

    ci, pi = np.unravel_index(result.argmin(), result.shape)
    context_size = context_sizes[ci]
    prediction_horizon = prediction_horizons[pi]
    print(f"Best context size and prediction horizon: {context_size}, {prediction_horizon}.")


if __name__ == "__main__":
    tyro.cli(run_search)
