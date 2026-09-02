import os
from datetime import datetime

import numpy as np
import tyro
from tqdm import trange, tqdm

from src.config import Config, TrainConfig, write_config
from src.datasets import ForecastDataset, load_dataset, load_schema, preprocess_dataset
from src.model_registry import create_model, is_shifted_output, get_default_config
from src.train import train_model
from src.util.util import pick_device
from src.validation import get_long_horizon_validation_metrics

DEFAULT_CONFIGS = [
    # get_default_config("linear"),
    # get_default_config("linear_features"),
    get_default_config("tcn")
]


def run_cross_validation(configs: list[Config] = DEFAULT_CONFIGS,
                         num_blocks: int = 3, block_size: int = 2500,
                         num_epochs: int = 10,
                         num_predictions: int = 2 * 336, log_dir_name="logs",
                         seeds: list[int] = [42], metric_name: str = "WAPE"):
    """
    Blocked cross-validation.
    Select blocks of data spread across the dataset and run training from scratch on each of them.
    For each block predict the values that immediately follow it using the trained model.
    This allows evaluating different model configurations
    and checking how well they predict the data they have never seen before
    (which would not be possible when training on the whole dataset).
    The downside is that each block has less data than the whole dataset, so results could be worse than during real training.

     Args:
         configs: configurations to compare
         num_blocks: number of blocks for cross-validation
         block_size: size of the training block
         num_epochs: number of training epochs per block
         num_predictions: how many future values to predict for validation
         log_dir_name: name of the log directory
         seeds: random seeds to use
         metric_name: metric to use for comparison
    """

    dataframe = load_dataset("train")
    schema = load_schema()

    preprocess_dataset(dataframe, schema)

    device = pick_device()

    if num_blocks > 1:
        block_shift = (schema.n_training_steps - num_predictions - block_size) // (num_blocks - 1)
    else:
        block_shift = 0

    result = np.ones(len(configs)) * float('inf')
    for config_idx, config in enumerate(configs):
        print(f"Running training for config {config}.")

        scores = []
        for block_idx in trange(num_blocks):
            block_start = block_idx * block_shift
            block_end = block_start + block_size
            train_df = dataframe.groupby(schema.series_id_column).nth(slice(block_start, block_end, None))

            dataset = ForecastDataset(train_df, schema, context_size=config.context_size,
                                      prediction_horizon=config.prediction_horizon,
                                      stride=8,
                                      is_shifted_output=is_shifted_output(config.model_name), device=device)
            for seed in seeds:
                train_config = TrainConfig(seed=seed, dataset_stride=8, num_epochs=num_epochs)
                train_config.set_seed()

                model = create_model(config, schema).to(device)

                run_name = (datetime.now().strftime("%Y_%m_%d_%H_%M_%S") +
                            f"_{config.context_size}_{config.prediction_horizon}_{seed}")
                log_dir = os.path.join(log_dir_name, "cross_validation", config.model_name, run_name)
                os.makedirs(log_dir, exist_ok=True)

                write_config(log_dir, config, train_config)

                train_model(model, dataset, None, train_config, log_dir, -1, device=device)

                metrics = get_long_horizon_validation_metrics(model, config, dataframe, schema, block_end,
                                                              num_predictions, device=device)
                if not metric_name in metrics:
                    raise ValueError(f"Metric {metric_name} is not found. Available metrics: {metrics.keys()}.")
                score = metrics[metric_name]
                tqdm.write(f"Block {block_idx}, seed {seed}, score {score:.4f}")
                scores.append(score)
        total_score = np.mean(scores)
        print(f"Config: {config}:\nScore: {total_score:.4f} ({scores})")
        result[config_idx] = total_score

    print(f"Cross-validation results:\n{result}.")

    if len(configs) > 1:
        config_idx = result.argmin()
        config = configs[config_idx]
        print(f"Best config: {config}.")


if __name__ == "__main__":
    tyro.cli(run_cross_validation)
