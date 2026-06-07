import os.path
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import yaml

from plot import plot_prediction_comparison
from src.baselines import make_all_baselines
from src.datasets import load_dataset, load_metadata, Schema
from src.linear import LinearModel
from src.plot import plot_series


def find_last_checkpoint(parent_directory: str) -> str | None:
    if not os.path.exists(parent_directory):
        return None
    parent_path = Path(parent_directory)
    checkpoints = list(parent_path.glob("**/*.pt"))
    if not checkpoints:
        return None
    return max(checkpoints, key=os.path.getmtime)


def get_config(checkpoint_path: str) -> dict[str, Any] | None:
    checkpoint_dir = os.path.dirname(checkpoint_path)
    config_path = os.path.join(checkpoint_dir, "config.yml")
    if not os.path.exists(config_path):
        return None
    return yaml.safe_load(open(config_path, "r"))


def predict(model: torch.nn.Module, train_df: pd.DataFrame, val_df: pd.DataFrame,
            schema: Schema, context_size: int, prediction_horizon: int) -> pd.DataFrame:
    val_series_ids = schema.get_series_ids(val_df)

    input = torch.zeros((len(val_series_ids), context_size))
    train_series_groups = schema.get_series_groups(train_df)
    for series_idx, series_id in enumerate(val_series_ids):
        series_df = train_series_groups.get_group(series_id)
        series_history = torch.Tensor(series_df.iloc[-context_size:][schema.target_column].to_numpy())
        input[series_idx, :] = series_history

    # TODO we may need test_horizon instead of validation_horizon?
    # or just detect horizon from the val_df
    validation_horizon = schema.validation_horizon
    result = torch.zeros((schema.n_series, validation_horizon))
    for timestep in range(0, validation_horizon, prediction_horizon):
        output = model(input)  # TODO pass features to the model
        result[:, timestep:min(timestep + prediction_horizon, validation_horizon)] = output

        input = torch.cat([input, output], dim=-1)
        input = input[:, -context_size:]
    result = result.detach().numpy()

    result_df = val_df[[schema.series_id_column, "timestamp"]].copy()
    result_df["prediction"] = 0.0
    for series_idx, series_id in enumerate(val_series_ids):
        result_df.loc[result_df[schema.series_id_column].eq(series_id), ["prediction"]] = result[series_idx]

    return result_df


if __name__ == "__main__":
    log_dir = "logs"
    checkpoint_path = find_last_checkpoint(log_dir)
    if not checkpoint_path:
        print(f"Could not find checkpoint to use in {log_dir}")
        sys.exit(1)

    print(f"Using checkpoint {checkpoint_path}")

    config = get_config(checkpoint_path)
    if not config:
        # TODO use default config if config file is not found
        print(f"Could not find config file for checkpoint {checkpoint_path}")
        sys.exit(1)

    print(f"Using model config {config}")
    model_name = config["model_name"]
    context_size = config["context_size"]
    prediction_horizon = config["prediction_horizon"]

    if model_name == "linear":
        model = LinearModel(context_size, prediction_horizon)
    else:
        print(f"Unknown model name {model_name}")
        sys.exit(1)

    model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    model.eval()

    train_df = load_dataset("train")
    val_df = load_dataset("validation")
    metadata = load_metadata()
    schema = Schema.from_metadata(metadata)

    # TODO move prediction code to `predict.py`, in this script only read output csvs
    model_result = predict(model, train_df, val_df, schema, context_size, prediction_horizon)
    baselines_results = make_all_baselines(train_df, val_df)
    all_results = dict(baselines_results,
                       **{f"{model_name}:{context_size}:{prediction_horizon}": model_result})

    plot_series(model_result, title=f"Predictions for {model_name} model, "
                                    f"context {context_size}, prediction {prediction_horizon}",
                x_key="timestamp", y_key="prediction", num_series=-1)
    plot_prediction_comparison(all_results, schema, "unit_000")
