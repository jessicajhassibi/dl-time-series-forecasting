import os.path
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
import yaml

from src.baselines import make_all_baselines
from src.datasets import load_dataset, load_metadata, Schema
from src.linear import LinearModel


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
            schema: Schema, context_size: int, prediction_horizon: int) -> np.ndarray:
    # TODO instead of predicting for all series, run predictions for val_df
    input = torch.zeros((schema.n_series, context_size))
    series_groups = schema.get_series_groups(train_df)
    for series_idx, series_id in enumerate(schema.get_series_ids(train_df)):
        series_df = series_groups.get_group(series_id)
        series_history = torch.Tensor(series_df.iloc[-context_size:][schema.target_column].to_numpy())
        input[series_idx, :] = series_history

    validation_horizon = schema.validation_horizon
    result = torch.zeros((schema.n_series, validation_horizon))
    for timestep in range(0, validation_horizon, prediction_horizon):
        output = model(input)  # TODO add features for models using features
        result[:, timestep:min(timestep + prediction_horizon, validation_horizon)] = output

        input = torch.cat([input, output], dim=-1)
        input = input[:, -context_size:]

    return result.detach().numpy()  # TODO return Dataframe


def plot_predictions(result: np.ndarray, series_ids: list[str]):
    fig = go.Figure()
    for series_id, series_prediction in zip(series_ids, result):
        timesteps = list(range(len(series_prediction)))  # TODO
        fig.add_trace(go.Scatter(x=timesteps, y=series_prediction,
                                 mode="lines", name=series_id))
    fig.update_layout(title=f"Predictions for {model_name} model, "
                            f"context {context_size}, prediction {prediction_horizon}",
                      xaxis_title="Time", yaxis_title="Prediction")
    fig.show()


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

    # TODO move prediction code to `predict.py`, in this script only read output csv
    model_result = predict(model, train_df, val_df, schema, context_size, prediction_horizon)
    baselines_results = make_all_baselines(train_df, val_df)

    series_ids = schema.get_series_ids(train_df)
    plot_predictions(model_result, series_ids)

    series_id = "unit_000"
    series_idx = series_ids.index(series_id)
    fig = go.Figure()
    for baseline_name, baseline_result in baselines_results.items():
        series_df = baseline_result.groupby(schema.series_id_column).get_group(series_id)
        xs = series_df["timestamp"]  # TODO move to schema
        ys = series_df["prediction"]
        fig.add_trace(go.Scatter(x=xs, y=ys,
                                 mode="lines", name=baseline_name))
    fig.add_trace(go.Scatter(x=xs, y=model_result[series_idx],
                             mode="lines", name=f"{model_name}:{context_size}:{prediction_horizon}"))
    fig.update_layout(title=f"Comparison of predictions for series {series_id}",
                      xaxis_title="Time", yaxis_title="Prediction")
    fig.show()
