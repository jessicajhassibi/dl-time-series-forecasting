import os.path
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch

from src.datasets import load_dataset, load_metadata, Schema
from src.linear import LinearModel


def find_last_checkpoint(parent_directory: str) -> str | None:
    if not os.path.exists(parent_directory):
        return None

    all_experiments = [os.path.join(parent_directory, d) for d in os.listdir(parent_directory)]
    all_experiments = [d for d in all_experiments if os.path.isdir(d)]
    if not all_experiments:
        return None

    last_experiment = max(all_experiments, key=os.path.getmtime)

    all_checkpoints = [os.path.join(last_experiment, d) for d in os.listdir(last_experiment)]
    all_checkpoints = [d for d in all_checkpoints if os.path.isfile(d) and d.endswith(".pt")]
    if not all_checkpoints:
        return None

    last_checkpoint = max(all_checkpoints, key=os.path.getmtime)
    return last_checkpoint


def predict(model: torch.nn.Module, train_df: pd.DataFrame, schema: Schema, context_size: int,
            prediction_horizon: int) -> np.ndarray:
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
        torch.roll(input, dims=(1,), shifts=(-prediction_horizon,))
        input[:, -prediction_horizon:] = output
        result[:, timestep:min(timestep + prediction_horizon, validation_horizon)] = output

    return result.detach().numpy()


if __name__ == "__main__":
    prediction_horizon = 336
    context_size = 3 * prediction_horizon
    # TODO write context_size and prediction_horizon to config file and load for prediction

    model_name = "linear"
    model = LinearModel(context_size, prediction_horizon)

    log_dir = os.path.join("logs", model_name)
    checkpoint_path = find_last_checkpoint(log_dir)
    if not checkpoint_path:
        print(f"Could not find checkpoint to use in {log_dir}")
        sys.exit(1)

    print(f"Using checkpoint {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    model.eval()

    train_df = load_dataset("train")
    metadata = load_metadata()
    schema = Schema.from_metadata(metadata)

    # TODO move prediction code to `predict.py`, in this script only read output csv
    result = predict(model, train_df, schema, context_size, prediction_horizon)

    # TODO add other baselines
    fig = go.Figure()
    for series_id, series_prediction in zip(schema.get_series_ids(train_df), result):
        timesteps = list(range(len(series_prediction)))  # TODO
        fig.add_trace(go.Scatter(x=timesteps, y=series_prediction,
                                 mode="lines", name=series_id))
    fig.update_layout(title=f"Predictions for {model_name}",
                      xaxis_title="Time", yaxis_title="Prediction")
    fig.show()
