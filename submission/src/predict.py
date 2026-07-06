"""Helper functions to run inference and perform predictions with the trained model."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import torch

from .datasets import Schema
from .models import create_model


def predict(model: torch.nn.Module, train_df: pd.DataFrame, val_df: pd.DataFrame,
            schema: Schema, context_size: int, prediction_horizon: int) -> pd.DataFrame:
    """Run predictions using a given model with given training and validation datasets."""
    val_series_ids = schema.get_series_ids(val_df)

    x_values = torch.zeros((len(val_series_ids), context_size))
    x_features = torch.zeros((len(val_series_ids), context_size, len(schema.feature_columns)))
    train_series_groups = schema.get_series_groups(train_df)
    for series_idx, series_id in enumerate(val_series_ids):
        series_df = train_series_groups.get_group(series_id)
        series_x = torch.Tensor(series_df.iloc[-context_size:][schema.target_column].to_numpy())
        series_features = torch.Tensor(series_df.iloc[-context_size:][schema.feature_columns].to_numpy())
        x_values[series_idx, :] = series_x
        x_features[series_idx, :, :] = series_features

    # TODO val_df does not necessarily come directly after train_df
    # therefore prediction should use timestamps from the dataset
    validation_horizon = schema.validation_horizon
    result = torch.zeros((schema.n_series, validation_horizon))
    for timestep in range(0, validation_horizon, prediction_horizon):
        y_values, y_features = model(x_values, x_features)
        result[:, timestep:min(timestep + prediction_horizon, validation_horizon)] = y_values

        x_values = torch.cat([x_values, y_values], dim=-1)
        x_values = x_values[:, -context_size:]
        if y_features is not None:
            x_features = torch.cat([x_features, y_features], dim=1)
            x_features = x_features[:, -context_size:, :]

    result = result.detach().numpy()

    result_df = val_df[[schema.series_id_column, schema.timestamp_column]].copy()
    result_df[schema.prediction_column] = 0.0
    for series_idx, series_id in enumerate(val_series_ids):
        result_df.loc[result_df[schema.series_id_column].eq(series_id), [schema.prediction_column]] = result[series_idx]

    return result_df


def predict_for_checkpoint(checkpoint: Path, config: dict[str, Any], train_df: pd.DataFrame, val_df: pd.DataFrame,
                           schema: Schema) -> pd.DataFrame:
    print(f"Using model config {config}")
    model_name = config["model_name"]
    context_size = config["context_size"]
    prediction_horizon = config["prediction_horizon"]
    model_config = config.get("model_config", {})

    model, _ = create_model(model_name, context_size, prediction_horizon, schema, model_config)

    checkpoint: dict = torch.load(checkpoint, map_location="cpu")
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    elif isinstance(checkpoint, dict):
        model.load_state_dict(checkpoint)
    else:
        raise ValueError("Checkpoint must be a state_dict or a dict containing `state_dict`.")

    model.eval()
    return predict(model, train_df, val_df, schema, context_size, prediction_horizon)
