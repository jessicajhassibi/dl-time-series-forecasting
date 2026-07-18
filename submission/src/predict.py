"""Helper functions to run inference and perform predictions with the trained model."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import torch

from .datasets import Schema
from .models import create_model


def predict(model: torch.nn.Module, context_df: pd.DataFrame, forecast_df: pd.DataFrame,
            schema: Schema, context_size: int, prediction_horizon: int) -> pd.DataFrame:
    """Run predictions using a given model with given context and forecast dataframes.
    If the forecast dataframe is outside the prediction horizon of the model,
    predictions are used as context to fill the gap.

    Assumptions:
     - given forecast dataset is a continuous block in the future;
     - each series in the forecast dataset has the same timestamp range;
     - predictions are done hourly.

    Args:
        model: model to use for predictions
        context_df: dataframe with historical data to use as context for the forecast
        forecast_df: dataframe with series ids and timestamps
        schema: dataset schema
        context_size: context size of the model
        prediction_horizon: prediction horizon of the model

    Return:
        A copy of forecast_df with a filled prediction column.
    """
    forecast_series_ids = schema.get_series_ids(forecast_df)

    # Fill initial context tensors for predictions
    x_values = torch.zeros((len(forecast_series_ids), context_size))
    x_features = torch.zeros((len(forecast_series_ids), context_size, len(schema.feature_columns)))
    context_series_groups = schema.get_series_groups(context_df)
    for series_idx, series_id in enumerate(forecast_series_ids):
        series_df = context_series_groups.get_group(series_id)
        series_x = torch.Tensor(series_df.iloc[-context_size:][schema.target_column].to_numpy().copy())
        series_features = torch.Tensor(series_df.iloc[-context_size:][schema.feature_columns].to_numpy())
        x_values[series_idx, :] = series_x
        x_features[series_idx, :, :] = series_features

    # Calculate how many steps (hours) in the future to predict
    forecast_timestamps = pd.to_datetime(forecast_df[schema.timestamp_column])
    min_forecast_timestamp = forecast_timestamps.min()
    max_forecast_timestamp = forecast_timestamps.max()
    max_context_timestamp = pd.to_datetime(context_df[schema.timestamp_column]).max()

    forecast_size = int((max_forecast_timestamp - min_forecast_timestamp).total_seconds() / 60 / 60) + 1
    forecast_horizon = int((max_forecast_timestamp - max_context_timestamp).total_seconds() / 60 / 60)
    print(f"Forecast size is {forecast_size}, forecast horizon is {forecast_horizon}")

    # Run the model until reaching the forecast horizon
    result = torch.zeros((len(forecast_series_ids), forecast_horizon))
    for timestep in range(0, forecast_horizon, prediction_horizon):
        y_values, y_features = model(x_values, x_features)
        result[:, timestep:min(timestep + prediction_horizon, forecast_horizon)] = y_values

        x_values = torch.cat([x_values, y_values], dim=-1)
        x_values = x_values[:, -context_size:]
        if y_features is not None:
            x_features = torch.cat([x_features, y_features], dim=1)
            x_features = x_features[:, -context_size:, :]

    result = result.detach().numpy()

    # Fill the dataframe with predictions by taking forecast_size values for each series
    # TODO use the timestamps in the forecast dataset for each series instead of just taking last block of values.
    result_df = forecast_df[[schema.series_id_column, schema.timestamp_column]].copy()
    result_df[schema.prediction_column] = 0.0
    for series_idx, series_id in enumerate(forecast_series_ids):
        series_mask = result_df[schema.series_id_column].eq(series_id)
        result_df.loc[series_mask, [schema.prediction_column]] = result[series_idx, -forecast_size:]

    return result_df


def predict_for_checkpoint(checkpoint: Path, config: dict[str, Any], context_df: pd.DataFrame,
                           forecast_df: pd.DataFrame, schema: Schema) -> pd.DataFrame:
    print(f"Using model config {config}")

    model_name = config["model_name"]
    context_size = config["context_size"]
    prediction_horizon = config["prediction_horizon"]
    model_config = config.get("model_config", {})

    model = create_model(model_name, context_size, prediction_horizon, schema, model_config)

    checkpoint: dict = torch.load(checkpoint, map_location="cpu")
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    elif isinstance(checkpoint, dict):
        model.load_state_dict(checkpoint)
    else:
        raise ValueError("Checkpoint must be a state_dict or a dict containing `state_dict`.")

    model.eval()
    return predict(model, context_df, forecast_df, schema, context_size, prediction_horizon)
