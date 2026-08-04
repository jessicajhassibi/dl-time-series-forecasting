"""Helper functions to run inference and perform predictions with the trained model."""
from __future__ import annotations

import pandas as pd
import torch
from pandas import DataFrame
from torch.nn import Module

from .datasets import Schema, get_slice_as_tensor


def predict(model: torch.nn.Module, context_df: pd.DataFrame, forecast_df: pd.DataFrame,
            schema: Schema, context_size: int, prediction_horizon: int,
            device: str | torch.device = "cpu") -> pd.DataFrame:
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
        device: device to use

    Return:
        A copy of forecast_df with a filled prediction column.
    """
    # Calculate how many steps (hours) in the future to predict
    forecast_timestamps = pd.to_datetime(forecast_df[schema.timestamp_column])
    min_forecast_timestamp = forecast_timestamps.min()
    max_forecast_timestamp = forecast_timestamps.max()
    max_context_timestamp = pd.to_datetime(context_df[schema.timestamp_column]).max()

    forecast_size = int((max_forecast_timestamp - min_forecast_timestamp).total_seconds() / 60 / 60) + 1
    forecast_horizon = int((max_forecast_timestamp - max_context_timestamp).total_seconds() / 60 / 60)
    print(f"Forecast size is {forecast_size}, forecast horizon is {forecast_horizon}")

    forecast_series_ids = schema.get_series_ids(forecast_df)

    result = predict_tensor(model, context_df, schema, context_size, prediction_horizon, forecast_series_ids,
                            forecast_horizon, device=device).cpu().detach().numpy()

    # Fill the dataframe with predictions by taking forecast_size values for each series
    # TODO use the timestamps in the forecast dataset for each series instead of just taking last block of values.
    result_df = forecast_df[[schema.series_id_column, schema.timestamp_column]].copy()
    result_df[schema.prediction_column] = 0.0
    for series_idx, series_id in enumerate(forecast_series_ids):
        series_mask = result_df[schema.series_id_column].eq(series_id)
        result_df.loc[series_mask, [schema.prediction_column]] = result[series_idx, -forecast_size:]

    return result_df


def predict_tensor(model: Module, context_df: DataFrame, schema: Schema, context_size: int,
                   prediction_horizon: int, forecast_series_ids: list[str], forecast_horizon: int,
                   device: str | torch.device = "cpu") -> torch.Tensor:
    # Get initial context tensors for predictions
    context_series_groups = schema.get_series_groups(context_df)
    x_values = get_slice_as_tensor(context_series_groups, forecast_series_ids, slice(-context_size, None),
                                   schema.target_column, device=device)
    x_features = get_slice_as_tensor(context_series_groups, forecast_series_ids, slice(-context_size, None),
                                     schema.feature_columns, device=device)

    # Run the model until reaching the forecast horizon
    result = torch.zeros((len(forecast_series_ids), forecast_horizon), device=device)
    for timestep in range(0, forecast_horizon, prediction_horizon):
        y_values, y_features = model(x_values, x_features)
        copied_size = min(prediction_horizon, forecast_horizon - timestep)
        result[:, timestep:timestep + copied_size] = y_values[:, :copied_size]

        x_values = torch.cat([x_values, y_values], dim=-1)
        x_values = x_values[:, -context_size:]
        if y_features is not None:
            x_features = torch.cat([x_features, y_features], dim=1)
            x_features = x_features[:, -context_size:, :]
        else:
            x_features[...] = torch.nan

    return result
