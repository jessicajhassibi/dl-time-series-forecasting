from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import torch

from .config import get_config
from .datasets import Schema
from .models import create_model


def predict(model: torch.nn.Module, train_df: pd.DataFrame, val_df: pd.DataFrame,
            schema: Schema, context_size: int, prediction_horizon: int) -> pd.DataFrame:
    """Run predictions using given model with given training and validation datasets."""
    val_series_ids = schema.get_series_ids(val_df)

    input = torch.zeros((len(val_series_ids), context_size))
    train_series_groups = schema.get_series_groups(train_df)
    for series_idx, series_id in enumerate(val_series_ids):
        series_df = train_series_groups.get_group(series_id)
        series_history = torch.Tensor(series_df.iloc[-context_size:][schema.target_column].to_numpy())
        input[series_idx, :] = series_history

    # TODO val_df does not necessarily come directly after train_df
    # therefore prediction should use timestamps from the dataset
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


def predict_for_checkpoint(checkpoint: Path, train_df: pd.DataFrame, val_df: pd.DataFrame,
                           schema: Schema) -> dict[str, Any]:
    config = get_config(checkpoint)
    if not config:
        # TODO use default config if config file is not found
        raise ValueError(f"Could not find config file for checkpoint {checkpoint}")

    print(f"Using model config {config}")
    model_name = config["model_name"]
    context_size = config["context_size"]
    prediction_horizon = config["prediction_horizon"]

    model, _ = create_model(model_name, context_size, prediction_horizon)

    checkpoint: dict = torch.load(checkpoint, map_location="cpu")
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    elif isinstance(checkpoint, dict):
        model.load_state_dict(checkpoint)
    else:
        raise ValueError("Checkpoint must be a state_dict or a dict containing `state_dict`.")

    model.eval()
    model_result = predict(model, train_df, val_df, schema, context_size, prediction_horizon)
    # TODO use TypedDict or dataclass
    return dict(result=model_result, config=config)
