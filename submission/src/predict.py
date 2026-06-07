import os
from pathlib import Path

import pandas as pd
import torch

from datasets import Schema


def find_last_checkpoint(parent_directory: str = "logs") -> Path | None:
    """"""
    if not os.path.exists(parent_directory):
        return None
    parent_path = Path(parent_directory)
    checkpoints = list(parent_path.glob("**/*.pt"))
    if not checkpoints:
        return None
    return max(checkpoints, key=os.path.getmtime)


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
