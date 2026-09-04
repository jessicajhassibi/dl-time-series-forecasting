from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import tyro

from src.validation import Metrics


def compute_metrics(predicted: np.ndarray, actual: np.ndarray) -> dict[str, float]:
    return Metrics().add_batch(torch.from_numpy(np.asarray(predicted, dtype=np.float64)),
                               torch.from_numpy(np.asarray(actual, dtype=np.float64))).compute()


def score(predictions: Path, labels: Path,
          series_col: str = "series_id", time_col: str = "timestamp",
          prediction_col: str = "prediction", target_col: str = "target") -> dict[str, float]:
    """Align predictions with labels on (series_id, timestamp) and score them.

    Args:
        predictions: CSV with series_id, timestamp, prediction
        labels: CSV with series_id, timestamp, target
        series_col: name of the series id column
        time_col: name of the timestamp column
        prediction_col: name of the prediction column
        target_col: name of the target column
    Returns:
        Dictionary of metric names to values.
    """
    pred_df = pd.read_csv(predictions)
    label_df = pd.read_csv(labels)

    for frame, name, required in ((pred_df, predictions, prediction_col), (label_df, labels, target_col)):
        missing = {series_col, time_col, required} - set(frame.columns)
        if missing:
            raise ValueError(f"{name} is missing column(s) {sorted(missing)}")

    for frame in (pred_df, label_df):
        frame[time_col] = pd.to_datetime(frame[time_col])

    merged = label_df.merge(pred_df, on=[series_col, time_col], how="left", validate="one_to_one")

    unmatched = int(merged[prediction_col].isna().sum())
    if unmatched:
        raise ValueError(
            f"{unmatched} of {len(merged)} label rows have no matching prediction. "
            f"Every row of the label file must be covered.")

    result = compute_metrics(merged[prediction_col].to_numpy(dtype=float),
                             merged[target_col].to_numpy(dtype=float))

    print(f"Scored {len(merged)} rows across {merged[series_col].nunique()} series.")
    for metric_name, metric_value in result.items():
        print(f"{metric_name}: {metric_value:.4f}")
    return result


if __name__ == "__main__":
    tyro.cli(score)
