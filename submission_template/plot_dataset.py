import json
import os
from itertools import islice

import pandas as pd
import plotly.graph_objects as go
from huggingface_hub import snapshot_download


def load_dataset(split_name: str, dataset_dir: str = "dataset") -> tuple[pd.DataFrame, dict]:
    splits = {'train': 'train.csv', 'validation': 'validation_input.csv'}
    csv_path = os.path.join(dataset_dir, splits[split_name])
    metadata_path = os.path.join(dataset_dir, "metadata.json")
    if (not os.path.exists(csv_path)) or (not os.path.exists(metadata_path)):
        snapshot_download(repo_id="AIML-TUDA/dlam-ts-project-data-2026", repo_type="dataset",
                          local_dir=dataset_dir)

    print(f"Reading '{split_name}' split from {csv_path} and metadata from {metadata_path}")
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return pd.read_csv(csv_path), metadata


def plot_series(df: pd.DataFrame, name: str, num_series: int = 10,
                x_key: str = "timestamp", y_key: str = "target"):
    fig = go.Figure()
    for series_id, series_df in islice(df.groupby('series_id'), num_series):
        fig.add_trace(go.Scatter(x=series_df[x_key], y=series_df[y_key],
                                 mode="lines", name=str(series_id), ))
    fig.update_layout(title=f"{name} Dataset Plot",
                      xaxis_title=x_key.capitalize(), yaxis_title=y_key.capitalize())
    fig.show()


def plot_keys(df: pd.DataFrame, name: str, series_id: str,
              x_key: str = "timestamp", y_keys: tuple[str, ...] = ("target",)):
    fig = go.Figure()
    series_df = df.groupby('series_id').get_group(series_id)
    for y_key in y_keys:
        fig.add_trace(go.Scatter(x=series_df[x_key], y=series_df[y_key],
                                 mode="lines", name=y_key, ))
    fig.update_layout(title=f"{name} Dataset Plot for {series_id}",
                      xaxis_title=x_key.capitalize(), yaxis_title="Value")
    fig.show()


if __name__ == "__main__":
    train_df, metadata = load_dataset("train")
    plot_series(train_df, name="Train")

    keys: list[str] = metadata["schema"]["train"]
    keys.remove("timestamp")
    keys.remove("series_id")
    plot_keys(train_df, name="Train", series_id="unit_000", y_keys=tuple(keys))
