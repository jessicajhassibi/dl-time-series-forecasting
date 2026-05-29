import json
import os
from itertools import islice

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly_express as px
from huggingface_hub import snapshot_download
from scipy.stats import pearsonr


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
    series_df = df.groupby('series_id').get_group(series_id)

    xs = series_df[x_key]

    fig = go.Figure()
    for y_key in y_keys:
        ys = series_df[y_key]
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name=y_key, ))
    fig.update_layout(title=f"{name} Dataset Plot for {series_id}",
                      xaxis_title=x_key.capitalize(), yaxis_title="Value")
    fig.show()


def plot_correlations(df: pd.DataFrame, series_id, y_keys):
    series_df = df.groupby('series_id').get_group(series_id)

    ys = [series_df[y_key] for y_key in y_keys]

    filtered_y_keys = []
    constant_y_keys = []
    filtered_ys = []
    for y_key, y in zip(y_keys, ys):
        std = np.std(y)
        if np.isclose(std, 0, 1e-6):
            constant_y_keys.append(y_key)
            continue
        filtered_y_keys.append(y_key)
        filtered_ys.append(y)

    print(f"Keys {constant_y_keys} have near-constant values")

    result = np.eye(len(filtered_y_keys))
    for i, y1 in enumerate(filtered_ys):
        for j, y2 in enumerate(filtered_ys):
            if i <= j:
                continue
            y_i = filtered_ys[i]
            y_j = filtered_ys[j]
            mask = ~np.isnan(y_i) & ~np.isnan(y_j)
            p = pearsonr(y_i[mask], y_j[mask])
            result[i, j] = p.statistic
            result[j, i] = result[i, j]

    fig = px.imshow(result, x=filtered_y_keys, y=filtered_y_keys, text_auto=".3f", aspect="auto",
                    labels=dict(color="Pearson Correlation Coefficient"), color_continuous_scale='RdBu')
    fig.update_layout(title=f"Correlations Between Variables in {series_id}")
    fig.show()


if __name__ == "__main__":
    train_df, metadata = load_dataset("train")

    keys: list[str] = metadata["schema"]["train"]
    keys.remove("timestamp")
    keys.remove("series_id")

    plot_series(train_df, name="Train")
    plot_keys(train_df, name="Train", series_id="unit_000", y_keys=tuple(keys))
    plot_correlations(train_df, series_id="unit_000", y_keys=tuple(keys))
