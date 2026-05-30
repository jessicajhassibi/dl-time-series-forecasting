import json
import os
from collections import defaultdict
from itertools import islice

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly_express as px
from huggingface_hub import snapshot_download
from plotly import colors as pc
from plotly.subplots import make_subplots
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


def partition_keys(df: pd.DataFrame, keys: list[str], eps=1e-6) -> tuple[list[str], list[str]]:
    variable_keys = []
    constant_keys = []
    constant_keys.extend(keys)
    for series_id, series_df in df.groupby('series_id'):
        keys_to_check = constant_keys.copy()
        for key in keys_to_check:
            std = np.std(series_df[key])
            if not np.isclose(std, 0, atol=eps):
                constant_keys.remove(key)
                variable_keys.append(key)

    return variable_keys, constant_keys


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


def plot_correlations(df: pd.DataFrame, series_id, y_keys: list[str]):
    series_df = df.groupby('series_id').get_group(series_id)

    ys = [series_df[y_key] for y_key in y_keys]

    result = np.eye(len(y_keys))
    for i, y1 in enumerate(ys):
        for j, y2 in enumerate(ys):
            if i <= j:
                continue
            y_i = ys[i]
            y_j = ys[j]
            mask = ~np.isnan(y_i) & ~np.isnan(y_j)
            p = pearsonr(y_i[mask], y_j[mask])
            result[i, j] = p.statistic
            result[j, i] = result[i, j]

    fig = px.imshow(result, x=y_keys, y=y_keys, text_auto=".3f", aspect="auto",
                    labels=dict(color="Pearson Correlation Coefficient"), color_continuous_scale='RdBu')
    fig.update_layout(title=f"Correlations Between Variables in {series_id}")
    fig.show()


def get_color(color_sequence_name: str, idx: int) -> str:
    color_sequence = getattr(pc.qualitative, color_sequence_name)
    return color_sequence[idx % len(color_sequence)]


def plot_distribution(df: pd.DataFrame, keys: list[str], max_columns: int = 4):
    values = defaultdict(list)
    for series_id, series_df in df.groupby('series_id'):
        for key in keys:
            mean_value = np.mean(series_df[key])
            values[key].append(mean_value)

    num_columns = min(len(keys), max_columns)
    fig = make_subplots(rows=len(keys) // max_columns + 1, cols=num_columns, subplot_titles=keys)
    for i, key in enumerate(keys):
        key_values = values[key]
        fig.add_trace(go.Histogram(y=key_values, name=key, marker_color=get_color("Prism", i)), row=(i // max_columns) + 1,
                      col=(i % max_columns) + 1)

    fig.update_layout(title=f"Histogram of Variables that are Constant in Each Series",
                      xaxis_title="Number of Series", yaxis_title="Value",
                      bargap=0.2)
    fig.show()


if __name__ == "__main__":
    train_df, metadata = load_dataset("train")

    keys: list[str] = metadata["schema"]["train"]
    keys.remove("timestamp")
    keys.remove("series_id")

    variable_keys, constant_keys = partition_keys(train_df, keys)

    plot_series(train_df, name="Train")
    plot_keys(train_df, name="Train", series_id="unit_000", y_keys=tuple(keys))
    plot_correlations(train_df, series_id="unit_000", y_keys=variable_keys)
    plot_distribution(train_df, keys=constant_keys)
