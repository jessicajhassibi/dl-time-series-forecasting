from collections import defaultdict

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly_express as px
from plotly import colors as pc
from plotly.subplots import make_subplots
from scipy.stats import pearsonr

from src.datasets import load_dataset, load_schema
from src.plot import plot_series


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


def plot_keys(df: pd.DataFrame, name: str, series_id: str,
              x_key: str, y_keys: list[str]):
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
        fig.add_trace(go.Histogram(y=key_values, name=key, marker_color=get_color("Prism", i)),
                      row=(i // max_columns) + 1,
                      col=(i % max_columns) + 1)

    fig.update_layout(title=f"Histogram of Variables that are Constant in Each Series",
                      xaxis_title="Number of Series", yaxis_title="Value",
                      bargap=0.2)
    fig.show()


if __name__ == "__main__":
    train_df = load_dataset("train")
    schema = load_schema()

    print(f"Dataset Schema:\n{schema}")

    variable_keys, constant_keys = partition_keys(train_df, schema.feature_columns)

    plot_series(train_df, title=f"Training Dataset Plot", x_key="timestamp", y_key=schema.target_column)
    plot_keys(train_df, name="Train", series_id="unit_000", x_key="timestamp",
              y_keys=[schema.target_column] + schema.feature_columns)
    plot_correlations(train_df, series_id="unit_000", y_keys=[schema.target_column] + variable_keys)
    plot_distribution(train_df, keys=constant_keys)
