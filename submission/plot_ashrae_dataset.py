"""Plotting script for the additional dataset (ASHRAE GEPIII electricity).

Mirrors plot_dataset.py, but points at `dataset_ashrae` and adds plots that only make
sense for this dataset:
  - monthly mean load, which shows the cooling-driven summer peak (ASHRAE puts heating on
    separate steam/hot-water meters, so meter 0 peaks in July, not in winter);
  - the held-out validation window against its ground truth, which we can only draw because
    we own the ASHRAE labels -- the benchmark keeps them on the leaderboard server.

Run:  python plot_ashrae_dataset.py
"""
from calendar import month_abbr
from pathlib import Path

import pandas as pd
from plotly import graph_objects as go

from src.datasets import Schema, load_dataset, load_schema
from src.util.plot import (partition_keys, plot_correlations, plot_distribution, plot_keys,
                           plot_series)

DATASET_DIR = Path("dataset_ashrae")


def require_dataset(dataset_dir: Path = DATASET_DIR) -> None:
    """Fail early with a useful message if the dataset has not been converted yet.

    Without this, load_dataset() would treat the missing file as "not downloaded yet" and
    pull the *benchmark* dataset from Hugging Face into dataset_ashrae.
    """
    if not (dataset_dir / "train.csv").exists():
        raise FileNotFoundError(
            f"{dataset_dir}/train.csv not found. Build it first:\n"
            f"    python convert_ashrae.py --n-series 250\n"
            f"(raw Kaggle files are needed too -- see SETUP.md section 3)")


def load_validation_target(dataset_dir: Path = DATASET_DIR) -> pd.DataFrame:
    """Load our own held-out labels. Not part of load_dataset(), which only knows the
    'train' and 'validation' (= inputs only) splits."""
    return pd.read_csv(dataset_dir / "validation_target.csv")


def plot_monthly_profile(train_df: pd.DataFrame, val_df: pd.DataFrame, schema: Schema) -> None:
    """Mean target per calendar month across all series, over the full year."""
    columns = [schema.timestamp_column, schema.target_column]
    full_year = pd.concat([train_df[columns], val_df[columns]])
    timestamps = pd.to_datetime(full_year[schema.timestamp_column])
    monthly = full_year.groupby(timestamps.dt.month)[schema.target_column].mean()

    fig = go.Figure(go.Bar(x=[month_abbr[m] for m in monthly.index], y=monthly.to_numpy()))
    fig.update_layout(title="Mean Load per Month (all series, 2016)",
                      xaxis_title="Month", yaxis_title="Mean target (kWh)")
    fig.show()


def plot_validation_window(train_df: pd.DataFrame, val_target_df: pd.DataFrame, schema: Schema,
                           series_id: str, context_hours: int = 336) -> None:
    """Tail of the training history followed by the held-out ground truth for one series."""
    history = train_df[train_df[schema.series_id_column] == series_id].tail(context_hours)
    held_out = val_target_df[val_target_df[schema.series_id_column] == series_id]
    split_at = pd.to_datetime(held_out[schema.timestamp_column].iloc[0])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=history[schema.timestamp_column], y=history[schema.target_column],
                             mode="lines", name="history (train)"))
    fig.add_trace(go.Scatter(x=held_out[schema.timestamp_column], y=held_out[schema.target_column],
                             mode="lines", name="held-out target"))
    fig.add_vline(x=split_at, line_dash="dash", line_color="grey")
    fig.update_layout(title=f"Validation Window for {series_id} "
                            f"(last {context_hours}h of history + {len(held_out)}h held out)",
                      xaxis_title=schema.timestamp_column.capitalize(),
                      yaxis_title=f"{schema.target_column.capitalize()} (kWh)")
    fig.show()


if __name__ == "__main__":
    require_dataset()

    train_df = load_dataset("train", DATASET_DIR)
    val_target_df = load_validation_target()
    schema = load_schema(DATASET_DIR)

    print(f"Dataset Schema:\n{schema}")

    # the surviving building ids depend on the zero-run filter, so never hardcode one
    series_id = schema.get_series_ids(train_df)[0]
    variable_keys, constant_keys = partition_keys(train_df, schema.feature_columns)

    plot_series(train_df, title="ASHRAE Training Dataset Plot",
                x_key=schema.timestamp_column, y_key=schema.target_column)
    plot_keys(train_df, name="ASHRAE Train", series_id=series_id, x_key=schema.timestamp_column,
              y_keys=[schema.target_column] + schema.feature_columns)
    plot_correlations(train_df, series_id=series_id, y_keys=[schema.target_column] + variable_keys)
    plot_distribution(train_df, keys=constant_keys)

    plot_monthly_profile(train_df, val_target_df, schema)
    plot_validation_window(train_df, val_target_df, schema, series_id=series_id)
