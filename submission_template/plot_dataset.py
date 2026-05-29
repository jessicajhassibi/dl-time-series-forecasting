import os
from itertools import islice

import pandas as pd
import plotly.graph_objects as go
from huggingface_hub import snapshot_download


def load_dataset(split_name: str, dataset_dir: str = "dataset") -> pd.DataFrame:
    splits = {'train': 'train.csv', 'validation': 'validation_input.csv'}
    csv_path = os.path.join(dataset_dir, splits[split_name])
    if not os.path.exists(csv_path):
        snapshot_download(repo_id="AIML-TUDA/dlam-ts-project-data-2026", repo_type="dataset",
                          local_dir=dataset_dir)
    print(f"Reading '{split_name}' split from {csv_path}")
    return pd.read_csv(csv_path)


def plot_dataset(df: pd.DataFrame, name: str, num_series: int = 10, x_key="timestamp", y_key="target"):
    fig = go.Figure()
    for series_id, series_df in islice(df.groupby('series_id'), num_series):
        fig.add_trace(go.Scatter(x=series_df[x_key], y=series_df[y_key],
                                 mode="lines", name=str(series_id), ))
    fig.update_layout(title=f"{name} Dataset Plot",
                      xaxis_title=x_key.capitalize(), yaxis_title=y_key.capitalize())
    fig.show()


if __name__ == "__main__":
    train_df = load_dataset("train")
    plot_dataset(train_df, name="Train")
