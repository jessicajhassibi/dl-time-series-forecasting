from itertools import islice

import pandas as pd
from plotly import graph_objects as go

from .datasets import Schema


def plot_series(df: pd.DataFrame, title: str, x_key: str, y_key: str, num_series: int = 10):
    series_to_plot = df.groupby('series_id')
    if num_series is not None and num_series > 0:
        series_to_plot = islice(series_to_plot, num_series)

    fig = go.Figure()
    for series_id, series_df in series_to_plot:
        fig.add_trace(go.Scatter(x=series_df[x_key], y=series_df[y_key],
                                 mode="lines", name=series_id, ))
    fig.update_layout(title=title,
                      xaxis_title=x_key.capitalize(), yaxis_title=y_key.capitalize())
    fig.show()


def plot_prediction_comparison(results: dict[str, pd.DataFrame], schema: Schema,
                               series_id: str):
    fig = go.Figure()
    for name, result in results.items():
        series_df = result.groupby(schema.series_id_column).get_group(series_id)
        xs = series_df[schema.timestamp_column]
        ys = series_df[schema.prediction_column]
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name=name))
    fig.update_layout(title=f"Predictions Comparison for {series_id}",
                      xaxis_title=schema.timestamp_column.capitalize(),
                      yaxis_title=schema.prediction_column.capitalize())
    fig.show()
