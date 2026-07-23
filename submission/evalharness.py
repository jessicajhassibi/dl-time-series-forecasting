"""Offline evaluation mirroring the leaderboard: hold out the last `horizon` hours of each
training series, roll the model forward, and score WAPE/MAE/RMSE.

Run: python eval_harness.py --checkpoint logs/tcn_deep/<run_dir>/checkpoint-best.pt
"""
from __future__ import annotations
from pathlib import Path

import torch
import tyro

from src.config import get_config
from src.datasets import load_dataset, load_schema, preprocess_dataset, get_slice_as_tensor
from src.predict import predict_tensor
from src.train import load_model
from src.models.tcn_deep import TCNDeep


def evaluate(checkpoint: Path, horizon: int | None = None):
    config, _ = get_config(checkpoint)
    print(f"Loaded config: {config}")

    df = load_dataset("train")
    schema = load_schema()
    preprocess_dataset(df, schema)

    if horizon is None:
        horizon = schema.validation_horizon
    total = schema.n_training_steps
    prediction_start = total - horizon
    print(f"Series length {total}; holding out last {horizon} steps (start {prediction_start}).")

    model = TCNDeep(config.context_size, config.prediction_horizon,
                    len(schema.feature_columns), **config.model_config)
    load_model(checkpoint, model, device="cpu")
    model.eval()

    series_ids = schema.get_series_ids(df)
    groups = schema.get_series_groups(df)
    context_df = groups.head(prediction_start)

    with torch.no_grad():
        pred = predict_tensor(model, context_df, schema, config.context_size,
                              config.prediction_horizon, series_ids, horizon)
    actual = get_slice_as_tensor(groups, series_ids,
                                 slice(prediction_start, prediction_start + horizon),
                                 schema.target_column)

    diff = pred - actual
    wape = diff.abs().sum().item() / actual.abs().sum().item()
    mae = diff.abs().mean().item()
    rmse = (diff ** 2).mean().sqrt().item()
    print("\n=== Rollout evaluation (leaderboard-style) ===")
    print(f"WAPE: {wape:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")


if __name__ == "__main__":
    tyro.cli(evaluate)