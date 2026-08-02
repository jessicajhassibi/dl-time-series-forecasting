"""Offline evaluation mirroring the leaderboard: hold out the last `horizon` hours of each
training series, roll the model forward, and score WAPE/MAE/RMSE.

IMPORTANT:
Such evaluation results do not indicate the true model performance as the model may have already seen predicted values during training.

Run: python eval_harness.py --checkpoint logs/tcn_deep/<run_dir>/checkpoint-best.pt
"""
from __future__ import annotations

from pathlib import Path

import tyro

from src.datasets import load_dataset, load_schema, preprocess_dataset
from src.model_registry import create_model, get_config
from src.train import load_model
from src.validation import get_long_horizon_validation_metrics


def evaluate(checkpoint: Path, horizon: int | None = None):
    config = get_config(checkpoint)
    print(f"Loaded config: {config}")

    df = load_dataset("train")
    schema = load_schema()
    preprocess_dataset(df, schema)

    if horizon is None:
        horizon = schema.validation_horizon
    total = schema.n_training_steps
    prediction_start = total - horizon
    print(f"Series length {total}; holding out last {horizon} steps (start {prediction_start}).")

    model = create_model(config, schema)
    load_model(checkpoint, model)

    result = get_long_horizon_validation_metrics(model, config, df, schema, prediction_start, horizon)

    print("\n=== Rollout evaluation (leaderboard-style) ===")
    for metric_name, metric_value in result.items():
        print(f"{metric_name}: {metric_value:.4f}")


if __name__ == "__main__":
    tyro.cli(evaluate)
