"""Inference entrypoint for final private evaluation.

Students should replace the placeholder logic with their trained model.
"""

from __future__ import annotations

import argparse
import os.path
from pathlib import Path

import pandas as pd
import torch

from config import get_config
from datasets import load_dataset
from linear import LinearModel
from src.datasets import load_metadata, Schema
from src.predict import find_last_checkpoint, predict


def load_forecast_index(input_dir: Path) -> pd.DataFrame:
    """Load the rows that need predictions."""
    candidates = [
        input_dir / "forecast_index_test.csv",
        input_dir / "forecast_index_validation.csv",
    ]
    for forecast_index in candidates:
        if forecast_index.exists():
            return pd.read_csv(forecast_index)
    expected = ", ".join(path.name for path in candidates)
    raise FileNotFoundError(f"Expected one of {expected} in input_dir.")


def main() -> None:
    """Load a checkpoint and write placeholder private-test predictions."""
    parser = argparse.ArgumentParser(description="Generate private test predictions.")
    parser.add_argument("--input_dir", required=False, default=Path("dataset"), type=Path)
    parser.add_argument("--output_file", required=False, default=None, type=Path)
    parser.add_argument("--checkpoint", required=False, default=None, type=Path)
    args = parser.parse_args()

    checkpoint: Path | None = args.checkpoint
    if checkpoint is None:
        checkpoint = find_last_checkpoint()

    if checkpoint is None or not checkpoint.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")

    train_df = load_dataset("train", args.input_dir)
    val_df = load_forecast_index(args.input_dir)

    metadata = load_metadata(args.input_dir)
    schema = Schema.from_metadata(metadata)

    config = get_config(checkpoint)
    if not config:
        # TODO use default config if config file is not found
        print(f"Could not find config file for checkpoint {checkpoint}")
        return

    print(f"Using model config {config}")
    model_name = config["model_name"]
    context_size = config["context_size"]
    prediction_horizon = config["prediction_horizon"]

    if model_name == "linear":
        model = LinearModel(context_size, prediction_horizon)
    else:
        print(f"Unknown model name {model_name}")
        return

    checkpoint: dict = torch.load(checkpoint, map_location="cpu")
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    elif isinstance(checkpoint, dict):
        model.load_state_dict(checkpoint)
    else:
        raise ValueError("Checkpoint must be a state_dict or a dict containing `state_dict`.")

    model.eval()

    model_result = predict(model, train_df, val_df, schema, context_size, prediction_horizon)

    output_file = args.output_file
    if output_file is None:
        output_file = Path(os.path.join("predictions", f"{model_name}_{context_size}_{prediction_horizon}.csv"))
    output_file.parent.mkdir(parents=True, exist_ok=True)
    model_result.to_csv(output_file, index=False)
    print(f"Written predictions to {output_file}")


if __name__ == "__main__":
    main()
