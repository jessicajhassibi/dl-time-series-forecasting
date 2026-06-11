"""Inference entrypoint for final private evaluation.

Students should replace the placeholder logic with their trained model.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

from src.config import get_configuration_id
from src.datasets import load_dataset, load_schema
from src.predict import predict_for_checkpoint
from src.util import find_last_checkpoint


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


def do_predict(checkpoint: Path | None, input_dir: Path, output_file: Path | None):
    if checkpoint is None:
        checkpoint = find_last_checkpoint()

    if checkpoint is None or not checkpoint.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")

    train_df = load_dataset("train", input_dir)
    val_df = load_forecast_index(input_dir)
    schema = load_schema(input_dir)

    model_result_dict = predict_for_checkpoint(checkpoint, train_df, val_df, schema)
    model_result = model_result_dict["result"]

    if output_file is None:
        config = model_result_dict["config"]
        output_file = Path(os.path.join("predictions", get_configuration_id(config) + ".csv"))
    output_file.parent.mkdir(parents=True, exist_ok=True)
    model_result.to_csv(output_file, index=False)
    print(f"Written predictions to {output_file}")


def main() -> None:
    """Load a checkpoint and write predictions."""
    parser = argparse.ArgumentParser(description="Generate private test predictions.")
    parser.add_argument("--input_dir", required=False, default=Path("dataset"), type=Path)
    parser.add_argument("--output_file", required=False, default=None, type=Path)
    parser.add_argument("--checkpoint", required=False, default=None, type=Path)
    args = parser.parse_args()

    do_predict(args.checkpoint, args.input_dir, args.output_file)


if __name__ == "__main__":
    main()
