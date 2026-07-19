"""A simple plotting script to compare predictions:
1. Runs all baselines.
2. Reads all prediction files available in the `predictions` folder.
3. Runs prediction for the most recent checkpoint found in the `logs` folder."""
from pathlib import Path

import pandas as pd

from src.config import get_config
from src.datasets import load_dataset, load_forecast_index, Schema, preprocess_dataset
from src.datasets import load_schema
from src.model_registry import create_model
from src.models.baselines import make_all_baselines
from src.predict import predict
from src.train import load_model
from src.util.plot import plot_prediction_comparison, plot_series
from src.util.util import find_last_checkpoint


def load_previous_predictions(parent_dir: Path = Path("predictions")) -> dict[str, pd.DataFrame]:
    result = {}
    for prediction_file in list(parent_dir.glob("**/*.csv")):
        result["file:" + prediction_file.with_suffix("").name] = pd.read_csv(prediction_file)
    return result


def predict_with_last_checkpoint(context_df: pd.DataFrame, forecast_df: pd.DataFrame, schema: Schema,
                                 log_dir: str = "logs") -> dict[str, pd.DataFrame]:
    checkpoint_path = find_last_checkpoint(log_dir)
    if not checkpoint_path:
        print(f"Could not find checkpoint to use in {log_dir}")
        return {}

    print(f"Using last checkpoint: {checkpoint_path}")

    config, _ = get_config(checkpoint_path)
    print(f"Using model config {config}")

    model = create_model(config, schema)
    load_model(checkpoint_path, model, device="cpu")
    model.eval()
    model_result = predict(model, context_df, forecast_df, schema,
                           config.context_size, config.prediction_horizon)
    model_id = config.get_id()
    return {model_id: model_result}


if __name__ == "__main__":
    context_df = load_dataset("train")
    forecast_df = load_forecast_index()
    schema = load_schema()

    preprocess_dataset(context_df, schema)

    baselines_results = make_all_baselines(context_df, forecast_df)
    previous_results = load_previous_predictions()
    model_results = predict_with_last_checkpoint(context_df, forecast_df, schema)
    all_results = dict(baselines_results,
                       **previous_results,
                       **model_results)

    for model_name, model_prediction in model_results.items():
        plot_series(model_prediction, title=f"Predictions for {model_name}",
                    x_key=schema.timestamp_column, y_key=schema.prediction_column, num_series=-1)
    plot_prediction_comparison(all_results, schema, "unit_000")
