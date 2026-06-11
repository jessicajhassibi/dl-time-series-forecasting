from pathlib import Path

import pandas as pd

from src.baselines import make_all_baselines
from src.config import get_configuration_id
from src.datasets import load_dataset
from src.datasets import load_schema
from src.plot import plot_prediction_comparison
from src.plot import plot_series
from src.predict import predict_for_checkpoint
from src.util import find_last_checkpoint


def load_previous_predictions(parent_dir: Path = Path("predictions")) -> dict[str, pd.DataFrame]:
    result = {}
    for prediction_file in list(parent_dir.glob("**/*.csv")):
        result["file:" + prediction_file.with_suffix("").name] = pd.read_csv(prediction_file)
    return result


def predict_with_last_checkpoint(log_dir: str = "logs") -> dict[str, pd.DataFrame]:
    checkpoint_path = find_last_checkpoint(log_dir)
    if not checkpoint_path:
        print(f"Could not find checkpoint to use in {log_dir}")
        return {}

    print(f"Using last checkpoint: {checkpoint_path}")
    model_result_dict = predict_for_checkpoint(checkpoint_path, train_df, val_df, schema)
    model_id = get_configuration_id(model_result_dict["config"])
    model_result = model_result_dict["result"]
    return {model_id: model_result}


if __name__ == "__main__":
    train_df = load_dataset("train")
    val_df = load_dataset("validation")
    schema = load_schema()

    baselines_results = make_all_baselines(train_df, val_df)
    previous_results = load_previous_predictions()
    model_results = predict_with_last_checkpoint()
    all_results = dict(baselines_results,
                       **previous_results,
                       **model_results)

    for model_name, model_prediction in model_results.items():
        plot_series(model_prediction, title=f"Predictions for {model_name}",
                    x_key="timestamp", y_key="prediction", num_series=-1)
    plot_prediction_comparison(all_results, schema, "unit_000")
