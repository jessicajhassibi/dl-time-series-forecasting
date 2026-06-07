import sys
from pathlib import Path

import pandas as pd

from src.baselines import make_all_baselines
from src.config import get_configuration_id
from src.datasets import load_dataset, load_metadata, Schema
from src.plot import plot_prediction_comparison
from src.plot import plot_series
from src.predict import find_last_checkpoint, predict_for_checkpoint


def load_previous_predictions(parent_dir: Path = Path("predictions")) -> dict[str, pd.DataFrame]:
    result = {}
    for prediction_file in list(parent_dir.glob("**/*.csv")):
        result["file:" + prediction_file.with_suffix("").name] = pd.read_csv(prediction_file)
    return result


if __name__ == "__main__":
    log_dir = "logs"
    checkpoint_path = find_last_checkpoint(log_dir)
    if not checkpoint_path:
        print(f"Could not find checkpoint to use in {log_dir}")
        sys.exit(1)

    print(f"Using checkpoint {checkpoint_path}")

    train_df = load_dataset("train")
    val_df = load_dataset("validation")
    metadata = load_metadata()
    schema = Schema.from_metadata(metadata)

    model_result_dict = predict_for_checkpoint(checkpoint_path, train_df, val_df, schema)
    model_id = get_configuration_id(model_result_dict["config"])
    model_result = model_result_dict["result"]

    baselines_results = make_all_baselines(train_df, val_df)
    all_results = dict(baselines_results,
                       **load_previous_predictions(),
                       **{"last:" + model_id: model_result})

    plot_series(model_result, title=f"Predictions for {model_id} model",
                x_key="timestamp", y_key="prediction", num_series=-1)
    plot_prediction_comparison(all_results, schema, "unit_000")
