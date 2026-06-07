import sys

import torch

from config import get_config
from plot import plot_prediction_comparison
from src.baselines import make_all_baselines
from src.datasets import load_dataset, load_metadata, Schema
from src.linear import LinearModel
from src.plot import plot_series
from src.predict import predict, find_last_checkpoint

if __name__ == "__main__":
    log_dir = "logs"
    checkpoint_path = find_last_checkpoint(log_dir)
    if not checkpoint_path:
        print(f"Could not find checkpoint to use in {log_dir}")
        sys.exit(1)

    print(f"Using checkpoint {checkpoint_path}")

    config = get_config(checkpoint_path)
    if not config:
        # TODO use default config if config file is not found
        print(f"Could not find config file for checkpoint {checkpoint_path}")
        sys.exit(1)

    print(f"Using model config {config}")
    model_name = config["model_name"]
    context_size = config["context_size"]
    prediction_horizon = config["prediction_horizon"]

    if model_name == "linear":
        model = LinearModel(context_size, prediction_horizon)
    else:
        print(f"Unknown model name {model_name}")
        sys.exit(1)

    model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    model.eval()

    train_df = load_dataset("train")
    val_df = load_dataset("validation")
    metadata = load_metadata()
    schema = Schema.from_metadata(metadata)

    # TODO prediction code is already in `predict.py`, in this script only read output csvs
    model_result = predict(model, train_df, val_df, schema, context_size, prediction_horizon)
    baselines_results = make_all_baselines(train_df, val_df)
    all_results = dict(baselines_results,
                       **{f"{model_name}:{context_size}:{prediction_horizon}": model_result})

    plot_series(model_result, title=f"Predictions for {model_name} model, "
                                    f"context {context_size}, prediction {prediction_horizon}",
                x_key="timestamp", y_key="prediction", num_series=-1)
    plot_prediction_comparison(all_results, schema, "unit_000")
