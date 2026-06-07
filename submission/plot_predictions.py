import sys

from src.config import get_configuration_id
from src.plot import plot_prediction_comparison
from src.baselines import make_all_baselines
from src.datasets import load_dataset, load_metadata, Schema
from src.plot import plot_series
from src.predict import find_last_checkpoint, predict_for_checkpoint

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

    # TODO maybe read predictions from predictions directory instead of running a model
    model_result_dict = predict_for_checkpoint(checkpoint_path, train_df, val_df, schema)
    model_id = get_configuration_id(model_result_dict["config"])
    model_result = model_result_dict["result"]

    baselines_results = make_all_baselines(train_df, val_df)
    all_results = dict(baselines_results,
                       **{model_id: model_result})

    plot_series(model_result, title=f"Predictions for {model_id} model",
                x_key="timestamp", y_key="prediction", num_series=-1)
    plot_prediction_comparison(all_results, schema, "unit_000")
