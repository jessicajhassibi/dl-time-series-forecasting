import os.path
from datetime import datetime

from src.datasets import load_dataset, ForecastDataset
from src.linear import LinearModel
from src.train import train_model

if __name__ == "__main__":
    prediction_horizon = 336
    context_size = 3 * prediction_horizon

    model_name = "linear"
    model = LinearModel(context_size, prediction_horizon)
    is_shifted_output = False

    train_df, metadata = load_dataset("train")
    train_dataset = ForecastDataset(train_df, metadata, context_size=context_size,
                                    prediction_horizon=prediction_horizon,
                                    is_shifted_output=is_shifted_output)
    print(f"Length of the dataset is {len(train_dataset)}")

    log_dir = os.path.join("logs", model_name, datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
    os.makedirs(log_dir, exist_ok=True)
    print(f"Writing experiment data to {log_dir}")

    train_model(model, train_dataset, log_dir, num_epochs=3)
