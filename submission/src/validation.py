"""Validation functions to evaluate prediction results using various metrics."""
import math
from contextlib import contextmanager

import pandas as pd
import torch
from torch.nn import Module
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from .config import Config
from .datasets import ForecastSample, Schema, get_slice_as_tensor
from .predict import predict_tensor


class Metrics:
    def __init__(self):
        self.error_sum = 0.0
        self.error_squared_sum = 0.0
        self.value_sum = 0.0
        self.count = 0

    def add_batch(self, predicted_values: torch.Tensor, actual_values: torch.Tensor) -> Metrics:
        diff = actual_values - predicted_values
        self.error_sum += diff.abs().sum().item()
        self.error_squared_sum += (diff ** 2).sum().item()
        self.value_sum += actual_values.abs().sum().item()
        self.count += actual_values.numel()

        return self

    def compute(self) -> dict[str, float]:
        # TODO add additional metrics ?
        return dict(WAPE=self.error_sum / self.value_sum,
                    MAE=self.error_sum / self.count,
                    RMSE=math.sqrt(self.error_squared_sum / self.count))


@contextmanager
def evaluate(model: Module):
    """Switch the model to evaluation mode and disable gradients.

    Usage:

    .. code-block:: python

        with evaluate(model):
            # compute metrics
    """
    training_mode = model.training
    model.eval()
    try:
        with torch.no_grad():
            yield
    finally:
        model.train(mode=training_mode)


def get_validation_metrics(model: Module, dataset: Dataset[ForecastSample],
                           batch_size: int = 500) -> dict[str, float | int]:
    """
    Run prediction on the given dataset and compute accuracy metrics, such as WAPE.

    Args:
        model: model to use for prediction
        dataset: loader for the dataset to use for prediction
        batch_size: batch size to use
    Returns:
        dictionary containing metric names and metric values
    """
    with evaluate(model):
        val_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        metrics = Metrics()
        for sample in tqdm(val_loader, desc="Computing validation metrics", position=1, leave=False):
            x = sample['x']
            y = sample['y']
            x_features = sample['x_features']
            y_pred, _ = model(x, x_features)

            metrics.add_batch(y_pred, y)
        result = metrics.compute()

    return result


def get_long_horizon_validation_metrics(model: Module, config: Config,
                                        dataframe: pd.DataFrame, schema: Schema, prediction_start: int,
                                        n_predictions: int,
                                        device: str | torch.device = "cpu") -> dict[str, float]:
    """Take the provided dataset up to prediction_start as input, predict values after that point,
    and compare with the actual values in the dataset.
    Since the number of predictions can be bigger than the model prediction horizon,
    this may require the model to use its own predictions as input, which can amplify prediction errors.

    Args:
        model: model to use
        config: model configuration
        dataframe: dataset to use for validation
        schema: dataset schema
        prediction_start: the timestep at which to start predictions.
                          The data before this timestep are going to be used as context.
        n_predictions: how many steps in the future to predict.
                       Can be greater than the model prediction horizon to test if the model can make long-term predictions.
        device: device to use
    Returns:
        Dictionary containing metrics names and metric values."""
    with evaluate(model):
        series_ids = schema.get_series_ids(dataframe)
        series_groups = schema.get_series_groups(dataframe)
        context_df = series_groups.head(prediction_start)
        predicted_values = predict_tensor(model, context_df, schema, config.context_size, config.prediction_horizon,
                                          series_ids, n_predictions, device)
        actual_values = get_slice_as_tensor(series_groups, series_ids,
                                            slice(prediction_start, prediction_start + n_predictions),
                                            schema.target_column, device)

        assert predicted_values.shape == actual_values.shape

        result = Metrics().add_batch(predicted_values, actual_values).compute()
    return result
