"""Validation functions to evaluate prediction results using various metrics."""
from abc import ABC, abstractmethod
from contextlib import contextmanager

import pandas as pd
import torch
from torch.nn import Module
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from .datasets import ForecastSample, Schema, get_slice_as_tensor
from .predict import predict_tensor


class Metric(ABC):
    """Base class for validation metrics."""
    name: str

    @abstractmethod
    def add_batch(self, predicted_values: torch.Tensor, actual_values: torch.Tensor):
        pass

    @abstractmethod
    def compute(self) -> float:
        pass


class Wape(Metric):
    name: str = "wape"

    def __init__(self):
        self.sum_error = 0.0
        self.sum_values = 0.0

    def add_batch(self, predicted_values: torch.Tensor, actual_values: torch.Tensor):
        self.sum_error = self.sum_error + torch.sum(torch.abs(actual_values - predicted_values)).item()
        self.sum_values = self.sum_values + torch.sum(torch.abs(actual_values)).item()

    def compute(self) -> float:
        return self.sum_error / self.sum_values


# TODO add additional metrics
METRICS_LIST = [Wape]


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
        val_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        metrics = [m() for m in METRICS_LIST]
        for sample in tqdm(val_loader, desc="Computing validation metrics", position=1, leave=False):
            x = sample['x']
            y = sample['y']
            x_features = sample['x_features']
            y_pred, _ = model(x, x_features)

            for m in metrics:
                m.add_batch(y_pred, y)
        result = {m.name: m.compute() for m in metrics}

    return result


def get_long_horizon_validation_metrics(model: Module, context_size: int, prediction_horizon: int,
                                        dataframe: pd.DataFrame, schema: Schema, prediction_start: int,
                                        n_predictions: int) -> dict[str, float]:
    """Run predictions for a time interval in the future and compute accuracy metrics.
    This may require the model to use its own predictions as input, which can amplify prediction errors.

    Args:
        model: model to use
        context_size: context size of the model
        prediction_horizon: prediction horizon of the model
        dataframe: dataset to use for validation
        schema: dataset schema
        prediction_start: the timestep at which to start predictions.
                          The data before this timestep are going to be used as context.
        n_predictions: how many steps in the future to predict.
                       Can be greater than the model prediction horizon to test if the model can make long-term predictions.
    Returns:
        Dictionary containing metrics names and metric values."""
    with evaluate(model):
        series_ids = schema.get_series_ids(dataframe)
        series_groups = schema.get_series_groups(dataframe)
        context_df = series_groups.head(prediction_start)
        predicted_values = predict_tensor(model, context_df, schema, context_size, prediction_horizon,
                                          series_ids, n_predictions)
        actual_values = get_slice_as_tensor(series_groups, series_ids,
                                            slice(prediction_start, prediction_start + n_predictions),
                                            schema.target_column)

        assert predicted_values.shape == actual_values.shape

        metrics = [m() for m in METRICS_LIST]
        for m in metrics:
            m.add_batch(predicted_values, actual_values)
        result = {m.name: m.compute() for m in metrics}

    return result
