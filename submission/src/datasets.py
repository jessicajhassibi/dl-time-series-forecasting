import json
import os
import pandas as pd
from dataclasses import dataclass
from huggingface_hub import snapshot_download
from pandas.core.groupby import DataFrameGroupBy
from pathlib import Path
from torch import Tensor
from torch.utils.data import Dataset
from typing import TypedDict


def download_dataset(dataset_dir: str | Path, target_path: str):
    """Downloads the dataset into the specified directory if the target path is not present."""
    if not os.path.exists(target_path):
        snapshot_download(repo_id="AIML-TUDA/dlam-ts-project-data-2026", repo_type="dataset",
                          local_dir=dataset_dir)
    if not os.path.exists(target_path):
        raise FileNotFoundError(f"Required path {target_path} not found in the dataset")


def load_dataset(split_name: str, dataset_dir: str | Path = "dataset") -> pd.DataFrame:
    """
    Download the dataset from Hugging Face Hub or load it from disk if it is already downloaded.

    Args:
        split_name: name of the split to load ('train' or 'validation')
        dataset_dir: directory name for the dataset
    Returns:
        An instance of pd.DataFrame with the data.
    """
    splits = {'train': 'train.csv', 'validation': 'validation_input.csv'}
    csv_path = os.path.join(dataset_dir, splits[split_name])
    download_dataset(dataset_dir, csv_path)
    print(f"Reading '{split_name}' split from {csv_path}.")
    return pd.read_csv(csv_path)


def load_metadata(dataset_dir: str | Path = "dataset") -> dict:
    """
    Download the dataset metadata from Hugging Face Hub or load it from disk if it is already downloaded.

    Args:
        dataset_dir: directory name for the dataset
    Returns:
        An dict with the dataset metadata.
    """
    metadata_path = os.path.join(dataset_dir, "metadata.json")
    download_dataset(dataset_dir, metadata_path)
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    print(f"Reading dataset metadata from {metadata_path}.")
    return metadata


@dataclass(frozen=True)
class Schema:
    """
    Defines dataset schema.
    """
    series_id_column: str
    """Column with the id of the individual series"""
    target_column: str
    """Column with the prediction target"""
    feature_columns: list[str]
    """Columns with features"""

    n_series: int
    """Number of series in the dataset"""
    n_training_steps: int
    """Number of training steps in each series"""

    validation_horizon: int

    def get_series_ids(self, df: pd.DataFrame) -> list[str]:
        """Return a sorted list of unique series ids in the given dataset"""
        return sorted(df[self.series_id_column].unique())

    def get_series_groups(self, df: pd.DataFrame) -> DataFrameGroupBy:
        return df.groupby(self.series_id_column)

    @staticmethod
    def from_metadata(metadata: dict) -> 'Schema':
        schema = metadata["schema"]
        target = metadata["target_column"]
        labels = schema["labels"]
        feature_keys: list[str] = schema["train"].copy()
        for label in labels:
            feature_keys.remove(label)
        n_series = metadata["n_series"]
        validation_horizon = metadata["validation_horizon"]
        test_horizon = metadata["test_horizon"]
        n_training_steps = metadata["n_steps"] - validation_horizon - test_horizon
        return Schema("series_id", target, feature_keys, n_series, n_training_steps,
                      validation_horizon=validation_horizon)


def load_schema(dataset_dir: str | Path = "dataset") -> Schema:
    """Load dataset schema from the given directory"""
    metadata = load_metadata(dataset_dir)
    return Schema.from_metadata(metadata)


class ForecastSample(TypedDict):
    """Sample of the timeseries dataset"""
    x: Tensor
    """Historical values of shape (context_size,)"""
    y: Tensor
    """Future value to predict of shape (prediction_horizon,) or (context_size,)"""
    x_features: Tensor
    """Historical features of shape (context_size, num_features)"""
    y_features: Tensor
    """Future features of shape (prediction_horizon, num_features) or (context_size, num_features)"""


class ForecastDataset(Dataset[ForecastSample]):
    """
    Forecasting dataset with input of `context_size` past values predicting `prediction_horizon` future values.
    """

    def __init__(self, df: pd.DataFrame, schema: Schema, context_size: int, prediction_horizon: int = 1,
                 is_shifted_output: bool = False):
        """
        Create the ForecastingDataset.

        Args:
            df: dataset
            schema: schema of the dataset
            context_size: number of past values to use for the input
            prediction_horizon: number of future values to predict
            is_shifted_output: if the output should only contain the future values,
                               or if it should be the input shifted by the prediction horizon.
        """
        assert context_size >= 0, f"Negative context_size value {context_size}"
        assert prediction_horizon > 0, f"Non-positive prediction_horizon value {prediction_horizon}"

        self.schema = schema
        self.data = df

        # TODO ensure that each group is sorted by timestamp
        self.series_groups = self.schema.get_series_groups(df)
        self.series_ids = self.schema.get_series_ids(df)

        # TODO preprocess dataset to deal with nan values in some of the columns

        self.context_size = context_size
        self.prediction_horizon = prediction_horizon
        self.is_shifted_output = is_shifted_output

    @property
    def n_chunks(self) -> int:
        """Number of chunks we can split each series into"""
        return self.schema.n_training_steps - self.context_size - self.prediction_horizon + 1

    def __len__(self) -> int:
        return self.schema.n_series * self.n_chunks

    def __getitem__(self, index: int) -> ForecastSample:
        series_idx = index // self.n_chunks
        series_id = self.series_ids[series_idx]
        series_df = self.series_groups.get_group(series_id)

        input_idx_start = index % self.n_chunks
        input_idx_end = input_idx_start + self.context_size

        xs = series_df.iloc[input_idx_start:input_idx_end][self.schema.target_column].to_numpy()
        x_features = series_df.iloc[input_idx_start:input_idx_end][self.schema.feature_columns].to_numpy()

        output_idx_end = input_idx_start + self.context_size + self.prediction_horizon
        if self.is_shifted_output:
            # output is the context window shifted by prediction horizon
            output_idx_start = input_idx_start + self.prediction_horizon
        else:
            # output contains only the future predictions
            output_idx_start = input_idx_start + self.context_size
        ys = series_df.iloc[output_idx_start:output_idx_end][self.schema.target_column].to_numpy()
        y_features = series_df.iloc[output_idx_start:output_idx_end][self.schema.feature_columns].to_numpy()

        return ForecastSample(x=Tensor(xs), y=Tensor(ys),
                              x_features=Tensor(x_features), y_features=Tensor(y_features), )
