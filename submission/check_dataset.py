"""A script to check dataset correctness."""
from torch.utils.data import DataLoader

from src.datasets import load_dataset, ForecastDataset, load_schema

if __name__ == "__main__":
    train_df = load_dataset("train")
    schema = load_schema()

    context_size = 239
    prediction_horizon = 5
    stride = 2
    n_chunks = 3
    n_train = context_size + prediction_horizon + stride * (n_chunks - 1)

    train_df = schema.get_series_groups(train_df).head(n_train).copy()

    for is_shifted_output in (False, True):
        train_dataset = ForecastDataset(train_df, schema,
                                        context_size=context_size, prediction_horizon=prediction_horizon,
                                        stride=stride,
                                        is_shifted_output=is_shifted_output)

        assert len(train_dataset) == n_chunks * schema.n_series, \
            f"Expected {n_chunks * schema.n_series} elements in the dataset, got {len(train_dataset)}"

        batch_size = n_chunks * 2
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        sample = next(iter(train_loader))
        assert sample['x'].shape == (batch_size, context_size)
        assert sample['x_features'].shape == (batch_size, context_size, len(schema.feature_columns))
        assert sample['y'].shape == (batch_size, context_size if is_shifted_output else prediction_horizon)
        assert sample['y_features'].shape == (batch_size, context_size if is_shifted_output else prediction_horizon,
                                              len(schema.feature_columns))
