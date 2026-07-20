from src.datasets import load_dataset, load_schema
from src.util.plot import plot_series, partition_keys, plot_keys, plot_correlations, plot_distribution

if __name__ == "__main__":
    train_df = load_dataset("train")
    schema = load_schema()

    print(f"Dataset Schema:\n{schema}")

    variable_keys, constant_keys = partition_keys(train_df, schema.feature_columns)

    plot_series(train_df, title=f"Training Dataset Plot", x_key=schema.timestamp_column, y_key=schema.target_column)
    plot_keys(train_df, name="Train", series_id="unit_000", x_key=schema.timestamp_column,
              y_keys=[schema.target_column] + schema.feature_columns)
    plot_correlations(train_df, series_id="unit_000", y_keys=[schema.target_column] + variable_keys)
    plot_distribution(train_df, keys=constant_keys)
