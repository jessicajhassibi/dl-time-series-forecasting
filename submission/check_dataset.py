"""A script to load a dataset, iterate over a few batches,
and print data shapes to see that the dataset is implemented correctly."""
from itertools import islice

from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets import load_dataset, ForecastDataset, load_schema

if __name__ == "__main__":
    train_df = load_dataset("train")
    schema = load_schema()

    for is_shifted_output in (False, True):
        train_dataset = ForecastDataset(train_df, schema, context_size=239, prediction_horizon=5,
                                        is_shifted_output=is_shifted_output)

        print(f"Is shifted output {is_shifted_output}")
        print(f"Length of the dataset is {len(train_dataset)}")

        train_loader = DataLoader(train_dataset, batch_size=100, shuffle=False)
        n_batches = train_dataset.n_chunks * 3 // 100  # select number of batches to cover multiple series
        print(f"Iterating over first {n_batches} batches")
        loop = tqdm(islice(enumerate(train_loader), n_batches))
        for batch_idx, sample in loop:
            if batch_idx == 0:
                print(f"Batch {batch_idx} sample:")
                print(f"x shape: {sample['x'].shape}")
                print(f"y shape: {sample['y'].shape}")
                print(f"x features shape: {sample['x_features'].shape}")
                print(f"y features shape: {sample['y_features'].shape}")
            loop.set_description(f"Batch index {batch_idx}")
