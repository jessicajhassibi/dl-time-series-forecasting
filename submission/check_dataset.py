from itertools import islice

from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets import load_dataset, SimpleDataset

if __name__ == "__main__":
    train_df, metadata = load_dataset("train")
    train_dataset = SimpleDataset(train_df, metadata, context_size=239, prediction_horizon=5)

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
            print(f"features shape: {sample['features'].shape}")
        loop.set_description(f"Batch index {batch_idx}")
