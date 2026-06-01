import json
import os

import pandas as pd
from huggingface_hub import snapshot_download


def load_dataset(split_name: str, dataset_dir: str = "dataset") -> tuple[pd.DataFrame, dict]:
    """
    Download the dataset from Hugging Face Hub or load it from disk if it is already downloaded.

    Args:
        split_name: name of the split to load ('train' or 'validation')
        dataset_dir: directory name for the dataset
    Returns:
        A tuple of the pd.DataFrame with the data and metadata dictionary.
    """
    splits = {'train': 'train.csv', 'validation': 'validation_input.csv'}
    csv_path = os.path.join(dataset_dir, splits[split_name])
    metadata_path = os.path.join(dataset_dir, "metadata.json")
    if (not os.path.exists(csv_path)) or (not os.path.exists(metadata_path)):
        snapshot_download(repo_id="AIML-TUDA/dlam-ts-project-data-2026", repo_type="dataset",
                          local_dir=dataset_dir)

    print(f"Reading '{split_name}' split from {csv_path} and metadata from {metadata_path}")
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return pd.read_csv(csv_path), metadata
