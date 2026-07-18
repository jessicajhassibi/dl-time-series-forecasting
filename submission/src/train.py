"""Training functions."""
import os.path

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .datasets import ForecastDataset


def train_model(model: torch.nn.Module, dataset: ForecastDataset, log_dir: str,
                num_epochs: int = 1, batch_size: int = 128, validate_step: int = 500):
    """
    Train the model on the provided dataset.

    Args:
        model: model to train
        dataset: dataset to train on
        log_dir: directory for saving checkpoints and writing log files
        num_epochs: number of epochs
        batch_size: batch size
        validate_step: number of training steps between validations
    """
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [0.9, 0.1])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    criterion = torch.nn.MSELoss()
    writer = SummaryWriter(log_dir=log_dir)
    it = 0
    metrics = {}
    for epoch in range(num_epochs):
        num_batches = len(train_loader)
        inner_loop = tqdm(enumerate(train_loader), total=num_batches)
        for batch_idx, sample in inner_loop:
            optimizer.zero_grad()

            x = sample['x']
            y = sample['y']
            x_features = sample['x_features']
            y_features = sample['y_features']

            y_pred, y_features_pred = model(x, x_features)

            loss = criterion(y, y_pred)
            if y_features_pred is not None:
                loss += criterion(y_features, y_features_pred)
            loss.backward()

            optimizer.step()

            batch_loss = loss.item()
            inner_loop.set_postfix(loss=batch_loss)
            writer.add_scalar("batch_loss", batch_loss, global_step=it)
            if (it % validate_step == 0) or (it == num_batches - 1):
                model.eval()
                with torch.no_grad():
                    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
                    metrics = get_validation_metrics(model, val_loader)
                    for metric, value in metrics.items():
                        writer.add_scalar(metric, value, global_step=it)
                model.train()

            it += 1

        print(f"Validation metrics after epoch {epoch}:")
        print("\n".join([f"\t{k}: {v:.3f}" for k, v in metrics.items()]))

        torch.save(model.state_dict(), os.path.join(log_dir, f"checkpoint-{epoch}.pt"))


def get_validation_metrics(model: torch.nn.Module, dataloader: DataLoader) -> dict[str, float]:
    """
    Run prediction on the given dataset and compute accuracy metrics, such as WAPE.

    Args:
        model: model to use for prediction
        dataloader: loader for the dataset to use for prediction
    Returns:
        dictionary containing metrics names and metric values
    """
    sum_error = 0
    sum_values = 0
    for sample in tqdm(dataloader, desc="Computing validation metrics", position=1, leave=False):
        x = sample['x']
        y = sample['y']
        x_features = sample['x_features']
        y_pred, _ = model(x, x_features)

        sum_error = sum_error + torch.sum(torch.abs(y - y_pred)).item()
        sum_values = sum_values + torch.sum(torch.abs(y)).item()

    # TODO add additional metrics
    wape = sum_error / sum_values
    return dict(wape=wape)
