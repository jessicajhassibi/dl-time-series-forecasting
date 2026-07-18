"""Training functions."""
import os.path

import torch
from torch.nn import Module
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .datasets import ForecastSample


def train_model(model: Module, train_dataset: Dataset[ForecastSample], val_dataset: Dataset[ForecastSample] | None,
                log_dir: str, num_epochs: int = 1, batch_size: int = 128, validate_step: int = 500):
    """
    Train the model on the provided dataset.

    Args:
        model: model to train
        train_dataset: dataset to train on
        val_dataset: dataset to use for validation during training, pass None to skip validation
        log_dir: directory for saving checkpoints and writing log files
        num_epochs: number of epochs
        batch_size: batch size
        validate_step: number of training steps between validations, negative value means to skip validation
    """
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    criterion = torch.nn.MSELoss()
    writer = SummaryWriter(log_dir=log_dir)
    global_step = 0

    metrics = {}
    is_validation_enabled = validate_step > 0 and (val_dataset is not None)

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
            writer.add_scalar("Train/batch_loss", batch_loss, global_step=global_step)

            is_validation_step = (global_step % validate_step == 0) and (global_step > 0)
            is_last_batch = batch_idx == num_batches - 1
            if is_validation_enabled and (is_validation_step or is_last_batch):
                metrics = get_validation_metrics(model, val_dataset)
                for metric, value in metrics.items():
                    writer.add_scalar(f"Metrics/{metric}", value, global_step=global_step)
                torch.save(model.state_dict(), os.path.join(log_dir, f"checkpoint-{global_step}.pt"))

            global_step += 1

        if is_validation_enabled:
            print(f"Validation metrics after epoch {epoch}:")
            print("\n".join([f"\t{k}: {v:.3f}" for k, v in metrics.items()]))

        if not is_validation_enabled:
            torch.save(model.state_dict(), os.path.join(log_dir, f"checkpoint-{global_step}.pt"))


def get_validation_metrics(model: Module, dataset: Dataset[ForecastSample],
                           batch_size: int = 500) -> dict[str, float | int]:
    """
    Run prediction on the given dataset and compute accuracy metrics, such as WAPE.

    Args:
        model: model to use for prediction
        dataset: loader for the dataset to use for prediction
        batch_size: batch size to use
    Returns:
        dictionary containing metrics names and metric values
    """
    training_mode = model.training
    model.eval()

    with torch.no_grad():
        val_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        sum_error = 0
        sum_values = 0
        for sample in tqdm(val_loader, desc="Computing validation metrics", position=1, leave=False):
            x = sample['x']
            y = sample['y']
            x_features = sample['x_features']
            y_pred, _ = model(x, x_features)

            sum_error = sum_error + torch.sum(torch.abs(y - y_pred)).item()
            sum_values = sum_values + torch.sum(torch.abs(y)).item()
        # TODO add additional metrics
        wape = sum_error / sum_values

    model.train(mode=training_mode)

    return dict(wape=wape)
