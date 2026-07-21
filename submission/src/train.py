"""Training functions."""
import os.path
from pathlib import Path

import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .datasets import ForecastSample
from .validation import get_validation_metrics


def train_model(model: Module, train_dataset: Dataset[ForecastSample], val_dataset: Dataset[ForecastSample] | None,
                log_dir: str, num_epochs: int = 1, batch_size: int = 128, validate_step: int = 500,
                checkpoint: Path | None = None):
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
        checkpoint: checkpoint location to resume training
    """
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    criterion = torch.nn.MSELoss()
    start_epoch = 0
    global_step = 0

    if checkpoint is not None:
        global_step, start_epoch = load_model(checkpoint, model, optimizer)
        print(f"Loaded model and optimizer weights from checkpoint {checkpoint}.\n"
              f"Global step: {global_step}, start epoch: {start_epoch}.")

    model.train()

    writer = SummaryWriter(log_dir=log_dir)
    metrics = {}
    is_validation_enabled = validate_step > 0 and (val_dataset is not None)

    for epoch in range(start_epoch, start_epoch + num_epochs):
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
                save_model(model, optimizer, epoch, global_step, log_dir)

            global_step += 1

        if is_validation_enabled:
            print(f"Validation metrics after epoch {epoch}:")
            print("\n".join([f"\t{k}: {v:.3f}" for k, v in metrics.items()]))

        if not is_validation_enabled:
            save_model(model, optimizer, epoch, global_step - 1, log_dir)


def load_model(checkpoint: Path, model: Module, optimizer: Optimizer | None = None,
               device: str | None = None) -> tuple[int, int]:
    checkpoint_dict: dict = torch.load(checkpoint, map_location=device)
    model.load_state_dict(checkpoint_dict['state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint_dict['optimizer_state_dict'])
    start_epoch = checkpoint_dict.get('epoch', -1) + 1
    global_step = checkpoint_dict.get('global_step', -1) + 1
    # TODO this does not start training on the exactly same location
    #  if training was stopped in the middle of the epoch
    return global_step, start_epoch


def save_model(model: Module, optimizer: Optimizer, epoch: int, global_step: int, log_dir: str):
    # TODO we can save model config to checkpoint as well
    torch.save(dict(epoch=epoch, global_step=global_step,
                    state_dict=model.state_dict(), optimizer_state_dict=optimizer.state_dict()),
               os.path.join(log_dir, f"checkpoint-{global_step}.pt"))
