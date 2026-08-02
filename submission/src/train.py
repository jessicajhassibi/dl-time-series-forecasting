"""Training functions."""
import os.path
from pathlib import Path

import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .config import TrainConfig
from .datasets import ForecastSample
from .validation import get_validation_metrics


def train_model(model: Module, train_dataset: Dataset[ForecastSample], val_dataset: Dataset[ForecastSample] | None,
                train_config: TrainConfig, log_dir: str, num_epochs: int = 1,
                validate_step: int = 500, val_metric_name: str = "WAPE",
                checkpoint: Path | None = None, device: str | torch.device = "cpu"):
    """
    Train the model on the provided dataset.

    Args:
        model: model to train
        train_dataset: dataset to train on
        val_dataset: dataset to use for validation during training, pass None to skip validation
        train_config: training configuration parameters
        log_dir: directory for saving checkpoints and writing log files
        num_epochs: number of epochs
        validate_step: number of training steps between validations,
                       negative values means only validate after each epoch if validation dataset is provided
        val_metric_name: validation metrics to use
        checkpoint: checkpoint location to resume training
        device: device to use
    """
    train_loader = DataLoader(train_dataset, batch_size=train_config.batch_size,
                              shuffle=True, num_workers=0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.lr, weight_decay=train_config.weight_decay)
    criterion = torch.nn.HuberLoss(delta=train_config.huber_delta)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs * len(train_loader))

    start_epoch = 0
    global_step = 0

    if checkpoint is not None:
        global_step, start_epoch = load_model(checkpoint, model, optimizer, scheduler, device)
        print(f"Loaded model and optimizer weights from checkpoint {checkpoint}.\n"
              f"Global step: {global_step}, start epoch: {start_epoch}.")
    model.train()

    is_validation_enabled = val_dataset is not None
    best_metric_value = float("inf")
    best_checkpoint_name = "checkpoint-best.pt"

    with SummaryWriter(log_dir=log_dir) as writer:
        for epoch in range(start_epoch, start_epoch + num_epochs):
            metrics = {}

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
                scheduler.step()

                batch_loss = loss.item()
                inner_loop.set_postfix(loss=batch_loss)
                writer.add_scalar("Train/batch_loss", batch_loss, global_step=global_step)
                writer.add_scalar("Train/learning_rate", scheduler.get_last_lr()[0], global_step)

                is_validation_step = (validate_step > 0) and (global_step % validate_step == 0) and (global_step > 0)
                is_last_batch = batch_idx == num_batches - 1
                if is_validation_enabled and (is_validation_step or is_last_batch):
                    metrics = get_validation_metrics(model, val_dataset)
                    for metric, value in metrics.items():
                        writer.add_scalar(f"Val/{metric}", value, global_step=global_step)
                    save_model(model, optimizer, scheduler, epoch, global_step, log_dir)

                    val_metric_value = metrics[val_metric_name]
                    if val_metric_value < best_metric_value:
                        best_metric_value = val_metric_value
                        save_model(model, optimizer, scheduler, epoch, global_step, log_dir,
                                   checkpoint_name=best_checkpoint_name)
                        # TODO copy already saved checkpoint

                global_step += 1

            if is_validation_enabled:
                print(f"Epoch {epoch}: Best {val_metric_name}: {best_metric_value:.4f}, " +
                           ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()]))
            else:
                save_model(model, optimizer, scheduler, epoch, global_step - 1, log_dir)

    if is_validation_enabled:
        print(f"Best {val_metric_name}: {best_metric_value:.4f}, "
              f"best checkpoint at {os.path.join(log_dir, best_checkpoint_name)}")


def load_model(checkpoint: Path, model: Module, optimizer: Optimizer | None = None,
               scheduler: LRScheduler | None = None,
               device: str | torch.device = "cpu") -> tuple[int, int]:
    checkpoint_dict: dict = torch.load(checkpoint, map_location=device)
    model.load_state_dict(checkpoint_dict['state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint_dict['optimizer_state_dict'])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint_dict['scheduler_state_dict'])
    start_epoch = checkpoint_dict.get('epoch', -1) + 1
    global_step = checkpoint_dict.get('global_step', -1) + 1
    # TODO this does not start training on the exactly same location
    #  if training was stopped in the middle of the epoch
    return global_step, start_epoch


def save_model(model: Module, optimizer: Optimizer, scheduler: LRScheduler,
               epoch: int, global_step: int, log_dir: str,
               checkpoint_name: str | None = None):
    # TODO we can save model config to checkpoint as well
    if checkpoint_name is None:
        checkpoint_name = f"checkpoint-{global_step}.pt"
    torch.save(dict(epoch=epoch, global_step=global_step,
                    state_dict=model.state_dict(), optimizer_state_dict=optimizer.state_dict(),
                    scheduler_state_dict=scheduler.state_dict()),
               os.path.join(log_dir, checkpoint_name))
