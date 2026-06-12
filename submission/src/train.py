import os.path

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .datasets import ForecastDataset


def train_model(model: torch.nn.Module, dataset: ForecastDataset, log_dir: str,
                num_epochs: int = 1, batch_size: int = 128):
    """
    Train the model on the provided dataset.

    Args:
        model: model to train
        dataset: dataset to train on
        log_dir: directory for saving checkpoints and writing log files
        num_epochs: number of epochs
        batch_size: batch size
    """
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    # TODO split by train val, evaluate accuracy during training

    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    criterion = torch.nn.MSELoss()
    writer = SummaryWriter(log_dir=log_dir)
    it = 0
    for epoch in range(num_epochs):
        inner_loop = tqdm(enumerate(data_loader), total=len(data_loader))
        for batch_idx, sample in inner_loop:
            optimizer.zero_grad()

            x = sample['x']
            y = sample['y']
            x_features = sample['x_features']
            y_features = sample['y_features']

            y_pred, y_features_pred = model(x, x_features)

            loss = criterion(y, y_pred)
            if y_features_pred:
                loss += criterion(y_features, y_features_pred)
            loss.backward()

            optimizer.step()

            batch_loss = loss.item()
            inner_loop.set_postfix(loss=batch_loss)
            writer.add_scalar("batch_loss", batch_loss, global_step=it)
            it += 1

        torch.save(model.state_dict(), os.path.join(log_dir, f"checkpoint-{epoch}.pt"))
