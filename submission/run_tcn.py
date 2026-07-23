"""Standalone TCN trainer: Apple GPU (MPS), Huber loss, cosine LR, best-checkpoint saving.
Run: python run_tcn.py --help
"""
import os
from datetime import datetime

import torch
import tyro
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

from src.config import Config, TrainConfig, write_config
from src.datasets import ForecastDataset, load_dataset, load_schema
from src.models.tcn_deep import TCNDeep


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def run(context_size: int = 504, prediction_horizon: int = 336,
        num_epochs: int = 10, seed: int = 42, batch_size: int = 256, lr: float = 1e-3,
        weight_decay: float = 1e-2, huber_delta: float = 1.0,
        hidden: int = 64, levels: int = 7, kernel_size: int = 3, dropout: float = 0.1):
    train_config = TrainConfig(seed=seed)
    train_config.set_seed()
    device = pick_device()
    print(f"Using device: {device}")

    df = load_dataset("train")
    schema = load_schema()

    model = TCNDeep(context_size, prediction_horizon, len(schema.feature_columns),
                    hidden=hidden, levels=levels, kernel_size=kernel_size, dropout=dropout).to(device)

    dataset = ForecastDataset(df, schema, context_size=context_size,
                              prediction_horizon=prediction_horizon, is_shifted_output=False)
    print(f"Loaded training dataset of length {len(dataset)}")

    gen = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(dataset, [0.9, 0.1], generator=gen)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.HuberLoss(delta=huber_delta)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs * len(train_loader))

    run_dir = datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + f"_{context_size}_{prediction_horizon}_{seed}"
    log_dir = os.path.join("logs", "tcn_deep", run_dir)
    os.makedirs(log_dir, exist_ok=True)
    print(f"Writing experiment data to {log_dir}")
    writer = SummaryWriter(log_dir=log_dir)

    model_config = dict(hidden=hidden, levels=levels, kernel_size=kernel_size, dropout=dropout)
    write_config(log_dir, Config("tcn_deep", context_size, prediction_horizon, model_config), train_config)

    global_step, best_wape = 0, float("inf")
    for epoch in range(num_epochs):
        model.train()
        for sample in train_loader:
            x, y = sample['x'].to(device), sample['y'].to(device)
            x_features = sample['x_features'].to(device)
            optimizer.zero_grad()
            y_pred, _ = model(x, x_features)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            scheduler.step()
            writer.add_scalar("Train/loss", loss.item(), global_step)
            writer.add_scalar("Train/lr", scheduler.get_last_lr()[0], global_step)
            global_step += 1

        model.eval()
        err, tot = 0.0, 0.0
        with torch.no_grad():
            for sample in val_loader:
                x, y = sample['x'].to(device), sample['y'].to(device)
                x_features = sample['x_features'].to(device)
                y_pred, _ = model(x, x_features)
                err += (y_pred - y).abs().sum().item()
                tot += y.abs().sum().item()
        val_wape = err / tot
        writer.add_scalar("Val/wape", val_wape, global_step)
        print(f"Epoch {epoch}: val single-shot WAPE = {val_wape:.4f}")

        ckpt = dict(epoch=epoch, global_step=global_step - 1,
                    state_dict=model.state_dict(), optimizer_state_dict=optimizer.state_dict())
        torch.save(ckpt, os.path.join(log_dir, "checkpoint-last.pt"))
        if val_wape < best_wape:
            best_wape = val_wape
            torch.save(ckpt, os.path.join(log_dir, "checkpoint-best.pt"))
            print(f"  new best WAPE {best_wape:.4f} -> saved checkpoint-best.pt")

    writer.close()
    print(f"Done. Best single-shot val WAPE: {best_wape:.4f}\nCheckpoints in: {log_dir}")


if __name__ == "__main__":
    tyro.cli(run)