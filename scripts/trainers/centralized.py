"""Baseline 1: Centralized DLinear training (upper bound)."""
from __future__ import annotations

import copy
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from scripts.config import ExperimentConfig
from scripts.data.dataset import ClientData, CentralizedDataset
from scripts.models.revin_dlinear import RevINDLinear
from scripts.utils.tools import EarlyStopping, make_run_dir, save_results


def _make_model(cfg: ExperimentConfig, channels: int) -> RevINDLinear:
    return RevINDLinear(
        seq_len=cfg.model.seq_len,
        pred_len=cfg.model.pred_len,
        channels=channels,
        kernel_size=cfg.model.kernel_size,
        individual=cfg.model.individual,
        revin_affine=cfg.model.revin_affine,
    )


def run_centralized(
    config: ExperimentConfig,
    clients: list[ClientData],
    device: torch.device | None = None,
) -> RevINDLinear:
    """
    Pool all clients into one dataset and train a single DLinear model.

    All clients are single-channel (shape [N, 1]), so channels=1.
    The model is shared across all clients via the centralized dataset.

    Saves best checkpoint to {checkpoint_dir}/{dataset}_centralized/best.pt.
    Returns the best model (loaded from checkpoint).
    """
    if device is None:
        device = torch.device(config.device if torch.cuda.is_available() else "cpu")

    model = _make_model(config, channels=1).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    train_ds = CentralizedDataset(clients, "train", config.model.seq_len, config.model.pred_len)
    val_ds = CentralizedDataset(clients, "val", config.model.seq_len, config.model.pred_len)

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, num_workers=0)

    ckpt_dir = Path(config.checkpoint_dir) / f"{config.dataset}_centralized"
    early_stop = EarlyStopping(patience=config.patience)

    for epoch in range(1, config.epochs + 1):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                val_loss += criterion(model(x), y).item()
        val_loss /= len(val_loader)

        print(f"[Centralized] Epoch {epoch:3d}/{config.epochs} | "
              f"train_loss={train_loss:.6f}  val_loss={val_loss:.6f}")

        early_stop(val_loss, model, ckpt_dir)
        if early_stop.early_stop:
            print(f"[Centralized] Early stopping at epoch {epoch}.")
            break

    # Load best weights
    model.load_state_dict(torch.load(ckpt_dir / "best.pt", map_location=device))
    return model
