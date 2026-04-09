"""Baseline 2: FedAvg Federated Learning training."""
from __future__ import annotations

import copy
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from scripts.config import ExperimentConfig
from scripts.data.dataset import ClientData, ClientDataset, CentralizedDataset
from scripts.models.revin_dlinear import RevINDLinear
from scripts.utils.tools import EarlyStopping, make_run_dir


def _make_model(cfg: ExperimentConfig) -> RevINDLinear:
    return RevINDLinear(
        seq_len=cfg.model.seq_len,
        pred_len=cfg.model.pred_len,
        channels=1,
        kernel_size=cfg.model.kernel_size,
        individual=cfg.model.individual,
        revin_affine=cfg.model.revin_affine,
    )


def run_local_epochs(
    model: RevINDLinear,
    client: ClientData,
    config: ExperimentConfig,
    device: torch.device,
) -> tuple[dict, int]:
    """
    Train model locally for config.local_epochs epochs on a single client.

    Returns:
        state_dict — updated model weights after local training
        n_windows  — number of training windows (used for weighted FedAvg)
    """
    local_model = copy.deepcopy(model).to(device)
    optimizer = torch.optim.Adam(local_model.parameters(), lr=config.lr)
    criterion = nn.MSELoss()

    ds = ClientDataset(client, "train", config.model.seq_len, config.model.pred_len)
    if len(ds) == 0:
        return local_model.state_dict(), 0

    loader = DataLoader(ds, batch_size=config.batch_size, shuffle=True, num_workers=0)

    local_model.train()
    for _ in range(config.local_epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(local_model(x), y)
            loss.backward()
            optimizer.step()

    return local_model.state_dict(), len(ds)


def fedavg_aggregate(
    global_model: RevINDLinear,
    local_results: list[tuple[dict, int]],
) -> None:
    """
    Weighted FedAvg: update global_model in-place.
    Weight for each client = number of training windows.
    """
    total_weight = sum(w for _, w in local_results if w > 0)
    if total_weight == 0:
        return

    new_state = copy.deepcopy(local_results[0][0])
    for key in new_state:
        new_state[key] = torch.zeros_like(new_state[key], dtype=torch.float32)

    for state_dict, weight in local_results:
        if weight == 0:
            continue
        for key in new_state:
            new_state[key] += state_dict[key].float() * (weight / total_weight)

    global_model.load_state_dict(new_state)


def run_fedavg(
    config: ExperimentConfig,
    clients: list[ClientData],
    device: torch.device | None = None,
) -> RevINDLinear:
    """
    Run FedAvg FL training for config.global_rounds rounds.

    Each round: local training for config.local_epochs epochs on each client,
    then FedAvg aggregation. Best global model saved by validation loss.
    """
    if device is None:
        device = torch.device(config.device if torch.cuda.is_available() else "cpu")

    global_model = _make_model(config).to(device)
    ckpt_dir = Path(config.checkpoint_dir) / f"{config.dataset}_fed"
    best_val_loss = float("inf")

    criterion = nn.MSELoss()
    val_ds = CentralizedDataset(clients, "val", config.model.seq_len, config.model.pred_len)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, num_workers=0)

    for rnd in range(1, config.global_rounds + 1):
        # Local training on all clients
        local_results = []
        for client in clients:
            state_dict, n_windows = run_local_epochs(global_model, client, config, device)
            local_results.append((state_dict, n_windows))

        # FedAvg aggregation
        fedavg_aggregate(global_model, local_results)

        # Validation on pooled val set
        global_model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                val_loss += criterion(global_model(x), y).item()
        val_loss /= max(len(val_loader), 1)

        print(f"[FedAvg] Round {rnd:3d}/{config.global_rounds} | val_loss={val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save(global_model.state_dict(), ckpt_dir / "best.pt")
            print(f"  → Best model saved (val_loss={val_loss:.6f})")

    # Load best weights
    global_model.load_state_dict(torch.load(ckpt_dir / "best.pt", map_location=device))
    return global_model
