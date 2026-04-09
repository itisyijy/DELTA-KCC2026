"""Utility helpers: early stopping, reproducibility, result persistence."""
from __future__ import annotations

import json
import os
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch


class EarlyStopping:
    """
    Stop training when validation loss stops improving.

    Saves best checkpoint to path/checkpoint.pt.
    """

    def __init__(self, patience: int = 7, delta: float = 0.0, verbose: bool = True):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.counter = 0
        self.best_score: float | None = None
        self.early_stop = False
        self.best_val_loss = np.inf

    def __call__(self, val_loss: float, model: torch.nn.Module, checkpoint_dir: Path) -> None:
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self._save(model, checkpoint_dir, val_loss)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self._save(model, checkpoint_dir, val_loss)
            self.counter = 0

    def _save(self, model: torch.nn.Module, checkpoint_dir: Path, val_loss: float) -> None:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), checkpoint_dir / "best.pt")
        self.best_val_loss = val_loss
        if self.verbose:
            print(f"Checkpoint saved (val_loss={val_loss:.6f})")


def seed_everything(seed: int) -> None:
    """Set all relevant random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_run_dir(output_dir: str | Path, dataset: str, baseline: str) -> Path:
    """Create and return a timestamped run directory."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    pid = os.getpid()
    run_dir = Path(output_dir) / f"{ts}_{dataset}_{baseline}_pid{pid}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_results(run_dir: Path, metrics: dict, config_dict: dict | None = None) -> None:
    """Save metrics (and optionally config) as JSON to the run directory."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    if config_dict is not None:
        with open(run_dir / "config.json", "w") as f:
            json.dump(config_dict, f, indent=2)
    print(f"Results saved to {run_dir / 'metrics.json'}")
