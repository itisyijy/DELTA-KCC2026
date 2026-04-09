"""Utility helpers: early stopping, reproducibility, result persistence."""
from __future__ import annotations

import json
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

CHECKPOINT_METADATA_SCHEMA_VERSION = 1


def checkpoint_metadata_path(checkpoint_path: str | Path) -> Path:
    """Return the sidecar metadata path for a checkpoint file."""
    checkpoint_path = Path(checkpoint_path)
    return checkpoint_path.with_name(f"{checkpoint_path.name}.meta.json")


def resolve_checkpoint_path(
    checkpoint_path_override: str | Path | None,
    checkpoint_dir: str | Path,
    dataset: str,
    baseline: str,
) -> Path:
    """Resolve a checkpoint path from an explicit override or the canonical default."""
    if checkpoint_path_override:
        return Path(checkpoint_path_override)
    return Path(checkpoint_dir) / f"{dataset}_{baseline}" / "best.pt"


def build_checkpoint_metadata(config: Any, source_baseline: str) -> dict[str, Any]:
    """Create strict checkpoint metadata for reuse validation."""
    metadata: dict[str, Any] = {
        "schema_version": CHECKPOINT_METADATA_SCHEMA_VERSION,
        "source_baseline": source_baseline,
        "dataset": config.dataset,
        "data": {
            "data_format": config.data_format,
            "data_path": config.data_path,
            "parquet_clients_dir": config.parquet_clients_dir,
            "timestamp_col": config.timestamp_col,
            "max_clients": config.max_clients,
        },
        "model": {
            "seq_len": config.model.seq_len,
            "pred_len": config.model.pred_len,
            "kernel_size": config.model.kernel_size,
            "individual": config.model.individual,
            "revin_affine": config.model.revin_affine,
        },
        "training": {
            "batch_size": config.batch_size,
            "lr": config.lr,
            "seed": config.seed,
        },
    }
    if source_baseline == "centralized":
        metadata["training"]["epochs"] = config.epochs
        metadata["training"]["patience"] = config.patience
    elif source_baseline == "fed":
        metadata["training"]["local_epochs"] = config.local_epochs
        metadata["training"]["global_rounds"] = config.global_rounds
    else:
        raise ValueError(f"Unsupported source_baseline for checkpoint metadata: {source_baseline!r}")
    return metadata


def write_checkpoint_metadata(checkpoint_path: str | Path, metadata: dict[str, Any]) -> Path:
    """Persist checkpoint sidecar metadata next to the checkpoint file."""
    metadata_path = checkpoint_metadata_path(checkpoint_path)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    return metadata_path


def load_checkpoint_metadata(checkpoint_path: str | Path) -> dict[str, Any] | None:
    """Load sidecar checkpoint metadata if present."""
    metadata_path = checkpoint_metadata_path(checkpoint_path)
    if not metadata_path.exists():
        return None
    with open(metadata_path) as f:
        return json.load(f)


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

    def __call__(self, val_loss: float, model: torch.nn.Module, checkpoint_path: Path) -> None:
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self._save(model, checkpoint_path, val_loss)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self._save(model, checkpoint_path, val_loss)
            self.counter = 0

    def _save(self, model: torch.nn.Module, checkpoint_path: Path, val_loss: float) -> None:
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), checkpoint_path)
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
