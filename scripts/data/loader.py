"""
Data loaders for the DLinear FED-TTA framework.

Two formats:
  load_csv_as_clients()     — CSV with timestamp column + N numeric columns
                               (electricity.csv, solar.csv)
  load_parquet_as_clients() — kcc2026 legacy parquet format
                               (manifest.json + clients/*.parquet)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .dataset import ClientData


def load_csv_as_clients(
    csv_path: str | Path,
    timestamp_col: str,
    seq_len: int,
    pred_len: int,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    max_clients: int | None = None,
) -> list[ClientData]:
    """
    Load a wide-format CSV (one column per client) as a list of ClientData.

    The Global StandardScaler is fit independently on each client's train split.
    Global scale stats (mean, std) are stored in ClientData for sMAPE inversion.

    Args:
        csv_path:      Path to CSV file.
        timestamp_col: Name of the datetime column to drop ('date' or 'LocalTime').
        seq_len:       Input window length (used only to validate data length).
        pred_len:      Prediction length (used only to validate data length).
        train_ratio:   Fraction of rows for training.
        val_ratio:     Fraction of rows for validation (test = 1 - train - val).
        max_clients:   If set, load only the first max_clients columns.
    """
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)
    df = df.drop(columns=[timestamp_col], errors="ignore")

    # Drop any remaining non-numeric columns
    df = df.select_dtypes(include=[np.number])

    col_names = list(df.columns)
    if max_clients is not None:
        col_names = col_names[:max_clients]

    n = len(df)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val

    min_len = seq_len + pred_len
    if n_train < min_len or n_val < min_len or n_test < min_len:
        raise ValueError(
            f"Dataset too short: n_train={n_train}, n_val={n_val}, n_test={n_test}, "
            f"but need at least seq_len+pred_len={min_len} per split."
        )

    split_indices = {
        "train": (0, n_train),
        "val":   (n_train, n_train + n_val),
        "test":  (n_train + n_val, n),
    }

    clients: list[ClientData] = []
    for col in col_names:
        series = df[col].values.astype(np.float32).reshape(-1, 1)  # [N, 1]

        # Fit scaler on train portion only
        scaler = StandardScaler()
        scaler.fit(series[:n_train])
        scaled = scaler.transform(series).astype(np.float32)

        clients.append(
            ClientData(
                client_id=str(col),
                values=scaled,
                split_indices=split_indices,
                global_mean=float(scaler.mean_[0]),
                global_std=float(scaler.scale_[0]),
            )
        )

    return clients


def load_parquet_as_clients(
    manifest_path: str | Path,
    clients_dir: str | Path,
    seq_len: int,
    pred_len: int,
    max_clients: int | None = None,
) -> list[ClientData]:
    """
    Load the kcc2026 legacy parquet format:
        manifest.json — contains per-client records with split_counts and scale stats
        clients/*.parquet — one parquet file per client, already globally scaled

    Expected manifest structure (per-client entry):
    {
        "client_id": "...",
        "status": "ready",
        "split_counts": {"train": int, "val": int, "test": int},
        "mean_full": float,   or "mean": float
        "std_full": float,    or "std": float
    }
    """
    manifest_path = Path(manifest_path)
    clients_dir = Path(clients_dir)

    with open(manifest_path) as f:
        manifest = json.load(f)

    # Support both list-of-records and dict-of-records manifest formats
    if isinstance(manifest, dict):
        records = list(manifest.values())
    else:
        records = manifest

    # Filter ready clients
    records = [r for r in records if r.get("status", "ready") == "ready"]
    if max_clients is not None:
        records = records[:max_clients]

    clients: list[ClientData] = []
    for rec in records:
        client_id = rec["client_id"]
        parquet_path = clients_dir / f"{client_id}.parquet"
        if not parquet_path.exists():
            continue

        df = pd.read_parquet(parquet_path)
        # Value column: first non-index numeric column
        value_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        series = df[value_cols[0]].values.astype(np.float32).reshape(-1, 1)

        # Split counts
        sc = rec.get("split_counts", {})
        n_train = int(sc.get("train", len(series) * 7 // 10))
        n_val = int(sc.get("val", len(series) // 10))
        n_test = len(series) - n_train - n_val

        if n_test <= 0:
            continue

        split_indices = {
            "train": (0, n_train),
            "val":   (n_train, n_train + n_val),
            "test":  (n_train + n_val, len(series)),
        }

        # Global scale stats (for inverse transform at sMAPE evaluation)
        mean = float(rec.get("mean_full", rec.get("mean", 0.0)))
        std = float(rec.get("std_full", rec.get("std", 1.0)))

        clients.append(
            ClientData(
                client_id=client_id,
                values=series,
                split_indices=split_indices,
                global_mean=mean,
                global_std=std,
            )
        )

    return clients
