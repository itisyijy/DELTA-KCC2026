from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scripts.data.loader import load_parquet_as_clients


def _write_parquet_client(path: Path) -> None:
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=6, freq="15min"),
            "split": ["train", "train", "train", "val", "test", "test"],
            "p": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )
    df.to_parquet(path)


def test_load_parquet_supports_murata_manifest_shape(tmp_path: Path) -> None:
    clients_dir = tmp_path / "clients"
    clients_dir.mkdir()
    _write_parquet_client(clients_dir / "client_a.parquet")
    manifest = {
        "channels": ["p"],
        "clients": [
            {
                "client_id": "client_a",
                "status": "ready",
                "split_counts": {"train": 3, "val": 1, "test": 2},
                "channel_stats": {"p": {"mean_full": 1.25, "std_full": 2.5}},
            }
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))

    clients = load_parquet_as_clients(manifest_path, clients_dir, seq_len=1, pred_len=1)

    assert len(clients) == 1
    assert clients[0].client_id == "client_a"
    assert clients[0].split_indices == {"train": (0, 3), "val": (3, 4), "test": (4, 6)}
    assert clients[0].global_mean == 1.0
    assert round(clients[0].global_std, 6) == round((2.0 / 3.0) ** 0.5, 6)
    assert round(float(clients[0].train_values().mean()), 6) == 0.0


def test_load_parquet_keeps_legacy_manifest_support(tmp_path: Path) -> None:
    clients_dir = tmp_path / "clients"
    clients_dir.mkdir()
    _write_parquet_client(clients_dir / "client_b.parquet")
    manifest = {
        "client_b": {
            "client_id": "client_b",
            "status": "ready",
            "split_counts": {"train": 3, "val": 1, "test": 2},
            "mean_full": 3.0,
            "std_full": 4.0,
        }
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))

    clients = load_parquet_as_clients(manifest_path, clients_dir, seq_len=1, pred_len=1)

    assert len(clients) == 1
    assert clients[0].client_id == "client_b"
    assert clients[0].global_mean == 1.0
    assert round(clients[0].global_std, 6) == round((2.0 / 3.0) ** 0.5, 6)
