from __future__ import annotations

import json
import textwrap
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.data.dataset import ClientData
from scripts.run_backbone_eval import main as run_backbone_eval_main
from scripts.tta.engine import evaluate_client
from scripts.utils.efficiency import RunEfficiency


class ZeroModel(torch.nn.Module):
    def __init__(self, pred_len: int):
        super().__init__()
        self.pred_len = pred_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros((x.shape[0], self.pred_len, x.shape[-1]), device=x.device)


def _client() -> ClientData:
    values = np.arange(20, dtype=np.float32).reshape(-1, 1)
    return ClientData(
        client_id="toy",
        values=values,
        split_indices={"train": (0, 8), "val": (8, 10), "test": (10, 20)},
        global_mean=0.0,
        global_std=1.0,
    )


def _write(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(body).strip() + "\n")


def test_run_efficiency_uses_rounds_6_to_20() -> None:
    tracker = RunEfficiency(torch.device("cpu"))
    tracker.start()
    for idx in range(25):
        tracker.record_round(idx + 1.0, (idx + 1.0) / 10.0)
    payload = tracker.payload(seed=7, trainable_params=12, total_backbone_params=120)

    assert payload["seed"] == 7
    assert payload["measured_round_start"] == 6
    assert payload["measured_round_end"] == 20
    assert payload["avg_round_sec"] == pytest.approx(sum(range(6, 21)) / 15.0)
    assert payload["avg_local_adapt_sec"] == pytest.approx(sum(i / 10.0 for i in range(6, 21)) / 15.0)
    assert payload["trainable_param_ratio"] == pytest.approx(0.1)


def test_evaluate_client_records_window_timings() -> None:
    tracker = RunEfficiency(torch.device("cpu"))
    tracker.start()
    metrics = evaluate_client(
        model=ZeroModel(pred_len=2),
        client=_client(),
        seq_len=4,
        pred_len=2,
        device=torch.device("cpu"),
        efficiency=tracker,
    )

    assert metrics["mse"] >= 0.0
    assert len(tracker.round_secs) == 5
    assert all(value >= 0.0 for value in tracker.round_secs)
    assert tracker.local_adapt_secs == [0.0] * 5


def test_run_backbone_eval_writes_efficiency_fields(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "configs" / "toy" / "fed_tta.yaml"
    _write(
        config_path,
        f"""
        baseline: fed_tta
        dataset: toy
        data_path: {tmp_path / "unused.csv"}
        data_format: csv
        timestamp_col: date
        checkpoint_path: {tmp_path / "dummy.pt"}
        model:
          seq_len: 4
          pred_len: 2
          kernel_size: 3
          individual: false
          revin_affine: true
        batch_size: 8
        device: cpu
        seed: 3
        output_dir: {tmp_path / "runs"}
        checkpoint_dir: {tmp_path / "checkpoints"}
        """,
    )

    monkeypatch.setattr("scripts.run_backbone_eval._load_clients", lambda config: [_client()])
    monkeypatch.setattr("scripts.run_backbone_eval.load_model", lambda config, device: ZeroModel(pred_len=2))
    run_backbone_eval_main(["--config", str(config_path)])

    run_dir = next((tmp_path / "runs").iterdir())
    payload = json.loads(next(run_dir.rglob("metrics.json")).read_text())
    assert payload["seed"] == 3
    assert payload["wall_clock_sec"] >= 0.0
    assert payload["avg_round_sec"] >= 0.0
    assert payload["avg_local_adapt_sec"] == 0.0
    assert payload["trainable_param_ratio"] == 0.0
