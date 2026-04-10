from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from scripts.run import BASELINE_RUNNERS, main
from scripts.utils.run_logging import MilestoneLogger


def _write(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(body).strip() + "\n")


def test_main_writes_run_log_and_keeps_console(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = tmp_path / "configs" / "toy" / "centralized.yaml"
    _write(
        config_path,
        f"""
        baseline: centralized
        dataset: toy
        data_path: {tmp_path / "unused.csv"}
        data_format: csv
        timestamp_col: date
        model:
          seq_len: 4
          pred_len: 2
          kernel_size: 3
          individual: false
          revin_affine: true
        batch_size: 8
        epochs: 1
        lr: 0.001
        patience: 1
        device: cpu
        seed: 0
        output_dir: {tmp_path / "runs"}
        checkpoint_dir: {tmp_path / "checkpoints"}
        """,
    )

    monkeypatch.setattr("scripts.run._load_clients", lambda config: [])

    def fake_runner(config, clients, device, run_dir, checkpoint_path_override=None) -> None:
        print("runner stdout")

    monkeypatch.setitem(BASELINE_RUNNERS, "centralized", fake_runner)
    main(["--config", str(config_path)])

    captured = capsys.readouterr()
    run_dirs = list((tmp_path / "runs").iterdir())
    assert len(run_dirs) == 1
    run_log = run_dirs[0] / "run.log"
    assert run_log.exists()
    contents = run_log.read_text()
    assert "run_dir=" in contents
    assert "run_log=" in contents
    assert "runner stdout" in contents
    assert "runner stdout" in captured.out


def test_milestone_logger_emits_each_threshold_once(
    capsys: pytest.CaptureFixture[str],
) -> None:
    progress = MilestoneLogger("TTA", 10)
    progress.start(detail="clients=2")
    for step in (1, 1, 2, 5, 5, 10):
        progress.update(step)
    progress.finish(detail="clients=2")

    out = capsys.readouterr().out
    assert out.count("Progress 10%") == 1
    assert out.count("Progress 20%") == 1
    assert out.count("Progress 50%") == 1
    assert out.count("Progress 100%") == 1
    assert out.count("Complete") == 1
