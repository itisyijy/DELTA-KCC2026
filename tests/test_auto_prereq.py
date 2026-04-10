from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest
import torch

from scripts.checkpoints import make_model, resolve_or_build_prereq_checkpoint
from scripts.config import load_config
from scripts.run import BASELINE_RUNNERS, main
from scripts.utils.tools import (
    build_checkpoint_metadata,
    checkpoint_metadata_path,
    resolve_checkpoint_path,
)


def _write(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(body).strip() + "\n")


def _fed_case(tmp_path: Path, checkpoint_path: Path | None = None) -> tuple[Path, object, object, Path]:
    config_dir = tmp_path / "configs" / "toy"
    common = textwrap.dedent(
        f"""
        dataset: toy
        data_path: {tmp_path / "toy.csv"}
        data_format: csv
        timestamp_col: date
        model:
          seq_len: 4
          pred_len: 2
          kernel_size: 3
          individual: false
          revin_affine: true
        batch_size: 8
        lr: 0.001
        device: cpu
        seed: 0
        output_dir: {tmp_path / "runs"}
        checkpoint_dir: {tmp_path / "checkpoints"}
        """
    ).strip()
    _write(
        config_dir / "fed.yaml",
        "\n".join(
            [
                "baseline: fed",
                common,
                "local_epochs: 1",
                "global_rounds: 1",
            ]
        ),
    )
    tta_path = config_dir / "fed_tta.yaml"
    tta_lines = ["baseline: fed_tta", common]
    if checkpoint_path:
        tta_lines.append(f"checkpoint_path: {checkpoint_path}")
    tta_lines.extend(
        [
            "tta:",
            "  k_ratio: 0.5",
            "  alpha: 1.0",
            "  lambda0: 1.0",
            "  gamma: 1.0",
            "  lr: 0.001",
            "  grad_clip: 1.0",
            "  rollback_threshold: 3.0",
            "  rollback_window: 2",
        ]
    )
    _write(
        tta_path,
        "\n".join(tta_lines),
    )
    tta_config = load_config(tta_path)
    prereq_config = load_config(tta_path.with_name("fed.yaml"))
    default_checkpoint = resolve_checkpoint_path(
        None,
        prereq_config.checkpoint_dir,
        prereq_config.dataset,
        "fed",
        metadata=build_checkpoint_metadata(prereq_config, "fed"),
    )
    return tta_path, tta_config, prereq_config, default_checkpoint


def _save_checkpoint(config, checkpoint_path: Path, with_metadata: bool = True) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(make_model(config, torch.device("cpu")).state_dict(), checkpoint_path)
    if with_metadata:
        checkpoint_metadata_path(checkpoint_path).write_text(
            json.dumps(build_checkpoint_metadata(config, "fed"), indent=2)
        )


def test_auto_prereq_builds_missing_custom_checkpoint(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    custom_checkpoint = tmp_path / "custom" / "fed_best.pt"
    tta_path, config, prereq_config, _ = _fed_case(tmp_path, checkpoint_path=custom_checkpoint)
    built: list[Path] = []
    run_dirs: list[Path] = []

    def fake_fed_runner(config, clients, device, run_dir, checkpoint_path_override=None) -> None:
        built.append(Path(checkpoint_path_override))
        run_dirs.append(Path(run_dir))
        print("fake prereq runner")
        _save_checkpoint(config, Path(checkpoint_path_override))

    monkeypatch.setitem(BASELINE_RUNNERS, "fed", fake_fed_runner)
    resolved = resolve_or_build_prereq_checkpoint(
        config, tta_path, [], torch.device("cpu"), auto_prereq=True, baseline_runners=BASELINE_RUNNERS
    )

    assert prereq_config.baseline == "fed"
    assert built == [custom_checkpoint]
    assert len(run_dirs) == 1
    assert run_dirs[0].parent == tmp_path / "runs"
    assert (run_dirs[0] / "run.log").exists()
    assert "fake prereq runner" in (run_dirs[0] / "run.log").read_text()
    assert resolved.checkpoint_path == str(custom_checkpoint)
    assert json.loads(checkpoint_metadata_path(custom_checkpoint).read_text())["source_baseline"] == "fed"


def test_auto_prereq_reuses_matching_checkpoint(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    tta_path, config, prereq_config, checkpoint_path = _fed_case(tmp_path)
    _save_checkpoint(prereq_config, checkpoint_path)
    rebuilt = False

    def fake_fed_runner(*args, **kwargs) -> None:
        nonlocal rebuilt
        rebuilt = True

    monkeypatch.setitem(BASELINE_RUNNERS, "fed", fake_fed_runner)
    resolved = resolve_or_build_prereq_checkpoint(
        config, tta_path, [], torch.device("cpu"), auto_prereq=True, baseline_runners=BASELINE_RUNNERS
    )

    assert resolved.checkpoint_path == str(checkpoint_path)
    assert rebuilt is False


def test_legacy_checkpoint_is_allowed_without_auto_prereq(tmp_path: Path) -> None:
    tta_path, config, prereq_config, checkpoint_path = _fed_case(tmp_path)
    _save_checkpoint(prereq_config, checkpoint_path, with_metadata=False)

    resolved = resolve_or_build_prereq_checkpoint(
        config, tta_path, [], torch.device("cpu"), auto_prereq=False, baseline_runners=BASELINE_RUNNERS
    )

    assert resolved.checkpoint_path == str(checkpoint_path)
    assert not checkpoint_metadata_path(checkpoint_path).exists()


def test_metadata_mismatch_fails_without_auto_prereq(tmp_path: Path) -> None:
    tta_path, config, prereq_config, checkpoint_path = _fed_case(tmp_path)
    _save_checkpoint(prereq_config, checkpoint_path)
    metadata_path = checkpoint_metadata_path(checkpoint_path)
    metadata = json.loads(metadata_path.read_text())
    metadata["model"]["seq_len"] = 5
    metadata_path.write_text(json.dumps(metadata, indent=2))

    with pytest.raises(ValueError, match="checkpoint metadata mismatch"):
        resolve_or_build_prereq_checkpoint(
            config, tta_path, [], torch.device("cpu"), auto_prereq=False, baseline_runners=BASELINE_RUNNERS
        )


def test_main_rejects_auto_prereq_for_non_tta(tmp_path: Path) -> None:
    config_path = tmp_path / "configs" / "toy" / "centralized.yaml"
    _write(
        config_path,
        f"""
        baseline: centralized
        dataset: toy
        data_path: {tmp_path / "toy.csv"}
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

    with pytest.raises(SystemExit):
        main(["--config", str(config_path), "--auto-prereq"])
