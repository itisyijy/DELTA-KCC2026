from __future__ import annotations

import dataclasses
from pathlib import Path

import torch

from scripts.config import ExperimentConfig, load_config
from scripts.data.dataset import ClientData
from scripts.models.revin_dlinear import RevINDLinear
from scripts.utils.tools import (
    build_checkpoint_metadata,
    load_checkpoint_metadata,
    make_run_dir,
    resolve_checkpoint_path,
)

TTA_PREREQ_BASELINES = {
    "dlinear_tta": "centralized",
    "fed_tta": "fed",
    "fed_tta_loop": "fed",
}


def make_model(config: ExperimentConfig, device: torch.device) -> RevINDLinear:
    """Instantiate a RevIN-DLinear model for the given config."""
    return RevINDLinear(
        seq_len=config.model.seq_len,
        pred_len=config.model.pred_len,
        channels=1,
        kernel_size=config.model.kernel_size,
        individual=config.model.individual,
        revin_affine=config.model.revin_affine,
    ).to(device)


def load_model(config: ExperimentConfig, device: torch.device) -> RevINDLinear:
    """Load a pre-trained model from checkpoint_path."""
    if not config.checkpoint_path:
        raise ValueError("checkpoint_path must be set for TTA baselines.")
    model = make_model(config, device)
    state = torch.load(config.checkpoint_path, map_location=device)
    model.load_state_dict(state, strict=True)
    return model


def _shared_prereq_settings(config: ExperimentConfig) -> tuple:
    return (
        config.dataset,
        config.data_format,
        config.data_path,
        config.parquet_clients_dir,
        config.timestamp_col,
        config.max_clients,
        config.model.seq_len,
        config.model.pred_len,
        config.model.kernel_size,
        config.model.individual,
        config.model.revin_affine,
    )


def resolve_or_build_prereq_checkpoint(
    config: ExperimentConfig,
    config_path: str | Path,
    clients: list[ClientData],
    device: torch.device,
    auto_prereq: bool,
    baseline_runners: dict,
) -> ExperimentConfig:
    """Resolve, validate, and optionally rebuild a prerequisite checkpoint for TTA baselines."""
    prereq_baseline = TTA_PREREQ_BASELINES.get(config.baseline)
    if prereq_baseline is None:
        return config

    prereq_config_path = Path(config_path).with_name(f"{prereq_baseline}.yaml")
    if not prereq_config_path.exists():
        raise FileNotFoundError(
            f"Missing prerequisite config for baseline {config.baseline!r}: {prereq_config_path}"
        )
    prereq_config = dataclasses.replace(
        load_config(prereq_config_path),
        max_clients=config.max_clients,
        output_dir=config.output_dir,
        checkpoint_dir=config.checkpoint_dir,
        device=config.device,
        seed=config.seed,
    )
    if _shared_prereq_settings(config) != _shared_prereq_settings(prereq_config):
        raise ValueError(
            f"{config.baseline!r} requires {prereq_baseline!r} to use matching dataset/model settings."
        )

    expected_metadata = build_checkpoint_metadata(prereq_config, prereq_baseline)
    checkpoint_path = resolve_checkpoint_path(
        config.checkpoint_path or None,
        config.checkpoint_dir,
        config.dataset,
        prereq_baseline,
        metadata=expected_metadata,
    )

    def checkpoint_error(allow_legacy: bool) -> str | None:
        if not checkpoint_path.exists():
            return f"checkpoint missing: {checkpoint_path}"
        try:
            metadata = load_checkpoint_metadata(checkpoint_path)
        except Exception as exc:
            return f"checkpoint metadata unreadable: {exc}"
        if metadata is None and not allow_legacy:
            return "checkpoint metadata missing"
        if metadata is not None and metadata != expected_metadata:
            return "checkpoint metadata mismatch"
        try:
            model = make_model(prereq_config, device)
            state = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(state, strict=True)
        except Exception as exc:
            return f"checkpoint load validation failed: {exc}"
        return None

    error = checkpoint_error(allow_legacy=not auto_prereq)
    if error is not None:
        if not auto_prereq:
            raise ValueError(
                f"Invalid checkpoint for baseline {config.baseline!r} at {checkpoint_path}: {error}"
            )
        print(f"[AutoPrereq] Rebuilding {prereq_baseline} checkpoint at {checkpoint_path}")
        print(f"  - {error}")
        prereq_run_dir = make_run_dir(config.output_dir, config.dataset, prereq_baseline)
        baseline_runners[prereq_baseline](
            prereq_config,
            clients,
            device,
            prereq_run_dir,
            checkpoint_path_override=checkpoint_path,
        )
        error = checkpoint_error(allow_legacy=False)
        if error is not None:
            raise RuntimeError(
                f"Failed to prepare prerequisite checkpoint at {checkpoint_path}: {error}"
            )

    return dataclasses.replace(config, checkpoint_path=str(checkpoint_path))
