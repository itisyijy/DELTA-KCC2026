from __future__ import annotations

import argparse
import dataclasses

from scripts.checkpoints import TTA_PREREQ_BASELINES
from scripts.config import ExperimentConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="DLinear FED-TTA — unified experiment runner"
    )
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--device", default=None, help="Override device (e.g. cpu, cuda:1)")
    parser.add_argument("--checkpoint-path", default=None, dest="checkpoint_path")
    parser.add_argument("--output-dir", default=None, dest="output_dir")
    parser.add_argument("--checkpoint-dir", default=None, dest="checkpoint_dir")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--global-rounds", type=int, default=None, dest="global_rounds")
    parser.add_argument("--local-epochs", type=int, default=None, dest="local_epochs")
    parser.add_argument("--max-clients", type=int, default=None, dest="max_clients")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--auto-prereq", action="store_true")
    return parser


def apply_cli_overrides(
    config: ExperimentConfig,
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> ExperimentConfig:
    for field_name in (
        "device",
        "checkpoint_path",
        "output_dir",
        "checkpoint_dir",
        "epochs",
        "global_rounds",
        "local_epochs",
        "max_clients",
        "seed",
    ):
        value = getattr(args, field_name)
        if value is not None:
            config = dataclasses.replace(config, **{field_name: value})
    if args.auto_prereq and config.baseline not in TTA_PREREQ_BASELINES:
        parser.error("--auto-prereq is only supported for dlinear_tta, fed_tta, and fed_tta_loop")
    return config
