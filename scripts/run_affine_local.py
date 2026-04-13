from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import numpy as np
import torch

from scripts.baselines import BASELINE_RUNNERS
from scripts.checkpoints import TTA_PREREQ_BASELINES, load_model, resolve_or_build_prereq_checkpoint
from scripts.cli import apply_cli_overrides, build_parser
from scripts.config import ExperimentConfig, load_config
from scripts.data.dataset import ClientData
from scripts.data.loader import load_csv_as_clients, load_parquet_as_clients
from scripts.tta.loop import run_local_tta_loop
from scripts.utils.run_logging import tee_run_output
from scripts.utils.tools import make_run_dir, save_results, seed_everything


def _load_clients(config: ExperimentConfig) -> list[ClientData]:
    if config.data_format == "csv":
        return load_csv_as_clients(
            csv_path=config.data_path,
            timestamp_col=config.timestamp_col,
            seq_len=config.model.seq_len,
            pred_len=config.model.pred_len,
            max_clients=config.max_clients,
        )
    if config.data_format == "parquet":
        clients_dir = config.parquet_clients_dir or str(Path(config.data_path).parent / "clients")
        return load_parquet_as_clients(
            manifest_path=config.data_path,
            clients_dir=clients_dir,
            seq_len=config.model.seq_len,
            pred_len=config.model.pred_len,
            max_clients=config.max_clients,
        )
    raise ValueError(f"Unknown data_format: {config.data_format!r}")


def _aggregate(per_client: dict[str, dict[str, float]]) -> dict[str, float]:
    keys = ("mse", "mae", "smape")
    return {
        key: float(np.mean([m[key] for m in per_client.values() if not np.isnan(m[key])]))
        for key in keys
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = build_parser()
    parser.description = "Affine-Adapter local TTA runner"
    parser.add_argument("--max-workers", type=int, default=4)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    config = apply_cli_overrides(load_config(args.config), args, parser)
    seed_everything(config.seed)
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    run_dir = make_run_dir(config.output_dir, config.dataset, f"{config.baseline}_affine")

    with tee_run_output(run_dir) as log_path:
        print(f"=== Affine Local TTA | baseline={config.baseline} | dataset={config.dataset} ===")
        print(f"  device={device}  seed={config.seed}  max_workers={args.max_workers}")
        print(f"  run_dir={run_dir}")
        print(f"  run_log={log_path}")

        clients = _load_clients(config)
        print(f"  Loaded {len(clients)} clients.")

        if config.baseline not in TTA_PREREQ_BASELINES:
            raise ValueError(
                f"Affine local TTA expects one of {sorted(TTA_PREREQ_BASELINES)}; "
                f"got {config.baseline!r}."
            )
        if config.checkpoint_path:
            print(f"  Using explicit checkpoint_path={config.checkpoint_path}")
        else:
            config = resolve_or_build_prereq_checkpoint(
                config=config,
                config_path=args.config,
                clients=clients,
                device=device,
                auto_prereq=args.auto_prereq,
                baseline_runners=BASELINE_RUNNERS,
            )

        per_client, per_client_diagnostics, diagnostic_summary = run_local_tta_loop(
            global_model=load_model(config, device),
            clients=clients,
            seq_len=config.model.seq_len,
            pred_len=config.model.pred_len,
            k=config.k(),
            tta_config=config.tta,
            device=device,
            max_workers=max(1, args.max_workers),
        )
        avg = _aggregate(per_client)
        print(
            f"\n[Affine Local TTA] Avg: MSE={avg['mse']:.4f}  "
            f"MAE={avg['mae']:.4f}  sMAPE={avg['smape']:.2f}%"
        )
        print(
            "[Affine Local TTA] Diagnostics: "
            f"adapt_rate={diagnostic_summary['adapt_rate']:.3f}  "
            f"accept_gate={diagnostic_summary['accept_gate_skip_rate']:.3f}  "
            f"hard_gate={diagnostic_summary['hard_gate_skip_rate']:.3f}  "
            f"drift_skip={diagnostic_summary['drift_skip_rate']:.3f}  "
            f"rollback_skip={diagnostic_summary['rollback_skip_rate']:.3f}  "
            f"reset/adapt={diagnostic_summary['reset_rate_given_adapt']:.3f}  "
            f"mean|gamma-1|={diagnostic_summary['mean_gamma_l1']:.6f}  "
            f"mean|delta|={diagnostic_summary['mean_delta_l1']:.6f}"
        )
        payload = {
            "engine": "affine_local_tta",
            "per_client": per_client,
            "per_client_diagnostics": per_client_diagnostics,
            "avg": avg,
            "diagnostic_summary": diagnostic_summary,
            "max_workers": max(1, args.max_workers),
        }
        save_results(run_dir, payload, dataclasses.asdict(config))


if __name__ == "__main__":
    main()
