from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import numpy as np
import torch

from scripts.checkpoints import load_model
from scripts.cli import apply_cli_overrides, build_parser
from scripts.config import ExperimentConfig, load_config
from scripts.data.dataset import ClientData
from scripts.data.loader import load_csv_as_clients, load_parquet_as_clients
from scripts.tta.engine import evaluate_client
from scripts.utils.efficiency import RunEfficiency, count_parameters
from scripts.utils.metrics import wasserstein_noniid
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
    clients_dir = config.parquet_clients_dir or str(Path(config.data_path).parent / "clients")
    return load_parquet_as_clients(
        manifest_path=config.data_path,
        clients_dir=clients_dir,
        seq_len=config.model.seq_len,
        pred_len=config.model.pred_len,
        max_clients=config.max_clients,
    )


def _aggregate(per_client: dict[str, dict[str, float]]) -> dict[str, float]:
    keys = ("mse", "mae", "smape")
    return {
        key: float(np.mean([m[key] for m in per_client.values() if not np.isnan(m[key])]))
        for key in keys
    }


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    parser.description = "Evaluate a backbone checkpoint without TTA"
    args = parser.parse_args(argv)
    config = apply_cli_overrides(load_config(args.config), args, parser)
    if not config.checkpoint_path:
        raise ValueError("checkpoint_path must be set for backbone evaluation.")

    seed_everything(config.seed)
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    run_dir = make_run_dir(config.output_dir, config.dataset, f"{config.baseline}_backbone_eval")

    with tee_run_output(run_dir) as log_path:
        print(f"=== Backbone Eval | baseline={config.baseline} | dataset={config.dataset} ===")
        print(f"  device={device}  seed={config.seed}")
        print(f"  run_dir={run_dir}")
        print(f"  run_log={log_path}")
        print(f"  checkpoint_path={config.checkpoint_path}")

        clients = _load_clients(config)
        print(f"  Loaded {len(clients)} clients.")
        model = load_model(config, device)
        efficiency = RunEfficiency(device)
        efficiency.start()
        per_client = {
            client.client_id: evaluate_client(
                model=model,
                client=client,
                seq_len=config.model.seq_len,
                pred_len=config.model.pred_len,
                device=device,
                efficiency=efficiency,
            )
            for client in clients
        }
        avg = _aggregate(per_client)
        wd = wasserstein_noniid(clients)
        print(
            f"\n[Backbone Eval] Avg: MSE={avg['mse']:.4f}  "
            f"MAE={avg['mae']:.4f}  sMAPE={avg['smape']:.2f}%  Non-IID(W)={wd:.4f}"
        )
        save_results(
            run_dir,
            {
                "engine": "backbone_eval",
                "per_client": per_client,
                "avg": avg,
                "wasserstein_noniid": wd,
                **efficiency.payload(
                    seed=config.seed,
                    trainable_params=0,
                    total_backbone_params=count_parameters(model),
                ),
            },
            dataclasses.asdict(config),
        )


if __name__ == "__main__":
    main()
