from __future__ import annotations

from pathlib import Path

import torch

from scripts.baselines import BASELINE_RUNNERS
from scripts.checkpoints import TTA_PREREQ_BASELINES, resolve_or_build_prereq_checkpoint
from scripts.cli import apply_cli_overrides, build_parser
from scripts.config import ExperimentConfig, load_config
from scripts.data.dataset import ClientData
from scripts.data.loader import load_csv_as_clients, load_parquet_as_clients
from scripts.utils.run_logging import tee_run_output
from scripts.utils.tools import make_run_dir, seed_everything


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


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = apply_cli_overrides(load_config(args.config), args, parser)
    seed_everything(config.seed)
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    run_dir = make_run_dir(config.output_dir, config.dataset, config.baseline)
    runner = BASELINE_RUNNERS.get(config.baseline)
    if runner is None:
        raise ValueError(
            f"Unknown baseline: {config.baseline!r}. Choose from: {list(BASELINE_RUNNERS)}"
        )
    with tee_run_output(run_dir) as log_path:
        print(f"=== DLinear FED-TTA | baseline={config.baseline} | dataset={config.dataset} ===")
        print(f"  device={device}  seed={config.seed}")
        print(f"  run_dir={run_dir}")
        print(f"  run_log={log_path}")
        clients = _load_clients(config)
        print(f"  Loaded {len(clients)} clients.")
        if config.baseline in TTA_PREREQ_BASELINES:
            config = resolve_or_build_prereq_checkpoint(
                config=config,
                config_path=args.config,
                clients=clients,
                device=device,
                auto_prereq=args.auto_prereq,
                baseline_runners=BASELINE_RUNNERS,
            )
        runner(config, clients, device, run_dir)


if __name__ == "__main__":
    main()
