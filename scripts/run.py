"""
Unified entry point for all 5 DLinear FED-TTA baselines.

Usage:
    python scripts/run.py --config configs/solar/fed_tta_loop.yaml
    python scripts/run.py --config configs/electricity/centralized.yaml --device cpu
    python scripts/run.py --config configs/solar/dlinear_tta.yaml \\
        --checkpoint-path checkpoints/solar_centralized/best.pt

The baseline is determined by the 'baseline' field in the YAML config.
Results are saved to runs/{timestamp}_{dataset}_{baseline}/metrics.json.
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
from pathlib import Path

import numpy as np
import torch

from scripts.config import ExperimentConfig, load_config
from scripts.data.dataset import ClientData
from scripts.data.loader import load_csv_as_clients, load_parquet_as_clients
from scripts.models.revin_dlinear import RevINDLinear
from scripts.trainers.centralized import run_centralized
from scripts.trainers.fedavg import run_fedavg
from scripts.tta.adapter import prepare_tta_model
from scripts.tta.engine import (
    RollbackGuard, ReconTracker,
    build_hindcast_inputs, evaluate_client, run_tta_step,
)
from scripts.tta.loop import run_fed_tta_loop
from scripts.tta.loss import TTALoss
from scripts.utils.metrics import mse, mae, smape, inverse_global_scale, wasserstein_noniid
from scripts.utils.tools import seed_everything, make_run_dir, save_results


def _load_clients(config: ExperimentConfig) -> list[ClientData]:
    """Load clients according to data_format in config."""
    if config.data_format == "csv":
        return load_csv_as_clients(
            csv_path=config.data_path,
            timestamp_col=config.timestamp_col,
            seq_len=config.model.seq_len,
            pred_len=config.model.pred_len,
            max_clients=config.max_clients,
        )
    elif config.data_format == "parquet":
        clients_dir = config.parquet_clients_dir or str(Path(config.data_path).parent / "clients")
        return load_parquet_as_clients(
            manifest_path=config.data_path,
            clients_dir=clients_dir,
            seq_len=config.model.seq_len,
            pred_len=config.model.pred_len,
            max_clients=config.max_clients,
        )
    else:
        raise ValueError(f"Unknown data_format: {config.data_format!r}")


def _load_model(config: ExperimentConfig, device: torch.device) -> RevINDLinear:
    """Load a pre-trained model from checkpoint_path."""
    if not config.checkpoint_path:
        raise ValueError("checkpoint_path must be set for TTA baselines.")
    model = RevINDLinear(
        seq_len=config.model.seq_len,
        pred_len=config.model.pred_len,
        channels=1,
        kernel_size=config.model.kernel_size,
        individual=config.model.individual,
        revin_affine=config.model.revin_affine,
    ).to(device)
    state = torch.load(config.checkpoint_path, map_location=device)
    model.load_state_dict(state)
    return model


def _aggregate_metrics(per_client: dict[str, dict[str, float]]) -> dict[str, float]:
    """Average per-client metrics (skip NaN clients)."""
    keys = ["mse", "mae", "smape"]
    agg: dict[str, list[float]] = {k: [] for k in keys}
    for m in per_client.values():
        for k in keys:
            v = m.get(k, float("nan"))
            if not np.isnan(v):
                agg[k].append(v)
    return {k: float(np.mean(v)) if v else float("nan") for k, v in agg.items()}


# ---------------------------------------------------------------------------
# Baseline runners
# ---------------------------------------------------------------------------

def run_baseline_centralized(config: ExperimentConfig, clients: list[ClientData],
                              device: torch.device, run_dir: Path) -> None:
    model = run_centralized(config, clients, device)
    per_client = {c.client_id: evaluate_client(
        model=model, client=c,
        seq_len=config.model.seq_len, pred_len=config.model.pred_len, device=device,
    ) for c in clients}
    avg = _aggregate_metrics(per_client)
    print(f"\n[Centralized] Avg: MSE={avg['mse']:.4f}  MAE={avg['mae']:.4f}  sMAPE={avg['smape']:.2f}%")
    save_results(run_dir, {"per_client": per_client, "avg": avg},
                 dataclasses.asdict(config))


def run_baseline_fed(config: ExperimentConfig, clients: list[ClientData],
                     device: torch.device, run_dir: Path) -> None:
    model = run_fedavg(config, clients, device)
    per_client = {c.client_id: evaluate_client(
        model=model, client=c,
        seq_len=config.model.seq_len, pred_len=config.model.pred_len, device=device,
    ) for c in clients}
    avg = _aggregate_metrics(per_client)
    wd = wasserstein_noniid(clients)
    print(f"\n[FED] Avg: MSE={avg['mse']:.4f}  MAE={avg['mae']:.4f}  "
          f"sMAPE={avg['smape']:.2f}%  Non-IID(W)={wd:.4f}")
    save_results(run_dir, {"per_client": per_client, "avg": avg, "wasserstein_noniid": wd},
                 dataclasses.asdict(config))


def _run_tta_eval(
    config: ExperimentConfig,
    clients: list[ClientData],
    model: RevINDLinear,
    device: torch.device,
    run_dir: Path,
    label: str,
) -> None:
    """Shared TTA evaluation loop for dlinear_tta and fed_tta (no server feedback)."""
    k = config.k()
    tta_loss_fn = TTALoss(
        k=k,
        alpha=config.tta.alpha,
        lambda0=config.tta.lambda0,
        gamma=config.tta.gamma,
    )

    per_client: dict[str, dict[str, float]] = {}
    for client in clients:
        cm = copy.deepcopy(model).to(device)
        cm, anchor_t, anchor_s = prepare_tta_model(cm)
        tta_params = [p for p in cm.dlinear.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(tta_params, lr=config.tta.lr)
        tracker = ReconTracker(window_size=config.tta.rollback_window)
        guard = RollbackGuard(threshold=config.tta.rollback_threshold, tracker=tracker)

        test_s, test_e = client.split_indices["test"]
        preds_g, targets_g = [], []

        for t_abs in range(test_s + config.model.seq_len + k - 1, test_e):
            inputs = build_hindcast_inputs(client.values, t_abs, config.model.seq_len, k)
            if inputs is None:
                continue
            x_input, x_recent = inputs

            run_tta_step(
                model=cm, optimizer=optimizer, tta_loss_fn=tta_loss_fn,
                anchor_trend_w=anchor_t, anchor_season_w=anchor_s,
                x_input=x_input, x_recent=x_recent,
                mu_hist=client.global_mean, sigma_hist=max(client.global_std, 1e-8),
                rollback_guard=guard, device=device, grad_clip=config.tta.grad_clip,
            )

            # Standard prediction window
            pred_start = t_abs - config.model.seq_len + 1
            if pred_start >= 0 and pred_start + config.model.seq_len <= len(client.values):
                x_np = client.values[pred_start : pred_start + config.model.seq_len]
                x_t = torch.from_numpy(x_np).unsqueeze(0).to(device)
                cm.eval()
                with torch.no_grad():
                    pred = cm(x_t).cpu().numpy()[0]
                cm.train()
                target = client.values[
                    pred_start + config.model.seq_len :
                    pred_start + config.model.seq_len + config.model.pred_len
                ]
                if len(target) == config.model.pred_len:
                    preds_g.append(pred)
                    targets_g.append(target)

        if preds_g:
            pg = np.concatenate(preds_g, axis=0)
            tg = np.concatenate(targets_g, axis=0)
            po = inverse_global_scale(pg, client.global_mean, client.global_std)
            to_ = inverse_global_scale(tg, client.global_mean, client.global_std)
            per_client[client.client_id] = {
                "mse": mse(pg, tg), "mae": mae(pg, tg), "smape": smape(po, to_),
            }
        else:
            per_client[client.client_id] = {
                "mse": float("nan"), "mae": float("nan"), "smape": float("nan"),
            }

    avg = _aggregate_metrics(per_client)
    print(f"\n[{label}] Avg: MSE={avg['mse']:.4f}  MAE={avg['mae']:.4f}  "
          f"sMAPE={avg['smape']:.2f}%")
    save_results(run_dir, {"per_client": per_client, "avg": avg},
                 dataclasses.asdict(config))


def run_baseline_dlinear_tta(config: ExperimentConfig, clients: list[ClientData],
                              device: torch.device, run_dir: Path) -> None:
    model = _load_model(config, device)
    _run_tta_eval(config, clients, model, device, run_dir, label="DLinear-TTA")


def run_baseline_fed_tta(config: ExperimentConfig, clients: list[ClientData],
                          device: torch.device, run_dir: Path) -> None:
    model = _load_model(config, device)
    _run_tta_eval(config, clients, model, device, run_dir, label="FED-TTA")


def run_baseline_fed_tta_loop(config: ExperimentConfig, clients: list[ClientData],
                               device: torch.device, run_dir: Path) -> None:
    model = _load_model(config, device)
    k = config.k()
    per_client = run_fed_tta_loop(
        global_model=model,
        clients=clients,
        seq_len=config.model.seq_len,
        pred_len=config.model.pred_len,
        k=k,
        tta_config=config.tta,
        feedback_config=config.feedback,
        device=device,
    )
    avg = _aggregate_metrics(per_client)
    wd = wasserstein_noniid(clients)
    print(f"\n[FED-TTA Loop] Avg: MSE={avg['mse']:.4f}  MAE={avg['mae']:.4f}  "
          f"sMAPE={avg['smape']:.2f}%  Non-IID(W)={wd:.4f}")
    save_results(run_dir, {"per_client": per_client, "avg": avg, "wasserstein_noniid": wd},
                 dataclasses.asdict(config))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

BASELINE_RUNNERS = {
    "centralized":   run_baseline_centralized,
    "fed":           run_baseline_fed,
    "dlinear_tta":   run_baseline_dlinear_tta,
    "fed_tta":       run_baseline_fed_tta,
    "fed_tta_loop":  run_baseline_fed_tta_loop,
}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="DLinear FED-TTA — unified experiment runner"
    )
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--device", default=None, help="Override device (e.g. cpu, cuda:1)")
    parser.add_argument("--checkpoint-path", default=None, dest="checkpoint_path",
                        help="Override checkpoint_path in config (for TTA baselines)")
    parser.add_argument("--max-clients", type=int, default=None, dest="max_clients",
                        help="Override max_clients in config")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args(argv)

    config = load_config(args.config)

    # CLI overrides
    if args.device is not None:
        config = dataclasses.replace(config, device=args.device)
    if args.checkpoint_path is not None:
        config = dataclasses.replace(config, checkpoint_path=args.checkpoint_path)
    if args.max_clients is not None:
        config = dataclasses.replace(config, max_clients=args.max_clients)
    if args.seed is not None:
        config = dataclasses.replace(config, seed=args.seed)

    seed_everything(config.seed)
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")

    print(f"=== DLinear FED-TTA | baseline={config.baseline} | dataset={config.dataset} ===")
    print(f"  device={device}  seed={config.seed}")

    clients = _load_clients(config)
    print(f"  Loaded {len(clients)} clients.")

    run_dir = make_run_dir(config.output_dir, config.dataset, config.baseline)

    runner = BASELINE_RUNNERS.get(config.baseline)
    if runner is None:
        raise ValueError(
            f"Unknown baseline: {config.baseline!r}. "
            f"Choose from: {list(BASELINE_RUNNERS)}"
        )

    runner(config, clients, device, run_dir)


if __name__ == "__main__":
    main()
