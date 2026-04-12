from __future__ import annotations

import copy
import dataclasses
from pathlib import Path

import numpy as np
import torch

from scripts.checkpoints import load_model
from scripts.config import ExperimentConfig
from scripts.data.dataset import ClientData
from scripts.models.revin_dlinear import RevINDLinear
from scripts.trainers.centralized import run_centralized
from scripts.trainers.fedavg import run_fedavg
from scripts.tta.adapter import prepare_tta_model
from scripts.tta.engine import RollbackGuard, ReconTracker, build_hindcast_inputs, evaluate_client, run_tta_step
from scripts.tta.loop import run_fed_tta_loop
from scripts.tta.loss import TTALoss
from scripts.tta.policy import get_tta_parameters
from scripts.utils.metrics import inverse_global_scale, mae, mse, smape, wasserstein_noniid
from scripts.utils.run_logging import MilestoneLogger
from scripts.utils.tools import save_results


def _aggregate_metrics(per_client: dict[str, dict[str, float]]) -> dict[str, float]:
    keys = ["mse", "mae", "smape"]
    agg: dict[str, list[float]] = {k: [] for k in keys}
    for metrics in per_client.values():
        for key in keys:
            value = metrics.get(key, float("nan"))
            if not np.isnan(value):
                agg[key].append(value)
    return {key: float(np.mean(values)) if values else float("nan") for key, values in agg.items()}


def _save_metrics(run_dir: Path, config: ExperimentConfig, payload: dict) -> None:
    save_results(run_dir, payload, dataclasses.asdict(config))


def _run_supervised_baseline(
    config: ExperimentConfig,
    clients: list[ClientData],
    device: torch.device,
    run_dir: Path,
    label: str,
    trainer,
    include_wasserstein: bool,
    checkpoint_path_override: str | Path | None,
) -> None:
    model = trainer(config, clients, device, checkpoint_path_override=checkpoint_path_override)
    per_client = {
        client.client_id: evaluate_client(
            model=model,
            client=client,
            seq_len=config.model.seq_len,
            pred_len=config.model.pred_len,
            device=device,
        )
        for client in clients
    }
    avg = _aggregate_metrics(per_client)
    extras = {}
    suffix = ""
    if include_wasserstein:
        extras["wasserstein_noniid"] = wasserstein_noniid(clients)
        suffix = f"  Non-IID(W)={extras['wasserstein_noniid']:.4f}"
    print(f"\n[{label}] Avg: MSE={avg['mse']:.4f}  MAE={avg['mae']:.4f}  sMAPE={avg['smape']:.2f}%{suffix}")
    _save_metrics(run_dir, config, {"per_client": per_client, "avg": avg, **extras})


def _count_tta_steps(clients: list[ClientData], seq_len: int, k: int) -> int:
    return sum(
        max(0, client.split_indices["test"][1] - (client.split_indices["test"][0] + seq_len + k - 1))
        for client in clients
    )


def _run_tta_eval(
    config: ExperimentConfig,
    clients: list[ClientData],
    model: RevINDLinear,
    device: torch.device,
    run_dir: Path,
    label: str,
) -> None:
    k = config.k()
    progress = MilestoneLogger(label=label, total_steps=_count_tta_steps(clients, config.model.seq_len, k))
    tta_loss_fn = TTALoss(k=k, alpha=config.tta.alpha, lambda0=config.tta.lambda0, gamma=config.tta.gamma, lambda_func=config.tta.lambda_func, hindcast_mask_threshold=config.tta.hindcast_mask_threshold)
    per_client: dict[str, dict[str, float]] = {}
    completed_steps = 0
    progress.start(detail=f"clients={len(clients)}")
    anchor_model = copy.deepcopy(model).to(device).eval() if config.tta.lambda_func > 0.0 else None
    for param in (anchor_model.parameters() if anchor_model else []):
        param.requires_grad_(False)

    for client in clients:
        model_copy = copy.deepcopy(model).to(device)
        model_copy, anchor_t, anchor_s = prepare_tta_model(model_copy, config.tta.update_scope)
        optimizer = torch.optim.Adam(get_tta_parameters(model_copy), lr=config.tta.lr)
        guard = RollbackGuard(
            threshold=config.tta.rollback_threshold,
            tracker=ReconTracker(window_size=config.tta.rollback_window),
        )
        # EMA anchor (mechanism 2): starts at FL checkpoint, slowly drifts toward adapted weights.
        # ema_beta=1.0 means fixed anchor (disabled), <1.0 enables progressive personalization.
        ema_anchor_t = anchor_t.clone()
        ema_anchor_s = anchor_s.clone()
        ema_beta = config.tta.ema_beta

        preds_g, targets_g = [], []
        test_s, test_e = client.split_indices["test"]
        for t_abs in range(test_s + config.model.seq_len + k - 1, test_e):
            completed_steps += 1
            inputs = build_hindcast_inputs(client.values, t_abs, config.model.seq_len, k)
            if inputs is None:
                progress.update(completed_steps)
                continue
            x_input, x_recent = inputs
            run_tta_step(
                model=model_copy,
                optimizer=optimizer,
                tta_loss_fn=tta_loss_fn,
                anchor_trend_w=ema_anchor_t,
                anchor_season_w=ema_anchor_s,
                x_input=x_input,
                x_recent=x_recent,
                mu_hist=client.global_mean,
                sigma_hist=max(client.global_std, 1e-8),
                rollback_guard=guard,
                device=device,
                grad_clip=config.tta.grad_clip,
                drift_gate_threshold=config.tta.drift_gate_threshold,
                min_active_frac=config.tta.min_active_frac,
                anchor_model=anchor_model,
            )

            # Update EMA anchor from current model weights (after potential adaptation)
            if ema_beta < 1.0:
                lt = model_copy.linear_trend
                ls = model_copy.linear_seasonal
                with torch.no_grad():
                    if isinstance(lt, torch.nn.ModuleList):
                        cur_t = torch.stack([l.weight for l in lt])
                        cur_s = torch.stack([l.weight for l in ls])
                    else:
                        cur_t = lt.weight
                        cur_s = ls.weight
                    ema_anchor_t = ema_beta * ema_anchor_t + (1 - ema_beta) * cur_t.detach()
                    ema_anchor_s = ema_beta * ema_anchor_s + (1 - ema_beta) * cur_s.detach()
            pred_start = t_abs - config.model.seq_len + 1
            if pred_start >= 0 and pred_start + config.model.seq_len <= len(client.values):
                x_t = torch.from_numpy(client.values[pred_start : pred_start + config.model.seq_len]).unsqueeze(0).to(device)
                model_copy.eval()
                with torch.no_grad():
                    pred = model_copy(x_t).cpu().numpy()[0]
                model_copy.train()
                target = client.values[pred_start + config.model.seq_len : pred_start + config.model.seq_len + config.model.pred_len]
                if len(target) == config.model.pred_len:
                    preds_g.append(pred)
                    targets_g.append(target)
            progress.update(completed_steps)
        if preds_g:
            preds_arr = np.concatenate(preds_g, axis=0)
            targets_arr = np.concatenate(targets_g, axis=0)
            per_client[client.client_id] = {
                "mse": mse(preds_arr, targets_arr),
                "mae": mae(preds_arr, targets_arr),
                "smape": smape(
                    inverse_global_scale(preds_arr, client.global_mean, client.global_std),
                    inverse_global_scale(targets_arr, client.global_mean, client.global_std),
                ),
            }
        else:
            per_client[client.client_id] = {"mse": float("nan"), "mae": float("nan"), "smape": float("nan")}
    avg = _aggregate_metrics(per_client)
    progress.finish(detail=f"clients={len(clients)}")
    print(f"\n[{label}] Avg: MSE={avg['mse']:.4f}  MAE={avg['mae']:.4f}  sMAPE={avg['smape']:.2f}%")
    _save_metrics(run_dir, config, {"per_client": per_client, "avg": avg})


def run_baseline_centralized(config, clients, device, run_dir, checkpoint_path_override=None) -> None:
    _run_supervised_baseline(config, clients, device, run_dir, "Centralized", run_centralized, False, checkpoint_path_override)


def run_baseline_fed(config, clients, device, run_dir, checkpoint_path_override=None) -> None:
    _run_supervised_baseline(config, clients, device, run_dir, "FED", run_fedavg, True, checkpoint_path_override)


def run_baseline_dlinear_tta(config, clients, device, run_dir, checkpoint_path_override=None) -> None:
    _run_tta_eval(config, clients, load_model(config, device), device, run_dir, "DLinear-TTA")


def run_baseline_fed_tta(config, clients, device, run_dir, checkpoint_path_override=None) -> None:
    _run_tta_eval(config, clients, load_model(config, device), device, run_dir, "FED-TTA")


def run_baseline_fed_tta_loop(config, clients, device, run_dir, checkpoint_path_override=None) -> None:
    per_client = run_fed_tta_loop(
        global_model=load_model(config, device),
        clients=clients,
        seq_len=config.model.seq_len,
        pred_len=config.model.pred_len,
        k=config.k(),
        tta_config=config.tta,
        feedback_config=config.feedback,
        device=device,
    )
    avg = _aggregate_metrics(per_client)
    wd = wasserstein_noniid(clients)
    print(f"\n[FED-TTA Loop] Avg: MSE={avg['mse']:.4f}  MAE={avg['mae']:.4f}  sMAPE={avg['smape']:.2f}%  Non-IID(W)={wd:.4f}")
    _save_metrics(run_dir, config, {"per_client": per_client, "avg": avg, "wasserstein_noniid": wd})


BASELINE_RUNNERS = {
    "centralized": run_baseline_centralized,
    "fed": run_baseline_fed,
    "dlinear_tta": run_baseline_dlinear_tta,
    "fed_tta": run_baseline_fed_tta,
    "fed_tta_loop": run_baseline_fed_tta_loop,
}
