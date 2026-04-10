from __future__ import annotations

import copy

import numpy as np
import torch

from scripts.config import TTAConfig, FeedbackConfig
from scripts.data.dataset import ClientData
from scripts.models.revin_dlinear import RevINDLinear
from scripts.tta.adapter import prepare_tta_model
from scripts.tta.delta import ClientDelta, aggregate_deltas, apply_server_feedback, clip_delta, compute_delta, get_model_weights
from scripts.tta.engine import (
    RollbackGuard,
    ReconTracker,
    build_hindcast_inputs,
    run_tta_step,
)
from scripts.tta.loss import TTALoss
from scripts.utils.metrics import mae, mse, smape, inverse_global_scale
from scripts.utils.run_logging import MilestoneLogger


def run_fed_tta_loop(
    *,
    global_model: RevINDLinear,
    clients: list[ClientData],
    seq_len: int,
    pred_len: int,
    k: int,
    tta_config: TTAConfig,
    feedback_config: FeedbackConfig,
    device: torch.device,
) -> dict[str, dict[str, float]]:
    """
    Run the FED-TTA Loop over all clients' test splits.

    For every test time step t (in lock-step across all clients):
      1. Each client runs one TTA step.
      2. Collect deltas from non-skipped clients, clip + aggregate.
      3. Apply aggregated delta to global_model.
      4. Clients load the updated global weights before the next step.

    Returns per-client evaluation metrics.
    """
    tta_loss_fn = TTALoss(
        k=k,
        alpha=tta_config.alpha,
        lambda0=tta_config.lambda0,
        gamma=tta_config.gamma,
    )

    # Per-client state
    client_models: list[RevINDLinear] = []
    client_anchors: list[tuple[torch.Tensor, torch.Tensor]] = []
    client_guards: list[RollbackGuard] = []
    client_optimizers: list[torch.optim.Optimizer] = []

    for client in clients:
        cm = copy.deepcopy(global_model).to(device)
        cm, anchor_t, anchor_s = prepare_tta_model(cm)
        client_models.append(cm)
        client_anchors.append((anchor_t, anchor_s))
        tracker = ReconTracker(window_size=tta_config.rollback_window)
        client_guards.append(
            RollbackGuard(threshold=tta_config.rollback_threshold, tracker=tracker)
        )
        tta_params = [p for p in cm.dlinear.parameters() if p.requires_grad]
        client_optimizers.append(torch.optim.Adam(tta_params, lr=tta_config.lr))

    # Determine the test time range (use first client as reference)
    # All clients share the same split structure (loaded from same CSV)
    test_s, test_e = clients[0].split_indices["test"]
    total_steps = max(0, test_e - (test_s + seq_len + k - 1))
    progress = MilestoneLogger("FED-TTA Loop", total_steps)
    progress.start(detail=f"clients={len(clients)}")

    # Accumulate predictions for final evaluation
    # per-client: list of (pred [pred_len, 1], target [pred_len, 1])
    all_preds: list[list[np.ndarray]] = [[] for _ in clients]
    all_targets: list[list[np.ndarray]] = [[] for _ in clients]

    # Lock-step over test steps
    for t_rel, t_abs in enumerate(range(test_s + seq_len + k - 1, test_e)):
        client_deltas: list[ClientDelta] = []

        for ci, (client, cm, (anchor_t, anchor_s), guard, opt) in enumerate(
            zip(clients, client_models, client_anchors, client_guards, client_optimizers)
        ):
            inputs = build_hindcast_inputs(client.values, t_abs, seq_len, k)
            if inputs is None:
                continue
            x_input, x_recent = inputs

            result = run_tta_step(
                model=cm,
                optimizer=opt,
                tta_loss_fn=tta_loss_fn,
                anchor_trend_w=anchor_t,
                anchor_season_w=anchor_s,
                x_input=x_input,
                x_recent=x_recent,
                mu_hist=client.global_mean,
                sigma_hist=max(client.global_std, 1e-8),
                rollback_guard=guard,
                device=device,
                grad_clip=tta_config.grad_clip,
            )

            if not result.skipped:
                dt, ds = compute_delta(cm, anchor_t, anchor_s)
                dt = clip_delta(dt, feedback_config.delta_clip_norm)
                ds = clip_delta(ds, feedback_config.delta_clip_norm)
                client_deltas.append(
                    ClientDelta(
                        client_id=client.client_id,
                        delta_trend=dt,
                        delta_season=ds,
                        n_valid_steps=1,
                    )
                )

            # Store standard prediction for evaluation
            x_pred_start = t_abs - seq_len + 1
            if x_pred_start + seq_len <= len(client.values):
                x_np = client.values[x_pred_start : x_pred_start + seq_len]  # [seq_len, 1]
                x_t = torch.from_numpy(x_np).unsqueeze(0).to(device)
                cm.eval()
                with torch.no_grad():
                    pred = cm(x_t).cpu().numpy()[0]  # [pred_len, 1]
                cm.train()  # back to train for next step
                target = client.values[
                    x_pred_start + seq_len : x_pred_start + seq_len + pred_len
                ]
                if len(target) == pred_len:
                    all_preds[ci].append(pred)
                    all_targets[ci].append(target)

        # --- Server feedback ---
        aggregated = aggregate_deltas(client_deltas, feedback_config.decay_factor)
        if aggregated is not None:
            agg_dt, agg_ds = aggregated
            apply_server_feedback(global_model, agg_dt, agg_ds)
            for ci, cm in enumerate(client_models):
                cm.load_state_dict(global_model.state_dict())
                client_anchors[ci] = get_model_weights(cm)
                client_optimizers[ci].state.clear()
        progress.update(t_rel + 1, detail=f"active_deltas={len(client_deltas)}")

    # --- Compute per-client metrics ---
    progress.finish(detail=f"clients={len(clients)}")
    results: dict[str, dict[str, float]] = {}
    for ci, client in enumerate(clients):
        preds = all_preds[ci]
        targets = all_targets[ci]
        if not preds:
            results[client.client_id] = {
                "mse": float("nan"), "mae": float("nan"), "smape": float("nan")
            }
            continue

        preds_g = np.concatenate(preds, axis=0)
        targets_g = np.concatenate(targets, axis=0)
        preds_orig = inverse_global_scale(preds_g, client.global_mean, client.global_std)
        targets_orig = inverse_global_scale(targets_g, client.global_mean, client.global_std)

        results[client.client_id] = {
            "mse":   mse(preds_g, targets_g),
            "mae":   mae(preds_g, targets_g),
            "smape": smape(preds_orig, targets_orig),
        }

    return results
