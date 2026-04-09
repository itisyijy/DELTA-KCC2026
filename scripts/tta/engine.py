"""
TTA inference engine: step-by-step adaptation and evaluation.

Plan.md §2.3–2.4:
  - Pre-update L_recon measurement (no-grad) for rollback gating
  - Rollback: skip update if L_recon_pre > threshold × rolling_mean(L_recon)
  - Standard evaluation: MSE/MAE in Global Scale, sMAPE in original units
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn

from scripts.data.dataset import ClientData
from scripts.models.revin_dlinear import RevINDLinear
from scripts.tta.loss import TTALoss
from scripts.utils.metrics import mae, mse, smape, inverse_global_scale


class ReconTracker:
    """Maintains a rolling average of L_recon values for rollback gating."""

    def __init__(self, window_size: int = 20):
        self.history: deque[float] = deque(maxlen=window_size)

    def update(self, loss_val: float) -> None:
        self.history.append(loss_val)

    def rolling_mean(self) -> float:
        return float(np.mean(self.history)) if self.history else float("inf")


class RollbackGuard:
    """
    Pre-update rollback: if L_recon_pre > threshold * rolling_mean → skip.

    When history is empty, the guard never skips (warm-up phase).
    """

    def __init__(self, threshold: float = 3.0, tracker: ReconTracker | None = None):
        self.threshold = threshold
        self.tracker = tracker if tracker is not None else ReconTracker()

    def should_skip(self, l_recon_pre: float) -> bool:
        avg = self.tracker.rolling_mean()
        if avg == float("inf"):   # no history yet
            return False
        return l_recon_pre > self.threshold * avg


def build_hindcast_inputs(
    series: np.ndarray,  # [N, 1] Global Scale
    t: int,              # absolute index into series (last observed step)
    seq_len: int,
    k: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Build (x_input, x_recent) for hindcast TTA at time step t.

    X_input  = series[t - seq_len - k + 1 : t - k + 1]  shape [seq_len, 1]
    X_recent = series[t - k + 1 : t + 1]                 shape [k, 1]

    Returns None if t is too small (insufficient history).
    """
    start_input = t - seq_len - k + 1
    end_input = t - k + 1       # exclusive
    start_recent = t - k + 1
    end_recent = t + 1          # exclusive

    if start_input < 0:
        return None

    x_input = series[start_input:end_input]    # [seq_len, 1]
    x_recent = series[start_recent:end_recent]  # [k, 1]
    return x_input, x_recent


@dataclass
class TTAStepResult:
    skipped: bool
    skip_reason: str | None
    l_recon_pre: float
    l_recon_post: float | None = None
    l_reg: float | None = None


def run_tta_step(
    *,
    model: RevINDLinear,
    optimizer: torch.optim.Optimizer,
    tta_loss_fn: TTALoss,
    anchor_trend_w: torch.Tensor,
    anchor_season_w: torch.Tensor,
    x_input: np.ndarray,    # [seq_len, 1]
    x_recent: np.ndarray,   # [k, 1]
    mu_hist: float,
    sigma_hist: float,
    rollback_guard: RollbackGuard,
    device: torch.device,
    grad_clip: float = 1.0,
) -> TTAStepResult:
    """
    One TTA gradient step.

    Algorithm (plan.md §2.4 Safety Mechanisms):
      1. Pre-measure L_recon with no_grad (rollback gate)
      2. If should_skip → return skipped result (do NOT update weights)
      3. optimizer.zero_grad()
      4. Forward pass (with grad) → y_hat
      5. Compute TTALoss → backward → clip grads → step
      6. Update ReconTracker with pre-update L_recon
    """
    x_input_t = torch.from_numpy(x_input).unsqueeze(0).to(device)   # [1, seq_len, 1]
    x_recent_t = torch.from_numpy(x_recent).to(device)               # [k, 1]

    # --- Step 1: pre-update L_recon (no grad) ---
    model.eval()
    with torch.no_grad():
        y_hat_pre = model(x_input_t)  # [1, pred_len, 1]
        l_recon_pre = tta_loss_fn.hindcast(y_hat_pre, x_recent_t).item()

    # --- Step 2: rollback gate ---
    if rollback_guard.should_skip(l_recon_pre):
        rollback_guard.tracker.update(l_recon_pre)
        return TTAStepResult(
            skipped=True,
            skip_reason="rollback",
            l_recon_pre=l_recon_pre,
        )

    # --- Steps 3–6: gradient update ---
    model.train()
    optimizer.zero_grad()

    y_hat = model(x_input_t)  # [1, pred_len, 1]
    total_loss, l_recon, l_reg = tta_loss_fn(
        y_hat, x_recent_t, model,
        anchor_trend_w, anchor_season_w,
        mu_hist, sigma_hist,
    )
    total_loss.backward()

    # Gradient clipping on TTA parameters only
    tta_params = [p for p in model.dlinear.parameters() if p.requires_grad]
    nn.utils.clip_grad_norm_(tta_params, grad_clip)

    optimizer.step()
    rollback_guard.tracker.update(l_recon_pre)

    return TTAStepResult(
        skipped=False,
        skip_reason=None,
        l_recon_pre=l_recon_pre,
        l_recon_post=l_recon.item(),
        l_reg=l_reg.item(),
    )


def evaluate_client(
    *,
    model: RevINDLinear,
    client: ClientData,
    seq_len: int,
    pred_len: int,
    device: torch.device,
) -> dict[str, float]:
    """
    Standard (non-TTA) evaluation over a client's test split.

    Computes:
      - MSE, MAE in Global Scale (after RevIN denorm)
      - sMAPE in original physical units (after global scaler inverse)
    """
    model.eval()
    test_s, test_e = client.split_indices["test"]
    test_data = client.values  # full series [N, 1]

    preds_global, targets_global = [], []

    with torch.no_grad():
        for start in range(test_s, test_e - seq_len - pred_len + 1):
            x = test_data[start : start + seq_len]
            y = test_data[start + seq_len : start + seq_len + pred_len]
            x_t = torch.from_numpy(x).unsqueeze(0).to(device)  # [1, seq_len, 1]
            pred = model(x_t).cpu().numpy()[0]                  # [pred_len, 1]
            preds_global.append(pred)
            targets_global.append(y)

    if not preds_global:
        return {"mse": float("nan"), "mae": float("nan"), "smape": float("nan")}

    preds_g = np.concatenate(preds_global, axis=0)    # [N_test_windows * pred_len, 1]
    targets_g = np.concatenate(targets_global, axis=0)

    # Inverse transform for sMAPE
    preds_orig = inverse_global_scale(preds_g, client.global_mean, client.global_std)
    targets_orig = inverse_global_scale(targets_g, client.global_mean, client.global_std)

    return {
        "mse":   mse(preds_g, targets_g),
        "mae":   mae(preds_g, targets_g),
        "smape": smape(preds_orig, targets_orig),
    }
