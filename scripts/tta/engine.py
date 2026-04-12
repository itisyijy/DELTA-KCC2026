"""
TTA inference engine: adaptation steps and per-client evaluation.

Two step functions are provided:

run_tta_step (legacy)
    Updates W_trend / W_season in-place.
    Used by run_fed_tta_loop.

run_tta_step_affine (new)
    Backbone completely frozen; only AffineAdapter (γ, δ ∈ R^C) updated.
    Uses HybridTTALoss (L_hind + L_cons + L_anchor with Bounded Adaptive Weighting).
    Used by run_local_tta_loop.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn

from scripts.data.dataset import ClientData
from scripts.models.revin_dlinear import RevINDLinear
from scripts.tta.adapter import AffineAdapter
from scripts.tta.loss import HybridTTALoss, TTALoss
from scripts.tta.policy import get_tta_parameters, should_adapt
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
    """Skip unstable updates after a warm-up history is available."""

    def __init__(self, threshold: float = 3.0, tracker: ReconTracker | None = None):
        self.threshold = threshold
        self.tracker = tracker if tracker is not None else ReconTracker()

    def should_skip(self, l_recon_pre: float) -> bool:
        avg = self.tracker.rolling_mean()
        if avg == float("inf"):
            return False
        return l_recon_pre > self.threshold * avg


def build_hindcast_inputs(
    series: np.ndarray,  # [N, 1] Global Scale
    t: int,              # absolute index into series (last observed step)
    seq_len: int,
    k: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Build the hindcast input/recent pair for one absolute time index."""
    start_input = t - seq_len - k + 1
    end_input = t - k + 1       # exclusive
    start_recent = t - k + 1
    end_recent = t + 1          # exclusive

    if start_input < 0:
        return None

    x_input = series[start_input:end_input]
    x_recent = series[start_recent:end_recent]
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
    drift_gate_threshold: float = 0.0,
    min_active_frac: float = 0.0,
    anchor_model: RevINDLinear | None = None,
) -> TTAStepResult:
    """Run one guarded TTA update step."""
    x_input_t = torch.from_numpy(x_input).unsqueeze(0).to(device)
    x_recent_t = torch.from_numpy(x_recent).to(device)

    # --- Inactive-window skip (mechanism 3) ---
    # Skip TTA when x_recent has too few above-zero-floor steps.
    # Uses the same threshold as HindcastLoss masking for consistency:
    #   active = x_recent > zero_floor + hindcast_mask_threshold
    # Solar PV values are always >= zero_floor, so 'zero_floor - 0.2' would
    # never trigger — we must use >= zero_floor (or + mask_threshold) instead.
    if min_active_frac > 0.0:
        zero_floor = -mu_hist / max(sigma_hist, 1e-8)
        active_frac = (x_recent_t > zero_floor + tta_loss_fn.hindcast_mask_threshold).float().mean().item()
        if active_frac < min_active_frac:
            return TTAStepResult(
                skipped=True,
                skip_reason="inactive_window",
                l_recon_pre=float("nan"),
            )

    model.eval()
    with torch.no_grad():
        y_hat_pre = model(x_input_t)
        # Use same masking as the gradient step for consistent rollback tracking
        l_recon_pre = tta_loss_fn.hindcast(
            y_hat_pre, x_recent_t, mu_hist, sigma_hist, tta_loss_fn.hindcast_mask_threshold
        ).item()

    if not should_adapt(x_recent_t, mu_hist, sigma_hist, drift_gate_threshold):
        rollback_guard.tracker.update(l_recon_pre)
        return TTAStepResult(
            skipped=True,
            skip_reason="drift_gate",
            l_recon_pre=l_recon_pre,
        )

    if rollback_guard.should_skip(l_recon_pre):
        rollback_guard.tracker.update(l_recon_pre)
        return TTAStepResult(
            skipped=True,
            skip_reason="rollback",
            l_recon_pre=l_recon_pre,
        )

    y_hat_anchor = None
    if anchor_model is not None:
        with torch.no_grad():
            y_hat_anchor = anchor_model(x_input_t)

    model.train()
    optimizer.zero_grad()

    y_hat = model(x_input_t)
    total_loss, l_recon, l_reg = tta_loss_fn(
        y_hat, x_recent_t, model,
        anchor_trend_w, anchor_season_w,
        mu_hist, sigma_hist,
        y_hat_anchor=y_hat_anchor,
    )

    # If all hindcast steps are masked (e.g. fully-inactive window with masking
    # enabled but min_active_frac disabled), total_loss has no grad_fn.  Skip
    # the update rather than crashing.
    if not total_loss.requires_grad:
        rollback_guard.tracker.update(l_recon_pre)
        return TTAStepResult(
            skipped=True,
            skip_reason="no_active_signal",
            l_recon_pre=l_recon_pre,
        )

    total_loss.backward()

    tta_params = get_tta_parameters(model)
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
    """Standard evaluation over one client's test split."""
    model.eval()
    test_s, test_e = client.split_indices["test"]
    test_data = client.values  # full series [N, 1]

    preds_global, targets_global = [], []

    with torch.no_grad():
        for start in range(test_s, test_e - seq_len - pred_len + 1):
            x = test_data[start : start + seq_len]
            y = test_data[start + seq_len : start + seq_len + pred_len]
            x_t = torch.from_numpy(x).unsqueeze(0).to(device)
            pred = model(x_t).cpu().numpy()[0]
            preds_global.append(pred)
            targets_global.append(y)

    if not preds_global:
        return {"mse": float("nan"), "mae": float("nan"), "smape": float("nan")}

    preds_g = np.concatenate(preds_global, axis=0)
    targets_g = np.concatenate(targets_global, axis=0)
    preds_orig = inverse_global_scale(preds_g, client.global_mean, client.global_std)
    targets_orig = inverse_global_scale(targets_g, client.global_mean, client.global_std)

    return {
        "mse":   mse(preds_g, targets_g),
        "mae":   mae(preds_g, targets_g),
        "smape": smape(preds_orig, targets_orig),
    }


# ─────────────────────────────────────────────────────────────────────────────
# New: Affine-Adapter TTA step
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TTAStepResultV2:
    """Result of one run_tta_step_affine call."""
    skipped: bool
    skip_reason: str | None
    l_hind: float
    l_cons: float
    l_anchor: float
    boost: float
    reset_applied: bool
    y_out: torch.Tensor | None  # [1, pred_len, C] — None when skipped


def run_tta_step_affine(
    *,
    frozen_model: RevINDLinear,
    adapter: AffineAdapter,
    optimizer: torch.optim.Optimizer,
    loss_fn: HybridTTALoss,
    x_input: np.ndarray,             # [seq_len, C]  Global Scale
    x_recent: np.ndarray,            # [k, C]        Global Scale (Hindcast targets)
    y_prev: torch.Tensor | None,     # [1, pred_len, C] 이전 창 예측 (detached)
    rollback_guard: RollbackGuard,
    device: torch.device,
    max_grad_norm: float = 1.0,
    drift_gate_threshold: float = 0.0,
    mu_hist: float = 0.0,
    sigma_hist: float = 1.0,
    reset_threshold: float = float("inf"),
    n_inner: int = 1,
) -> TTAStepResultV2:
    """
    One TTA update step using the Affine Adapter strategy.

    Backbone (frozen_model) is never modified.  Only adapter.gamma / adapter.delta
    receive gradient updates.

    Space contract
    --------------
    frozen_model(x_input_t) outputs in the same space as x_recent.
    Both must be Global Scale (RevIN denorm is part of frozen_model.forward).
    y_true_k passed to loss_fn is x_recent unsqueezed to [1, k, C].

    Rollback / skip gates
    ---------------------
    1. Drift gate   — skip if |mean/std shift| < drift_gate_threshold
    2. Rollback     — skip if pre-step L_hind > threshold × rolling mean
    3. Anomaly gate — reset adapter if post-update L_hind > reset_threshold
    """
    # ── 전처리: numpy → tensor ────────────────────────────────────────────────
    x_input_t = torch.from_numpy(x_input).unsqueeze(0).float().to(device)  # [1, L, C]
    x_recent_t = (
        torch.from_numpy(x_recent).unsqueeze(0).float().to(device)         # [1, k, C]
    )

    # ── Gate 1: Drift gate ────────────────────────────────────────────────────
    if not should_adapt(
        torch.from_numpy(x_recent).float().to(device),
        mu_hist, sigma_hist, drift_gate_threshold,
    ):
        with torch.no_grad():
            y_out = adapter(frozen_model(x_input_t))
        rollback_guard.tracker.update(float("nan"))
        return TTAStepResultV2(
            skipped=True, skip_reason="drift_gate",
            l_hind=float("nan"), l_cons=float("nan"),
            l_anchor=float("nan"), boost=1.0, reset_applied=False,
            y_out=y_out.detach(),
        )

    # ── Pre-step L_hind (no grad) ─────────────────────────────────────────────
    with torch.no_grad():
        y_pre = adapter(frozen_model(x_input_t))                            # [1, H, C]
        k = x_recent_t.shape[1]
        l_hind_pre = (y_pre[:, :k, :] - x_recent_t).pow(2).mean().item()

    # ── Gate 2: Rollback guard ────────────────────────────────────────────────
    if rollback_guard.should_skip(l_hind_pre):
        rollback_guard.tracker.update(l_hind_pre)
        return TTAStepResultV2(
            skipped=True, skip_reason="rollback",
            l_hind=l_hind_pre, l_cons=float("nan"),
            l_anchor=float("nan"), boost=1.0, reset_applied=False,
            y_out=y_pre.detach(),
        )

    # ── Inner gradient steps ──────────────────────────────────────────────────
    logs: dict[str, float] = {}
    for _ in range(n_inner):
        optimizer.zero_grad()

        # Backbone forward: no grad_fn → gradients cannot reach frozen_model
        with torch.no_grad():
            y_hat = frozen_model(x_input_t)         # [1, H, C]

        # Affine adapter: Ŷ_final = γ ⊙ y_hat + δ  (only γ, δ get gradients)
        y_final = adapter(y_hat)                    # [1, H, C]

        loss, logs = loss_fn(
            y_curr=y_final,
            y_prev=y_prev,
            y_true_k=x_recent_t,
            gamma=adapter.gamma,
            delta=adapter.delta,
        )

        if not loss.requires_grad:
            # 안전장치: gradient 없으면 skip (예: L_hind=0이고 y_prev=None)
            break

        loss.backward()
        nn.utils.clip_grad_norm_(adapter.parameters(), max_norm=max_grad_norm)
        optimizer.step()

    # ── Gate 3: Anomaly gate — 오차 폭발 시 adapter 리셋 ─────────────────────
    reset_applied = logs.get("L_hind", 0.0) > reset_threshold
    if reset_applied:
        adapter.reset()

    rollback_guard.tracker.update(l_hind_pre)

    # ── 최종 예측 ─────────────────────────────────────────────────────────────
    with torch.no_grad():
        y_out = adapter(frozen_model(x_input_t))    # [1, H, C]

    return TTAStepResultV2(
        skipped=False,
        skip_reason=None,
        l_hind=logs.get("L_hind", float("nan")),
        l_cons=logs.get("L_cons", float("nan")),
        l_anchor=logs.get("L_anchor", float("nan")),
        boost=logs.get("boost", 1.0),
        reset_applied=reset_applied,
        y_out=y_out.detach(),
    )
