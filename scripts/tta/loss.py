"""
TTA loss functions for the DLinear FED-TTA framework.

Plan.md §2.3:
    L_TTA = L_recon + α * L_reg

    L_recon = (1/k) * Σ_i ||ŷ_i - x_recent_i||²
        (in Global Scale space, after RevIN denorm)

    L_reg = λ_trend * ||W_trend_TTA - W_trend_FL||²
          + λ_season * ||W_season_TTA - W_season_FL||²

    λ_trend  = λ₀ · exp(-γ · |μ_curr - μ_hist| / σ_hist)
    λ_season = λ₀ · exp(-γ · |σ_curr - σ_hist| / σ_hist)

    Statistics (μ, σ) extracted from x_recent in Global Scale space.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from scripts.models.revin_dlinear import RevINDLinear


class HindcastLoss(nn.Module):
    """
    Shifted-Window Hindcast (Self-Reconstruction) Loss.

    Compares the first k steps of the model's prediction with the most
    recently observed k steps (x_recent).  Both tensors are in Global Scale.

    Args:
        k: Hindcast length.  Must satisfy k <= pred_len.
    """

    def __init__(self, k: int):
        super().__init__()
        self.k = k

    def forward(
        self,
        y_hat: torch.Tensor,           # [1, pred_len, 1] in Global Scale
        x_recent: torch.Tensor,        # [k, 1] in Global Scale
        mu_hist: float | None = None,  # train-split mean (Global Scale) — enables masking
        sigma_hist: float | None = None,  # train-split std (Global Scale)
        active_threshold: float = 0.0, # delta above GS zero_floor; steps below are masked
    ) -> torch.Tensor:
        """Compute MSE loss over hindcast steps, optionally masking near-zero steps.

        When mu_hist/sigma_hist are provided, steps at or below
        (zero_floor + active_threshold) in Global Scale are excluded from
        the loss so that near-zero (nighttime) readings don't dominate.
        zero_floor = -mu_hist / sigma_hist  (original-scale 0 in GS).
        """
        # y_hat[0, :k, :] shape: [k, 1]
        pred_k = y_hat[0, : self.k, :]

        if mu_hist is not None and sigma_hist is not None:
            zero_floor = -mu_hist / max(float(sigma_hist), 1e-8)
            # keep steps where actual value is above zero_floor + threshold
            mask = (x_recent > zero_floor + active_threshold).float()
            active_count = mask.sum()
            if active_count == 0:
                return torch.zeros(1, device=y_hat.device)
            return (mask * (pred_k - x_recent).pow(2)).sum() / active_count

        return torch.mean((pred_k - x_recent) ** 2)


class DynamicRegularizer(nn.Module):
    """
    Continuous Dynamic Component Preservation Penalty.

    Regularises TTA updates via exponentially decaying coefficients that
    reflect the magnitude of the current distribution shift.

    λ_trend  = λ₀ · exp(-γ · |μ_curr - μ_hist| / σ_hist)
    λ_season = λ₀ · exp(-γ · |σ_curr - σ_hist| / σ_hist)
    """

    def __init__(self, lambda0: float = 1.0, gamma: float = 1.0):
        super().__init__()
        self.lambda0 = lambda0
        self.gamma = gamma

    def forward(
        self,
        model: RevINDLinear,
        anchor_trend_w: torch.Tensor,    # frozen copy of W_trend from FL ckpt
        anchor_season_w: torch.Tensor,   # frozen copy of W_season from FL ckpt
        x_recent: torch.Tensor,          # [k, 1] Global Scale (before RevIN)
        mu_hist: float,                  # train-split mean (Global Scale)
        sigma_hist: float,               # train-split std  (Global Scale)
    ) -> torch.Tensor:
        # --- Compute current statistics from x_recent ---
        mu_curr = x_recent.mean().item()
        sigma_curr = x_recent.std().item()

        eps = 1e-8
        sigma_hist = max(sigma_hist, eps)

        lambda_trend = self.lambda0 * torch.exp(
            torch.tensor(
                -self.gamma * abs(mu_curr - mu_hist) / sigma_hist,
                dtype=torch.float32,
            )
        )
        lambda_season = self.lambda0 * torch.exp(
            torch.tensor(
                -self.gamma * abs(sigma_curr - sigma_hist) / sigma_hist,
                dtype=torch.float32,
            )
        )

        # --- Weight deviations ---
        linear_trend = model.linear_trend
        linear_season = model.linear_seasonal

        if isinstance(linear_trend, nn.ModuleList):
            # individual=True: stacked weights [C, pred_len, seq_len]
            w_trend = torch.stack([l.weight for l in linear_trend])
            w_season = torch.stack([l.weight for l in linear_season])
        else:
            w_trend = linear_trend.weight
            w_season = linear_season.weight

        reg = (
            lambda_trend * (w_trend - anchor_trend_w).pow(2).sum()
            + lambda_season * (w_season - anchor_season_w).pow(2).sum()
        )
        return reg


# ─────────────────────────────────────────────────────────────────────────────
# New: Hybrid TTA Loss (Affine-Adapter 전용)
# ─────────────────────────────────────────────────────────────────────────────

class HybridTTALoss(nn.Module):
    """
    Hybrid TTA objective for Affine-Adapter TTA.

    L_TTA = α_eff · L_cons + β_eff · L_hind + λ · L_anchor

    Components
    ----------
    L_hind  [주신호] 최근 k 스텝 실측값과의 MSE → 현실 Grounding
              ``y_curr[:, :k, :] vs y_true_k``
    L_cons  [보조]   연속 창간 예측 급변 억제 (Temporal Smoothness)
              ``y_curr[:, :H-1, :] vs sg(y_prev[:, 1:H, :])``
    L_anchor [안전망] γ → 1, δ → 0 유지 (Long-term Drift 방지)
              ``(γ-1)² + δ²  normalized by C``

    Bounded Adaptive Weighting
    --------------------------
    도메인 시프트가 강할수록 실데이터 의존도↑, 자기 일관성↓:

        boost   = clip(1 + ρ · sg(L_hind), 1, max_boost)
        α_eff   = α / boost
        β_eff   = β × boost

    Parameters
    ----------
    alpha         : L_cons base weight (보조 역할; default 0.3)
    beta          : L_hind base weight (주신호;  default 1.0)
    lambda_anchor : L_anchor weight                (default 0.1)
    sensitivity   : adaptive boost 강도 ρ          (default 1.0)
    max_boost     : boost 상한 — Gradient Explosion 방어 (default 5.0)
    """

    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 1.0,
        lambda_anchor: float = 0.1,
        sensitivity: float = 1.0,
        max_boost: float = 5.0,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.lambda_anchor = lambda_anchor
        self.sensitivity = sensitivity
        self.max_boost = max_boost

    def forward(
        self,
        y_curr: torch.Tensor,                    # [B, H, C]  현재 예측 (adapter 포함)
        y_prev: torch.Tensor | None,             # [B, H, C]  이전 예측 (detached)
        y_true_k: torch.Tensor | None,           # [B, k, C]  최근 k 스텝 실측값
        gamma: nn.Parameter,                     # [1, 1, C]
        delta: nn.Parameter,                     # [1, 1, C]
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Returns
        -------
        loss : scalar tensor (has grad_fn through y_curr / gamma / delta)
        logs : dict with L_hind, L_cons, L_anchor, alpha_eff, beta_eff, boost
        """
        device = y_curr.device

        # ── L_hind: 실측값 Grounding ──────────────────────────────────────────
        if y_true_k is not None:
            k = y_true_k.shape[1]
            L_hind = (y_curr[:, :k, :] - y_true_k).pow(2).mean()
        else:
            L_hind = y_curr.new_tensor(0.0)

        # ── L_cons: 보조 정규화 — 급변 억제 ──────────────────────────────────
        H = y_curr.shape[1]
        if y_prev is not None and H > 1:
            # curr[:,  0:H-1] ↔ prev[:,  1:H] (동일 미래 타임스텝)
            L_cons = (
                y_curr[:, :H - 1, :] - y_prev[:, 1:, :].detach()
            ).pow(2).mean()
        else:
            L_cons = y_curr.new_tensor(0.0)

        # ── L_anchor: 항등 변환 근방 유지 (γ→1, δ→0) ─────────────────────────
        C = gamma.shape[-1]
        L_anchor = (
            (gamma - 1.0).pow(2).sum() + delta.pow(2).sum()
        ) / C

        # ── Bounded Adaptive Weighting ────────────────────────────────────────
        e = L_hind.detach().item()
        boost = min(1.0 + self.sensitivity * e, self.max_boost)
        alpha_eff = self.alpha / boost
        beta_eff = self.beta * boost

        loss = (
            alpha_eff * L_cons
            + beta_eff * L_hind
            + self.lambda_anchor * L_anchor
        )

        logs: dict[str, float] = {
            "L_hind": L_hind.item(),
            "L_cons": L_cons.item(),
            "L_anchor": L_anchor.item(),
            "alpha_eff": alpha_eff,
            "beta_eff": beta_eff,
            "boost": boost,
        }
        return loss, logs


# ─────────────────────────────────────────────────────────────────────────────
# Legacy: Weight-Update TTA Loss
# ─────────────────────────────────────────────────────────────────────────────

class TTALoss(nn.Module):
    """
    Total TTA objective: L_TTA = L_recon + α * L_reg + λ_func * L_func

    L_func = ||ŷ_TTA - ŷ_anchor||²  (functional regularization in output space)

    Returns (total_loss, l_recon, l_reg) for logging.
    """

    def __init__(
        self,
        k: int,
        alpha: float = 1.0,
        lambda0: float = 1.0,
        gamma: float = 1.0,
        lambda_func: float = 0.0,
        hindcast_mask_threshold: float = 0.0,
    ):
        super().__init__()
        self.hindcast = HindcastLoss(k)
        self.regularizer = DynamicRegularizer(lambda0=lambda0, gamma=gamma)
        self.alpha = alpha
        self.lambda_func = lambda_func
        self.hindcast_mask_threshold = hindcast_mask_threshold

    def forward(
        self,
        y_hat: torch.Tensor,
        x_recent: torch.Tensor,
        model: RevINDLinear,
        anchor_trend_w: torch.Tensor,
        anchor_season_w: torch.Tensor,
        mu_hist: float,
        sigma_hist: float,
        y_hat_anchor: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        l_recon = self.hindcast(y_hat, x_recent, mu_hist, sigma_hist, self.hindcast_mask_threshold)
        l_reg = self.regularizer(
            model, anchor_trend_w, anchor_season_w, x_recent, mu_hist, sigma_hist
        )
        l_func = (
            torch.mean((y_hat - y_hat_anchor.detach()) ** 2)
            if self.lambda_func > 0.0 and y_hat_anchor is not None
            else torch.zeros(1, device=y_hat.device)
        )
        total = l_recon + self.alpha * l_reg + self.lambda_func * l_func
        return total, l_recon, l_reg
