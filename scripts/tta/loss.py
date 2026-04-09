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
        y_hat: torch.Tensor,    # [1, pred_len, 1] in Global Scale
        x_recent: torch.Tensor, # [k, 1] in Global Scale
    ) -> torch.Tensor:
        # y_hat[0, :k, :] shape: [k, 1]
        pred_k = y_hat[0, : self.k, :]
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


class TTALoss(nn.Module):
    """
    Total TTA objective: L_TTA = L_recon + α * L_reg

    Returns (total_loss, l_recon, l_reg) for logging.
    """

    def __init__(
        self,
        k: int,
        alpha: float = 1.0,
        lambda0: float = 1.0,
        gamma: float = 1.0,
    ):
        super().__init__()
        self.hindcast = HindcastLoss(k)
        self.regularizer = DynamicRegularizer(lambda0=lambda0, gamma=gamma)
        self.alpha = alpha

    def forward(
        self,
        y_hat: torch.Tensor,
        x_recent: torch.Tensor,
        model: RevINDLinear,
        anchor_trend_w: torch.Tensor,
        anchor_season_w: torch.Tensor,
        mu_hist: float,
        sigma_hist: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        l_recon = self.hindcast(y_hat, x_recent)
        l_reg = self.regularizer(
            model, anchor_trend_w, anchor_season_w, x_recent, mu_hist, sigma_hist
        )
        total = l_recon + self.alpha * l_reg
        return total, l_recon, l_reg
