"""
TTA model preparation utilities.

Two strategies are supported:

(Legacy) Weight-Update TTA  — prepare_tta_model
    RevIN affine params (γ_RevIN, β_RevIN) → Freeze
    W_trend, W_season → update targets
    Used by run_tta_step / run_fed_tta_loop (existing approach).

(New) Affine-Adapter TTA  — AffineAdapter + prepare_frozen_backbone
    ALL backbone params → completely frozen (no_grad)
    Separate AffineAdapter(γ, δ ∈ R^{1×1×C}) → only update target
    Used by run_tta_step_affine / run_local_tta_loop (new approach).

    학술적 의미:
      "TTA는 Temporal Dynamics를 DLinear에 전적으로 위임하고,
       오직 채널별 공간적 Level/Amplitude 변화(Spatial Shift)만 보정한다."

    Ŷ_final = γ ⊙ Ŷ_DLinear + δ
      γ ∈ R^{1×1×C}  (init=1, Amplitude 변화 대응)
      δ ∈ R^{1×1×C}  (init=0, Level 변화 대응)
"""
from __future__ import annotations

import torch
import torch.nn as nn

from scripts.models.revin_dlinear import RevINDLinear
from scripts.tta.policy import configure_tta_model


# ─────────────────────────────────────────────────────────────────────────────
# New: Affine Adapter
# ─────────────────────────────────────────────────────────────────────────────

class AffineAdapter(nn.Module):
    """
    Channel-wise affine output adapter for TTA.

    Ŷ_final = γ ⊙ Ŷ_backbone + δ

    Parameters
    ----------
    n_channels : int
        Number of output channels (C).

    Shapes
    ------
    gamma : [1, 1, C]  broadcast over (batch, time) dims  — init 1
    delta : [1, 1, C]  broadcast over (batch, time) dims  — init 0

    Design rationale
    ----------------
    Time-invariant (shared across H steps) so that k-step Hindcast gradients
    propagate uniformly over the full pred_len — no Disjoint artifact at step k.
    Parameter count = 2C (e.g. 2 scalars for univariate), making overfitting
    practically impossible.
    """

    def __init__(self, n_channels: int) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(1, 1, n_channels))
        self.delta = nn.Parameter(torch.zeros(1, 1, n_channels))

    def forward(self, y_hat: torch.Tensor) -> torch.Tensor:
        """Apply affine transform.  y_hat: [B, H, C] → [B, H, C]."""
        return self.gamma * y_hat + self.delta

    @torch.no_grad()
    def reset(self) -> None:
        """Restore identity transform (Anomaly Gate)."""
        self.gamma.fill_(1.0)
        self.delta.fill_(0.0)


def prepare_frozen_backbone(model: RevINDLinear) -> RevINDLinear:
    """
    Freeze every parameter of the backbone for Affine-Adapter TTA.

    The returned model should always run in eval() mode during TTA so that
    RevIN instance-norm statistics are computed from the current window only.
    """
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Legacy: Weight-Update TTA
# ─────────────────────────────────────────────────────────────────────────────

def prepare_tta_model(
    model: RevINDLinear,
    update_scope: str = "all",
) -> tuple[RevINDLinear, torch.Tensor, torch.Tensor]:
    """
    Prepare model for TTA:
      1. Freeze RevIN affine parameters (affine_weight, affine_bias).
      2. Ensure DLinear linear_seasonal and linear_trend are trainable.
      3. Return detached anchor copies of trend and seasonal weights.

    Returns:
        model              — same model, modified in-place
        anchor_trend_w     — frozen copy of W_trend  (for L_reg reference)
        anchor_season_w    — frozen copy of W_season (for L_reg reference)

    For individual=True models, weights are stacked into a single tensor
    [C, pred_len, seq_len] for efficient L_reg computation.
    """
    configure_tta_model(model, update_scope)

    # Capture anchors
    linear_trend = model.linear_trend
    linear_season = model.linear_seasonal

    if isinstance(linear_trend, nn.ModuleList):
        anchor_trend_w = torch.stack(
            [layer.weight.detach().clone() for layer in linear_trend]
        )
        anchor_season_w = torch.stack(
            [layer.weight.detach().clone() for layer in linear_season]
        )
    else:
        anchor_trend_w = linear_trend.weight.detach().clone()
        anchor_season_w = linear_season.weight.detach().clone()

    return model, anchor_trend_w, anchor_season_w
