"""
TTA model preparation: freeze RevIN, expose DLinear weights as update targets.

Plan.md §2.1:
    RevIN affine params (γ_RevIN, β_RevIN) → Freeze during TTA
    W_trend, W_season → update targets
"""
from __future__ import annotations

import torch
import torch.nn as nn

from scripts.models.revin_dlinear import RevINDLinear
from scripts.tta.policy import configure_tta_model


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
