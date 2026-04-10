from __future__ import annotations

import torch

from scripts.models.revin_dlinear import RevINDLinear
from scripts.tta.adapter import prepare_tta_model
from scripts.tta.policy import compute_drift_score, get_tta_parameters, should_adapt


def _model() -> RevINDLinear:
    return RevINDLinear(8, 4, 1, kernel_size=3, individual=False, revin_affine=True)


def test_prepare_tta_model_norm_scope_updates_only_revin() -> None:
    model, _, _ = prepare_tta_model(_model(), "norm")
    trainable = {name for name, param in model.named_parameters() if param.requires_grad}
    assert trainable == {"revin.affine_weight", "revin.affine_bias"}
    assert len(get_tta_parameters(model)) == 2


def test_prepare_tta_model_trend_scope_freezes_seasonal_branch() -> None:
    model, _, _ = prepare_tta_model(_model(), "trend")
    trainable = {name for name, param in model.named_parameters() if param.requires_grad}
    assert trainable == {"dlinear.linear_trend.weight", "dlinear.linear_trend.bias"}


def test_drift_gate_uses_recent_mean_and_scale_shift() -> None:
    recent = torch.full((4, 1), 3.0)
    assert compute_drift_score(recent, mu_hist=0.0, sigma_hist=1.0) >= 3.0
    assert should_adapt(recent, mu_hist=0.0, sigma_hist=1.0, threshold=2.0)
    assert not should_adapt(recent, mu_hist=3.0, sigma_hist=0.0, threshold=1.0)
