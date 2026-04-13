from __future__ import annotations

import pytest
import torch

from scripts.tta.adapter import AffineAdapter
from scripts.tta.diagnostics import ClientTTADiagnostics
from scripts.tta.engine import TTAStepResultV2


PRED_LEN = 8
CHANNELS = 2


def test_horizon_conv_param_shapes() -> None:
    adapter = AffineAdapter(CHANNELS, pred_len=PRED_LEN, mode="horizon_conv")
    assert adapter.gamma.shape == (1, 3, CHANNELS)
    assert adapter.delta.shape == (1, 1, CHANNELS)


def test_horizon_conv_is_identity_at_init() -> None:
    adapter = AffineAdapter(CHANNELS, pred_len=PRED_LEN, mode="horizon_conv")
    y_hat = torch.randn(4, PRED_LEN, CHANNELS)
    assert torch.allclose(adapter(y_hat), y_hat)


def test_horizon_conv_reset_restores_identity() -> None:
    adapter = AffineAdapter(CHANNELS, pred_len=PRED_LEN, mode="horizon_conv")
    with torch.no_grad():
        adapter.gamma[0, 0, :].fill_(1.3)
        adapter.gamma[0, 1, :].fill_(0.8)
        adapter.delta.fill_(2.0)
    adapter.reset()
    assert torch.allclose(adapter.gamma, torch.ones(1, 3, CHANNELS))
    assert torch.allclose(adapter.delta, torch.zeros(1, 1, CHANNELS))


def test_horizon_conv_shifts_local_mass() -> None:
    adapter = AffineAdapter(1, pred_len=PRED_LEN, mode="horizon_conv")
    with torch.no_grad():
        adapter.gamma[0, :, 0] = torch.tensor([1.5, 1.0, 1.0])
    y_hat = torch.zeros(1, PRED_LEN, 1)
    y_hat[0, 3, 0] = 1.0
    out = adapter(y_hat)[0, :, 0]
    assert out[4].item() > 0.0
    assert out[3].item() > 0.0


def test_horizon_conv_diagnostics_reuse_gamma_delta_fields() -> None:
    diag = ClientTTADiagnostics()
    result = TTAStepResultV2(
        skipped=False,
        skip_reason=None,
        l_hind=0.1,
        l_cons=0.0,
        l_anchor=0.0,
        boost=1.0,
        reset_applied=False,
        y_out=torch.zeros(1, PRED_LEN, 1),
    )
    diag.update(result, gamma_l1=0.03, delta_l1=0.01)
    stats = diag.as_dict()
    assert stats["mean_gamma_l1"] == pytest.approx(0.03)
    assert stats["mean_delta_l1"] == pytest.approx(0.01)
