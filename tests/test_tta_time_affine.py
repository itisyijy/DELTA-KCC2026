from __future__ import annotations

import pytest
import torch

from scripts.tta.adapter import AffineAdapter
from scripts.tta.diagnostics import ClientTTADiagnostics
from scripts.tta.engine import TTAStepResultV2


PRED_LEN = 8
CHANNELS = 3


def test_channel_affine_shape_is_backward_compatible() -> None:
    adapter = AffineAdapter(CHANNELS)
    assert adapter.gamma.shape == (1, 1, CHANNELS)
    assert adapter.delta.shape == (1, 1, CHANNELS)


def test_time_affine_uses_horizon_specific_params() -> None:
    adapter = AffineAdapter(CHANNELS, pred_len=PRED_LEN, mode="time_affine")
    assert adapter.gamma.shape == (1, PRED_LEN, CHANNELS)
    assert adapter.delta.shape == (1, PRED_LEN, CHANNELS)


def test_time_affine_is_identity_at_init() -> None:
    adapter = AffineAdapter(CHANNELS, pred_len=PRED_LEN, mode="time_affine")
    y_hat = torch.randn(2, PRED_LEN, CHANNELS)
    assert torch.allclose(adapter(y_hat), y_hat)


def test_time_affine_reset_restores_identity() -> None:
    adapter = AffineAdapter(CHANNELS, pred_len=PRED_LEN, mode="time_affine")
    with torch.no_grad():
        adapter.gamma.fill_(2.0)
        adapter.delta.fill_(3.0)
    adapter.reset()
    assert torch.allclose(adapter.gamma, torch.ones(1, PRED_LEN, CHANNELS))
    assert torch.allclose(adapter.delta, torch.zeros(1, PRED_LEN, CHANNELS))


def test_time_affine_forward_broadcasts_over_batch() -> None:
    adapter = AffineAdapter(CHANNELS, pred_len=PRED_LEN, mode="time_affine")
    with torch.no_grad():
        adapter.gamma[:, 0, :].fill_(2.0)
        adapter.delta[:, 1, :].fill_(1.0)
    y_hat = torch.ones(4, PRED_LEN, CHANNELS)
    out = adapter(y_hat)
    assert out.shape == (4, PRED_LEN, CHANNELS)
    assert torch.allclose(out[:, 0, :], torch.full((4, CHANNELS), 2.0))
    assert torch.allclose(out[:, 1, :], torch.full((4, CHANNELS), 2.0))


def test_unknown_adapter_mode_raises() -> None:
    with pytest.raises(ValueError):
        AffineAdapter(CHANNELS, pred_len=PRED_LEN, mode="bad_mode")


def test_diagnostics_aggregate_time_affine_magnitudes() -> None:
    diag = ClientTTADiagnostics()
    result = TTAStepResultV2(
        skipped=False,
        skip_reason=None,
        l_hind=0.2,
        l_cons=0.1,
        l_anchor=0.05,
        boost=1.3,
        reset_applied=False,
        y_out=torch.zeros(1, PRED_LEN, CHANNELS),
    )
    diag.update(result, gamma_l1=0.02, delta_l1=0.03)
    stats = diag.as_dict()
    assert stats["adapt_rate"] == pytest.approx(1.0)
    assert stats["mean_gamma_l1"] == pytest.approx(0.02)
    assert stats["mean_delta_l1"] == pytest.approx(0.03)
    assert stats["final_gamma_l1"] == pytest.approx(0.02)
    assert stats["final_delta_l1"] == pytest.approx(0.03)


def test_diagnostics_track_hard_gate_skip() -> None:
    diag = ClientTTADiagnostics()
    result = TTAStepResultV2(
        skipped=True,
        skip_reason="hard_gate",
        l_hind=0.2,
        l_cons=float("nan"),
        l_anchor=float("nan"),
        boost=1.0,
        reset_applied=False,
        y_out=torch.zeros(1, PRED_LEN, CHANNELS),
    )
    diag.update(result, gamma_l1=0.0, delta_l1=0.0)
    stats = diag.as_dict()
    assert stats["adapt_rate"] == pytest.approx(0.0)
    assert stats["hard_gate_skip_rate"] == pytest.approx(1.0)


def test_diagnostics_track_accept_gate_skip() -> None:
    diag = ClientTTADiagnostics()
    result = TTAStepResultV2(
        skipped=True,
        skip_reason="accept_gate",
        l_hind=0.2,
        l_cons=float("nan"),
        l_anchor=float("nan"),
        boost=1.0,
        reset_applied=False,
        y_out=torch.zeros(1, PRED_LEN, CHANNELS),
    )
    diag.update(result, gamma_l1=0.0, delta_l1=0.0)
    stats = diag.as_dict()
    assert stats["adapt_rate"] == pytest.approx(0.0)
    assert stats["accept_gate_skip_rate"] == pytest.approx(1.0)
