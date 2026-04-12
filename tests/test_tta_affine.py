"""
Tests for Affine-Adapter TTA (new approach).

Covers:
  1. AffineAdapter — 형상, 초기값, reset, gradient 격리
  2. HybridTTALoss — L_hind / L_cons / L_anchor 계산, Bounded Adaptive Weighting
  3. prepare_frozen_backbone — 백본 완전 동결 확인
  4. run_tta_step_affine — 1-step 업데이트 후 adapter 변화 확인
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from scripts.models.revin_dlinear import RevINDLinear
from scripts.tta.adapter import AffineAdapter, prepare_frozen_backbone
from scripts.tta.engine import ReconTracker, RollbackGuard, run_tta_step_affine
from scripts.tta.loop import _client_step_worker
from scripts.tta.loss import HybridTTALoss


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

SEQ_LEN  = 16
PRED_LEN = 8
K        = 4
CHANNELS = 3


def _model() -> RevINDLinear:
    return RevINDLinear(
        seq_len=SEQ_LEN, pred_len=PRED_LEN, channels=CHANNELS,
        kernel_size=3, individual=False, revin_affine=True,
    )


def _adapter() -> AffineAdapter:
    return AffineAdapter(n_channels=CHANNELS)


def _loss_fn(**kwargs) -> HybridTTALoss:
    defaults = dict(alpha=0.3, beta=1.0, lambda_anchor=0.1, sensitivity=1.0, max_boost=5.0)
    defaults.update(kwargs)
    return HybridTTALoss(**defaults)


# ─────────────────────────────────────────────────────────────────────────────
# 1. AffineAdapter
# ─────────────────────────────────────────────────────────────────────────────

def test_affine_adapter_initial_values() -> None:
    """gamma=1, delta=0 → forward는 항등 변환이어야 한다."""
    adapter = _adapter()
    assert torch.allclose(adapter.gamma, torch.ones(1, 1, CHANNELS))
    assert torch.allclose(adapter.delta, torch.zeros(1, 1, CHANNELS))


def test_affine_adapter_forward_shape() -> None:
    adapter = _adapter()
    y_hat = torch.randn(2, PRED_LEN, CHANNELS)
    out = adapter(y_hat)
    assert out.shape == y_hat.shape


def test_affine_adapter_is_identity_at_init() -> None:
    adapter = _adapter()
    y_hat = torch.randn(1, PRED_LEN, CHANNELS)
    assert torch.allclose(adapter(y_hat), y_hat)


def test_affine_adapter_reset_restores_identity() -> None:
    adapter = _adapter()
    with torch.no_grad():
        adapter.gamma.fill_(2.0)
        adapter.delta.fill_(3.0)
    adapter.reset()
    assert torch.allclose(adapter.gamma, torch.ones(1, 1, CHANNELS))
    assert torch.allclose(adapter.delta, torch.zeros(1, 1, CHANNELS))


def test_affine_adapter_only_has_two_params() -> None:
    adapter = _adapter()
    params = list(adapter.parameters())
    assert len(params) == 2  # gamma, delta


def test_affine_adapter_param_count() -> None:
    """파라미터 총 수 = 2 * C (시간 축 독립)."""
    adapter = _adapter()
    total = sum(p.numel() for p in adapter.parameters())
    assert total == 2 * CHANNELS


# ─────────────────────────────────────────────────────────────────────────────
# 2. prepare_frozen_backbone
# ─────────────────────────────────────────────────────────────────────────────

def test_prepare_frozen_backbone_freezes_all_params() -> None:
    model = _model()
    prepare_frozen_backbone(model)
    for name, param in model.named_parameters():
        assert not param.requires_grad, f"{name} should be frozen"


def test_prepare_frozen_backbone_sets_eval_mode() -> None:
    model = _model()
    model.train()
    prepare_frozen_backbone(model)
    assert not model.training


def test_frozen_backbone_no_grad_flows_to_model() -> None:
    """adapter 업데이트 시 frozen_model 파라미터에 gradient가 쌓이지 않아야 한다."""
    model = prepare_frozen_backbone(_model())
    adapter = _adapter()
    opt = torch.optim.Adam(adapter.parameters(), lr=1e-3)

    x = torch.randn(1, SEQ_LEN, CHANNELS)
    opt.zero_grad()
    with torch.no_grad():
        y_hat = model(x)
    y_final = adapter(y_hat)
    loss = y_final.sum()
    loss.backward()

    for param in model.parameters():
        assert param.grad is None, "frozen_model must not receive gradients"


# ─────────────────────────────────────────────────────────────────────────────
# 3. HybridTTALoss
# ─────────────────────────────────────────────────────────────────────────────

def _make_adapter_params(gamma_val: float = 1.0, delta_val: float = 0.0):
    adapter = _adapter()
    with torch.no_grad():
        adapter.gamma.fill_(gamma_val)
        adapter.delta.fill_(delta_val)
    return adapter.gamma, adapter.delta


def test_hybrid_loss_all_zero_at_identity_no_prev_no_true() -> None:
    """y_prev=None, y_true_k=None → L_hind=L_cons=0, L_anchor도 0."""
    loss_fn = _loss_fn()
    y_curr = torch.randn(1, PRED_LEN, CHANNELS)
    gamma, delta = _make_adapter_params(1.0, 0.0)
    _, logs = loss_fn(y_curr=y_curr, y_prev=None, y_true_k=None,
                      gamma=gamma, delta=delta)
    assert logs["L_hind"]   == pytest.approx(0.0)
    assert logs["L_cons"]   == pytest.approx(0.0)
    assert logs["L_anchor"] == pytest.approx(0.0)


def test_hybrid_loss_l_hind_matches_mse() -> None:
    y_curr   = torch.zeros(1, PRED_LEN, CHANNELS)
    y_true_k = torch.ones(1, K, CHANNELS)
    gamma, delta = _make_adapter_params()
    loss_fn = _loss_fn()
    _, logs = loss_fn(y_curr=y_curr, y_prev=None, y_true_k=y_true_k,
                      gamma=gamma, delta=delta)
    assert logs["L_hind"] == pytest.approx(1.0)


def test_hybrid_loss_l_anchor_nonzero_when_gamma_shifted() -> None:
    gamma, delta = _make_adapter_params(gamma_val=2.0, delta_val=0.0)
    loss_fn = _loss_fn(lambda_anchor=1.0)
    y_curr = torch.zeros(1, PRED_LEN, CHANNELS)
    _, logs = loss_fn(y_curr=y_curr, y_prev=None, y_true_k=None,
                      gamma=gamma, delta=delta)
    # L_anchor = (gamma-1)^2 / C = 1^2 = 1.0
    assert logs["L_anchor"] == pytest.approx(1.0)


def test_hybrid_loss_l_cons_computed_with_prev() -> None:
    y_curr = torch.zeros(1, PRED_LEN, CHANNELS)
    y_prev = torch.ones(1, PRED_LEN, CHANNELS).detach()
    gamma, delta = _make_adapter_params()
    loss_fn = _loss_fn()
    _, logs = loss_fn(y_curr=y_curr, y_prev=y_prev, y_true_k=None,
                      gamma=gamma, delta=delta)
    assert logs["L_cons"] > 0.0


def test_hybrid_loss_boost_clipped_at_max_boost() -> None:
    """L_hind가 매우 클 때 boost가 max_boost를 넘지 않아야 한다."""
    max_boost = 3.0
    loss_fn = _loss_fn(beta=1.0, sensitivity=1000.0, max_boost=max_boost)
    y_curr   = torch.zeros(1, PRED_LEN, CHANNELS)
    y_true_k = torch.full((1, K, CHANNELS), 100.0)   # MSE = 10000
    gamma, delta = _make_adapter_params()
    _, logs = loss_fn(y_curr=y_curr, y_prev=None, y_true_k=y_true_k,
                      gamma=gamma, delta=delta)
    assert logs["boost"] == pytest.approx(max_boost)
    assert logs["beta_eff"] == pytest.approx(loss_fn.beta * max_boost)


def test_hybrid_loss_boost_increases_beta_eff_when_hind_large() -> None:
    loss_fn = _loss_fn(alpha=0.3, beta=1.0, sensitivity=1.0, max_boost=5.0)
    y_curr   = torch.zeros(1, PRED_LEN, CHANNELS)
    y_true_k = torch.ones(1, K, CHANNELS)   # L_hind = 1.0 → boost = 2.0
    gamma, delta = _make_adapter_params()
    _, logs = loss_fn(y_curr=y_curr, y_prev=None, y_true_k=y_true_k,
                      gamma=gamma, delta=delta)
    assert logs["boost"]     == pytest.approx(2.0)
    assert logs["beta_eff"]  == pytest.approx(2.0)
    assert logs["alpha_eff"] == pytest.approx(0.3 / 2.0)


def test_hybrid_loss_has_grad_fn() -> None:
    """loss는 gamma/delta에 대한 gradient를 가져야 한다."""
    adapter  = _adapter()
    loss_fn  = _loss_fn()
    y_curr   = adapter(torch.randn(1, PRED_LEN, CHANNELS).detach())
    y_true_k = torch.randn(1, K, CHANNELS)
    loss, _  = loss_fn(y_curr=y_curr, y_prev=None, y_true_k=y_true_k,
                       gamma=adapter.gamma, delta=adapter.delta)
    loss.backward()
    assert adapter.gamma.grad is not None
    assert adapter.delta.grad is not None


# ─────────────────────────────────────────────────────────────────────────────
# 4. run_tta_step_affine
# ─────────────────────────────────────────────────────────────────────────────

def _make_step_inputs():
    x_input  = np.random.randn(SEQ_LEN, CHANNELS).astype(np.float32)
    x_recent = np.random.randn(K, CHANNELS).astype(np.float32)
    return x_input, x_recent


def test_tta_step_affine_updates_adapter() -> None:
    """TTA step 이후 adapter 파라미터가 초기값에서 변해야 한다."""
    model   = prepare_frozen_backbone(_model())
    adapter = _adapter()
    gamma_before = adapter.gamma.clone()
    delta_before = adapter.delta.clone()

    opt      = torch.optim.Adam(adapter.parameters(), lr=1e-2)
    loss_fn  = _loss_fn()
    guard    = RollbackGuard(threshold=1e9, tracker=ReconTracker(20))
    x_input, x_recent = _make_step_inputs()

    result = run_tta_step_affine(
        frozen_model=model, adapter=adapter, optimizer=opt,
        loss_fn=loss_fn, x_input=x_input, x_recent=x_recent,
        y_prev=None, rollback_guard=guard, device=torch.device("cpu"),
    )

    assert not result.skipped
    changed = (
        not torch.allclose(adapter.gamma, gamma_before)
        or not torch.allclose(adapter.delta, delta_before)
    )
    assert changed, "At least one adapter parameter should have changed"


def test_tta_step_affine_hard_gate_skips_easy_window() -> None:
    model = prepare_frozen_backbone(_model())
    adapter = _adapter()
    opt = torch.optim.Adam(adapter.parameters(), lr=1e-2)
    loss_fn = _loss_fn()
    guard = RollbackGuard(threshold=1e9, tracker=ReconTracker(20))
    for loss_val in (1.0, 1.0, 1.0, 1.0):
        guard.tracker.update(loss_val)
    x_input = np.zeros((SEQ_LEN, CHANNELS), dtype=np.float32)
    x_recent = np.zeros((K, CHANNELS), dtype=np.float32)

    result = run_tta_step_affine(
        frozen_model=model, adapter=adapter, optimizer=opt,
        loss_fn=loss_fn, x_input=x_input, x_recent=x_recent,
        y_prev=None, rollback_guard=guard, device=torch.device("cpu"),
        hard_gate_scale=1.05, hard_gate_min_history=4,
    )

    assert result.skipped
    assert result.skip_reason == "hard_gate"


def test_recon_tracker_ignores_nan_in_rolling_mean() -> None:
    tracker = ReconTracker(4)
    tracker.update(float("nan"))
    tracker.update(2.0)
    tracker.update(4.0)
    assert tracker.finite_count() == 2
    assert tracker.rolling_mean() == pytest.approx(3.0)


def test_tta_step_affine_does_not_update_frozen_model() -> None:
    """TTA step 이후 frozen_model 파라미터가 변하지 않아야 한다."""
    model   = prepare_frozen_backbone(_model())
    adapter = _adapter()
    params_before = {n: p.clone() for n, p in model.named_parameters()}

    opt     = torch.optim.Adam(adapter.parameters(), lr=1e-2)
    loss_fn = _loss_fn()
    guard   = RollbackGuard(threshold=1e9, tracker=ReconTracker(20))
    x_input, x_recent = _make_step_inputs()

    run_tta_step_affine(
        frozen_model=model, adapter=adapter, optimizer=opt,
        loss_fn=loss_fn, x_input=x_input, x_recent=x_recent,
        y_prev=None, rollback_guard=guard, device=torch.device("cpu"),
    )

    for name, param in model.named_parameters():
        assert torch.allclose(param, params_before[name]), \
            f"frozen param {name} changed after TTA step"


def test_tta_step_affine_returns_valid_y_out_shape() -> None:
    model   = prepare_frozen_backbone(_model())
    adapter = _adapter()
    opt     = torch.optim.Adam(adapter.parameters(), lr=1e-3)
    loss_fn = _loss_fn()
    guard   = RollbackGuard(threshold=1e9, tracker=ReconTracker(20))
    x_input, x_recent = _make_step_inputs()

    result = run_tta_step_affine(
        frozen_model=model, adapter=adapter, optimizer=opt,
        loss_fn=loss_fn, x_input=x_input, x_recent=x_recent,
        y_prev=None, rollback_guard=guard, device=torch.device("cpu"),
    )

    assert result.y_out is not None
    assert result.y_out.shape == (1, PRED_LEN, CHANNELS)


def test_tta_step_affine_skips_on_drift_gate() -> None:
    """drift_gate_threshold가 매우 높으면 항상 skip되어야 한다."""
    model   = prepare_frozen_backbone(_model())
    adapter = _adapter()
    opt     = torch.optim.Adam(adapter.parameters(), lr=1e-3)
    loss_fn = _loss_fn()
    guard   = RollbackGuard(threshold=1e9, tracker=ReconTracker(20))
    x_input, x_recent = _make_step_inputs()

    result = run_tta_step_affine(
        frozen_model=model, adapter=adapter, optimizer=opt,
        loss_fn=loss_fn, x_input=x_input, x_recent=x_recent,
        y_prev=None, rollback_guard=guard, device=torch.device("cpu"),
        drift_gate_threshold=1e9,  # 절대 넘을 수 없는 임계값
    )

    assert result.skipped
    assert result.skip_reason == "drift_gate"


def test_tta_step_affine_anomaly_gate_resets_adapter() -> None:
    """reset_threshold=0 → 항상 reset → adapter가 항등 변환으로 복귀해야 한다."""
    model   = prepare_frozen_backbone(_model())
    adapter = _adapter()
    with torch.no_grad():
        adapter.gamma.fill_(2.0)
        adapter.delta.fill_(3.0)

    opt     = torch.optim.Adam(adapter.parameters(), lr=1e-3)
    loss_fn = _loss_fn()
    guard   = RollbackGuard(threshold=1e9, tracker=ReconTracker(20))
    x_input, x_recent = _make_step_inputs()

    run_tta_step_affine(
        frozen_model=model, adapter=adapter, optimizer=opt,
        loss_fn=loss_fn, x_input=x_input, x_recent=x_recent,
        y_prev=None, rollback_guard=guard, device=torch.device("cpu"),
        reset_threshold=0.0,  # 항상 reset 발동
    )

    assert torch.allclose(adapter.gamma, torch.ones(1, 1, CHANNELS))
    assert torch.allclose(adapter.delta, torch.zeros(1, 1, CHANNELS))


def test_tta_step_affine_temporal_consistency_with_y_prev() -> None:
    """y_prev가 있을 때 L_cons가 기록되어야 한다."""
    model   = prepare_frozen_backbone(_model())
    adapter = _adapter()
    opt     = torch.optim.Adam(adapter.parameters(), lr=1e-3)
    loss_fn = _loss_fn()
    guard   = RollbackGuard(threshold=1e9, tracker=ReconTracker(20))
    x_input, x_recent = _make_step_inputs()

    # 첫 스텝
    result1 = run_tta_step_affine(
        frozen_model=model, adapter=adapter, optimizer=opt,
        loss_fn=loss_fn, x_input=x_input, x_recent=x_recent,
        y_prev=None, rollback_guard=guard, device=torch.device("cpu"),
    )
    # 두 번째 스텝: y_prev를 첫 스텝의 y_out으로 전달
    result2 = run_tta_step_affine(
        frozen_model=model, adapter=adapter, optimizer=opt,
        loss_fn=loss_fn, x_input=x_input, x_recent=x_recent,
        y_prev=result1.y_out, rollback_guard=guard, device=torch.device("cpu"),
    )

    assert not result2.skipped
    # L_cons는 y_prev가 있을 때 NaN이 아니어야 함
    assert not np.isnan(result2.l_cons)


# ─────────────────────────────────────────────────────────────────────────────
# 5. 클라이언트 병렬화 (_client_step_worker / ThreadPoolExecutor)
# ─────────────────────────────────────────────────────────────────────────────

from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor
from scripts.data.dataset import ClientData


def _make_tta_config(**overrides):
    """TTAConfig import 없이 동작하는 최소 config 객체."""
    cfg = dict(
        grad_clip=1.0, drift_gate_threshold=0.0, reset_threshold=float("inf"),
        hard_gate_scale=0.0, hard_gate_min_history=0,
        alpha=0.3, beta=1.0, lambda_anchor=0.1, sensitivity=1.0, max_boost=5.0,
        lr=1e-3, rollback_threshold=1e9, rollback_window=20,
    )
    cfg.update(overrides)
    return SimpleNamespace(**cfg)


def _make_client(cid: str, n: int = 100, scale: float = 1.0) -> ClientData:
    """
    유효한 t_abs 범위가 보장되는 ClientData 픽스처.

    N=100, test split: [75, 100]
    유효한 t_abs 범위: SEQ_LEN+K-1 <= t_abs-75 (즉 t_abs >= 94)
    → t_abs=94 사용 시:
        x_input  = values[75:91]  (16 elements) ✓
        x_recent = values[91:95]  (4 elements)  ✓
    """
    values = (np.random.randn(n, CHANNELS) * scale).astype(np.float64)
    return ClientData(
        client_id=cid, values=values,
        split_indices={"train": (0, 60), "val": (60, 75), "test": (75, n)},
        global_mean=0.0, global_std=max(scale, 1.0),
    )


# 병렬 테스트의 공통 t_abs: test_start(75) + SEQ_LEN(16) + K(4) - 1 = 94
_T_ABS = 94


def test_client_step_worker_returns_correct_ci() -> None:
    """_client_step_worker가 입력 ci를 그대로 반환해야 한다."""
    model   = prepare_frozen_backbone(_model())
    adapter = _adapter()
    opt     = torch.optim.Adam(adapter.parameters(), lr=1e-3)
    guard   = RollbackGuard(threshold=1e9, tracker=ReconTracker(20))

    ci_in = 42
    ci_out, _, _, _, _ = _client_step_worker(
        ci=ci_in, client=_make_client("c0"),
        fm=model, adapter=adapter, opt=opt, guard=guard, y_prev=None,
        t_abs=_T_ABS, seq_len=SEQ_LEN, pred_len=PRED_LEN, k=K,
        loss_fn=_loss_fn(), tta_config=_make_tta_config(),
        device=torch.device("cpu"),
    )
    assert ci_out == ci_in


def test_parallel_workers_update_all_adapters() -> None:
    """
    N개 클라이언트가 병렬로 실행된 후 모든 adapter가 초기값에서 변해야 한다.

    각 클라이언트는 독립적인 fm / adapter / opt를 가지므로
    ThreadPoolExecutor 내부에서 경합 없이 각자 업데이트됨.
    """
    NUM_CLIENTS = 4
    adapters = [_adapter() for _ in range(NUM_CLIENTS)]
    models   = [prepare_frozen_backbone(_model()) for _ in range(NUM_CLIENTS)]
    opts     = [torch.optim.Adam(a.parameters(), lr=1e-2) for a in adapters]
    guards   = [RollbackGuard(threshold=1e9, tracker=ReconTracker(20)) for _ in range(NUM_CLIENTS)]
    clients  = [_make_client(f"c{i}", scale=float(i + 1)) for i in range(NUM_CLIENTS)]
    cfg      = _make_tta_config()

    with ThreadPoolExecutor(max_workers=NUM_CLIENTS) as ex:
        futures = [
            ex.submit(
                _client_step_worker,
                ci=ci, client=clients[ci], fm=models[ci], adapter=adapters[ci],
                opt=opts[ci], guard=guards[ci], y_prev=None,
                t_abs=_T_ABS, seq_len=SEQ_LEN, pred_len=PRED_LEN, k=K,
                loss_fn=_loss_fn(), tta_config=cfg, device=torch.device("cpu"),
            )
            for ci in range(NUM_CLIENTS)
        ]
        results = [f.result() for f in futures]

    # 모든 future가 올바른 ci를 반환했는지 확인
    returned_cis = {r[0] for r in results}
    assert returned_cis == set(range(NUM_CLIENTS))

    # 모든 adapter가 업데이트됐는지 확인
    for ci, adapter in enumerate(adapters):
        changed = (
            not torch.allclose(adapter.gamma, torch.ones(1, 1, CHANNELS))
            or not torch.allclose(adapter.delta, torch.zeros(1, 1, CHANNELS))
        )
        assert changed, f"adapter[{ci}] was not updated after parallel step"


def test_parallel_workers_do_not_corrupt_each_other() -> None:
    """
    병렬 실행 시 클라이언트 i의 adapter 업데이트가
    클라이언트 j (i≠j)의 adapter에 영향을 주지 않아야 한다.

    검증 방법:
    - 순차 실행 후 각 adapter 값을 저장.
    - 동일 초기 가중치로 병렬 실행.
    - 두 결과의 gamma/delta가 element-wise 일치하면 교차 오염 없음.
    """
    NUM_CLIENTS = 4
    cfg     = _make_tta_config()

    # 재현 가능한 동일 모델 초기화를 위해 같은 state_dict 사용
    reference_model = _model()
    ref_state = reference_model.state_dict()

    clients = [_make_client(f"c{i}", scale=float(i + 1)) for i in range(NUM_CLIENTS)]

    def _run(parallel: bool):
        ad_list = [_adapter() for _ in range(NUM_CLIENTS)]
        md_list = []
        for _ in range(NUM_CLIENTS):
            m = _model()
            m.load_state_dict(ref_state)
            md_list.append(prepare_frozen_backbone(m))
        op_list  = [torch.optim.Adam(a.parameters(), lr=1e-2) for a in ad_list]
        gd_list  = [RollbackGuard(threshold=1e9, tracker=ReconTracker(20)) for _ in range(NUM_CLIENTS)]

        kwargs_list = [
            dict(ci=ci, client=clients[ci], fm=md_list[ci], adapter=ad_list[ci],
                 opt=op_list[ci], guard=gd_list[ci], y_prev=None,
                 t_abs=_T_ABS, seq_len=SEQ_LEN, pred_len=PRED_LEN, k=K,
                 loss_fn=_loss_fn(), tta_config=cfg, device=torch.device("cpu"))
            for ci in range(NUM_CLIENTS)
        ]
        if parallel:
            with ThreadPoolExecutor(max_workers=NUM_CLIENTS) as ex:
                futures = [ex.submit(_client_step_worker, **kw) for kw in kwargs_list]
                for f in futures:
                    f.result()
        else:
            for kw in kwargs_list:
                _client_step_worker(**kw)

        return [(a.gamma.detach().clone(), a.delta.detach().clone()) for a in ad_list]

    seq_results = _run(parallel=False)
    par_results = _run(parallel=True)

    for ci in range(NUM_CLIENTS):
        gamma_seq, delta_seq = seq_results[ci]
        gamma_par, delta_par = par_results[ci]
        assert torch.allclose(gamma_seq, gamma_par, atol=1e-5), \
            f"client {ci} gamma differs: seq={gamma_seq} par={gamma_par}"
        assert torch.allclose(delta_seq, delta_par, atol=1e-5), \
            f"client {ci} delta differs: seq={delta_seq} par={delta_par}"
