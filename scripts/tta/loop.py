from __future__ import annotations

import concurrent.futures
import copy
import threading

import numpy as np
import torch

# Thread-local: torch.set_num_threads(1)을 워커 스레드마다 1회만 실행하기 위함
_tls = threading.local()

from scripts.config import TTAConfig, FeedbackConfig
from scripts.data.dataset import ClientData
from scripts.models.revin_dlinear import RevINDLinear
from scripts.tta.adapter import AffineAdapter, prepare_frozen_backbone, prepare_tta_model
from scripts.tta.delta import ClientDelta, aggregate_deltas, apply_server_feedback, clip_delta, compute_delta, get_model_weights
from scripts.tta.diagnostics import ClientTTADiagnostics
from scripts.tta.engine import (
    RollbackGuard,
    ReconTracker,
    TTAStepResultV2,
    build_hindcast_inputs,
    run_tta_step,
    run_tta_step_affine,
)
from scripts.tta.loss import HybridTTALoss, TTALoss
from scripts.tta.policy import get_tta_parameters
from scripts.utils.metrics import mae, mse, smape, inverse_global_scale
from scripts.utils.run_logging import MilestoneLogger


def run_fed_tta_loop(
    *,
    global_model: RevINDLinear,
    clients: list[ClientData],
    seq_len: int,
    pred_len: int,
    k: int,
    tta_config: TTAConfig,
    feedback_config: FeedbackConfig,
    device: torch.device,
) -> dict[str, dict[str, float]]:
    """
    Run the FED-TTA Loop over all clients' test splits.

    For every test time step t (in lock-step across all clients):
      1. Each client runs one TTA step.
      2. Collect deltas from non-skipped clients, clip + aggregate.
      3. Apply aggregated delta to global_model.
      4. Clients load the updated global weights before the next step.

    Returns per-client evaluation metrics.
    """
    tta_loss_fn = TTALoss(
        k=k,
        alpha=tta_config.alpha,
        lambda0=tta_config.lambda0,
        gamma=tta_config.gamma,
        lambda_func=tta_config.lambda_func,
        hindcast_mask_threshold=tta_config.hindcast_mask_threshold,
    )

    # Per-client state
    client_models: list[RevINDLinear] = []
    client_anchors: list[tuple[torch.Tensor, torch.Tensor]] = []
    client_anchor_models: list[RevINDLinear | None] = []
    client_guards: list[RollbackGuard] = []
    client_optimizers: list[torch.optim.Optimizer] = []

    for client in clients:
        cm = copy.deepcopy(global_model).to(device)
        cm, anchor_t, anchor_s = prepare_tta_model(cm, tta_config.update_scope)
        client_models.append(cm)
        client_anchors.append((anchor_t, anchor_s))
        if tta_config.lambda_func > 0.0:
            am = copy.deepcopy(global_model).to(device).eval()
            for p in am.parameters():
                p.requires_grad_(False)
            client_anchor_models.append(am)
        else:
            client_anchor_models.append(None)
        tracker = ReconTracker(window_size=tta_config.rollback_window)
        client_guards.append(
            RollbackGuard(threshold=tta_config.rollback_threshold, tracker=tracker)
        )
        client_optimizers.append(torch.optim.Adam(get_tta_parameters(cm), lr=tta_config.lr))

    # Determine the test time range (use first client as reference)
    # All clients share the same split structure (loaded from same CSV)
    test_s, test_e = clients[0].split_indices["test"]
    total_steps = max(0, test_e - (test_s + seq_len + k - 1))
    progress = MilestoneLogger("FED-TTA Loop", total_steps)
    progress.start(detail=f"clients={len(clients)}")

    # Accumulate predictions for final evaluation
    # per-client: list of (pred [pred_len, 1], target [pred_len, 1])
    all_preds: list[list[np.ndarray]] = [[] for _ in clients]
    all_targets: list[list[np.ndarray]] = [[] for _ in clients]

    # Lock-step over test steps
    for t_rel, t_abs in enumerate(range(test_s + seq_len + k - 1, test_e)):
        client_deltas: list[ClientDelta] = []

        for ci, (client, cm, (anchor_t, anchor_s), am, guard, opt) in enumerate(
            zip(clients, client_models, client_anchors, client_anchor_models, client_guards, client_optimizers)
        ):
            inputs = build_hindcast_inputs(client.values, t_abs, seq_len, k)
            if inputs is None:
                continue
            x_input, x_recent = inputs

            result = run_tta_step(
                model=cm,
                optimizer=opt,
                tta_loss_fn=tta_loss_fn,
                anchor_trend_w=anchor_t,
                anchor_season_w=anchor_s,
                x_input=x_input,
                x_recent=x_recent,
                mu_hist=client.global_mean,
                sigma_hist=max(client.global_std, 1e-8),
                rollback_guard=guard,
                device=device,
                grad_clip=tta_config.grad_clip,
                drift_gate_threshold=tta_config.drift_gate_threshold,
                min_active_frac=tta_config.min_active_frac,
                anchor_model=am,
            )

            if not result.skipped:
                dt, ds = compute_delta(cm, anchor_t, anchor_s)
                dt = clip_delta(dt, feedback_config.delta_clip_norm)
                ds = clip_delta(ds, feedback_config.delta_clip_norm)
                client_deltas.append(
                    ClientDelta(
                        client_id=client.client_id,
                        delta_trend=dt,
                        delta_season=ds,
                        n_valid_steps=1,
                    )
                )

            # Store standard prediction for evaluation
            x_pred_start = t_abs - seq_len + 1
            if x_pred_start + seq_len <= len(client.values):
                x_np = client.values[x_pred_start : x_pred_start + seq_len]  # [seq_len, 1]
                x_t = torch.from_numpy(x_np).unsqueeze(0).to(device)
                cm.eval()
                with torch.no_grad():
                    pred = cm(x_t).cpu().numpy()[0]  # [pred_len, 1]
                cm.train()  # back to train for next step
                target = client.values[
                    x_pred_start + seq_len : x_pred_start + seq_len + pred_len
                ]
                if len(target) == pred_len:
                    all_preds[ci].append(pred)
                    all_targets[ci].append(target)

        # --- Server feedback ---
        aggregated = aggregate_deltas(client_deltas, feedback_config.decay_factor)
        if aggregated is not None:
            agg_dt, agg_ds = aggregated
            apply_server_feedback(global_model, agg_dt, agg_ds)
            for ci, cm in enumerate(client_models):
                cm.load_state_dict(global_model.state_dict())
                client_anchors[ci] = get_model_weights(cm)
                client_optimizers[ci].state.clear()
        progress.update(t_rel + 1, detail=f"active_deltas={len(client_deltas)}")

    # --- Compute per-client metrics ---
    progress.finish(detail=f"clients={len(clients)}")
    results: dict[str, dict[str, float]] = {}
    for ci, client in enumerate(clients):
        preds = all_preds[ci]
        targets = all_targets[ci]
        if not preds:
            results[client.client_id] = {
                "mse": float("nan"), "mae": float("nan"), "smape": float("nan")
            }
            continue

        preds_g = np.concatenate(preds, axis=0)
        targets_g = np.concatenate(targets, axis=0)
        preds_orig = inverse_global_scale(preds_g, client.global_mean, client.global_std)
        targets_orig = inverse_global_scale(targets_g, client.global_mean, client.global_std)

        results[client.client_id] = {
            "mse":   mse(preds_g, targets_g),
            "mae":   mae(preds_g, targets_g),
            "smape": smape(preds_orig, targets_orig),
        }

    return results


# ─────────────────────────────────────────────────────────────────────────────
# New: Affine-Adapter Local TTA Loop (backbone completely frozen)
# ─────────────────────────────────────────────────────────────────────────────

def _client_step_worker(
    *,
    ci: int,
    client: ClientData,
    fm: RevINDLinear,
    adapter: AffineAdapter,
    opt: torch.optim.Optimizer,
    guard: RollbackGuard,
    y_prev: torch.Tensor | None,
    t_abs: int,
    seq_len: int,
    pred_len: int,
    k: int,
    loss_fn: HybridTTALoss,
    tta_config: "TTAConfig",
    device: torch.device,
) -> tuple[int, torch.Tensor | None, "np.ndarray | None", "np.ndarray | None", TTAStepResultV2 | None]:
    """
    한 시점(t_abs)에서 한 클라이언트의 TTA 스텝을 처리. ThreadPoolExecutor 워커용.

    Thread-safety 보장 조건
    ----------------------
    - fm, adapter, opt, guard : 클라이언트 전용 객체 (공유 없음)
    - loss_fn                 : 순수 함수 (내부 상태 없음) → 읽기 전용 공유 안전
    - y_prev                  : 이전 타임스텝에서 이미 detach()된 텐서 → 읽기 전용

    PyTorch over-subscription 방지
    --------------------------------
    멀티 스레드 환경에서 각 워커가 PyTorch 내부 스레드 풀(기본 = CPU 코어 수)을
    모두 사용하면 N_clients × N_cores 스레드가 경합한다.
    torch.set_num_threads(1)로 워커당 PyTorch 내부 스레드를 1개로 제한한다.
    (스레드-로컬 플래그로 워커 생애 주기 중 1회만 호출)

    Returns
    -------
    (ci, new_y_prev, pred_np, target_np)
    """
    # 워커 스레드 최초 실행 시에만 설정 (threading.local 사용)
    if not getattr(_tls, "num_threads_set", False):
        torch.set_num_threads(1)
        _tls.num_threads_set = True

    inputs = build_hindcast_inputs(client.values, t_abs, seq_len, k)
    if inputs is None:
        return ci, y_prev, None, None, None

    x_input, x_recent = inputs

    result = run_tta_step_affine(
        frozen_model=fm,
        adapter=adapter,
        optimizer=opt,
        loss_fn=loss_fn,
        x_input=x_input,
        x_recent=x_recent,
        y_prev=y_prev,
        rollback_guard=guard,
        device=device,
        max_grad_norm=tta_config.grad_clip,
        drift_gate_threshold=tta_config.drift_gate_threshold,
        mu_hist=client.global_mean,
        sigma_hist=max(client.global_std, 1e-8),
        reset_threshold=tta_config.reset_threshold,
        hard_gate_scale=tta_config.hard_gate_scale,
        hard_gate_min_history=tta_config.hard_gate_min_history,
    )

    new_y_prev = result.y_out  # already detached in run_tta_step_affine

    # 평가용 예측 수집
    x_pred_start = t_abs - seq_len + 1
    pred_np: np.ndarray | None = None
    target_np: np.ndarray | None = None

    if x_pred_start + seq_len <= len(client.values):
        x_np = client.values[x_pred_start : x_pred_start + seq_len]
        x_t = torch.from_numpy(x_np).unsqueeze(0).float().to(device)
        with torch.no_grad():
            pred_np = adapter(fm(x_t)).cpu().numpy()[0]  # [pred_len, C]
        target = client.values[
            x_pred_start + seq_len : x_pred_start + seq_len + pred_len
        ]
        if len(target) == pred_len:
            target_np = target
        else:
            pred_np = None  # target 길이 불일치 시 둘 다 버림

    return ci, new_y_prev, pred_np, target_np, result


def run_local_tta_loop(
    *,
    global_model: RevINDLinear,
    clients: list[ClientData],
    seq_len: int,
    pred_len: int,
    k: int,
    tta_config: "TTAConfig",
    device: torch.device,
    max_workers: int = 1,
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]], dict[str, float]]:
    """
    Affine-Adapter TTA: 백본 완전 동결, 클라이언트별 AffineAdapter(γ,δ) 만 업데이트.

    병렬화 모델
    ----------
    - 시간 축(t): y_prev 의존성으로 인해 직렬 순회 필수.
    - 클라이언트 축: 동일 t에서 클라이언트 간 공유 상태 없음 → 병렬 안전.
      max_workers > 1 이면 ThreadPoolExecutor로 클라이언트를 병렬 처리.

    max_workers 가이드
    ------------------
    - 1(기본): 순차 실행. 클라이언트 수가 적거나 디버깅 시 사용.
    - N > 1: 스레드 N개로 클라이언트 병렬화.
      · CPU 추론: max_workers = min(len(clients), os.cpu_count()) 권장.
      · 단일 GPU: GPU 연산은 기본 CUDA 스트림에서 직렬화되지만,
                  데이터 변환·optimizer 등 CPU 작업은 병렬화 이득 있음.
      · 멀티 GPU: 클라이언트별 device를 다르게 할당하면 진정한 병렬 가능
                  (현재 구현은 단일 device 공유).

    Thread-safety 보장
    ------------------
    각 클라이언트는 전용 fm / adapter / optimizer / guard를 보유하여
    스레드 간 공유 상태가 없다. loss_fn은 순수 함수(읽기 전용)이므로
    스레드 간 공유해도 안전하다.
    워커 내부에서 torch.set_num_threads(1)을 호출하여 PyTorch 내부
    스레드 풀의 과다 사용(over-subscription)을 방지한다.

    Space contract
    --------------
    frozen_model.forward(x) outputs Global Scale (RevIN denorm 포함).
    x_recent (= Hindcast target) 도 Global Scale.

    Args
    ----
    tta_config  : TTAConfig
    max_workers : 병렬 클라이언트 수. 1 = 순차. N > 1 = ThreadPoolExecutor.
    """
    # ── 공유 Loss 함수 ─────────────────────────────────────────────────────────
    loss_fn = HybridTTALoss(
        alpha=tta_config.alpha,
        beta=tta_config.beta,
        lambda_anchor=tta_config.lambda_anchor,
        sensitivity=tta_config.sensitivity,
        max_boost=tta_config.max_boost,
    )

    # ── 클라이언트별 상태 초기화 ──────────────────────────────────────────────
    frozen_models: list[RevINDLinear] = []
    adapters:      list[AffineAdapter] = []
    optimizers:    list[torch.optim.Optimizer] = []
    guards:        list[RollbackGuard] = []

    channels = global_model.revin.num_features

    for client in clients:
        # 각 클라이언트는 글로벌 모델의 독립적인 복사본 (완전 동결)
        fm = copy.deepcopy(global_model).to(device)
        fm = prepare_frozen_backbone(fm)
        frozen_models.append(fm)

        adapter = AffineAdapter(
            channels,
            pred_len=pred_len,
            mode=tta_config.adapter_mode,
        ).to(device)
        adapters.append(adapter)

        opt = torch.optim.Adam(adapter.parameters(), lr=tta_config.lr)
        optimizers.append(opt)

        tracker = ReconTracker(window_size=tta_config.rollback_window)
        guards.append(
            RollbackGuard(threshold=tta_config.rollback_threshold, tracker=tracker)
        )

    # ── 테스트 구간 설정 ──────────────────────────────────────────────────────
    test_s, test_e = clients[0].split_indices["test"]
    total_steps = max(0, test_e - (test_s + seq_len + k - 1))
    progress = MilestoneLogger("Local-TTA Loop", total_steps)
    progress.start(detail=f"clients={len(clients)}, channels={channels}")

    # 클라이언트별 누적 예측/정답
    all_preds:   list[list[np.ndarray]] = [[] for _ in clients]
    all_targets: list[list[np.ndarray]] = [[] for _ in clients]
    diagnostics = [ClientTTADiagnostics() for _ in clients]

    # 클라이언트별 이전 창 예측 (Temporal Consistency Loss용)
    y_prevs: list[torch.Tensor | None] = [None] * len(clients)

    # ── 시간 축 순회 ──────────────────────────────────────────────────────────
    # 클라이언트 병렬화: ThreadPoolExecutor를 루프 밖에서 생성해 스레드 생성 비용 절감
    executor = (
        concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        if max_workers > 1
        else None
    )

    try:
        for t_rel, t_abs in enumerate(range(test_s + seq_len + k - 1, test_e)):

            if executor is None:
                # ── 순차 실행 (max_workers == 1) ──────────────────────────────
                for ci, (client, fm, adapter, opt, guard) in enumerate(
                    zip(clients, frozen_models, adapters, optimizers, guards)
                ):
                    _, new_y_prev, pred_np, target_np, result = _client_step_worker(
                        ci=ci, client=client, fm=fm, adapter=adapter,
                        opt=opt, guard=guard, y_prev=y_prevs[ci],
                        t_abs=t_abs, seq_len=seq_len, pred_len=pred_len, k=k,
                        loss_fn=loss_fn, tta_config=tta_config, device=device,
                    )
                    diagnostics[ci].update(
                        result,
                        gamma_l1=float((adapter.gamma - 1.0).abs().mean().item()),
                        delta_l1=float(adapter.delta.abs().mean().item()),
                    )
                    y_prevs[ci] = new_y_prev
                    if pred_np is not None and target_np is not None:
                        all_preds[ci].append(pred_np)
                        all_targets[ci].append(target_np)

            else:
                # ── 병렬 실행 (max_workers > 1) ───────────────────────────────
                # Step 1: 모든 클라이언트를 동시에 제출
                #   y_prevs[ci]는 이 시점에서 읽기 전용 (이전 t의 결과)
                futures: dict[concurrent.futures.Future, int] = {
                    executor.submit(
                        _client_step_worker,
                        ci=ci, client=client, fm=fm, adapter=adapter,
                        opt=opt, guard=guard, y_prev=y_prevs[ci],
                        t_abs=t_abs, seq_len=seq_len, pred_len=pred_len, k=k,
                        loss_fn=loss_fn, tta_config=tta_config, device=device,
                    ): ci
                    for ci, (client, fm, adapter, opt, guard) in enumerate(
                        zip(clients, frozen_models, adapters, optimizers, guards)
                    )
                }

                # Step 2: 결과 수집 (메인 스레드에서 순차적으로)
                #   y_prevs 쓰기는 모든 future가 완료된 후 메인 스레드에서만 수행
                for future in concurrent.futures.as_completed(futures):
                    ci, new_y_prev, pred_np, target_np, result = future.result()
                    diagnostics[ci].update(
                        result,
                        gamma_l1=float((adapters[ci].gamma - 1.0).abs().mean().item()),
                        delta_l1=float(adapters[ci].delta.abs().mean().item()),
                    )
                    y_prevs[ci] = new_y_prev
                    if pred_np is not None and target_np is not None:
                        all_preds[ci].append(pred_np)
                        all_targets[ci].append(target_np)

            progress.update(t_rel + 1)

    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    # ── 클라이언트별 메트릭 집계 ─────────────────────────────────────────────
    progress.finish(detail=f"clients={len(clients)}")
    results: dict[str, dict[str, float]] = {}

    for ci, client in enumerate(clients):
        preds   = all_preds[ci]
        targets = all_targets[ci]

        if not preds:
            results[client.client_id] = {
                "mse": float("nan"), "mae": float("nan"), "smape": float("nan")
            }
            continue

        preds_g   = np.concatenate(preds,   axis=0)
        targets_g = np.concatenate(targets, axis=0)
        preds_orig   = inverse_global_scale(preds_g,   client.global_mean, client.global_std)
        targets_orig = inverse_global_scale(targets_g, client.global_mean, client.global_std)

        results[client.client_id] = {
            "mse":   mse(preds_g, targets_g),
            "mae":   mae(preds_g, targets_g),
            "smape": smape(preds_orig, targets_orig),
        }

    per_client_diagnostics = {
        client.client_id: diagnostics[ci].as_dict()
        for ci, client in enumerate(clients)
    }
    summary_keys = tuple(next(iter(per_client_diagnostics.values())).keys()) if per_client_diagnostics else ()
    diagnostic_summary = {
        key: float(np.mean([stats[key] for stats in per_client_diagnostics.values()]))
        for key in summary_keys
    }
    return results, per_client_diagnostics, diagnostic_summary
