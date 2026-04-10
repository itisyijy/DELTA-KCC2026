```
scripts/                         configs/
├── __init__.py                  ├── electricity/{centralized,fed,dlinear_tta,fed_tta,fed_tta_loop}.yaml
├── config.py                    ├── solar/         (동일 5개)
├── run.py                       └── murata/        (경로 placeholder, 동일 5개)
├── models/
│   ├── dlinear.py               (backgrounds/LTSF-Linear 어댑트)
│   └── revin_dlinear.py         (RevIN wrapper — NEW)
├── data/
│   ├── dataset.py               (ClientData, ClientDataset, CentralizedDataset)
│   └── loader.py                (CSV / kcc2026 parquet 양쪽 지원)
├── trainers/
│   ├── centralized.py           (Baseline 1)
│   └── fedavg.py                (Baseline 2)
├── tta/
│   ├── loss.py                  (HindcastLoss, DynamicRegularizer, TTALoss)
│   ├── adapter.py               (RevIN freeze + anchor 추출)
│   ├── engine.py                (run_tta_step, RollbackGuard)
│   └── loop.py                  (FED-TTA Loop — Baseline 5)
└── utils/
    ├── metrics.py               (MSE/MAE/sMAPE/Wasserstein)
    └── tools.py                 (EarlyStopping, seed_everything)
```

# 실행 방법 (kcc2026 conda env 사용)

``` BASH
PYTHON=/home/jylee/miniconda3/envs/kcc2026/bin/python
cd /home/jylee/DLinear-Season-Trend
```

현재 기본 config는 모두 `cuda:1`을 사용합니다.

- `batch_size: 256`
- `epochs: 15` for centralized
- `global_rounds: 15` for federated baselines
- dataset-specific `seq_len`
- dataset-specific `kernel_size`
- `solar`: `288 -> 96` (`48h -> 16h`)
- `solar kernel_size`: `73`
- `murata`: `192 -> 96` (`48h -> 24h`)
- `murata kernel_size`: `49`
- `electricity`: `336 -> 96` (`14d -> 4d`)
- `electricity kernel_size`: `25`

`dlinear_tta`, `fed_tta`, `fed_tta_loop`는 현재 config와 동일한 `seq_len/pred_len`로 다시 학습한 체크포인트를 사용해야 합니다. 예전 `96 -> 96` 체크포인트는 그대로 재사용할 수 없습니다.

`--auto-prereq`를 사용하면 TTA 계열 baseline 실행 시 필요한 선행 체크포인트를 자동으로 검증합니다. 체크포인트가 없거나, sidecar metadata(`best.pt.meta.json`) 기준으로 현재 prerequisite config와 맞지 않으면 해당 prerequisite baseline을 먼저 다시 학습한 뒤 이어서 실행합니다.

## 현재 상태 메모

`fed_tta_loop` 발산 버그 수정 완료 (2026-04-11, `scripts/tta/loop.py`).

- **원인**: 서버 피드백 후 anchor가 갱신되지 않아 delta = `W_current - W_original` (누적 drift 전체)이 됐고, 매 스텝마다 이미 반영된 drift를 0.9배로 재적용하는 양의 피드백 루프로 MSE가 (1.9)ⁿ 급수로 발산.
- **수정**: broadcast 시 `client_anchors`를 새 global weights로 갱신 + `optimizer.state.clear()`로 stale Adam 모멘텀 초기화.

모든 실행 예시는 기본적으로 `tmux` detached session 기준입니다.

``` BASH
tmux ls
tmux attach -t <session_name>
tmux kill-session -t <session_name>
```

## Baseline 1: Centralized 학습
``` BASH
# solar
tmux new-session -d -s solar_centralized \
    "$PYTHON -m scripts.run --config configs/solar/centralized.yaml"

# electricity
tmux new-session -d -s electricity_centralized \
    "$PYTHON -m scripts.run --config configs/electricity/centralized.yaml"

# murata
tmux new-session -d -s murata_centralized \
    "$PYTHON -m scripts.run --config configs/murata/centralized.yaml"
```

## Baseline 2: FedAvg 학습
``` BASH
# solar
tmux new-session -d -s solar_fed \
    "$PYTHON -m scripts.run --config configs/solar/fed.yaml"

# electricity
tmux new-session -d -s electricity_fed \
    "$PYTHON -m scripts.run --config configs/electricity/fed.yaml"

# murata
tmux new-session -d -s murata_fed \
    "$PYTHON -m scripts.run --config configs/murata/fed.yaml"
```

## Baseline 3: Centralized 모델 + TTA
``` BASH
# solar
tmux new-session -d -s solar_dlinear_tta \
    "$PYTHON -m scripts.run --config configs/solar/dlinear_tta.yaml --checkpoint-path checkpoints/solar_centralized/best.pt"

# electricity
tmux new-session -d -s electricity_dlinear_tta \
    "$PYTHON -m scripts.run --config configs/electricity/dlinear_tta.yaml --checkpoint-path checkpoints/electricity_centralized/best.pt"

# murata
tmux new-session -d -s murata_dlinear_tta \
    "$PYTHON -m scripts.run --config configs/murata/dlinear_tta.yaml --checkpoint-path checkpoints/murata_centralized/best.pt"
```

``` BASH
# checkpoint가 없거나 mismatch면 centralized를 먼저 다시 학습
tmux new-session -d -s murata_dlinear_tta_auto \
    "$PYTHON -m scripts.run --config configs/murata/dlinear_tta.yaml --auto-prereq"
```

## Baseline 4: FL 모델 + 일회성 TTA
``` BASH
# solar
tmux new-session -d -s solar_fed_tta \
    "$PYTHON -m scripts.run --config configs/solar/fed_tta.yaml --checkpoint-path checkpoints/solar_fed/best.pt"

# electricity
tmux new-session -d -s electricity_fed_tta \
    "$PYTHON -m scripts.run --config configs/electricity/fed_tta.yaml --checkpoint-path checkpoints/electricity_fed/best.pt"

# murata
tmux new-session -d -s murata_fed_tta \
    "$PYTHON -m scripts.run --config configs/murata/fed_tta.yaml --checkpoint-path checkpoints/murata_fed/best.pt"
```

``` BASH
# checkpoint가 없거나 mismatch면 fed를 먼저 다시 학습
tmux new-session -d -s murata_fed_tta_auto \
    "$PYTHON -m scripts.run --config configs/murata/fed_tta.yaml --auto-prereq"
```

## Baseline 5: FED-TTA Loop

``` BASH
# solar
tmux new-session -d -s solar_fed_tta_loop \
    "$PYTHON -m scripts.run --config configs/solar/fed_tta_loop.yaml --checkpoint-path checkpoints/solar_fed/best.pt"

# electricity
tmux new-session -d -s electricity_fed_tta_loop \
    "$PYTHON -m scripts.run --config configs/electricity/fed_tta_loop.yaml --checkpoint-path checkpoints/electricity_fed/best.pt"

# murata
tmux new-session -d -s murata_fed_tta_loop \
    "$PYTHON -m scripts.run --config configs/murata/fed_tta_loop.yaml --checkpoint-path checkpoints/murata_fed/best.pt"
```

``` BASH
# checkpoint가 없거나 mismatch면 fed를 먼저 다시 학습
tmux new-session -d -s murata_fed_tta_loop_auto \
    "$PYTHON -m scripts.run --config configs/murata/fed_tta_loop.yaml --auto-prereq"
```
