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

``` BASH
# 짧은 실험 예시: solar fed를 라운드 줄여서 실행
tmux new-session -d -s solar_fed_seed0 \
    "$PYTHON -m scripts.run --config configs/solar/fed.yaml --seed 0 --global-rounds 10 --local-epochs 1 --output-dir runs/bg_solar_fed_seed0 --checkpoint-dir checkpoints/bg_solar_fed_seed0"

# 같은 seed 유지, 다른 실험 2개를 동시에 실행하는 예시
tmux new-session -d -s solar_centralized_seed0 \
    "$PYTHON -m scripts.run --config configs/solar/centralized.yaml --seed 0 --epochs 10 --output-dir runs/bg_solar_centralized_seed0 --checkpoint-dir checkpoints/bg_solar_centralized_seed0"

tmux new-session -d -s solar_fed_seed0_short \
    "$PYTHON -m scripts.run --config configs/solar/fed.yaml --seed 0 --global-rounds 10 --local-epochs 1 --output-dir runs/bg_solar_fed_seed0_short --checkpoint-dir checkpoints/bg_solar_fed_seed0_short"
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

## Baseline 4: FL 모델 + 일회성 TTA
``` BASH
# solar
tmux new-session -d -s solar_fed_tta \
    "$PYTHON -m scripts.run --config configs/solar/fed_tta.yaml"

# electricity
tmux new-session -d -s electricity_fed_tta \
    "$PYTHON -m scripts.run --config configs/electricity/fed_tta.yaml"

# murata
tmux new-session -d -s murata_fed_tta \
    "$PYTHON -m scripts.run --config configs/murata/fed_tta.yaml"
```

## Baseline 5 (제안 기법): FED-TTA Loop
``` BASH
# solar
tmux new-session -d -s solar_fed_tta_loop \
    "$PYTHON -m scripts.run --config configs/solar/fed_tta_loop.yaml"

# electricity
tmux new-session -d -s electricity_fed_tta_loop \
    "$PYTHON -m scripts.run --config configs/electricity/fed_tta_loop.yaml"

# murata
tmux new-session -d -s murata_fed_tta_loop \
    "$PYTHON -m scripts.run --config configs/murata/fed_tta_loop.yaml"
```
