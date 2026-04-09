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
## Baseline 1: Centralized 학습
``` BASH
$PYTHON scripts/run.py --config configs/solar/centralized.yaml
```

## Baseline 2: FedAvg 학습
``` BASH
$PYTHON scripts/run.py --config configs/solar/fed.yaml
```

## Baseline 3: Centralized 모델 + TTA
``` BASH
$PYTHON scripts/run.py --config configs/solar/dlinear_tta.yaml \
    --checkpoint-path checkpoints/solar_centralized/best.pt
```

## Baseline 4: FL 모델 + 일회성 TTA
``` BASH
$PYTHON scripts/run.py --config configs/solar/fed_tta.yaml
```

## Baseline 5 (제안 기법): FED-TTA Loop
``` BASH
$PYTHON scripts/run.py --config configs/solar/fed_tta_loop.yaml
```

