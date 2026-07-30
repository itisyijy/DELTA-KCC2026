# DELTA-KCC2026

DELTA는 장기 시계열 예측 모델의 배포 후 분포 이동에 대응하는 선택적 경량
Test-Time Adaptation(TTA) 프레임워크입니다. DLinear backbone은 동결하고
time-affine adapter만 갱신하며, drift gate·rollback guard·reset rule로 불필요하거나
해로운 적응을 억제합니다.

이 저장소는 KCC 2026 연구를 위한 학습·평가 코드, 재현 스크립트, 리포트 생성기와
원고 산출물을 포함합니다.

## 핵심 구성

```text
scripts/
├── run.py                       # 공통 학습/TTA 진입점
├── models/                      # DLinear, RevIN-DLinear
├── data/                        # CSV 및 kcc2026 parquet 로더
├── trainers/                    # Centralized, FedAvg
├── tta/                         # adapter, loss, engine, loop
├── run_dataset_pipeline.sh      # 데이터셋별 5-baseline 파이프라인
├── run_kcc_drift_gate_sweep.sh  # drift-gate threshold sweep
└── report_kcc_*.py              # 효율성, CI, multi-seed 리포트

configs/
├── murata/
├── solar/
└── electricity/

docs/
├── KCC_장기시계열TTA_jhlim_v6.docx
├── v7_추가원고.md
└── report.md
```

각 데이터셋 config는 다음 다섯 baseline을 공유합니다.

1. `centralized`: 중앙집중 DLinear 학습
2. `fed`: FedAvg DLinear 학습
3. `dlinear_tta`: 중앙집중 모델 + TTA
4. `fed_tta`: FedAvg 모델 + 일회성 TTA
5. `fed_tta_loop`: FedAvg 모델 + 반복 서버 피드백 TTA

## 데이터셋 설정

| 데이터셋 | 입력 길이 | 예측 길이 | 커널 크기 | 시간 범위 |
| --- | ---: | ---: | ---: | --- |
| Murata | 192 | 96 | 49 | 48h → 24h |
| Solar | 288 | 96 | 73 | 48h → 16h |
| Electricity | 336 | 96 | 25 | 14d → 4d |
| Traffic | 336 | 96 | 25 | 14d → 4d |

Murata·Solar·Electricity 재현 config는 저장소에 포함됩니다. Traffic은 862개
센서 확장성 평가에 사용하며, 데이터 경로가 환경에 의존하므로
`configs/traffic/`을 로컬 config로 관리합니다.

TTA baseline은 동일한 `seq_len`/`pred_len`로 학습된 선행 체크포인트가 필요합니다.
`--auto-prereq`는 체크포인트와 `best.pt.meta.json`을 검사하고, 누락되었거나 config와
맞지 않으면 prerequisite baseline을 먼저 학습합니다.

## 실행 환경

```bash
cd /home/jylee/DELTA-KCC2026
export PYTHON=/home/jylee/miniconda3/envs/kcc2026/bin/python3
export DEVICE=cuda:1
```

모든 GPU 실험은 `cuda:1`에서 실행합니다. 장시간 작업은 tmux detached session으로
실행하고, 시작 전에 RAM과 VRAM 여유를 확인합니다.

```bash
free -h
nvidia-smi
tmux ls
```

## 빠른 실행

아래 예시는 Murata 기준이며 `DATASET`만 `solar` 또는 `electricity`로 바꿀 수
있습니다.

```bash
DATASET=murata

# Centralized
tmux new-session -d -s "${DATASET}_centralized" \
  "$PYTHON -m scripts.run --config configs/$DATASET/centralized.yaml --device $DEVICE"

# FedAvg
tmux new-session -d -s "${DATASET}_fed" \
  "$PYTHON -m scripts.run --config configs/$DATASET/fed.yaml --device $DEVICE"

# Centralized + TTA
tmux new-session -d -s "${DATASET}_dlinear_tta" \
  "$PYTHON -m scripts.run --config configs/$DATASET/dlinear_tta.yaml --device $DEVICE --auto-prereq"

# FedAvg + TTA
tmux new-session -d -s "${DATASET}_fed_tta" \
  "$PYTHON -m scripts.run --config configs/$DATASET/fed_tta.yaml --device $DEVICE --auto-prereq"

# FedAvg + loop TTA
tmux new-session -d -s "${DATASET}_fed_tta_loop" \
  "$PYTHON -m scripts.run --config configs/$DATASET/fed_tta_loop.yaml --device $DEVICE --auto-prereq"
```

실행 상태와 로그는 다음처럼 확인합니다.

```bash
tmux attach -t murata_fed_tta
tail -f logs/<run-log>.log
```

## Drift-gate 실험과 리포트

효율성 pilot → 전체 데이터셋 확장 → multi-seed 검증은 백그라운드 자동화
스크립트로 실행합니다.

```bash
STAMP=$(date +%Y%m%d_%H%M%S)
STAMP=$STAMP DEVICE=cuda:1 sh scripts/run_kcc_rebuttal_overnight.sh
```

기존 결과를 재사용해 threshold sweep만 실행할 수도 있습니다.

```bash
STAMP=<experiment-stamp> REUSE_EXISTING=1 DATASETS="murata electricity solar" \
  DEVICE=cuda:1 sh scripts/run_kcc_drift_gate_sweep.sh
```

리포트 생성기는 쉼표로 구분한 데이터셋 목록을 받습니다. Traffic 결과 source가
준비된 경우 동일한 공통 threshold 분석에 포함할 수 있습니다.

```bash
$PYTHON scripts/report_kcc_drift_gate_sweep.py \
  --stamp <experiment-stamp> \
  --datasets murata,electricity,solar,traffic

$PYTHON scripts/report_kcc_drift_gate_ci.py \
  --stamp <experiment-stamp> \
  --datasets murata,electricity,solar,traffic
```

실험 결과는 `runs/`, 체크포인트는 `checkpoints/`, 로그는 `logs/` 아래에 생성되며
Git에는 포함하지 않습니다.

## 검증

```bash
$PYTHON -m pytest -q
sh -n scripts/run_kcc_drift_gate_sweep.sh
$PYTHON -m py_compile scripts/report_kcc_drift_gate_sweep.py
```

연구 주장과 표 병합용 최신 초안은
[`docs/v7_추가원고.md`](docs/v7_추가원고.md), 상세 구현·실험 기록은
[`docs/report.md`](docs/report.md)에서 확인할 수 있습니다.
