#!/bin/sh

set -eu

cd /home/jylee/DLinear-Season-Trend
PYTHON=/home/jylee/miniconda3/envs/kcc2026/bin/python
LOG_DIR=logs
STAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p "$LOG_DIR"

"$PYTHON" -u -m scripts.run --config configs/murata/centralized.yaml --device cuda:0 \
  > "$LOG_DIR/${STAMP}_murata_centralized_cuda0.log" 2>&1 &
PID1=$!

"$PYTHON" -u -m scripts.run --config configs/murata/fed.yaml --device cuda:1 \
  > "$LOG_DIR/${STAMP}_murata_fed_cuda1.log" 2>&1 &
PID2=$!

wait "$PID1" "$PID2"

"$PYTHON" -u -m scripts.run --config configs/murata/dlinear_tta.yaml --device cuda:0 \
  > "$LOG_DIR/${STAMP}_murata_dlinear_tta_cuda0.log" 2>&1 &
PID3=$!

"$PYTHON" -u -m scripts.run --config configs/murata/fed_tta.yaml --device cuda:1 \
  > "$LOG_DIR/${STAMP}_murata_fed_tta_cuda1.log" 2>&1 &
PID4=$!

wait "$PID3" "$PID4"

"$PYTHON" -u -m scripts.run --config configs/murata/fed_tta_loop.yaml --device cuda:0 \
  > "$LOG_DIR/${STAMP}_murata_fed_tta_loop_cuda0.log" 2>&1
