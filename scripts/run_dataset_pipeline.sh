#!/bin/sh

set -eu

if [ "$#" -ne 3 ]; then
  echo "Usage: $0 <dataset> <device_a> <device_b>" >&2
  exit 1
fi

cd /home/jylee/DLinear-Season-Trend
DATASET="$1"
DEVICE_A="$2"
DEVICE_B="$3"
PYTHON=/home/jylee/miniconda3/envs/kcc2026/bin/python
LOG_DIR=logs
STAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p "$LOG_DIR"

"$PYTHON" -u -m scripts.run --config "configs/$DATASET/centralized.yaml" --device "$DEVICE_A" \
  > "$LOG_DIR/${STAMP}_${DATASET}_centralized_${DEVICE_A}.log" 2>&1 &
PID1=$!

"$PYTHON" -u -m scripts.run --config "configs/$DATASET/fed.yaml" --device "$DEVICE_B" \
  > "$LOG_DIR/${STAMP}_${DATASET}_fed_${DEVICE_B}.log" 2>&1 &
PID2=$!

wait "$PID1" "$PID2"

"$PYTHON" -u -m scripts.run --config "configs/$DATASET/dlinear_tta.yaml" --device "$DEVICE_A" \
  > "$LOG_DIR/${STAMP}_${DATASET}_dlinear_tta_${DEVICE_A}.log" 2>&1 &
PID3=$!

"$PYTHON" -u -m scripts.run --config "configs/$DATASET/fed_tta.yaml" --device "$DEVICE_B" \
  > "$LOG_DIR/${STAMP}_${DATASET}_fed_tta_${DEVICE_B}.log" 2>&1 &
PID4=$!

wait "$PID3" "$PID4"

"$PYTHON" -u -m scripts.run --config "configs/$DATASET/fed_tta_loop.yaml" --device "$DEVICE_A" \
  > "$LOG_DIR/${STAMP}_${DATASET}_fed_tta_loop_${DEVICE_A}.log" 2>&1
