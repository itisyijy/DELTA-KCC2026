#!/bin/sh
# Run the full 5-baseline pipeline for a given dataset.
#
# Usage:
#   ./scripts/run_dataset_pipeline.sh <dataset> <device_a> <device_b>
#
# Stage 1 (parallel): centralized on device_a, fed on device_b
# Stage 2 (parallel): dlinear_tta on device_a, fed_tta on device_b
#                     Both use --auto-prereq so missing/stale checkpoints are
#                     rebuilt automatically from Stage 1 output.
# Stage 3 (serial):   fed_tta_loop on device_a (depends on Stage 1 fed ckpt)

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

# --- Stage 1: train prerequisite models ---
echo "[${DATASET}] Stage 1: centralized + fed"

"$PYTHON" -u -m scripts.run --config "configs/$DATASET/centralized.yaml" --device "$DEVICE_A" \
  > "$LOG_DIR/${STAMP}_${DATASET}_centralized_${DEVICE_A}.log" 2>&1 &
PID1=$!

"$PYTHON" -u -m scripts.run --config "configs/$DATASET/fed.yaml" --device "$DEVICE_B" \
  > "$LOG_DIR/${STAMP}_${DATASET}_fed_${DEVICE_B}.log" 2>&1 &
PID2=$!

wait "$PID1" "$PID2"

# --- Stage 2: TTA eval (checkpoints resolved/rebuilt automatically) ---
echo "[${DATASET}] Stage 2: dlinear_tta + fed_tta"

"$PYTHON" -u -m scripts.run --config "configs/$DATASET/dlinear_tta.yaml" --device "$DEVICE_A" \
  --auto-prereq \
  > "$LOG_DIR/${STAMP}_${DATASET}_dlinear_tta_${DEVICE_A}.log" 2>&1 &
PID3=$!

"$PYTHON" -u -m scripts.run --config "configs/$DATASET/fed_tta.yaml" --device "$DEVICE_B" \
  --auto-prereq \
  > "$LOG_DIR/${STAMP}_${DATASET}_fed_tta_${DEVICE_B}.log" 2>&1 &
PID4=$!

wait "$PID3" "$PID4"

# --- Stage 3: fed_tta_loop (sequential; uses fed checkpoint) ---
echo "[${DATASET}] Stage 3: fed_tta_loop"

"$PYTHON" -u -m scripts.run --config "configs/$DATASET/fed_tta_loop.yaml" --device "$DEVICE_A" \
  --auto-prereq \
  > "$LOG_DIR/${STAMP}_${DATASET}_fed_tta_loop_${DEVICE_A}.log" 2>&1

echo "[${DATASET}] Pipeline complete. Logs in ${LOG_DIR}/"
