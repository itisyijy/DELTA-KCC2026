#!/bin/sh
set -eu

ROOT=/home/jylee/DLinear-Season-Trend
PYTHON=${PYTHON:-/home/jylee/miniconda3/envs/kcc2026/bin/python3}
DEVICE=${DEVICE:-cuda:1}
STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}
EFF_MAX_WORKERS=${EFF_MAX_WORKERS:-1}
SEED_MAX_WORKERS=${SEED_MAX_WORKERS:-2}
SEEDS=${SEEDS:-"0 1 2"}
RUN_ELECTRICITY_MULTI_SEED=${RUN_ELECTRICITY_MULTI_SEED:-1}
LOG_DIR="$ROOT/logs"
mkdir -p "$LOG_DIR"

driver_log="$LOG_DIR/${STAMP}_overnight_driver.log"

wait_for_session() {
  name=$1
  while tmux has-session -t "$name" 2>/dev/null; do
    sleep 60
  done
}

report_efficiency() {
  datasets=$1
  cd "$ROOT"
  "$PYTHON" scripts/report_kcc_efficiency.py --stamp "$STAMP" --datasets $datasets
}

decision_field() {
  "$PYTHON" - "$ROOT/runs/kcc_drift_gate_sweep/$STAMP/report_outputs/efficiency_decision.json" "$1" <<'PY'
import json
import sys

payload = json.loads(open(sys.argv[1]).read())
value = payload[sys.argv[2]]
print("true" if value is True else "false" if value is False else value)
PY
}

launch_pilot_if_needed() {
  if [ -f "$ROOT/runs/kcc_drift_gate_sweep/$STAMP/report_outputs/efficiency_decision.json" ]; then
    return
  fi
  if tmux has-session -t "kcc_${STAMP}_seq" 2>/dev/null; then
    return
  fi
  cd "$ROOT"
  STAMP=$STAMP LAUNCH_MODE=sequential REUSE_EXISTING=0 DATASETS="murata" \
    MAX_WORKERS=$EFF_MAX_WORKERS DEVICE=$DEVICE sh scripts/run_kcc_drift_gate_sweep.sh
}

launch_expansion() {
  if tmux has-session -t "kcc_${STAMP}_seq" 2>/dev/null; then
    return
  fi
  cd "$ROOT"
  STAMP=$STAMP LAUNCH_MODE=sequential REUSE_EXISTING=1 DATASETS="electricity solar" \
    MAX_WORKERS=$EFF_MAX_WORKERS DEVICE=$DEVICE sh scripts/run_kcc_drift_gate_sweep.sh
}

launch_multi_seed() {
  dataset=$1
  gate=$2
  seed_stamp="${STAMP}_${dataset}_seed"
  session="kcc_seed_${seed_stamp}_${dataset}"
  if tmux has-session -t "$session" 2>/dev/null; then
    wait_for_session "$session"
    return
  fi
  cd "$ROOT"
  STAMP=$seed_stamp DATASET=$dataset VARIANTS="control $gate" SEEDS="$SEEDS" \
    DEVICE=$DEVICE MAX_WORKERS=$SEED_MAX_WORKERS PYTHON=$PYTHON sh scripts/run_kcc_multi_seed.sh
  wait_for_session "$session"
  "$PYTHON" scripts/report_kcc_multi_seed.py --stamp "$seed_stamp" --dataset "$dataset" --variants control "$gate"
}

worker_main() {
  echo "=== KCC overnight automation | stamp=$STAMP | device=$DEVICE ==="
  launch_pilot_if_needed
  wait_for_session "kcc_${STAMP}_seq"
  report_efficiency "murata"
  gate=$(decision_field selected_gate)
  pilot_ok=$(decision_field pilot_clear_gain)
  echo "[murata pilot] gate=$gate clear_gain=$pilot_ok"

  if [ "$pilot_ok" = "true" ]; then
    launch_expansion
    wait_for_session "kcc_${STAMP}_seq"
    report_efficiency "murata electricity solar"
    gate=$(decision_field selected_gate)
    echo "[full efficiency] gate=$gate"
  else
    echo "[murata pilot] skipped electricity/solar expansion because clear gain was not met."
  fi

  launch_multi_seed murata "$gate"
  if [ "$pilot_ok" = "true" ] && [ "$RUN_ELECTRICITY_MULTI_SEED" = "1" ]; then
    launch_multi_seed electricity "$gate"
  fi
  echo "=== KCC overnight automation complete | stamp=$STAMP | gate=$gate ==="
}

if [ "${1:-}" = "--worker" ]; then
  worker_main
  exit 0
fi

session="kcc_auto_${STAMP}"
tmux new-session -d -s "$session" \
  "cd $ROOT && STAMP=$STAMP DEVICE=$DEVICE EFF_MAX_WORKERS=$EFF_MAX_WORKERS SEED_MAX_WORKERS=$SEED_MAX_WORKERS SEEDS='$SEEDS' RUN_ELECTRICITY_MULTI_SEED=$RUN_ELECTRICITY_MULTI_SEED PYTHON=$PYTHON sh scripts/run_kcc_rebuttal_overnight.sh --worker > $driver_log 2>&1"
echo "[overnight] session=$session -> $driver_log"
echo "Using stamp=$STAMP"
