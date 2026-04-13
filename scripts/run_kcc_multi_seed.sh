#!/bin/sh
set -eu

ROOT=/home/jylee/DLinear-Season-Trend
PYTHON=${PYTHON:-/home/jylee/miniconda3/envs/kcc2026/bin/python}
DEVICE=${DEVICE:-cuda:1}
DATASET=${DATASET:-murata}
VARIANTS=${VARIANTS:-"control gate_1p0"}
SEEDS=${SEEDS:-"0 1 2"}
MAX_WORKERS=${MAX_WORKERS:-2}
STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}
RUN_ROOT="$ROOT/runs/kcc_multi_seed/$STAMP"
LOG_DIR="$ROOT/logs"
mkdir -p "$RUN_ROOT" "$LOG_DIR"

write_cfg() {
  "$PYTHON" - "$1" "$2" "$3" "$DEVICE" "$4" "$5" "$6" <<'PY'
import sys
import yaml

base_cfg, dst, prereq_dst, device, outdir, seed, gate = sys.argv[1:]
with open(base_cfg) as f:
    cfg = yaml.safe_load(f)
cfg["device"] = device
cfg["seed"] = int(seed)
cfg["output_dir"] = outdir
cfg.setdefault("tta", {})
cfg["tta"]["adapter_mode"] = "time_affine"
cfg["tta"]["k_ratio"] = 0.0625
cfg["tta"]["lr"] = 1e-4
cfg["tta"]["alpha"] = 0.3
cfg["tta"]["beta"] = 1.0
cfg["tta"]["lambda_anchor"] = 0.1
cfg["tta"]["sensitivity"] = 1.0
cfg["tta"]["max_boost"] = 5.0
cfg["tta"]["reset_threshold"] = 2.5
cfg["tta"]["drift_gate_threshold"] = float(gate)
cfg["tta"]["hard_gate_scale"] = 0.0
cfg["tta"]["hard_gate_min_history"] = 0
cfg["tta"]["acceptance_margin"] = -1.0
with open(dst, "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
with open(prereq_dst, "w") as f:
    yaml.safe_dump({"baseline": "fed"}, f, sort_keys=False)
PY
}

gate_value() {
  case "$1" in
    control) printf "0.0" ;;
    gate_0p3) printf "0.3" ;;
    gate_0p5) printf "0.5" ;;
    gate_1p0) printf "1.0" ;;
    *) echo "unknown variant: $1" >&2; exit 1 ;;
  esac
}

run_one() {
  variant=$1
  seed=$2
  gate=$(gate_value "$variant")
  outdir="$RUN_ROOT/${variant}_s${seed}"
  cfg="$outdir/config.yaml"
  prereq="$outdir/fed.yaml"
  log="$LOG_DIR/${STAMP}_${DATASET}_${variant}_s${seed}.log"
  mkdir -p "$outdir"
  write_cfg "$ROOT/configs/$DATASET/fed_tta.yaml" "$cfg" "$prereq" "$outdir" "$seed" "$gate"
  echo "[$DATASET/$variant/seed=$seed] -> $log"
  cd "$ROOT"
  "$PYTHON" -u -m scripts.run_affine_local --config "$cfg" --max-workers "$MAX_WORKERS" > "$log" 2>&1
}

worker_main() {
  echo "=== KCC Multi-Seed | stamp=$STAMP | dataset=$DATASET | device=$DEVICE ==="
  for seed in $SEEDS; do
    for variant in $VARIANTS; do
      run_one "$variant" "$seed"
    done
  done
  echo "Completed multi-seed runs under $RUN_ROOT"
}

if [ "${1:-}" = "--worker" ]; then
  worker_main
  exit 0
fi

SESSION="kcc_seed_${STAMP}_${DATASET}"
DRIVER_LOG="$LOG_DIR/${STAMP}_${DATASET}_multi_seed_driver.log"
tmux new-session -d -s "$SESSION" \
  "cd $ROOT && STAMP=$STAMP DATASET=$DATASET VARIANTS='$VARIANTS' SEEDS='$SEEDS' DEVICE=$DEVICE MAX_WORKERS=$MAX_WORKERS PYTHON=$PYTHON sh scripts/run_kcc_multi_seed.sh --worker > $DRIVER_LOG 2>&1"
echo "[multi-seed] session=$SESSION -> $DRIVER_LOG"
echo "Runs will be written under $RUN_ROOT"
