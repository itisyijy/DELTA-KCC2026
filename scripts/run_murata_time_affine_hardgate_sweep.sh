#!/bin/sh
set -eu

ROOT=/home/jylee/DELTA-KCC2026
PYTHON=${PYTHON:-/home/jylee/miniconda3/envs/kcc2026/bin/python}
DEVICE=${DEVICE:-cuda:1}
MAX_WORKERS=${MAX_WORKERS:-4}
RUN_ROOT="$ROOT/runs/murata_time_affine_hardgate_sweep"
LOG_DIR="$ROOT/logs"
STAMP=$(date +%Y%m%d_%H%M%S)
mkdir -p "$RUN_ROOT" "$LOG_DIR"

write_cfg() {
  dst=$1
  prereq_dst=$2
  outdir=$3
  gate_scale=$4
  "$PYTHON" - "$ROOT/configs/murata/fed_tta.yaml" "$ROOT/configs/murata/fed.yaml" "$dst" \
    "$prereq_dst" "$DEVICE" "$outdir" "$gate_scale" <<'PY'
import math
import sys
import yaml

base_cfg, prereq_cfg, dst, prereq_dst, device, outdir, gate_scale = sys.argv[1:]
with open(base_cfg) as f:
    cfg = yaml.safe_load(f)
with open(prereq_cfg) as f:
    prereq = yaml.safe_load(f)

cfg["device"] = device
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
cfg["tta"]["drift_gate_threshold"] = 0.0
cfg["tta"]["hard_gate_scale"] = float(gate_scale)
cfg["tta"]["hard_gate_min_history"] = 20

prereq["device"] = device
prereq["output_dir"] = outdir

with open(dst, "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
with open(prereq_dst, "w") as f:
    yaml.safe_dump(prereq, f, sort_keys=False)
PY
}

launch() {
  name=$1
  gate_scale=$2
  outdir="$RUN_ROOT/$STAMP/$name"
  cfg="$outdir/config.yaml"
  prereq_dst="$outdir/fed.yaml"
  log="$LOG_DIR/${STAMP}_${name}.log"
  session="mur_hgate_${STAMP}_${name}"
  mkdir -p "$outdir"
  write_cfg "$cfg" "$prereq_dst" "$outdir" "$gate_scale"
  tmux new-session -d -s "$session" \
    "cd $ROOT && $PYTHON -u -m scripts.run_affine_local --config $cfg --max-workers $MAX_WORKERS > $log 2>&1"
  echo "[$name] session=$session -> $log"
}

cd "$ROOT"
echo "=== Murata Time-Affine Hard-Gate Sweep | stamp=$STAMP | device=$DEVICE | max_workers=$MAX_WORKERS ==="

launch time_best_control 0.0
launch time_gate1p00 1.00
launch time_gate1p02 1.02
launch time_gate1p05 1.05
launch time_gate1p10 1.10

echo ""
echo "Launched 5 full-scale runs."
echo "Monitor:"
echo "  grep 'Avg:' $LOG_DIR/${STAMP}_*.log"
