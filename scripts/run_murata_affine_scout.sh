#!/bin/sh
set -eu

ROOT=/home/jylee/DELTA-KCC2026
PYTHON=${PYTHON:-/home/jylee/miniconda3/envs/kcc2026/bin/python}
DEVICE=${DEVICE:-cuda:1}
MAX_CLIENTS=${MAX_CLIENTS:-8}
MAX_WORKERS=${MAX_WORKERS:-4}
RUN_ROOT="$ROOT/runs/murata_affine_scout"
LOG_DIR="$ROOT/logs"
STAMP=$(date +%Y%m%d_%H%M%S)
mkdir -p "$RUN_ROOT" "$LOG_DIR"

write_cfg() {
  base_cfg=$1
  prereq_cfg=$2
  dst=$3
  prereq_dst=$4
  outdir=$5
  adapter_mode=$6
  k_ratio=$7
  reset_threshold=$8
  "$PYTHON" - "$base_cfg" "$prereq_cfg" "$dst" "$prereq_dst" "$DEVICE" "$outdir" \
    "$adapter_mode" "$k_ratio" "$reset_threshold" <<'PY'
import math
import sys
import yaml

base_cfg, prereq_cfg, dst, prereq_dst, device, outdir, mode, k_ratio, reset_thr = sys.argv[1:]
with open(base_cfg) as f:
    cfg = yaml.safe_load(f)
with open(prereq_cfg) as f:
    prereq = yaml.safe_load(f)

cfg["device"] = device
cfg["output_dir"] = outdir
cfg.setdefault("tta", {})
cfg["tta"]["adapter_mode"] = mode
cfg["tta"]["k_ratio"] = float(k_ratio)
cfg["tta"]["lr"] = 1e-4
cfg["tta"]["alpha"] = 0.3
cfg["tta"]["beta"] = 1.0
cfg["tta"]["lambda_anchor"] = 0.1
cfg["tta"]["sensitivity"] = 1.0
cfg["tta"]["max_boost"] = 5.0
cfg["tta"]["reset_threshold"] = math.inf if reset_thr == "inf" else float(reset_thr)
cfg["tta"]["drift_gate_threshold"] = 0.0

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
  adapter_mode=$2
  k_ratio=$3
  reset_threshold=$4
  outdir="$RUN_ROOT/$STAMP/$name"
  cfg="$outdir/config.yaml"
  prereq_dst="$outdir/fed.yaml"
  log="$LOG_DIR/${STAMP}_${name}.log"
  session="mur_aff_${STAMP}_${name}"
  mkdir -p "$outdir"
  write_cfg "$ROOT/configs/murata/fed_tta.yaml" "$ROOT/configs/murata/fed.yaml" \
    "$cfg" "$prereq_dst" "$outdir" "$adapter_mode" "$k_ratio" "$reset_threshold"
  tmux new-session -d -s "$session" \
    "cd $ROOT && $PYTHON -u -m scripts.run_affine_local --config $cfg --max-clients $MAX_CLIENTS --max-workers $MAX_WORKERS > $log 2>&1"
  echo "[$name] session=$session -> $log"
}

cd "$ROOT"
echo "=== Murata Affine Scout V2 | stamp=$STAMP | device=$DEVICE | max_clients=$MAX_CLIENTS | max_workers=$MAX_WORKERS ==="

launch channel_base               channel_affine 0.25  inf
launch channel_short_k            channel_affine 0.125 inf
launch channel_reset_guard        channel_affine 0.25  2.5
launch time_base                  time_affine    0.25  inf
launch time_short_k               time_affine    0.125 inf
launch time_short_k_reset_guard   time_affine    0.125 2.5

echo ""
echo "Launched 6 scout runs."
echo "Monitor:"
echo "  grep 'Avg:' $LOG_DIR/${STAMP}_*.log"
echo "Report:"
echo "  $PYTHON -m scripts.report_murata_affine_scout --stamp $STAMP"
