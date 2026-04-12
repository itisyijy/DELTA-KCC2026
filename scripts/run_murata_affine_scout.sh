#!/bin/sh
set -eu

ROOT=/home/jylee/DLinear-Season-Trend
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
  k_ratio=$6
  lr=$7
  alpha=$8
  beta=$9
  lambda_anchor=${10}
  sensitivity=${11}
  max_boost=${12}
  reset_threshold=${13}
  "$PYTHON" - "$base_cfg" "$prereq_cfg" "$dst" "$prereq_dst" "$DEVICE" "$outdir" \
    "$k_ratio" "$lr" "$alpha" "$beta" "$lambda_anchor" "$sensitivity" "$max_boost" \
    "$reset_threshold" <<'PY'
import math
import sys
import yaml

(
    base_cfg,
    prereq_cfg,
    dst,
    prereq_dst,
    device,
    outdir,
    k_ratio,
    lr,
    alpha,
    beta,
    lambda_anchor,
    sensitivity,
    max_boost,
    reset_threshold,
) = sys.argv[1:]

with open(base_cfg) as f:
    cfg = yaml.safe_load(f)
with open(prereq_cfg) as f:
    prereq = yaml.safe_load(f)

cfg["device"] = device
cfg["output_dir"] = outdir
cfg.setdefault("tta", {})
cfg["tta"]["k_ratio"] = float(k_ratio)
cfg["tta"]["lr"] = float(lr)
cfg["tta"]["alpha"] = float(alpha)
cfg["tta"]["beta"] = float(beta)
cfg["tta"]["lambda_anchor"] = float(lambda_anchor)
cfg["tta"]["sensitivity"] = float(sensitivity)
cfg["tta"]["max_boost"] = float(max_boost)
cfg["tta"]["reset_threshold"] = math.inf if reset_threshold == "inf" else float(reset_threshold)
cfg["tta"]["drift_gate_threshold"] = 0.0
cfg["tta"]["rollback_threshold"] = 3.0
cfg["tta"]["rollback_window"] = 20

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
  base_cfg=$2
  prereq_cfg=$3
  outdir="$RUN_ROOT/$STAMP/$name"
  cfg="$outdir/config.yaml"
  prereq_dst="$outdir/$(basename "$prereq_cfg")"
  log="$LOG_DIR/${STAMP}_${name}.log"
  session="mur_aff_${STAMP}_${name}"
  mkdir -p "$outdir"
  write_cfg "$base_cfg" "$prereq_cfg" "$cfg" "$prereq_dst" "$outdir" \
    "$4" "$5" "$6" "$7" "$8" "$9" "${10}" "${11}"
  tmux new-session -d -s "$session" \
    "cd $ROOT && $PYTHON -u -m scripts.run_affine_local --config $cfg --max-clients $MAX_CLIENTS --max-workers $MAX_WORKERS > $log 2>&1"
  echo "[$name] session=$session -> $log"
}

cd "$ROOT"

FED_CFG="$ROOT/configs/murata/fed_tta.yaml"
FED_PREREQ="$ROOT/configs/murata/fed.yaml"
DLINEAR_CFG="$ROOT/configs/murata/dlinear_tta.yaml"
DLINEAR_PREREQ="$ROOT/configs/murata/centralized.yaml"

echo "=== Murata Affine Scout | stamp=$STAMP | device=$DEVICE | max_clients=$MAX_CLIENTS | max_workers=$MAX_WORKERS ==="

launch fed_base          "$FED_CFG" "$FED_PREREQ"         0.25 1e-4 0.3 1.0 0.10 1.0 5.0 inf
launch fed_fast_lr       "$FED_CFG" "$FED_PREREQ"         0.25 3e-4 0.3 1.0 0.10 1.0 5.0 inf
launch fed_short_k       "$FED_CFG" "$FED_PREREQ"         0.125 1e-4 0.3 1.0 0.10 1.0 5.0 inf
launch fed_long_k        "$FED_CFG" "$FED_PREREQ"         0.50 1e-4 0.3 1.0 0.10 1.0 5.0 inf
launch fed_cons_low      "$FED_CFG" "$FED_PREREQ"         0.25 1e-4 0.1 1.0 0.10 1.0 5.0 inf
launch fed_cons_high     "$FED_CFG" "$FED_PREREQ"         0.25 1e-4 0.6 1.0 0.10 1.0 5.0 inf
launch fed_anchor_light  "$FED_CFG" "$FED_PREREQ"         0.25 1e-4 0.3 1.0 0.03 1.0 5.0 inf
launch fed_anchor_strong "$FED_CFG" "$FED_PREREQ"         0.25 1e-4 0.3 1.0 0.30 1.0 5.0 inf
launch fed_boost_soft    "$FED_CFG" "$FED_PREREQ"         0.25 1e-4 0.3 1.0 0.10 0.5 3.0 inf
launch fed_boost_aggr    "$FED_CFG" "$FED_PREREQ"         0.25 1e-4 0.3 1.0 0.10 2.0 8.0 inf
launch fed_reset_guard   "$FED_CFG" "$FED_PREREQ"         0.25 1e-4 0.3 1.0 0.10 1.0 5.0 2.5
launch dlinear_base      "$DLINEAR_CFG" "$DLINEAR_PREREQ" 0.25 1e-4 0.3 1.0 0.10 1.0 5.0 inf

echo ""
echo "Launched 12 scout runs."
echo "Monitor:"
echo "  ls $LOG_DIR/${STAMP}_*.log"
echo "  grep 'Avg:' $LOG_DIR/${STAMP}_*.log"
