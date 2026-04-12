#!/bin/sh
set -eu

ROOT=/home/jylee/DLinear-Season-Trend
PYTHON=${PYTHON:-/home/jylee/miniconda3/envs/kcc2026/bin/python}
DEVICE=${DEVICE:-cuda:1}
MAX_WORKERS=${MAX_WORKERS:-4}
RUN_ROOT="$ROOT/runs/elec_solar_compare"
LOG_DIR="$ROOT/logs"
STAMP=$(date +%Y%m%d_%H%M%S)
mkdir -p "$RUN_ROOT" "$LOG_DIR"

write_tta_cfg() {
  base_cfg=$1
  dst=$2
  prereq_dst=$3
  outdir=$4
  k_ratio=$5
  reset_threshold=$6
  "$PYTHON" - "$base_cfg" "$dst" "$prereq_dst" "$DEVICE" "$outdir" "$k_ratio" "$reset_threshold" <<'PY'
import math
import sys
import yaml

base_cfg, dst, prereq_dst, device, outdir, k_ratio, reset_thr = sys.argv[1:]
with open(base_cfg) as f:
    cfg = yaml.safe_load(f)
cfg["device"] = device
cfg["output_dir"] = outdir
cfg.setdefault("tta", {})
cfg["tta"]["adapter_mode"] = "time_affine"
cfg["tta"]["k_ratio"] = float(k_ratio)
cfg["tta"]["lr"] = 1e-4
cfg["tta"]["alpha"] = 0.3
cfg["tta"]["beta"] = 1.0
cfg["tta"]["lambda_anchor"] = 0.1
cfg["tta"]["sensitivity"] = 1.0
cfg["tta"]["max_boost"] = 5.0
cfg["tta"]["reset_threshold"] = math.inf if reset_thr == "inf" else float(reset_thr)
cfg["tta"]["drift_gate_threshold"] = 0.0
with open(dst, "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
with open(prereq_dst, "w") as f:
    yaml.safe_dump({"baseline": "fed"}, f, sort_keys=False)
PY
}

write_backbone_cfg() {
  src=$1
  dst=$2
  outdir=$3
  "$PYTHON" - "$src" "$dst" "$DEVICE" "$outdir" <<'PY'
import sys
import yaml

src, dst, device, outdir = sys.argv[1:]
with open(src) as f:
    cfg = yaml.safe_load(f)
cfg["device"] = device
cfg["output_dir"] = outdir
with open(dst, "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
PY
}

launch_backbone() {
  dataset=$1
  cfg_src="$ROOT/configs/$dataset/fed_tta.yaml"
  outdir="$RUN_ROOT/$STAMP/${dataset}_backbone"
  cfg="$outdir/config.yaml"
  log="$LOG_DIR/${STAMP}_${dataset}_backbone.log"
  session="cmp_${STAMP}_${dataset}_backbone"
  mkdir -p "$outdir"
  write_backbone_cfg "$cfg_src" "$cfg" "$outdir"
  tmux new-session -d -s "$session" \
    "cd $ROOT && $PYTHON -u -m scripts.run_backbone_eval --config $cfg > $log 2>&1"
  echo "[$dataset/backbone] session=$session -> $log"
}

launch_tta() {
  dataset=$1
  name=$2
  k_ratio=$3
  reset_threshold=$4
  outdir="$RUN_ROOT/$STAMP/${dataset}_${name}"
  cfg="$outdir/config.yaml"
  prereq="$outdir/fed.yaml"
  log="$LOG_DIR/${STAMP}_${dataset}_${name}.log"
  session="cmp_${STAMP}_${dataset}_${name}"
  mkdir -p "$outdir"
  write_tta_cfg "$ROOT/configs/$dataset/fed_tta.yaml" "$cfg" "$prereq" "$outdir" "$k_ratio" "$reset_threshold"
  tmux new-session -d -s "$session" \
    "cd $ROOT && $PYTHON -u -m scripts.run_affine_local --config $cfg --max-workers $MAX_WORKERS > $log 2>&1"
  echo "[$dataset/$name] session=$session -> $log"
}

echo "=== Elec/Solar Backbone vs TTA | stamp=$STAMP | device=$DEVICE | max_workers=$MAX_WORKERS ==="
for dataset in electricity solar; do
  launch_backbone "$dataset"
  launch_tta "$dataset" time_k0125_reset2p5 0.125 2.5
  launch_tta "$dataset" time_k00625_reset2p5 0.0625 2.5
done

echo ""
echo "Launched 6 runs."
echo "Monitor:"
echo "  grep 'Avg:' $LOG_DIR/${STAMP}_*.log"
