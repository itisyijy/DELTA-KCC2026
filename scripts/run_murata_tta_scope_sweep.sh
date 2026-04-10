#!/bin/sh
set -eu

ROOT=/home/jylee/DLinear-Season-Trend
PYTHON=${PYTHON:-python3}
DEVICE=${DEVICE:-cuda:1}
BASE_CFG="$ROOT/configs/murata/fed_tta.yaml"
PREREQ_CFG="$ROOT/configs/murata/fed.yaml"
RUN_ROOT="$ROOT/runs/murata_tta_scope_sweep"
LOG_DIR="$ROOT/logs"
STAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p "$RUN_ROOT" "$LOG_DIR"

launch() {
  name=$1
  scope=$2
  gate=$3
  lr=$4
  outdir="$RUN_ROOT/$STAMP/$name"
  cfg="$outdir/config.yaml"
  prereq="$outdir/fed.yaml"
  log="$LOG_DIR/${STAMP}_${name}.log"
  mkdir -p "$outdir"
  "$PYTHON" - "$BASE_CFG" "$cfg" "$scope" "$gate" "$lr" "$outdir" "$DEVICE" <<'PY'
import sys, yaml
src, dst, scope, gate, lr, outdir, device = sys.argv[1:]
with open(src) as f:
    cfg = yaml.safe_load(f)
cfg["device"] = device
cfg["output_dir"] = outdir
cfg["tta"]["update_scope"] = scope
cfg["tta"]["drift_gate_threshold"] = float(gate)
cfg["tta"]["lr"] = float(lr)
with open(dst, "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
PY
  cp "$PREREQ_CFG" "$prereq"
  nohup "$PYTHON" -u -m scripts.run --config "$cfg" > "$log" 2>&1 &
  echo "$name $!"
}

cd "$ROOT"
launch murata_fed_tta_norm_only norm 0.0 1e-4
launch murata_fed_tta_trend_only trend 0.0 5e-5
launch murata_fed_tta_conditional all 0.75 1e-5
