#!/bin/sh
set -eu

ROOT=/home/jylee/DELTA-KCC2026
PYTHON=${PYTHON:-/home/jylee/miniconda3/envs/kcc2026/bin/python}
DEVICE=${DEVICE:-cuda:1}
RUN_ROOT="$ROOT/runs/murata_backbone_eval"
LOG_DIR="$ROOT/logs"
STAMP=$(date +%Y%m%d_%H%M%S)
OUTDIR="$RUN_ROOT/$STAMP"
CFG="$OUTDIR/config.yaml"
LOG="$LOG_DIR/${STAMP}_murata_fed_backbone_eval.log"
SESSION="mur_backbone_${STAMP}"
mkdir -p "$OUTDIR" "$LOG_DIR"

"$PYTHON" - "$ROOT/configs/murata/fed_tta.yaml" "$CFG" "$DEVICE" "$OUTDIR" <<'PY'
import sys
import yaml

src, dst, device, outdir = sys.argv[1:]
with open(src) as f:
    cfg = yaml.safe_load(f)
cfg["baseline"] = "fed_tta"
cfg["device"] = device
cfg["output_dir"] = outdir
with open(dst, "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
PY

tmux new-session -d -s "$SESSION" \
  "cd $ROOT && $PYTHON -u -m scripts.run_backbone_eval --config $CFG > $LOG 2>&1"

echo "=== Murata Fed Backbone Eval | stamp=$STAMP | device=$DEVICE ==="
echo "session=$SESSION -> $LOG"
