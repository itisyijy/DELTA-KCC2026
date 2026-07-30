#!/bin/sh
set -eu

ROOT=/home/jylee/DELTA-KCC2026
PYTHON=${PYTHON:-/home/jylee/miniconda3/envs/kcc2026/bin/python}
DEVICE=${DEVICE:-cuda:1}
MAX_WORKERS=${MAX_WORKERS:-2}
DATASETS=${DATASETS:-"murata solar"}
REUSE_EXISTING=${REUSE_EXISTING:-1}
LAUNCH_MODE=${LAUNCH_MODE:-parallel}
STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}
RUN_ROOT="$ROOT/runs/kcc_drift_gate_sweep"
LOG_DIR="$ROOT/logs"
STAMP_ROOT="$RUN_ROOT/$STAMP"
SOURCES="$STAMP_ROOT/sources.tsv"
QUEUE_SH="$STAMP_ROOT/queue.sh"
mkdir -p "$STAMP_ROOT" "$LOG_DIR"
[ -f "$SOURCES" ] || printf "dataset\tkind\tstatus\tpath\n" > "$SOURCES"
if [ ! -f "$QUEUE_SH" ]; then
  printf "#!/bin/sh\nset -eu\n" > "$QUEUE_SH"
fi

record_source() {
  printf "%s\t%s\t%s\t%s\n" "$1" "$2" "$3" "$4" >> "$SOURCES"
}

schedule_run() {
  session=$1
  cmd=$2
  if [ "$LAUNCH_MODE" = "sequential" ]; then
    printf "%s\n" "$cmd" >> "$QUEUE_SH"
  else
    tmux new-session -d -s "$session" "$cmd"
  fi
}

find_existing() {
  "$PYTHON" - "$ROOT" "$1" "$2" <<'PY'
from pathlib import Path
import sys

root = Path(sys.argv[1])
dataset = sys.argv[2]
kind = sys.argv[3]
patterns = {
    ("murata", "backbone"): [root / "runs" / "murata_backbone_eval" / "*" / "*" / "metrics.json"],
    ("murata", "control"): [root / "runs" / "murata_time_affine_sweep" / "*" / "time_k00625_reset2p5" / "*" / "metrics.json"],
    ("electricity", "backbone"): [root / "runs" / "elec_solar_compare" / "*" / "electricity_backbone" / "*" / "metrics.json"],
    ("electricity", "control"): [root / "runs" / "elec_solar_compare" / "*" / "electricity_time_k00625_reset2p5" / "*" / "metrics.json"],
    ("solar", "backbone"): [root / "runs" / "elec_solar_compare" / "*" / "solar_backbone" / "*" / "metrics.json"],
    ("solar", "control"): [root / "runs" / "elec_solar_compare" / "*" / "solar_time_k00625_reset2p5" / "*" / "metrics.json"],
}
cands = []
for pattern in patterns.get((dataset, kind), []):
    cands.extend(sorted(root.glob(str(pattern.relative_to(root)))))
if not cands:
    print("")
    raise SystemExit(0)
latest = cands[-1]
print(latest.parent if kind == "backbone" and dataset == "murata" else latest.parent.parent)
PY
}

write_backbone_cfg() {
  "$PYTHON" - "$1" "$2" "$DEVICE" "$3" <<'PY'
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

write_tta_cfg() {
  "$PYTHON" - "$1" "$2" "$3" "$DEVICE" "$4" "$5" <<'PY'
import math
import sys
import yaml

base_cfg, dst, prereq_dst, device, outdir, gate = sys.argv[1:]
with open(base_cfg) as f:
    cfg = yaml.safe_load(f)
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

launch_backbone() {
  dataset=$1
  outdir="$STAMP_ROOT/${dataset}_backbone"
  cfg="$outdir/config.yaml"
  log="$LOG_DIR/${STAMP}_${dataset}_backbone.log"
  session="kcc_${STAMP}_${dataset}_backbone"
  mkdir -p "$outdir"
  write_backbone_cfg "$ROOT/configs/$dataset/fed_tta.yaml" "$cfg" "$outdir"
  schedule_run "$session" \
    "cd $ROOT && $PYTHON -u -m scripts.run_backbone_eval --config $cfg > $log 2>&1"
  record_source "$dataset" backbone launched "$outdir"
  echo "[$dataset/backbone] session=$session -> $log"
}

launch_control() {
  dataset=$1
  outdir="$STAMP_ROOT/${dataset}_control"
  cfg="$outdir/config.yaml"
  prereq="$outdir/fed.yaml"
  log="$LOG_DIR/${STAMP}_${dataset}_control.log"
  session="kcc_${STAMP}_${dataset}_control"
  mkdir -p "$outdir"
  write_tta_cfg "$ROOT/configs/$dataset/fed_tta.yaml" "$cfg" "$prereq" "$outdir" 0.0
  schedule_run "$session" \
    "cd $ROOT && $PYTHON -u -m scripts.run_affine_local --config $cfg --max-workers $MAX_WORKERS > $log 2>&1"
  record_source "$dataset" control launched "$outdir"
  echo "[$dataset/control] session=$session -> $log"
}

launch_gate() {
  dataset=$1
  label=$2
  gate=$3
  outdir="$STAMP_ROOT/${dataset}_${label}"
  cfg="$outdir/config.yaml"
  prereq="$outdir/fed.yaml"
  log="$LOG_DIR/${STAMP}_${dataset}_${label}.log"
  session="kcc_${STAMP}_${dataset}_${label}"
  mkdir -p "$outdir"
  write_tta_cfg "$ROOT/configs/$dataset/fed_tta.yaml" "$cfg" "$prereq" "$outdir" "$gate"
  schedule_run "$session" \
    "cd $ROOT && $PYTHON -u -m scripts.run_affine_local --config $cfg --max-workers $MAX_WORKERS > $log 2>&1"
  record_source "$dataset" "$label" launched "$outdir"
  echo "[$dataset/$label] session=$session -> $log"
}

echo "=== KCC Drift-Gate Sweep | stamp=$STAMP | device=$DEVICE | max_workers=$MAX_WORKERS ==="
echo "Datasets: $DATASETS"
for dataset in $DATASETS; do
  backbone=""
  control=""
  if [ "$REUSE_EXISTING" = "1" ]; then
    backbone=$(find_existing "$dataset" backbone)
    control=$(find_existing "$dataset" control)
  fi
  if [ -n "$backbone" ]; then
    record_source "$dataset" backbone reused "$backbone"
    echo "[$dataset/backbone] reuse -> $backbone"
  else
    launch_backbone "$dataset"
  fi
  if [ -n "$control" ]; then
    record_source "$dataset" control reused "$control"
    echo "[$dataset/control] reuse -> $control"
  else
    launch_control "$dataset"
  fi
  launch_gate "$dataset" gate_0p3 0.3
  launch_gate "$dataset" gate_0p5 0.5
  launch_gate "$dataset" gate_1p0 1.0
done

if [ "$LAUNCH_MODE" = "sequential" ]; then
  session="kcc_${STAMP}_seq"
  tmux new-session -d -s "$session" "sh $QUEUE_SH"
  echo "Sequential queue session: $session -> $QUEUE_SH"
fi

echo ""
echo "Sources manifest: $SOURCES"
echo "Monitor:"
echo "  grep 'Avg:' $LOG_DIR/${STAMP}_*.log"
echo "Resume with same stamp:"
echo "  STAMP=$STAMP DATASETS=\"electricity\" sh scripts/run_kcc_drift_gate_sweep.sh"
