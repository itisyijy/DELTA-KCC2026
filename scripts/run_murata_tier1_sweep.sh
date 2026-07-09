#!/bin/sh
# Tier 1 TTA sweep for murata/fed_tta
#
# A. drift_gate_threshold: 0.5, 1.0, 2.0
# B. update_scope:         norm, trend, season
# C. lambda0:              5.0, 10.0
#
# Usage: ./scripts/run_murata_tier1_sweep.sh [device]
#   device defaults to cuda:1

set -eu

cd /home/jylee/DLinear-Season-Trend
PYTHON=/home/jylee/miniconda3/envs/kcc2026/bin/python
DEVICE="${1:-cuda:1}"
STAMP=$(date +%Y%m%d_%H%M%S)
BASE_CONFIG=configs/murata/fed_tta.yaml
TMPDIR=/tmp/murata_tier1_${STAMP}
mkdir -p "$TMPDIR" logs
# prereq resolution looks for fed.yaml in the same dir as the config
cp configs/murata/fed.yaml "$TMPDIR/fed.yaml"

patch_and_run() {
    label="$1"
    field="$2"
    value="$3"

    cfg="${TMPDIR}/${label}.yaml"
    $PYTHON - "$BASE_CONFIG" "$cfg" "$field" "$value" <<'PYEOF'
import sys, yaml
src, dst, field, value = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
with open(src) as f:
    cfg = yaml.safe_load(f)
# parse value as float or string
try:
    value = float(value)
except ValueError:
    pass
section, key = field.split(".")
cfg[section][key] = value
with open(dst, "w") as f:
    yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
PYEOF

    log="logs/${STAMP}_murata_tier1_${label}.log"
    $PYTHON -u -m scripts.run --config "$cfg" --device "$DEVICE" > "$log" 2>&1 &
    echo "[$label] PID=$! -> $log"
}

echo "=== Murata Tier 1 Sweep | device=$DEVICE | stamp=$STAMP ==="

# A. drift_gate_threshold
patch_and_run "A_drift0.5"   "tta.drift_gate_threshold" "0.5"
patch_and_run "A_drift1.0"   "tta.drift_gate_threshold" "1.0"
patch_and_run "A_drift2.0"   "tta.drift_gate_threshold" "2.0"

# B. update_scope
patch_and_run "B_scope_norm"   "tta.update_scope" "norm"
patch_and_run "B_scope_trend"  "tta.update_scope" "trend"
patch_and_run "B_scope_season" "tta.update_scope" "season"

# C. lambda0
patch_and_run "C_lambda5"  "tta.lambda0" "5.0"
patch_and_run "C_lambda10" "tta.lambda0" "10.0"

echo ""
echo "All 8 runs launched. Monitor with:"
echo "  tail -f logs/${STAMP}_murata_tier1_*.log"
echo ""
echo "Summary when done:"
echo "  grep 'Avg:' logs/${STAMP}_murata_tier1_*.log"
