#!/bin/sh
# update_scope=norm + lr sweep for electricity.
# Tests RevIN-only adaptation (γ, β) with conservative lr values.
#
# Usage: ./scripts/run_elec_norm_sweep.sh [device]
#   device defaults to cuda:0

set -eu

cd /home/jylee/DLinear-Season-Trend
PYTHON=/home/jylee/miniconda3/envs/kcc2026/bin/python
DEV="${1:-cuda:0}"
STAMP=$(date +%Y%m%d_%H%M%S)
TMPDIR=/tmp/elec_norm_${STAMP}
mkdir -p "$TMPDIR/electricity" logs

cp configs/electricity/centralized.yaml "$TMPDIR/electricity/"
cp configs/electricity/fed.yaml         "$TMPDIR/electricity/"

patch_and_run() {
    baseline="$1"
    lr="$2"

    label="electricity_${baseline}_norm_lr${lr}"
    src="configs/electricity/${baseline}.yaml"
    cfg="${TMPDIR}/electricity/${baseline}_norm_lr${lr}.yaml"

    $PYTHON - "$src" "$cfg" "$lr" <<'PYEOF'
import sys, yaml
src, dst, lr = sys.argv[1], sys.argv[2], sys.argv[3]
with open(src) as f:
    cfg = yaml.safe_load(f)
cfg.setdefault('tta', {})['update_scope'] = 'norm'
cfg['tta']['lr'] = float(lr)
with open(dst, 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
PYEOF

    log="logs/${STAMP}_${label}.log"
    $PYTHON -u -m scripts.run --config "$cfg" --device "$DEV" \
        > "$log" 2>&1 &
    echo "[$label] PID=$! log=$log"
}

echo "=== Electricity norm sweep | stamp=$STAMP ==="
echo "    DEV=$DEV  update_scope=norm  lr in {1e-4, 5e-4, 1e-3}"
echo ""

# Batch 1: dlinear_tta x3
echo ">>> Batch 1: dlinear_tta ..."
for lr in 0.0001 0.0005 0.001; do
    patch_and_run dlinear_tta "$lr"
done
wait
echo ">>> Batch 1 done."
echo ""

# Batch 2: fed_tta x3
echo ">>> Batch 2: fed_tta ..."
for lr in 0.0001 0.0005 0.001; do
    patch_and_run fed_tta "$lr"
done
wait
echo ">>> Batch 2 done."
echo ""

echo "=== All done ==="
grep "Avg:" logs/${STAMP}_electricity_*.log | sort
