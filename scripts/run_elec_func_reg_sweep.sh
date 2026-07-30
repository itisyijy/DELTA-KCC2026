#!/bin/sh
# Functional regularization sweep — electricity only.
# Runs dlinear_tta and fed_tta with lambda_func in {0.5, 1.0, 2.0}.
# Strategy: 3 processes at a time (electricity has 321 clients → heavy RAM).
#
# Usage: ./scripts/run_elec_func_reg_sweep.sh [device]
#   device defaults to cuda:0

set -eu

cd /home/jylee/DELTA-KCC2026
PYTHON=/home/jylee/miniconda3/envs/kcc2026/bin/python
DEV="${1:-cuda:0}"
STAMP=$(date +%Y%m%d_%H%M%S)
TMPDIR=/tmp/elec_func_reg_${STAMP}
mkdir -p "$TMPDIR/electricity" logs

# Copy prereq configs so resolve_or_build_prereq_checkpoint can find them
cp configs/electricity/centralized.yaml "$TMPDIR/electricity/"
cp configs/electricity/fed.yaml         "$TMPDIR/electricity/"

patch_and_run() {
    baseline="$1"
    lambda_func="$2"

    label="electricity_${baseline}_lfunc${lambda_func}"
    src="configs/electricity/${baseline}.yaml"
    cfg="${TMPDIR}/electricity/${baseline}_lfunc${lambda_func}.yaml"

    $PYTHON - "$src" "$cfg" "$lambda_func" <<'PYEOF'
import sys, yaml
src, dst, lf = sys.argv[1], sys.argv[2], float(sys.argv[3])
with open(src) as f:
    cfg = yaml.safe_load(f)
cfg.setdefault('tta', {})['lambda_func'] = lf
with open(dst, 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
PYEOF

    log="logs/${STAMP}_${label}.log"
    $PYTHON -u -m scripts.run --config "$cfg" --device "$DEV" \
        > "$log" 2>&1 &
    echo "[$label] PID=$! log=$log"
}

echo "=== Electricity Functional Regularization Sweep | stamp=$STAMP ==="
echo "    DEV=$DEV  (3 processes at a time)"
echo ""

# Batch 1: dlinear_tta x3
echo ">>> Batch 1: dlinear_tta (lfunc 0.5, 1.0, 2.0) ..."
for lf in 0.5 1.0 2.0; do
    patch_and_run dlinear_tta "$lf"
done
echo ">>> Waiting for batch 1..."
wait
echo ">>> Batch 1 done."
echo ""

# Batch 2: fed_tta x3
echo ">>> Batch 2: fed_tta (lfunc 0.5, 1.0, 2.0) ..."
for lf in 0.5 1.0 2.0; do
    patch_and_run fed_tta "$lf"
done
echo ">>> Waiting for batch 2..."
wait
echo ">>> Batch 2 done."
echo ""

echo "=== All done ==="
echo "Results:"
grep "Avg:" logs/${STAMP}_electricity_*.log | sort
