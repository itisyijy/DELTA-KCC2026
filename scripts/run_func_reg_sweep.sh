#!/bin/sh
# Functional regularization sweep across all datasets.
#
# Runs dlinear_tta and fed_tta with lambda_func in {0.5, 1.0, 2.0}
# for electricity, solar, murata.
#
# Usage: ./scripts/run_func_reg_sweep.sh [device_a] [device_b]
#   device_a defaults to cuda:0  (electricity, solar)
#   device_b defaults to cuda:1  (murata)

set -eu

cd /home/jylee/DELTA-KCC2026
PYTHON=/home/jylee/miniconda3/envs/kcc2026/bin/python
DEV_A="${1:-cuda:0}"
DEV_B="${2:-cuda:1}"
STAMP=$(date +%Y%m%d_%H%M%S)
TMPDIR=/tmp/func_reg_sweep_${STAMP}
mkdir -p "$TMPDIR" logs

# Copy prereq configs so resolve_or_build_prereq_checkpoint can find them
for dataset in electricity solar murata; do
    mkdir -p "$TMPDIR/$dataset"
    cp "configs/$dataset/centralized.yaml" "$TMPDIR/$dataset/"
    cp "configs/$dataset/fed.yaml"         "$TMPDIR/$dataset/"
done

patch_and_run() {
    dataset="$1"
    baseline="$2"
    lambda_func="$3"
    device="$4"

    label="${dataset}_${baseline}_lfunc${lambda_func}"
    src="configs/${dataset}/${baseline}.yaml"
    cfg="${TMPDIR}/${dataset}/${baseline}_lfunc${lambda_func}.yaml"

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
    $PYTHON -u -m scripts.run --config "$cfg" --device "$device" \
        > "$log" 2>&1 &
    echo "[$label] PID=$! device=$device"
}

echo "=== Functional Regularization Sweep | stamp=$STAMP ==="
echo "    DEV_A=$DEV_A  DEV_B=$DEV_B"
echo "    Strategy: dataset-sequential, within-dataset parallel (6 runs at a time)"
echo ""

for dataset in electricity solar murata; do
    if [ "$dataset" = "murata" ]; then
        device="$DEV_B"
    else
        device="$DEV_A"
    fi

    echo ">>> [$dataset] launching 6 runs on $device ..."
    for lf in 0.5 1.0 2.0; do
        for baseline in dlinear_tta fed_tta; do
            patch_and_run "$dataset" "$baseline" "$lf" "$device"
        done
    done

    echo ">>> [$dataset] waiting for completion..."
    wait
    echo ">>> [$dataset] done."
    echo ""
done

echo "All datasets complete. Summary:"
echo "  grep 'Avg:' logs/${STAMP}_*.log | sort"
