#!/bin/sh
# Electricity TTA improvement sweep.
#
# Hypothesis: standard TTA (update_scope=all, lr=1e-3) causes destructive drift
# over 1.5M steps. Two mechanisms to fix:
#   1. update_scope=norm  — only update RevIN affine (mean/std shift adaptation)
#   2. drift_gate         — skip TTA on stable timesteps (no real shift)
#
# Batches (2 processes at a time — 321 clients is heavy):
#   Batch A: norm + lr in {1e-4, 5e-4} x {fed_tta, dlinear_tta}
#   Batch B: norm + lr=1e-4 + drift_gate in {0.3, 0.5, 1.0} x fed_tta
#   Batch C: best combos — norm + lr=1e-4 + drift=0.5 + alpha in {3.0, 5.0}
#
# Usage: ./scripts/run_elec_tta_improve_sweep.sh [device]
#   device defaults to cuda:0

set -eu

cd /home/jylee/DELTA-KCC2026
PYTHON=/home/jylee/miniconda3/envs/kcc2026/bin/python
DEV="${1:-cuda:0}"
STAMP=$(date +%Y%m%d_%H%M%S)
TMPDIR=/tmp/elec_improve_${STAMP}
mkdir -p "$TMPDIR/electricity" logs

# prereq resolution needs these in the same dir as the patched config
cp configs/electricity/centralized.yaml "$TMPDIR/electricity/"
cp configs/electricity/fed.yaml         "$TMPDIR/electricity/"

# patch_and_run <baseline> <label_suffix> [key=value ...]
# Patches configs/electricity/<baseline>.yaml with the given key=value overrides
# (supports nested keys like tta.lr, tta.update_scope) and launches in background.
patch_and_run() {
    baseline="$1"
    label_suffix="$2"
    shift 2

    label="electricity_${baseline}_${label_suffix}"
    src="configs/electricity/${baseline}.yaml"
    cfg="${TMPDIR}/electricity/${baseline}_${label_suffix}.yaml"

    # Build the override pairs into a Python dict and apply
    $PYTHON - "$src" "$cfg" "$@" <<'PYEOF'
import sys, yaml

src, dst = sys.argv[1], sys.argv[2]
pairs = sys.argv[3:]

with open(src) as f:
    cfg = yaml.safe_load(f)

for pair in pairs:
    key, val = pair.split('=', 1)
    # parse value
    try:
        val = float(val)
    except ValueError:
        if val.lower() == 'true':
            val = True
        elif val.lower() == 'false':
            val = False

    if '.' in key:
        section, subkey = key.split('.', 1)
        cfg.setdefault(section, {})[subkey] = val
    else:
        cfg[key] = val

with open(dst, 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
PYEOF

    log="logs/${STAMP}_${label}.log"
    $PYTHON -u -m scripts.run --config "$cfg" --device "$DEV" \
        > "$log" 2>&1 &
    echo "[$label] PID=$! log=$log"
}

echo "=== Electricity TTA Improve Sweep | stamp=$STAMP | DEV=$DEV ==="
echo "  Baselines: centralized=0.1464  fed=0.1533  (targets to beat)"
echo "  fed_tta=0.2536  dlinear_tta=0.2460  fed_tta_loop=0.1691  (current TTA)"
echo ""

# ------------------------------------------------------------------
# Batch A: update_scope=norm + lr sweep
#   Hypothesis: norm-only adapts just RevIN affine weights (mean/std shift).
#   Much less destructive than full-model TTA.
# ------------------------------------------------------------------
echo ">>> Batch A: update_scope=norm, lr in {1e-4, 5e-4} ..."

patch_and_run fed_tta      "norm_lr1e-4"  tta.update_scope=norm tta.lr=0.0001
patch_and_run dlinear_tta  "norm_lr1e-4"  tta.update_scope=norm tta.lr=0.0001
wait

patch_and_run fed_tta      "norm_lr5e-4"  tta.update_scope=norm tta.lr=0.0005
patch_and_run dlinear_tta  "norm_lr5e-4"  tta.update_scope=norm tta.lr=0.0005
wait

echo ">>> Batch A done."
echo ""

# ------------------------------------------------------------------
# Batch B: norm + lr=1e-4 + drift_gate sweep
#   Hypothesis: skip TTA on stable timesteps (drift_score < threshold).
#   Gate computed as max(|Δμ|/σ, |Δσ|/σ) from x_recent vs train stats.
# ------------------------------------------------------------------
echo ">>> Batch B: norm + lr=1e-4 + drift_gate in {0.3, 0.5, 1.0} ..."

patch_and_run fed_tta  "norm_lr1e-4_gate0p3"  tta.update_scope=norm tta.lr=0.0001 tta.drift_gate_threshold=0.3
patch_and_run fed_tta  "norm_lr1e-4_gate0p5"  tta.update_scope=norm tta.lr=0.0001 tta.drift_gate_threshold=0.5
wait

patch_and_run fed_tta  "norm_lr1e-4_gate1p0"  tta.update_scope=norm tta.lr=0.0001 tta.drift_gate_threshold=1.0
patch_and_run dlinear_tta  "norm_lr1e-4_gate0p5"  tta.update_scope=norm tta.lr=0.0001 tta.drift_gate_threshold=0.5
wait

echo ">>> Batch B done."
echo ""

# ------------------------------------------------------------------
# Batch C: norm + lr=1e-4 + drift=0.5 + stronger regularization
#   Hypothesis: higher alpha prevents excessive RevIN parameter drift.
# ------------------------------------------------------------------
echo ">>> Batch C: norm + lr=1e-4 + drift=0.5 + alpha in {3.0, 5.0} ..."

patch_and_run fed_tta  "norm_lr1e-4_gate0p5_alpha3"  \
    tta.update_scope=norm tta.lr=0.0001 tta.drift_gate_threshold=0.5 tta.alpha=3.0
patch_and_run fed_tta  "norm_lr1e-4_gate0p5_alpha5"  \
    tta.update_scope=norm tta.lr=0.0001 tta.drift_gate_threshold=0.5 tta.alpha=5.0
wait

echo ">>> Batch C done."
echo ""

# ------------------------------------------------------------------
# Batch D: norm + k_ratio=0.5 (k=48) — longer hindcast signal
#   Hypothesis: k=24 gives too weak a signal; k=48 covers 2 daily cycles.
# ------------------------------------------------------------------
echo ">>> Batch D: norm + lr=1e-4 + k_ratio=0.5 ..."

patch_and_run fed_tta  "norm_lr1e-4_k0p5"  \
    tta.update_scope=norm tta.lr=0.0001 tta.k_ratio=0.5
patch_and_run fed_tta  "norm_lr1e-4_gate0p5_k0p5"  \
    tta.update_scope=norm tta.lr=0.0001 tta.drift_gate_threshold=0.5 tta.k_ratio=0.5
wait

echo ">>> Batch D done."
echo ""

# ------------------------------------------------------------------
echo "=== All done ==="
echo ""
echo "Results (reference: centralized MSE=0.1464, fed_tta_loop MSE=0.1691):"
grep "Avg:" logs/${STAMP}_electricity_*.log | sort
