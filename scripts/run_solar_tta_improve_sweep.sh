#!/bin/sh
# Solar TTA improvement sweep.
#
# Hypothesis: solar TTA is nearly neutral (dlinear_tta=0.2254 vs centralized=0.2232),
# but is blocked from improvement by:
#   1. Nighttime zero windows polluting the hindcast loss
#   2. TTA firing on stable clear-day sequences with no real shift
#   3. federated loop aggregation destroying per-client personalization
#
# Strategy: dlinear_tta only (NO federation loop — loop was 46% worse).
# Key mechanisms:
#   min_active_frac    — skip TTA when x_recent is mostly inactive (nighttime)
#   hindcast_mask_threshold — exclude near-zero dawn/dusk steps from L_recon
#   drift_gate         — skip TTA on stable days (no weather-driven shift)
#   update_scope=norm  — limit adaptation to RevIN affine params
#
# Batches (3 processes at a time — 137 clients, manageable):
#   Batch A: min_active_frac in {0.2, 0.3, 0.5}
#   Batch B: hindcast_mask_threshold in {0.2, 0.5} + combo with min_active_frac
#   Batch C: drift_gate in {0.2, 0.3, 0.5}
#   Batch D: best combos (min_active_frac=0.3 + mask=0.3 + drift=0.3)
#   Batch E: update_scope=norm variants
#
# Usage: ./scripts/run_solar_tta_improve_sweep.sh [device]
#   device defaults to cuda:0

set -eu

cd /home/jylee/DELTA-KCC2026
PYTHON=/home/jylee/miniconda3/envs/kcc2026/bin/python
DEV="${1:-cuda:0}"
STAMP=$(date +%Y%m%d_%H%M%S)
TMPDIR=/tmp/solar_improve_${STAMP}
mkdir -p "$TMPDIR/solar" logs

# prereq resolution needs these
cp configs/solar/centralized.yaml "$TMPDIR/solar/"
cp configs/solar/fed.yaml         "$TMPDIR/solar/"

# patch_and_run <label_suffix> [key=value ...]
# Always uses dlinear_tta as baseline (no federation loop).
patch_and_run() {
    label_suffix="$1"
    shift

    label="solar_dlinear_tta_${label_suffix}"
    src="configs/solar/dlinear_tta.yaml"
    cfg="${TMPDIR}/solar/dlinear_tta_${label_suffix}.yaml"

    $PYTHON - "$src" "$cfg" "$@" <<'PYEOF'
import sys, yaml

src, dst = sys.argv[1], sys.argv[2]
pairs = sys.argv[3:]

with open(src) as f:
    cfg = yaml.safe_load(f)

for pair in pairs:
    key, val = pair.split('=', 1)
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

echo "=== Solar TTA Improve Sweep | stamp=$STAMP | DEV=$DEV ==="
echo "  Baselines: centralized=0.2232  fed=0.2242  (targets to beat)"
echo "  dlinear_tta=0.2254  fed_tta=0.2268  (current TTA — barely worse)"
echo "  fed_tta_loop=0.3268  (catastrophic — NOT tested here)"
echo ""

# ------------------------------------------------------------------
# Batch A: min_active_frac sweep
#   Skips the TTA update entirely when fewer than X% of x_recent steps
#   are above the zero_floor. Key for solar's nighttime dead zones.
# ------------------------------------------------------------------
echo ">>> Batch A: min_active_frac in {0.2, 0.3, 0.5} ..."

patch_and_run "maf0p2"  tta.min_active_frac=0.2
patch_and_run "maf0p3"  tta.min_active_frac=0.3
patch_and_run "maf0p5"  tta.min_active_frac=0.5
wait

echo ">>> Batch A done."
echo ""

# ------------------------------------------------------------------
# Batch B: hindcast_mask_threshold sweep + combo with min_active_frac
#   mask_threshold=0.0 masks exact-zero steps; positive values also
#   mask near-zero dawn/dusk readings (active only if x > zero_floor + t).
# ------------------------------------------------------------------
echo ">>> Batch B: hindcast_mask_threshold in {0.2, 0.5} + combo ..."

patch_and_run "mask0p2"         tta.hindcast_mask_threshold=0.2
patch_and_run "mask0p5"         tta.hindcast_mask_threshold=0.5
patch_and_run "maf0p3_mask0p3"  tta.min_active_frac=0.3 tta.hindcast_mask_threshold=0.3
wait

echo ">>> Batch B done."
echo ""

# ------------------------------------------------------------------
# Batch C: drift_gate sweep
#   skip TTA on stable windows (no weather shift).
#   Solar has clear-day stretches where adaptation is counterproductive.
# ------------------------------------------------------------------
echo ">>> Batch C: drift_gate in {0.2, 0.3, 0.5} ..."

patch_and_run "gate0p2"  tta.drift_gate_threshold=0.2
patch_and_run "gate0p3"  tta.drift_gate_threshold=0.3
patch_and_run "gate0p5"  tta.drift_gate_threshold=0.5
wait

echo ">>> Batch C done."
echo ""

# ------------------------------------------------------------------
# Batch D: combined mechanism sweeps
#   min_active_frac + mask + drift_gate together
# ------------------------------------------------------------------
echo ">>> Batch D: combined (maf + mask + gate) ..."

patch_and_run "maf0p3_mask0p3_gate0p3" \
    tta.min_active_frac=0.3 tta.hindcast_mask_threshold=0.3 tta.drift_gate_threshold=0.3

patch_and_run "maf0p3_mask0p3_gate0p5" \
    tta.min_active_frac=0.3 tta.hindcast_mask_threshold=0.3 tta.drift_gate_threshold=0.5

patch_and_run "maf0p5_mask0p3_gate0p3" \
    tta.min_active_frac=0.5 tta.hindcast_mask_threshold=0.3 tta.drift_gate_threshold=0.3
wait

echo ">>> Batch D done."
echo ""

# ------------------------------------------------------------------
# Batch E: update_scope=norm (RevIN affine only)
#   Keeps linear weights frozen; adapts only mean/std normalization.
#   Combined with best selective-skip settings from above.
# ------------------------------------------------------------------
echo ">>> Batch E: update_scope=norm + best selective settings ..."

patch_and_run "norm_lr1e-4" \
    tta.update_scope=norm tta.lr=0.0001

patch_and_run "norm_lr1e-4_maf0p3" \
    tta.update_scope=norm tta.lr=0.0001 tta.min_active_frac=0.3

patch_and_run "norm_lr1e-4_maf0p3_gate0p3" \
    tta.update_scope=norm tta.lr=0.0001 tta.min_active_frac=0.3 tta.drift_gate_threshold=0.3
wait

echo ">>> Batch E done."
echo ""

# ------------------------------------------------------------------
echo "=== All done ==="
echo ""
echo "Results (reference: centralized MSE=0.2232, dlinear_tta MSE=0.2254):"
grep "Avg:" logs/${STAMP}_solar_*.log | sort
