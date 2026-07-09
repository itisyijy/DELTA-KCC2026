#!/bin/sh
# Sweep the three new TTA mechanisms for murata:
#   Mechanism 1: Active-step masked hindcast (hindcast_mask_threshold)
#   Mechanism 2: EMA anchor (ema_beta)
#   Mechanism 3: Inactive-window skip (min_active_frac)
set -eu

ROOT=/home/jylee/DLinear-Season-Trend
PYTHON=${PYTHON:-/home/jylee/miniconda3/envs/kcc2026/bin/python}
DEVICE=${DEVICE:-cuda:1}
BASE_CFG="$ROOT/configs/murata/fed_tta.yaml"
CKPT="checkpoints/murata_fed/seq192_pred96_k49_lr1e-3_r15_s0_0c6f19/best.pt"
RUN_ROOT="$ROOT/runs/murata_mechanism_sweeps"
LOG_DIR="$ROOT/logs"
STAMP=$(date +%Y%m%d_%H%M%S)
mkdir -p "$RUN_ROOT" "$LOG_DIR"

# Write a TTA config YAML with mechanism-specific overrides.
# All runs use: lr=1e-4, scope=norm, k_ratio=0.25, no drift gate, default rollback.
write_cfg() {
  name=$1
  mask_thr=$2      # hindcast_mask_threshold
  min_frac=$3      # min_active_frac
  ema_beta=$4      # ema_beta (1.0 = disabled)
  lr=${5:-1e-4}    # tta lr

  outdir="$RUN_ROOT/$STAMP/$name"
  dst="$outdir/config.yaml"
  mkdir -p "$outdir"

  "$PYTHON" - "$BASE_CFG" "$dst" "$DEVICE" "$outdir" "$CKPT" \
    "$mask_thr" "$min_frac" "$ema_beta" "$lr" <<'PY'
import sys
import yaml

base, dst, device, outdir, ckpt, mask_thr, min_frac, ema_beta, lr = sys.argv[1:]

with open(base) as f:
    cfg = yaml.safe_load(f)

cfg["device"]       = device
cfg["output_dir"]   = outdir
cfg["checkpoint_path"] = ckpt

cfg.setdefault("tta", {})
cfg["tta"]["lr"]                      = float(lr)
cfg["tta"]["update_scope"]            = "norm"
cfg["tta"]["k_ratio"]                 = 0.25
cfg["tta"]["drift_gate_threshold"]    = 0.0
cfg["tta"]["rollback_threshold"]      = 3.0
cfg["tta"]["rollback_window"]         = 20
cfg["tta"]["hindcast_mask_threshold"] = float(mask_thr)
cfg["tta"]["min_active_frac"]         = float(min_frac)
cfg["tta"]["ema_beta"]                = float(ema_beta)

with open(dst, "w") as f:
    yaml.safe_dump(cfg, f)
PY
  echo "$outdir/config.yaml"
}

run_one() {
  name=$1
  mask_thr=$2
  min_frac=$3
  ema_beta=$4
  lr=${5:-1e-4}

  cfg=$(write_cfg "$name" "$mask_thr" "$min_frac" "$ema_beta" "$lr")
  log="$LOG_DIR/${STAMP}_murata_${name}.log"
  echo "==> $name  mask_thr=$mask_thr  min_frac=$min_frac  ema_beta=$ema_beta  lr=$lr"
  "$PYTHON" -u -m scripts.run --config "$cfg" > "$log" 2>&1
  tail -3 "$log"
}

cd "$ROOT"

# --- Baseline (all mechanisms disabled, replicates existing behavior) ---
run_one "baseline_norm"   0.0  0.0  1.0  1e-4

# --- Mechanism 1: Active-step masked hindcast ---
# threshold=0.0: mask exact zero steps (strict above zero_floor)
run_one "m1_mask_t0p0"   0.0  0.0  1.0  1e-4
# threshold=0.1: require 0.1 GS units of signal above zero_floor
run_one "m1_mask_t0p1"   0.1  0.0  1.0  1e-4
# threshold=0.3: require meaningful daytime generation
run_one "m1_mask_t0p3"   0.3  0.0  1.0  1e-4

# --- Mechanism 3: Inactive-window skip ---
run_one "m3_skip_0p2"    0.0  0.2  1.0  1e-4
run_one "m3_skip_0p3"    0.0  0.3  1.0  1e-4
run_one "m3_skip_0p5"    0.0  0.5  1.0  1e-4

# --- Mechanism 2: EMA anchor ---
# beta=0.999  → half-life ~693 steps (≈7 days at 15min × sliding window)
run_one "m2_ema_0p999"   0.0  0.0  0.999   1e-4
# beta=0.9995 → half-life ~1386 steps (≈14 days)
run_one "m2_ema_0p9995"  0.0  0.0  0.9995  1e-4
# beta=0.9999 → half-life ~6931 steps (≈72 days)
run_one "m2_ema_0p9999"  0.0  0.0  0.9999  1e-4

# --- Combinations ---
# M1 + M3: mask zero steps AND skip fully-inactive windows
run_one "comb_m1m3_t0_f0p3"    0.0  0.3  1.0   1e-4
run_one "comb_m1m3_t0p1_f0p3"  0.1  0.3  1.0   1e-4

# M1 + M2 + M3: all three mechanisms together
run_one "comb_all_t0_f0p3_b0p999"    0.0  0.3  0.999   1e-4
run_one "comb_all_t0p1_f0p3_b0p999"  0.1  0.3  0.999   1e-4
run_one "comb_all_t0p1_f0p3_b0p9995" 0.1  0.3  0.9995  1e-4
