#!/bin/sh
# Tier-1 sweep: verify individual effect of each new TTA mechanism on murata.
# 5 experiments, all launched in parallel.  Run tier-2 after reviewing results.
#
#   M1: active-step masked hindcast (hindcast_mask_threshold)
#   M3: inactive-window skip         (min_active_frac)
#   (M2 / EMA anchor deferred to tier-2)
set -eu

ROOT=/home/jylee/DLinear-Season-Trend
PYTHON=${PYTHON:-/home/jylee/miniconda3/envs/kcc2026/bin/python}
DEVICE=${DEVICE:-cuda:1}
BASE_CFG="$ROOT/configs/murata/fed_tta.yaml"
PREREQ_CFG="$ROOT/configs/murata/fed.yaml"
CKPT="checkpoints/murata_fed/seq192_pred96_k49_lr1e-3_r15_s0_0c6f19/best.pt"
RUN_ROOT="$ROOT/runs/murata_mech_tier1"
LOG_DIR="$ROOT/logs"
STAMP=$(date +%Y%m%d_%H%M%S)
mkdir -p "$RUN_ROOT" "$LOG_DIR"

write_cfg() {
  name=$1; mask_thr=$2; min_frac=$3; ema_beta=$4
  outdir="$RUN_ROOT/$STAMP/$name"
  dst="$outdir/config.yaml"
  prereq_dst="$outdir/fed.yaml"
  mkdir -p "$outdir"
  "$PYTHON" - "$BASE_CFG" "$PREREQ_CFG" "$dst" "$prereq_dst" "$DEVICE" "$outdir" "$CKPT" \
    "$mask_thr" "$min_frac" "$ema_beta" <<'PY'
import sys, yaml
base, prereq_base, dst, prereq_dst, device, outdir, ckpt, mask_thr, min_frac, ema_beta = sys.argv[1:]
with open(base) as f:
    cfg = yaml.safe_load(f)
cfg["device"]       = device
cfg["output_dir"]   = outdir
cfg["checkpoint_path"] = ckpt
cfg.setdefault("tta", {})
cfg["tta"]["lr"]                      = 1e-4
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
# write prereq fed.yaml alongside the TTA config
with open(prereq_base) as f:
    prereq = yaml.safe_load(f)
prereq["device"]     = device
prereq["output_dir"] = outdir
with open(prereq_dst, "w") as f:
    yaml.safe_dump(prereq, f)
PY
  echo "$outdir/config.yaml"
}

run_one() {
  name=$1; mask_thr=$2; min_frac=$3; ema_beta=$4
  cfg=$(write_cfg "$name" "$mask_thr" "$min_frac" "$ema_beta")
  log="$LOG_DIR/${STAMP}_murata_${name}.log"
  "$PYTHON" -u -m scripts.run --config "$cfg" > "$log" 2>&1
  echo "[done] $name  $(grep 'Avg:' "$log" | tail -1)"
}

cd "$ROOT"

# --- Tier 1: 5 experiments in parallel ---
#                                          mask_thr  min_frac  ema_beta
run_one  baseline_norm         0.0       0.0       1.0  &   # 기준점
run_one  m1_mask_t0p0          0.0       0.0       1.0  &   # M1: strict zero 마스킹
run_one  m1_mask_t0p1          0.1       0.0       1.0  &   # M1: 신호 0.1 이상만
run_one  m3_skip_0p3           0.0       0.3       1.0  &   # M3: 30% 미만 활성 스킵
run_one  comb_m1m3_t0p1_f0p3   0.1       0.3       1.0  &   # M1+M3 조합

wait
echo ""
echo "=== Tier-1 results ==="
for name in baseline_norm m1_mask_t0p0 m1_mask_t0p1 m3_skip_0p3 comb_m1m3_t0p1_f0p3; do
  log="$LOG_DIR/${STAMP}_murata_${name}.log"
  result=$(grep 'Avg:' "$log" | tail -1)
  printf '%-35s  %s\n' "$name" "$result"
done
