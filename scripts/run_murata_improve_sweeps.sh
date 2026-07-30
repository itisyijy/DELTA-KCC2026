#!/bin/sh
set -eu

ROOT=/home/jylee/DELTA-KCC2026
PYTHON=${PYTHON:-python3}
DEVICE=${DEVICE:-cuda:1}
BASE_CFG="$ROOT/configs/murata/fed_tta.yaml"
PREREQ_CFG="$ROOT/configs/murata/fed.yaml"
RUN_ROOT="$ROOT/runs/murata_improve_sweeps"
LOG_DIR="$ROOT/logs"
STAMP=$(date +%Y%m%d_%H%M%S)
mkdir -p "$RUN_ROOT" "$LOG_DIR"

write_tta_cfg() {
  base=$1
  dst=$2
  outdir=$3
  checkpoint=$4
  seq_len=$5
  pred_len=$6
  kernel=$7
  individual=$8
  revin_affine=$9
  k_ratio=${10}
  tta_lr=${11}
  scope=${12}
  gate=${13}
  rb_th=${14}
  rb_win=${15}
  batch=${16}
  clear_ckpt=${17}
  "$PYTHON" - "$base" "$dst" "$DEVICE" "$outdir" "$checkpoint" "$seq_len" "$pred_len" \
    "$kernel" "$individual" "$revin_affine" "$k_ratio" "$tta_lr" "$scope" "$gate" \
    "$rb_th" "$rb_win" "$batch" "$clear_ckpt" <<'PY'
import sys
import yaml

(
    base,
    dst,
    device,
    outdir,
    checkpoint,
    seq_len,
    pred_len,
    kernel,
    individual,
    revin_affine,
    k_ratio,
    tta_lr,
    scope,
    gate,
    rb_th,
    rb_win,
    batch,
    clear_ckpt,
) = sys.argv[1:]

with open(base) as f:
    cfg = yaml.safe_load(f)

cfg["device"] = device
cfg["output_dir"] = outdir
cfg["model"]["seq_len"] = int(seq_len)
cfg["model"]["pred_len"] = int(pred_len)
cfg["model"]["kernel_size"] = int(kernel)
cfg["model"]["individual"] = individual.lower() == "true"
cfg["model"]["revin_affine"] = revin_affine.lower() == "true"
if batch:
    cfg["batch_size"] = int(batch)

cfg["tta"]["k_ratio"] = float(k_ratio)
cfg["tta"]["lr"] = float(tta_lr)
cfg["tta"]["update_scope"] = scope
cfg["tta"]["drift_gate_threshold"] = float(gate)
cfg["tta"]["rollback_threshold"] = float(rb_th)
cfg["tta"]["rollback_window"] = int(rb_win)

if clear_ckpt == "1":
    cfg["checkpoint_path"] = ""
elif checkpoint:
    cfg["checkpoint_path"] = checkpoint

with open(dst, "w") as f:
    yaml.safe_dump(cfg, f)
PY
}

write_prereq_cfg() {
  base=$1
  dst=$2
  outdir=$3
  seq_len=$4
  pred_len=$5
  kernel=$6
  individual=$7
  revin_affine=$8
  batch=$9
  "$PYTHON" - "$base" "$dst" "$DEVICE" "$outdir" "$seq_len" "$pred_len" "$kernel" \
    "$individual" "$revin_affine" "$batch" <<'PY'
import sys
import yaml

(
    base,
    dst,
    device,
    outdir,
    seq_len,
    pred_len,
    kernel,
    individual,
    revin_affine,
    batch,
) = sys.argv[1:]

with open(base) as f:
    cfg = yaml.safe_load(f)

cfg["device"] = device
cfg["output_dir"] = outdir
cfg["model"]["seq_len"] = int(seq_len)
cfg["model"]["pred_len"] = int(pred_len)
cfg["model"]["kernel_size"] = int(kernel)
cfg["model"]["individual"] = individual.lower() == "true"
cfg["model"]["revin_affine"] = revin_affine.lower() == "true"
if batch:
    cfg["batch_size"] = int(batch)

with open(dst, "w") as f:
    yaml.safe_dump(cfg, f)
PY
}

run_one() {
  name=$1
  auto_prereq=$2
  seq_len=$3
  pred_len=$4
  kernel=$5
  individual=$6
  revin_affine=$7
  k_ratio=$8
  tta_lr=$9
  scope=${10}
  gate=${11}
  rb_th=${12}
  rb_win=${13}
  checkpoint=${14}
  batch=${15}
  clear_ckpt=${16}

  outdir="$RUN_ROOT/$STAMP/$name"
  cfg="$outdir/config.yaml"
  prereq="$outdir/fed.yaml"
  log="$LOG_DIR/${STAMP}_${name}.log"
  mkdir -p "$outdir"

  write_tta_cfg "$BASE_CFG" "$cfg" "$outdir" "$checkpoint" "$seq_len" "$pred_len" \
    "$kernel" "$individual" "$revin_affine" "$k_ratio" "$tta_lr" "$scope" "$gate" \
    "$rb_th" "$rb_win" "$batch" "$clear_ckpt"
  write_prereq_cfg "$PREREQ_CFG" "$prereq" "$outdir" "$seq_len" "$pred_len" "$kernel" \
    "$individual" "$revin_affine" "$batch"

  if [ "$auto_prereq" = "1" ]; then
    "$PYTHON" -u -m scripts.run --config "$cfg" --auto-prereq > "$log" 2>&1
  else
    "$PYTHON" -u -m scripts.run --config "$cfg" > "$log" 2>&1
  fi
}

cd "$ROOT"

# Experiment 1: Norm-only TTA hyperparameter sweep (k_ratio, lr, rollback)
run_one e1_norm_k020_lr5e-5 0 192 96 49 false true 0.20 5e-05 norm 0.0 3.0 20 "checkpoints/murata_fed/seq192_pred96_k49_lr1e-3_r15_s0_0c6f19/best.pt" "" 0
run_one e1_norm_k020_lr1e-4 0 192 96 49 false true 0.20 1e-04 norm 0.0 3.0 20 "checkpoints/murata_fed/seq192_pred96_k49_lr1e-3_r15_s0_0c6f19/best.pt" "" 0
run_one e1_norm_k025_lr5e-5 0 192 96 49 false true 0.25 5e-05 norm 0.0 3.0 20 "checkpoints/murata_fed/seq192_pred96_k49_lr1e-3_r15_s0_0c6f19/best.pt" "" 0
run_one e1_norm_k025_lr1e-4 0 192 96 49 false true 0.25 1e-04 norm 0.0 3.0 20 "checkpoints/murata_fed/seq192_pred96_k49_lr1e-3_r15_s0_0c6f19/best.pt" "" 0
run_one e1_norm_k030_lr1e-4 0 192 96 49 false true 0.30 1e-04 norm 0.0 3.0 20 "checkpoints/murata_fed/seq192_pred96_k49_lr1e-3_r15_s0_0c6f19/best.pt" "" 0
run_one e1_norm_k030_lr1p5e-4 0 192 96 49 false true 0.30 1.5e-04 norm 0.0 3.0 20 "checkpoints/murata_fed/seq192_pred96_k49_lr1e-3_r15_s0_0c6f19/best.pt" "" 0
run_one e1_norm_rb2_w10 0 192 96 49 false true 0.25 1e-04 norm 0.0 2.0 10 "checkpoints/murata_fed/seq192_pred96_k49_lr1e-3_r15_s0_0c6f19/best.pt" "" 0
run_one e1_norm_rb4_w30 0 192 96 49 false true 0.25 1e-04 norm 0.0 4.0 30 "checkpoints/murata_fed/seq192_pred96_k49_lr1e-3_r15_s0_0c6f19/best.pt" "" 0

# Experiment 2: Drift gate sweep (norm-only)
run_one e2_gate_0p10 0 192 96 49 false true 0.25 1e-04 norm 0.10 3.0 20 "checkpoints/murata_fed/seq192_pred96_k49_lr1e-3_r15_s0_0c6f19/best.pt" "" 0
run_one e2_gate_0p20 0 192 96 49 false true 0.25 1e-04 norm 0.20 3.0 20 "checkpoints/murata_fed/seq192_pred96_k49_lr1e-3_r15_s0_0c6f19/best.pt" "" 0
run_one e2_gate_0p30 0 192 96 49 false true 0.25 1e-04 norm 0.30 3.0 20 "checkpoints/murata_fed/seq192_pred96_k49_lr1e-3_r15_s0_0c6f19/best.pt" "" 0

# Experiment 3: RevIN affine + individual ablations (auto-prereq)
run_one e3_revin_affine_false 1 192 96 49 false false 0.25 1e-04 norm 0.0 3.0 20 "" "" 1
run_one e3_individual_true 1 192 96 49 true true 0.25 1e-04 norm 0.0 3.0 20 "" "" 1

# Experiment 4: Longer context windows (auto-prereq)
run_one e4_seq336 1 336 96 49 false true 0.25 1e-04 norm 0.0 3.0 20 "" "" 1
run_one e4_seq672 1 672 96 49 false true 0.25 1e-04 norm 0.0 3.0 20 "" 128 1
