from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path("/home/jylee/DELTA-KCC2026")
RUN_ROOT = ROOT / "runs" / "murata_time_affine_hardgate_sweep"


def _latest_stamp() -> str:
    stamps = sorted(path.name for path in RUN_ROOT.iterdir() if path.is_dir())
    if not stamps:
        raise FileNotFoundError(f"No sweep runs found under {RUN_ROOT}")
    return stamps[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Murata time-affine hard-gate sweep results")
    parser.add_argument("--stamp", default=None, help="Sweep stamp under runs/murata_time_affine_hardgate_sweep")
    args = parser.parse_args()
    stamp = args.stamp or _latest_stamp()
    rows = []
    for variant_dir in sorted((RUN_ROOT / stamp).iterdir()):
        if not variant_dir.is_dir():
            continue
        run_dirs = sorted(p for p in variant_dir.iterdir() if p.is_dir())
        if not run_dirs:
            continue
        metrics_path = run_dirs[-1] / "metrics.json"
        if not metrics_path.exists():
            continue
        payload = json.loads(metrics_path.read_text())
        rows.append((variant_dir.name, payload["avg"], payload["diagnostic_summary"]))
    if not rows:
        raise FileNotFoundError(f"No metrics found for stamp {stamp}")

    print(f"=== Murata Time-Affine Hard-Gate Sweep Report | stamp={stamp} ===")
    for name, avg, diag in sorted(rows, key=lambda row: row[1]["mse"]):
        print(
            f"{name:18s} "
            f"MSE={avg['mse']:.4f} MAE={avg['mae']:.4f} sMAPE={avg['smape']:.2f}% "
            f"adapt={diag['adapt_rate']:.3f} hard={diag['hard_gate_skip_rate']:.3f} "
            f"rollback={diag['rollback_skip_rate']:.3f} reset={diag['reset_rate_given_adapt']:.3f}"
        )


if __name__ == "__main__":
    main()
