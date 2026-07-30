from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path("/home/jylee/DELTA-KCC2026")
RUN_ROOT = ROOT / "runs" / "elec_solar_compare"


def _latest_stamp() -> str:
    stamps = sorted(path.name for path in RUN_ROOT.iterdir() if path.is_dir())
    if not stamps:
        raise FileNotFoundError(f"No compare runs found under {RUN_ROOT}")
    return stamps[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize electricity/solar backbone vs TTA results")
    parser.add_argument("--stamp", default=None, help="Run stamp under runs/elec_solar_compare")
    args = parser.parse_args()
    stamp = args.stamp or _latest_stamp()
    stamp_root = RUN_ROOT / stamp
    rows = []
    for variant_dir in sorted(p for p in stamp_root.iterdir() if p.is_dir()):
        run_dirs = sorted(p for p in variant_dir.iterdir() if p.is_dir())
        if not run_dirs:
            continue
        metrics_path = run_dirs[-1] / "metrics.json"
        if not metrics_path.exists():
            continue
        payload = json.loads(metrics_path.read_text())
        avg = payload["avg"]
        rows.append((variant_dir.name, avg))
    if not rows:
        raise FileNotFoundError(f"No metrics found for stamp {stamp}")

    print(f"=== Elec/Solar Backbone vs TTA | stamp={stamp} ===")
    for dataset in ("electricity", "solar"):
        subset = [(name, avg) for name, avg in rows if name.startswith(dataset + "_")]
        subset.sort(key=lambda row: row[1]["mse"])
        print(f"[{dataset}]")
        for name, avg in subset:
            print(f"{name:32s} MSE={avg['mse']:.4f} MAE={avg['mae']:.4f} sMAPE={avg['smape']:.2f}%")


if __name__ == "__main__":
    main()
