from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path("/home/jylee/DELTA-KCC2026")
RUN_ROOT = ROOT / "runs" / "murata_affine_scout"


def _load_metrics(stamp: str) -> list[dict]:
    rows: list[dict] = []
    stamp_root = RUN_ROOT / stamp
    for variant_dir in sorted(p for p in stamp_root.iterdir() if p.is_dir()):
        run_dirs = sorted(p for p in variant_dir.iterdir() if p.is_dir())
        if not run_dirs:
            continue
        metrics_path = run_dirs[-1] / "metrics.json"
        if not metrics_path.exists():
            continue
        payload = json.loads(metrics_path.read_text())
        diag = payload["diagnostic_summary"]
        avg = payload["avg"]
        rows.append({
            "name": variant_dir.name,
            "mse": avg["mse"],
            "mae": avg["mae"],
            "smape": avg["smape"],
            "adapt_rate": diag["adapt_rate"],
            "rollback_skip_rate": diag["rollback_skip_rate"],
            "reset_rate_given_adapt": diag["reset_rate_given_adapt"],
            "mean_gamma_l1": diag["mean_gamma_l1"],
            "mean_delta_l1": diag["mean_delta_l1"],
            "final_gamma_l1": diag["final_gamma_l1"],
            "final_delta_l1": diag["final_delta_l1"],
        })
    return sorted(rows, key=lambda row: row["mse"])


def _promotions(rows: list[dict]) -> list[str]:
    channel_rows = [row for row in rows if row["name"].startswith("channel_")]
    time_rows = [row for row in rows if row["name"].startswith("time_")]
    best_channel = min(channel_rows, key=lambda row: row["mse"])
    best_time = min(time_rows, key=lambda row: row["mse"])
    if best_channel["mse"] - best_time["mse"] >= 0.001:
        return [row["name"] for row in sorted(time_rows, key=lambda row: row["mse"])[:2]]
    return [best_time["name"], best_channel["name"]]


def _latest_stamp() -> str:
    stamps = sorted(path.name for path in RUN_ROOT.iterdir() if path.is_dir())
    if not stamps:
        raise FileNotFoundError(f"No scout runs found under {RUN_ROOT}")
    return stamps[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Murata affine scout results")
    parser.add_argument("--stamp", default=None, help="Scout run stamp under runs/murata_affine_scout")
    args = parser.parse_args()
    stamp = args.stamp or _latest_stamp()
    rows = _load_metrics(stamp)
    if not rows:
        raise FileNotFoundError(f"No metrics found for stamp {stamp}")

    print(f"=== Murata Affine Scout Report | stamp={stamp} ===")
    for row in rows:
        print(
            f"{row['name']:24s} "
            f"MSE={row['mse']:.4f} MAE={row['mae']:.4f} sMAPE={row['smape']:.2f}% "
            f"adapt={row['adapt_rate']:.3f} rollback={row['rollback_skip_rate']:.3f} "
            f"reset={row['reset_rate_given_adapt']:.3f} "
            f"|g-1|={row['mean_gamma_l1']:.6f} |d|={row['mean_delta_l1']:.6f} "
            f"final_g={row['final_gamma_l1']:.6f} final_d={row['final_delta_l1']:.6f}"
        )

    promoted = _promotions(rows)
    print("")
    print("Promotion candidates:")
    for name in promoted:
        print(name)


if __name__ == "__main__":
    main()
