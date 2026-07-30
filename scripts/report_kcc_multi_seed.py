from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path("/home/jylee/DELTA-KCC2026")
RUN_ROOT = ROOT / "runs" / "kcc_multi_seed"


def latest_stamp(run_root: Path) -> str:
    stamps = sorted(path.name for path in run_root.iterdir() if path.is_dir())
    if not stamps:
        raise FileNotFoundError(f"No multi-seed runs found under {run_root}")
    return stamps[-1]


def resolve_run_dir(base: Path) -> Path:
    if (base / "metrics.json").exists():
        return base
    runs = sorted(path.parent for path in base.rglob("metrics.json"))
    if len(runs) != 1:
        raise FileNotFoundError(f"Expected exactly one metrics.json under {base}, found {len(runs)}")
    return runs[0]


def load_payload(base: Path) -> dict:
    return json.loads(resolve_run_dir(base).joinpath("metrics.json").read_text())


def summarize(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=0))


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize DLST multi-seed variance runs")
    parser.add_argument("--run-root", default=str(RUN_ROOT))
    parser.add_argument("--stamp", default=None)
    parser.add_argument("--dataset", default="murata")
    parser.add_argument("--variants", nargs="+", default=["control", "gate_1p0"])
    parser.add_argument("--seeds", nargs="+", default=["0", "1", "2"])
    args = parser.parse_args()

    run_root = Path(args.run_root)
    stamp = args.stamp or latest_stamp(run_root)
    stamp_root = run_root / stamp
    rows = [["variant", "mse_mean", "mse_std", "mae_mean", "mae_std", "smape_mean", "smape_std", "adapt_rate_mean", "adapt_rate_std"]]
    md = [
        f"# KCC Multi-Seed Summary ({stamp})",
        "",
        f"Dataset: `{args.dataset}`",
        "",
        "| variant | MSE | MAE | sMAPE | adapt rate |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]

    for variant in args.variants:
        payloads = []
        for seed in args.seeds:
            run_base = stamp_root / f"{variant}_s{seed}"
            if not run_base.exists():
                raise FileNotFoundError(f"Missing run directory for variant={variant}, seed={seed}: {run_base}")
            payloads.append(load_payload(run_base))
        mse_mean, mse_std = summarize([payload["avg"]["mse"] for payload in payloads])
        mae_mean, mae_std = summarize([payload["avg"]["mae"] for payload in payloads])
        smape_mean, smape_std = summarize([payload["avg"]["smape"] for payload in payloads])
        adapt_mean, adapt_std = summarize([
            payload.get("diagnostic_summary", {}).get("adapt_rate", 0.0)
            for payload in payloads
        ])
        rows.append([
            variant,
            f"{mse_mean:.6f}",
            f"{mse_std:.6f}",
            f"{mae_mean:.6f}",
            f"{mae_std:.6f}",
            f"{smape_mean:.6f}",
            f"{smape_std:.6f}",
            f"{adapt_mean:.6f}",
            f"{adapt_std:.6f}",
        ])
        md.append(
            f"| {variant} | {mse_mean:.4f} ± {mse_std:.4f} | {mae_mean:.4f} ± {mae_std:.4f} | "
            f"{smape_mean:.2f} ± {smape_std:.2f} | {adapt_mean:.3f} ± {adapt_std:.3f} |"
        )

    out_dir = stamp_root / "report_outputs"
    out_dir.mkdir(exist_ok=True)
    with (out_dir / "multi_seed_summary.csv").open("w", newline="") as handle:
        csv.writer(handle).writerows(rows)
    (out_dir / "multi_seed_summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"\nWrote multi-seed summary to {out_dir}")


if __name__ == "__main__":
    main()
