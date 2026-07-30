from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path("/home/jylee/DELTA-KCC2026")
RUN_ROOT = ROOT / "runs" / "kcc_drift_gate_sweep"
DATASETS = ("murata", "electricity", "solar")
# High-variate CSV datasets where a degenerate adapt-rate (~0 or ~1) invalidates the gate.
HIGH_VARIATE = {"electricity", "solar", "traffic"}
GATES = (("gate_0p3", 0.3), ("gate_0p5", 0.5), ("gate_1p0", 1.0))


def latest_stamp(run_root: Path) -> str:
    stamps = sorted(p.name for p in run_root.iterdir() if p.is_dir())
    if not stamps:
        raise FileNotFoundError(f"No sweep runs found under {run_root}")
    return stamps[-1]


def load_sources(stamp_root: Path) -> dict[tuple[str, str], Path]:
    lines = stamp_root.joinpath("sources.tsv").read_text().strip().splitlines()[1:]
    return {(a, b): Path(d) for a, b, _, d in (line.split("\t") for line in lines)}


def resolve_run_dir(base: Path) -> Path:
    if (base / "metrics.json").exists():
        return base
    runs = sorted(p.parent for p in base.rglob("metrics.json"))
    if not runs:
        raise FileNotFoundError(f"No metrics.json under {base}")
    return runs[-1]


def load_payload(base: Path) -> dict:
    run_dir = resolve_run_dir(base)
    return json.loads(run_dir.joinpath("metrics.json").read_text())


def select_gate(rows: dict[str, dict[str, dict]], datasets: tuple[str, ...]) -> tuple[str, str]:
    candidates = []
    for kind, thr in GATES:
        valid = True
        mean_adapt = 0.0
        for dataset in datasets:
            gated = rows[dataset][kind]
            if dataset in HIGH_VARIATE and (gated["adapt_rate"] <= 0.01 or gated["adapt_rate"] >= 0.99):
                valid = False
            if gated["rel_deg"] > 0.5:
                valid = False
            mean_adapt += gated["adapt_rate"]
        candidates.append((kind, thr, valid, mean_adapt / len(datasets)))
    valid_rows = [row for row in candidates if row[2]]
    if not valid_rows:
        return "gate_0p5", "No shared threshold satisfied constraints; defaulting to gate_0p5."
    valid_rows.sort(key=lambda row: (row[3], abs(row[1] - 0.5)))
    return valid_rows[0][0], f"Selected {valid_rows[0][0]} as shared drift threshold."


def bootstrap_mean_ci(values: np.ndarray, n_boot: int, alpha: float, seed: int) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    n = len(values)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    idx = rng.integers(0, n, size=(n_boot, n))
    means = values[idx].mean(axis=1)
    lo = np.quantile(means, alpha / 2)
    hi = np.quantile(means, 1 - alpha / 2)
    return float(values.mean()), float(lo), float(hi)


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap CI for KCC drift-gate sweep")
    parser.add_argument("--stamp", default=None)
    parser.add_argument("--run-root", default=str(RUN_ROOT))
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--datasets", default=",".join(DATASETS), help="Comma-separated dataset list")
    args = parser.parse_args()

    datasets = tuple(d.strip() for d in args.datasets.split(",") if d.strip())
    run_root = Path(args.run_root)
    stamp = args.stamp or latest_stamp(run_root)
    stamp_root = run_root / stamp
    sources = load_sources(stamp_root)

    rows: dict[str, dict[str, dict]] = {d: {} for d in datasets}
    for dataset in datasets:
        backbone = load_payload(sources[(dataset, "backbone")])
        control = load_payload(sources[(dataset, "control")])
        rows[dataset]["backbone"] = {"avg": backbone["avg"], "per_client": backbone["per_client"]}
        rows[dataset]["control"] = {
            "avg": control["avg"],
            "per_client": control["per_client"],
            "per_diag": control.get("per_client_diagnostics", {}),
        }
        for kind, _ in GATES:
            payload = load_payload(sources[(dataset, kind)])
            rows[dataset][kind] = {
                "avg": payload["avg"],
                "per_client": payload["per_client"],
                "per_diag": payload.get("per_client_diagnostics", {}),
            }
            rel_deg = 100.0 * (payload["avg"]["mse"] - backbone["avg"]["mse"]) / backbone["avg"]["mse"]
            rows[dataset][kind]["rel_deg"] = rel_deg
            rows[dataset][kind]["adapt_rate"] = payload.get("diagnostic_summary", {}).get("adapt_rate", 0.0)

    gate_kind, gate_msg = select_gate(rows, datasets)

    lines = [f"# KCC Drift-Gate Bootstrap CI ({stamp})", "", gate_msg, ""]
    lines.append("| dataset | metric | mean diff | CI low | CI high | n_clients |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: |")

    for dataset in datasets:
        base = rows[dataset]["backbone"]["per_client"]
        gate = rows[dataset][gate_kind]["per_client"]
        ids = sorted(set(base.keys()) & set(gate.keys()))
        if not ids:
            continue
        base_mse = np.array([base[i]["mse"] for i in ids], dtype=float)
        gate_mse = np.array([gate[i]["mse"] for i in ids], dtype=float)
        base_mae = np.array([base[i]["mae"] for i in ids], dtype=float)
        gate_mae = np.array([gate[i]["mae"] for i in ids], dtype=float)
        base_smape = np.array([base[i]["smape"] for i in ids], dtype=float)
        gate_smape = np.array([gate[i]["smape"] for i in ids], dtype=float)

        for name, delta in (
            ("mse", gate_mse - base_mse),
            ("mae", gate_mae - base_mae),
            ("smape", gate_smape - base_smape),
        ):
            mean, lo, hi = bootstrap_mean_ci(delta, args.n_bootstrap, args.alpha, args.seed)
            lines.append(f"| {dataset} | {name} | {mean:.6f} | {lo:.6f} | {hi:.6f} | {len(ids)} |")

        ctrl_diag = rows[dataset]["control"]["per_diag"]
        gate_diag = rows[dataset][gate_kind]["per_diag"]
        diag_ids = sorted(set(ctrl_diag.keys()) & set(gate_diag.keys()))
        if diag_ids:
            ctrl_adapt = np.array([ctrl_diag[i]["adapt_rate"] for i in diag_ids], dtype=float)
            gate_adapt = np.array([gate_diag[i]["adapt_rate"] for i in diag_ids], dtype=float)
            reduction = 1.0 - gate_adapt / np.maximum(ctrl_adapt, 1e-8)
            mean, lo, hi = bootstrap_mean_ci(reduction, args.n_bootstrap, args.alpha, args.seed)
            lines.append(f"| {dataset} | adapt_reduction | {mean:.6f} | {lo:.6f} | {hi:.6f} | {len(diag_ids)} |")

    out_dir = stamp_root / "report_outputs"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "ci_summary.md"
    out_path.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nWrote CI summary to {out_path}")


if __name__ == "__main__":
    main()
