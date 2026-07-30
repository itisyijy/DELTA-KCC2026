from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.models.revin_dlinear import RevINDLinear
from scripts.tta.adapter import AffineAdapter

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


def load_sources(stamp_root: Path) -> dict[tuple[str, str], tuple[str, Path]]:
    lines = stamp_root.joinpath("sources.tsv").read_text().strip().splitlines()[1:]
    return {(a, b): (c, Path(d)) for a, b, c, d in (line.split("\t") for line in lines)}


def resolve_run_dir(base: Path) -> Path:
    if (base / "metrics.json").exists():
        return base
    runs = sorted(p.parent for p in base.rglob("metrics.json"))
    if not runs:
        raise FileNotFoundError(f"No metrics.json under {base}")
    return runs[-1]


def load_result(base: Path) -> tuple[dict, dict]:
    run_dir = resolve_run_dir(base)
    return json.loads(run_dir.joinpath("metrics.json").read_text()), json.loads(run_dir.joinpath("config.json").read_text())


def param_stats(cfg: dict) -> tuple[int, int, float]:
    model = cfg["model"]
    backbone = RevINDLinear(model["seq_len"], model["pred_len"], 1, model["kernel_size"], model["individual"], model["revin_affine"])
    adapter = AffineAdapter(1, pred_len=model["pred_len"], mode=cfg["tta"].get("adapter_mode", "time_affine"))
    b_count = sum(p.numel() for p in backbone.parameters())
    a_count = sum(p.numel() for p in adapter.parameters())
    return a_count, b_count, a_count / b_count


def gate_eval(backbone: float, control: float, gated: float, control_adapt: float, gated_adapt: float) -> tuple[float, float]:
    rel_deg = 100.0 * (gated - backbone) / backbone
    adapt_drop = 100.0 * (1.0 - gated_adapt / max(control_adapt, 1e-8))
    return rel_deg, adapt_drop


def choose_threshold(rows: dict[str, dict[str, dict]], datasets: tuple[str, ...]) -> tuple[str | None, str]:
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
        return None, "Murata-only primary: no shared drift threshold satisfied no-harm and selectivity."
    valid_rows.sort(key=lambda row: (row[3], abs(row[1] - 0.5)))
    return valid_rows[0][0], f"Selected {valid_rows[0][0]} as shared drift threshold."


def write_csv(path: Path, rows: list[list[str]]) -> None:
    with path.open("w", newline="") as f:
        csv.writer(f).writerows(rows)


def write_svg(path: Path, selected: str, rows: dict[str, dict[str, dict]], datasets: tuple[str, ...]) -> None:
    labels = list(datasets)
    vals = [(rows[d]["control"]["adapt_rate"], rows[d][selected]["adapt_rate"]) for d in labels]
    bars, width = [], 420
    for i, (ctrl, gate) in enumerate(vals):
        y = 40 + 60 * i
        bars.append(f'<text x="10" y="{y+12}" font-size="12">{labels[i]}</text>')
        bars.append(f'<rect x="110" y="{y}" width="{280*ctrl:.1f}" height="14" fill="#9ecae1"/>')
        bars.append(f'<rect x="110" y="{y+20}" width="{280*gate:.1f}" height="14" fill="#3182bd"/>')
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="460" height="230">', '<text x="10" y="20" font-size="14">Adapt Rate: control vs selected gate</text>']
    svg.extend(bars)
    svg.extend(['<text x="110" y="215" font-size="11">light=control dark=selected</text>', f'<line x1="110" y1="200" x2="{110+width}" y2="200" stroke="#333"/>', "</svg>"])
    path.write_text("\n".join(svg))


def murata_hardgate_rows() -> list[str]:
    root = ROOT / "runs" / "murata_time_affine_hardgate_sweep"
    if not root.exists():
        return []
    stamps = sorted(p for p in root.iterdir() if p.is_dir())
    if not stamps:
        return []
    stamp = stamps[-1]
    lines = ["", "| Murata Ablation | MSE | adapt | hard skip |", "| --- | ---: | ---: | ---: |"]
    for name in ("time_best_control", "time_gate1p10"):
        payload, _ = load_result(stamp / name)
        diag = payload.get("diagnostic_summary", {})
        lines.append(f"| {name} | {payload['avg']['mse']:.4f} | {diag.get('adapt_rate', 0.0):.3f} | {diag.get('hard_gate_skip_rate', 0.0):.3f} |")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize KCC drift-gate sweep")
    parser.add_argument("--run-root", default=str(RUN_ROOT))
    parser.add_argument("--stamp", default=None)
    parser.add_argument("--datasets", default=",".join(DATASETS), help="Comma-separated dataset list")
    args = parser.parse_args()
    datasets = tuple(d.strip() for d in args.datasets.split(",") if d.strip())
    run_root = Path(args.run_root)
    stamp = args.stamp or latest_stamp(run_root)
    stamp_root = run_root / stamp
    sources = load_sources(stamp_root)
    rows: dict[str, dict[str, dict]] = {d: {} for d in datasets}
    for dataset in datasets:
        for kind in ("backbone", "control", *(name for name, _ in GATES)):
            payload, cfg = load_result(sources[(dataset, kind)][1])
            diag = payload.get("diagnostic_summary", {})
            a_cnt, b_cnt, ratio = param_stats(cfg) if kind != "backbone" else (0, *param_stats(cfg)[1:])
            rows[dataset][kind] = {
                "mse": payload["avg"]["mse"],
                "mae": payload["avg"]["mae"],
                "smape": payload["avg"]["smape"],
                "adapt_rate": diag.get("adapt_rate", 0.0),
                "drift_skip_rate": diag.get("drift_skip_rate", 0.0),
                "rollback_skip_rate": diag.get("rollback_skip_rate", 0.0),
                "reset_rate_given_adapt": diag.get("reset_rate_given_adapt", 0.0),
                "mean_gamma_l1": diag.get("mean_gamma_l1", 0.0),
                "mean_delta_l1": diag.get("mean_delta_l1", 0.0),
                "adapter_params": a_cnt,
                "param_ratio": ratio,
            }
        for kind, _ in GATES:
            rel_deg, adapt_drop = gate_eval(rows[dataset]["backbone"]["mse"], rows[dataset]["control"]["mse"], rows[dataset][kind]["mse"], rows[dataset]["control"]["adapt_rate"], rows[dataset][kind]["adapt_rate"])
            rows[dataset][kind]["rel_deg"] = rel_deg
            rows[dataset][kind]["adapt_drop"] = adapt_drop
    selected, verdict = choose_threshold(rows, datasets)
    deg_headers = " | ".join(f"{d} deg%" for d in datasets)
    md = [f"# KCC Drift-Gate Summary ({stamp})", "", verdict, "", f"| threshold | {deg_headers} | avg adapt |", "| --- |" + " ---: |" * (len(datasets) + 1)]
    sweep_csv = [["threshold", *(f"{d}_rel_deg" for d in datasets), "avg_adapt_rate"]]
    for kind, thr in GATES:
        avg_adapt = sum(rows[d][kind]["adapt_rate"] for d in datasets) / len(datasets)
        degs = " | ".join(f"{rows[d][kind]['rel_deg']:.3f}" for d in datasets)
        md.append(f"| {thr:.1f} | {degs} | {avg_adapt:.3f} |")
        sweep_csv.append([f"{thr:.1f}", *(f"{rows[d][kind]['rel_deg']:.6f}" for d in datasets), f"{avg_adapt:.6f}"])
    selected = selected or "gate_0p5"
    main_csv = [["dataset", "method", "mse", "mae", "smape"]]
    eff_csv = [["dataset", "control_adapt_rate", "gate_adapt_rate", "drift_skip_rate", "rollback_skip_rate", "param_ratio"]]
    md.extend(["", "| dataset | method | MSE | MAE | sMAPE |", "| --- | --- | ---: | ---: | ---: |"])
    for dataset in datasets:
        for kind, label in (("backbone", "backbone"), ("control", "control"), (selected, selected)):
            row = rows[dataset][kind]
            md.append(f"| {dataset} | {label} | {row['mse']:.4f} | {row['mae']:.4f} | {row['smape']:.2f} |")
            main_csv.append([dataset, label, f"{row['mse']:.6f}", f"{row['mae']:.6f}", f"{row['smape']:.6f}"])
        gate = rows[dataset][selected]
        eff_csv.append([dataset, f"{rows[dataset]['control']['adapt_rate']:.6f}", f"{gate['adapt_rate']:.6f}", f"{gate['drift_skip_rate']:.6f}", f"{gate['rollback_skip_rate']:.6f}", f"{gate['param_ratio']:.6f}"])
    md.extend(["", "| dataset | control adapt | gated adapt | drift skip | rollback skip | param ratio |", "| --- | ---: | ---: | ---: | ---: | ---: |"])
    for row in eff_csv[1:]:
        md.append(f"| {row[0]} | {float(row[1]):.3f} | {float(row[2]):.3f} | {float(row[3]):.3f} | {float(row[4]):.3f} | {100*float(row[5]):.3f}% |")
    md.extend(murata_hardgate_rows())
    out = stamp_root / "report_outputs"
    out.mkdir(exist_ok=True)
    (out / "summary.md").write_text("\n".join(md) + "\n")
    write_csv(out / "threshold_sweep.csv", sweep_csv)
    write_csv(out / "main_table.csv", main_csv)
    write_csv(out / "efficiency_table.csv", eff_csv)
    write_svg(out / "adapt_rate.svg", selected, rows, datasets)
    print("\n".join(md))
    print(f"\nWrote report artifacts to {out}")


if __name__ == "__main__":
    main()
