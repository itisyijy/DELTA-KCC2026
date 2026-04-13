from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ROOT = Path("/home/jylee/DLinear-Season-Trend")
RUN_ROOT = ROOT / "runs" / "kcc_drift_gate_sweep"
GATES = ("gate_0p3", "gate_0p5", "gate_1p0")


def latest_stamp(run_root: Path) -> str:
    stamps = sorted(path.name for path in run_root.iterdir() if path.is_dir())
    if not stamps:
        raise FileNotFoundError(f"No sweep runs found under {run_root}")
    return stamps[-1]


def load_sources(stamp_root: Path) -> dict[tuple[str, str], Path]:
    lines = stamp_root.joinpath("sources.tsv").read_text().strip().splitlines()[1:]
    return {(a, b): Path(d) for a, b, _, d in (line.split("\t") for line in lines)}


def resolve_run_dir(base: Path) -> Path:
    if (base / "metrics.json").exists():
        return base
    runs = sorted(path.parent for path in base.rglob("metrics.json"))
    if not runs:
        raise FileNotFoundError(f"No metrics.json under {base}")
    return runs[-1]


def load_payload(base: Path) -> dict:
    return json.loads(resolve_run_dir(base).joinpath("metrics.json").read_text())


def available_datasets(
    sources: dict[tuple[str, str], Path],
    requested: list[str] | None,
) -> list[str]:
    candidates = requested or sorted({dataset for dataset, _ in sources})
    need = ("backbone", "control", *GATES)
    datasets = [dataset for dataset in candidates if all((dataset, kind) in sources for kind in need)]
    if not datasets:
        raise FileNotFoundError("No dataset has complete backbone/control/gate artifacts.")
    return datasets


def pct_drop(base: float, new: float) -> float:
    return 100.0 * (1.0 - new / max(base, 1e-8))


def metric(payload: dict, key: str) -> float:
    return float(payload.get(key, 0.0))


def rel_deg(backbone: dict, variant: dict) -> float:
    return 100.0 * (variant["avg"]["mse"] - backbone["avg"]["mse"]) / backbone["avg"]["mse"]


def choose_gate(rows: dict[str, dict[str, dict]], datasets: list[str]) -> str:
    candidates: list[tuple[float, float, str]] = []
    for gate in GATES:
        valid = True
        mean_adapt = 0.0
        for dataset in datasets:
            variant = rows[dataset][gate]
            adapt = float(variant.get("diagnostic_summary", {}).get("adapt_rate", 0.0))
            mean_adapt += adapt
            if rel_deg(rows[dataset]["backbone"], variant) > 0.5:
                valid = False
            if dataset in {"electricity", "solar"} and (adapt <= 0.01 or adapt >= 0.99):
                valid = False
        if valid:
            dist = abs(float(gate.split("_")[-1].replace("p", ".")) - 0.5)
            candidates.append((mean_adapt / len(datasets), dist, gate))
    return sorted(candidates)[0][2] if candidates else "gate_1p0"


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize DLST wall-clock efficiency evidence")
    parser.add_argument("--run-root", default=str(RUN_ROOT))
    parser.add_argument("--stamp", default=None)
    parser.add_argument("--datasets", nargs="*", default=None)
    args = parser.parse_args()

    run_root = Path(args.run_root)
    stamp = args.stamp or latest_stamp(run_root)
    stamp_root = run_root / stamp
    sources = load_sources(stamp_root)
    datasets = available_datasets(sources, args.datasets)
    rows = {
        dataset: {kind: load_payload(sources[(dataset, kind)]) for kind in ("backbone", "control", *GATES)}
        for dataset in datasets
    }
    gate = choose_gate(rows, datasets)

    table = [[
        "dataset", "backbone_wall_clock_sec", "control_wall_clock_sec", "gate_wall_clock_sec",
        "wall_reduction_pct", "control_avg_round_sec", "gate_avg_round_sec",
        "control_avg_local_adapt_sec", "gate_avg_local_adapt_sec", "local_adapt_reduction_pct",
        "adapt_rate_drop_pct", "gate_peak_vram_mb", "trainable_param_ratio", "relative_mse_change_pct",
    ]]
    md = [
        f"# KCC Efficiency Summary ({stamp})",
        "",
        f"Datasets: `{', '.join(datasets)}`",
        f"Selected gate: `{gate}`",
        "",
        "| dataset | backbone wall | control wall | gate wall | wall drop | local adapt drop | control adapt | gate adapt | peak VRAM | param ratio |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    wall_drops, round_drops, local_drops, adapt_drops, rel_degs = [], [], [], [], []
    max_vram, max_ratio = 0.0, 0.0
    for dataset in datasets:
        backbone, control, gated = rows[dataset]["backbone"], rows[dataset]["control"], rows[dataset][gate]
        control_wall, gate_wall = metric(control, "wall_clock_sec"), metric(gated, "wall_clock_sec")
        control_round, gate_round = metric(control, "avg_round_sec"), metric(gated, "avg_round_sec")
        control_local, gate_local = metric(control, "avg_local_adapt_sec"), metric(gated, "avg_local_adapt_sec")
        control_adapt = float(control.get("diagnostic_summary", {}).get("adapt_rate", 0.0))
        gate_adapt = float(gated.get("diagnostic_summary", {}).get("adapt_rate", 0.0))
        wall_drop = pct_drop(control_wall, gate_wall)
        round_drop = pct_drop(control_round, gate_round)
        local_drop = pct_drop(control_local, gate_local)
        adapt_drop = pct_drop(control_adapt, gate_adapt)
        deg = rel_deg(backbone, gated)
        max_vram = max(max_vram, metric(gated, "peak_vram_mb"))
        max_ratio = max(max_ratio, metric(gated, "trainable_param_ratio"))
        wall_drops.append(wall_drop)
        round_drops.append(round_drop)
        local_drops.append(local_drop)
        adapt_drops.append(adapt_drop)
        rel_degs.append(deg)
        table.append([
            dataset, f"{metric(backbone, 'wall_clock_sec'):.6f}", f"{control_wall:.6f}",
            f"{gate_wall:.6f}", f"{wall_drop:.6f}", f"{control_round:.6f}", f"{gate_round:.6f}",
            f"{control_local:.6f}", f"{gate_local:.6f}", f"{local_drop:.6f}", f"{adapt_drop:.6f}",
            f"{metric(gated, 'peak_vram_mb'):.3f}", f"{metric(gated, 'trainable_param_ratio'):.6f}",
            f"{deg:.6f}",
        ])
        md.append(
            f"| {dataset} | {metric(backbone, 'wall_clock_sec'):.2f}s | {control_wall:.2f}s | {gate_wall:.2f}s | "
            f"{wall_drop:.1f}% | {local_drop:.1f}% | {control_adapt:.3f} | {gate_adapt:.3f} | "
            f"{metric(gated, 'peak_vram_mb'):.1f} MB | {100*metric(gated, 'trainable_param_ratio'):.3f}% |"
        )

    decision = {
        "stamp": stamp,
        "datasets": datasets,
        "selected_gate": gate,
        "mean_wall_clock_reduction_pct": sum(wall_drops) / len(wall_drops),
        "mean_round_reduction_pct": sum(round_drops) / len(round_drops),
        "mean_local_adapt_reduction_pct": sum(local_drops) / len(local_drops),
        "mean_adapt_rate_drop_pct": sum(adapt_drops) / len(adapt_drops),
        "max_relative_mse_change_pct": max(rel_degs),
        "no_harm_ok": max(rel_degs) <= 0.5,
    }
    decision["pilot_clear_gain"] = bool(
        decision["no_harm_ok"] and (
            decision["mean_wall_clock_reduction_pct"] >= 3.0
            or decision["mean_local_adapt_reduction_pct"] >= 5.0
        )
    )

    paragraph = (
        f"On `cuda:1`, the selected DLST gate `{gate}` over `{', '.join(datasets)}` reduced end-to-end wall-clock time by "
        f"{decision['mean_wall_clock_reduction_pct']:.1f}% on average relative to the always-adapting control while lowering "
        f"per-window round time by {decision['mean_round_reduction_pct']:.1f}% and local adaptation time by "
        f"{decision['mean_local_adapt_reduction_pct']:.1f}%. Adaptation frequency dropped by "
        f"{decision['mean_adapt_rate_drop_pct']:.1f}%, peak GPU memory stayed below {max_vram:.1f} MB, "
        f"and the trainable parameter ratio remained {100*max_ratio:.3f}% of the frozen backbone."
    )

    out_dir = stamp_root / "report_outputs"
    out_dir.mkdir(exist_ok=True)
    with (out_dir / "efficiency_table.csv").open("w", newline="") as handle:
        csv.writer(handle).writerows(table)
    (out_dir / "efficiency_summary.md").write_text("\n".join(md) + "\n")
    (out_dir / "rebuttal_paragraph.md").write_text(paragraph + "\n")
    (out_dir / "efficiency_decision.json").write_text(json.dumps(decision, indent=2) + "\n")
    print("\n".join(md))
    print(f"\n{paragraph}")
    print(f"\nDecision: {json.dumps(decision)}")
    print(f"\nWrote efficiency artifacts to {out_dir}")


if __name__ == "__main__":
    main()
