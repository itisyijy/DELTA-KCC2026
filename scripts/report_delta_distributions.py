from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.data.loader import load_csv_as_clients, load_parquet_as_clients

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STAMP = ROOT / "runs/kcc_drift_gate_sweep/20260724_combined4"
DATASETS = ("murata", "solar", "electricity", "traffic")
EXPECTED_DIFF = {
    "murata": "0.000047",
    "solar": "0.000000",
    "electricity": "-0.000148",
    "traffic": "0.000122",
}
EXPECTED_N = {"murata": 30, "solar": 137, "electricity": 321, "traffic": 862}


def resolve_source(path_text: str) -> Path:
    path = Path(path_text)
    if not path.exists():
        path = Path(path_text.replace("/home/jylee/DLinear-Season-Trend", str(ROOT)))
    runs = [path] if (path / "metrics.json").exists() else sorted(
        p.parent for p in path.rglob("metrics.json")
    )
    if not runs:
        raise FileNotFoundError(f"No metrics.json under {path}")
    return runs[-1]


def load_sources(stamp: Path) -> dict[tuple[str, str], Path]:
    lines = stamp.joinpath("sources.tsv").read_text().splitlines()[1:]
    return {
        (dataset, kind): resolve_source(path)
        for dataset, kind, _, path in (line.split("\t") for line in lines)
    }


def load_run(path: Path) -> tuple[dict, dict]:
    metrics = json.loads(path.joinpath("metrics.json").read_text())
    config = json.loads(path.joinpath("config.json").read_text())
    return metrics, config


def validate_config(config: dict, *, gated: bool) -> None:
    assert config["seed"] == 0
    assert config["model"]["pred_len"] == 96
    if gated:
        assert config["tta"]["k_ratio"] == 0.0625
        assert config["tta"]["adapter_mode"] == "time_affine"


def no_harm_stats(backbone: dict, gated: dict) -> dict:
    ids = sorted(set(backbone["per_client"]) & set(gated["per_client"]))
    base = np.array([backbone["per_client"][i]["mse"] for i in ids])
    gate = np.array([gated["per_client"][i]["mse"] for i in ids])
    delta = gate - base
    rel = 100.0 * delta / base
    over = rel > 0.5
    return {
        "n": len(ids),
        "mean_delta": float(delta.mean()),
        "rel_gt_0p5_count": int(over.sum()),
        "rel_gt_0p5_rate": float(over.mean()),
        "rel_gt_0_count": int((rel > 0).sum()),
        "rel_gt_0_rate": float((rel > 0).mean()),
        "rel_percentiles": dict(
            zip(("p50", "p95", "max"), map(float, np.quantile(rel, (0.5, 0.95, 1.0))))
        ),
        "over_0p5_clients": [
            {
                "client_id": ids[i],
                "mse_backbone": float(base[i]),
                "delta": float(delta[i]),
                "rel_percent": float(rel[i]),
            }
            for i in np.flatnonzero(over)
        ],
    }


def load_clients(config: dict) -> list:
    model = config["model"]
    common = {
        "seq_len": model["seq_len"],
        "pred_len": model["pred_len"],
        "max_clients": config["max_clients"],
    }
    if config["data_format"] == "csv":
        return load_csv_as_clients(
            config["data_path"], config["timestamp_col"], **common
        )
    return load_parquet_as_clients(
        config["data_path"], config["parquet_clients_dir"], **common
    )


def client_scores(client, start: int, steps: int, k: int) -> np.ndarray:
    windows = torch.from_numpy(client.values[:, 0]).unfold(0, k, 1)
    windows = windows[start : start + steps]
    means = windows.mean(1).numpy().astype(np.float64)
    stds = windows.std(1, unbiased=False).numpy().astype(np.float64)
    scale = max(client.global_std, 1e-8)
    mean_shift = np.abs((means - client.global_mean) / scale)
    std_shift = np.abs((stds - client.global_std) / scale)
    return np.maximum(mean_shift, std_shift)


def drift_stats(config: dict) -> dict:
    clients = load_clients(config)
    model = config["model"]
    k = int(config["tta"]["k_ratio"] * model["pred_len"])
    test_start, test_end = clients[0].split_indices["test"]
    start = test_start + model["seq_len"]
    steps = test_end - (test_start + model["seq_len"] + k - 1)
    per_client = [client_scores(client, start, steps, k) for client in clients]
    scores = np.concatenate(per_client)

    def prefix_rate(frac: float) -> tuple[float, int]:
        count = int(steps * frac)
        prefix = np.concatenate([values[:count] for values in per_client])
        return float((prefix < 1.0).mean()), count

    front_10, windows_10 = prefix_rate(0.10)
    front_05, windows_05 = prefix_rate(0.05)
    return {
        "n_clients": len(clients),
        "windows_per_client": steps,
        "client_window_pairs": len(scores),
        "skip_rate": {
            str(threshold): float((scores < threshold).mean())
            for threshold in (0.3, 0.5, 1.0)
        },
        "front_10pct_lt_1p0": front_10,
        "front_10pct_windows_per_client": windows_10,
        "front_05pct_lt_1p0": front_05,
        "front_05pct_windows_per_client": windows_05,
        "score_percentiles": dict(
            zip(("p25", "p50", "p75"), map(float, np.quantile(scores, (0.25, 0.5, 0.75))))
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stamp", type=Path, default=DEFAULT_STAMP)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    sources = load_sources(args.stamp)
    report = {}
    for dataset in DATASETS:
        backbone, backbone_cfg = load_run(sources[dataset, "backbone"])
        gated, gated_cfg = load_run(sources[dataset, "gate_1p0"])
        validate_config(backbone_cfg, gated=False)
        validate_config(gated_cfg, gated=True)
        no_harm = no_harm_stats(backbone, gated)
        assert no_harm["n"] == EXPECTED_N[dataset]
        assert f'{no_harm["mean_delta"]:.6f}' == EXPECTED_DIFF[dataset]
        drift = drift_stats(gated_cfg)
        for kind, threshold in (("gate_0p3", 0.3), ("gate_0p5", 0.5), ("gate_1p0", 1.0)):
            metrics, config = load_run(sources[dataset, kind])
            validate_config(config, gated=True)
            assert config["tta"]["drift_gate_threshold"] == threshold
            logged = metrics["diagnostic_summary"]["drift_skip_rate"]
            assert np.isclose(drift["skip_rate"][str(threshold)], logged, atol=1e-15)
        report[dataset] = {"no_harm": no_harm, "drift": drift}
    text = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
