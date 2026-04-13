from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from time import perf_counter
from typing import Iterable

import torch


def count_parameters(
    source: torch.nn.Module | Iterable[torch.nn.Parameter],
    *,
    trainable_only: bool = False,
) -> int:
    params = source.parameters() if hasattr(source, "parameters") else source
    return sum(
        int(param.numel())
        for param in params
        if not trainable_only or param.requires_grad
    )


def _sample_window(
    values: list[float],
    *,
    warmup_rounds: int,
    measured_rounds: int,
) -> tuple[list[float], int, int]:
    if not values:
        return [], 0, 0
    if len(values) <= warmup_rounds:
        return values, 1, len(values)
    end = min(len(values), warmup_rounds + measured_rounds)
    return values[warmup_rounds:end], warmup_rounds + 1, end


def _avg(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def peak_vram_mb(device: torch.device) -> float:
    if device.type != "cuda" or not torch.cuda.is_available():
        return 0.0
    try:
        return float(torch.cuda.max_memory_allocated(device) / (1024 ** 2))
    except RuntimeError:
        return 0.0


@dataclass
class RunEfficiency:
    device: torch.device
    warmup_rounds: int = 5
    measured_rounds: int = 15
    round_secs: list[float] = field(default_factory=list)
    local_adapt_secs: list[float] = field(default_factory=list)
    run_start_ts: str = ""
    run_end_ts: str = ""
    wall_clock_sec: float = 0.0
    _started_at: float = 0.0

    def start(self) -> None:
        self.run_start_ts = datetime.now().isoformat(timespec="seconds")
        self._started_at = perf_counter()
        if self.device.type == "cuda" and torch.cuda.is_available():
            try:
                torch.cuda.reset_peak_memory_stats(self.device)
            except RuntimeError:
                pass

    def record_round(self, round_sec: float, local_adapt_sec: float = 0.0) -> None:
        self.round_secs.append(max(0.0, float(round_sec)))
        self.local_adapt_secs.append(max(0.0, float(local_adapt_sec)))

    def finish(self) -> None:
        if not self._started_at:
            return
        self.run_end_ts = datetime.now().isoformat(timespec="seconds")
        self.wall_clock_sec = max(0.0, perf_counter() - self._started_at)

    def payload(
        self,
        *,
        seed: int,
        trainable_params: int,
        total_backbone_params: int,
    ) -> dict[str, float | int | str]:
        if not self.run_end_ts:
            self.finish()
        round_sample, start_idx, end_idx = _sample_window(
            self.round_secs,
            warmup_rounds=self.warmup_rounds,
            measured_rounds=self.measured_rounds,
        )
        adapt_sample, _, _ = _sample_window(
            self.local_adapt_secs,
            warmup_rounds=self.warmup_rounds,
            measured_rounds=self.measured_rounds,
        )
        ratio = (
            float(trainable_params / total_backbone_params)
            if total_backbone_params > 0
            else 0.0
        )
        return {
            "seed": int(seed),
            "run_start_ts": self.run_start_ts,
            "run_end_ts": self.run_end_ts,
            "wall_clock_sec": float(self.wall_clock_sec),
            "avg_round_sec": _avg(round_sample),
            "avg_local_adapt_sec": _avg(adapt_sample),
            "peak_vram_mb": peak_vram_mb(self.device),
            "round_count": len(self.round_secs),
            "measured_round_start": start_idx,
            "measured_round_end": end_idx,
            "round_unit": "client_test_window",
            "trainable_params": int(trainable_params),
            "total_backbone_params": int(total_backbone_params),
            "trainable_param_ratio": ratio,
        }
