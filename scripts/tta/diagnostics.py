from __future__ import annotations

from dataclasses import dataclass

from scripts.tta.engine import TTAStepResultV2


@dataclass
class ClientTTADiagnostics:
    total_steps: int = 0
    adapted_steps: int = 0
    accept_gate_skips: int = 0
    hard_gate_skips: int = 0
    drift_skips: int = 0
    rollback_skips: int = 0
    reset_steps: int = 0
    l_hind_sum: float = 0.0
    l_cons_sum: float = 0.0
    l_anchor_sum: float = 0.0
    boost_sum: float = 0.0
    gamma_l1_sum: float = 0.0
    delta_l1_sum: float = 0.0
    final_gamma_l1: float = 0.0
    final_delta_l1: float = 0.0

    def update(
        self,
        result: TTAStepResultV2 | None,
        *,
        gamma_l1: float,
        delta_l1: float,
    ) -> None:
        if result is None:
            return
        self.total_steps += 1
        self.gamma_l1_sum += gamma_l1
        self.delta_l1_sum += delta_l1
        self.final_gamma_l1 = gamma_l1
        self.final_delta_l1 = delta_l1
        if result.skip_reason == "accept_gate":
            self.accept_gate_skips += 1
            return
        if result.skip_reason == "hard_gate":
            self.hard_gate_skips += 1
            return
        if result.skip_reason == "drift_gate":
            self.drift_skips += 1
            return
        if result.skip_reason == "rollback":
            self.rollback_skips += 1
            return
        self.adapted_steps += 1
        self.reset_steps += int(result.reset_applied)
        self.l_hind_sum += result.l_hind
        self.l_cons_sum += result.l_cons
        self.l_anchor_sum += result.l_anchor
        self.boost_sum += result.boost

    def as_dict(self) -> dict[str, float]:
        denom = max(1, self.total_steps)
        adapted = max(1, self.adapted_steps)
        return {
            "total_steps": self.total_steps,
            "adapted_steps": self.adapted_steps,
            "accept_gate_skips": self.accept_gate_skips,
            "hard_gate_skips": self.hard_gate_skips,
            "drift_skips": self.drift_skips,
            "rollback_skips": self.rollback_skips,
            "reset_steps": self.reset_steps,
            "adapt_rate": self.adapted_steps / denom,
            "accept_gate_skip_rate": self.accept_gate_skips / denom,
            "hard_gate_skip_rate": self.hard_gate_skips / denom,
            "drift_skip_rate": self.drift_skips / denom,
            "rollback_skip_rate": self.rollback_skips / denom,
            "reset_rate_given_adapt": self.reset_steps / adapted,
            "mean_l_hind": self.l_hind_sum / adapted,
            "mean_l_cons": self.l_cons_sum / adapted,
            "mean_l_anchor": self.l_anchor_sum / adapted,
            "mean_boost": self.boost_sum / adapted,
            "mean_gamma_l1": self.gamma_l1_sum / denom,
            "mean_delta_l1": self.delta_l1_sum / denom,
            "final_gamma_l1": self.final_gamma_l1,
            "final_delta_l1": self.final_delta_l1,
        }


def summarize_diagnostics(per_client: dict[str, dict[str, float]]) -> dict[str, float]:
    if not per_client:
        return {}
    keys = tuple(next(iter(per_client.values())).keys())
    summary: dict[str, float] = {}
    for key in keys:
        values = [float(stats[key]) for stats in per_client.values()]
        summary[key] = sum(values) / len(values)
    return summary
