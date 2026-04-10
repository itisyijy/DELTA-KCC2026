from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from scripts.models.revin_dlinear import RevINDLinear


@dataclass
class ClientDelta:
    client_id: str
    delta_trend: torch.Tensor
    delta_season: torch.Tensor
    n_valid_steps: int


def _stack_weights(layer_group: nn.Module | nn.ModuleList) -> torch.Tensor:
    if isinstance(layer_group, nn.ModuleList):
        return torch.stack([layer.weight for layer in layer_group])
    return layer_group.weight


def compute_delta(
    model: RevINDLinear,
    anchor_trend_w: torch.Tensor,
    anchor_season_w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        (_stack_weights(model.linear_trend) - anchor_trend_w).detach(),
        (_stack_weights(model.linear_seasonal) - anchor_season_w).detach(),
    )


def clip_delta(delta: torch.Tensor, max_norm: float) -> torch.Tensor:
    if max_norm <= 0:
        return delta
    norm = delta.norm()
    if norm > max_norm:
        delta = delta * (max_norm / (norm + 1e-8))
    return delta


def get_model_weights(model: RevINDLinear) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        _stack_weights(model.linear_trend).detach().clone(),
        _stack_weights(model.linear_seasonal).detach().clone(),
    )


def aggregate_deltas(
    client_deltas: list[ClientDelta],
    decay_factor: float = 0.9,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    valid = [delta for delta in client_deltas if delta.n_valid_steps > 0]
    if not valid:
        return None
    total_weight = sum(delta.n_valid_steps for delta in valid)
    agg_trend = torch.zeros_like(valid[0].delta_trend)
    agg_season = torch.zeros_like(valid[0].delta_season)
    for delta in valid:
        weight = delta.n_valid_steps / total_weight
        agg_trend += delta.delta_trend.float() * weight
        agg_season += delta.delta_season.float() * weight
    return agg_trend * decay_factor, agg_season * decay_factor


def apply_server_feedback(
    global_model: RevINDLinear,
    agg_delta_trend: torch.Tensor,
    agg_delta_season: torch.Tensor,
) -> None:
    linear_trend = global_model.linear_trend
    linear_season = global_model.linear_seasonal
    if isinstance(linear_trend, nn.ModuleList):
        for idx, layer in enumerate(linear_trend):
            layer.weight.data += agg_delta_trend[idx]
        for idx, layer in enumerate(linear_season):
            layer.weight.data += agg_delta_season[idx]
        return
    linear_trend.weight.data += agg_delta_trend
    linear_season.weight.data += agg_delta_season
