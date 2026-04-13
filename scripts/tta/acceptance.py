from __future__ import annotations

import torch

from scripts.tta.adapter import AffineAdapter


def capture_adapter_state(adapter: AffineAdapter) -> tuple[torch.Tensor, torch.Tensor]:
    return adapter.gamma.detach().clone(), adapter.delta.detach().clone()


@torch.no_grad()
def restore_adapter_state(
    adapter: AffineAdapter,
    state: tuple[torch.Tensor, torch.Tensor],
) -> None:
    gamma, delta = state
    adapter.gamma.copy_(gamma)
    adapter.delta.copy_(delta)


def accepts_update(
    l_hind_pre: float,
    l_hind_post: float,
    acceptance_margin: float,
) -> bool:
    if acceptance_margin < 0.0:
        return True
    target = l_hind_pre * (1.0 - acceptance_margin)
    return l_hind_post <= target
