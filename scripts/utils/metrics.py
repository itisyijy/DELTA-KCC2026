"""
Evaluation metrics for the DLinear FED-TTA framework.

MSE / MAE  — computed in Global Scale space (after RevIN denorm, before global scaler inversion)
sMAPE      — computed in original physical units (after full inverse transform)
"""
from __future__ import annotations

import random

import numpy as np

from scripts.data.dataset import ClientData


def mse(pred: np.ndarray, target: np.ndarray) -> float:
    """Mean Squared Error in Global Scale space."""
    return float(np.mean((pred - target) ** 2))


def mae(pred: np.ndarray, target: np.ndarray) -> float:
    """Mean Absolute Error in Global Scale space."""
    return float(np.mean(np.abs(pred - target)))


def inverse_global_scale(
    values: np.ndarray,
    mean: float,
    std: float,
) -> np.ndarray:
    """Invert the Global StandardScaler: values * std + mean."""
    return values * std + mean


def smape(
    pred: np.ndarray,
    target: np.ndarray,
    eps: float = 1e-8,
) -> float:
    """
    Symmetric Mean Absolute Percentage Error (%).
    Both pred and target must be in ORIGINAL physical units.
    """
    numerator = np.abs(pred - target)
    denominator = (np.abs(pred) + np.abs(target)) / 2.0 + eps
    return float(np.mean(numerator / denominator) * 100.0)


def wasserstein_noniid(
    clients: list[ClientData],
    n_pairs: int = 50,
    seed: int = 0,
) -> float:
    """
    Quantify Non-IID degree via Wasserstein Distance.

    Samples n_pairs of client pairs and computes the 1D Wasserstein Distance
    between their training split value distributions.  Returns the mean distance.
    """
    try:
        from scipy.stats import wasserstein_distance
    except ImportError:
        raise ImportError("scipy is required for wasserstein_noniid. Install with: pip install scipy")

    if len(clients) < 2:
        return 0.0

    rng = random.Random(seed)
    pairs = [
        rng.sample(range(len(clients)), 2)
        for _ in range(min(n_pairs, len(clients) * (len(clients) - 1) // 2))
    ]

    distances = []
    for i, j in pairs:
        ci = clients[i].train_values().flatten()
        cj = clients[j].train_values().flatten()
        distances.append(wasserstein_distance(ci, cj))

    return float(np.mean(distances)) if distances else 0.0


def count_params(model) -> int:
    """Count total trainable parameters (communication cost proxy)."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
