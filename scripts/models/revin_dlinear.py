import torch
import torch.nn as nn

from .dlinear import DLinearModel


class RevIN(nn.Module):
    """Reversible Instance Normalization (from backgrounds/RevIN/RevIN.py)."""

    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        if mode == "norm":
            self._get_statistics(x)
            return self._normalize(x)
        elif mode == "denorm":
            return self._denormalize(x)
        raise NotImplementedError(f"Unknown RevIN mode: {mode}")

    def _get_statistics(self, x: torch.Tensor) -> None:
        # Reduce over all dims except batch and channel
        dim2reduce = tuple(range(1, x.ndim - 1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(
            torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps
        ).detach()

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.mean) / self.stdev
        if self.affine:
            x = x * self.affine_weight + self.affine_bias
        return x

    def _denormalize(self, x: torch.Tensor) -> torch.Tensor:
        if self.affine:
            x = (x - self.affine_bias) / (self.affine_weight + self.eps ** 2)
        x = x * self.stdev + self.mean
        return x


class RevINDLinear(nn.Module):
    """
    DLinear wrapped with Reversible Instance Normalization.

    Forward pass:
        x (Global Scale) → RevIN.norm → DLinear → RevIN.denorm → ŷ (Global Scale)

    TTA target parameters:
        self.dlinear.linear_seasonal  (W_season)
        self.dlinear.linear_trend     (W_trend)

    RevIN affine params are FROZEN during TTA (see tta/adapter.py).
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        channels: int,
        kernel_size: int = 25,
        individual: bool = False,
        revin_affine: bool = True,
        revin_eps: float = 1e-5,
    ):
        super().__init__()
        self.revin = RevIN(channels, eps=revin_eps, affine=revin_affine)
        self.dlinear = DLinearModel(seq_len, pred_len, channels, kernel_size, individual)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, seq_len, C]  in Global Scale space
        x_norm = self.revin(x, "norm")       # per-window z-score
        y_norm = self.dlinear(x_norm)         # [B, pred_len, C]
        y_hat = self.revin(y_norm, "denorm")  # back to Global Scale
        return y_hat

    @property
    def linear_seasonal(self) -> nn.Linear | nn.ModuleList:
        return self.dlinear.linear_seasonal

    @property
    def linear_trend(self) -> nn.Linear | nn.ModuleList:
        return self.dlinear.linear_trend
