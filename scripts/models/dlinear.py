import torch
import torch.nn as nn


class moving_avg(nn.Module):
    """Moving average block to highlight the trend of time series."""

    def __init__(self, kernel_size: int, stride: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C]
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """Series decomposition block: returns (seasonal, trend)."""

    def __init__(self, kernel_size: int):
        super().__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean  # (seasonal, trend)


class DLinearModel(nn.Module):
    """
    Decomposition-Linear (DLinear).

    Attributes exposed for TTA:
      self.linear_seasonal  — nn.Linear or nn.ModuleList (individual=True)
      self.linear_trend     — nn.Linear or nn.ModuleList (individual=True)
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        channels: int,
        kernel_size: int = 25,
        individual: bool = False,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = channels
        self.individual = individual

        self.decomposition = series_decomp(kernel_size)

        if individual:
            self.linear_seasonal = nn.ModuleList(
                [nn.Linear(seq_len, pred_len) for _ in range(channels)]
            )
            self.linear_trend = nn.ModuleList(
                [nn.Linear(seq_len, pred_len) for _ in range(channels)]
            )
        else:
            self.linear_seasonal = nn.Linear(seq_len, pred_len)
            self.linear_trend = nn.Linear(seq_len, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, seq_len, C]
        seasonal_init, trend_init = self.decomposition(x)
        # permute to [B, C, seq_len] for linear layers
        seasonal_init = seasonal_init.permute(0, 2, 1)
        trend_init = trend_init.permute(0, 2, 1)

        if self.individual:
            seasonal_out = torch.zeros(
                seasonal_init.size(0), seasonal_init.size(1), self.pred_len,
                dtype=seasonal_init.dtype, device=seasonal_init.device,
            )
            trend_out = torch.zeros_like(seasonal_out)
            for i in range(self.channels):
                seasonal_out[:, i, :] = self.linear_seasonal[i](seasonal_init[:, i, :])
                trend_out[:, i, :] = self.linear_trend[i](trend_init[:, i, :])
        else:
            seasonal_out = self.linear_seasonal(seasonal_init)
            trend_out = self.linear_trend(trend_init)

        out = seasonal_out + trend_out
        return out.permute(0, 2, 1)  # [B, pred_len, C]
