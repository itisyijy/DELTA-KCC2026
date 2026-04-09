from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class ClientData:
    """
    Preprocessed time series for a single FL client (one column).

    values        — [N, 1] float32, Global Scale (StandardScaler applied on train split)
    split_indices — {'train': (start, end), 'val': (start, end), 'test': (start, end)}
                    end is exclusive; train end == val start, val end == test start
    global_mean   — scalar mean of the train split (for sMAPE inverse transform)
    global_std    — scalar std of the train split
    client_id     — string identifier
    """

    client_id: str
    values: np.ndarray          # [N, 1] float32
    split_indices: dict         # {'train': (int, int), 'val': (int, int), 'test': (int, int)}
    global_mean: float
    global_std: float

    def train_values(self) -> np.ndarray:
        s, e = self.split_indices["train"]
        return self.values[s:e]

    def val_values(self) -> np.ndarray:
        s, e = self.split_indices["val"]
        return self.values[s:e]

    def test_values(self) -> np.ndarray:
        s, e = self.split_indices["test"]
        return self.values[s:e]


class ClientDataset(Dataset):
    """
    Sliding-window dataset for a single client.

    Each sample: (x [seq_len, 1], y [pred_len, 1]) in Global Scale space.
    Windows are computed with stride=1 within the given split.
    """

    def __init__(
        self,
        client: ClientData,
        split: str,
        seq_len: int,
        pred_len: int,
    ):
        assert split in ("train", "val", "test"), f"Unknown split: {split}"
        s, e = client.split_indices[split]
        self.data = client.values[s:e]          # [N_split, 1] float32
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n_windows = max(0, len(self.data) - seq_len - pred_len + 1)

    def __len__(self) -> int:
        return self.n_windows

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.data[idx : idx + self.seq_len]                              # [seq_len, 1]
        y = self.data[idx + self.seq_len : idx + self.seq_len + self.pred_len]  # [pred_len, 1]
        return torch.from_numpy(x), torch.from_numpy(y)


class CentralizedDataset(Dataset):
    """
    Concatenation of all clients' sliding-window datasets for centralized training.
    """

    def __init__(
        self,
        clients: list[ClientData],
        split: str,
        seq_len: int,
        pred_len: int,
    ):
        self.datasets = [ClientDataset(c, split, seq_len, pred_len) for c in clients]
        self.cumulative_sizes = []
        total = 0
        for ds in self.datasets:
            total += len(ds)
            self.cumulative_sizes.append(total)

    def __len__(self) -> int:
        return self.cumulative_sizes[-1] if self.cumulative_sizes else 0

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Binary search for the right sub-dataset
        lo, hi = 0, len(self.datasets) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if self.cumulative_sizes[mid] <= idx:
                lo = mid + 1
            else:
                hi = mid
        offset = self.cumulative_sizes[lo - 1] if lo > 0 else 0
        return self.datasets[lo][idx - offset]
