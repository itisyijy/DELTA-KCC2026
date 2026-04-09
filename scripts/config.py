"""
Configuration dataclasses for the DLinear FED-TTA framework.

All configs are loadable from YAML via load_config(yaml_path).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class ModelConfig:
    seq_len: int = 96
    pred_len: int = 96
    kernel_size: int = 25
    individual: bool = False
    revin_affine: bool = True


@dataclass
class TTAConfig:
    # k = int(k_ratio * pred_len); k_ratio ∈ {0.25, 0.5, 1.0} guarantees k ≤ pred_len
    k_ratio: float = 0.25
    alpha: float = 1.0        # L_TTA = L_recon + alpha * L_reg
    lambda0: float = 1.0      # base penalty coefficient
    gamma: float = 1.0        # exponential decay sensitivity
    lr: float = 1e-3
    grad_clip: float = 1.0
    rollback_threshold: float = 3.0
    rollback_window: int = 20


@dataclass
class FeedbackConfig:
    delta_clip_norm: float = 1.0
    decay_factor: float = 0.9


@dataclass
class ExperimentConfig:
    # --- Required ---
    baseline: str             # centralized | fed | dlinear_tta | fed_tta | fed_tta_loop
    dataset: str              # electricity | solar | murata
    data_path: str            # CSV file path or parquet manifest.json path

    # --- Data ---
    timestamp_col: str = "date"   # CSV column to drop ('date' or 'LocalTime')
    data_format: str = "csv"      # 'csv' or 'parquet'
    parquet_clients_dir: str = "" # only for parquet format: path to clients/ directory

    # --- Model ---
    model: ModelConfig = field(default_factory=ModelConfig)

    # --- Training ---
    batch_size: int = 64
    epochs: int = 50          # centralized training epochs
    lr: float = 1e-3
    patience: int = 7         # early stopping patience

    # --- FL ---
    local_epochs: int = 3     # sync period (local epochs per FL round)
    global_rounds: int = 50

    # --- TTA / FED-TTA ---
    checkpoint_path: str = ""     # path to pre-trained model checkpoint
    tta: TTAConfig = field(default_factory=TTAConfig)
    feedback: FeedbackConfig = field(default_factory=FeedbackConfig)

    # --- Misc ---
    max_clients: int | None = None
    output_dir: str = "runs"
    checkpoint_dir: str = "checkpoints"
    device: str = "cuda:0"
    seed: int = 0

    def k(self) -> int:
        """Hindcast length derived from k_ratio and pred_len."""
        return max(1, int(self.tta.k_ratio * self.model.pred_len))


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base."""
    result = base.copy()
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def load_config(yaml_path: str | Path) -> ExperimentConfig:
    """
    Load an ExperimentConfig from a YAML file.

    Top-level keys map directly to ExperimentConfig fields.
    Nested dicts 'model', 'tta', 'feedback' map to their respective dataclasses.
    """
    with open(yaml_path) as f:
        raw = yaml.safe_load(f)

    model_cfg = ModelConfig(**{
        k: v for k, v in raw.pop("model", {}).items()
        if k in ModelConfig.__dataclass_fields__
    })
    tta_cfg = TTAConfig(**{
        k: v for k, v in raw.pop("tta", {}).items()
        if k in TTAConfig.__dataclass_fields__
    })
    feedback_cfg = FeedbackConfig(**{
        k: v for k, v in raw.pop("feedback", {}).items()
        if k in FeedbackConfig.__dataclass_fields__
    })

    valid_keys = ExperimentConfig.__dataclass_fields__.keys()
    filtered = {k: v for k, v in raw.items() if k in valid_keys}

    return ExperimentConfig(
        model=model_cfg,
        tta=tta_cfg,
        feedback=feedback_cfg,
        **filtered,
    )
