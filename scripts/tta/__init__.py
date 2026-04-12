from .loss import HindcastLoss, DynamicRegularizer, TTALoss, HybridTTALoss
from .adapter import prepare_tta_model, AffineAdapter, prepare_frozen_backbone
from .engine import (
    ReconTracker, RollbackGuard,
    build_hindcast_inputs, run_tta_step, evaluate_client,
    TTAStepResult,
    TTAStepResultV2, run_tta_step_affine,
)
from .delta import ClientDelta, aggregate_deltas, apply_server_feedback, clip_delta, compute_delta
from .loop import run_fed_tta_loop, run_local_tta_loop

__all__ = [
    # Legacy (Weight-Update TTA)
    "HindcastLoss", "DynamicRegularizer", "TTALoss",
    "prepare_tta_model",
    "ReconTracker", "RollbackGuard",
    "build_hindcast_inputs", "run_tta_step", "evaluate_client", "TTAStepResult",
    "ClientDelta", "compute_delta", "clip_delta",
    "aggregate_deltas", "apply_server_feedback", "run_fed_tta_loop",
    # New (Affine-Adapter TTA)
    "HybridTTALoss",
    "AffineAdapter", "prepare_frozen_backbone",
    "TTAStepResultV2", "run_tta_step_affine",
    "run_local_tta_loop",
]
