"""
Harness state management -- placeholder.

This module will track handle IDs across MCP tool calls once the harness
agent is designed (after all 43 tools are independently tested).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class RunState:
    """Holds all handle IDs and metadata for a single ML experiment run."""

    dataset_id: Optional[str] = None
    dataset_path: Optional[str] = None
    split_id: Optional[str] = None
    target_column: Optional[str] = None
    task_type: Optional[str] = None
    resolved_strategy: Optional[str] = None
    folds_id: Optional[str] = None
    n_folds: Optional[int] = None
    imputer_spec: Optional[dict] = None
    encoder_spec: Optional[dict] = None
    scaler_spec: Optional[dict] = None
    feature_selection_spec: Optional[dict] = None
    outlier_spec: Optional[dict] = None
    derived_feature_spec: Optional[dict] = None
    pipeline_spec_id: Optional[str] = None
    model_family: Optional[str] = None
    hyperparams: Optional[dict] = None
    cv_result: Optional[dict] = None
    imbalance_spec: Optional[dict] = None
    model_id: Optional[str] = None
    calibrated_model_id: Optional[str] = None
    experiment_id: Optional[str] = None
    run_id: Optional[str] = None
    extra: dict = field(default_factory=dict)

    def update(self, **kwargs: Any) -> "RunState":
        import copy
        new_state = copy.copy(self)
        for key, value in kwargs.items():
            if hasattr(new_state, key):
                setattr(new_state, key, value)
            else:
                new_state.extra[key] = value
        return new_state
