"""
Model registry and model factory for modeling_server.

MODEL_REGISTRY is deliberately small and fixed -- model choice is not a
search dimension in this project.  hyperparams passed to
cross_validate_model overrides defaults, never replaces the family.
"""

from __future__ import annotations

import json
from typing import Optional

from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge

from shared.cache import compute_handle_id


MODEL_REGISTRY: dict[str, dict[str, type]] = {
    "linear": {
        "regression": Ridge,
        "classification": LogisticRegression,
    },
    "tree_ensemble": {
        "regression": HistGradientBoostingRegressor,
        "classification": HistGradientBoostingClassifier,
    },
}
"""Deliberately small and fixed -- model choice is not a search dimension
in this project.  hyperparams passed to cross_validate_model overrides
these defaults, never replaces the family."""

_DEFAULT_HYPERPARAMS: dict[str, dict[str, dict]] = {
    "linear": {
        "regression": {"alpha": 1.0},
        "classification": {"C": 1.0, "max_iter": 1000, "solver": "lbfgs"},
    },
    "tree_ensemble": {
        "regression": {"max_iter": 200, "learning_rate": 0.1, "random_state": 42},
        "classification": {"max_iter": 200, "learning_rate": 0.1, "random_state": 42},
    },
}


def get_model(
    model_family: str,
    task_type: str,
    hyperparams: Optional[dict] = None,
) -> "sklearn.base.BaseEstimator":
    """Input: model_family in {'linear', 'tree_ensemble'}, task_type in
    {'regression', 'classification'}, and an optional dict of hyperparameter
    overrides.  Processing: looks up MODEL_REGISTRY[model_family][task_type]
    and instantiates it, merging hyperparams over the class's defaults rather
    than replacing them.  Output: an unfitted scikit-learn estimator instance,
    attached as the final step of a cloned pipeline inside cv.py and
    finalize.py."""
    if model_family not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model_family: {model_family!r}. Choose from {list(MODEL_REGISTRY)}")
    family = MODEL_REGISTRY[model_family]
    if task_type not in family:
        raise ValueError(f"Unknown task_type: {task_type!r}. Choose from {list(family)}")

    cls = family[task_type]
    defaults = dict(_DEFAULT_HYPERPARAMS.get(model_family, {}).get(task_type, {}))
    if hyperparams:
        defaults.update(hyperparams)

    # Filter to only kwargs the class actually accepts
    import inspect
    valid_params = set(inspect.signature(cls.__init__).parameters) - {"self"}
    filtered = {k: v for k, v in defaults.items() if k in valid_params}

    return cls(**filtered)


def compute_model_id(
    folds_id: str,
    pipeline_spec_id: str,
    model_family: str,
    hyperparams: Optional[dict],
) -> str:
    """sha1 of the sorted, JSON-serialized tuple of all four inputs.
    Deterministic -- finalize_model checks this before fitting, so calling
    it twice with identical inputs returns the existing model_id instead
    of refitting."""
    return compute_handle_id(
        folds_id,
        pipeline_spec_id,
        model_family,
        json.dumps(hyperparams or {}, sort_keys=True),
    )