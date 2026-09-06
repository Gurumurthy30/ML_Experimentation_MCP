"""
Regularization path search and decision-threshold sweep for modeling_server.
"""

from __future__ import annotations

import json
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score


def alpha_grid_default(model_family: str) -> list[float]:
    """A log-spaced default grid (e.g. 0.001 to 100), used by
    regularization_path_search when the caller doesn't supply alpha_grid."""
    if model_family == "linear":
        return list(np.logspace(-3, 2, num=20))
    # tree_ensemble doesn't use alpha; return an empty list
    return []


def sweep_regularization(
    folds: list[tuple[list[int], list[int]]],
    pipeline_spec_id: str,
    alpha_grid: list[float],
    train_df: pd.DataFrame,
    target_column: str,
    task_type: str,
) -> dict:
    """Calls run_cross_validation once per alpha value in alpha_grid,
    collecting the mean CV score for each.  Returns the best alpha and
    per-alpha scores."""
    from .cv import run_cross_validation

    results = {}
    for alpha in alpha_grid:
        if task_type == "regression":
            hyperparams = {"alpha": float(alpha)}
        else:
            # LogisticRegression uses C = 1/alpha
            hyperparams = {"C": float(1.0 / max(alpha, 1e-8)), "max_iter": 1000}

        cv_result = run_cross_validation(
            folds=folds,
            pipeline_spec_id=pipeline_spec_id,
            model_family="linear",
            hyperparams=hyperparams,
            train_df=train_df,
            target_column=target_column,
            task_type=task_type,
        )
        metrics = cv_result.get("metrics", {})
        # Primary metric: roc_auc for classification, rmse (negated) for regression
        if task_type == "classification":
            primary = metrics.get("roc_auc", metrics.get("accuracy", {}))
        else:
            primary = metrics.get("r2", metrics.get("rmse", {}))
        mean_score = primary.get("mean", 0.0) if isinstance(primary, dict) else 0.0
        results[float(alpha)] = {"mean_cv_score": mean_score, "metrics": metrics}

    if not results:
        return {"best_alpha": None, "cv_score_per_alpha": {}}

    # For regression with rmse, minimize; for everything else, maximize
    if task_type == "regression" and "rmse" in next(iter(results.values()))["metrics"]:
        best_alpha = min(results, key=lambda a: results[a]["mean_cv_score"])
    else:
        best_alpha = max(results, key=lambda a: results[a]["mean_cv_score"])

    return {
        "best_alpha": best_alpha,
        "cv_score_per_alpha": {str(k): v for k, v in results.items()},
    }


def threshold_sweep(
    y_true: pd.Series,
    y_proba: "np.ndarray",
    optimize_for: str,
) -> float:
    """Input: true binary labels, predicted probabilities for the positive
    class, and which metric to optimize for ('f1'/'precision'/'recall').
    Processing: tries a grid of thresholds between 0 and 1, scoring each
    with optimize_for's corresponding sklearn metric.  Output: the threshold
    that scored highest."""
    thresholds = np.linspace(0.01, 0.99, 99)
    best_threshold = 0.5
    best_score = -1.0

    y_true_arr = np.asarray(y_true)
    y_proba_arr = np.asarray(y_proba)

    for t in thresholds:
        y_pred = (y_proba_arr >= t).astype(int)
        try:
            if optimize_for == "f1":
                score = f1_score(y_true_arr, y_pred, zero_division=0)
            elif optimize_for == "precision":
                score = precision_score(y_true_arr, y_pred, zero_division=0)
            elif optimize_for == "recall":
                score = recall_score(y_true_arr, y_pred, zero_division=0)
            else:
                raise ValueError(f"Unknown optimize_for: {optimize_for!r}")
        except Exception:
            score = 0.0

        if score > best_score:
            best_score = score
            best_threshold = float(t)

    return best_threshold