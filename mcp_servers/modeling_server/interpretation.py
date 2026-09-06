"""
Model interpretation and error analysis for modeling_server.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance


def explain_linear(
    model: "sklearn.base.BaseEstimator",
    feature_names: list[str],
) -> dict:
    """Input: a fitted linear model (Ridge or LogisticRegression) and the
    ordered list of feature names coming out of the pipeline's preprocessing
    step.  Processing: reads model.coef_ directly.  Output:
    {feature_name: coefficient} dict sorted by absolute value descending."""
    # Unwrap Pipeline if needed
    estimator = _unwrap_estimator(model)

    if not hasattr(estimator, "coef_"):
        raise AttributeError(
            f"Model {type(estimator).__name__!r} has no coef_ attribute; "
            "expected Ridge or LogisticRegression."
        )

    coef = np.asarray(estimator.coef_)
    if coef.ndim == 2:
        # Multi-class LogisticRegression: average absolute coefficients
        coef = coef.mean(axis=0)

    names = list(feature_names) if feature_names else [f"f{i}" for i in range(len(coef))]
    paired = dict(zip(names, coef.tolist()))
    return dict(sorted(paired.items(), key=lambda kv: abs(kv[1]), reverse=True))


def explain_tree_ensemble(
    model: "sklearn.base.BaseEstimator",
    X: pd.DataFrame,
    feature_names: list[str],
    method: str = "permutation",
) -> dict:
    """Input: a fitted tree-ensemble model, the feature matrix to explain
    against (typically the train partition), and feature_names.
    Processing: method='permutation' shuffles each feature column and
    measures the resulting score drop.  Output: {feature_name:
    importance_score} dict, sorted descending."""
    estimator = _unwrap_estimator(model)

    if method == "permutation":
        X_arr = X if isinstance(X, np.ndarray) else X.to_numpy()
        result = permutation_importance(
            estimator, X_arr, None,
            n_repeats=10, random_state=42,
        )
        importances = result.importances_mean
    else:
        # Fall back to built-in feature_importances_ if available
        if hasattr(estimator, "feature_importances_"):
            importances = estimator.feature_importances_
        else:
            importances = np.ones(X.shape[1]) / X.shape[1]

    names = list(feature_names) if feature_names else [f"f{i}" for i in range(len(importances))]
    paired = dict(zip(names, importances.tolist()))
    return dict(sorted(paired.items(), key=lambda kv: kv[1], reverse=True))


def segment_error_rates(
    y_true: pd.Series,
    y_pred: pd.Series,
    X: pd.DataFrame,
    segment_by: list[str],
) -> dict:
    """Input: true labels, predicted labels, the feature matrix rows they
    came from, and which columns to break the error rate down by.
    Processing: groups rows by the values in segment_by's columns (binning
    any numeric column into quantiles first), then computes the error rate
    within each group.  Output: {segment_description: error_rate} dict,
    sorted worst-first."""
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    errors = (y_true_arr != y_pred_arr).astype(float)

    result = {}

    for col in segment_by:
        if col not in X.columns:
            continue
        series = X[col].reset_index(drop=True)

        if pd.api.types.is_numeric_dtype(series):
            try:
                binned = pd.qcut(series, q=4, duplicates="drop", labels=False)
                bin_labels = pd.qcut(series, q=4, duplicates="drop")
                series = bin_labels.astype(str)
            except Exception:
                series = series.astype(str)
        else:
            series = series.astype(str)

        for val in series.unique():
            mask = (series == val).to_numpy()
            if mask.sum() == 0:
                continue
            error_rate = float(errors[mask].mean())
            key = f"{col}={val}"
            result[key] = error_rate

    return dict(sorted(result.items(), key=lambda kv: kv[1], reverse=True))


def auto_select_segment_columns(
    X: pd.DataFrame,
    error: pd.Series,
) -> list[str]:
    """Input: the feature matrix and a per-row error-magnitude series.
    Processing: computes each column's correlation with error and ranks
    descending.  Output: top columns by absolute correlation with error,
    used as segment_error_rates' segment_by argument when the caller
    omits it."""
    error_arr = error.reset_index(drop=True)
    correlations = {}

    for col in X.columns:
        series = X[col].reset_index(drop=True)
        if pd.api.types.is_numeric_dtype(series):
            try:
                corr = abs(float(series.corr(error_arr)))
                if pd.notnull(corr):
                    correlations[col] = corr
            except Exception:
                pass
        else:
            # For categorical, use point-biserial proxy via dummy encoding
            try:
                dummies = pd.get_dummies(series, drop_first=True)
                max_corr = max(
                    abs(float(dummies[c].corr(error_arr)))
                    for c in dummies.columns
                    if pd.notnull(dummies[c].corr(error_arr))
                )
                correlations[col] = max_corr
            except Exception:
                pass

    ranked = sorted(correlations, key=lambda c: correlations[c], reverse=True)
    return ranked[:5]  # Return top 5 by default


def _unwrap_estimator(model):
    """Unwrap the final step from a Pipeline if needed."""
    from sklearn.pipeline import Pipeline
    if isinstance(model, Pipeline):
        return model[-1]
    return model