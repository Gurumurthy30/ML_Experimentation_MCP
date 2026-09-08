"""
Cross-validation logic for modeling_server.

clone_pipeline_from_spec() + run_cross_validation() form the core loop that
cross_validate_model and diagnose_fit both invoke.  Nothing here writes to
_cache/models/ -- scoring only.
"""

from __future__ import annotations

import json
from typing import Optional

import numpy as np
import pandas as pd
from sklearn import clone
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from shared.cache import get_cache_dir


def _load_folds(folds_id: str) -> dict:
    """Load the folds JSON written by generate_cv_folds."""
    folds_path = get_cache_dir() / "folds" / f"{folds_id}.json"
    if not folds_path.exists():
        raise FileNotFoundError(f"Folds not found: {folds_id!r}")
    with open(folds_path) as f:
        return json.load(f)


def _load_train(split_id: str) -> pd.DataFrame:
    split_dir = get_cache_dir() / "splits" / split_id
    return pd.read_parquet(split_dir / "train.parquet")


def _load_split_meta(split_id: str) -> dict:
    meta_path = get_cache_dir() / "splits" / split_id / "meta.json"
    if not meta_path.exists():
        return {}
    with open(meta_path) as f:
        return json.load(f)


def _load_pipeline_spec(pipeline_spec_id: str) -> Pipeline:
    """Load the unfitted pipeline object from the specs cache."""
    import joblib
    spec_path = get_cache_dir() / "specs" / f"{pipeline_spec_id}.joblib"
    if not spec_path.exists():
        raise FileNotFoundError(f"Pipeline spec not found: {pipeline_spec_id!r}")
    return joblib.load(spec_path)


def clone_pipeline_from_spec(pipeline_spec_id: str) -> Pipeline:
    """Input: a pipeline_spec_id string.  Processing: loads the unfitted
    Pipeline from the specs cache and calls sklearn.clone() on it.  Output:
    a fresh, still-unfitted copy of the same structure every time it's called
    -- this is what makes correct per-fold refitting possible without any
    state leaking from one fold's fit into the next."""
    spec = _load_pipeline_spec(pipeline_spec_id)
    return clone(spec)


def _score_fold(model, X_val: pd.DataFrame, y_val: pd.Series, task_type: str) -> dict:
    """Compute task-appropriate metrics for one fold."""
    y_pred = model.predict(X_val)

    if task_type == "classification":
        metrics = {
            "accuracy": float(accuracy_score(y_val, y_pred)),
        }
        # F1 (macro for multi-class)
        try:
            metrics["f1"] = float(f1_score(y_val, y_pred, average="macro", zero_division=0))
        except Exception:
            pass
        # ROC-AUC (only if predict_proba available)
        if hasattr(model, "predict_proba"):
            try:
                proba = model.predict_proba(X_val)
                n_classes = proba.shape[1]
                if n_classes == 2:
                    metrics["roc_auc"] = float(roc_auc_score(y_val, proba[:, 1]))
                else:
                    metrics["roc_auc"] = float(
                        roc_auc_score(y_val, proba, multi_class="ovr", average="macro")
                    )
            except Exception:
                pass
    else:
        mse = mean_squared_error(y_val, y_pred)
        metrics = {
            "rmse": float(np.sqrt(mse)),
            "mae": float(mean_absolute_error(y_val, y_pred)),
            "r2": float(r2_score(y_val, y_pred)),
        }
    return metrics


def run_cross_validation(
    folds: list[tuple[list[int], list[int]]],
    pipeline_spec_id: str,
    model_family: str,
    hyperparams: Optional[dict],
    train_df: pd.DataFrame,
    target_column: str,
    task_type: str,
    capture_train_scores: bool = False,
    imbalance_spec: Optional[dict] = None,
) -> dict:
    """Input: the (train_idx, val_idx) list, the unfitted pipeline spec ID,
    and which model family/hyperparams to attach.  Processing: for each fold,
    clone_pipeline_from_spec(), attach get_model(model_family, ...), .fit()
    on train_idx rows only, score on val_idx rows.  Output:
    {metric: {mean, std}, per_fold_scores} -- aggregated across all folds."""
    from .models import get_model  # local import to avoid circularity

    X = train_df.drop(columns=[target_column])
    y = train_df[target_column]

    per_fold: list[dict] = []
    per_fold_train: list[dict] = []

    for train_idx, val_idx in folds:
        X_tr = X.iloc[train_idx].reset_index(drop=True)
        y_tr = y.iloc[train_idx].reset_index(drop=True)
        X_val = X.iloc[val_idx].reset_index(drop=True)
        y_val = y.iloc[val_idx].reset_index(drop=True)

        # Apply imbalance strategy if specified
        if imbalance_spec and imbalance_spec.get("strategy") not in (None, "none"):
            X_tr, y_tr = _apply_imbalance(X_tr, y_tr, imbalance_spec)

        pipe = clone_pipeline_from_spec(pipeline_spec_id)
        estimator = get_model(model_family, task_type, hyperparams)

        # Build final pipeline: preprocessing + model
        final_pipe = Pipeline(list(pipe.steps) + [("model", estimator)])
        final_pipe.fit(X_tr, y_tr)

        val_scores = _score_fold(final_pipe, X_val, y_val, task_type)
        per_fold.append(val_scores)

        if capture_train_scores:
            train_scores = _score_fold(final_pipe, X_tr, y_tr, task_type)
            per_fold_train.append(train_scores)

    # Aggregate
    all_metrics = set(k for fold in per_fold for k in fold)
    aggregated = {}
    for metric in all_metrics:
        vals = [fold[metric] for fold in per_fold if metric in fold]
        aggregated[metric] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
        }

    result = {
        "metrics": aggregated,
        "per_fold_scores": per_fold,
        "n_folds": len(folds),
    }

    if capture_train_scores:
        train_aggregated = {}
        for metric in all_metrics:
            vals = [fold[metric] for fold in per_fold_train if metric in fold]
            if vals:
                train_aggregated[metric] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                }
        result["train_metrics"] = train_aggregated
        result["per_fold_train_scores"] = per_fold_train

    return result


def _apply_imbalance(X, y, imbalance_spec: dict):
    """Apply imbalance handling strategy inside a fold's training data."""
    strategy = imbalance_spec.get("strategy", "class_weight")
    if strategy in ("class_weight", "none"):
        return X, y

    try:
        min_class_count = int(y.value_counts().min())
        if min_class_count <= 1:
            return X, y

        if strategy == "oversample_smote":
            from imblearn.over_sampling import SMOTE
            k_neighbors = max(1, min(5, min_class_count - 1))
            sm = SMOTE(random_state=42, k_neighbors=k_neighbors)
            X_res, y_res = sm.fit_resample(X, y)
            return pd.DataFrame(X_res, columns=X.columns), pd.Series(y_res, name=y.name)
        elif strategy == "oversample_random":
            from imblearn.over_sampling import RandomOverSampler
            ros = RandomOverSampler(random_state=42)
            X_res, y_res = ros.fit_resample(X, y)
            return pd.DataFrame(X_res, columns=X.columns), pd.Series(y_res, name=y.name)
        elif strategy == "undersample_random":
            from imblearn.under_sampling import RandomUnderSampler
            rus = RandomUnderSampler(random_state=42)
            X_res, y_res = rus.fit_resample(X, y)
            return pd.DataFrame(X_res, columns=X.columns), pd.Series(y_res, name=y.name)
    except (ImportError, ValueError, Exception):
        # Fall back to original data if imbalanced-learn is not installed or resampling fails (e.g. non-numeric columns)
        pass

    return X, y




def compute_fold_gap(cv_result: dict) -> dict:
    """Input: a cv_result dict as returned by run_cross_validation with
    capture_train_scores=True, plus each fold's train-set score captured
    alongside it.  Processing: train score minus val score, averaged across
    folds, plus the standard deviation of that gap across folds.  Output:
    {mean_gap: float, gap_std: float}."""
    per_fold_val = cv_result.get("per_fold_scores", [])
    per_fold_train = cv_result.get("per_fold_train_scores", [])

    if not per_fold_val or not per_fold_train:
        return {"mean_gap": 0.0, "gap_std": 0.0, "primary_metric": None}

    # Use the first metric that appears in both
    primary = None
    for metric in per_fold_val[0]:
        if metric in per_fold_train[0]:
            primary = metric
            break

    if primary is None:
        return {"mean_gap": 0.0, "gap_std": 0.0, "primary_metric": None}

    gaps = [
        tr.get(primary, 0.0) - vl.get(primary, 0.0)
        for tr, vl in zip(per_fold_train, per_fold_val)
    ]
    return {
        "mean_gap": float(np.mean(gaps)),
        "gap_std": float(np.std(gaps)),
        "primary_metric": primary,
    }


def verdict_from_gap(gap_stats: dict) -> str:
    """Input: the dict returned by compute_fold_gap.  Processing: fixed
    threshold logic -- large positive mean_gap -> 'overfitting'; both train
    and val scores low -> 'underfitting'; otherwise 'good_fit'.  Output:
    one of those three strings."""
    mean_gap = gap_stats.get("mean_gap", 0.0)
    gap_std = gap_stats.get("gap_std", 0.0)

    if mean_gap > 0.15 or (mean_gap > 0.10 and gap_std > 0.08):
        return "overfitting"
    # We don't have absolute val score here; use gap < 0 as a rough proxy
    if mean_gap < -0.05:
        return "underfitting"
    return "good_fit"