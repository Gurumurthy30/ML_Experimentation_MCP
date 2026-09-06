"""
Modeling MCP tools — exposed as MCP tools by server.py.

model_family is a small, fixed enum ('linear' / 'tree_ensemble') -- model
choice is deliberately not a search dimension in this project.
"""

from __future__ import annotations

import json
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

try:
    from ..shared.cache import compute_handle_id, get_cache_dir, load_object, object_exists, save_object
except ImportError:
    from shared.cache import compute_handle_id, get_cache_dir, load_object, object_exists, save_object


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_folds(folds_id: str) -> dict:
    folds_path = get_cache_dir() / "folds" / f"{folds_id}.json"
    if not folds_path.exists():
        raise FileNotFoundError(f"Folds not found for folds_id={folds_id!r}")
    with open(folds_path) as f:
        return json.load(f)


def _load_split_meta(split_id: str) -> dict:
    meta_path = get_cache_dir() / "splits" / split_id / "meta.json"
    if not meta_path.exists():
        return {}
    with open(meta_path) as f:
        return json.load(f)


def _load_train(split_id: str) -> pd.DataFrame:
    return pd.read_parquet(get_cache_dir() / "splits" / split_id / "train.parquet")


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

def establish_baseline(split_id: str, task_type: str) -> dict:
    """Fit a dummy baseline model to determine performance lower bound.

    PURPOSE & WHEN TO USE:
        Call this tool before training candidate models to establish the baseline performance floor
        (majority class for classification, mean prediction for regression).

    INPUT PARAMETERS:
        - `split_id` (str): Active split handle ID.
        - `task_type` (str): Problem type ('classification' or 'regression').

    RETURNS:
        - Dict with `baseline_metrics` (accuracy/f1/auc or rmse/mae/r2) and `strategy`.

    AI AGENT GUIDELINES:
        - Use `baseline_metrics` as the benchmark that all subsequent models must beat.
    """
    meta = _load_split_meta(split_id)
    target_column = meta.get("target_column", "")

    train_df = _load_train(split_id)
    X_train = train_df.drop(columns=[target_column], errors="ignore")
    y_train = train_df[target_column]

    if task_type == "classification":
        dummy = DummyClassifier(strategy="most_frequent", random_state=42)
        dummy.fit(X_train, y_train)
        y_pred = dummy.predict(X_train)
        metrics = {
            "accuracy": float(accuracy_score(y_train, y_pred)),
            "f1_macro": float(f1_score(y_train, y_pred, average="macro", zero_division=0)),
        }
        try:
            proba = dummy.predict_proba(X_train)
            if proba.shape[1] == 2:
                metrics["roc_auc"] = float(roc_auc_score(y_train, proba[:, 1]))
        except Exception:
            pass
    else:
        dummy = DummyRegressor(strategy="mean")
        dummy.fit(X_train, y_train)
        y_pred = dummy.predict(X_train)
        mse = mean_squared_error(y_train, y_pred)
        metrics = {
            "rmse": float(np.sqrt(mse)),
            "mae": float(mean_absolute_error(y_train, y_pred)),
            "r2": float(r2_score(y_train, y_pred)),
        }

    return {"baseline_metrics": metrics, "strategy": "most_frequent" if task_type == "classification" else "mean"}


def cross_validate_model(
    folds_id: str,
    pipeline_spec_id: str,
    model_family: str,
    hyperparams: Optional[dict] = None,
    imbalance_spec: Optional[dict] = None,
) -> dict:
    """Run k-fold cross-validation for a candidate model pipeline configuration.

    PURPOSE & WHEN TO USE:
        Call this tool as the primary scoring mechanism for model candidates.
        Performs per-fold fitting of the preprocessor pipeline and model on training rows only, scoring on validation rows.

    INPUT PARAMETERS:
        - `folds_id` (str): CV folds handle ID returned by `generate_cv_folds`.
        - `pipeline_spec_id` (str): Preprocessing pipeline handle ID returned by `build_preprocessing_pipeline`.
        - `model_family` (str): Model family choice ('linear' or 'tree_ensemble').
        - `hyperparams` (dict, optional): Model hyperparameters dict (e.g. {'C': 1.0} or {'learning_rate': 0.1}).
        - `imbalance_spec` (dict, optional): Class imbalance strategy recipe from `handle_class_imbalance`.

    RETURNS:
        - Dict with `metrics` (mean and std for metrics) and `per_fold_scores`.

    AI AGENT GUIDELINES:
        - Next, call `diagnose_fit` to check for overfitting or underfitting.
    """
    from .cv import run_cross_validation

    folds_data = _load_folds(folds_id)
    split_id = folds_data["split_id"]
    meta = _load_split_meta(split_id)
    target_column = meta.get("target_column", "")
    task_type = meta.get("task_type", "classification")

    train_df = _load_train(split_id)
    folds = [(f[0], f[1]) for f in folds_data["folds"]]

    result = run_cross_validation(
        folds=folds,
        pipeline_spec_id=pipeline_spec_id,
        model_family=model_family,
        hyperparams=hyperparams,
        train_df=train_df,
        target_column=target_column,
        task_type=task_type,
        capture_train_scores=False,
        imbalance_spec=imbalance_spec,
    )
    return result


def diagnose_fit(
    folds_id: str,
    pipeline_spec_id: str,
    model_family: str,
    cv_result: dict,
) -> dict:
    """Diagnose overfitting or underfitting by evaluating train-vs-validation metric gaps across folds.

    PURPOSE & WHEN TO USE:
        Call this tool after `cross_validate_model` to compare training set scores against validation scores.

    INPUT PARAMETERS:
        - `folds_id` (str): CV folds handle ID.
        - `pipeline_spec_id` (str): Pipeline spec handle ID.
        - `model_family` (str): Model family ('linear' or 'tree_ensemble').
        - `cv_result` (dict): Dict output returned by `cross_validate_model`.

    RETURNS:
        - Dict with `train_val_gap`, `gap_std_across_folds`, `verdict` ('good_fit', 'overfitting', 'underfitting'), `train_metrics`, `val_metrics`.

    AI AGENT GUIDELINES:
        - If verdict is 'overfitting' for linear models, run `regularization_path_search`.
    """
    from .cv import run_cross_validation, compute_fold_gap, verdict_from_gap

    folds_data = _load_folds(folds_id)
    split_id = folds_data["split_id"]
    meta = _load_split_meta(split_id)
    target_column = meta.get("target_column", "")
    task_type = meta.get("task_type", "classification")

    train_df = _load_train(split_id)
    folds = [(f[0], f[1]) for f in folds_data["folds"]]

    full_result = run_cross_validation(
        folds=folds,
        pipeline_spec_id=pipeline_spec_id,
        model_family=model_family,
        hyperparams=None,
        train_df=train_df,
        target_column=target_column,
        task_type=task_type,
        capture_train_scores=True,
    )

    gap_stats = compute_fold_gap(full_result)
    verdict = verdict_from_gap(gap_stats)

    return {
        "train_val_gap": gap_stats["mean_gap"],
        "gap_std_across_folds": gap_stats["gap_std"],
        "primary_metric": gap_stats["primary_metric"],
        "verdict": verdict,
        "train_metrics": full_result.get("train_metrics", {}),
        "val_metrics": full_result.get("metrics", {}),
    }


def regularization_path_search(
    folds_id: str,
    pipeline_spec_id: str,
    model_family: str = "linear",
    alpha_grid: Optional[list[float]] = None,
) -> dict:
    """Search optimal L2 regularization penalty alpha for linear models over cross-validation folds.

    PURPOSE & WHEN TO USE:
        Call this tool when `diagnose_fit` flags overfitting on a linear model to tune alpha.

    INPUT PARAMETERS:
        - `folds_id` (str): CV folds handle ID.
        - `pipeline_spec_id` (str): Pipeline spec handle ID.
        - `model_family` (str, default='linear'): Model family.
        - `alpha_grid` (list[float], optional): Grid of regularization penalty values to search.

    RETURNS:
        - Dict with `best_alpha`, `cv_score_per_alpha`, and best metrics.
    """
    from .tuning import alpha_grid_default, sweep_regularization
    from .cv import run_cross_validation

    if alpha_grid is None:
        alpha_grid = alpha_grid_default(model_family)

    if not alpha_grid:
        return {"best_alpha": None, "cv_score_per_alpha": {}, "message": "No alpha grid for tree_ensemble"}

    folds_data = _load_folds(folds_id)
    split_id = folds_data["split_id"]
    meta = _load_split_meta(split_id)
    target_column = meta.get("target_column", "")
    task_type = meta.get("task_type", "classification")

    train_df = _load_train(split_id)
    folds = [(f[0], f[1]) for f in folds_data["folds"]]

    return sweep_regularization(
        folds=folds,
        pipeline_spec_id=pipeline_spec_id,
        alpha_grid=alpha_grid,
        train_df=train_df,
        target_column=target_column,
        task_type=task_type,
    )


def handle_class_imbalance(
    split_id: str,
    target_column: str,
    strategy: str = "auto",
    severity_threshold: Optional[float] = None,
) -> dict:
    """Configure class imbalance handling techniques (SMOTE, class_weight, or undersampling).

    PURPOSE & WHEN TO USE:
        Call this tool for imbalanced classification tasks. Resampling is applied strictly inside each CV fold's training partition.

    INPUT PARAMETERS:
        - `split_id` (str): Active split handle ID.
        - `target_column` (str): Target column name.
        - `strategy` (str, default='auto'): Resampling strategy ('auto', 'smote', 'class_weight', 'random_undersample', 'none').
        - `severity_threshold` (float, optional): Imbalance ratio threshold (minority/majority) to trigger action.

    RETURNS:
        - Dict containing `imbalance_spec`, `severity_ratio`, `resolved_strategy`, `action_taken`.
    """
    from .imbalance import detect_severity_ratio, resolve_auto_strategy, build_imbalance_spec

    train_df = _load_train(split_id)
    y = train_df[target_column]
    severity_ratio = detect_severity_ratio(y)
    threshold = severity_threshold or 0.3

    if strategy == "auto":
        resolved = resolve_auto_strategy(severity_ratio, len(train_df))
    else:
        resolved = strategy

    spec = build_imbalance_spec(resolved, {})
    return {
        "imbalance_spec": spec,
        "severity_ratio": round(severity_ratio, 4),
        "resolved_strategy": resolved,
        "action_taken": resolved != "none",
    }


def calibrate_probabilities(model_id: str, method: str = "platt") -> dict:
    """Legacy stub for probability calibration (requires split_id context).

    AI AGENT GUIDELINES:
        - Call `calibrate_probabilities_with_split` instead to calibrate probabilities using a validation slice.
    """
    raise NotImplementedError(
        "calibrate_probabilities requires split_id. Call calibrate_probabilities_with_split(model_id, split_id, method)."
    )


def calibrate_probabilities_with_split(
    model_id: str,
    split_id: str,
    method: str = "platt",
) -> dict:
    """Calibrate classification probability outputs using Platt scaling or Isotonic regression.

    PURPOSE & WHEN TO USE:
        Call this tool on a finalized classification model to produce calibrated prediction probabilities.

    INPUT PARAMETERS:
        - `model_id` (str): Finalized model handle ID returned by `finalize_model`.
        - `split_id` (str): Active split handle ID.
        - `method` (str, default='platt'): Calibration method ('platt' or 'isotonic').

    RETURNS:
        - Dict with `calibrated_model_id` and `method`.
    """
    from .finalize import load_model, save_model
    from .calibration import platt_calibration, isotonic_calibration
    from .models import compute_model_id

    meta = _load_split_meta(split_id)
    target_column = meta.get("target_column", "")
    train_df = _load_train(split_id)

    cutoff = int(len(train_df) * 0.8)
    cal_df = train_df.iloc[cutoff:]
    X_cal = cal_df.drop(columns=[target_column], errors="ignore")
    y_cal = cal_df[target_column]

    model = load_model(model_id)
    if method == "isotonic":
        calibrated = isotonic_calibration(model, X_cal, y_cal)
    else:
        calibrated = platt_calibration(model, X_cal, y_cal)

    cal_model_id = compute_handle_id(model_id, method)
    save_model(calibrated, cal_model_id)
    return {"calibrated_model_id": cal_model_id, "method": method}


def tune_decision_threshold(
    model_id: str,
    split_id: str,
    optimize_for: str = "f1",
) -> dict:
    """Sweep classification probability decision thresholds to optimize F1, Precision, or Recall.

    PURPOSE & WHEN TO USE:
        Call this tool post-finalization on binary classification models to adjust the decision boundary from 0.5 to an optimal threshold.

    INPUT PARAMETERS:
        - `model_id` (str): Finalized model handle ID.
        - `split_id` (str): Active split handle ID.
        - `optimize_for` (str, default='f1'): Metric to maximize ('f1', 'precision', 'recall').

    RETURNS:
        - Dict with `optimal_threshold`, `metric_at_threshold`, `optimize_for`.
    """
    from .finalize import load_model
    from .tuning import threshold_sweep

    meta = _load_split_meta(split_id)
    target_column = meta.get("target_column", "")

    train_df = _load_train(split_id)
    X_train = train_df.drop(columns=[target_column], errors="ignore")
    y_train = train_df[target_column]

    model = load_model(model_id)
    if not hasattr(model, "predict_proba"):
        raise ValueError("tune_decision_threshold requires a model with predict_proba (classification only)")

    if y_train.nunique() > 2:
        return {
            "optimal_threshold": 0.5,
            "metric_at_threshold": None,
            "optimize_for": optimize_for,
            "message": "Threshold tuning is only applicable to binary classification tasks.",
        }

    proba = model.predict_proba(X_train)[:, 1]
    optimal_threshold = threshold_sweep(y_train, proba, optimize_for)

    y_pred_opt = (proba >= optimal_threshold).astype(int)
    if optimize_for == "f1":
        metric_val = float(f1_score(y_train, y_pred_opt, average="binary", zero_division=0))
    elif optimize_for == "precision":
        metric_val = float(precision_score(y_train, y_pred_opt, average="binary", zero_division=0))
    else:
        metric_val = float(recall_score(y_train, y_pred_opt, average="binary", zero_division=0))

    return {
        "optimal_threshold": round(optimal_threshold, 4),
        "metric_at_threshold": round(metric_val, 4),
        "optimize_for": optimize_for,
    }


def explain_predictions(model_id: str, split_id: str, top_n: int = 5) -> dict:
    """Generate feature importances and sample prediction explanations for a finalized model.

    PURPOSE & WHEN TO USE:
        Call this tool post-finalization to inspect feature importance weights (linear coefficients or permutation importance) and sample predictions.

    INPUT PARAMETERS:
        - `model_id` (str): Finalized model handle ID.
        - `split_id` (str): Active split handle ID.
        - `top_n` (int, default=5): Number of sample prediction rows to inspect.

    RETURNS:
        - Dict containing `feature_importances`, `method`, `top_examples`.
    """
    from .finalize import load_model
    from .interpretation import explain_linear, explain_tree_ensemble
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge, LogisticRegression

    meta = _load_split_meta(split_id)
    target_column = meta.get("target_column", "")

    train_df = _load_train(split_id)
    X_train = train_df.drop(columns=[target_column], errors="ignore")
    y_train = train_df[target_column]

    model = load_model(model_id)

    try:
        feature_names = list(model[:-1].get_feature_names_out())
    except Exception:
        feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]

    from sklearn.pipeline import Pipeline as SKPipeline
    final_estimator = model[-1] if isinstance(model, SKPipeline) else model

    if isinstance(final_estimator, (Ridge, LogisticRegression)):
        importances = explain_linear(model, feature_names)
        method = "coefficients"
    else:
        try:
            X_transformed = model[:-1].transform(X_train)
            importances = explain_tree_ensemble(model[-1], X_transformed, feature_names)
        except Exception:
            importances = {}
        method = "permutation_importance"

    sample = train_df.head(top_n)
    X_sample = sample.drop(columns=[target_column], errors="ignore")
    y_pred = model.predict(X_sample)
    examples = []
    for i, (idx, row) in enumerate(sample.iterrows()):
        examples.append({
            "inputs": {col: (row[col].item() if hasattr(row[col], "item") else row[col])
                       for col in X_sample.columns},
            "actual": row[target_column],
            "predicted": y_pred[i],
        })

    return {
        "feature_importances": dict(list(importances.items())[:20]),
        "method": method,
        "top_examples": examples,
    }


def analyze_prediction_errors(
    model_id: str,
    split_id: str,
    segment_by: Optional[list[str]] = None,
) -> dict:
    """Analyze error distributions across feature subgroups to detect systematic weaknesses.

    PURPOSE & WHEN TO USE:
        Call this tool post-finalization to perform error slice analysis across categorical or binned continuous feature segments.

    INPUT PARAMETERS:
        - `model_id` (str): Finalized model handle ID.
        - `split_id` (str): Active split handle ID.
        - `segment_by` (list[str], optional): List of column names to segment error rates by. Auto-selected if None.

    RETURNS:
        - Dict with `error_by_segment`, `overall_error_rate`, `segments_analyzed`.
    """
    from .finalize import load_model
    from .interpretation import segment_error_rates, auto_select_segment_columns

    meta = _load_split_meta(split_id)
    target_column = meta.get("target_column", "")

    train_df = _load_train(split_id)
    X_train = train_df.drop(columns=[target_column], errors="ignore")
    y_train = train_df[target_column]

    model = load_model(model_id)
    y_pred = pd.Series(model.predict(X_train), name="predicted")

    error = (y_train.reset_index(drop=True) != y_pred).astype(float)

    if not segment_by:
        segment_by = auto_select_segment_columns(X_train.reset_index(drop=True), error)

    if not segment_by:
        return {
            "error_by_segment": {},
            "overall_error_rate": float(error.mean()),
            "segments_analyzed": [],
            "message": "No columns with meaningful correlation to error found",
        }

    error_by_segment = segment_error_rates(
        y_train.reset_index(drop=True),
        y_pred,
        X_train.reset_index(drop=True),
        segment_by,
    )

    return {
        "error_by_segment": error_by_segment,
        "overall_error_rate": float(error.mean()),
        "segments_analyzed": segment_by,
    }


def finalize_model(
    folds_id: str,
    pipeline_spec_id: str,
    model_family: str,
    hyperparams: Optional[dict] = None,
) -> dict:
    """Fit the winning pipeline specification on the FULL training partition and persist the final model.

    PURPOSE & WHEN TO USE:
        Call this tool after completing CV candidate evaluation to train the final model on the entire training set.
        This is the ONLY tool that creates a persisted, reusable fitted model handle (`model_id`).

    INPUT PARAMETERS:
        - `folds_id` (str): CV folds handle ID of winning candidate run.
        - `pipeline_spec_id` (str): Preprocessing pipeline handle ID.
        - `model_family` (str): Winning model family ('linear' or 'tree_ensemble').
        - `hyperparams` (dict, optional): Hyperparameters for final fit.

    RETURNS:
        - Dict containing:
            - `model_id` (str): Unique handle ID assigned to the fitted, saved model artifact.
            - `task_type` (str): Problem task type.
            - `model_family` (str): Model family.

    AI AGENT GUIDELINES:
        - Pass `model_id` to `final_test_evaluation` for one-shot test set evaluation.
    """
    from .finalize import fit_final_pipeline

    folds_data = _load_folds(folds_id)
    split_id = folds_data["split_id"]
    meta = _load_split_meta(split_id)
    target_column = meta.get("target_column", "")
    task_type = meta.get("task_type", "classification")

    train_df = _load_train(split_id)
    X_train = train_df.drop(columns=[target_column], errors="ignore")
    y_train = train_df[target_column]

    model_id = fit_final_pipeline(
        pipeline_spec_id=pipeline_spec_id,
        model_family=model_family,
        hyperparams=hyperparams,
        X_train_full=X_train,
        y_train_full=y_train,
        task_type=task_type,
        folds_id=folds_id,
    )

    return {"model_id": model_id, "task_type": task_type, "model_family": model_family}


def final_test_evaluation(model_id: str, split_id: str) -> dict:
    """Perform one-shot evaluation of the final model on the reserved held-out test set.

    PURPOSE & WHEN TO USE:
        Call this tool as the VERY LAST STEP in the modeling lifecycle.
        This tool reads `test.parquet` and evaluates final generalization performance.
        It enforces a strict single-evaluation rule per split_id.

    INPUT PARAMETERS:
        - `model_id` (str): Finalized model handle ID returned by `finalize_model`.
        - `split_id` (str): Active split handle ID.

    RETURNS:
        - Dict with final test set evaluation metrics (accuracy/f1/auc or rmse/mae/r2).

    AI AGENT GUIDELINES:
        - Never call this tool more than once per split with different models.
    """
    from .finalize import (
        check_test_not_already_evaluated,
        evaluate_on_reserved_test,
        mark_test_evaluated,
    )

    if not check_test_not_already_evaluated(split_id):
        marker_path = get_cache_dir() / "splits" / split_id / ".test_evaluated"
        with open(marker_path) as f:
            marker = json.load(f)
        if marker.get("model_id") != model_id:
            raise RuntimeError(
                f"final_test_evaluation has already been called for split_id={split_id!r} "
                f"with model_id={marker['model_id']!r}. Calling again with a different model_id "
                "violates the one-shot test rule."
            )
        return evaluate_on_reserved_test(model_id, split_id)

    metrics = evaluate_on_reserved_test(model_id, split_id)
    mark_test_evaluated(split_id, model_id)
    return metrics