"""
Model finalization and test-set evaluation for modeling_server.

This file contains THE ONLY tool that writes to _cache/models/.
fit_final_pipeline() is the one place a real, reusable, persisted fitted
model gets created.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from shared.cache import compute_handle_id, get_cache_dir, object_exists


def _load_split_meta(split_id: str) -> dict:
    meta_path = get_cache_dir() / "splits" / split_id / "meta.json"
    if not meta_path.exists():
        return {}
    with open(meta_path) as f:
        return json.load(f)


def fit_final_pipeline(
    pipeline_spec_id: str,
    model_family: str,
    hyperparams: Optional[dict],
    X_train_full: pd.DataFrame,
    y_train_full: pd.Series,
    task_type: str,
    folds_id: str,
) -> str:
    """Input: the unfitted pipeline spec's ID, model family/hyperparams,
    the full train partition's features and target.  Processing: the one
    place a real, reusable, persisted model gets created.  Clones
    pipeline_spec_id ONCE, attaches get_model(...), fits on the entire train
    partition.  Computes model_id first and returns the existing one if
    already fit.  Output: a model_id string."""
    from .models import compute_model_id, get_model
    from .cv import clone_pipeline_from_spec

    model_id = compute_model_id(folds_id, pipeline_spec_id, model_family, hyperparams)

    if object_exists("models", model_id):
        return model_id

    pipe = clone_pipeline_from_spec(pipeline_spec_id)
    estimator = get_model(model_family, task_type, hyperparams)

    # For class_weight strategy, pass class_weight='balanced' if supported
    if hasattr(estimator, "class_weight") and estimator.class_weight is None:
        pass  # keep default

    final_pipe = Pipeline(list(pipe.steps) + [("model", estimator)])
    final_pipe.fit(X_train_full, y_train_full)

    save_model(final_pipe, model_id)
    return model_id


def save_model(model: Pipeline, model_id: str) -> None:
    """Input: a fitted pipeline+model object and its handle_id.
    Processing: joblib.dump.  Output: none -- writes to
    _cache/models/{model_id}.joblib."""
    models_dir = get_cache_dir() / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, models_dir / f"{model_id}.joblib")


def load_model(model_id: str) -> Pipeline:
    """Input: a model_id string.  Processing: joblib.load from _cache/models/.
    Output: the fitted pipeline+model object -- used by
    final_test_evaluation, explain_predictions, analyze_prediction_errors,
    and tune_decision_threshold."""
    model_path = get_cache_dir() / "models" / f"{model_id}.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_id!r}")
    return joblib.load(model_path)


def evaluate_on_reserved_test(model_id: str, split_id: str) -> dict:
    """Input: a finalized model_id and the split_id to score against.
    Processing: loads test.parquet (the one file in the whole system read
    exactly once), runs predict()/predict_proba(), and scores against
    task_type-appropriate metrics.  Output: {metric_name: value} dict."""
    meta = _load_split_meta(split_id)
    task_type = meta.get("task_type", "classification")
    target_column = meta.get("target_column")

    test_path = get_cache_dir() / "splits" / split_id / "test.parquet"
    test_df = pd.read_parquet(test_path)
    X_test = test_df.drop(columns=[target_column], errors="ignore")
    y_test = test_df[target_column]

    model = load_model(model_id)
    y_pred = model.predict(X_test)

    if task_type == "classification":
        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        }
        if hasattr(model, "predict_proba"):
            try:
                proba = model.predict_proba(X_test)
                if proba.shape[1] == 2:
                    metrics["roc_auc"] = float(roc_auc_score(y_test, proba[:, 1]))
                else:
                    metrics["roc_auc"] = float(
                        roc_auc_score(y_test, proba, multi_class="ovr", average="macro")
                    )
            except Exception:
                pass
    else:
        mse = mean_squared_error(y_test, y_pred)
        metrics = {
            "rmse": float(np.sqrt(mse)),
            "mae": float(mean_absolute_error(y_test, y_pred)),
            "r2": float(r2_score(y_test, y_pred)),
        }

    return metrics


def check_test_not_already_evaluated(split_id: str) -> bool:
    """Input: a split_id string.  Processing: checks a marker file written
    the first time final_test_evaluation succeeds for this split_id.
    Output: True if no prior evaluation marker exists (safe to proceed),
    False otherwise -- enforcing the one-shot rule."""
    marker_path = get_cache_dir() / "splits" / split_id / ".test_evaluated"
    return not marker_path.exists()


def mark_test_evaluated(split_id: str, model_id: str) -> None:
    """Write the evaluation marker file so a second call is refused."""
    marker_path = get_cache_dir() / "splits" / split_id / ".test_evaluated"
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    with open(marker_path, "w") as f:
        json.dump({"model_id": model_id}, f)