"""
Experiment Tracking MCP tools.

All MLflow interactions go through mlflow_client.py.
All case-library interactions go through case_library.py.
"""

from __future__ import annotations

from typing import Optional

import mlflow
from mlflow.tracking import MlflowClient

try:
    from .mlflow_client import configure_mlflow, get_or_create_experiment
    from .case_library import (
        compute_dataset_fingerprint,
        find_similar_cases,
        store_case,
    )
except ImportError:
    from mlflow_client import configure_mlflow, get_or_create_experiment
    from case_library import (
        compute_dataset_fingerprint,
        find_similar_cases,
        store_case,
    )

# Ensure tracking URI is set when this module is imported
configure_mlflow()


def create_experiment(name: str) -> dict:
    """Get or create an MLflow experiment by name.

    PURPOSE & WHEN TO USE:
        Call this tool at the start of a project or experimentation session to set up or recover an MLflow experiment container.

    INPUT PARAMETERS:
        - `name` (str): Unique experiment name (e.g. 'housing_price_prediction').

    RETURNS:
        - Dict containing:
            - `experiment_id` (str): MLflow experiment ID string.
            - `name` (str): Experiment name.
    """
    experiment_id = get_or_create_experiment(name)
    return {"experiment_id": experiment_id, "name": name}


def start_run(experiment_id: str, run_name: str) -> dict:
    """Start a new MLflow tracking run within an experiment.

    PURPOSE & WHEN TO USE:
        Call this tool before evaluating a candidate pipeline configuration or training a model to begin recording parameters and metrics.

    INPUT PARAMETERS:
        - `experiment_id` (str): MLflow experiment ID string returned by `create_experiment`.
        - `run_name` (str): Descriptive run name (e.g. 'linear_baseline_v1').

    RETURNS:
        - Dict with `run_id` (str).
    """
    mlflow.set_experiment(experiment_id=experiment_id)
    run = mlflow.start_run(run_name=run_name)
    return {"run_id": run.info.run_id}


def log_parameters(
    run_id: str,
    params: dict,
    split_id: str,
    pipeline_spec_id: str,
    folds_id: str,
    model_family: str,
) -> dict:
    """Log model hyperparameters and artifact handle IDs to an active MLflow run.

    PURPOSE & WHEN TO USE:
        Call this tool after starting an MLflow run to log configuration parameters and handle IDs for complete reproducibility.

    INPUT PARAMETERS:
        - `run_id` (str): Active MLflow run ID string.
        - `params` (dict): Model hyperparameter dictionary.
        - `split_id` (str): Split handle ID.
        - `pipeline_spec_id` (str): Pipeline handle ID.
        - `folds_id` (str): CV folds handle ID.
        - `model_family` (str): Model family choice.

    RETURNS:
        - Dict with `logged` (list of parameter keys logged).
    """
    client = MlflowClient()
    all_params = {
        "split_id": split_id,
        "pipeline_spec_id": pipeline_spec_id,
        "folds_id": folds_id,
        "model_family": model_family,
        **{str(k): str(v) for k, v in (params or {}).items()},
    }
    for key, value in all_params.items():
        client.log_param(run_id, key, str(value)[:500])
    return {"logged": list(all_params.keys())}


def log_metrics(run_id: str, metrics: dict) -> dict:
    """Log numerical performance metrics to an active MLflow run.

    PURPOSE & WHEN TO USE:
        Call this tool after cross-validation or evaluation to record accuracy, F1, AUC, RMSE, MAE, or R2 scores into MLflow.

    INPUT PARAMETERS:
        - `run_id` (str): Active MLflow run ID.
        - `metrics` (dict): Dictionary mapping metric names to floating point numbers.

    RETURNS:
        - Dict with `logged` (list of metric keys logged).
    """
    client = MlflowClient()
    for key, value in (metrics or {}).items():
        try:
            client.log_metric(run_id, key, float(value))
        except (TypeError, ValueError):
            pass
    return {"logged": list(metrics.keys())}


def log_artifact(run_id: str, path: str) -> dict:
    """Log a file artifact (model binary, confusion matrix plot, or spec file) to an MLflow run.

    PURPOSE & WHEN TO USE:
        Call this tool to persist local files or plots associated with a run into MLflow storage.

    INPUT PARAMETERS:
        - `run_id` (str): Active MLflow run ID.
        - `path` (str): Absolute or relative path to local artifact file.

    RETURNS:
        - Dict with `logged_artifact` path string.
    """
    client = MlflowClient()
    client.log_artifact(run_id, path)
    return {"logged_artifact": path}


def end_run(run_id: str, status: str = "finished") -> dict:
    """Complete and terminate an active MLflow run.

    PURPOSE & WHEN TO USE:
        Call this tool when run evaluation finishes or fails to finalize the run state in MLflow.

    INPUT PARAMETERS:
        - `run_id` (str): Active MLflow run ID.
        - `status` (str, default='finished'): Completion status ('finished' or 'failed').

    RETURNS:
        - Dict with `run_id` and `status`.
    """
    client = MlflowClient()
    mlflow_status = "FINISHED" if status.lower() == "finished" else "FAILED"
    client.set_terminated(run_id, status=mlflow_status)
    return {"run_id": run_id, "status": status}


def get_run(run_id: str) -> dict:
    """Retrieve full run metadata, parameters, and logged metrics for a given run ID.

    PURPOSE & WHEN TO USE:
        Call this tool to inspect recorded details of a specific MLflow run.

    INPUT PARAMETERS:
        - `run_id` (str): MLflow run ID.

    RETURNS:
        - Dict containing `run_id`, `status`, `params`, `metrics`, `tags`.
    """
    client = MlflowClient()
    run = client.get_run(run_id)
    return {
        "run_id": run_id,
        "status": run.info.status,
        "params": dict(run.data.params),
        "metrics": dict(run.data.metrics),
        "tags": dict(run.data.tags),
    }


def compare_runs(experiment_id: str, metric: str) -> dict:
    """Rank and compare all recorded runs within an experiment by a target metric.

    PURPOSE & WHEN TO USE:
        Call this tool to list all runs in an experiment ordered by performance.

    INPUT PARAMETERS:
        - `experiment_id` (str): MLflow experiment ID.
        - `metric` (str): Metric name to sort by (e.g. 'f1' or 'rmse').

    RETURNS:
        - Dict containing list of `runs` sorted in descending order of performance.
    """
    client = MlflowClient()
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        order_by=[f"metrics.{metric} DESC"],
    )
    return {
        "runs": [
            {
                "run_id": r.info.run_id,
                "run_name": r.info.run_name,
                "status": r.info.status,
                "metrics": dict(r.data.metrics),
                "params": dict(r.data.params),
            }
            for r in runs
        ]
    }


def get_best_run(experiment_id: str, metric: str, mode: str = "max") -> dict:
    """Retrieve the single best-performing run in an experiment based on a metric and direction.

    PURPOSE & WHEN TO USE:
        Call this tool to identify the top candidate run to finalize.

    INPUT PARAMETERS:
        - `experiment_id` (str): MLflow experiment ID.
        - `metric` (str): Metric name (e.g. 'accuracy' or 'rmse').
        - `mode` (str, default='max'): Optimization direction ('max' for accuracy/f1/auc, 'min' for loss/rmse/mae).

    RETURNS:
        - Dict containing `run_id`, `run_name`, `metrics`, `params`.
    """
    order = "DESC" if mode == "max" else "ASC"
    client = MlflowClient()
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        order_by=[f"metrics.{metric} {order}"],
        max_results=1,
    )
    if not runs:
        return {"run_id": None, "metrics": {}, "message": "No runs found"}
    best = runs[0]
    return {
        "run_id": best.info.run_id,
        "run_name": best.info.run_name,
        "metrics": dict(best.data.metrics),
        "params": dict(best.data.params),
    }


def log_case(
    dataset_fingerprint: dict,
    task_type: str,
    approach_summary: str,
    outcome_metrics: dict,
) -> dict:
    """Save an anonymized ML experiment case history entry into the case library for future reference.

    PURPOSE & WHEN TO USE:
        Call this tool after completing a full ML experiment cycle to store dataset meta-features, problem type, pipeline approach, and outcome metrics into the case library.

    INPUT PARAMETERS:
        - `dataset_fingerprint` (dict): Anonymized structural fingerprint dict.
        - `task_type` (str): Task type ('classification' or 'regression').
        - `approach_summary` (str): Text description of preprocessing, model family, and tuning approach used.
        - `outcome_metrics` (dict): Final test performance metrics achieved.

    RETURNS:
        - Dict with `stored` (bool) and `task_type`.
    """
    store_case(
        fingerprint=dataset_fingerprint,
        task_type=task_type,
        summary=approach_summary,
        metrics=outcome_metrics,
        scope="private",
    )
    return {"stored": True, "task_type": task_type}


def retrieve_similar_cases(
    dataset_fingerprint: dict,
    task_type: str,
    k: int = 5,
    scope: str = "private",
) -> dict:
    """Query the case library for historical ML experiments on similar datasets.

    PURPOSE & WHEN TO USE:
        Call this tool before planning a new ML experiment to inspect similar past datasets and successful preprocessing strategies.

    INPUT PARAMETERS:
        - `dataset_fingerprint` (dict): Anonymized structural fingerprint dict.
        - `task_type` (str): Task type ('classification' or 'regression').
        - `k` (int, default=5): Number of top similar cases to return.
        - `scope` (str, default='private'): Case search scope ('private' or 'shared').

    RETURNS:
        - Dict with list of `similar_cases`, `k`, and `scope`.
    """
    cases = find_similar_cases(
        fingerprint=dataset_fingerprint,
        task_type=task_type,
        k=k,
        scope=scope if scope != "shared" else "all",
    )
    return {"similar_cases": cases, "k": k, "scope": scope}