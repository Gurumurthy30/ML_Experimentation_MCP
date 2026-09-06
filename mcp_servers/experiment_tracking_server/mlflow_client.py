"""
MLflow client helpers for experiment_tracking_server.

configure_mlflow() is called once at server startup.
All other modules import get_or_create_experiment() for idempotent access.
"""

from __future__ import annotations

import os

import mlflow
from mlflow.tracking import MlflowClient


def configure_mlflow() -> None:
    """Configures MLflow tracking URI with local file-store fallback."""
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
    default_uri = "file:///./mlruns"
    uri = os.environ.get("MLFLOW_TRACKING_URI", default_uri)
    try:
        mlflow.set_tracking_uri(uri)
    except Exception:
        mlflow.set_tracking_uri(default_uri)


def get_or_create_experiment(name: str) -> str:
    """Wraps MlflowClient's get_experiment_by_name / create_experiment so
    create_experiment (tools.py) is idempotent."""
    client = MlflowClient()
    experiment = client.get_experiment_by_name(name)
    if experiment is not None:
        return experiment.experiment_id
    experiment_id = client.create_experiment(name)
    return experiment_id
