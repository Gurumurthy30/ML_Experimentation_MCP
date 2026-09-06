def configure_mlflow() -> None:
    """mlflow.set_tracking_uri('sqlite:///mlflow.db') -- called once at
    server startup."""

def get_or_create_experiment(name: str) -> str:
    """Wraps MlflowClient's get_experiment_by_name / create_experiment so
    create_experiment (tools.py) is idempotent."""
