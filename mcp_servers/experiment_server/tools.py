def create_experiment(name: str) -> dict          # get-or-create via mlflow_client
def start_run(experiment_id: str, run_name: str) -> dict
def log_parameters(run_id: str, params: dict) -> dict
def log_metrics(run_id: str, metrics: dict) -> dict
def log_artifact(run_id: str, path: str) -> dict   # e.g. confusion-matrix PNG, the joblib file
def end_run(run_id: str, status: str = "FINISHED") -> dict
def get_run(run_id: str) -> dict
def compare_runs(experiment_id: str, metric: str = "roc_auc") -> dict
def get_best_run(experiment_id: str, metric: str = "roc_auc", mode: str = "max") -> dict