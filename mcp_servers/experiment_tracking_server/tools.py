def create_experiment(name: str) -> dict:
    """Get-or-create against MLflow -- calling again with the same name
    returns the same experiment_id rather than erroring or duplicating."""

def start_run(experiment_id: str, run_name: str) -> dict:
    """Opens a new MLflow run under experiment_id. Returns {run_id}."""

def log_parameters(run_id: str, params: dict, split_id: str, pipeline_spec_id: str, folds_id: str, model_family: str) -> dict:
    """Logs params plus the four handle IDs explicitly, so a run's exact
    configuration is fully reconstructible later from IDs alone, without
    needing to remember what those handles pointed to at the time."""

def log_metrics(run_id: str, metrics: dict) -> dict:
    """Shape of metrics varies by task_type (rmse/mae/r2 for regression,
    accuracy/f1/roc_auc for classification) -- MLflow just stores
    whatever's given."""

def log_artifact(run_id: str, path: str) -> dict:
    """Typically the cached .joblib for a model_id, or a saved
    confusion-matrix image."""

def end_run(run_id: str, status: str = "finished") -> dict:
    """Closes the MLflow run with status in {finished, failed}."""

def get_run(run_id: str) -> dict:
    """Full run record: params, metrics, artifacts, status."""

def compare_runs(experiment_id: str, metric: str) -> dict:
    """All runs in experiment_id, sorted by metric descending."""

def get_best_run(experiment_id: str, metric: str, mode: str) -> dict:
    """mode in {max, min}. Returns {run_id, metrics} for the top
    result."""

def log_case(dataset_fingerprint: dict, task_type: str, approach_summary: str, outcome_metrics: dict) -> dict:
    """dataset_fingerprint comes from
    case_library.compute_dataset_fingerprint() -- a structural signature
    (row-count bucket, role mix, target balance), NEVER the raw dataset,
    so cases are shareable without leaking anyone's actual data.
    Typically called right after end_run."""

def retrieve_similar_cases(dataset_fingerprint: dict, task_type: str, k: int = 5, scope: str = "private") -> dict:
    """Called before planning begins on a new dataset. scope='shared'
    also searches an opt-in, community-contributed case pool. Returns the
    top-k most similar past cases by case_library.fingerprint_distance()."""