def fit_final_pipeline(pipeline_spec_id: str, model_family: str, hyperparams: dict, X_train_full, y_train_full) -> str:
    """The one place a real, reusable, persisted model gets created.
    Clones pipeline_spec_id ONCE, fits on the entire train partition (no
    folding), attaches get_model(...). Computes model_id via
    models.compute_model_id() first and returns the existing one if it's
    already been fit before, instead of redoing the work."""

def save_model(model, model_id: str) -> None:
    """joblib.dump under _cache/models/{model_id}.joblib."""

def load_model(model_id: str):
    """joblib.load -- used by final_test_evaluation, explain_predictions,
    analyze_prediction_errors, tune_decision_threshold."""

def evaluate_on_reserved_test(model_id: str, split_id: str) -> dict:
    """Loads split_id's test.parquet -- the one file in the whole system
    read exactly once -- and scores model_id against it."""

def check_test_not_already_evaluated(split_id: str) -> bool:
    """Checks a marker written the first time final_test_evaluation
    succeeds for this split_id. Enforces the one-shot rule -- a second
    call with a different model_id raises rather than silently
    re-evaluating."""