MODEL_REGISTRY = {
    "logistic_regression": (LogisticRegression, dict(max_iter=1000, class_weight="balanced")),
    "random_forest":       (RandomForestClassifier, dict(n_estimators=300, class_weight="balanced", n_jobs=-1)),
    "gradient_boosting":   (HistGradientBoostingClassifier, dict(max_iter=200)),
}

def get_model(model_type: str, hyperparams: dict | None = None):
    """Merges hyperparams over MODEL_REGISTRY defaults, instantiates."""

def compute_model_id(split_id: str, pipeline_id: str, model_type: str, hyperparams: dict | None) -> str:
    """sha1 of the sorted, JSON-serialized tuple. Deterministic -- this IS
    the idempotency mechanism train_model relies on."""

def model_exists(model_id: str) -> bool