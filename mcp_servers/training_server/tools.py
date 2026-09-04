def split_dataset(dataset_id: str, target_column: str, test_size: float = 0.2,
                   strategy: str = "stratified", random_state: int = 42) -> dict:
    """Writes _cache/splits/{split_id}/{train,test}.parquet. split_id is a
    hash of (dataset_id, target_column, test_size, strategy, random_state)
    -- same inputs always resolve to the same split_id (idempotent)."""

def create_preprocessing_pipeline(dataset_id: str, target_column: str, config: dict) -> dict:
    """config = {numeric_impute, categorical_impute, encoding_strategy,
    scale: bool}. Delegates to pipelines.build_column_transformer, saves
    the unfitted Pipeline via joblib under a config-derived pipeline_id."""

def train_model(split_id: str, pipeline_id: str, model_type: str,
                 hyperparams: dict | None = None) -> dict:
    """model_id = hash(split_id, pipeline_id, model_type, hyperparams).
    If _cache/models/{model_id}.joblib already exists, return it unchanged
    -- this is the idempotency check that makes crash-recovery safe.
    Otherwise: load train split, clone pipeline, attach estimator from
    models.get_model(), fit(X_train, y_train), joblib.dump. Calls
    failure_injection.maybe_inject_failure(model_type) before fitting."""

def evaluate_model(model_id: str, split_id: str) -> dict:
    """Loads fitted pipeline+model, predicts on test split, returns
    {accuracy, precision, recall, f1, roc_auc, confusion_matrix}."""

def cross_validate_model(dataset_id: str, pipeline_id: str, model_type: str,
                          hyperparams: dict | None = None, cv_folds: int = 5) -> dict:
    """sklearn.model_selection.cross_validate with StratifiedKFold. Used
    only on the final chosen model, not every candidate."""

def predict(model_id: str, rows: list[dict]) -> dict:
    """Loads fitted pipeline+model, wraps rows in a one-row DataFrame per
    call, returns predictions + probabilities. Used by the Gradio
    'try it yourself' panel."""