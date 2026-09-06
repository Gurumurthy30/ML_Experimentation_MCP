def clone_pipeline_from_spec(pipeline_spec_id: str):
    """Loads the unfitted Pipeline via pipeline.load_pipeline_spec() (Data
    Preparation MCP's shared cache) and calls sklearn.clone() on it --
    returns a fresh, still-unfitted copy every time, which is what makes
    correct per-fold refitting possible without cross-fold
    contamination."""

def run_cross_validation(folds: list, pipeline_spec_id: str, model_family: str, hyperparams: dict) -> dict:
    """For each (train_idx, val_idx) in folds: clone_pipeline_from_spec(),
    attach get_model(model_family, ...), .fit() on train_idx rows, score
    on val_idx rows. Aggregates per-fold scores into mean/std."""

def compute_fold_gap(cv_result: dict) -> dict:
    """Train score minus val score, averaged across folds, plus the
    standard deviation of that gap across folds -- a fold with unusually
    high variance is itself evidence of instability, on top of the
    average gap."""

def verdict_from_gap(gap_stats: dict) -> str:
    """Threshold logic on compute_fold_gap's output: large positive gap ->
    'overfitting'; both train and val scores low -> 'underfitting';
    otherwise 'good_fit'."""