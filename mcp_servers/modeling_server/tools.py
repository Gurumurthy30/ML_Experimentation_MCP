def establish_baseline(split_id: str, task_type: str) -> dict:
    """Fits a trivial predictor once against split_id's train partition --
    mean/median for regression, majority-class/stratified for
    classification -- and evaluates it. No CV needed since a constant
    predictor can't overfit. Returns {metric: value}, the floor every
    later model gets compared against."""

def cross_validate_model(folds_id: str, pipeline_spec_id: str, model_family: str, hyperparams: dict = None) -> dict:
    """THE default scoring call for every candidate configuration.
    Internally: for each (train_idx, val_idx) in folds_id, clones
    pipeline_spec_id fresh via cv.clone_pipeline_from_spec(), fits on
    train_idx rows only (this is where imputation/encoding/scaling/
    feature selection actually get fit, correctly, per fold), scores on
    val_idx. Produces no persisted model -- scoring only. Returns
    {metric: {mean, std}} plus per_fold_scores."""

def diagnose_fit(folds_id: str, pipeline_spec_id: str, model_family: str, cv_result: dict) -> dict:
    """Re-runs the same per-fold fit as cross_validate_model but also
    captures each fold's train-set score (omitted from cv_result to keep
    that payload small) and compares it to the val score already
    computed. Returns {train_val_gap, gap_std_across_folds, verdict} via
    cv.verdict_from_gap()."""

def regularization_path_search(folds_id: str, pipeline_spec_id: str, model_family: str = "linear", alpha_grid: list[float] = None) -> dict:
    """Only meaningful for model_family='linear'. Runs
    cross_validate_model once per alpha in alpha_grid (a sensible default
    if omitted), picks the alpha with the best mean CV score. Called when
    diagnose_fit returns verdict='overfitting' on a linear model. Returns
    {best_alpha, cv_score_per_alpha}."""

def handle_class_imbalance(split_id: str, target_column: str, strategy: str = "auto", severity_threshold: float = None) -> dict:
    """strategy='auto' checks imbalance.detect_severity_ratio() against
    severity_threshold. Returns imbalance_spec -- a recipe, not a
    resampled dataset: the actual resampling (SMOTE, undersampling, or
    just class_weight) gets applied inside each fold's training data only,
    inside cross_validate_model, never before folding."""

def calibrate_probabilities(model_id: str, method: str = "platt") -> dict:
    """Only meaningful post-finalize_model. method in {platt, isotonic}.
    Returns a NEW calibrated_model_id -- the original model_id is left
    untouched."""

def tune_decision_threshold(model_id: str, split_id: str, optimize_for: str) -> dict:
    """Post-finalize_model. Sweeps thresholds against split_id's train
    partition (not the reserved test partition) to find the one that
    optimizes optimize_for (f1/precision/recall). Returns
    {optimal_threshold, metric_at_threshold}."""

def explain_predictions(model_id: str, split_id: str, top_n: int) -> dict:
    """Post-finalize_model, read-only. model_family='linear' returns
    coefficients directly; 'tree_ensemble' returns permutation importance
    (more reliable than built-in impurity importance when features are
    correlated). Also returns top_n example predictions with inputs."""

def analyze_prediction_errors(model_id: str, split_id: str, segment_by: list[str] = None) -> dict:
    """Post-finalize_model. Breaks the error rate down by feature-value
    bucket for each column in segment_by (auto-selected via
    interpretation.auto_select_segment_columns() if omitted). This is
    where 'fine overall, bad for tenure < 3 months' surfaces --
    explain_predictions only shows the aggregate view."""

def finalize_model(folds_id: str, pipeline_spec_id: str, model_family: str, hyperparams: dict = None) -> dict:
    """THE ONLY tool that produces a persisted, reusable fitted model.
    Clones pipeline_spec_id once (not per fold) and fits it on the entire
    train partition. Writes to _cache/models/{model_id}.joblib. folds_id
    is passed in only to confirm which configuration won, not used for
    folding here. Returns {model_id}."""

def final_test_evaluation(model_id: str, split_id: str) -> dict:
    """The only tool permitted to read splits/{split_id}/test.parquet.
    Checks finalize.check_test_not_already_evaluated() first -- a second
    call against the same split_id with a DIFFERENT model_id is refused,
    preventing the finalist from being re-rolled against the same
    held-out data. Returns {metric: value}."""