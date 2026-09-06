def alpha_grid_default(model_family: str) -> list[float]:
    """A log-spaced default grid (e.g. 0.001 to 100), used by
    regularization_path_search when the caller doesn't supply alpha_grid."""

def sweep_regularization(folds_id: str, pipeline_spec_id: str, alpha_grid: list[float]) -> dict:
    """Calls run_cross_validation once per alpha value, collects the mean
    CV score for each."""

def threshold_sweep(y_true, y_proba, optimize_for: str) -> float:
    """Tries a grid of thresholds between 0 and 1, returns the one that
    maximizes optimize_for's corresponding sklearn metric."""