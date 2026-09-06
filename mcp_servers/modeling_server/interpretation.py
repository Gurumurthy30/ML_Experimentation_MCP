def explain_linear(model, feature_names: list[str]) -> dict:
    """Returns each feature's coefficient directly from the fitted linear
    model, paired with feature_names."""

def explain_tree_ensemble(model, X, feature_names: list[str], method: str = "permutation") -> dict:
    """Permutation importance (shuffles each feature, measures the score
    drop) by default -- more reliable than built-in impurity-based
    importance when features are correlated."""

def segment_error_rates(y_true, y_pred, X, segment_by: list[str]) -> dict:
    """Groups rows by the values in segment_by's columns (binning numeric
    columns into quantiles first) and computes the error rate within each
    group, sorted worst-first."""

def auto_select_segment_columns(X, error) -> list[str]:
    """When segment_by is omitted, picks the columns whose values
    correlate most with per-row error magnitude, as a starting point for
    segment_error_rates."""