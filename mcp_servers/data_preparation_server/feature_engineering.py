def apply_row_wise_transforms(df: pd.DataFrame, feature_defs: list[dict]) -> pd.DataFrame:
    """Computes date-part extraction, ratios, and other single-row
    transforms directly -- safe to run before any split/fold logic since
    a row's derived value never depends on any other row."""

def apply_aggregation_transforms(train_df: pd.DataFrame, feature_defs: list[dict]) -> dict:
    """Computes group means/counts/etc. MUST be fit on train data only,
    per fold, since an aggregation depends on every row in the group at
    once. Returns a fitted lookup table applied to val/test rows,
    computed inside Modeling MCP's per-fold loop, not here."""

SELECT_METHODS = {"filter": ..., "wrapper": ..., "embedded": ...}
"""filter = mutual_info/ANOVA-F/chi2 ranking (cheap); embedded = Lasso
coefficients or tree importances via SelectFromModel; wrapper =
RFE/RFECV (expensive, only used when filter/embedded results are
ambiguous)."""

def build_feature_selection_spec(method: str, k_or_percentile) -> dict:
    """Returns {method, k_or_percentile} only -- see select_features's
    docstring in tools.py for why this can never be a frozen column
    list."""