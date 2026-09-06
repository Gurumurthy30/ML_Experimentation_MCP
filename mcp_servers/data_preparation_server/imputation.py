IMPUTE_STRATEGIES = {
    "mean": ..., "median": ..., "mode": ..., "constant": ...,
    "drop_rows": ..., "model_based": ...,
}
"""Registry of strategy names to their sklearn/pandas implementation --
referenced by name in imputer_spec, resolved to an actual transformer only
when the pipeline spec gets cloned and fit, inside Modeling MCP."""

def resolve_auto_strategy(column_role: str, mar_flag: bool, missing_pct: float) -> str:
    """Numeric columns default to 'median' ('model_based', i.e. a KNN or
    iterative imputer, when mar_flag=True -- structured missingness
    benefits from using other columns to predict the missing value).
    Categorical columns default to a '__missing__' constant category
    rather than mode, since mode silently erases the fact anything was
    missing. missing_pct above ~60% resolves to 'drop_rows' instead,
    flagged back to the caller rather than done silently."""

def build_imputer_spec(strategy_map: dict, column_roles: dict, mar_flags: dict) -> dict:
    """Applies resolve_auto_strategy per column when strategy_map='auto',
    otherwise uses the caller-supplied per-role overrides directly."""