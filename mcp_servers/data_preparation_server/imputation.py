"""
Imputation strategy registry + resolution logic. Everything here produces
or reasons about *specs* -- plain strategy names and parameters. The actual
fitted imputer objects only get built when a spec is turned into a real
sklearn transformer inside pipeline.py.
"""

from __future__ import annotations

from typing import Any

from sklearn.experimental import enable_iterative_imputer  # noqa: F401  (registers IterativeImputer)
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer

try:
    from ..shared.schemas import ColumnRole
except ImportError:
    from shared.schemas import ColumnRole

NUMERIC_ROLES = {ColumnRole.NUMERIC_CONTINUOUS.value, ColumnRole.NUMERIC_DISCRETE.value}
CATEGORICAL_ROLES = {ColumnRole.CATEGORICAL_NOMINAL.value, ColumnRole.CATEGORICAL_ORDINAL.value, ColumnRole.BOOLEAN.value}
# Roles this server actually imputes/scales/encodes. Identifiers, raw
# datetimes, free text, and constant columns are left alone here -- an
# identifier shouldn't be median-imputed, and a raw datetime should be
# turned into row_wise features (derive_features) before it's numeric at
# all. Skipping them means they simply never get a strategy entry, so
# _build_preprocessor's ColumnTransformer drops them via remainder="drop"
# instead of a numeric imputer choking on a datetime64 column.
ACTIONABLE_ROLES = NUMERIC_ROLES | CATEGORICAL_ROLES

IMPUTE_STRATEGIES: dict[str, Any] = {
    "mean": lambda: SimpleImputer(strategy="mean"),
    "median": lambda: SimpleImputer(strategy="median"),
    "mode": lambda: SimpleImputer(strategy="most_frequent"),
    "constant": lambda: SimpleImputer(strategy="constant"),
    # 'drop_rows' has no transformer -- a fitted step can't change row
    # count, so this is handled upstream (row filtering before the
    # pipeline is fit), never inside pipeline.py's ColumnTransformer.
    "drop_rows": None,
    "model_based": lambda: IterativeImputer(max_iter=10, random_state=0),
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
    role = column_role.value if hasattr(column_role, "value") else column_role
    if role not in ACTIONABLE_ROLES:
        return None
    if missing_pct >= 60.0:
        return "drop_rows"
    if role in NUMERIC_ROLES:
        return "model_based" if mar_flag else "median"
    return "constant"


def build_imputer_spec(strategy_map: dict | str, column_roles: dict, mar_flags: dict) -> dict:
    """Applies resolve_auto_strategy per column when strategy_map='auto',
    otherwise uses the caller-supplied per-role overrides directly.

    mar_flags is expected in the shape Data Understanding MCP's
    analyze_missing_values (or this server's lightweight internal
    equivalent) produces: {column: {"missing_pct": float, "likely_mar":
    bool}}.
    """
    if strategy_map in ("auto", None):
        per_column = {}
        for column, role in column_roles.items():
            entry = (mar_flags or {}).get(column, {})
            strategy = resolve_auto_strategy(
                role, bool(entry.get("likely_mar", False)), float(entry.get("missing_pct", 0.0))
            )
            if strategy is not None:
                per_column[column] = strategy
        return {"strategies": per_column}

    if isinstance(strategy_map, dict):
        role_values = {ColumnRole(r).value if not isinstance(r, str) else r for r in ColumnRole}
        keys_are_roles = len(strategy_map) > 0 and all(key in role_values for key in strategy_map)
        if keys_are_roles:
            per_column = {}
            for column, role in column_roles.items():
                role_value = role.value if hasattr(role, "value") else role
                if role_value in strategy_map:
                    per_column[column] = strategy_map[role_value]
                    continue
                entry = (mar_flags or {}).get(column, {})
                strategy = resolve_auto_strategy(
                    role_value, bool(entry.get("likely_mar", False)), float(entry.get("missing_pct", 0.0))
                )
                if strategy is not None:
                    per_column[column] = strategy
            return {"strategies": per_column}
        # Keys are already column names -- caller-supplied per-column map,
        # always honored as-is even for a role this server wouldn't have
        # touched automatically (the caller asked explicitly).
        return {"strategies": dict(strategy_map)}

    raise ValueError(f"Unsupported strategy_map value: {strategy_map!r}")