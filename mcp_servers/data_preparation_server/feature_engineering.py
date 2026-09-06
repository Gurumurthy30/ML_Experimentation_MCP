"""
Feature derivation (row-wise + aggregation) and feature-selection method
registry for data_preparation_server.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import (
    RFE,
    SelectFromModel,
    SelectPercentile,
    f_classif,
    f_regression,
    mutual_info_classif,
    mutual_info_regression,
)


def apply_row_wise_transforms(df: pd.DataFrame, feature_defs: list[dict]) -> pd.DataFrame:
    """Computes date-part extraction, ratios, and other single-row
    transforms directly -- safe to run before any split/fold logic since a
    row's derived value never depends on any other row."""
    out = df.copy()
    for feature_def in feature_defs:
        if feature_def.get("expression_type") != "row_wise":
            continue
        out[feature_def["name"]] = _evaluate_row_wise(out, feature_def["definition"])
    return out


def _evaluate_row_wise(df: pd.DataFrame, definition: dict) -> pd.Series:
    kind = definition.get("kind")
    if kind == "datetime_part":
        # e.g. {"kind": "datetime_part", "column": "signup_date", "part": "month"}
        series = pd.to_datetime(df[definition["column"]], errors="coerce")
        return getattr(series.dt, definition["part"])
    if kind == "ratio":
        # e.g. {"kind": "ratio", "numerator": "charges", "denominator": "tenure"}
        numerator = df[definition["numerator"]]
        denominator = df[definition["denominator"]].replace(0, np.nan)
        return numerator / denominator
    if kind == "arithmetic":
        # e.g. {"kind": "arithmetic", "expression": "monthly_charges * tenure"}
        return df.eval(definition["expression"])
    raise ValueError(f"Unsupported row_wise feature kind: {kind!r}")


def apply_aggregation_transforms(train_df: pd.DataFrame, feature_defs: list[dict]) -> dict:
    """Computes group means/counts/etc. MUST be fit on train data only, per
    fold, since an aggregation depends on every row in the group at once.
    Returns a fitted lookup table applied to val/test rows, computed
    inside Modeling MCP's per-fold loop (via DeriveFeaturesTransformer in
    pipeline.py), not here directly -- this function is the reusable piece
    that per-fold call delegates to."""
    lookups: dict[str, dict] = {}
    for feature_def in feature_defs:
        if feature_def.get("expression_type") != "aggregation":
            continue
        definition = feature_def["definition"]
        group_cols = definition["group_by"]
        value_col = definition["value_column"]
        agg = definition.get("agg", "mean")
        table = train_df.groupby(group_cols)[value_col].agg(agg)
        lookups[feature_def["name"]] = {
            "group_by": group_cols,
            "table": table,
            "global_fallback": float(train_df[value_col].agg(agg)),
        }
    return lookups


SELECT_METHODS = {"filter": "filter", "wrapper": "wrapper", "embedded": "embedded"}
"""filter = mutual_info/ANOVA-F ranking (cheap); embedded = tree importances
via SelectFromModel; wrapper = RFE (expensive, only used when filter/
embedded results are ambiguous)."""


def build_feature_selection_spec(method: str, k_or_percentile: int | float) -> dict:
    """Input: the resolved selection method and either an integer feature
    count or a float percentile (0-1) of features to keep. Processing:
    packages the two into a plain recipe, no fitting. Output: {method,
    k_or_percentile} only -- see select_features's docstring in tools.py
    for why this can never be a frozen column list; the actual column
    subset is recomputed per fold, downstream in Modeling MCP."""
    if method not in SELECT_METHODS:
        raise ValueError(f"Unknown feature selection method: {method!r}")
    return {"method": method, "k_or_percentile": k_or_percentile}


class FeatureSelector(BaseEstimator, TransformerMixin):
    """The transformer pipeline.py wires in as the pipeline's final
    'select' step. Deliberately holds no frozen column list at construction
    time -- .fit() recomputes which columns to keep from whatever rows it's
    given, so a fresh clone fit per CV fold in Modeling MCP re-derives the
    selection from that fold's training rows only (see
    build_feature_selection_spec's docstring for why a frozen list would
    leak).
    """

    def __init__(
        self,
        method: str = "filter",
        k_or_percentile: float | int = 0.8,
        task_type: str = "classification",
        random_state: int = 0,
    ):
        self.method = method
        self.k_or_percentile = k_or_percentile
        self.task_type = task_type
        self.random_state = random_state

    def _percentile(self, n_features: int) -> float:
        if isinstance(self.k_or_percentile, float) and 0 < self.k_or_percentile <= 1:
            return self.k_or_percentile * 100
        if isinstance(self.k_or_percentile, int):
            return max(1.0, min(100.0, 100.0 * self.k_or_percentile / max(n_features, 1)))
        return 80.0

    def fit(self, X, y=None):
        X = _ensure_2d_numeric(X)
        n_features = X.shape[1]
        is_classification = self.task_type == "classification"

        if n_features <= 1 or y is None:
            self.selector_ = None
            self.support_ = np.ones(n_features, dtype=bool)
            return self

        if self.method == "wrapper":
            base_model = (
                RandomForestClassifier(n_estimators=100, random_state=self.random_state)
                if is_classification
                else RandomForestRegressor(n_estimators=100, random_state=self.random_state)
            )
            n_keep = self._n_keep(n_features)
            self.selector_ = RFE(base_model, n_features_to_select=n_keep)
            self.selector_.fit(X, y)
            self.support_ = self.selector_.get_support()
        elif self.method == "embedded":
            base_model = (
                RandomForestClassifier(n_estimators=200, random_state=self.random_state)
                if is_classification
                else RandomForestRegressor(n_estimators=200, random_state=self.random_state)
            )
            self.selector_ = SelectFromModel(
                base_model, threshold="median", max_features=self._n_keep(n_features)
            )
            self.selector_.fit(X, y)
            self.support_ = self.selector_.get_support()
        else:  # 'filter'
            score_func = f_classif if is_classification else f_regression
            self.selector_ = SelectPercentile(score_func, percentile=self._percentile(n_features))
            self.selector_.fit(X, y)
            self.support_ = self.selector_.get_support()

        if self.support_.sum() == 0:
            # Degenerate case (e.g. every feature scored identically) --
            # keep everything rather than emitting an empty matrix.
            self.support_ = np.ones(n_features, dtype=bool)
        return self

    def _n_keep(self, n_features: int) -> int:
        if isinstance(self.k_or_percentile, float) and 0 < self.k_or_percentile <= 1:
            return max(1, int(round(n_features * self.k_or_percentile)))
        if isinstance(self.k_or_percentile, int):
            return max(1, min(self.k_or_percentile, n_features))
        return max(1, int(round(n_features * 0.8)))

    def transform(self, X):
        X = _ensure_2d_numeric(X)
        return X[:, self.support_]

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return np.array([f"f{i}" for i in range(self.support_.sum())])
        input_features = np.asarray(input_features)
        return input_features[self.support_]


def _ensure_2d_numeric(X) -> np.ndarray:
    if isinstance(X, pd.DataFrame):
        return X.to_numpy(dtype=float, na_value=0.0)
    arr = np.asarray(X)
    return arr.astype(float)