"""
Encoding strategy registry + resolution logic, and the K-fold target
encoder used inside pipeline.py's ColumnTransformer.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold, StratifiedKFold

try:
    from ..shared.schemas import ColumnRole
except ImportError:
    from shared.schemas import ColumnRole

CATEGORICAL_ROLES = {
    ColumnRole.CATEGORICAL_NOMINAL.value,
    ColumnRole.CATEGORICAL_ORDINAL.value,
    ColumnRole.BOOLEAN.value,
}

ENCODE_STRATEGIES: dict[str, Any] = {
    "one_hot": "one_hot",
    "label": "label",
    "target_encoding": "target_encoding",
    "frequency": "frequency",
}
"""one_hot is the default for low-cardinality nominal columns,
target_encoding for high-cardinality ones. Values here are just canonical
names; pipeline.py maps each name to the actual transformer class."""


def build_encoder_spec(strategy: str, column_roles: dict, cardinalities: dict) -> dict:
    """strategy='auto' picks one_hot when a column's cardinality <= 15,
    target_encoding above that. Only columns whose role is one of
    CATEGORICAL_ROLES get an entry -- numeric/datetime/etc. columns are
    left for imputation/scaling instead."""
    per_column = {}
    for column, role in column_roles.items():
        role_value = role.value if hasattr(role, "value") else role
        if role_value not in CATEGORICAL_ROLES:
            continue
        if strategy in ("auto", None):
            cardinality = cardinalities.get(column, 0)
            per_column[column] = "one_hot" if cardinality <= 15 else "target_encoding"
        elif isinstance(strategy, dict):
            per_column[column] = strategy.get(column, strategy.get(role_value, "one_hot"))
        else:
            per_column[column] = strategy
    return {"strategies": per_column, "target_encoder_config": cv_safe_target_encoder_config()}


def cv_safe_target_encoder_config() -> dict:
    """Default smoothing/fold-count parameters for the K-fold out-of-fold
    target-encoding scheme -- this only configures the approach. The
    actual out-of-fold computation happens inside KFoldTargetEncoder.fit()
    below, using training rows only, never against the whole train pool
    naively -- which is what would leak target information into the
    encoding."""
    return {"smoothing": 10.0, "n_splits": 5, "random_state": 0}


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """Maps each category to its relative frequency in the fitted data.
    Unseen categories at transform time map to 0.0."""

    def fit(self, X, y=None):
        column = _as_series(X)
        counts = column.value_counts(normalize=True, dropna=False)
        self.freq_map_ = counts.to_dict()
        return self

    def transform(self, X):
        column = _as_series(X)
        mapped = column.map(self.freq_map_).fillna(0.0)
        return mapped.to_numpy().reshape(-1, 1)

    def get_feature_names_out(self, input_features=None):
        name = input_features[0] if input_features else "frequency"
        return np.array([f"{name}_freq"])


class KFoldTargetEncoder(BaseEstimator, TransformerMixin):
    """Out-of-fold target encoding, computed entirely inside .fit()/
    .fit_transform() so the *training* rows themselves never see a target
    mean derived from their own row. At transform() time (val/test rows,
    which were never part of fitting) it uses the global per-category mean
    learned on the full fitted column, smoothed toward the overall mean
    for low-count categories.

    This is exactly what encoding.py's docstring means by "the K-fold
    out-of-fold logic is configured here but executed per fold downstream"
    -- 'per fold' here means per CV fold in Modeling MCP, which clones and
    re-fits this transformer fresh on each fold's training rows only.
    """

    def __init__(self, smoothing: float = 10.0, n_splits: int = 5, random_state: int = 0):
        self.smoothing = smoothing
        self.n_splits = n_splits
        self.random_state = random_state

    def fit(self, X, y=None):
        if y is None:
            raise ValueError("KFoldTargetEncoder requires y at fit time")
        column = _as_series(X)
        target = pd.Series(np.asarray(y), index=column.index)
        self.global_mean_ = float(target.mean())
        stats = target.groupby(column).agg(["mean", "count"])
        self.category_means_ = (
            (stats["mean"] * stats["count"] + self.global_mean_ * self.smoothing)
            / (stats["count"] + self.smoothing)
        ).to_dict()
        return self

    def fit_transform(self, X, y=None):
        if y is None:
            raise ValueError("KFoldTargetEncoder requires y at fit time")
        column = _as_series(X)
        target = pd.Series(np.asarray(y), index=column.index)
        self.global_mean_ = float(target.mean())

        out = pd.Series(np.nan, index=column.index, dtype=float)
        n_splits = max(2, min(self.n_splits, column.nunique(dropna=False) if False else self.n_splits))
        splitter_cls = StratifiedKFold if _looks_discrete(target) else KFold
        try:
            splitter = splitter_cls(n_splits=n_splits, shuffle=True, random_state=self.random_state)
            split_iter = splitter.split(column, target) if splitter_cls is StratifiedKFold else splitter.split(column)
        except ValueError:
            splitter = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
            split_iter = splitter.split(column)

        for train_pos, holdout_pos in split_iter:
            train_idx = column.index[train_pos]
            holdout_idx = column.index[holdout_pos]
            fold_stats = target.loc[train_idx].groupby(column.loc[train_idx]).agg(["mean", "count"])
            fold_means = (
                (fold_stats["mean"] * fold_stats["count"] + self.global_mean_ * self.smoothing)
                / (fold_stats["count"] + self.smoothing)
            )
            out.loc[holdout_idx] = column.loc[holdout_idx].map(fold_means).fillna(self.global_mean_)

        # Fit the full-data mapping too, used later by transform() on
        # genuinely unseen (val/test) rows.
        full_stats = target.groupby(column).agg(["mean", "count"])
        self.category_means_ = (
            (full_stats["mean"] * full_stats["count"] + self.global_mean_ * self.smoothing)
            / (full_stats["count"] + self.smoothing)
        ).to_dict()
        return out.to_numpy().reshape(-1, 1)

    def transform(self, X):
        column = _as_series(X)
        mapped = column.map(self.category_means_).fillna(self.global_mean_)
        return mapped.to_numpy().reshape(-1, 1)

    def get_feature_names_out(self, input_features=None):
        name = input_features[0] if input_features else "target_enc"
        return np.array([f"{name}_target_enc"])


def _as_series(X) -> pd.Series:
    """ColumnTransformer hands sub-transformers a 2D slice (DataFrame or
    ndarray) for a single column; normalize that down to a 1D Series."""
    if isinstance(X, pd.DataFrame):
        return X.iloc[:, 0]
    if isinstance(X, pd.Series):
        return X
    arr = np.asarray(X)
    return pd.Series(arr.reshape(-1))


def _looks_discrete(y: pd.Series) -> bool:
    return y.nunique(dropna=True) <= max(20, int(0.05 * len(y)))