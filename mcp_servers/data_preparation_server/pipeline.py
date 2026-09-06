"""
Assembles the specs produced by tools.py (imputer_spec, encoder_spec,
scaler_spec, feature_selection_spec, outlier_spec, derived_feature_spec)
into one real, UNFITTED sklearn Pipeline. Nothing in this file ever calls
.fit() -- that only happens in Modeling MCP (once per CV fold, and once
for real in finalize_model).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder, RobustScaler, StandardScaler

from . import feature_engineering
from .encoding import FrequencyEncoder, KFoldTargetEncoder
from .feature_engineering import FeatureSelector, apply_aggregation_transforms, apply_row_wise_transforms
from .outliers import compute_iqr_bounds
try:
    from ..shared import cache
except ImportError:
    from shared import cache


class DeriveFeaturesTransformer(BaseEstimator, TransformerMixin):
    """Pipeline step 1 ('derive'). Row-wise features (date parts, ratios)
    are stateless and applied identically in fit and transform. Aggregation
    features (group means/counts) are refit from whatever rows .fit() sees
    -- when Modeling MCP clones this pipeline and fits it once per CV fold,
    that means the aggregation lookup is recomputed from that fold's
    training rows only, which is what keeps it leakage-safe without any
    special-casing in Modeling MCP itself.
    """

    def __init__(self, feature_defs: Optional[list[dict]] = None):
        self.feature_defs = feature_defs or []

    def fit(self, X, y=None):
        agg_defs = [fd for fd in self.feature_defs if fd.get("expression_type") == "aggregation"]
        self.aggregation_lookups_ = apply_aggregation_transforms(X, agg_defs) if agg_defs else {}
        return self

    def transform(self, X):
        out = apply_row_wise_transforms(X, self.feature_defs)
        for name, lookup in self.aggregation_lookups_.items():
            group_cols = lookup["group_by"]
            table = lookup["table"]
            mapped = out[group_cols].apply(tuple, axis=1).map(table) if len(group_cols) > 1 else out[group_cols[0]].map(table)
            out[name] = mapped.fillna(lookup["global_fallback"])
        return out

    def get_feature_names_out(self, input_features=None):
        base = list(input_features) if input_features is not None else []
        derived = [fd["name"] for fd in self.feature_defs]
        return np.array(base + [d for d in derived if d not in base])


class OutlierHandlerTransformer(BaseEstimator, TransformerMixin):
    """Pipeline step 2 ('handle_outliers'). Bounds passed in from
    outlier_spec are used only to know *which* columns to touch -- the
    actual lower/upper bounds are always recomputed in .fit() from
    whatever rows are passed in, so a fresh per-fold fit never reuses
    bounds derived from the full train pool (the same leakage class as
    fitting a scaler before splitting)."""

    def __init__(self, strategy: str = "cap", columns: Optional[list[str]] = None):
        self.strategy = strategy
        self.columns = columns

    def fit(self, X, y=None):
        columns = self.columns if self.columns else list(X.select_dtypes(include="number").columns)
        self.columns_ = [c for c in columns if c in X.columns]
        self.bounds_ = {}
        for column in self.columns_:
            series = X[column].dropna()
            if series.empty:
                continue
            self.bounds_[column] = compute_iqr_bounds(series)
        return self

    def transform(self, X):
        out = X.copy()
        for column, (lower, upper) in self.bounds_.items():
            if self.strategy == "cap":
                out[column] = out[column].clip(lower=lower, upper=upper)
            elif self.strategy == "remove":
                out = out[(out[column].isna()) | ((out[column] >= lower) & (out[column] <= upper))]
            elif self.strategy == "transform":
                out[column] = np.sign(out[column]) * np.log1p(out[column].abs())
        return out

    def get_feature_names_out(self, input_features=None):
        return np.array(list(input_features)) if input_features is not None else None


def _make_numeric_imputer(strategy: str):
    if strategy == "mean":
        return SimpleImputer(strategy="mean")
    if strategy == "constant":
        return SimpleImputer(strategy="constant", fill_value=0.0)
    if strategy == "model_based":
        # NOTE: this ColumnTransformer wires one sub-pipeline per column,
        # so 'model_based' only sees its own column here rather than the
        # full numeric block an IterativeImputer would ideally use for
        # cross-column signal. KNNImputer degrades gracefully in that
        # single-column case; swap in a shared multi-column
        # IterativeImputer grouped by strategy if that matters for your
        # dataset.
        return KNNImputer(n_neighbors=5)
    if strategy == "drop_rows":
        # Rows are filtered upstream, before .fit() -- a fitted transformer
        # can't change row count. Fall back defensively so the pipeline
        # still fits if a caller wires this strategy in directly.
        return SimpleImputer(strategy="median")
    return SimpleImputer(strategy="median")


def _make_categorical_imputer(strategy: str):
    if strategy == "mode":
        return SimpleImputer(strategy="most_frequent")
    if strategy == "drop_rows":
        return SimpleImputer(strategy="most_frequent")
    return SimpleImputer(strategy="constant", fill_value="__missing__")


def _make_scaler(method: str):
    if method == "standard":
        return StandardScaler()
    if method == "minmax":
        return MinMaxScaler()
    if method == "robust":
        return RobustScaler()
    return None  # 'none' -> no scaling step for this column


def _make_encoder(strategy: str, target_encoder_config: dict):
    if strategy == "label":
        return OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    if strategy == "frequency":
        return FrequencyEncoder()
    if strategy == "target_encoding":
        return KFoldTargetEncoder(**target_encoder_config)
    return OneHotEncoder(handle_unknown="ignore", sparse_output=False)


def _build_preprocessor(imputer_spec: dict, encoder_spec: dict, scaler_spec: dict) -> ColumnTransformer:
    """One ColumnTransformer covering impute -> encode -> scale, built as
    a small per-column sub-Pipeline for every column: numeric columns get
    [impute, scale], categorical columns get [impute, encode]. This is the
    "one sklearn ColumnTransformer" the plan doc refers to -- impute/
    encode/scale are still three logical stages, just expressed as ordered
    steps inside each column's sub-pipeline rather than three separate
    top-level pipeline steps.
    """
    impute_strategies = (imputer_spec or {}).get("strategies", {})
    encode_strategies = (encoder_spec or {}).get("strategies", {})
    target_encoder_config = (encoder_spec or {}).get("target_encoder_config", {})
    scale_method = (scaler_spec or {}).get("method", "none")

    categorical_cols = list(encode_strategies.keys())
    numeric_cols = [c for c in impute_strategies.keys() if c not in categorical_cols]

    transformers = []
    for column in numeric_cols:
        steps = [("impute", _make_numeric_imputer(impute_strategies.get(column, "median")))]
        scaler = _make_scaler(scale_method)
        if scaler is not None:
            steps.append(("scale", scaler))
        transformers.append((f"num_{column}", Pipeline(steps), [column]))

    for column in categorical_cols:
        steps = [
            ("impute", _make_categorical_imputer(impute_strategies.get(column, "constant"))),
            ("encode", _make_encoder(encode_strategies.get(column, "one_hot"), target_encoder_config)),
        ]
        transformers.append((f"cat_{column}", Pipeline(steps), [column]))

    if not transformers:
        # No specs given yet (e.g. assembling a pipeline with only a
        # selector for a fully-numeric, already-clean dataset) -- pass
        # everything through unchanged rather than raising.
        return ColumnTransformer(transformers=[("passthrough", "passthrough", slice(0, None))], remainder="drop")

    return ColumnTransformer(transformers=transformers, remainder="drop", verbose_feature_names_out=False)


def _build_selector(feature_selection_spec: Optional[dict]):
    if not feature_selection_spec:
        return "passthrough"
    return FeatureSelector(
        method=feature_selection_spec.get("method", "filter"),
        k_or_percentile=feature_selection_spec.get("k_or_percentile", 0.8),
        task_type=feature_selection_spec.get("task_type", "classification"),
    )


def assemble_column_transformer(
    imputer_spec: dict,
    encoder_spec: dict,
    scaler_spec: dict,
    feature_selection_spec: dict,
    outlier_spec: Optional[dict] = None,
    derived_feature_spec: Optional[dict] = None,
) -> Pipeline:
    """Input: the spec dicts produced upstream by tools.py's
    handle_missing_values / encode_categorical / scale_numeric_features /
    select_features / handle_outliers -- each a plain recipe of strategy
    names and parameters, no fitted state. Processing: builds one sklearn
    ColumnTransformer wrapped in a Pipeline, wiring the steps together in a
    fixed order: derive -> handle_outliers -> impute -> encode -> scale ->
    select. Nothing is fit here -- this function only assembles structure.
    Output: an UNFITTED sklearn Pipeline object."""
    steps = []
    if derived_feature_spec and derived_feature_spec.get("feature_defs"):
        steps.append(("derive", DeriveFeaturesTransformer(derived_feature_spec["feature_defs"])))
    if outlier_spec and outlier_spec.get("bounds"):
        steps.append(
            ("handle_outliers", OutlierHandlerTransformer(
                strategy=outlier_spec.get("strategy", "cap"),
                columns=list(outlier_spec["bounds"].keys()),
            ))
        )
    steps.append(("preprocess", _build_preprocessor(imputer_spec, encoder_spec, scaler_spec)))
    steps.append(("select", _build_selector(feature_selection_spec)))
    return Pipeline(steps)


def save_pipeline_spec(spec: Pipeline, pipeline_spec_id: str) -> None:
    """Input: the unfitted Pipeline from assemble_column_transformer and
    its handle_id. Processing: joblib.dump (via shared.cache.save_object,
    which picks .joblib for anything that isn't a DataFrame). Output: none
    -- writes to _cache/specs/{pipeline_spec_id}.joblib, the shared path
    every server reads through shared.cache."""
    cache.save_object(spec, "specs", pipeline_spec_id)


def load_pipeline_spec(pipeline_spec_id: str) -> Pipeline:
    """Input: a pipeline_spec_id string. Processing: joblib.load from
    _cache/specs/ (via shared.cache.load_object). Output: the same
    unfitted, clonable Pipeline object that was saved. Callers (Modeling
    MCP's cv.py, finalize.py) are responsible for calling sklearn.clone()
    before ever calling .fit() -- this file never fits anything itself."""
    return cache.load_object("specs", pipeline_spec_id)