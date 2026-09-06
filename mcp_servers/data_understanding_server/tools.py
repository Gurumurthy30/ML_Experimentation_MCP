"""
Dataset understanding tools, exposed as MCP tools by server.py.

Workflow:
    1. Call `load_dataset(path)` once per file. It hashes the file's
       absolute path + mtime + size into a `dataset_id`, caches the
       data as Parquet under `_cache/datasets/{dataset_id}.parquet`,
       and returns that `dataset_id`.
    2. Every other function here takes that `dataset_id` (not the
       original file path) and reads the cached Parquet file via
       `cache.load_dataset_df`.

All returned dicts are built from native Python types (str/int/float/
list/dict) rather than numpy/pandas scalars, since the values need to
survive JSON serialization when returned over MCP.

NOTE ON THIS VERSION: docstrings below describe the CURRENT, AS-WRITTEN
behavior of each function, including known bugs and unimplemented
pieces, because these docstrings double as the prompt an LLM agent
reads before deciding how to call these tools. Where behavior doesn't
match what the function name implies, that's called out explicitly
rather than glossed over. See the "Known bug" / "Current behavior"
notes -- these are things to fix, not things to rely on.
"""

import math
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

from shared.cache import (
    compute_handle_id,
    save_object,
    load_object,
    object_exists,
)

from stats import(
    test_normality,
    compute_kurtosis,
    spearman_correlation,
    cramers_v,
    anova_f_eta_squared,
    mutual_information,
    auto_select_association_method
)


def _safe_float(value) -> Optional[float]:
    """Casts to float and converts NaN/Infinity to None. Plain JSON has
    no representation for NaN/Infinity, so leaving them in as float
    silently breaks structured-output serialization for the whole
    response -- better to surface them as an explicit null."""
    value = float(value)
    return value if math.isfinite(value) else None


def load_dataset(path: str, delimiter: Optional[str] = None, encoding: Optional[str] = None) -> dict:
    """dataset_id = compute_handle_id(abspath, mtime, size). On a cache
    miss, reads CSV (via delimiter/encoding, only used to override
    auto-detection) or Parquet based on file extension, and writes it to
    _cache/datasets/{id}.parquet. On a cache hit, skips straight to
    reading the cached Parquet. Calling this again on the same
    unmodified file is a cache hit; editing the file changes its mtime
    and size, so it gets a new id and is treated as a fresh dataset.
    Returns {dataset_id, n_rows, n_cols, dtypes}.
    """
    resolved = Path(path).resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"No such file: {resolved}")

    stat = resolved.stat()
    dataset_id = compute_handle_id(str(resolved), stat.st_mtime, stat.st_size)

    if not object_exists("datasets", dataset_id):
        if resolved.suffix.lower() == ".parquet":
            df = pd.read_parquet(resolved)
        else:
            read_kwargs = {}
            if delimiter is not None:
                read_kwargs["sep"] = delimiter
            if encoding is not None:
                read_kwargs["encoding"] = encoding
            df = pd.read_csv(resolved, **read_kwargs)
        save_object(df, "datasets", dataset_id)

    df = load_object("datasets", dataset_id)
    return dict(
        dataset_id=dataset_id,
        n_rows=int(df.shape[0]),
        n_cols=int(df.shape[1]),
        dtypes={col: str(dtype) for col, dtype in df.dtypes.items()},
    )


def infer_column_roles(dataset_id: str, mode: str = "permissive") -> dict:
    """Loads the cached DataFrame and returns basic per-column
    introspection: dtype, number of unique values, and up to 5 sample
    non-null values.

    Current behavior (as implemented):
    - Does NOT classify columns into semantic roles (numeric_continuous,
      numeric_discrete, categorical_nominal, categorical_ordinal,
      datetime, boolean, text_freeform, identifier, constant) despite
      the function name -- no `role` or `confidence` field is produced.
    - The `mode` parameter ('permissive'/'strict') is accepted but has
      no effect; there's no low-confidence-guess logic for it to toggle.

    Returns {column_name: {dtype, n_unique, sample_values}}.
    """
    df = load_object("datasets", dataset_id)
    inferred = {}
    for col in df.columns:
        inferred[col] = {
            "dtype": str(df[col].dtype),
            "n_unique": int(df[col].nunique()),
            "sample_values": df[col].dropna().head(5).tolist(),
        }
    return inferred


def profile_dataset(dataset_id: str, depth: str = "full") -> dict:
    """Per-column stats appropriate to its dtype. Every column always
    gets {dtype, missing_pct} at minimum, for both depths -- depth only
    controls how much *extra* detail is added on top, it never drops a
    column from the output.

    - Numeric: min/max/mean/std/skew always included. depth='full' adds
      kurtosis and a normality test (scipy.stats.normaltest, requires
      >=8 non-null values; noted separately if there aren't enough).
      depth='quick' skips both to keep the payload small.
    - Datetime: min/max and pandas-inferred frequency, both depths.
    - Categorical (object, category, or bool dtype -- this now also
      covers plain string columns from a CSV, not just pandas' explicit
      'category' dtype): cardinality always included; depth='full' adds
      top-10 values with counts.
    - Any float that would serialize as NaN/Infinity (e.g. std of a
      column with one non-null value) is reported as null instead,
      since raw NaN/Infinity breaks JSON output.

    Returns {column_name: {dtype, missing_pct, ...dtype-specific
    fields}}.
    """
    df = load_object("datasets", dataset_id)
    profile = {}

    for col in df.columns:
        col_data = df[col]
        col_profile = {
            "dtype": str(col_data.dtype),
            "missing_pct": _safe_float(col_data.isnull().mean() * 100),
        }
        non_null = col_data.dropna()

        if pd.api.types.is_numeric_dtype(col_data):
            col_profile.update({
                "min": _safe_float(col_data.min()),
                "max": _safe_float(col_data.max()),
                "mean": _safe_float(col_data.mean()),
                "std": _safe_float(col_data.std()),
                "skew": _safe_float(col_data.skew()),
            })
            if depth == "full":
                col_profile["kurtosis"] = _safe_float(compute_kurtosis(col_data))
                if len(non_null) >= 8:
                    normality = test_normality(col_data)
                    col_profile["is_normal"] = bool(normality["is_normal"])
                    col_profile["normality_stat"] = _safe_float(normality["statistic"])
                    col_profile["normality_p_value"] = _safe_float(normality["p_value"])
                else:
                    col_profile["normality_test_skipped_reason"] = (
                        f"needs >=8 non-null values, has {len(non_null)}"
                    )

        elif pd.api.types.is_datetime64_any_dtype(col_data):
            col_profile.update({
                "min": str(col_data.min()),
                "max": str(col_data.max()),
                "inferred_freq": pd.infer_freq(non_null),
            })

        elif (
            col_data.dtype == object
            or isinstance(col_data.dtype, pd.CategoricalDtype)
            or pd.api.types.is_bool_dtype(col_data)
        ):
            col_profile["cardinality"] = int(col_data.nunique())
            if depth == "full":
                top_values = col_data.value_counts().head(10)
                col_profile["top_values"] = {
                    str(k): int(v) for k, v in top_values.items()
                }

        profile[col] = col_profile
    return profile


def analyze_target_and_infer_task(dataset_id: str, goal_text: str, target_column: Optional[str] = None) -> dict:
    """Classifies task_type (classification vs regression) from the
    target column's dtype and cardinality.

    Current behavior (as implemented):
    - `target_column` is required in practice: the function does
      `df[target_column]` unconditionally, so passing None raises an
      error immediately. There is no logic to infer a target column
      from `goal_text` or from column roles -- `goal_text` is accepted
      but unused.
    - Classification rule: the target is treated as classification if
      it's non-numeric, OR if it has <=20 unique values AND is not a
      float dtype. NOTE: a low-cardinality float target (e.g. a binary
      0.0/1.0 label stored as float) will be routed to regression under
      this rule.
    - Does NOT return `confidence`, `reasoning`, or
      `needs_clarification`. Callers cannot currently tell a confident
      inference from a shaky one, and there's no path for flagging
      genuine ambiguity back to the caller instead of guessing.

    Returns, for classification: {task_type, class_balance}.
    Returns, for regression: {task_type, distribution_stats}.
    """
    df = load_object("datasets", dataset_id)
    target_data = df[target_column]

    is_classification = not pd.api.types.is_numeric_dtype(target_data) or (
        target_data.nunique() <= 20 and not pd.api.types.is_float_dtype(target_data)
    )

    if is_classification:
        class_balance = target_data.value_counts(normalize=True)
        return {
            "task_type": "classification",
            "class_balance": {str(k): float(v) for k, v in class_balance.items()},
        }

    return {
        "task_type": "regression",
        "distribution_stats": {
            "min": float(target_data.min()),
            "max": float(target_data.max()),
            "mean": float(target_data.mean()),
            "std": float(target_data.std()),
            "skew": float(target_data.skew()),
        },
    }


def analyze_missing_values(dataset_id: str) -> dict:
    """Per-column missing percentage, plus pairwise correlation between
    each column's missingness mask and every other column's not-null
    mask.

    Current behavior (as implemented):
    - Does NOT compute or return a `likely_mar` flag. It returns the raw
      correlation values against every other column, unthresholded --
      nothing is actually flagged as "likely missing-at-random" or
      otherwise. Anything downstream that branches on a `likely_mar` key
      (e.g. an 'auto' strategy in a missing-value-handling tool) will
      not find it here yet.

    Returns {column_name: {missing_pct, missingness_correlations?}},
    where missingness_correlations is only present for columns with
    missing_pct > 0, and maps every other column name to a correlation
    value (or None if undefined, e.g. a constant column).
    """
    df = load_object("datasets", dataset_id)
    missing_info = {}
    for col in df.columns:
        missing_pct = float(df[col].isnull().mean() * 100)
        missing_info[col] = {"missing_pct": missing_pct}
        if missing_pct > 0:
            missing_mask = df[col].isnull()
            correlations = {}
            for other_col in df.columns:
                if other_col != col:
                    corr = missing_mask.corr(df[other_col].notnull())
                    correlations[other_col] = float(corr) if pd.notnull(corr) else None
            missing_info[col]["missingness_correlations"] = correlations
    return missing_info


def detect_outliers(dataset_id: str, method: str = "iqr") -> dict:
    """IQR method (Q1 - 1.5*IQR, Q3 + 1.5*IQR) by default, 'zscore'
    (mean +/- 3*std) as the alternative. Only numeric columns are
    considered.

    Current behavior (as implemented):
    - The per-column result key is `example_indices` (NOT
      `example_row_indices`) -- callers reading the result must use
      `example_indices`.
    - Raises ValueError for any `method` other than 'iqr' or 'zscore'.

    Returns {column_name: {outlier_count, example_indices}}, where
    example_indices is capped at the first 5 matching row indices.
    """
    df = load_object("datasets", dataset_id)
    outlier_info = {}
    for col in df.select_dtypes(include=["number"]).columns:
        col_data = df[col].dropna()
        if method == "iqr":
            Q1 = col_data.quantile(0.25)
            Q3 = col_data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
        elif method == "zscore":
            mean = col_data.mean()
            std = col_data.std()
            lower_bound = mean - 3 * std
            upper_bound = mean + 3 * std
        else:
            raise ValueError(f"Unsupported method: {method}")

        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        outlier_info[col] = {
            "outlier_count": int(len(outliers)),
            "example_indices": outliers.index.tolist()[:5],
        }
    return outlier_info


def detect_target_leakage(dataset_id: str, target_column: str, outcome_time_column: Optional[str] = None, correlation_threshold: float = 0.98) -> dict:
    """Intended to be a heuristic, advisory-only leakage check (never
    auto-dropping columns): identifier-like columns, columns over a
    correlation threshold with the target, and -- if given -- features
    timestamped after outcome_time_column.

    Current behavior (as implemented) -- substantially not yet built:
    - Returns a flat {column: correlation_with_target} dict, not the
      documented {column, reason, evidence} structure.
    - No identifier-ratio check (n_unique/n_rows > 0.95) is performed.
    - `correlation_threshold` is accepted but never used -- nothing is
      actually filtered or flagged against it.
    - `outcome_time_column` is accepted but never used.
    - Non-numeric feature columns will produce NaN from
      `target.corr(...)` rather than being handled, encoded, or
      excluded with a reason.
    - KNOWN BUG (mutates shared state): `df.pop(target_column)` mutates
      whatever DataFrame object `load_object` returns. If that's a
      cached reference rather than a fresh copy, this permanently
      removes the target column from the cached dataset for this
      `dataset_id` -- affecting every subsequent call to this or any
      other function against the same `dataset_id`, not just this call.

    Returns {column_name: correlation_with_target_or_None}.
    """
    df = load_object("datasets", dataset_id)
    target = df.pop(target_column)
    return {col: target.corr(df[col]) for col in df.columns}


def measure_associations(dataset_id: str, column_a: str, column_b: str,
                          role_a: str, role_b: str, method: str = "auto",
                          weak_threshold: float = 0.1) -> dict:
    """Computes an association statistic between two columns of a cached
    dataset. If method='auto', picks the method via
    auto_select_association_method based on (role_a, role_b). If the
    primary statistic's effect size is below weak_threshold, escalates to
    mutual_information, since Spearman/Cramer's V/ANOVA-F only catch
    monotonic, categorical-association, or mean-shift relationships
    respectively.

    role_a/role_b are not inferred here -- infer_column_roles does not
    yet classify semantic roles, so the caller must pass "numeric" or
    "categorical" for each column explicitly.

    Returns {method, ...method-specific fields, strength, escalated_to_mi,
    mutual_information?}.
    """
    df = load_object("datasets", dataset_id)
    x = df[column_a]
    y = df[column_b]

    if method == "auto":
        method = auto_select_association_method(role_a, role_b)

    result = {"method": method}

    if method == "spearman":
        corr = spearman_correlation(x, y)
        rho = corr["rho"]
        result["rho"] = _safe_float(rho)
        result["p_value"] = _safe_float(corr["p_value"])
        strength = abs(rho) if rho is not None and not math.isnan(rho) else 0.0
    elif method == "cramers_v":
        v = cramers_v(x, y)
        result["cramers_v"] = _safe_float(v)
        strength = v if v is not None and not math.isnan(v) else 0.0
    elif method == "anova":
        categorical, numeric = (x, y) if role_a == "categorical" else (y, x)
        anova = anova_f_eta_squared(categorical, numeric)
        result["f_statistic"] = _safe_float(anova["f_statistic"])
        result["p_value"] = _safe_float(anova["p_value"])
        result["eta_squared"] = _safe_float(anova["eta_squared"])
        eta = anova["eta_squared"]
        strength = math.sqrt(eta) if eta is not None and not math.isnan(eta) else 0.0
    else:
        raise ValueError(f"Unknown method: {method}")

    result["strength"] = _safe_float(strength)
    result["escalated_to_mi"] = False

    if strength < weak_threshold:
        mi = mutual_information(x, y, role_a, role_b)
        result["mutual_information"] = _safe_float(mi)
        result["escalated_to_mi"] = True

    return result