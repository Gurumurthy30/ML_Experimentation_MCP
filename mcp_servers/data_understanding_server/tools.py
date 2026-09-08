"""
Dataset understanding tools, exposed as MCP tools by server.py.

Workflow:
    1. Call `load_dataset(path)` once per file to cache it and get a `dataset_id`.
    2. Call diagnostic and profiling tools (`infer_column_roles`, `profile_dataset`,
       `analyze_target_and_infer_task`, `analyze_missing_values`, `detect_target_leakage`)
       passing `dataset_id`.
"""

import math
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

try:
    from ..shared.cache import (
        compute_handle_id,
        save_object,
        load_object,
        object_exists,
    )
except ImportError:
    from shared.cache import (
        compute_handle_id,
        save_object,
        load_object,
        object_exists,
    )

try:
    from .stats import (
        test_normality,
        compute_kurtosis,
        spearman_correlation,
        cramers_v,
        anova_f_eta_squared,
        mutual_information,
        auto_select_association_method,
    )
except ImportError:
    from stats import (
        test_normality,
        compute_kurtosis,
        spearman_correlation,
        cramers_v,
        anova_f_eta_squared,
        mutual_information,
        auto_select_association_method,
    )


def _safe_float(value) -> Optional[float]:
    """Casts to float and converts NaN/Infinity to None for clean JSON serialization."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def load_dataset(path: str, delimiter: Optional[str] = None, encoding: Optional[str] = None) -> dict:
    """Load a raw dataset file into the shared cache and return its metadata.

    PURPOSE & WHEN TO USE:
        Call this tool at the start of analysis to ingest a raw file (CSV or Parquet)
        from disk into the TabularML cache system. Hashes file path, mtime, and size
        to produce a unique dataset_id handle.

    INPUT PARAMETERS:
        - `path` (str): Absolute file path to the CSV or Parquet dataset.
        - `delimiter` (str, optional): Custom delimiter character for CSV (e.g. ',' or '\t').
        - `encoding` (str, optional): File text encoding (e.g. 'utf-8' or 'latin1').

    RETURNS:
        - Dict with keys:
            - `dataset_id` (str): Unique handle hash assigned to this dataset in cache.
            - `n_rows` (int): Total row count in dataset.
            - `n_cols` (int): Total column count in dataset.
            - `dtypes` (dict[str, str]): Mapping of column names to string pandas dtypes.

    AI AGENT GUIDELINES:
        - Use the returned `dataset_id` string as input for all subsequent diagnostic tools.
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
    """Classify each column into a semantic role for ML pipelines.

    PURPOSE & WHEN TO USE:
        Call this tool after loading a dataset to discover column semantics (numeric_continuous,
        numeric_discrete, categorical_nominal, boolean, datetime, text_freeform, identifier, constant).
        This drives downstream preprocessing decisions like imputation and encoding.

    INPUT PARAMETERS:
        - `dataset_id` (str): Cached dataset handle ID returned by `load_dataset`.
        - `mode` (str, default='permissive'): 'permissive' uses fallback heuristics; 'strict' assigns 'unknown' if low confidence.

    RETURNS:
        - Dict mapping column name to dict containing:
            - `role` (str): Inferred semantic role.
            - `confidence` (str): Confidence level ('high', 'medium', 'low').
            - `n_unique` (int): Number of distinct values.
            - `dtype` (str): Underlying pandas dtype.

    AI AGENT GUIDELINES:
        - Columns flagged as `identifier` or `constant` should be dropped before modeling.
    """
    df = load_object("datasets", dataset_id)
    n_rows = len(df)
    inferred = {}

    for col in df.columns:
        series = df[col]
        n_unique = int(series.nunique(dropna=True))
        n_missing = int(series.isnull().sum())
        n_non_null = n_rows - n_missing

        role = "unknown"
        confidence = "low"

        # Constant column
        if n_unique <= 1:
            role = "constant"
            confidence = "high"

        # Datetime
        elif pd.api.types.is_datetime64_any_dtype(series):
            role = "datetime"
            confidence = "high"

        # Try to parse as datetime (object columns)
        elif series.dtype == object and n_non_null > 0:
            sample = series.dropna().head(min(50, n_non_null))
            try:
                pd.to_datetime(sample, errors="raise")
                role = "datetime"
                confidence = "medium"
            except Exception:
                pass

        # Bool
        if role == "unknown":
            if pd.api.types.is_bool_dtype(series):
                role = "boolean"
                confidence = "high"
            elif n_unique == 2 and n_non_null > 0:
                vals = set(str(v).lower() for v in series.dropna().unique())
                bool_sets = [{"true", "false"}, {"yes", "no"}, {"0", "1"}, {"1", "2"}]
                if any(vals <= bs for bs in bool_sets):
                    role = "boolean"
                    confidence = "medium"

        # Identifier
        if role == "unknown" and n_non_null > 0:
            if n_unique / n_rows > 0.95:
                role = "identifier"
                confidence = "high"

        # Numeric
        if role == "unknown" and pd.api.types.is_numeric_dtype(series):
            unique_ratio = n_unique / max(n_non_null, 1)
            if n_unique > 20 or unique_ratio > 0.05:
                role = "numeric_continuous"
                confidence = "high"
            else:
                role = "numeric_discrete"
                confidence = "medium"

        # Categorical / text freeform
        if role == "unknown" and (series.dtype == object or isinstance(series.dtype, pd.CategoricalDtype)):
            avg_len = series.dropna().astype(str).str.len().mean() if n_non_null > 0 else 0
            if n_unique / max(n_non_null, 1) > 0.5 and avg_len > 30:
                role = "text_freeform"
                confidence = "medium"
            else:
                role = "categorical_nominal"
                confidence = "high" if n_unique <= 50 else "medium"

        # Fallback for strict mode
        if mode == "strict" and confidence == "low":
            role = "unknown"

        inferred[col] = {
            "role": role,
            "confidence": confidence,
            "n_unique": n_unique,
            "dtype": str(series.dtype),
        }

    return inferred


def profile_dataset(dataset_id: str, depth: str = "full") -> dict:
    """Generate per-column descriptive statistics and distribution metrics.

    PURPOSE & WHEN TO USE:
        Call this tool to inspect summary statistics (missing percentage, min/max, mean/std, skewness,
        cardinality, and normality test) across all columns in a dataset.

    INPUT PARAMETERS:
        - `dataset_id` (str): Cached dataset handle ID returned by `load_dataset`.
        - `depth` (str, default='full'): 'full' includes kurtosis, normality tests, and top values; 'quick' computes basic stats only.

    RETURNS:
        - Dict mapping column name to feature profile stats dict containing `dtype`, `missing_pct`, min/max/mean/std, skewness, etc.

    AI AGENT GUIDELINES:
        - Check `missing_pct` and `skew` to guide scaling and imputation choices.
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


def analyze_target_and_infer_task(
    dataset_id: str, goal_text: str, target_column: Optional[str] = None
) -> dict:
    """Analyze target column characteristics and determine task type (classification vs regression).

    PURPOSE & WHEN TO USE:
        Call this tool to infer whether the ML problem is 'classification' or 'regression'
        and check target distribution / class balance.

    INPUT PARAMETERS:
        - `dataset_id` (str): Cached dataset handle ID.
        - `goal_text` (str): User goal prompt description (used to find target column if unstated).
        - `target_column` (str, optional): Target column name. If None, auto-detected from goal_text or dataset.

    RETURNS:
        - Dict containing:
            - `task_type` (str): 'classification' or 'regression'.
            - `target_column` (str): Resolved target column name.
            - `class_balance` (dict, classification): Normalized label proportions.
            - `distribution_stats` (dict, regression): Summary min/max/mean/std/skew.
            - `confidence` (str): Task type inference confidence ('high', 'medium', 'low').
            - `reasoning` (str): Explanation for task type decision.
            - `needs_clarification` (bool): True if target type is ambiguous.
    """
    df = load_object("datasets", dataset_id)

    if target_column is None:
        goal_lower = goal_text.lower()
        candidates = [col for col in df.columns if col.lower() in goal_lower]
        if len(candidates) == 1:
            target_column = candidates[0]
        elif len(candidates) > 1:
            target_column = max(candidates, key=lambda c: goal_lower.rfind(c.lower()))
        else:
            target_column = df.columns[-1]

    target_data = df[target_column]
    n_unique = target_data.nunique(dropna=True)
    is_float = pd.api.types.is_float_dtype(target_data)
    is_numeric = pd.api.types.is_numeric_dtype(target_data)

    needs_clarification = False

    if not is_numeric:
        is_classification = True
        confidence = "high"
        reasoning = f"Column '{target_column}' has non-numeric dtype '{target_data.dtype}' -- classification."
    elif is_float and n_unique > 20:
        is_classification = False
        confidence = "high"
        reasoning = f"Column '{target_column}' is float with {n_unique} unique values -- regression."
    elif n_unique <= 2:
        is_classification = True
        confidence = "high" if not is_float else "medium"
        reasoning = f"Column '{target_column}' has {n_unique} unique values -- binary classification."
        if is_float:
            needs_clarification = True
    elif n_unique <= 20 and not is_float:
        is_classification = True
        confidence = "high"
        reasoning = f"Column '{target_column}' has {n_unique} unique values and integer dtype -- classification."
    elif n_unique <= 20 and is_float:
        is_classification = False
        confidence = "low"
        reasoning = (
            f"Column '{target_column}' is float with only {n_unique} unique values. "
            "Routing to regression; set task_type='classification' explicitly if this is a label."
        )
        needs_clarification = True
    else:
        is_classification = False
        confidence = "medium"
        reasoning = f"Column '{target_column}' is numeric with {n_unique} unique values -- regression."

    if is_classification:
        class_balance = target_data.value_counts(normalize=True)
        result = {
            "task_type": "classification",
            "target_column": target_column,
            "class_balance": {str(k): float(v) for k, v in class_balance.items()},
            "confidence": confidence,
            "reasoning": reasoning,
            "needs_clarification": needs_clarification,
        }
    else:
        result = {
            "task_type": "regression",
            "target_column": target_column,
            "distribution_stats": {
                "min": _safe_float(target_data.min()),
                "max": _safe_float(target_data.max()),
                "mean": _safe_float(target_data.mean()),
                "std": _safe_float(target_data.std()),
                "skew": _safe_float(target_data.skew()),
            },
            "confidence": confidence,
            "reasoning": reasoning,
            "needs_clarification": needs_clarification,
        }
    return result


def analyze_missing_values(dataset_id: str) -> dict:
    """Analyze per-column missingness and detect structured missingness (MAR).

    PURPOSE & WHEN TO USE:
        Call this tool to inspect missing value percentages across columns and check if missingness
        correlates with other features (likely_mar=True). This result dictates whether `handle_missing_values`
        uses simple median/mode imputation or model-based (KNN) imputation.

    INPUT PARAMETERS:
        - `dataset_id` (str): Cached dataset handle ID.

    RETURNS:
        - Dict mapping column names to:
            - `missing_pct` (float): Percentage of missing rows.
            - `likely_mar` (bool): True if missingness correlates with another feature at |r| >= 0.3.
            - `missingness_correlations` (dict, optional): Pairwise correlation between missingness mask and other features.
    """
    df = load_object("datasets", dataset_id)
    missing_info = {}
    MAR_THRESHOLD = 0.3

    for col in df.columns:
        missing_pct = float(df[col].isnull().mean() * 100)
        entry: dict = {"missing_pct": missing_pct, "likely_mar": False}

        if missing_pct > 0:
            missing_mask = df[col].isnull().astype(float)
            correlations = {}
            max_abs_corr = 0.0

            for other_col in df.columns:
                if other_col == col:
                    continue
                other = df[other_col]
                if pd.api.types.is_numeric_dtype(other):
                    corr = missing_mask.corr(other)
                elif other.dtype == object or isinstance(other.dtype, pd.CategoricalDtype):
                    corr = missing_mask.corr(other.astype("category").cat.codes.astype(float))
                else:
                    corr = float("nan")

                val = float(corr) if pd.notnull(corr) else None
                correlations[other_col] = val
                if val is not None:
                    max_abs_corr = max(max_abs_corr, abs(val))

            entry["likely_mar"] = max_abs_corr >= MAR_THRESHOLD
            entry["missingness_correlations"] = correlations

        missing_info[col] = entry
    return missing_info


def detect_outliers(dataset_id: str, method: str = "iqr") -> dict:
    """Detect outlier data points in numeric columns using IQR or Z-score bounds.

    PURPOSE & WHEN TO USE:
        Call this tool during data understanding to identify extreme values and guide decisions in `handle_outliers`.

    INPUT PARAMETERS:
        - `dataset_id` (str): Cached dataset handle ID.
        - `method` (str, default='iqr'): Outlier detection method ('iqr' or 'zscore').

    RETURNS:
        - Dict mapping numeric column names to dict with `outlier_count`, `example_indices`, `lower_bound`, `upper_bound`.
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
            if std == 0:
                outlier_info[col] = {"outlier_count": 0, "example_indices": []}
                continue
            lower_bound = mean - 3 * std
            upper_bound = mean + 3 * std
        else:
            raise ValueError(f"Unsupported method: {method}. Use 'iqr' or 'zscore'.")

        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        outlier_info[col] = {
            "outlier_count": int(len(outliers)),
            "example_indices": [int(i) for i in outliers.index.tolist()[:5]],
            "lower_bound": _safe_float(lower_bound),
            "upper_bound": _safe_float(upper_bound),
        }
    return outlier_info


def detect_target_leakage(
    dataset_id: str,
    target_column: str,
    outcome_time_column: Optional[str] = None,
    correlation_threshold: float = 0.98,
) -> dict:
    """Detect potential data leakage features before splitting or training.

    PURPOSE & WHEN TO USE:
        Call this tool to audit features for target leakage (identifier-like columns, near-perfect
        correlation with target, or timestamps occurring after the outcome).

    INPUT PARAMETERS:
        - `dataset_id` (str): Cached dataset handle ID.
        - `target_column` (str): Target column name.
        - `outcome_time_column` (str, optional): Timestamp column marking outcome occurrence.
        - `correlation_threshold` (float, default=0.98): Pearson correlation threshold for flagging leakage.

    RETURNS:
        - Dict containing:
            - `leakage_suspects` (list[dict]): List of flagged columns with `column`, `reason`, `evidence`.
            - `checked_columns` (int): Number of feature columns inspected.

    AI AGENT GUIDELINES:
        - Review `leakage_suspects` and drop flagged features before splitting data.
    """
    df = load_object("datasets", dataset_id)
    n_rows = len(df)
    suspects = []

    target = df[target_column]
    feature_cols = [c for c in df.columns if c != target_column]

    for col in feature_cols:
        series = df[col]
        reasons = []
        evidence = {}

        # Identifier-like check
        n_unique = series.nunique(dropna=True)
        id_ratio = n_unique / max(n_rows, 1)
        if id_ratio > 0.95:
            reasons.append("identifier_like")
            evidence["unique_ratio"] = round(id_ratio, 4)

        # High correlation with target (numeric only)
        if pd.api.types.is_numeric_dtype(series) and pd.api.types.is_numeric_dtype(target):
            try:
                corr = float(series.corr(target))
                if pd.notnull(corr) and abs(corr) >= correlation_threshold:
                    reasons.append("high_correlation")
                    evidence["pearson_corr"] = round(corr, 4)
            except Exception:
                pass

        # Time-based leakage
        if outcome_time_column and outcome_time_column in df.columns:
            if pd.api.types.is_datetime64_any_dtype(series) or (series.dtype == object):
                try:
                    feat_times = pd.to_datetime(series, errors="coerce")
                    outcome_times = pd.to_datetime(df[outcome_time_column], errors="coerce")
                    after_outcome = (feat_times > outcome_times).mean()
                    if after_outcome > 0.5:
                        reasons.append("timestamped_after_outcome")
                        evidence["pct_after_outcome"] = round(float(after_outcome), 4)
                except Exception:
                    pass

        if reasons:
            suspects.append({
                "column": col,
                "reason": reasons,
                "evidence": evidence,
            })

    return {"leakage_suspects": suspects, "checked_columns": len(feature_cols)}


def measure_associations(
    dataset_id: str,
    column_a: str,
    column_b: str,
    role_a: str,
    role_b: str,
    method: str = "auto",
    weak_threshold: float = 0.1,
) -> dict:
    """Compute statistical association metrics between two columns.

    PURPOSE & WHEN TO USE:
        Call this tool to measure feature-feature or feature-target relationships using Spearman correlation,
        Cramér's V, or ANOVA F-test / Eta-squared, with fallback to Mutual Information.

    INPUT PARAMETERS:
        - `dataset_id` (str): Cached dataset handle ID.
        - `column_a` (str): First column name.
        - `column_b` (str): Second column name.
        - `role_a` (str): Role of first column ('numeric' or 'categorical').
        - `role_b` (str): Role of second column ('numeric' or 'categorical').
        - `method` (str, default='auto'): Association method ('spearman', 'cramers_v', 'anova', or 'auto').
        - `weak_threshold` (float, default=0.1): Threshold below which analysis escalates to Mutual Information.

    RETURNS:
        - Dict with `method`, `strength`, `escalated_to_mi`, and method-specific metrics.
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