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
"""

import pandas as pd
from cache import (
    compute_dataset_id,
    dataset_parquet_path,
    save_dataset,
    load_dataset_df,
)


def load_dataset(path: str) -> dict:
    """Ingest a CSV/Parquet file, caching it for all later tool calls.

    Computes a deterministic `dataset_id` from the file's absolute
    path, modification time, and size (see
    `cache.compute_dataset_id`). If a cached Parquet copy for that ID
    doesn't already exist, the file is read with pandas and written
    to the cache. Every other function in this module takes the
    returned `dataset_id`, not the original file path.

    Args:
        path: Path to a `.csv` or `.parquet` file on disk.

    Returns:
        {
            "dataset_id": str,   # pass this to every other tool
            "n_rows": int,
            "n_cols": int,
            "dtypes": {column_name: dtype_as_string, ...},
        }

    Raises:
        FileNotFoundError: if `path` does not exist.
        ValueError: if `path` doesn't end in `.csv` or `.parquet`.
    """
    dataset_id = compute_dataset_id(path)
    cache_path = dataset_parquet_path(dataset_id)

    if cache_path.exists():
        df = load_dataset_df(dataset_id)
    else:
        if path.endswith(".csv"):
            df = pd.read_csv(path)
        elif path.endswith(".parquet"):
            df = pd.read_parquet(path)
        else:
            raise ValueError(f"Unsupported file format: {path}")
        save_dataset(df, dataset_id)

    return dict(
        dataset_id=dataset_id,
        n_rows=int(df.shape[0]),
        n_cols=int(df.shape[1]),
        dtypes={col: str(dtype) for col, dtype in df.dtypes.items()},
    )


def infer_column_eoles(dataset_id: str) -> dict:
    """Summarize each column's dtype, cardinality, and sample values.

    Args:
        dataset_id: ID returned by `load_dataset`.

    Returns:
        dict mapping column name -> {
            "dtype": str,
            "n_unique": int,          # distinct non-null values
            "sample_values": list,    # up to 5 non-null values
        }

    Raises:
        FileNotFoundError: if `dataset_id` hasn't been cached yet
            (call `load_dataset` first).
    """
    df = load_dataset_df(dataset_id)
    inferred = {}
    for col in df.columns:
        inferred[col] = {
            "dtype": str(df[col].dtype),
            "n_unique": int(df[col].nunique()),
            "sample_values": df[col].dropna().head(5).tolist(),
        }
    return inferred


def profile_dataset(dataset_id: str) -> dict:
    """Build a compact statistical profile of every column.

    Numeric columns get min/max/mean/std/skew (skew via pandas'
    `Series.skew`, a bias-corrected estimate equivalent to
    `scipy.stats.skew(..., bias=False)`). Non-numeric columns get
    their 10 most frequent values with counts. Every column reports
    its missing-value percentage.

    Kept intentionally compact -- this is the payload a planning step
    reasons over, not a full data dump.

    Args:
        dataset_id: ID returned by `load_dataset`.

    Returns:
        dict mapping column name -> a profile dict. Numeric columns:
        {dtype, missing_pct, min, max, mean, std, skew}. Non-numeric
        columns: {dtype, missing_pct, top_values} where top_values
        maps value -> count for up to 10 values.

    Raises:
        FileNotFoundError: if `dataset_id` hasn't been cached yet.
    """
    df = load_dataset_df(dataset_id)
    profile = {}
    for col in df.columns:
        col_data = df[col]
        col_profile = {
            "dtype": str(col_data.dtype),
            "missing_pct": float(col_data.isnull().mean() * 100),
        }
        if pd.api.types.is_numeric_dtype(col_data):
            col_profile.update({
                "min": float(col_data.min()),
                "max": float(col_data.max()),
                "mean": float(col_data.mean()),
                "std": float(col_data.std()),
                "skew": float(col_data.skew()),
            })
        else:
            top_values = col_data.value_counts().head(10)
            col_profile["top_values"] = {
                str(k): int(v) for k, v in top_values.items()
            }
        profile[col] = col_profile
    return profile


def analyze_target_and_infer_task_type(dataset_id: str, target_column: str) -> dict:
    """Infer classification vs. regression for a target column.

    Treated as classification if the column is non-numeric (e.g. a
    string/category label), or if it's numeric with at most 20
    distinct values and isn't a float dtype. Otherwise treated as
    regression.

    Args:
        dataset_id: ID returned by `load_dataset`.
        target_column: Name of the column to analyze.

    Returns:
        Classification: {"task_type": "classification",
            "class_balance": {label: proportion, ...}}
        Regression: {"task_type": "regression",
            "distribution_stats": {min, max, mean, std, skew}}

    Raises:
        FileNotFoundError: if `dataset_id` hasn't been cached yet.
        KeyError: if `target_column` isn't in the dataset.
    """
    df = load_dataset_df(dataset_id)
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
    """Report missing-value rates and flag likely non-random missingness.

    For every column with at least one missing value, computes the
    correlation between "value is missing in this column" and "value
    is present in each other column". A strong correlation hints the
    missingness isn't purely random (MCAR) -- e.g. a 'TotalCharges'
    column that's null exactly when 'tenure' == 0 will show up as a
    high correlation between the two.

    Args:
        dataset_id: ID returned by `load_dataset`.

    Returns:
        dict mapping column name -> {
            "missing_pct": float,
            # present only when missing_pct > 0:
            "missingness_correlations": {other_col: float | None, ...}
        }
        A `None` correlation means it was undefined (e.g. a constant
        column), not that there's no relationship.

    Raises:
        FileNotFoundError: if `dataset_id` hasn't been cached yet.
    """
    df = load_dataset_df(dataset_id)
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
    """Flag numeric outliers in every numeric column.

    "iqr" (default): values outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR].
    "zscore": values more than 3 standard deviations from the mean.

    Args:
        dataset_id: ID returned by `load_dataset`.
        method: "iqr" or "zscore".

    Returns:
        dict mapping numeric column name -> {
            "outlier_count": int,
            "example_indices": list,  # up to 5 row indices
        }

    Raises:
        FileNotFoundError: if `dataset_id` hasn't been cached yet.
        ValueError: if `method` isn't "iqr" or "zscore".
    """
    df = load_dataset_df(dataset_id)
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


def detect_correlations(dataset_id: str, threshold: float = 0.8) -> dict:
    """Find pairs of numeric columns correlated above a threshold.

    Args:
        dataset_id: ID returned by `load_dataset`.
        threshold: Minimum absolute Pearson correlation to report.

    Returns:
        dict mapping column name -> list of (other_column, correlation)
        pairs whose absolute correlation exceeds `threshold`. Each
        pair appears twice, once under each column's key.

    Raises:
        FileNotFoundError: if `dataset_id` hasn't been cached yet.
    """
    df = load_dataset_df(dataset_id)
    numeric_df = df.select_dtypes(include=["number"])
    corr_matrix = numeric_df.corr()
    correlated_pairs = {}
    for col in corr_matrix.columns:
        for idx in corr_matrix.index:
            if col != idx and abs(corr_matrix.loc[idx, col]) > threshold:
                correlated_pairs.setdefault(col, []).append(
                    (idx, float(corr_matrix.loc[idx, col]))
                )
    return correlated_pairs


def detect_multicollinearity(dataset_id: str, threshold: float = 0.8) -> dict:
    """Find numeric columns involved in any high-correlation pair.

    A flat-list companion to `detect_correlations` -- useful as a
    quick "which columns are candidates to drop" check, without the
    pair-level detail.

    Args:
        dataset_id: ID returned by `load_dataset`.
        threshold: Minimum absolute Pearson correlation for two
            columns to count as multicollinear.

    Returns:
        {"multicollinear_columns": [column names appearing in at
         least one pair with |correlation| > threshold]}

    Raises:
        FileNotFoundError: if `dataset_id` hasn't been cached yet.
    """
    df = load_dataset_df(dataset_id)
    numeric_df = df.select_dtypes(include=["number"])
    corr_matrix = numeric_df.corr()
    multicollinear_cols = set()
    for col in corr_matrix.columns:
        for idx in corr_matrix.index:
            if col != idx and abs(corr_matrix.loc[idx, col]) > threshold:
                multicollinear_cols.add(col)
                multicollinear_cols.add(idx)
    return {"multicollinear_columns": list(multicollinear_cols)}