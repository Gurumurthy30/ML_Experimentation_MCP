"""
Data Preparation MCP tools.

All tools here operate on `split_id` handles (after the leakage firewall
that split_dataset erects) and produce *spec* dicts or a pipeline_spec_id
-- never fitted objects. The actual sklearn .fit() calls happen inside
Modeling MCP (once per CV fold and once for real in finalize_model).
"""

from __future__ import annotations

import json
from typing import Optional

import pandas as pd

try:
    from ..shared.cache import (
        compute_handle_id,
        get_cache_dir,
        load_object,
        object_exists,
        save_object,
    )
    from ..shared.schemas import ColumnRole
except ImportError:
    from shared.cache import (
        compute_handle_id,
        get_cache_dir,
        load_object,
        object_exists,
        save_object,
    )
    from shared.schemas import ColumnRole

try:
    from .imputation import build_imputer_spec
    from .encoding import build_encoder_spec
    from .scaling import build_scaler_spec
    from .outliers import build_outlier_spec, compute_iqr_bounds
    from .feature_engineering import build_feature_selection_spec
    from .pipeline import assemble_column_transformer, save_pipeline_spec
    from . import splitting
except ImportError:
    from imputation import build_imputer_spec
    from encoding import build_encoder_spec
    from scaling import build_scaler_spec
    from outliers import build_outlier_spec, compute_iqr_bounds
    from feature_engineering import build_feature_selection_spec
    from pipeline import assemble_column_transformer, save_pipeline_spec
    import splitting


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_train(split_id: str) -> pd.DataFrame:
    """Load only the train partition for a given split_id."""
    cache_dir = get_cache_dir()
    split_dir = cache_dir / "splits" / split_id
    train_path = split_dir / "train.parquet"
    if not train_path.exists():
        raise FileNotFoundError(f"Train partition not found for split_id={split_id!r}")
    return pd.read_parquet(train_path)


def _load_test(split_id: str) -> pd.DataFrame:
    """Load the test partition for a given split_id (only called by finalize)."""
    cache_dir = get_cache_dir()
    split_dir = cache_dir / "splits" / split_id
    test_path = split_dir / "test.parquet"
    if not test_path.exists():
        raise FileNotFoundError(f"Test partition not found for split_id={split_id!r}")
    return pd.read_parquet(test_path)


def _save_split(split_id: str, train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    cache_dir = get_cache_dir()
    split_dir = cache_dir / "splits" / split_id
    split_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_parquet(split_dir / "train.parquet", index=False)
    test_df.to_parquet(split_dir / "test.parquet", index=False)


def _load_split_meta(split_id: str) -> dict:
    cache_dir = get_cache_dir()
    meta_path = cache_dir / "splits" / split_id / "meta.json"
    if not meta_path.exists():
        return {}
    with open(meta_path) as f:
        return json.load(f)


def _save_split_meta(split_id: str, meta: dict) -> None:
    cache_dir = get_cache_dir()
    split_dir = cache_dir / "splits" / split_id
    split_dir.mkdir(parents=True, exist_ok=True)
    with open(split_dir / "meta.json", "w") as f:
        json.dump(meta, f)


def _infer_column_roles_from_train(train_df: pd.DataFrame, target_column: str) -> dict:
    """Lightweight role inference for use inside data preparation tools."""
    roles = {}
    n_rows = len(train_df)
    for col in train_df.columns:
        if col == target_column:
            continue
        series = train_df[col]
        n_unique = series.nunique(dropna=True)
        if pd.api.types.is_numeric_dtype(series):
            if n_unique > 20 or n_unique / max(n_rows, 1) > 0.05:
                roles[col] = ColumnRole.NUMERIC_CONTINUOUS.value
            else:
                roles[col] = ColumnRole.NUMERIC_DISCRETE.value
        elif pd.api.types.is_datetime64_any_dtype(series):
            roles[col] = ColumnRole.DATETIME.value
        elif pd.api.types.is_bool_dtype(series):
            roles[col] = ColumnRole.BOOLEAN.value
        elif series.dtype == object or isinstance(series.dtype, pd.CategoricalDtype):
            if n_unique / max(n_rows, 1) > 0.95:
                roles[col] = ColumnRole.IDENTIFIER.value
            else:
                roles[col] = ColumnRole.CATEGORICAL_NOMINAL.value
        else:
            roles[col] = ColumnRole.UNKNOWN.value if hasattr(ColumnRole, "UNKNOWN") else "unknown"
    return roles


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

def split_dataset(
    dataset_id: str,
    target_column: str,
    task_type: str,
    split_strategy: str = "auto",
    group_column: Optional[str] = None,
    time_column: Optional[str] = None,
    test_size: float = 0.2,
) -> dict:
    """Erect the data leakage firewall by partitioning raw dataset into train and reserved test sets.

    PURPOSE & WHEN TO USE:
        Call this tool immediately after dataset diagnostics to create train.parquet and test.parquet.
        This is the single most critical leakage firewall step -- test.parquet is saved away and NEVER
        touched again until final evaluation.

    INPUT PARAMETERS:
        - `dataset_id` (str): Cached dataset handle ID returned by `load_dataset`.
        - `target_column` (str): Target column name.
        - `task_type` (str): Task type ('classification' or 'regression').
        - `split_strategy` (str, default='auto'): Splitting strategy ('auto', 'random', 'stratified', 'group', 'time_series').
        - `group_column` (str, optional): Group ID column name for group splitting.
        - `time_column` (str, optional): Date/time column name for temporal splitting.
        - `test_size` (float, default=0.2): Proportion of data held out for final test evaluation (0.0 < test_size < 1.0).

    RETURNS:
        - Dict with keys:
            - `split_id` (str): Unique handle hash assigned to this data split.
            - `resolved_strategy` (str): The actual split strategy selected ('stratified', 'time_series', etc.).
            - `n_train` (int): Number of training partition rows.
            - `n_test` (int): Number of reserved test partition rows.

    AI AGENT GUIDELINES:
        - Pass `split_id` to all downstream data preparation and modeling tools.
    """
    df = load_object("datasets", dataset_id)

    if split_strategy == "auto":
        resolved_strategy = splitting.auto_detect_split_strategy(
            df, target_column, group_column, time_column
        )
    else:
        resolved_strategy = split_strategy

    RANDOM_STATE = 42
    if resolved_strategy == "time_series":
        train_df, test_df = splitting.time_series_split(df, time_column, test_size)
    elif resolved_strategy == "group":
        train_df, test_df = splitting.group_split(df, group_column, test_size, RANDOM_STATE)
    elif resolved_strategy == "stratified":
        train_df, test_df = splitting.stratified_split(df, target_column, test_size, RANDOM_STATE)
    else:
        train_df, test_df = splitting.random_split(df, test_size, RANDOM_STATE)

    split_id = compute_handle_id(dataset_id, target_column, resolved_strategy, test_size)

    if not (get_cache_dir() / "splits" / split_id / "train.parquet").exists():
        _save_split(split_id, train_df, test_df)
        _save_split_meta(split_id, {
            "dataset_id": dataset_id,
            "target_column": target_column,
            "task_type": task_type,
            "resolved_strategy": resolved_strategy,
            "group_column": group_column,
            "time_column": time_column,
            "test_size": test_size,
            "n_train": len(train_df),
            "n_test": len(test_df),
        })

    return {
        "split_id": split_id,
        "resolved_strategy": resolved_strategy,
        "n_train": len(train_df),
        "n_test": len(test_df),
    }


def generate_cv_folds(split_id: str, n_folds: str | int = "auto") -> dict:
    """Generate cross-validation fold row-indices from the training partition.

    PURPOSE & WHEN TO USE:
        Call this tool after `split_dataset` to generate leakage-safe CV fold index lists.
        Fold indices are stored as JSON metadata and inherited from the split strategy.

    INPUT PARAMETERS:
        - `split_id` (str): Active split handle ID.
        - `n_folds` (int or 'auto', default='auto'): Number of CV folds. 'auto' computes adaptive count based on dataset size.

    RETURNS:
        - Dict with `folds_id`, `n_folds`, and `strategy`.

    AI AGENT GUIDELINES:
        - Pass `folds_id` to `establish_baseline`, `cross_validate_model`, and `regularization_path_search`.
    """
    train_df = _load_train(split_id)
    meta = _load_split_meta(split_id)
    resolved_strategy = meta.get("resolved_strategy", "random")
    group_column = meta.get("group_column")
    time_column = meta.get("time_column")
    target_column = meta.get("target_column")

    if n_folds == "auto":
        n_folds = splitting.compute_adaptive_fold_count(len(train_df))

    folds = splitting.generate_folds(
        train_df,
        strategy=resolved_strategy,
        n_folds=n_folds,
        group_column=group_column,
        time_column=time_column,
        target_column=target_column,
    )

    folds_id = compute_handle_id(split_id, n_folds, resolved_strategy)
    cache_dir = get_cache_dir()
    folds_path = cache_dir / "folds" / f"{folds_id}.json"
    if not folds_path.exists():
        folds_path.parent.mkdir(parents=True, exist_ok=True)
        with open(folds_path, "w") as f:
            json.dump({"folds": folds, "split_id": split_id, "n_folds": n_folds, "strategy": resolved_strategy}, f)

    return {
        "folds_id": folds_id,
        "n_folds": n_folds,
        "strategy": resolved_strategy,
    }


def validate_split_quality(split_id: str, force: bool = False) -> dict:
    """Validate train vs test distribution alignment to detect covariate shift.

    PURPOSE & WHEN TO USE:
        Call this tool on small datasets (or with force=True) to check for severe distribution shifts between train and test splits using KS / Chi-square tests.

    INPUT PARAMETERS:
        - `split_id` (str): Active split handle ID.
        - `force` (bool, default=False): If True, forces distribution testing even on large datasets (>1000 rows).

    RETURNS:
        - Dict containing `skipped` flag, `columns_tested`, `flagged_columns`, and test details per column.
    """
    train_df = _load_train(split_id)
    meta = _load_split_meta(split_id)

    if splitting.should_skip_shift_check(len(train_df), force):
        return {
            "skipped": True,
            "reason": f"Train has {len(train_df)} rows (> 1000); set force=True to run anyway.",
        }

    test_df = _load_test(split_id)
    target_column = meta.get("target_column", "")
    results = {}

    for col in train_df.columns:
        if col == target_column:
            continue
        train_col = train_df[col].dropna()
        test_col = test_df[col].dropna()

        if len(train_col) == 0 or len(test_col) == 0:
            results[col] = {"p_value": None, "flagged": False, "reason": "empty column"}
            continue

        if pd.api.types.is_numeric_dtype(train_col):
            stat, p_value = splitting.ks_test_numeric(train_col, test_col)
            results[col] = {
                "p_value": round(float(p_value), 4),
                "statistic": round(float(stat), 4),
                "flagged": p_value < 0.05,
                "test": "ks",
            }
        else:
            stat, p_value = splitting.chi_square_categorical(train_col, test_col)
            results[col] = {
                "p_value": round(float(p_value), 4),
                "statistic": round(float(stat), 4),
                "flagged": p_value < 0.05,
                "test": "chi_square",
            }

    flagged = [col for col, r in results.items() if r.get("flagged")]
    return {
        "skipped": False,
        "columns_tested": len(results),
        "flagged_columns": flagged,
        "results": results,
    }


def derive_features(split_id: str, feature_defs: list[dict]) -> dict:
    """Define row-wise or aggregation feature engineering definitions.

    PURPOSE & WHEN TO USE:
        Call this tool to specify new feature definitions (e.g. date extractions, mathematical combinations, group aggregations). Returns a spec recipe without mutating raw data.

    INPUT PARAMETERS:
        - `split_id` (str): Active split handle ID.
        - `feature_defs` (list[dict]): List of dicts specifying `name`, `expression_type` ('row_wise' or 'aggregation'), and `definition`.

    RETURNS:
        - Dict with `derived_feature_spec`.
    """
    if not feature_defs:
        return {"derived_feature_spec": {"feature_defs": []}}

    row_wise = [fd for fd in feature_defs if fd.get("expression_type") == "row_wise"]
    aggregation = [fd for fd in feature_defs if fd.get("expression_type") == "aggregation"]
    other = [fd for fd in feature_defs if fd.get("expression_type") not in ("row_wise", "aggregation")]

    if other:
        raise ValueError(
            f"Unsupported expression_type for: {[fd['name'] for fd in other]}. "
            "Must be 'row_wise' or 'aggregation'."
        )

    return {
        "derived_feature_spec": {
            "feature_defs": feature_defs,
            "n_row_wise": len(row_wise),
            "n_aggregation": len(aggregation),
        }
    }


def handle_missing_values(split_id: str, strategy_map: dict | str = "auto") -> dict:
    """Define missing value imputation strategy per column.

    PURPOSE & WHEN TO USE:
        Call this tool to resolve missingness strategies ('median', 'mean', 'mode', 'constant', 'model_based').
        When strategy_map='auto', uses column roles and MAR flags to pick median/mode for MCAR or KNN for MAR.

    INPUT PARAMETERS:
        - `split_id` (str): Active split handle ID.
        - `strategy_map` (dict or 'auto', default='auto'): Either 'auto' or explicit dict mapping column names to strategy strings.

    RETURNS:
        - Dict containing `imputer_spec`.
    """
    train_df = _load_train(split_id)
    meta = _load_split_meta(split_id)
    target_column = meta.get("target_column", "")

    feature_df = train_df.drop(columns=[target_column], errors="ignore")
    column_roles = _infer_column_roles_from_train(train_df, target_column)

    mar_flags = {}
    for col in feature_df.columns:
        missing_pct = float(feature_df[col].isnull().mean() * 100)
        likely_mar = False
        if missing_pct > 0:
            mask = feature_df[col].isnull().astype(float)
            for other_col in feature_df.columns:
                if other_col == col:
                    continue
                other = feature_df[other_col]
                if pd.api.types.is_numeric_dtype(other):
                    try:
                        corr = abs(float(mask.corr(other)))
                        if corr >= 0.3:
                            likely_mar = True
                            break
                    except Exception:
                        pass
        mar_flags[col] = {"missing_pct": missing_pct, "likely_mar": likely_mar}

    imputer_spec = build_imputer_spec(strategy_map, column_roles, mar_flags)
    return {"imputer_spec": imputer_spec}


def encode_categorical(
    split_id: str,
    target_column: str,
    strategy: str = "auto",
) -> dict:
    """Define categorical encoding strategies for non-numeric features.

    PURPOSE & WHEN TO USE:
        Call this tool to select categorical encoding methods ('one_hot', 'label', 'target_encoding', 'frequency', 'auto').

    INPUT PARAMETERS:
        - `split_id` (str): Active split handle ID.
        - `target_column` (str): Target column name.
        - `strategy` (str, default='auto'): Strategy name. 'auto' chooses one-hot for cardinality <= 15 and target encoding above.

    RETURNS:
        - Dict containing `encoder_spec`.
    """
    train_df = _load_train(split_id)

    column_roles = _infer_column_roles_from_train(train_df, target_column)
    cardinalities = {
        col: int(train_df[col].nunique(dropna=True))
        for col in train_df.columns
        if col != target_column
    }

    encoder_spec = build_encoder_spec(strategy, column_roles, cardinalities)
    return {"encoder_spec": encoder_spec}


def scale_numeric_features(split_id: str, method: str = "standard") -> dict:
    """Define scaling strategy for continuous numeric features.

    PURPOSE & WHEN TO USE:
        Call this tool to configure numeric feature scaling ('standard', 'minmax', 'robust', 'none').

    INPUT PARAMETERS:
        - `split_id` (str): Active split handle ID.
        - `method` (str, default='standard'): Scaling method name. Use 'none' when training tree-based models.

    RETURNS:
        - Dict containing `scaler_spec`.
    """
    scaler_spec = build_scaler_spec(method)
    return {"scaler_spec": scaler_spec}


def handle_outliers(split_id: str, strategy: str = "cap") -> dict:
    """Define outlier handling bounds for numeric columns.

    PURPOSE & WHEN TO USE:
        Call this tool to configure how outliers are handled ('cap', 'remove', 'transform').

    INPUT PARAMETERS:
        - `split_id` (str): Active split handle ID.
        - `strategy` (str, default='cap'): Outlier strategy.

    RETURNS:
        - Dict containing `outlier_spec`.
    """
    train_df = _load_train(split_id)
    meta = _load_split_meta(split_id)
    target_column = meta.get("target_column", "")

    outlier_report = {}
    for col in train_df.select_dtypes(include=["number"]).columns:
        if col == target_column:
            continue
        series = train_df[col].dropna()
        if len(series) < 4:
            continue
        lower, upper = compute_iqr_bounds(series)
        outliers = train_df[(train_df[col] < lower) | (train_df[col] > upper)]
        outlier_report[col] = {
            "lower_bound": float(lower),
            "upper_bound": float(upper),
            "outlier_count": int(len(outliers)),
        }

    outlier_spec = build_outlier_spec(strategy, outlier_report)
    return {"outlier_spec": outlier_spec}


def select_features(
    split_id: str,
    target_column: str,
    task_type: str,
    method: str = "auto",
) -> dict:
    """Define feature selection technique for reducing dimensionality.

    PURPOSE & WHEN TO USE:
        Call this tool to set feature selection approach ('filter', 'wrapper', 'embedded', 'auto').

    INPUT PARAMETERS:
        - `split_id` (str): Active split handle ID.
        - `target_column` (str): Target column name.
        - `task_type` (str): Task type ('classification' or 'regression').
        - `method` (str, default='auto'): Selection method.

    RETURNS:
        - Dict containing `feature_selection_spec`.
    """
    train_df = _load_train(split_id)
    feature_cols = [c for c in train_df.columns if c != target_column]
    n_features = len(feature_cols)

    if method == "auto":
        if n_features > 50:
            resolved_method = "embedded"
        else:
            resolved_method = "filter"
    else:
        resolved_method = method

    k_or_percentile = 0.8

    spec = build_feature_selection_spec(resolved_method, k_or_percentile)
    spec["task_type"] = task_type
    return {"feature_selection_spec": spec}


def build_preprocessing_pipeline(
    split_id: str,
    target_column: str,
    imputer_spec: dict,
    encoder_spec: dict,
    scaler_spec: dict,
    feature_selection_spec: dict,
    outlier_spec: Optional[dict] = None,
    derived_feature_spec: Optional[dict] = None,
) -> dict:
    """Assemble all individual data preparation specs into an unfitted sklearn Pipeline.

    PURPOSE & WHEN TO USE:
        Call this tool after defining imputer, encoder, scaler, and selection specs.
        It assembles an unfitted scikit-learn Pipeline object and saves it to cache, returning `pipeline_spec_id`.

    INPUT PARAMETERS:
        - `split_id` (str): Active split handle ID.
        - `target_column` (str): Target column name.
        - `imputer_spec` (dict): Imputer recipe from `handle_missing_values`.
        - `encoder_spec` (dict): Encoder recipe from `encode_categorical`.
        - `scaler_spec` (dict): Scaler recipe from `scale_numeric_features`.
        - `feature_selection_spec` (dict): Selection recipe from `select_features`.
        - `outlier_spec` (dict, optional): Outlier recipe from `handle_outliers`.
        - `derived_feature_spec` (dict, optional): Feature engineering recipe from `derive_features`.

    RETURNS:
        - Dict containing:
            - `pipeline_spec_id` (str): Unique handle ID for the assembled unfitted Pipeline spec.

    AI AGENT GUIDELINES:
        - Pass `pipeline_spec_id` to Modeling MCP tools like `cross_validate_model` and `finalize_model`.
    """
    pipeline_spec_id = compute_handle_id(
        split_id,
        target_column,
        json.dumps(imputer_spec, sort_keys=True, default=str),
        json.dumps(encoder_spec, sort_keys=True, default=str),
        json.dumps(scaler_spec, sort_keys=True, default=str),
        json.dumps(feature_selection_spec, sort_keys=True, default=str),
        json.dumps(outlier_spec or {}, sort_keys=True, default=str),
        json.dumps(derived_feature_spec or {}, sort_keys=True, default=str),
    )

    if not object_exists("specs", pipeline_spec_id):
        pipeline = assemble_column_transformer(
            imputer_spec=imputer_spec,
            encoder_spec=encoder_spec,
            scaler_spec=scaler_spec,
            feature_selection_spec=feature_selection_spec,
            outlier_spec=outlier_spec,
            derived_feature_spec=derived_feature_spec,
        )
        save_pipeline_spec(pipeline, pipeline_spec_id)

    return {"pipeline_spec_id": pipeline_spec_id}