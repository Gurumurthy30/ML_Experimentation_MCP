def split_dataset(dataset_id: str, target_column: str, task_type: str, split_strategy: str = "auto", group_column: str = None, time_column: str = None, test_size: float = 0.2) -> dict:
    """The leakage firewall -- the first thing that happens to a
    dataset_id after diagnostics. split_strategy='auto' calls
    splitting.auto_detect_split_strategy(): a time_column with a temporal
    goal -> time-series split (no shuffle); a repeating group_column (e.g.
    customer_id) -> group-aware split so one entity never spans both
    partitions; classification -> stratified; else plain random. Writes
    train.parquet + test.parquet under a new split_id. test.parquet is not
    read again until final_test_evaluation. Returns {split_id,
    resolved_strategy}."""

def generate_cv_folds(split_id: str, n_folds: str | int = "auto") -> dict:
    """n_folds='auto' resolves via splitting.compute_adaptive_fold_count:
    ~10 folds under 1000 rows, 5 up to 50k, 3 above -- small datasets need
    more folds for a stable estimate. Fold type is inherited from
    split_dataset's resolved_strategy, not re-chosen here. Writes only
    row-index lists (not a data copy) under a new folds_id."""

def validate_split_quality(split_id: str, force: bool = False) -> dict:
    """Compares each feature's train vs. test distribution:
    splitting.ks_test_numeric() for numeric columns, chi-square for
    categorical. Auto-skips (returns {skipped: True, reason}) above ~1000
    rows unless force=True -- large samples rarely have a real
    representativeness problem; this exists to catch an unlucky split on
    small data. Returns {column: {p_value, flagged}} or the skip dict."""

def derive_features(split_id: str, feature_defs: list[dict]) -> dict:
    """Each feature_def is {name, expression_type, definition}.
    expression_type='row_wise' (date parts, ratios) is safe to compute
    directly since it never looks across rows. expression_type=
    'aggregation' (group means/counts) only stores the recipe here -- the
    actual group statistics get computed on train data only, per fold,
    downstream in Modeling MCP, for the same leakage reason as everything
    else in this file. Returns {derived_feature_spec}."""

def handle_missing_values(split_id: str, strategy_map: dict | str = "auto") -> dict:
    """strategy_map='auto' resolves a strategy per column via
    imputation.resolve_auto_strategy(), using that column's role plus
    analyze_missing_values' likely_mar flag and missing percentage.
    Returns imputer_spec -- a strategy per column, not fitted values. The
    actual median/mode gets computed per fold, later, in Modeling MCP's
    cv.py."""

def encode_categorical(split_id: str, target_column: str, strategy: str = "auto") -> dict:
    """strategy in {one_hot, label, target_encoding, frequency, 'auto'}.
    target_column is required even though this only encodes features,
    because target_encoding's out-of-fold scheme needs to know what it's
    encoding against. Returns encoder_spec -- a recipe, not fitted encoder
    state; the K-fold out-of-fold logic is configured here but executed
    per fold downstream."""

def scale_numeric_features(split_id: str, method: str = "standard") -> dict:
    """method in {standard, minmax, robust, none}. 'none' is the expected
    choice when the pipeline will later pair with model_family=
    'tree_ensemble', which doesn't need scaling. Returns scaler_spec."""

def handle_outliers(split_id: str, strategy: str = "cap") -> dict:
    """strategy in {cap, remove, transform}. Bounds computed from the
    train partition only. Returns outlier_spec -- like every other spec
    here, bounds get recomputed per fold downstream rather than frozen
    once."""

def select_features(split_id: str, target_column: str, task_type: str, method: str = "auto") -> dict:
    """method in {filter, wrapper, embedded, 'auto'}. Deliberately returns
    a method + target feature count, NOT a frozen list of selected column
    names -- freezing the actual selection here and reusing it across
    every CV fold would leak each fold's validation rows into the
    selection decision, the same category of mistake as fitting a scaler
    before splitting. The concrete column subset gets recomputed inside
    each fold's fit, in Modeling MCP. Returns feature_selection_spec."""

def build_preprocessing_pipeline(split_id: str, target_column: str, imputer_spec: dict, encoder_spec: dict, scaler_spec: dict, feature_selection_spec: dict, outlier_spec: dict = None, derived_feature_spec: dict = None) -> dict:
    """Assembles every spec above into one sklearn ColumnTransformer +
    Pipeline via pipeline.assemble_column_transformer(). The returned
    pipeline is UNFITTED -- nothing in it has seen data yet. Saved to
    _cache/specs/{pipeline_spec_id}.joblib. It only ever gets .fit()
    called on it inside Modeling MCP: once per fold in
    cross_validate_model, and once for real in finalize_model. Returns
    {pipeline_spec_id}."""