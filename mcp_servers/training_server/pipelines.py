def infer_column_types(df: pd.DataFrame, target_column: str) -> tuple[list[str], list[str]]:
    """Returns (numeric_cols, categorical_cols) excluding target_column,
    using dtype + a cardinality check for object/int columns that are
    really categorical."""

def build_column_transformer(dataset_id: str, target_column: str, config: dict) -> ColumnTransformer:
    """Numeric branch: SimpleImputer(strategy='median') [+ StandardScaler
    if config['scale']]. Categorical branch: OneHotEncoder(handle_unknown
    ='ignore') if n_unique <= 15 else OrdinalEncoder. Two variants get
    built per plan -- scaled (for logistic_regression) and unscaled (for
    tree models) -- both cached separately as pipeline_ids."""

def save_pipeline(pipeline, pipeline_id: str) -> None   # joblib.dump under _cache/pipelines/
def load_pipeline(pipeline_id: str)                      # joblib.load