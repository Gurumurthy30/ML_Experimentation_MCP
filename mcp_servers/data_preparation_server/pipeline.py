def assemble_column_transformer(imputer_spec, encoder_spec, scaler_spec, feature_selection_spec, outlier_spec, derived_feature_spec) -> ColumnTransformer:
    """Builds one sklearn ColumnTransformer + Pipeline from the five
    specs, applied in a fixed order: derive -> handle_outliers -> impute
    -> encode -> scale -> select. Nothing is fit here -- this function
    only wires the steps together structurally."""

def save_pipeline_spec(spec, pipeline_spec_id: str) -> None:
    """joblib.dump the unfitted Pipeline under
    _cache/specs/{pipeline_spec_id}.joblib."""

def load_pipeline_spec(pipeline_spec_id: str):
    """joblib.load -- returns the same unfitted, clonable Pipeline.
    Callers (Modeling MCP's cv.py, finalize.py) are responsible for
    calling sklearn.clone() before ever calling .fit() -- this file never
    fits anything itself."""