SCALE_METHODS = {"standard": ..., "minmax": ..., "robust": ..., "none": ...}
"""robust (median/IQR-based) is preferred over standard when
detect_outliers flagged that column, since a plain StandardScaler gets
distorted by outliers it hasn't been told about."""

def build_scaler_spec(method: str, model_family_hint: str = None) -> dict:
    """method='auto' with model_family_hint='tree_ensemble' resolves to
    'none' -- tree-based models don't benefit from scaling, so skipping it
    saves a pipeline step for free."""