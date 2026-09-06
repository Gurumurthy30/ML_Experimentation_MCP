OUTLIER_STRATEGIES = {"cap": ..., "remove": ..., "transform": ...}
"""cap = winsorize to the IQR fence; remove = drop the row (train
partition only); transform = a variance-stabilizing transform like log1p
or Yeo-Johnson instead of touching the row at all."""

def build_outlier_spec(strategy: str, outlier_report: dict) -> dict:
    """outlier_report comes from Data Understanding MCP's detect_outliers.
    Bounds/thresholds are computed from the train partition only --
    included in the spec so they get recomputed correctly inside each CV
    fold rather than frozen from the full train pool once."""