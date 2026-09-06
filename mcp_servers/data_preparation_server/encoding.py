ENCODE_STRATEGIES = {"one_hot": ..., "label": ..., "target_encoding": ..., "frequency": ...}
"""one_hot is the default for low-cardinality nominal columns,
target_encoding for high-cardinality ones."""

def build_encoder_spec(strategy: str, column_roles: dict, cardinalities: dict) -> dict:
    """strategy='auto' picks one_hot when a column's cardinality <= 15,
    target_encoding above that."""

def cv_safe_target_encoder_config() -> dict:
    """Default smoothing/fold-count parameters for the K-fold
    out-of-fold target-encoding scheme -- this only configures the
    approach. The actual out-of-fold computation happens per CV fold in
    Modeling MCP, never against the whole train pool at once, which is
    what would leak target information into the encoding."""