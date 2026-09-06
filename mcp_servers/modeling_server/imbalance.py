IMBALANCE_STRATEGIES = {
    "class_weight": ..., "oversample_smote": ..., "oversample_random": ...,
    "undersample_random": ..., "combined_smote_undersample": ...,
}
"""class_weight is the zero-new-dependency default; the oversample/
undersample options require imbalanced-learn."""

def detect_severity_ratio(y) -> float:
    """minority class count / majority class count."""

def resolve_auto_strategy(severity_ratio: float, n_rows: int) -> str:
    """Ratio above ~0.3 -> no strategy needed. Below that, with enough
    rows to safely synthesize minority examples -> oversample_smote.
    Below that on small data (where SMOTE's nearest-neighbor basis gets
    unreliable) -> class_weight instead."""

def build_imbalance_spec(strategy: str, params: dict) -> dict:
    """Returns a recipe. Applied inside cv.run_cross_validation's per-fold
    training step, never before folding -- resampling before the split
    would let synthetic points derived from validation-fold neighbors leak
    into training."""