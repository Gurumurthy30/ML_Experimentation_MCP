"""
Class imbalance detection and strategy resolution for modeling_server.
"""

from __future__ import annotations

import pandas as pd


IMBALANCE_STRATEGIES: dict[str, str] = {
    "class_weight": "class_weight",
    "oversample_smote": "oversample_smote",
    "oversample_random": "oversample_random",
    "undersample_random": "undersample_random",
    "combined_smote_undersample": "combined_smote_undersample",
    "none": "none",
}
"""class_weight is the zero-new-dependency default; the oversample/
undersample options require imbalanced-learn.  'none' is returned when the
imbalance is mild enough to ignore."""


def detect_severity_ratio(y: pd.Series) -> float:
    """Input: the target column's values, train partition only.
    Processing: counts occurrences of each class.  Output: a single float
    -- minority class count divided by majority class count -- close to
    1.0 means balanced, close to 0 means severe imbalance."""
    counts = y.value_counts()
    if len(counts) < 2:
        return 1.0  # Only one class -- degenerate case
    minority = int(counts.min())
    majority = int(counts.max())
    return minority / max(majority, 1)


def resolve_auto_strategy(severity_ratio: float, n_rows: int) -> str:
    """Ratio above ~0.3 -> no strategy needed.  Below that, with enough
    rows to safely synthesize minority examples -> oversample_smote.
    Below that on small data (where SMOTE's nearest-neighbor basis gets
    unreliable) -> class_weight instead."""
    if severity_ratio >= 0.3:
        return "none"
    # Need at least ~6 minority examples for SMOTE (5 neighbours + 1)
    if n_rows >= 200 and severity_ratio * n_rows >= 6:
        return "oversample_smote"
    return "class_weight"


def build_imbalance_spec(strategy: str, params: dict) -> dict:
    """Returns a recipe.  Applied inside cv.run_cross_validation's per-fold
    training step, never before folding -- resampling before the split
    would let synthetic points derived from validation-fold neighbors leak
    into training."""
    if strategy not in IMBALANCE_STRATEGIES:
        raise ValueError(
            f"Unknown imbalance strategy: {strategy!r}. "
            f"Choose from {list(IMBALANCE_STRATEGIES)}"
        )
    return {"strategy": strategy, "params": params or {}}