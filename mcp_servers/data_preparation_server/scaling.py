"""
Scaling method registry + resolution logic.
"""

from __future__ import annotations

from typing import Optional

SCALE_METHODS = {"standard": "standard", "minmax": "minmax", "robust": "robust", "none": "none"}
"""robust (median/IQR-based) is preferred over standard when detect_outliers
flagged that column, since a plain StandardScaler gets distorted by
outliers it hasn't been told about. Values here are canonical names;
pipeline.py maps each name to the actual sklearn scaler class."""


def build_scaler_spec(method: str = "auto", model_family_hint: Optional[str] = None) -> dict:
    """method='auto' with model_family_hint='tree_ensemble' resolves to
    'none' -- tree-based models don't benefit from scaling, so skipping it
    saves a pipeline step for free."""
    if method in ("auto", None):
        resolved = "none" if model_family_hint == "tree_ensemble" else "standard"
    else:
        resolved = method
    if resolved not in SCALE_METHODS:
        raise ValueError(f"Unknown scaling method: {resolved!r}")
    return {"method": resolved}