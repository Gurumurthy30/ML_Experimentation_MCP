"""
Outlier-handling strategy registry + spec builder.
"""

from __future__ import annotations

import pandas as pd

OUTLIER_STRATEGIES = {"cap": "cap", "remove": "remove", "transform": "transform"}
"""cap = winsorize to the IQR fence; remove = drop the row (train partition
only); transform = a variance-stabilizing transform like log1p or
Yeo-Johnson instead of touching the row at all."""


def compute_iqr_bounds(series: pd.Series, k: float = 1.5) -> tuple[float, float]:
    """Input: a numeric Series (already NA-dropped by the caller) and the
    IQR multiplier. Processing: the classic Tukey fence, Q1 - k*IQR to
    Q3 + k*IQR. Output: (lower_bound, upper_bound). This is the same
    computation tools.py's handle_outliers runs against the train
    partition only -- never the full pre-split dataset -- so the bounds
    themselves can't leak test-row information."""
    q1 = float(series.quantile(0.25))
    q3 = float(series.quantile(0.75))
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr


def build_outlier_spec(strategy: str, outlier_report: dict) -> dict:
    """outlier_report comes from tools.py's handle_outliers, which computes
    it fresh from the train partition via compute_iqr_bounds (mirroring
    Data Understanding MCP's detect_outliers shape, but leakage-safe).
    Bounds are packaged into the spec so they get recomputed correctly
    inside each CV fold downstream rather than frozen from the full train
    pool once."""
    if strategy not in OUTLIER_STRATEGIES:
        raise ValueError(f"Unknown outlier strategy: {strategy!r}")
    bounds = {
        column: {"lower": info["lower_bound"], "upper": info["upper_bound"]}
        for column, info in (outlier_report or {}).items()
        if isinstance(info, dict) and "lower_bound" in info and "upper_bound" in info
    }
    return {"strategy": strategy, "bounds": bounds}