"""
Probability calibration for modeling_server.

Both functions produce a NEW calibrated model wrapping the original --
the original fitted model is never mutated.
"""

from __future__ import annotations

import pandas as pd
from sklearn.calibration import CalibratedClassifierCV

try:
    from sklearn.frozen import FrozenEstimator
except ImportError:
    FrozenEstimator = None


def platt_calibration(
    model: "sklearn.base.BaseEstimator",
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> CalibratedClassifierCV:
    """Input: an already-fitted classifier (finalize_model's output) plus
    a held-out validation slice it wasn't fit on. Fits a sigmoid mapping."""
    if FrozenEstimator is not None:
        calibrated = CalibratedClassifierCV(estimator=FrozenEstimator(model), method="sigmoid")
    else:
        calibrated = CalibratedClassifierCV(estimator=model, method="sigmoid", cv="prefit")
    calibrated.fit(X_val, y_val)
    return calibrated


def isotonic_calibration(
    model: "sklearn.base.BaseEstimator",
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> CalibratedClassifierCV:
    """Input: fitted model plus validation slice. Fits isotonic mapping."""
    if FrozenEstimator is not None:
        calibrated = CalibratedClassifierCV(estimator=FrozenEstimator(model), method="isotonic")
    else:
        calibrated = CalibratedClassifierCV(estimator=model, method="isotonic", cv="prefit")
    calibrated.fit(X_val, y_val)
    return calibrated