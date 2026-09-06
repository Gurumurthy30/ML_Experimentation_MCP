def platt_calibration(model, X_val, y_val):
    """Fits a sigmoid (logistic) mapping from raw scores to calibrated
    probabilities -- sklearn.calibration.CalibratedClassifierCV(
    method='sigmoid')."""

def isotonic_calibration(model, X_val, y_val):
    """Non-parametric monotonic mapping -- more flexible than Platt but
    needs more validation data to avoid overfitting the calibration curve
    itself."""