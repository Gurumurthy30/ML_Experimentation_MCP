import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

def test_normality(series: pd.Series) -> dict:
    """Shapiro-Wilk under 5000 non-null values (its valid range),
    D'Agostino-Pearson above that. Returns {is_normal, statistic,
    p_value} at alpha=0.05."""
    series = series.dropna()
    if len(series) < 5000:
        statistic, p_value = stats.shapiro(series)
    else:
        statistic, p_value = stats.normaltest(series)
    is_normal = p_value > 0.05
    return {"is_normal": is_normal, "statistic": statistic, "p_value": p_value}

def compute_kurtosis(series: pd.Series) -> float:
    """Fisher's definition (normal distribution = 0). Flags heavy-tailed
    columns, which feeds outlier-handling strategy downstream."""
    k = stats.kurtosis(series.dropna(), fisher=True)
    return k

def spearman_correlation(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    """Rank-based correlation -- catches monotonic-but-non-linear
    relationships Pearson misses, robust to outliers. Returns
    (rho, p_value)."""
    rho, p_value = stats.spearmanr(x.dropna(), y.dropna())
    return {"rho": rho, "p_value": p_value}

def cramers_v(x: pd.Series, y: pd.Series) -> float:
    """Chi-square-based association strength for two categorical columns,
    bias-corrected, bounded to [0, 1]."""
    table = pd.crosstab(x, y)
    chi2, p, dof, expected = stats.chi2_contingency(table)
    n = table.to_numpy().sum()
    r, k = table.shape
    _cramers_v = np.sqrt(chi2 / (n * (min(r - 1, k - 1))))
    return _cramers_v

def anova_f_eta_squared(categorical: pd.Series, numeric: pd.Series) -> dict:
    """One-way ANOVA F-statistic plus eta-squared (share of the numeric
    column's variance explained by categorical group membership). Returns
    {f_statistic, p_value, eta_squared}."""
    groups = numeric.groupby(categorical)
    samples = [group.dropna() for _, group in groups]
    f_statistic, p_value = stats.f_oneway(*samples)
    ss_total = ((numeric - numeric.mean()) ** 2).sum()
    ss_between = sum(len(group) * (group.mean() - numeric.mean()) ** 2 for group in samples)
    eta_squared = ss_between / ss_total if ss_total > 0 else np.nan
    return {"f_statistic": f_statistic, "p_value": p_value, "eta_squared": eta_squared}

def mutual_information(x: pd.Series, y: pd.Series, x_role: str, y_role: str) -> float:
    """sklearn mutual_info_regression/classif, chosen by role combination.
    This is measure_associations' escalation method when Spearman/
    Cramer's V/ANOVA-F all report a weak relationship, since mutual
    information catches dependence that isn't monotonic or linear at
    all."""
    if x_role == "categorical":
        x = x.astype("category").cat.codes
        x_discrete = True
    else:
        x = x.astype(float)
        x_discrete = False

    if y_role == "categorical":
        y = y.astype("category").cat.codes
    
    X = x.to_numpy().reshape(-1, 1)
    Y = y.to_numpy()
    
    if y_role == "numeric":
        return mutual_info_regression(X, Y,
                                    discrete_features=[x_discrete],
                                    random_state=42)[0]
    elif y_role == "categorical":
        return mutual_info_classif(X, Y,
                                    discrete_features=[x_discrete],
                                    random_state=42)[0]
    else:
        raise ValueError(f"mutual_info not implemented for {x_role}+{y_role}")

def auto_select_association_method(role_a: str, role_b: str) -> str:
    """Pure lookup: numeric+numeric -> 'spearman', categorical+categorical
    -> 'cramers_v', categorical+numeric -> 'anova'. Used when
    measure_associations' method='auto'."""
    if role_a == "numeric" and role_b == "numeric":
        return "spearman"
    elif role_a == "categorical" and role_b == "categorical":
        return "cramers_v"
    elif (role_a == "categorical" and role_b == "numeric") or (role_a == "numeric" and role_b == "categorical"):
        return "anova"
    else:
        raise ValueError(f"auto_select_association_method not implemented for {role_a}+{role_b}")