"""
Package allowlist policy for code_execution_server.

Any package not in ALLOWED_PACKAGES is refused by validate_install_request,
regardless of what install_package is called with.
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Allowlist
# ---------------------------------------------------------------------------

ALLOWED_PACKAGES: dict[str, str] = {
    # Core data science stack
    "numpy": ">=1.24.0",
    "pandas": ">=2.0.0",
    "scipy": ">=1.10.0",
    "scikit-learn": ">=1.3.0",
    "joblib": ">=1.3.0",
    "pyarrow": ">=12.0.0",
    # Plotting
    "matplotlib": ">=3.7.0",
    "seaborn": ">=0.12.0",
    "plotly": ">=5.14.0",
    # Stats
    "statsmodels": ">=0.14.0",
    "pingouin": ">=0.5.0",
    # ML extras
    "xgboost": ">=1.7.0",
    "lightgbm": ">=3.3.0",
    "catboost": ">=1.1.0",
    "imbalanced-learn": ">=0.10.0",
    "shap": ">=0.41.0",
    "optuna": ">=3.0.0",
    # Utilities
    "tqdm": ">=4.65.0",
    "tabulate": ">=0.9.0",
}
"""Locked allowlist.  Anything not on this list is refused by
validate_install_request regardless of what install_package is asked for."""


def validate_install_request(
    package_spec: str,
    policy: str = "locked",
) -> tuple[bool, str]:
    """policy='locked' checks package_spec against ALLOWED_PACKAGES (name
    only, version ignored on the caller's side -- we check the name is on the
    list but allow any version spec the caller passes, since restricting to a
    specific pinned version on the caller's end is too rigid for an interactive
    research environment).

    Returns (allowed, reason).
    """
    if policy != "locked":
        return False, f"Unknown policy: {policy!r}. Only 'locked' is supported."

    if not package_spec or not isinstance(package_spec, str):
        return False, "Invalid package spec: must be a non-empty string."

    # Parse the package name from a spec like 'pandas==2.2.0', 'numpy>=1.24',
    # 'scipy', etc.
    name_match = re.match(r"^([A-Za-z0-9_.\-]+)", package_spec.strip())
    if not name_match:
        return False, f"Could not parse package name from: {package_spec!r}"

    package_name = name_match.group(1).lower().replace("_", "-")

    # Check against the allowlist (normalize both sides)
    for allowed_name in ALLOWED_PACKAGES:
        if allowed_name.lower().replace("_", "-") == package_name:
            return True, f"Package '{package_name}' is on the allowlist."

    return (
        False,
        f"Package '{package_name}' is not on the allowlist. "
        f"Allowed packages: {sorted(ALLOWED_PACKAGES)}",
    )