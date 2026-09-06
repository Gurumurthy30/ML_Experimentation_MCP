ALLOWED_PACKAGES = {...}
"""Locked, pinned allowlist. Anything not on this list is refused by
validate_install_request regardless of what install_package is asked
for."""

def validate_install_request(package_spec: str, policy: str = "locked") -> tuple[bool, str]:
    """policy='locked' checks package_spec against ALLOWED_PACKAGES exactly
    (name and pinned version). Returns (allowed, reason)."""