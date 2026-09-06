"""
Unit tests for code_execution_server sandbox and tools.
"""

from code_execution_server.package_policy import validate_install_request
from code_execution_server.sandbox import execute_in_sandbox


def test_package_policy():
    allowed, reason = validate_install_request("scikit-learn==1.3.0")
    assert allowed is True

    allowed_disallowed, reason = validate_install_request("malicious_package_123")
    assert allowed_disallowed is False
    assert "allowlist" in reason.lower()


def test_sandbox_execution(isolated_cache_dir):
    code = "result = 1 + 1\nprint(f'Computed {result}')"
    res = execute_in_sandbox(code, mounted_handles={}, timeout_s=10)
    assert res["stderr"] == ""
    assert "Computed 2" in res["stdout"]
