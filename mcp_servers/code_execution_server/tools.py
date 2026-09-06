"""
Code Execution MCP tools.

Provides run_python (sandboxed execution with handle injection),
install_package (allowlist-gated), and list_available_tools (introspection).
"""

from __future__ import annotations

import subprocess
import sys
from typing import Optional

try:
    from ..shared.cache import load_object
except ImportError:
    from shared.cache import load_object

try:
    from .package_policy import validate_install_request
    from .sandbox import execute_in_sandbox, enforce_output_caps
    from .tool_stubs import generate_client_stubs, _TOOL_SCHEMAS
except ImportError:
    from package_policy import validate_install_request
    from sandbox import execute_in_sandbox, enforce_output_caps
    from tool_stubs import generate_client_stubs, _TOOL_SCHEMAS


def run_python(
    code: str,
    input_handles: dict[str, str],
    timeout_s: int = 60,
) -> dict:
    """Execute Python code in an isolated sandbox with handle injection and MCP client tool stubs.

    PURPOSE & WHEN TO USE:
        Call this tool when custom data analysis, plotting, transformation, or model inspection is required that is not covered by existing MCP server tools.
        Cached datasets and models specified in `input_handles` are automatically loaded into local variable names inside the sandbox.

    INPUT PARAMETERS:
        - `code` (str): Python source code string to execute.
        - `input_handles` (dict[str, str]): Mapping of variable names to handle IDs (e.g. `{"df": "a1b2c3"}`).
        - `timeout_s` (int, default=60): Wall-clock execution timeout in seconds.

    RETURNS:
        - Dict containing:
            - `stdout` (str): Captured standard output text.
            - `stderr` (str): Captured error output text.
            - `new_handles` (dict[str, str]): Auto-persisted handles for DataFrames or models assigned in code.

    AI AGENT GUIDELINES:
        - Code executed inside the sandbox can invoke other MCP server tools directly via auto-injected client stubs.
    """
    if not isinstance(code, str) or not code.strip():
        return {"stdout": "", "stderr": "Error: code must be a non-empty string.", "new_handles": {}}

    mounted_handles: dict = {}
    for var_name, handle_id in (input_handles or {}).items():
        try:
            obj = load_object_by_guess(handle_id)
            mounted_handles[var_name] = obj
        except Exception as e:
            return {
                "stdout": "",
                "stderr": f"Error loading handle '{handle_id}' as '{var_name}': {e}",
                "new_handles": {},
            }

    stubs_code = generate_client_stubs()
    full_code = stubs_code + "\n\n" + code

    result = execute_in_sandbox(full_code, mounted_handles, timeout_s)
    return result


def load_object_by_guess(handle_id: str):
    """Tries each subdirectory in turn until the handle is found."""
    for subdir in ("datasets", "models", "specs"):
        try:
            return load_object(subdir, handle_id)
        except FileNotFoundError:
            pass
    raise FileNotFoundError(f"Handle not found in any cache subdirectory: {handle_id!r}")


def install_package(package_spec: str) -> dict:
    """Install a Python package into the execution environment subject to allowlist security policy.

    PURPOSE & WHEN TO USE:
        Call this tool if sandboxed code requires an allowed third-party library (e.g. 'xgboost', 'lightgbm', 'shap', 'plotly', 'statsmodels').

    INPUT PARAMETERS:
        - `package_spec` (str): Package specification string (e.g. 'xgboost==2.0.0').

    RETURNS:
        - Dict with `success` (bool) and `message` (str).
    """
    allowed, reason = validate_install_request(package_spec)
    if not allowed:
        return {"success": False, "message": reason}

    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", package_spec, "--quiet"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode == 0:
            return {"success": True, "message": f"Installed: {package_spec}"}
        else:
            return {
                "success": False,
                "message": f"pip install failed: {result.stderr[:500]}",
            }
    except subprocess.TimeoutExpired:
        return {"success": False, "message": "pip install timed out after 120s"}
    except Exception as e:
        return {"success": False, "message": str(e)}


def list_available_tools() -> dict:
    """List tool names, signatures, and parameter specifications across all TabularML MCP servers.

    PURPOSE & WHEN TO USE:
        Call this tool for introspection to discover available tools and their parameter schemas across all 5 MCP servers.

    INPUT PARAMETERS:
        None.

    RETURNS:
        - Dict mapping server names to lists of tool schema dicts (`tool_name`, `params`).
    """
    return {
        server: [
            {"tool_name": t["name"], "params": t["params"]}
            for t in tools
        ]
        for server, tools in _TOOL_SCHEMAS.items()
    }