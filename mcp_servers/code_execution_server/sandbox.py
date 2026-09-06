"""
Sandbox execution environment for code_execution_server.

Uses a subprocess + restricted exec() rather than Docker, making it
runnable on Windows without container infrastructure.  The sandbox:
  - Executes user code in a separate Python subprocess
  - Injects handle objects into the namespace before execution
  - Captures stdout/stderr and applies output caps
  - Scans the result namespace for new DataFrame/model objects and
    auto-persists them as new cache handles

SECURITY NOTE: This is a research/local-use sandbox.  For production
or multi-user environments, replace with a proper container sandbox.
"""

from __future__ import annotations

import io
import os
import pickle
import subprocess
import sys
import tempfile
import textwrap
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from typing import Any

import pandas as pd


# Output caps
STDOUT_MAX_CHARS = 4000
STDERR_MAX_CHARS = 2000


def build_sandbox_container() -> dict:
    """Returns a sandbox config dict rather than a Docker container object
    (no Docker required).  Defines resource limits and scratch directory
    location -- used by execute_in_sandbox to configure the execution
    environment.

    In a production deployment, replace this function with one that
    launches an actual Docker container with --network=none and cgroup
    limits."""
    scratch_dir = tempfile.mkdtemp(prefix="tabularml_sandbox_")
    return {
        "scratch_dir": scratch_dir,
        "timeout_s": 60,
        "max_stdout_chars": STDOUT_MAX_CHARS,
        "max_stderr_chars": STDERR_MAX_CHARS,
    }


def execute_in_sandbox(
    code: str,
    mounted_handles: dict[str, Any],
    timeout_s: int,
) -> dict:
    """Runs code with mounted_handles pre-loaded into the namespace.
    Returns {stdout, stderr, return_value, new_handles}."""

    # Build a wrapper script that:
    # 1. Loads the handles from a pickle file
    # 2. Executes user code with stdout/stderr captured
    # 3. Saves the resulting namespace to another pickle file
    with tempfile.TemporaryDirectory(prefix="tabularml_exec_") as tmpdir:
        tmp = Path(tmpdir)

        # Serialize handles for the subprocess
        handles_path = tmp / "handles.pkl"
        with open(handles_path, "wb") as f:
            pickle.dump(mounted_handles, f)

        result_path = tmp / "result.pkl"
        code_path = tmp / "user_code.py"
        code_path.write_text(code, encoding="utf-8")

        wrapper = textwrap.dedent(f"""
import sys, io, pickle, traceback

# Load pre-injected handles
with open({str(handles_path)!r}, "rb") as _f:
    _handles = pickle.load(_f)

# Build execution namespace
_ns = dict(**_handles)

# Capture stdout/stderr
_stdout_buf = io.StringIO()
_stderr_buf = io.StringIO()

_error = None
try:
    with open({str(code_path)!r}, "r", encoding="utf-8") as _cf:
        _user_code = _cf.read()

    import sys as _sys
    old_stdout, old_stderr = _sys.stdout, _sys.stderr
    _sys.stdout, _sys.stderr = _stdout_buf, _stderr_buf
    try:
        exec(compile(_user_code, "<sandbox>", "exec"), _ns)
    except Exception as _e:
        _stderr_buf.write(traceback.format_exc())
    finally:
        _sys.stdout, _sys.stderr = old_stdout, old_stderr
except Exception as _outer:
    _stderr_buf.write(traceback.format_exc())

# Collect result: only include picklable data objects (DataFrames, Series, models, dicts, etc.)
_picklable_ns = {{}}
for k, v in _ns.items():
    if k.startswith("_") or callable(v) or isinstance(v, type):
        continue
    try:
        pickle.dumps(v)
        _picklable_ns[k] = v
    except Exception:
        pass

_result = {{
    "stdout": _stdout_buf.getvalue(),
    "stderr": _stderr_buf.getvalue(),
    "namespace": _picklable_ns,
}}

with open({str(result_path)!r}, "wb") as _rf:
    pickle.dump(_result, _rf)
""")
        wrapper_path = tmp / "sandbox_wrapper.py"
        wrapper_path.write_text(wrapper, encoding="utf-8")

        try:
            proc = subprocess.run(
                [sys.executable, str(wrapper_path)],
                timeout=timeout_s,
                capture_output=True,
                text=True,
                env={**os.environ, "PYTHONPATH": str(Path(__file__).parent.parent.parent)},
            )
        except subprocess.TimeoutExpired:
            return {
                "stdout": "",
                "stderr": f"[Sandbox timed out after {timeout_s}s]",
                "new_handles": {},
                "error": "timeout",
            }

        if not result_path.exists() or result_path.stat().st_size == 0:
            return {
                "stdout": proc.stdout[:STDOUT_MAX_CHARS] if proc.stdout else "",
                "stderr": (proc.stderr or "[Sandbox process failed to produce output]")[:STDERR_MAX_CHARS],
                "new_handles": {},
                "error": "sandbox_crash",
            }

        with open(result_path, "rb") as f:
            result = pickle.load(f)

        stdout, stderr = enforce_output_caps(result["stdout"], result["stderr"])
        new_handles = persist_new_handles(result.get("namespace", {}))

        return {
            "stdout": stdout,
            "stderr": stderr,
            "new_handles": new_handles,
        }


def enforce_output_caps(stdout: str, stderr: str) -> tuple[str, str]:
    """Truncates stdout to ~4000 characters server-side -- enforced here
    regardless of what the sandboxed code tries to print, not just
    requested of the model."""
    if len(stdout) > STDOUT_MAX_CHARS:
        stdout = stdout[:STDOUT_MAX_CHARS] + f"\n[... truncated to {STDOUT_MAX_CHARS} chars ...]"
    if len(stderr) > STDERR_MAX_CHARS:
        stderr = stderr[:STDERR_MAX_CHARS] + f"\n[... truncated to {STDERR_MAX_CHARS} chars ...]"
    return stdout, stderr


def persist_new_handles(namespace: dict) -> dict[str, str]:
    """Scans the sandbox namespace after execution for anything that looks
    like a DataFrame or sklearn model/pipeline, calls
    shared.cache.save_object() on each, returns {variable_name: new_handle_id}."""
    from shared.cache import save_object, compute_handle_id
    import sklearn.base
    import sklearn.pipeline

    new_handles = {}
    for name, obj in namespace.items():
        if name.startswith("_"):
            continue

        if isinstance(obj, pd.DataFrame):
            handle_id = compute_handle_id("sandbox_df", name, str(obj.shape), str(obj.columns.tolist()))
            save_object(obj, "datasets", handle_id)
            new_handles[name] = handle_id

        elif isinstance(obj, (sklearn.base.BaseEstimator, sklearn.pipeline.Pipeline)):
            handle_id = compute_handle_id("sandbox_model", name, str(type(obj).__name__))
            save_object(obj, "models", handle_id)
            new_handles[name] = handle_id

    return new_handles