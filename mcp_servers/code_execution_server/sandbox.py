def build_sandbox_container():
    """Docker container, --network=none (no egress), read-only bind mount
    of data/_cache plus one writable scratch subdir torn down after each
    invocation, CPU/memory/wall-clock limits via cgroups."""

def execute_in_sandbox(code: str, mounted_handles: dict, timeout_s: int) -> dict:
    """Runs code inside build_sandbox_container()'s container with
    mounted_handles already loaded and the tool_stubs functions injected.
    Hard-kills on timeout_s. Returns raw stdout/stderr/return value before
    enforce_output_caps() trims them."""

def enforce_output_caps(stdout: str, stderr: str) -> tuple[str, str]:
    """Truncates stdout to ~4000 characters server-side -- enforced here
    regardless of what the sandboxed code tries to print, not just
    requested of the model."""

def persist_new_handles(scratch_dir: str) -> dict:
    """Scans the sandbox's writable scratch directory after execution for
    anything that looks like a DataFrame/model/figure, calls
    shared.cache.save_object() on each, returns {variable_name:
    new_handle_id}."""