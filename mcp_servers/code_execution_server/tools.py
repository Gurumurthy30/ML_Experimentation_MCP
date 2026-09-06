def run_python(code: str, input_handles: dict, timeout_s: int = 60) -> dict:
    """input_handles maps a local variable name to a handle_id (e.g.
    {"df": "a1b2c3"}) -- loaded via shared.cache.load_object and injected
    into the sandbox namespace before code runs. Delegates execution to
    sandbox.execute_in_sandbox(). Anything the code assigns that looks
    like a DataFrame/model/figure is auto-persisted by
    sandbox.persist_new_handles() and returned as a new handle rather than
    inlined. stdout/stderr are capped; large return values become handles,
    never raw bytes in the tool response."""

def install_package(package_spec: str) -> dict:
    """Checked against package_policy.validate_install_request() before
    anything is actually installed inside the container. Returns
    {success, message}. Affects only later run_python calls in the same
    session -- doesn't touch the shared cache."""

def list_available_tools() -> dict:
    """Introspects the other four servers' registered tools (name, params,
    return shape). Used by tool_stubs.generate_client_stubs() to build the
    function stubs injected into the sandbox, so generated code can call
    e.g. modeling.cross_validate_model(...) directly instead of that
    requiring a separate top-level MCP call."""