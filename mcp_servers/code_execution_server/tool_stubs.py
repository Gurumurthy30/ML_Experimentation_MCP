def generate_client_stubs() -> str:
    """Uses list_available_tools()'s schema to generate Python source
    defining one thin wrapper function per tool on Data Understanding,
    Data Preparation, Modeling, and Experiment Tracking MCP. Injected into
    the sandbox namespace before every run_python call, so one call can
    batch what would otherwise be several separate MCP round-trips into a
    single script."""