def maybe_inject_failure(model_type: str) -> None:
    """Reads TABULARML_INJECT_FAILURE env var. If it matches model_type
    AND a one-shot marker file (_cache/.failure_markers/{model_type})
    doesn't exist yet: creates the marker, raises RuntimeError. Second
    call for the same model_type passes through -- simulates 'it crashed
    once, then succeeded on retry'."""