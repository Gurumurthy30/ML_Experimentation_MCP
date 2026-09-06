"""
Shared vocabulary imported by all 5 MCP servers.

Nothing here does any computation -- it only fixes shapes, names, and
enum values so a field name or enum value means the same thing regardless
of which server process produced or consumes it.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import TypedDict


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ColumnRole(str, Enum):
    """The closed vocabulary infer_column_roles (Data Understanding MCP) is
    allowed to return.  Imported by data_preparation_server's imputation.py,
    encoding.py, and scaling.py so a role string is spelled identically in
    all three places instead of drifting across separate processes."""

    NUMERIC_CONTINUOUS = "numeric_continuous"
    NUMERIC_DISCRETE = "numeric_discrete"
    CATEGORICAL_NOMINAL = "categorical_nominal"
    CATEGORICAL_ORDINAL = "categorical_ordinal"
    DATETIME = "datetime"
    BOOLEAN = "boolean"
    TEXT_FREEFORM = "text_freeform"
    IDENTIFIER = "identifier"
    CONSTANT = "constant"
    UNKNOWN = "unknown"


class TaskType(str, Enum):
    """REGRESSION or CLASSIFICATION -- the value analyze_target_and_infer_task
    resolves and every downstream tool (split_dataset, select_features,
    get_model, ...) accepts as a plain string matching one of these two
    values."""

    REGRESSION = "regression"
    CLASSIFICATION = "classification"


# ---------------------------------------------------------------------------
# TypedDicts
# ---------------------------------------------------------------------------

class DatasetProfile(TypedDict):
    """Documents the dict shape profile_dataset (Data Understanding MCP)
    actually returns.  Consumed by case_library.compute_dataset_fingerprint()
    and by data_preparation_server's imputation/encoding/scaling modules, so
    all of them read the same field names rather than each re-deriving the
    shape from raw tool output."""

    dataset_id: str
    n_rows: int
    n_cols: int
    # column_roles maps column name -> ColumnRole value string
    column_roles: dict[str, str]
    # columns maps column name -> role-appropriate stats dict
    columns: dict[str, dict]


class ToolError(TypedDict):
    """The one error shape every tool across all 5 servers raises through.

    recoverable=False marks cases the harness should not retry (e.g.
    final_test_evaluation's one-shot rule being violated).
    recoverable=True marks cases worth retrying with different arguments
    (e.g. an ambiguous target_column).  Lets the harness distinguish those
    two cases by field name alone, regardless of which server raised the
    error."""

    error_type: str
    message: str
    recoverable: bool


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

_HANDLE_RE = re.compile(r"^[0-9a-f]{12}$")


def validate_handle_id(value: str) -> bool:
    """Input: any string passed where a handle_id is expected as a tool
    argument.  Processing: checks value matches compute_handle_id's fixed
    output shape (12 lowercase hex characters) via a regex, nothing more.
    Output: True if well-formed, False otherwise -- called at the top of
    every tool that receives a handle_id parameter, so a malformed or
    hallucinated ID fails fast with a clear ToolError instead of a
    confusing file-not-found error surfacing three calls later."""
    if not isinstance(value, str):
        return False
    return bool(_HANDLE_RE.match(value))
