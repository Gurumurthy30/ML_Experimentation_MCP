"""
Generates thin Python wrapper stubs for all tools on the other 4 MCP servers,
injected into the sandbox namespace before every run_python call.

This lets generated code call e.g.
    modeling.cross_validate_model(folds_id=..., pipeline_spec_id=..., ...)
directly inside a script, instead of requiring a separate top-level MCP call
per tool invocation.
"""

from __future__ import annotations

import sys
from pathlib import Path


# Tool schemas for the other 4 servers (static, since these are compile-time
# known from the plan doc -- no MCP round-trip needed just to build stubs)
_TOOL_SCHEMAS: dict[str, list[dict]] = {
    "data_understanding": [
        {"name": "load_dataset", "params": ["path", "delimiter=None", "encoding=None"]},
        {"name": "infer_column_roles", "params": ["dataset_id", "mode='permissive'"]},
        {"name": "profile_dataset", "params": ["dataset_id", "depth='full'"]},
        {"name": "analyze_target_and_infer_task", "params": ["dataset_id", "goal_text", "target_column=None"]},
        {"name": "analyze_missing_values", "params": ["dataset_id"]},
        {"name": "detect_outliers", "params": ["dataset_id", "method='iqr'"]},
        {"name": "detect_target_leakage", "params": ["dataset_id", "target_column", "outcome_time_column=None", "correlation_threshold=0.98"]},
        {"name": "measure_associations", "params": ["dataset_id", "column_a", "column_b", "role_a", "role_b", "method='auto'"]},
    ],
    "data_preparation": [
        {"name": "split_dataset", "params": ["dataset_id", "target_column", "task_type", "split_strategy='auto'", "group_column=None", "time_column=None", "test_size=0.2"]},
        {"name": "generate_cv_folds", "params": ["split_id", "n_folds='auto'"]},
        {"name": "validate_split_quality", "params": ["split_id", "force=False"]},
        {"name": "derive_features", "params": ["split_id", "feature_defs"]},
        {"name": "handle_missing_values", "params": ["split_id", "strategy_map='auto'"]},
        {"name": "encode_categorical", "params": ["split_id", "target_column", "strategy='auto'"]},
        {"name": "scale_numeric_features", "params": ["split_id", "method='standard'"]},
        {"name": "handle_outliers", "params": ["split_id", "strategy='cap'"]},
        {"name": "select_features", "params": ["split_id", "target_column", "task_type", "method='auto'"]},
        {"name": "build_preprocessing_pipeline", "params": ["split_id", "target_column", "imputer_spec", "encoder_spec", "scaler_spec", "feature_selection_spec", "outlier_spec=None", "derived_feature_spec=None"]},
    ],
    "modeling": [
        {"name": "establish_baseline", "params": ["split_id", "task_type"]},
        {"name": "cross_validate_model", "params": ["folds_id", "pipeline_spec_id", "model_family", "hyperparams=None"]},
        {"name": "diagnose_fit", "params": ["folds_id", "pipeline_spec_id", "model_family", "cv_result"]},
        {"name": "regularization_path_search", "params": ["folds_id", "pipeline_spec_id", "model_family='linear'", "alpha_grid=None"]},
        {"name": "handle_class_imbalance", "params": ["split_id", "target_column", "strategy='auto'", "severity_threshold=None"]},
        {"name": "calibrate_probabilities", "params": ["model_id", "split_id", "method='platt'"]},
        {"name": "tune_decision_threshold", "params": ["model_id", "split_id", "optimize_for"]},
        {"name": "explain_predictions", "params": ["model_id", "split_id", "top_n"]},
        {"name": "analyze_prediction_errors", "params": ["model_id", "split_id", "segment_by=None"]},
        {"name": "finalize_model", "params": ["folds_id", "pipeline_spec_id", "model_family", "hyperparams=None"]},
        {"name": "final_test_evaluation", "params": ["model_id", "split_id"]},
    ],
    "experiment_tracking": [
        {"name": "create_experiment", "params": ["name"]},
        {"name": "start_run", "params": ["experiment_id", "run_name"]},
        {"name": "log_parameters", "params": ["run_id", "params", "split_id", "pipeline_spec_id", "folds_id", "model_family"]},
        {"name": "log_metrics", "params": ["run_id", "metrics"]},
        {"name": "log_artifact", "params": ["run_id", "path"]},
        {"name": "end_run", "params": ["run_id", "status='finished'"]},
        {"name": "get_run", "params": ["run_id"]},
        {"name": "compare_runs", "params": ["experiment_id", "metric"]},
        {"name": "get_best_run", "params": ["experiment_id", "metric", "mode"]},
        {"name": "log_case", "params": ["dataset_fingerprint", "task_type", "approach_summary", "outcome_metrics"]},
        {"name": "retrieve_similar_cases", "params": ["dataset_fingerprint", "task_type", "k=5", "scope='private'"]},
    ],
}


def generate_client_stubs() -> str:
    """Uses _TOOL_SCHEMAS to generate Python source defining one thin wrapper
    function per tool on Data Understanding, Data Preparation, Modeling, and
    Experiment Tracking MCP.  Injected into the sandbox namespace before every
    run_python call, so one call can batch what would otherwise be several
    separate MCP round-trips into a single script.

    The stubs import directly from the server modules (since the sandbox runs
    in-process with the same Python environment), so they call the actual tool
    functions rather than making another MCP network call.
    """
    lines = [
        "# Auto-generated client stubs -- do not edit",
        "import sys",
        "from pathlib import Path",
        "",
        "# Ensure project root is on path",
        f"_project_root = {str(Path(__file__).parent.parent.parent)!r}",
        "if _project_root not in sys.path:",
        "    sys.path.insert(0, _project_root)",
        "",
    ]

    module_map = {
        "data_understanding": "data_understanding_server.tools",
        "data_preparation": "data_preparation_server.tools",
        "modeling": "modeling_server.tools",
        "experiment_tracking": "experiment_tracking_server.tools",
    }

    for server_name, module_path in module_map.items():
        lines.append(f"# --- {server_name} ---")
        lines.append(f"import importlib as _il_{server_name}")
        lines.append(f"_mod_{server_name} = _il_{server_name}.import_module('mcp_servers.{module_path}')")
        lines.append(f"class _{server_name.replace('_', '').title()}Stubs:")
        for tool in _TOOL_SCHEMAS.get(server_name, []):
            tool_name = tool["name"]
            params_str = ", ".join(tool["params"])
            # Build kwargs pass-through
            kwarg_names = [p.split("=")[0].strip() for p in tool["params"]]
            kwargs_pass = ", ".join(f"{n}={n}" for n in kwarg_names)
            lines.append(f"    def {tool_name}(self, {params_str}):")
            lines.append(f"        return _mod_{server_name}.{tool_name}({kwargs_pass})")
        alias = server_name.replace("_", "")
        lines.append(f"{server_name} = _{server_name.replace('_', '').title()}Stubs()")
        lines.append("")

    return "\n".join(lines)