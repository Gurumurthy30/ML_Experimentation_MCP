"""
Smoke test -- exercises the core tool pipeline end-to-end without the harness.

Run from the project root:
    python -m scripts.smoke_test_server

This script:
  1. Seeds a sample dataset (iris for classification)
  2. Calls each tool in order, simulating a full ML run
  3. Prints results at each step
  4. Verifies no errors occur

It does NOT run the MCP servers over stdio -- it imports the tool functions
directly, so it works as a quick integration test during development.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

# Set up the project root on sys.path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "mcp_servers"))

# Use a temporary cache dir so smoke tests don't pollute the real cache
tmp_cache = tempfile.mkdtemp(prefix="tabularml_smoke_")
os.environ["TABULARML_CACHE_DIR"] = tmp_cache
print(f"Cache dir: {tmp_cache}\n")


def run_smoke_test():
    # Step 1: Seed data
    from scripts.seed_sample_data import seed_iris
    iris_path = seed_iris()
    print(f"[1] Dataset ready: {iris_path}")

    # Step 2: Load dataset
    from data_understanding_server.tools import (
        load_dataset, infer_column_roles, profile_dataset,
        analyze_target_and_infer_task, analyze_missing_values,
    )

    result = load_dataset(str(iris_path))
    dataset_id = result["dataset_id"]
    print(f"[2] load_dataset -> dataset_id={dataset_id}, shape=({result['n_rows']},{result['n_cols']})")

    # Step 3: Infer column roles
    roles = infer_column_roles(dataset_id)
    print(f"[3] infer_column_roles -> {list(roles.keys())} roles assigned")

    # Step 4: Profile dataset
    profile = profile_dataset(dataset_id, depth="quick")
    print(f"[4] profile_dataset -> profiled {len(profile)} columns")

    # Step 5: Analyze target
    target_info = analyze_target_and_infer_task(dataset_id, "predict species", target_column="target")
    task_type = target_info["task_type"]
    target_column = target_info.get("target_column", "target")
    print(f"[5] analyze_target -> task_type={task_type}, target={target_column}")

    # Step 6: Split dataset
    from data_preparation_server.tools import (
        split_dataset, generate_cv_folds, handle_missing_values,
        encode_categorical, scale_numeric_features, handle_outliers,
        select_features, build_preprocessing_pipeline,
    )

    split_result = split_dataset(dataset_id, target_column, task_type)
    split_id = split_result["split_id"]
    print(f"[6] split_dataset -> split_id={split_id}, strategy={split_result['resolved_strategy']}")
    print(f"     n_train={split_result['n_train']}, n_test={split_result['n_test']}")

    # Step 7: Generate CV folds
    folds_result = generate_cv_folds(split_id)
    folds_id = folds_result["folds_id"]
    print(f"[7] generate_cv_folds -> folds_id={folds_id}, n_folds={folds_result['n_folds']}")

    # Step 8: Handle missing values
    miss_result = handle_missing_values(split_id)
    imputer_spec = miss_result["imputer_spec"]
    print(f"[8] handle_missing_values -> imputer_spec has {len(imputer_spec.get('strategies', {}))} columns")

    # Step 9: Encode categorical
    enc_result = encode_categorical(split_id, target_column)
    encoder_spec = enc_result["encoder_spec"]
    print(f"[9] encode_categorical -> encoder_spec has {len(encoder_spec.get('strategies', {}))} columns")

    # Step 10: Scale
    scale_result = scale_numeric_features(split_id, method="standard")
    scaler_spec = scale_result["scaler_spec"]
    print(f"[10] scale_numeric_features -> method={scaler_spec['method']}")

    # Step 11: Outliers
    outlier_result = handle_outliers(split_id)
    outlier_spec = outlier_result["outlier_spec"]
    print(f"[11] handle_outliers -> {len(outlier_spec.get('bounds', {}))} columns flagged")

    # Step 12: Feature selection
    feat_result = select_features(split_id, target_column, task_type)
    feature_selection_spec = feat_result["feature_selection_spec"]
    print(f"[12] select_features -> method={feature_selection_spec['method']}")

    # Step 13: Build pipeline
    pipe_result = build_preprocessing_pipeline(
        split_id=split_id,
        target_column=target_column,
        imputer_spec=imputer_spec,
        encoder_spec=encoder_spec,
        scaler_spec=scaler_spec,
        feature_selection_spec=feature_selection_spec,
        outlier_spec=outlier_spec,
    )
    pipeline_spec_id = pipe_result["pipeline_spec_id"]
    print(f"[13] build_preprocessing_pipeline -> pipeline_spec_id={pipeline_spec_id}")

    # Step 14: Cross-validate
    from modeling_server.tools import (
        establish_baseline, cross_validate_model, diagnose_fit, finalize_model, final_test_evaluation,
    )

    baseline = establish_baseline(split_id, task_type)
    print(f"[14] establish_baseline -> {baseline['baseline_metrics']}")

    cv_result = cross_validate_model(folds_id, pipeline_spec_id, "tree_ensemble")
    print(f"[15] cross_validate_model -> metrics: {cv_result['metrics']}")

    # Step 15: Diagnose fit
    diag = diagnose_fit(folds_id, pipeline_spec_id, "tree_ensemble", cv_result)
    print(f"[16] diagnose_fit -> verdict={diag['verdict']}, gap={diag['train_val_gap']:.4f}")

    # Step 16: Finalize model
    fin_result = finalize_model(folds_id, pipeline_spec_id, "tree_ensemble")
    model_id = fin_result["model_id"]
    print(f"[17] finalize_model -> model_id={model_id}")

    # Step 17: Final test evaluation
    test_metrics = final_test_evaluation(model_id, split_id)
    print(f"[18] final_test_evaluation -> {test_metrics}")

if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    try:
        run_smoke_test()
    except Exception as e:
        import traceback
        print(f"\n[FAILED] Smoke test FAILED: {e}")
        traceback.print_exc()
        sys.exit(1)
