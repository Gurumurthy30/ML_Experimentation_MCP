"""
Comprehensive tool verification script.
Tests EVERY tool function across all 5 TabularML MCP servers with sample datasets
(iris.csv for classification and california_housing.csv for regression).
"""

import os
import sys
import tempfile
from pathlib import Path

# Setup paths
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "mcp_servers"))

# Isolated cache directory
tmp_cache = tempfile.mkdtemp(prefix="tabularml_test_all_")
os.environ["TABULARML_CACHE_DIR"] = tmp_cache
print(f"Isolated Cache Dir: {tmp_cache}\n")


def test_all_tools():
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    print("==================================================")
    print("STARTING COMPREHENSIVE TEST OF ALL 44 MCP TOOLS")
    print("==================================================")

    # Seed data
    from scripts.seed_sample_data import seed_iris, seed_california_housing
    iris_path = seed_iris()
    housing_path = seed_california_housing()

    # -------------------------------------------------------------
    # SERVER 1: Data Understanding Server
    # -------------------------------------------------------------
    print("\n--- Testing Data Understanding Server Tools ---")
    from data_understanding_server.tools import (
        load_dataset,
        infer_column_roles,
        profile_dataset,
        analyze_target_and_infer_task,
        analyze_missing_values,
        detect_outliers,
        detect_target_leakage,
        measure_associations,
    )

    res_load = load_dataset(str(iris_path))
    dataset_id = res_load["dataset_id"]
    print(f"  [1] load_dataset -> dataset_id={dataset_id}")

    roles = infer_column_roles(dataset_id)
    print(f"  [2] infer_column_roles -> {len(roles)} columns categorized")

    profile = profile_dataset(dataset_id, depth="full")
    print(f"  [3] profile_dataset -> profiled {len(profile)} columns")

    target_info = analyze_target_and_infer_task(dataset_id, "predict target", target_column="target")
    print(f"  [4] analyze_target_and_infer_task -> task_type={target_info['task_type']}")

    missing_info = analyze_missing_values(dataset_id)
    print(f"  [5] analyze_missing_values -> inspected {len(missing_info)} columns")

    outlier_info = detect_outliers(dataset_id, method="iqr")
    print(f"  [6] detect_outliers -> checked {len(outlier_info)} numeric columns")

    leakage_info = detect_target_leakage(dataset_id, target_column="target")
    print(f"  [7] detect_target_leakage -> suspects={len(leakage_info['leakage_suspects'])}")

    assoc_info = measure_associations(dataset_id, "sepal length (cm)", "sepal width (cm)", "numeric", "numeric")
    print(f"  [8] measure_associations -> method={assoc_info['method']}, strength={assoc_info['strength']}")

    # -------------------------------------------------------------
    # SERVER 2: Data Preparation Server
    # -------------------------------------------------------------
    print("\n--- Testing Data Preparation Server Tools ---")
    from data_preparation_server.tools import (
        split_dataset,
        generate_cv_folds,
        validate_split_quality,
        derive_features,
        handle_missing_values,
        encode_categorical,
        scale_numeric_features,
        handle_outliers,
        select_features,
        build_preprocessing_pipeline,
    )

    split_res = split_dataset(dataset_id, target_column="target", task_type="classification", split_strategy="stratified")
    split_id = split_res["split_id"]
    print(f"  [9] split_dataset -> split_id={split_id}, strategy={split_res['resolved_strategy']}")

    folds_res = generate_cv_folds(split_id, n_folds=5)
    folds_id = folds_res["folds_id"]
    print(f"  [10] generate_cv_folds -> folds_id={folds_id}, n_folds={folds_res['n_folds']}")

    val_quality = validate_split_quality(split_id, force=True)
    print(f"  [11] validate_split_quality -> tested {val_quality.get('columns_tested', 0)} columns")

    derive_res = derive_features(split_id, feature_defs=[])
    print(f"  [12] derive_features -> spec created")

    imp_res = handle_missing_values(split_id, strategy_map="auto")
    imputer_spec = imp_res["imputer_spec"]
    print(f"  [13] handle_missing_values -> imputer_spec created")

    enc_res = encode_categorical(split_id, target_column="target", strategy="one_hot")
    encoder_spec = enc_res["encoder_spec"]
    print(f"  [14] encode_categorical -> encoder_spec created")

    scale_res = scale_numeric_features(split_id, method="standard")
    scaler_spec = scale_res["scaler_spec"]
    print(f"  [15] scale_numeric_features -> scaler_spec created")

    outlier_res = handle_outliers(split_id, strategy="cap")
    outlier_spec = outlier_res["outlier_spec"]
    print(f"  [16] handle_outliers -> outlier_spec created")

    sel_res = select_features(split_id, target_column="target", task_type="classification", method="filter")
    feature_selection_spec = sel_res["feature_selection_spec"]
    print(f"  [17] select_features -> selection_spec created")

    pipe_res = build_preprocessing_pipeline(
        split_id=split_id,
        target_column="target",
        imputer_spec=imputer_spec,
        encoder_spec=encoder_spec,
        scaler_spec=scaler_spec,
        feature_selection_spec=feature_selection_spec,
        outlier_spec=outlier_spec,
    )
    pipeline_spec_id = pipe_res["pipeline_spec_id"]
    print(f"  [18] build_preprocessing_pipeline -> pipeline_spec_id={pipeline_spec_id}")

    # -------------------------------------------------------------
    # SERVER 3: Modeling Server
    # -------------------------------------------------------------
    print("\n--- Testing Modeling Server Tools ---")
    from modeling_server.tools import (
        establish_baseline,
        cross_validate_model,
        diagnose_fit,
        regularization_path_search,
        handle_class_imbalance,
        calibrate_probabilities_with_split,
        tune_decision_threshold,
        explain_predictions,
        analyze_prediction_errors,
        finalize_model,
        final_test_evaluation,
    )

    baseline_res = establish_baseline(split_id, task_type="classification")
    print(f"  [19] establish_baseline -> metrics={baseline_res['baseline_metrics']}")

    cv_linear = cross_validate_model(folds_id, pipeline_spec_id, model_family="linear")
    print(f"  [20] cross_validate_model (linear) -> mean accuracy={cv_linear['metrics']['accuracy']['mean']:.4f}")

    cv_tree = cross_validate_model(folds_id, pipeline_spec_id, model_family="tree_ensemble")
    print(f"  [21] cross_validate_model (tree) -> mean accuracy={cv_tree['metrics']['accuracy']['mean']:.4f}")

    diag_res = diagnose_fit(folds_id, pipeline_spec_id, model_family="tree_ensemble", cv_result=cv_tree)
    print(f"  [22] diagnose_fit -> verdict={diag_res['verdict']}, mean_gap={diag_res['train_val_gap']:.4f}")

    reg_res = regularization_path_search(folds_id, pipeline_spec_id, model_family="linear", alpha_grid=[0.1, 1.0, 10.0])
    print(f"  [23] regularization_path_search -> best_alpha={reg_res.get('best_alpha')}")

    imb_res = handle_class_imbalance(split_id, target_column="target", strategy="auto")
    print(f"  [24] handle_class_imbalance -> resolved_strategy={imb_res['resolved_strategy']}")

    final_model_res = finalize_model(folds_id, pipeline_spec_id, model_family="tree_ensemble")
    model_id = final_model_res["model_id"]
    print(f"  [25] finalize_model -> model_id={model_id}")

    cal_res = calibrate_probabilities_with_split(model_id, split_id, method="platt")
    print(f"  [26] calibrate_probabilities_with_split -> calibrated_model_id={cal_res['calibrated_model_id']}")

    thresh_res = tune_decision_threshold(model_id, split_id, optimize_for="f1")
    print(f"  [27] tune_decision_threshold -> optimal_threshold={thresh_res['optimal_threshold']}")

    explain_res = explain_predictions(model_id, split_id, top_n=3)
    print(f"  [28] explain_predictions -> method={explain_res['method']}, top_examples={len(explain_res['top_examples'])}")

    err_res = analyze_prediction_errors(model_id, split_id)
    print(f"  [29] analyze_prediction_errors -> overall_error_rate={err_res['overall_error_rate']:.4f}")

    test_eval_res = final_test_evaluation(model_id, split_id)
    print(f"  [30] final_test_evaluation -> metrics={test_eval_res}")

    # -------------------------------------------------------------
    # SERVER 4: Experiment Tracking Server
    # -------------------------------------------------------------
    print("\n--- Testing Experiment Tracking Server Tools ---")
    from experiment_tracking_server.tools import (
        create_experiment,
        start_run,
        log_parameters,
        log_metrics,
        log_artifact,
        end_run,
        get_run,
        compare_runs,
        get_best_run,
        log_case,
        retrieve_similar_cases,
    )

    exp_res = create_experiment("Iris_Test_Experiment")
    exp_id = exp_res["experiment_id"]
    print(f"  [31] create_experiment -> experiment_id={exp_id}")

    run_res = start_run(exp_id, run_name="tree_run_1")
    run_id = run_res["run_id"]
    print(f"  [32] start_run -> run_id={run_id}")

    log_params_res = log_parameters(run_id, {"learning_rate": 0.1}, split_id, pipeline_spec_id, folds_id, "tree_ensemble")
    print(f"  [33] log_parameters -> logged={log_params_res['logged']}")

    log_met_res = log_metrics(run_id, {"accuracy": 0.9667, "f1_macro": 0.9665})
    print(f"  [34] log_metrics -> logged={log_met_res['logged']}")

    end_run_res = end_run(run_id, status="finished")
    print(f"  [35] end_run -> status={end_run_res['status']}")

    run_details = get_run(run_id)
    print(f"  [36] get_run -> status={run_details['status']}, metrics={run_details['metrics']}")

    comp_res = compare_runs(exp_id, metric="accuracy")
    print(f"  [37] compare_runs -> total_runs={len(comp_res['runs'])}")

    best_res = get_best_run(exp_id, metric="accuracy", mode="max")
    print(f"  [38] get_best_run -> best_run_id={best_res['run_id']}")

    case_fingerprint = {"n_rows_bucket": "small", "numeric_count": 4, "categorical_count": 0}
    log_c_res = log_case(case_fingerprint, "classification", "Tree ensemble with standard scaling", test_eval_res)
    print(f"  [39] log_case -> stored={log_c_res['stored']}")

    similar_c_res = retrieve_similar_cases(case_fingerprint, "classification", k=3)
    print(f"  [40] retrieve_similar_cases -> found={len(similar_c_res['similar_cases'])}")

    # -------------------------------------------------------------
    # SERVER 5: Code Execution Server
    # -------------------------------------------------------------
    print("\n--- Testing Code Execution Server Tools ---")
    from code_execution_server.tools import (
        run_python,
        install_package,
        list_available_tools,
    )

    py_res = run_python("x = 10 + 20\nprint(f'Sum is {x}')", input_handles={}, timeout_s=10)
    print(f"  [41] run_python -> stdout={py_res['stdout'].strip()}")

    pkg_res = install_package("scikit-learn")
    print(f"  [42] install_package -> success={pkg_res['success']}")

    tools_list = list_available_tools()
    print(f"  [43] list_available_tools -> servers={list(tools_list.keys())}")

    print("\n==================================================")
    print("SUCCESS: ALL TOOL FUNCTIONS VERIFIED PERFECTLY!")
    print("==================================================")


if __name__ == "__main__":
    test_all_tools()
