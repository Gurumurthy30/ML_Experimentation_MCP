"""
Unit tests for data_understanding_server tools.
"""

import pandas as pd

from data_understanding_server.tools import (
    load_dataset,
    infer_column_roles,
    profile_dataset,
    analyze_target_and_infer_task,
    analyze_missing_values,
)


def test_data_understanding_flow(isolated_cache_dir, tmp_path, sample_classification_df):
    csv_path = tmp_path / "sample.csv"
    sample_classification_df.to_csv(csv_path, index=False)

    # 1. Load dataset
    res = load_dataset(str(csv_path))
    assert "dataset_id" in res
    assert res["n_rows"] == 100
    assert res["n_cols"] == 5
    dataset_id = res["dataset_id"]

    # 2. Infer column roles
    roles = infer_column_roles(dataset_id)
    assert len(roles) == 5

    # 3. Profile dataset
    profile = profile_dataset(dataset_id)
    assert "num_col1" in profile
    assert profile["num_col1"]["missing_pct"] == 6.0

    # 4. Analyze target
    target_info = analyze_target_and_infer_task(dataset_id, "classification task for target", "target")
    assert target_info["task_type"] == "classification"
    assert target_info["target_column"] == "target"

    # 5. Analyze missing values
    missing_info = analyze_missing_values(dataset_id)
    assert "num_col1" in missing_info
    assert missing_info["num_col1"]["missing_pct"] > 0
