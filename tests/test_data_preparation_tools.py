"""
Unit tests for data_preparation_server tools.
"""

import pandas as pd

from data_understanding_server.tools import load_dataset
from data_preparation_server.tools import (
    split_dataset,
    generate_cv_folds,
    handle_missing_values,
    scale_numeric_features,
    encode_categorical,
    select_features,
    build_preprocessing_pipeline,
)


def test_data_preparation_flow(isolated_cache_dir, tmp_path, sample_classification_df):
    csv_path = tmp_path / "sample.csv"
    sample_classification_df.to_csv(csv_path, index=False)

    ds_res = load_dataset(str(csv_path))
    dataset_id = ds_res["dataset_id"]

    # Split
    split_res = split_dataset(dataset_id, target_column="target", task_type="classification", split_strategy="stratified")
    assert "split_id" in split_res
    split_id = split_res["split_id"]

    # Folds
    folds_res = generate_cv_folds(split_id, n_folds=5)
    assert "folds_id" in folds_res

    # Impute
    imp_res = handle_missing_values(split_id, strategy_map="auto")
    imputer_spec = imp_res["imputer_spec"]

    # Scale
    scale_res = scale_numeric_features(split_id, method="standard")
    scaler_spec = scale_res["scaler_spec"]

    # Encode
    enc_res = encode_categorical(split_id, target_column="target", strategy="one_hot")
    encoder_spec = enc_res["encoder_spec"]

    # Select
    sel_res = select_features(split_id, target_column="target", task_type="classification", method="filter")
    feature_selection_spec = sel_res["feature_selection_spec"]

    # Pipeline
    pipe_res = build_preprocessing_pipeline(
        split_id=split_id,
        target_column="target",
        imputer_spec=imputer_spec,
        encoder_spec=encoder_spec,
        scaler_spec=scaler_spec,
        feature_selection_spec=feature_selection_spec,
    )
    assert "pipeline_spec_id" in pipe_res
