"""
Tests verifying data leakage guarantees across data preparation, modeling, and evaluation.
"""

import pandas as pd
import pytest

from data_understanding_server.tools import detect_target_leakage
from modeling_server.finalize import (
    check_test_not_already_evaluated,
    mark_test_evaluated,
)
from modeling_server.tools import final_test_evaluation


def test_target_leakage_detection(isolated_cache_dir):
    """Verifies target leakage detector flags columns perfectly correlated or identical to target."""
    df = pd.DataFrame({
        "feature_clean": [1, 2, 3, 4, 5],
        "feature_leaked": [0, 1, 0, 1, 1],  # Exact match to target
        "target": [0, 1, 0, 1, 1],
    })

    from shared.cache import save_object
    dataset_id = "test_leakage_ds"
    save_object(df, "datasets", dataset_id)

    res = detect_target_leakage(dataset_id, target_column="target")
    suspect_cols = [s["column"] for s in res["leakage_suspects"]]
    assert "feature_leaked" in suspect_cols


def test_eval_on_reserved_test_guarantee(isolated_cache_dir):
    """Verifies that evaluation on test split enforces single-eval rule."""
    split_id = "test_split_leakage_check"
    model_id = "test_model_1"

    # First check passes
    assert check_test_not_already_evaluated(split_id) is True

    # Mark as evaluated
    mark_test_evaluated(split_id, model_id)

    # Second check fails
    assert check_test_not_already_evaluated(split_id) is False


def test_one_shot_test_evaluation_rejection_with_different_model_id(isolated_cache_dir):
    """
    Verifies that calling final_test_evaluation a second time on the same split_id
    with a different model_id raises a RuntimeError.
    """
    split_id = "test_split_oneshot_check"
    model_id_1 = "model_alpha"
    model_id_2 = "model_beta"

    # Mark test as evaluated for model_alpha
    mark_test_evaluated(split_id, model_id_1)

    # Invoking final_test_evaluation with a different model_id must fail
    with pytest.raises(RuntimeError, match="violates the one-shot test rule"):
        final_test_evaluation(model_id_2, split_id)
