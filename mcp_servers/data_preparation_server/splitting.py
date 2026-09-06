"""
Splitting logic for data_preparation_server. Everything here operates on
plain DataFrames handed to it by tools.py -- no cache/handle awareness
lives in this file, only the actual split-strategy math.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit,
    KFold,
    StratifiedKFold,
    TimeSeriesSplit,
    train_test_split,
)

RANDOM_STATE = 42


def auto_detect_split_strategy(
    df: pd.DataFrame,
    target_column: str,
    group_column: Optional[str],
    time_column: Optional[str],
) -> str:
    """Resolution order: time_column present -> 'time_series'; a given
    group_column that actually repeats (n_unique < n_rows) -> 'group';
    task_type is classification -> 'stratified'; else 'random'."""
    if time_column:
        return "time_series"
    if group_column and df[group_column].nunique(dropna=False) < len(df):
        return "group"
    if _looks_like_classification_target(df[target_column]):
        return "stratified"
    return "random"


def _looks_like_classification_target(series: pd.Series) -> bool:
    """Cheap fallback used only when task_type isn't already resolved by
    the caller: a low-cardinality, non-float column looks like a label."""
    if series.dtype == object or isinstance(series.dtype, pd.CategoricalDtype) or series.dtype == bool:
        return True
    n_unique = series.nunique(dropna=True)
    return n_unique <= max(20, int(0.05 * len(series)))


def stratified_split(
    df: pd.DataFrame, target_column: str, test_size: float, random_state: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Input: the full loaded DataFrame, the target column name, the held-
    out fraction (0-1), and a fixed random_state for reproducibility.
    Processing: sklearn.model_selection.train_test_split with
    stratify=df[target_column], preserving the target's class proportions
    in both partitions. Output: (train_df, test_df) -- two row-disjoint
    DataFrames with identical columns to the input."""
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df[target_column]
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def group_split(
    df: pd.DataFrame, group_column: str, test_size: float, random_state: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Input: the full DataFrame and the name of the column identifying a
    repeating entity (e.g. customer_id). Processing:
    sklearn.model_selection.GroupShuffleSplit keyed on group_column --
    guarantees no single group_column value appears in both resulting
    partitions, which stratified/random splitting cannot promise. Output:
    (train_df, test_df), row-disjoint and group-disjoint."""
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(df, groups=df[group_column]))
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)


def time_series_split(
    df: pd.DataFrame, time_column: str, test_size: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Input: the full DataFrame and the name of its datetime column.
    Processing: sorts by time_column ascending and takes a chronological
    tail (test_size fraction of rows) as the test partition -- no
    shuffling, since shuffling a temporal split would leak the future into
    training. Output: (train_df, test_df), where every timestamp in
    train_df precedes every timestamp in test_df."""
    sort_key = pd.to_datetime(df[time_column], errors="coerce")
    ordered = df.assign(**{"__sort_key__": sort_key}).sort_values("__sort_key__", kind="mergesort")
    ordered = ordered.drop(columns="__sort_key__").reset_index(drop=True)
    cutoff = int(round(len(ordered) * (1 - test_size)))
    cutoff = min(max(cutoff, 1), len(ordered) - 1)
    return ordered.iloc[:cutoff].reset_index(drop=True), ordered.iloc[cutoff:].reset_index(drop=True)


def random_split(
    df: pd.DataFrame, test_size: float, random_state: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Plain, unstratified fallback used when auto_detect_split_strategy
    resolves to 'random' (regression target, no group/time column)."""
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def compute_adaptive_fold_count(n_rows: int) -> int:
    """Input: row count of the train partition only. Processing: a fixed
    threshold lookup. Output: an int -- 10 under 1000 rows, 5 between
    1000-50000, 3 above 50000 -- since smaller datasets need more folds for
    a stable score estimate, while more folds on large data mostly just
    costs compute for no added stability."""
    if n_rows < 1000:
        return 10
    if n_rows <= 50000:
        return 5
    return 3


def generate_folds(
    train_pool: pd.DataFrame,
    strategy: str,
    n_folds: int,
    group_column: Optional[str] = None,
    time_column: Optional[str] = None,
    target_column: Optional[str] = None,
) -> list[tuple[list[int], list[int]]]:
    """Input: the train partition only (test rows are never seen here),
    the split strategy already resolved by auto_detect_split_strategy (not
    re-chosen), and n_folds from compute_adaptive_fold_count. Processing:
    dispatches to StratifiedKFold / GroupKFold / TimeSeriesSplit / KFold
    matching strategy. Output: a list of (train_row_indices,
    val_row_indices) tuples of plain Python ints -- this is exactly what
    generate_cv_folds persists as folds_id, never a copy of the underlying
    rows.

    Note: target_column is an addition beyond the plan doc's literal
    signature, needed because StratifiedKFold requires labels and
    generate_cv_folds (tools.py) doesn't take target_column as a public
    parameter -- it's recovered from the split's stored metadata and
    threaded through here. Omit it (leave None) for any non-stratified
    strategy.
    """
    n = len(train_pool)
    positions = np.arange(n)
    folds: list[tuple[list[int], list[int]]] = []

    if strategy == "group":
        if not group_column:
            raise ValueError("group_column is required when strategy='group'")
        splitter = GroupKFold(n_splits=n_folds)
        groups = train_pool[group_column].to_numpy()
        for tr_idx, val_idx in splitter.split(positions, groups=groups):
            folds.append((tr_idx.tolist(), val_idx.tolist()))

    elif strategy == "time_series":
        if time_column:
            order = train_pool[time_column].argsort(kind="mergesort").to_numpy()
        else:
            order = positions
        splitter = TimeSeriesSplit(n_splits=n_folds)
        for tr_pos, val_pos in splitter.split(order):
            folds.append((order[tr_pos].tolist(), order[val_pos].tolist()))

    elif strategy == "stratified":
        if not target_column:
            raise ValueError("target_column is required when strategy='stratified'")
        splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
        y = train_pool[target_column].to_numpy()
        for tr_idx, val_idx in splitter.split(positions, y):
            folds.append((tr_idx.tolist(), val_idx.tolist()))

    else:  # 'random'
        splitter = KFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
        for tr_idx, val_idx in splitter.split(positions):
            folds.append((tr_idx.tolist(), val_idx.tolist()))

    return folds


def ks_test_numeric(train_col: pd.Series, test_col: pd.Series) -> tuple[float, float]:
    """Input: the same numeric column's values from the train and test
    partitions. Processing: scipy.stats.ks_2samp, a two-sample
    Kolmogorov-Smirnov test of whether both samples were drawn from the
    same distribution. Output: (statistic, p_value) -- a low p_value flags
    the column as one whose train/test distributions differ more than
    chance would predict."""
    result = stats.ks_2samp(train_col, test_col)
    return float(result.statistic), float(result.pvalue)


def chi_square_categorical(train_col: pd.Series, test_col: pd.Series) -> tuple[float, float]:
    """Input: the same categorical column's values from the train and test
    partitions. Processing: builds a category-frequency contingency table
    and runs a chi-square test of independence between partition and
    category. Output: (statistic, p_value), same interpretation as
    ks_test_numeric but for categorical columns."""
    categories = sorted(set(train_col.unique()) | set(test_col.unique()), key=str)
    train_counts = train_col.value_counts().reindex(categories, fill_value=0).to_numpy()
    test_counts = test_col.value_counts().reindex(categories, fill_value=0).to_numpy()
    contingency = np.vstack([train_counts, test_counts])
    contingency = contingency[:, contingency.sum(axis=0) > 0]
    if contingency.shape[1] < 2:
        # Fewer than 2 non-empty categories -- nothing meaningful to test.
        return 0.0, 1.0
    statistic, p_value, _, _ = stats.chi2_contingency(contingency)
    return float(statistic), float(p_value)


def should_skip_shift_check(n_rows: int, force: bool) -> bool:
    """Input: the train partition's row count and the caller's force flag.
    Processing: a single threshold comparison. Output: True (skip) when
    n_rows > 1000 and force is False -- large samples rarely have a real
    representativeness problem, so validate_split_quality skips the
    (non-trivial) per-column test work by default above that size."""
    return n_rows > 1000 and not force