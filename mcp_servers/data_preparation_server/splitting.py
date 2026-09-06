def auto_detect_split_strategy(df: pd.DataFrame, target_column: str, group_column: str | None, time_column: str | None) -> str:
    """Resolution order: time_column present -> 'time_series'; a given
    group_column that actually repeats (n_unique < n_rows) -> 'group';
    task_type is classification -> 'stratified'; else 'random'."""

def stratified_split(df, target_column, test_size, random_state) -> tuple:
    """sklearn.model_selection.train_test_split with
    stratify=df[target_column]."""

def group_split(df, group_column, test_size, random_state) -> tuple:
    """sklearn.model_selection.GroupShuffleSplit keyed on group_column --
    guarantees no group value appears in both resulting partitions."""

def time_series_split(df, time_column, test_size) -> tuple:
    """Sorts by time_column and takes a chronological tail as the test
    partition -- no shuffling, since shuffling a temporal split leaks
    the future into training."""

def compute_adaptive_fold_count(n_rows: int) -> int:
    """Returns 10 under 1000 rows, 5 between 1000-50000, 3 above 50000."""

def generate_folds(train_pool: pd.DataFrame, strategy: str, n_folds: int, group_column: str = None, time_column: str = None) -> list:
    """Dispatches to StratifiedKFold / GroupKFold / TimeSeriesSplit based
    on strategy, matching whatever split_dataset originally resolved.
    Returns a list of (train_indices, val_indices) tuples."""

def ks_test_numeric(train_col: pd.Series, test_col: pd.Series) -> tuple[float, float]:
    """scipy.stats.ks_2samp -- tests whether train and test come from the
    same distribution for one numeric column. Returns (statistic,
    p_value)."""

def chi_square_categorical(train_col: pd.Series, test_col: pd.Series) -> tuple[float, float]:
    """Chi-square test of independence on the two partitions' category
    frequency tables."""

def should_skip_shift_check(n_rows: int, force: bool) -> bool:
    """True (skip) when n_rows > 1000 and force is False."""