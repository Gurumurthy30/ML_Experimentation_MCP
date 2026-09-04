from pathlib import Path
import hashlib
import pandas as pd
import os


def get_cache_dir() -> Path:
    """
    Return the root cache directory.
    """
    cache_dir = Path("../Data/_Cache")
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir

def get_dataset_cache_dir() -> Path:
    """
    Return the dataset cache directory.
    """
    dataset_cache_dir = get_cache_dir() / "datasets"
    os.makedirs(dataset_cache_dir, exist_ok=True)
    return dataset_cache_dir

def compute_dataset_id(path: str) -> str:
    """
    Generate a deterministic ID for a dataset.

    The ID changes if:
    - absolute path changes
    - modification time changes
    - file size changes
    """
    file_path = Path(path).resolve()
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")
    stat = file_path.stat()
    identity = (f"{file_path}"f"|{stat.st_mtime_ns}"f"|{stat.st_size}")
    return hashlib.sha1(identity.encode("utf-8")).hexdigest()[:12]


def dataset_parquet_path(dataset_id: str) -> Path:
    """Return the cache path for a dataset ID."""
    return get_dataset_cache_dir() / f"{dataset_id}.parquet"


def save_dataset(df: pd.DataFrame, dataset_id: str) -> Path:
    """Save DataFrame into the dataset cache."""
    path = dataset_parquet_path(dataset_id)
    df.to_parquet(path,index=False)
    return path


def load_dataset_df(dataset_id: str) -> pd.DataFrame:
    """Load a cached dataset using its ID."""
    path = dataset_parquet_path(dataset_id)
    if not path.exists():
        raise FileNotFoundError(f"Cached dataset not found: {dataset_id}")
    return pd.read_parquet(path)