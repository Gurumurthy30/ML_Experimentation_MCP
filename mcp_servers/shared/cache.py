from pathlib import Path
import hashlib
import json
import os
import pandas as pd


def get_cache_dir() -> Path:
    """Reads the TABULARML_CACHE_DIR env var. Default is an absolute
    path (<project_root>/data/_cache), NOT a relative one, so all
    server processes resolve to the same directory regardless of each
    process's working directory at launch time."""
    project_root = Path(__file__).resolve().parent.parent.parent
    default_dir = project_root / "data" / "_cache"
    cache_dir = Path(os.environ.get("TABULARML_CACHE_DIR", str(default_dir))).resolve()
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir



def compute_handle_id(*parts) -> str:
    """sha1 of the JSON-serialized parts (order preserved, not sorted),
    truncated to 12 chars. Same inputs, same order, always produce the
    same ID."""
    m = hashlib.sha1()
    m.update(json.dumps(parts, default=str).encode("utf-8"))
    return m.hexdigest()[:12]


def save_object(obj, subdir: str, handle_id: str) -> None:
    """Writes obj under {cache_dir}/{subdir}/{handle_id}.*, choosing
    .parquet for DataFrames and .joblib for everything else (unfitted
    pipeline specs, fitted models, any other sklearn object)."""
    path = get_cache_dir() / subdir
    os.makedirs(path, exist_ok=True)
    if isinstance(obj, pd.DataFrame):
        obj.to_parquet(path / f"{handle_id}.parquet", index=False)
    else:
        import joblib
        joblib.dump(obj, path / f"{handle_id}.joblib")


def load_object(subdir: str, handle_id: str):
    """Reverse of save_object -- reconstitutes the real DataFrame, spec,
    or model from disk given just the handle_id string. This is the call
    every tool makes as its first line when it receives a handle as an
    argument."""
    path = get_cache_dir() / subdir
    parquet_path = path / f"{handle_id}.parquet"
    joblib_path = path / f"{handle_id}.joblib"
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    elif joblib_path.exists():
        import joblib
        return joblib.load(joblib_path)
    else:
        raise FileNotFoundError(f"Cached object not found: {subdir}/{handle_id}")


def object_exists(subdir: str, handle_id: str) -> bool:
    """Cheap existence check, used before recomputing anything expensive
    -- e.g. finalize_model checks this before actually fitting."""
    path = get_cache_dir() / subdir
    parquet_path = path / f"{handle_id}.parquet"
    joblib_path = path / f"{handle_id}.joblib"
    return parquet_path.exists() or joblib_path.exists()