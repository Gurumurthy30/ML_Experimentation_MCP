"""
Seed sample datasets into data/samples/ for smoke testing.

Downloads/generates datasets for classification (iris, churn) and
regression (California housing) use.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

SAMPLES_DIR = Path(__file__).parent.parent / "data" / "samples"


def _download(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"  Already exists: {dest.name}")
        return
    print(f"  Downloading {dest.name}...")
    urllib.request.urlretrieve(url, dest)
    print(f"  Saved: {dest}")


def seed_california_housing() -> Path:
    dest = SAMPLES_DIR / "california_housing.csv"
    if dest.exists():
        print(f"  Already exists: {dest.name}")
        return dest
    print("  Generating california_housing.csv from sklearn...")
    from sklearn.datasets import fetch_california_housing
    import pandas as pd
    data = fetch_california_housing(as_frame=True)
    data.frame.to_csv(dest, index=False)
    print(f"  Saved: {dest} ({len(data.frame)} rows)")
    return dest


def seed_iris() -> Path:
    dest = SAMPLES_DIR / "iris.csv"
    if dest.exists():
        print(f"  Already exists: {dest.name}")
        return dest
    print("  Generating iris.csv from sklearn...")
    from sklearn.datasets import load_iris
    import pandas as pd
    data = load_iris(as_frame=True)
    data.frame.to_csv(dest, index=False)
    print(f"  Saved: {dest} ({len(data.frame)} rows)")
    return dest


def seed_breast_cancer() -> Path:
    dest = SAMPLES_DIR / "breast_cancer.csv"
    if dest.exists():
        print(f"  Already exists: {dest.name}")
        return dest
    print("  Generating breast_cancer.csv from sklearn...")
    from sklearn.datasets import load_breast_cancer
    import pandas as pd
    data = load_breast_cancer(as_frame=True)
    data.frame.to_csv(dest, index=False)
    print(f"  Saved: {dest} ({len(data.frame)} rows)")
    return dest


if __name__ == "__main__":
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Seeding sample data into {SAMPLES_DIR}/\n")

    seed_california_housing()
    seed_iris()
    seed_breast_cancer()

    print("\nAvailable datasets:")
    for f in sorted(SAMPLES_DIR.glob("*.csv")):
        size_kb = f.stat().st_size // 1024
        print(f"  {f.name} ({size_kb} KB)")
