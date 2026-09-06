"""
pytest fixtures for ML Experimentation MCP server tests.
"""

import os
import sys
import tempfile
from pathlib import Path

import pandas as pd
import pytest

# Ensure project root & mcp_servers are in sys.path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "mcp_servers"))


@pytest.fixture(autouse=True)
def isolated_cache_dir():
    """Isolated cache directory for each test."""
    with tempfile.TemporaryDirectory(prefix="tabularml_test_") as tmpdir:
        old_env = os.environ.get("TABULARML_CACHE_DIR")
        os.environ["TABULARML_CACHE_DIR"] = tmpdir
        yield Path(tmpdir)
        if old_env is not None:
            os.environ["TABULARML_CACHE_DIR"] = old_env
        else:
            os.environ.pop("TABULARML_CACHE_DIR", None)


@pytest.fixture
def sample_classification_df():
    """Generates a small synthetic classification dataframe."""
    import numpy as np
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        "num_col1": np.random.randn(n),
        "num_col2": np.random.randn(n) * 10 + 5,
        "cat_col1": np.random.choice(["A", "B", "C"], size=n),
        "cat_col2": np.random.choice(["low", "high"], size=n),
        "target": np.random.choice([0, 1], size=n, p=[0.7, 0.3]),
    })
    # Add a few missing values
    df.loc[5:10, "num_col1"] = np.nan
    df.loc[15:18, "cat_col1"] = np.nan
    return df
