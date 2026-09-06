"""
Case library -- stores and retrieves dataset fingerprints from a local SQLite
database at data/_cache/cases/cases.sqlite.

A fingerprint is a structural signature of a dataset (row-count bucket, role
mix, target balance, task_type) -- NEVER actual data values -- so cases are
shareable without leaking anyone's actual data.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Optional

from shared.cache import get_cache_dir


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def _get_db_path() -> Path:
    cases_dir = get_cache_dir() / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)
    return cases_dir / "cases.sqlite"


def _get_conn() -> sqlite3.Connection:
    db_path = _get_db_path()
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    _ensure_table(conn)
    return conn


def _ensure_table(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS cases (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fingerprint_json TEXT NOT NULL,
            task_type TEXT NOT NULL,
            approach_summary TEXT NOT NULL,
            outcome_metrics_json TEXT NOT NULL,
            scope TEXT NOT NULL DEFAULT 'private',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_dataset_fingerprint(profile_summary: dict, column_roles: dict) -> dict:
    """Derives a structural signature from Data Understanding MCP's
    profile_dataset/infer_column_roles output: row-count bucket, the mix
    of column roles present, target balance, task_type.  Never touches
    the actual data values."""

    # Row-count bucket
    n_rows = sum(
        v.get("missing_pct", 0) >= 0  # just a truthy check; count from profile keys
        for v in profile_summary.values()
    )
    # Better: try to extract n_rows from a special key if present
    n_rows = profile_summary.get("_n_rows", len(profile_summary))
    if n_rows < 1000:
        row_bucket = "tiny"
    elif n_rows < 10_000:
        row_bucket = "small"
    elif n_rows < 100_000:
        row_bucket = "medium"
    else:
        row_bucket = "large"

    # Role mix: count each role type
    role_counts: dict[str, int] = {}
    for col, info in column_roles.items():
        role = info if isinstance(info, str) else info.get("role", "unknown")
        role_counts[role] = role_counts.get(role, 0) + 1

    return {
        "row_bucket": row_bucket,
        "n_columns": len(column_roles),
        "role_mix": role_counts,
    }


def store_case(
    fingerprint: dict,
    task_type: str,
    summary: str,
    metrics: dict,
    scope: str = "private",
) -> None:
    """Inserts a row into cases.sqlite, tagged with scope so 'private' and
    'shared' cases are distinguishable at retrieval time."""
    conn = _get_conn()
    try:
        conn.execute(
            """
            INSERT INTO cases (fingerprint_json, task_type, approach_summary,
                               outcome_metrics_json, scope)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                json.dumps(fingerprint, sort_keys=True),
                task_type,
                summary,
                json.dumps(metrics, sort_keys=True),
                scope,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def find_similar_cases(
    fingerprint: dict,
    task_type: str,
    k: int,
    scope: str,
) -> list[dict]:
    """Filters cases.sqlite by task_type and scope, ranks by
    fingerprint_distance(), returns the top k."""
    conn = _get_conn()
    try:
        if scope == "all":
            rows = conn.execute(
                "SELECT * FROM cases WHERE task_type = ?",
                (task_type,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM cases WHERE task_type = ? AND scope = ?",
                (task_type, scope),
            ).fetchall()
    finally:
        conn.close()

    if not rows:
        return []

    scored = []
    for row in rows:
        fp = json.loads(row["fingerprint_json"])
        dist = fingerprint_distance(fingerprint, fp)
        scored.append({
            "id": row["id"],
            "fingerprint": fp,
            "task_type": row["task_type"],
            "approach_summary": row["approach_summary"],
            "outcome_metrics": json.loads(row["outcome_metrics_json"]),
            "scope": row["scope"],
            "created_at": row["created_at"],
            "distance": dist,
        })

    scored.sort(key=lambda x: x["distance"])
    return scored[:k]


def fingerprint_distance(a: dict, b: dict) -> float:
    """A simple weighted distance over the fingerprint's fields (row-count
    bucket, role-mix overlap, target-balance similarity) -- no embeddings
    or vector store needed at this scale."""
    distance = 0.0

    # Row bucket mismatch (categorical)
    BUCKET_ORDER = ["tiny", "small", "medium", "large"]
    a_bucket = a.get("row_bucket", "medium")
    b_bucket = b.get("row_bucket", "medium")
    if a_bucket != b_bucket:
        a_idx = BUCKET_ORDER.index(a_bucket) if a_bucket in BUCKET_ORDER else 2
        b_idx = BUCKET_ORDER.index(b_bucket) if b_bucket in BUCKET_ORDER else 2
        distance += abs(a_idx - b_idx) * 0.3

    # Column count difference (normalized)
    a_cols = a.get("n_columns", 0)
    b_cols = b.get("n_columns", 0)
    max_cols = max(a_cols, b_cols, 1)
    distance += abs(a_cols - b_cols) / max_cols * 0.2

    # Role mix overlap (Jaccard-like)
    a_roles = a.get("role_mix", {})
    b_roles = b.get("role_mix", {})
    all_roles = set(a_roles) | set(b_roles)
    if all_roles:
        a_total = max(sum(a_roles.values()), 1)
        b_total = max(sum(b_roles.values()), 1)
        role_distance = 0.0
        for role in all_roles:
            a_frac = a_roles.get(role, 0) / a_total
            b_frac = b_roles.get(role, 0) / b_total
            role_distance += abs(a_frac - b_frac)
        distance += role_distance / len(all_roles) * 0.5

    return round(distance, 4)