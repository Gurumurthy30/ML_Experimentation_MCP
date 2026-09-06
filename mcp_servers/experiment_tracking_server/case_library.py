def compute_dataset_fingerprint(profile_summary: dict, column_roles: dict) -> dict:
    """Derives a structural signature from Data Understanding MCP's
    profile_dataset/infer_column_roles output: row-count bucket, the mix
    of column roles present, target balance, task_type. Never touches
    the actual data values."""

def store_case(fingerprint: dict, task_type: str, summary: str, metrics: dict, scope: str = "private") -> None:
    """Inserts a row into cases.sqlite, tagged with scope so 'private'
    and 'shared' cases are distinguishable at retrieval time."""

def find_similar_cases(fingerprint: dict, task_type: str, k: int, scope: str) -> list:
    """Filters cases.sqlite by task_type and scope, ranks by
    fingerprint_distance(), returns the top k."""

def fingerprint_distance(a: dict, b: dict) -> float:
    """A simple weighted distance over the fingerprint's fields (row-count
    bucket, role-mix overlap, target-balance similarity) -- no embeddings
    or vector store needed at this scale."""