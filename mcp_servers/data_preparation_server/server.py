"""
MCP server process for data_preparation_server -- pure wiring, no logic.
Registers the 10 tools from tools.py and runs over stdio. Must be launched
with TABULARML_CACHE_DIR set to the same path every other one of the 5
servers uses (see shared/cache.py).
"""

import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
PARENT_DIR = THIS_DIR.parent

for _p in (THIS_DIR, PARENT_DIR):
    _p_str = str(_p)
    if _p_str not in sys.path:
        sys.path.insert(0, _p_str)

from fastmcp import FastMCP

try:
    from . import tools
except ImportError:
    import tools


mcp = FastMCP("data-preparation-server")

mcp.tool()(tools.split_dataset)
mcp.tool()(tools.generate_cv_folds)
mcp.tool()(tools.validate_split_quality)
mcp.tool()(tools.derive_features)
mcp.tool()(tools.handle_missing_values)
mcp.tool()(tools.encode_categorical)
mcp.tool()(tools.scale_numeric_features)
mcp.tool()(tools.handle_outliers)
mcp.tool()(tools.select_features)
mcp.tool()(tools.build_preprocessing_pipeline)

if __name__ == "__main__":
    mcp.run(transport="stdio")