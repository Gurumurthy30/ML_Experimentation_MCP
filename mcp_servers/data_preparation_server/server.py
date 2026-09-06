"""
MCP server process for data_preparation_server -- pure wiring, no logic.
Registers the 10 tools from tools.py and runs over stdio. Must be launched
with TABULARML_CACHE_DIR set to the same path every other one of the 5
servers uses (see shared/cache.py).
"""

from mcp.server.fastmcp import FastMCP

from . import tools

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