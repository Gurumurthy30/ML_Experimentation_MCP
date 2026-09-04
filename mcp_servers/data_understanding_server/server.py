"""
Entry point for the dataset-understanding MCP server.

Runs either as a plain script:
    python server.py
or as a module from the project root (requires __init__.py in both
`mcp_servers/` and `mcp_servers/data_understanding_server/`):
    python -m mcp_servers.data_understanding_server.server
"""

from fastmcp import FastMCP
from tools import (
    load_dataset,
    infer_column_eoles,
    profile_dataset,
    analyze_target_and_infer_task_type,
    analyze_missing_values,
    detect_outliers,
    detect_correlations,
    detect_multicollinearity,
)

mcp = FastMCP("dataset-server")

mcp.tool()(load_dataset)
mcp.tool()(infer_column_eoles)
mcp.tool()(profile_dataset)
mcp.tool()(analyze_target_and_infer_task_type)
mcp.tool()(analyze_missing_values)
mcp.tool()(detect_outliers)
mcp.tool()(detect_correlations)
mcp.tool()(detect_multicollinearity)

if __name__ == "__main__":
    mcp.run(transport="stdio")