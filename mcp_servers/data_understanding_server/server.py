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
    infer_column_roles,
    profile_dataset,
    analyze_target_and_infer_task,
    analyze_missing_values,
    detect_outliers,
    detect_target_leakage,
    measure_associations,
)

mcp = FastMCP("data-understanding-server")

mcp.tool()(load_dataset)
mcp.tool()(infer_column_roles)
mcp.tool()(profile_dataset)
mcp.tool()(analyze_target_and_infer_task)
mcp.tool()(analyze_missing_values)
mcp.tool()(detect_outliers)
mcp.tool()(detect_target_leakage)
mcp.tool()(measure_associations)

if __name__ == "__main__":
    mcp.run(transport="stdio")