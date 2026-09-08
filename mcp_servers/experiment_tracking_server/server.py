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


mcp = FastMCP("experiment-tracking-server")

mcp.tool()(tools.create_experiment)
mcp.tool()(tools.start_run)
mcp.tool()(tools.log_parameters)
mcp.tool()(tools.log_metrics)
mcp.tool()(tools.log_artifact)
mcp.tool()(tools.end_run)
mcp.tool()(tools.get_run)
mcp.tool()(tools.compare_runs)
mcp.tool()(tools.get_best_run)
mcp.tool()(tools.log_case)
mcp.tool()(tools.retrieve_similar_cases)

if __name__ == "__main__":
    mcp.run(transport="stdio")