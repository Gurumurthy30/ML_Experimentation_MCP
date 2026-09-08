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


mcp = FastMCP("code-execution-server")

mcp.tool()(tools.run_python)
mcp.tool()(tools.install_package)
mcp.tool()(tools.list_available_tools)

if __name__ == "__main__":
    mcp.run(transport="stdio")