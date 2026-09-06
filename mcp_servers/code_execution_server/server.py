from mcp.server.fastmcp import FastMCP
from . import tools

mcp = FastMCP("code-execution-server")

mcp.tool()(tools.run_python)
mcp.tool()(tools.install_package)
mcp.tool()(tools.list_available_tools)

if __name__ == "__main__":
    mcp.run(transport="stdio")