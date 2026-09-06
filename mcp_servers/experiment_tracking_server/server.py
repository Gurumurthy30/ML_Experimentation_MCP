from mcp.server.fastmcp import FastMCP
from . import tools

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