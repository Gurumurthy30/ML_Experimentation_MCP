from mcp.server.fastmcp import FastMCP
from . import tools

mcp = FastMCP("training-server")

mcp.tool()(tools.split_dataset)
mcp.tool()(tools.create_preprocessing_pipeline)
mcp.tool()(tools.train_model)
mcp.tool()(tools.evaluate_model)
mcp.tool()(tools.cross_validate_model)
mcp.tool()(tools.predict)

if __name__ == "__main__":
    mcp.run(transport="stdio")