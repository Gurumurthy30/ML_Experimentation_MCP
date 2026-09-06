from mcp.server.fastmcp import FastMCP
from . import tools

mcp = FastMCP("modeling-server")

mcp.tool()(tools.establish_baseline)
mcp.tool()(tools.cross_validate_model)
mcp.tool()(tools.diagnose_fit)
mcp.tool()(tools.regularization_path_search)
mcp.tool()(tools.handle_class_imbalance)
mcp.tool()(tools.calibrate_probabilities)
mcp.tool()(tools.tune_decision_threshold)
mcp.tool()(tools.explain_predictions)
mcp.tool()(tools.analyze_prediction_errors)
mcp.tool()(tools.finalize_model)
mcp.tool()(tools.final_test_evaluation)

if __name__ == "__main__":
    mcp.run(transport="stdio")