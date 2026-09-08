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