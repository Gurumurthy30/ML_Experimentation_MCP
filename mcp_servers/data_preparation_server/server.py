from mcp.server.fastmcp import FastMCP
from tools import (
    split_dataset,
    generate_cv_folds,
    validate_split_quality,
    derive_features,
    handle_missing_values,
    encode_categorical,
    scale_numeric_features,
    handle_outliers,
    select_features,
    build_preprocessing_pipeline
)

mcp = FastMCP("data-preparation-server")

mcp.tool()(split_dataset)
mcp.tool()(generate_cv_folds)
mcp.tool()(validate_split_quality)
mcp.tool()(derive_features)
mcp.tool()(handle_missing_values)
mcp.tool()(encode_categorical)
mcp.tool()(scale_numeric_features)
mcp.tool()(handle_outliers)
mcp.tool()(select_features)
mcp.tool()(build_preprocessing_pipeline)

if __name__ == "__main__":
    mcp.run(transport="stdio")