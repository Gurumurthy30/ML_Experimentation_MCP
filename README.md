# TabularML MCP: Machine Learning Experimentation MCP Suite

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![MCP Protocol](https://img.shields.io/badge/MCP-1.0.0-purple.svg)](https://modelcontextprotocol.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**TabularML MCP** is a production-ready suite of 5 Model Context Protocol (MCP) servers that empower AI Agents (Claude Desktop, Antigravity, Cursor, etc.) to autonomously perform end-to-end Machine Learning experimentation on tabular datasets — with zero data leakage guarantees, prompt-optimized tool schemas, and sandboxed code execution.

---

## 🌟 Architecture Overview

```
                      +-----------------------------------+
                      |         Raw Dataset (CSV/Parquet) |
                      +-----------------+-----------------+
                                        |
                                        v [load_dataset]
                      +-----------------+-----------------+
                      |    1. Data Understanding Server   |
                      | (Profiling, Roles, Leakage Audit) |
                      +-----------------+-----------------+
                                        |
                                        v [split_dataset]  <--- 🛡️ LEAKAGE FIREWALL
                     /------------------+------------------\
                    /                                       \
                   v                                         v
   +---------------+---------------+          +--------------+--------------+
   | Train Partition (train.parquet)|          | Test Partition (test.parquet)|
   +---------------+---------------+          +--------------+--------------+
                   |                                   |  (Held out & locked)
                   v                                   |
   +---------------+---------------+                   |
   |    2. Data Preparation Server |                   |
   | (Impute, Scale, Encode Specs) |                   |
   +---------------+---------------+                   |
                   |                                   |
                   v [pipeline_spec_id]                |
   +---------------+---------------+                   |
   |       3. Modeling Server      |                   |
   | (Per-Fold CV, Baseline, Fit)  |                   |
   +---------------+---------------+                   |
                   |                                   |
                   v [finalize_model]                  |
   +---------------+---------------+                   |
   |        Persisted Model        |                   |
   +---------------+---------------+                   |
                   |                                   |
                   +-----------------+-----------------+
                                     |
                                     v [final_test_evaluation] (One-shot)
                   +-----------------+-----------------+
                   | 4. Experiment Tracking Server     |
                   | (MLflow & Case Library Logging)   |
                   +-----------------------------------+
```

---

## 🛠️ The 5 MCP Servers

| Server | Role & Purpose | Key Capabilities |
| :--- | :--- | :--- |
| 🔍 **`data_understanding_server`** | Dataset Ingestion & Diagnostics | File loading, 9-role semantic column inference, per-column profiling, task type inference, MAR missingness detection, advisory target leakage audits, association metrics. |
| 🧹 **`data_preparation_server`** | Leakage-Safe Preprocessing Recipes | Split firewall creation (`train`/`test`), CV fold generation, imputation, encoding, scaling, outlier capping, feature selection, and unfitted sklearn `Pipeline` spec assembly. |
| 🤖 **`modeling_server`** | Candidate Evaluation & Finalization | Baseline benchmarks, per-fold CV scoring, fit gap diagnosis (overfitting/underfitting), regularization paths, class imbalance handling, probability calibration, threshold tuning, and final model evaluation. |
| 📊 **`experiment_tracking_server`** | Experiment Governance | MLflow tracking integration (experiments, runs, params, metrics, artifacts) and an anonymized dataset fingerprint case library for historical retrieval. |
| ⚡ **`code_execution_server`** | Sandboxed Custom Analysis | Subprocess-isolated Python code execution with handle auto-injection, security allowlist package installer, and auto-generated MCP client stubs. |

---

## 🛡️ Key Features & Guarantees

- **Strict Data Leakage Protection**: Data preparation steps generate recipe specifications rather than fitted transformers. All sklearn `.fit()` operations occur inside cross-validation folds on training indices only. `test.parquet` is locked away and only readable during `final_test_evaluation` (enforcing a one-shot evaluation rule).
- **Prompt-Optimized Tool Schemas**: Every tool function features structured docstrings formatted specifically for LLM agents — detailing purpose, when to use, input constraints, return schemas, and operational guidelines.
- **Hash-Based Handle Architecture**: Datasets, pipeline specifications, CV folds, and trained models are stored immutably in a local cache directory (`_cache/`) and referenced via unique hash handle IDs (`dataset_id`, `split_id`, `folds_id`, `pipeline_spec_id`, `model_id`).
- **Windows-Compatible Sandbox**: `code_execution_server` uses an out-of-process subprocess execution sandbox with handle auto-injection and output capping — no Docker daemon required.

---

## 📂 Project Structure

```
ML_Experimentation_MCP/
├── mcp_servers/
│   ├── shared/                        # Shared schemas, cache engine & handle hashing
│   ├── data_understanding_server/     # Ingestion, profiling, target & leakage audits
│   ├── data_preparation_server/       # Splitting firewall, spec builders & pipeline assembly
│   ├── modeling_server/               # Baseline, per-fold CV, tuning & evaluation
│   ├── experiment_tracking_server/    # MLflow integration & dataset case library
│   └── code_execution_server/         # Subprocess sandbox, package policy & tool stubs
├── scripts/
│   ├── seed_sample_data.py            # Generates Iris, California Housing & Breast Cancer CSVs
│   ├── smoke_test_server.py           # 18-step end-to-end pipeline integration test
│   └── test_all_tools.py              # Verification script testing all 44 tools
├── tests/                             # Pytest unit test suite
│   ├── conftest.py
│   ├── test_data_understanding_tools.py
│   ├── test_data_preparation_tools.py
│   ├── test_code_execution_tools.py
│   └── test_leakage_guarantees.py
├── pyproject.toml                     # Dependency & package configuration
└── README.md
```

---

## 🚀 Quickstart & Setup

### 1. Prerequisites
- Python **3.10** or higher
- Git

### 2. Installation
Clone the repository and set up a virtual environment:

```bash
git clone https://github.com/Gurumurthy30/ML_Experimentation_MCP.git
cd ML_Experimentation_MCP

# Create & activate virtual environment
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -e .
pip install pytest mlflow-skinny
```

### 3. Seed Sample Datasets
Generate sample benchmark datasets (`iris.csv`, `california_housing.csv`, `breast_cancer.csv`):

```bash
python scripts/seed_sample_data.py
```

---

## 🧪 Testing & Verification

### Run End-to-End Integration Smoke Test (18 Steps)
Exercises the full ML pipeline from dataset ingestion through final test evaluation:

```bash
python scripts/smoke_test_server.py
```

### Test All 44 MCP Tools Across All 5 Servers
Runs a comprehensive test verifying that every single tool function executes cleanly and returns valid JSON dictionaries:

```bash
python scripts/test_all_tools.py
```

### Run Pytest Unit Test Suite
Runs unit tests covering leakage guarantees, data preparation, understanding, and code execution policies:

```bash
python -m pytest tests/
```

---

## 🔌 Connecting to MCP Clients

To connect TabularML MCP servers to MCP-compatible clients (e.g. Claude Desktop, Antigravity, or Cursor), add the servers to your MCP configuration file (e.g., `mcpServers` in `claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "tabularml_data_understanding": {
      "command": "python",
      "args": ["-m", "mcp_servers.data_understanding_server.server"],
      "cwd": "C:/Users/gurum/Documents/4_Projects/ML_Experimentation_MCP"
    },
    "tabularml_data_preparation": {
      "command": "python",
      "args": ["-m", "mcp_servers.data_preparation_server.server"],
      "cwd": "C:/Users/gurum/Documents/4_Projects/ML_Experimentation_MCP"
    },
    "tabularml_modeling": {
      "command": "python",
      "args": ["-m", "mcp_servers.modeling_server.server"],
      "cwd": "C:/Users/gurum/Documents/4_Projects/ML_Experimentation_MCP"
    },
    "tabularml_experiment_tracking": {
      "command": "python",
      "args": ["-m", "mcp_servers.experiment_tracking_server.server"],
      "cwd": "C:/Users/gurum/Documents/4_Projects/ML_Experimentation_MCP"
    },
    "tabularml_code_execution": {
      "command": "python",
      "args": ["-m", "mcp_servers.code_execution_server.server"],
      "cwd": "C:/Users/gurum/Documents/4_Projects/ML_Experimentation_MCP"
    }
  }
}
```

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
