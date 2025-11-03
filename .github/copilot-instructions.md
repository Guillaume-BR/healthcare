## Quick context for AI coding assistants

This repository contains an exploratory data analysis / ML notebook and small utility helpers for a healthcare dataset. Aim to be conservative: do not run code that expects private data, and prefer edits that are small, well-tested, and reversible.

Key files and folders
- `notebook/data_cleaning.ipynb` — main EDA + preprocessing notebook. Shows data loading from `../data/raw/dirty_v3_path.csv` and many analysis steps (visualization, normality tests, feature checks).
- `utils/functions.py` — helper utilities (e.g. `describe_dataframe`). `utils/__init__.py` is empty.
- `data/processed/` — target for processed outputs. `requirements.txt` exists but is empty (assume dependencies are documented inline in notebooks).

Project architecture & patterns
- This is primarily a notebook-driven project (EDA → modeling). There is no service, API or package structure. Prefer small, incremental changes that keep notebooks runnable.
- Data paths in notebooks are relative (`../data/raw/...`). When modifying code, prefer using `pathlib.Path` or a configurable `DATA_PATH` variable rather than hard-coding absolute paths.
- Utility functions are minimal; reuse `utils/functions.py` for shared helpers and export any new helpers there with docstrings and unit tests if applicable.

Developer workflows (discoverable)
- To run notebooks locally: open `notebook/data_cleaning.ipynb` in VS Code or Jupyter. Do not assume remote GPUs or large datasets are available.
- There is no test runner configured. If you add tests, follow standard `pytest` conventions and add a `requirements.txt` update.

Conventions and code style
- Notebooks contain helper function definitions (e.g., `print_divider`, `describe_dataframe`). When moving code into `utils/`, preserve function names and update imports in the notebooks.
- Keep changes minimal in notebooks: prefer creating small Python modules under `utils/` and import them from the notebook instead of editing large notebook cells.

Integration points & external dependencies
- Notebooks import many ML and viz libraries (scikit-learn, xgboost, lightgbm, catboost, tensorflow, plotly, shap, eli5). Because `requirements.txt` is empty, do not attempt to install or import heavy libs by default — ask the repo owner or add only required, pinned dependencies.

Safety & permissions
- Dataset files may be sensitive. Avoid adding code that uploads or shares data. If adding sample data or fixtures, use synthetic or anonymized examples and note this in the PR.

When making edits
- Prefer edits that are: small, documented, and reversible. Include unit tests for new Python modules. Update `requirements.txt` when introducing new dependencies.
- If you change data-loading logic, make the path configurable and keep original hard-coded path as a fallback.

Examples to reference
- Use `notebook/data_cleaning.ipynb` to see how `DATA_PATH` is used and how `describe_dataframe` is expected to behave.
- `utils/functions.py` contains a copy of `describe_dataframe` — consider consolidating duplicate implementations when refactoring.

If anything is unclear
- Ask for the dataset location, intended runtime environment (local vs cloud), and whether adding a dependency to `requirements.txt` is acceptable.

End of instructions — keep edits concise and repository-specific.
