"""
Constructs and executes the end-to-end Jupyter Notebook
`Project Attempt 1/notebooks/semiconductor_cointegration_ml_workflow.ipynb`
calling all modular `.py` functions in `src/` and embedding cell outputs and charts.
"""

from pathlib import Path
import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook
from nbclient import NotebookClient

PROJECT_DIR = Path(__file__).resolve().parent.parent
NOTEBOOKS_DIR = PROJECT_DIR / "notebooks"
NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)
NOTEBOOK_PATH = NOTEBOOKS_DIR / "semiconductor_cointegration_ml_workflow.ipynb"


def build_notebook() -> nbformat.NotebookNode:
    nb = new_notebook()
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }

    cells = [
        new_markdown_cell(
            "# US & Taiwan Semiconductor Supply-Chain Cointegration & Machine Learning Strategy\n"
            "**Stock Markets Analytics Zoomcamp 2026 — Capstone Project (Attempt 1)**\n\n"
            "This end-to-end research and automation notebook executes the complete quantitative workflow by importing "
            "and calling the modular `.py` packages in [`src/`](../src/):\n"
            "1. **Data Sources & Loading (`src.data_loader`)**: Combines daily OHLCV for 23 US-listed semiconductor equities, "
            "4 Taiwanese supply-chain bellwethers (`2330.TW`, `2303.TW`, `2454.TW`, `3711.TW`), sector/market benchmarks "
            "(`SOXX`, `SMH`, `^GSPC`, `^VIX`, `TWD=X`, `DX-Y.NYB`), 5 FRED macroeconomic series (`DGS10`, `DGS2`, `T10Y2Y`, "
            "`FEDFUNDS`, `BAMLH0A0HYM2`), and **point-in-time quarterly accounting fundamentals from the SEC EDGAR XBRL API**.\n"
            "2. **Transformations, Cointegration & EDA (`src.transformations`)**: Engineers 28 new domain-specific features "
            "(rolling 60-day OLS hedge ratio $\\beta_t$, log-price spread $S_t$, 20d/60d Spread Z-Scores $Z_t$, rolling ADF "
            "stationarity proxy, Ornstein-Uhlenbeck half-life $\\tau$, TSMC ADR premium, Taiwan supply-chain momentum, and "
            "SEC EDGAR inventory/revenue ratios), binned regime variables, and seasonal calendar dummies.\n"
            "3. **Machine Learning & Hyperparameter Tuning (`src.modeling`)**: Compares 4 domain hand-crafted rules with "
            "tuned `DecisionTreeClassifier`, `RandomForestClassifier`, `XGBClassifier`, penalized `Ridge` regression, "
            "and a 2-layer Deep Neural Network (`MLPClassifier`), culminating in the Cointegration + ML Consensus Hybrid rule.\n"
            "4. **Trading Simulation & Subgroup Diagnostics (`src.simulation`)**: Evaluates vector simulations (`sim1_` raw and "
            "`sim2_` stop-loss/take-profit managed) and an exact chronological `iterrows()` portfolio simulator (`sim3_`) with "
            "compounding capital reinvestment, Top-K concurrent position limits, and dynamic position sizing.\n"
            "5. **Live 10-Day Trade Signals & Broker API (`src.broker_api` & `src.monitor`)**: Produces the latest trade "
            "orders for the next 10 trading days with exact share quantities and bracket Stop-Loss / Take-Profit levels."
        ),
        new_code_cell(
            "import sys\n"
            "from pathlib import Path\n"
            "import json\n"
            "import numpy as np\n"
            "import pandas as pd\n"
            "import matplotlib.pyplot as plt\n"
            "from IPython.display import Image, display\n\n"
            "# Add Project Attempt 1 root to sys.path so `src` modules import cleanly\n"
            "PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\n"
            "if str(PROJECT_ROOT) not in sys.path:\n"
            "    sys.path.insert(0, str(PROJECT_ROOT))\n\n"
            "from src.config import (\n"
            "    COMBINED_DATA_PATH, FIGURES_DIR, REPORTS_DIR,\n"
            "    SUPPLY_CHAIN_PEER_MAP, TARGET_BINARY_COL, TARGET_GROWTH_COL, US_SEMI_TICKERS\n"
            ")\n"
            "from src.data_loader import load_or_download_raw_data\n"
            "from src.transformations import build_unified_dataset, get_feature_sets, run_pair_cointegration_adf_summary\n"
            "from src.modeling import evaluate_predictions_on_test, tune_and_train_models\n"
            "from src.simulation import (\n"
            "    run_deep_exploratory_error_analysis,\n"
            "    run_exact_iterrows_portfolio_simulation,\n"
            "    run_vector_simulations,\n"
            ")\n"
            "from src.broker_api import generate_latest_live_signals, submit_orders_to_broker\n"
            "from src.monitor import build_html_monitoring_dashboard, generate_all_charts\n\n"
            "pd.set_option('display.max_columns', 30)\n"
            "pd.set_option('display.width', 140)\n"
            "print(f'Project Root: {PROJECT_ROOT}')"
        ),
        new_markdown_cell(
            "## 1. Data Loading & Unified Dataset Construction (`mode='local'` / `'download'`)\n"
            "We load the unified multi-source dataset combining Yahoo Finance OHLCV (23 US Semi tickers + 4 Taiwan TWSE "
            "anchors + ETFs/FX/VIX), FRED macroeconomic yields, and point-in-time SEC EDGAR XBRL fundamentals."
        ),
        new_code_cell(
            "if COMBINED_DATA_PATH.exists():\n"
            "    df_base = pd.read_parquet(COMBINED_DATA_PATH)\n"
            "else:\n"
            "    df_ohlcv, df_ctx, df_sec = load_or_download_raw_data(mode='local', incremental=False)\n"
            "    df_base = build_unified_dataset(df_ohlcv, df_ctx, df_sec, save_parquet=True)\n\n"
            "df_full, feature_sets = get_feature_sets(df_base)\n"
            "print(f'Unified Dataset Shape: {df_full.shape[0]:,} rows x {df_full.shape[1]} columns')\n"
            "print('Date Range:', df_full['Date'].min().date(), 'to', df_full['Date'].max().date())\n"
            "for k, v in feature_sets.items():\n"
            "    print(f'  Feature Set {k:22s}: {len(v):3d} columns')\n\n"
            "df_full[['Date', 'Ticker', 'sub_vertical', 'peer_ticker', 'Close', 'pair_spread_zscore_20d', 'sec_rev_yoy', 'split']].tail(8)"
        ),
        new_markdown_cell(
            "## 2. Exploratory Data Analysis (EDA) & Supply-Chain Cointegration Verification\n"
            "We test all semiconductor sub-vertical supply-chain pairs for stationarity using the Augmented Dickey-Fuller (ADF) "
            "test on the rolling cointegrated log-price spread $S_t = \\ln(P_{A,t}) - (\\alpha_t + \\beta_t \\ln(P_{B,t}))$, "
            "and examine feature correlations with the 10-trading-day forward return target."
        ),
        new_code_cell(
            "df_coint_adf = run_pair_cointegration_adf_summary(df_full)\n"
            "display(df_coint_adf)\n\n"
            "# Target distribution summary by temporal split\n"
            "split_summary = (\n"
            "    df_full.groupby('split')\n"
            "    .agg(\n"
            "        Records=('Ticker', 'count'),\n"
            "        Min_Date=('Date', 'min'),\n"
            "        Max_Date=('Date', 'max'),\n"
            "        Positive_10d_Rate=(TARGET_BINARY_COL, 'mean'),\n"
            "        Mean_10d_Growth=(TARGET_GROWTH_COL, 'mean'),\n"
            "    )\n"
            "    .reset_index()\n"
            ")\n"
            "display(split_summary)"
        ),
        new_code_cell(
            "# Top 15 numerical features correlated with 10-day positive future growth\n"
            "num_cols = feature_sets['NUMERIC']\n"
            "corrs = df_full[num_cols].apply(lambda s: s.corr(df_full[TARGET_BINARY_COL])).dropna()\n"
            "df_corr = pd.DataFrame({\n"
            "    'Feature': corrs.index,\n"
            "    'Correlation': corrs.values,\n"
            "    'Abs_Correlation': corrs.abs().values\n"
            "}).sort_values('Abs_Correlation', ascending=False).head(15).reset_index(drop=True)\n"
            "display(df_corr)"
        ),
        new_markdown_cell(
            "## 3. Machine Learning Modeling, Hyperparameter Tuning & Out-of-Sample Precision\n"
            "We evaluate 4 manual domain rules (`pred0`..`pred3`) alongside tuned `DecisionTreeClassifier`, "
            "`RandomForestClassifier`, `XGBClassifier`, penalized `Ridge` regression, and a Deep Neural Network (`MLPClassifier`), "
            "plus probability threshold rules and the Cointegration + ML Consensus Hybrid strategy (`pred12_coint_ml_hybrid`)."
        ),
        new_code_cell(
            "df_full, models_dict, df_tuning, df_feat_imp = tune_and_train_models(df_full, feature_sets['ALL_FEATURES'])\n"
            "print('Hyperparameter Tuning Results on Temporal Validation Split:')\n"
            "display(df_tuning)\n\n"
            "print('Top 15 Most Important Features (Random Forest & XGBoost Combined):')\n"
            "display(df_feat_imp.head(15))"
        ),
        new_code_cell(
            "df_test_eval = evaluate_predictions_on_test(df_full)\n"
            "print('Out-of-Sample TEST Split Precision & Return Summary Across All 13 Predictors:')\n"
            "display(df_test_eval)"
        ),
        new_markdown_cell(
            "## 4. Trading Strategy Simulations (`sim1_`, `sim2_`, and Exact `iterrows()` Portfolio `sim3_`)\n"
            "We compare:\n"
            "- **Strategy 1 (`sim1_`)**: Unfiltered 10-day vector simulation ($1,000 per signal, 0.20% round-trip cost).\n"
            "- **Strategy 2 (`sim2_`)**: Risk-managed vector simulation with tail Stop-Loss (`-9%`) and Take-Profit (`+16%`).\n"
            "- **Strategy 3 (`sim3_`)**: Exact chronological `iterrows()` portfolio simulation starting with **$50,000** capital, "
            "compounding gains across up to 6 concurrent positions sized by ML conviction and inverse volatility."
        ),
        new_code_cell(
            "df_full, df_sim_summary = run_vector_simulations(df_full)\n"
            "display(df_sim_summary)"
        ),
        new_code_cell(
            "df_equity, df_trades, portfolio_metrics = run_exact_iterrows_portfolio_simulation(\n"
            "    df_full,\n"
            "    signal_col='pred12_coint_ml_hybrid',\n"
            ")\n"
            "print('Exact Chronological iterrows() Portfolio Simulation Metrics vs SOXX & S&P 500 Benchmarks:')\n"
            "display(pd.DataFrame([portfolio_metrics]).T.rename(columns={0: 'Value'}))\n\n"
            "print('Sample of Executed Portfolio Trades on Test Set:')\n"
            "display(df_trades.tail(10))"
        ),
        new_markdown_cell(
            "## 5. Deep Exploratory Analysis & False-Positive Error Debugging\n"
            "We inspect strategy performance across semiconductor sub-verticals (`Memory_Storage`, `WFE_Equipment`, "
            "`Foundry_Logic`, `Analog_Power_RF`), individual tickers, and Cointegration Spread Z-Score regimes, "
            "and diagnose the worst False Positive trades on the Test set."
        ),
        new_code_cell(
            "deep_analysis = run_deep_exploratory_error_analysis(df_full, signal_col='pred12_coint_ml_hybrid')\n"
            "print('1. Performance Breakdown by Semiconductor Sub-Vertical:')\n"
            "display(deep_analysis['by_subvertical'])\n\n"
            "print('2. Performance Breakdown by Cointegration Z-Score Regime:')\n"
            "display(deep_analysis['by_zscore_regime'])\n\n"
            "print('3. Top 10 Tickers by Risk-Managed 10-Day Return:')\n"
            "display(deep_analysis['by_ticker'].head(10))\n\n"
            "print('4. Diagnostic Table of Worst False Positive Predictions on Test Split:')\n"
            "display(deep_analysis['worst_false_positives'].head(8))"
        ),
        new_markdown_cell(
            "## 6. Visualizations & Latest Live 10-Day Trade Signals\n"
            "We render all 5 analytical charts and display the latest live 10-day trade signals and simulated Alpaca "
            "bracket orders for the upcoming 2-week holding horizon."
        ),
        new_code_cell(
            "chart_paths = generate_all_charts(df_full, df_feat_imp, df_sim_summary, df_equity, deep_analysis)\n"
            "for cp in chart_paths:\n"
            "    display(Image(filename=str(cp)))"
        ),
        new_code_cell(
            "df_signals = generate_latest_live_signals(df_full)\n"
            "broker_orders = submit_orders_to_broker(df_signals, dry_run=True)\n"
            "dashboard_path = build_html_monitoring_dashboard(portfolio_metrics, df_signals, df_sim_summary, df_coint_adf)\n\n"
            "print('Latest Live 10-Trading-Day Rankings & Recommended Bracket Orders:')\n"
            "display(df_signals.head(12))\n\n"
            "print(f'Logged {len(broker_orders)} simulated Alpaca bracket orders:')\n"
            "print(json.dumps(broker_orders[:3], indent=2))"
        ),
    ]

    nb.cells = cells
    with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)
    print(f"[NotebookBuilder] Wrote notebook structure to {NOTEBOOK_PATH}")
    return nb


def execute_notebook():
    nb = build_notebook()
    print("[NotebookBuilder] Executing notebook cells via nbclient...")
    client = NotebookClient(
        nb,
        timeout=600,
        kernel_name="python3",
        resources={"metadata": {"path": str(NOTEBOOKS_DIR)}},
    )
    client.execute()
    with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)
    print(f"[NotebookBuilder] Successfully executed and saved {NOTEBOOK_PATH} with embedded outputs!")


if __name__ == "__main__":
    execute_notebook()

