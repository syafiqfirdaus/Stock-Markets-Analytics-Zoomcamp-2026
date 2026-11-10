"""
End-to-End Automated Workflow Runner for Project Attempt 1:
US & Taiwan Semiconductor Supply-Chain Cointegration & Machine Learning Strategy.

Usage Examples:
  1. Full fresh download & end-to-end execution:
     python "Project Attempt 1/run_pipeline.py" --mode download

  2. Fast local cached execution (no API re-download if local parquet exists):
     python "Project Attempt 1/run_pipeline.py" --mode local

  3. Daily cron incremental update & live trade generation:
     python "Project Attempt 1/run_pipeline.py" --mode download --incremental
"""

import argparse
import json
from typing import Any, Dict
import pandas as pd

from src.broker_api import generate_latest_live_signals, submit_orders_to_broker
from src.config import COMBINED_DATA_PATH, PREDICTIONS_DATA_PATH, REPORTS_DIR
from src.data_loader import load_or_download_raw_data
from src.modeling import evaluate_predictions_on_test, tune_and_train_models
from src.monitor import (
    build_html_monitoring_dashboard,
    generate_all_charts,
    send_telegram_trade_alert,
)
from src.simulation import (
    run_deep_exploratory_error_analysis,
    run_exact_iterrows_portfolio_simulation,
    run_vector_simulations,
)
from src.transformations import (
    build_unified_dataset,
    get_feature_sets,
    run_pair_cointegration_adf_summary,
)


def execute_full_workflow(
    mode: str = "local",
    incremental: bool = False,
    execute_live_broker: bool = False,
) -> Dict[str, Any]:
    """Run the complete 6-stage quantitative research & trading automation workflow."""
    print("=" * 88)
    print("US & TAIWAN SEMICONDUCTOR SUPPLY-CHAIN COINTEGRATION & ML TRADING WORKFLOW")
    print(f"Execution Regime: mode='{mode}', incremental={incremental}")
    print("=" * 88)

    # Stage 1 & 2: Data Acquisition & Unified Feature Engineering
    if mode == "local" and not incremental and COMBINED_DATA_PATH.exists():
        print(f"[Workflow] Loading pre-built unified dataset from {COMBINED_DATA_PATH.name}...")
        df_base = pd.read_parquet(COMBINED_DATA_PATH)
    else:
        df_ohlcv, df_ctx, df_sec = load_or_download_raw_data(mode=mode, incremental=incremental)
        df_base = build_unified_dataset(df_ohlcv, df_ctx, df_sec, save_parquet=True)

    df_full, feature_sets = get_feature_sets(df_base)
    all_features = feature_sets["ALL_FEATURES"]

    print(
        f"[Workflow] Unified Dataset Ready: {len(df_full):,} rows | "
        f"{len(feature_sets['NUMERIC'])} numeric features | "
        f"{len(feature_sets['DUMMIES'])} dummy features | "
        f"{len(all_features)} total ML features."
    )

    # Stage 3: Supply-Chain Pair Cointegration ADF Stationarity Test Summary (EDA)
    df_coint_adf = run_pair_cointegration_adf_summary(df_full)
    df_coint_adf.to_csv(REPORTS_DIR / "cointegration_adf_summary.csv", index=False)

    # Stage 4: Model Training, Hyperparameter Tuning & Out-of-Sample Evaluation
    df_full, models_dict, df_tuning, df_feat_imp = tune_and_train_models(df_full, all_features)
    df_test_eval = evaluate_predictions_on_test(df_full)

    df_tuning.to_csv(REPORTS_DIR / "hyperparameter_tuning_results.csv", index=False)
    df_feat_imp.to_csv(REPORTS_DIR / "feature_importances.csv", index=False)
    df_test_eval.to_csv(REPORTS_DIR / "model_test_precision_summary.csv", index=False)

    # Stage 5: Vector Simulations (`sim1_`, `sim2_`) & Exact `iterrows()` Portfolio Simulation (`sim3_`)
    df_full, df_sim_summary = run_vector_simulations(df_full)
    df_sim_summary.to_csv(REPORTS_DIR / "vector_simulations_summary.csv", index=False)

    df_equity, df_trades, portfolio_metrics = run_exact_iterrows_portfolio_simulation(
        df_full,
        signal_col="pred12_coint_ml_hybrid",
    )
    df_equity.to_csv(REPORTS_DIR / "exact_portfolio_equity_curve.csv", index=False)
    df_trades.to_csv(REPORTS_DIR / "exact_portfolio_executed_trades.csv", index=False)

    with open(REPORTS_DIR / "exact_portfolio_metrics.json", "w", encoding="utf-8") as f:
        json.dump(portfolio_metrics, f, indent=2)

    # Save Test + Live predictions parquet for fast inspection
    df_full[df_full["split"].isin(["test", "live"])].to_parquet(PREDICTIONS_DATA_PATH, index=False)

    # Deep Exploratory Error & Subgroup Analysis
    deep_analysis = run_deep_exploratory_error_analysis(df_full, signal_col="pred12_coint_ml_hybrid")
    for key, sub_df in deep_analysis.items():
        sub_df.to_csv(REPORTS_DIR / f"deep_analysis_{key}.csv", index=False)

    # Stage 6: Live 10-Day Trade Signals, Broker Bracket Order Submission, Charts & HTML Dashboard
    df_signals = generate_latest_live_signals(df_full)
    broker_orders = submit_orders_to_broker(df_signals, dry_run=not execute_live_broker)

    chart_paths = generate_all_charts(df_full, df_feat_imp, df_sim_summary, df_equity, deep_analysis)
    dashboard_path = build_html_monitoring_dashboard(portfolio_metrics, df_signals, df_sim_summary, df_coint_adf)
    send_telegram_trade_alert(df_signals, portfolio_metrics)

    # Print Executive Summary to Console
    print("\n" + "=" * 88)
    print("EXECUTIVE SUMMARY - OUT-OF-SAMPLE TEST RESULTS")
    print("=" * 88)
    print("\n1. Supply-Chain Pair Cointegration (ADF Test Top Pairs):")
    print(df_coint_adf.head(8).to_string(index=False))

    print("\n2. Out-of-Sample Test Precision Across Manual & ML Predictors:")
    print(df_test_eval.to_string(index=False))

    print("\n3. Vector Simulation Summary (Sim1 Raw vs Sim2 Risk-Managed):")
    cols_show = [
        "Strategy_Predictor", "Test_Trades", "Avg_Trades_Per_Active_Day",
        "Required_Capital_USD", "Sim1_CAGR", "Sim2_CAGR", "Sim2_Sharpe", "Sim2_Win_Rate_Pct",
    ]
    print(df_sim_summary[cols_show].to_string(index=False))

    print("\n4. Exact Chronological iterrows() Portfolio Simulation vs Benchmarks ($50,000 Initial Capital):")
    for k, v in portfolio_metrics.items():
        print(f"   {k:34s}: {v}")

    print("\n5. Latest Live 10-Day Trade Recommendations:")
    live_cols = [
        "Date", "Ticker", "sub_vertical", "peer_ticker", "Close",
        "prob_ensemble", "pair_spread_zscore_20d", "Recommended_Action",
        "Order_Shares", "Allocated_Capital_USD", "Stop_Loss_Price", "Take_Profit_Price",
    ]
    print(df_signals[live_cols].head(10).to_string(index=False))
    print("=" * 88)

    return {
        "df_full": df_full,
        "feature_sets": feature_sets,
        "df_coint_adf": df_coint_adf,
        "df_tuning": df_tuning,
        "df_feat_imp": df_feat_imp,
        "df_test_eval": df_test_eval,
        "df_sim_summary": df_sim_summary,
        "df_equity": df_equity,
        "df_trades": df_trades,
        "portfolio_metrics": portfolio_metrics,
        "deep_analysis": deep_analysis,
        "df_signals": df_signals,
        "broker_orders": broker_orders,
        "chart_paths": chart_paths,
        "dashboard_path": dashboard_path,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run US & Taiwan Semiconductor Cointegration + ML Trading Strategy Workflow."
    )
    parser.add_argument(
        "--mode",
        choices=["local", "download"],
        default="local",
        help="Execution regime: 'local' uses cached dataset if available; 'download' fetches fresh API data.",
    )
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="Fetch only recent delta trading days and merge into existing local dataset.",
    )
    parser.add_argument(
        "--live-broker",
        action="store_true",
        help="Submit live paper bracket orders to Alpaca API if ALPACA_API_KEY is configured.",
    )
    args = parser.parse_args()
    execute_full_workflow(
        mode=args.mode,
        incremental=args.incremental,
        execute_live_broker=args.live_broker,
    )


if __name__ == "__main__":
    main()

