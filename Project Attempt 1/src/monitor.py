"""
Model & Strategy Monitoring, Visualization, HTML Dashboard & Telegram Alert Module
(Rubric Section 3, Section 5, and Section 7 Bonus).
"""

import os
from pathlib import Path
from typing import Dict, List
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns

from .config import FIGURES_DIR, REPORTS_DIR, TARGET_BINARY_COL, TARGET_GROWTH_COL

sns.set_theme(style="whitegrid", palette="deep")
DASHBOARD_HTML_PATH = REPORTS_DIR / "monitoring_dashboard.html"


def generate_all_charts(
    df: pd.DataFrame,
    df_feat_imp: pd.DataFrame,
    df_sim_summary: pd.DataFrame,
    df_equity: pd.DataFrame,
    deep_analysis: Dict[str, pd.DataFrame],
) -> List[Path]:
    """Generate 5 analytical figures for EDA, Modeling, and Trading Simulation."""
    saved_paths: List[Path] = []

    # 1. Cointegration Spread Z-Score Dynamics & Target Distribution (EDA)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.2))
    lrcx = df[(df["Ticker"] == "LRCX") & (df["Date"] >= "2022-01-01")].sort_values("Date")
    axes[0].plot(lrcx["Date"], lrcx["pair_spread_zscore_20d"], color="#1f77b4", lw=1.3, label="LRCX vs AMAT Spread Z-Score (20d)")
    axes[0].axhline(0.0, color="black", lw=0.9, ls="--")
    axes[0].axhline(-1.5, color="#2ca02c", lw=1.1, ls=":", label="Oversold Reversion Entry (-1.5 SD)")
    axes[0].axhline(1.5, color="#d62728", lw=1.1, ls=":", label="Overextended Spread (+1.5 SD)")
    axes[0].set_title("Supply-Chain Pair Cointegration Z-Score (LRCX / AMAT)", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("20-Day Spread Z-Score")
    axes[0].legend(loc="upper right", fontsize=9)

    valid_ret = (df[TARGET_GROWTH_COL].dropna() - 1.0) * 100.0
    valid_ret_clipped = valid_ret.clip(-25.0, 25.0)
    axes[1].hist(valid_ret_clipped, bins=65, color="#2b5c8f", edgecolor="white", alpha=0.85)
    axes[1].axvline(0.0, color="#d62728", lw=1.4, ls="--", label="Zero Return Threshold")
    axes[1].axvline(valid_ret.mean(), color="#2ca02c", lw=1.4, label=f"Mean 10d Return ({valid_ret.mean():.2f}%)")
    axes[1].set_title("Distribution of 10-Trading-Day Forward Returns (23 Semi Tickers)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("10-Day Forward Return (%)")
    axes[1].set_ylabel("Frequency")
    axes[1].legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    p1 = FIGURES_DIR / "01_cointegration_spreads_eda.png"
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    saved_paths.append(p1)

    # 2. Top Feature Correlations with Target & ML Feature Importances
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.0))
    num_cols = [
        c for c in df_feat_imp["Feature"].tolist()
        if c in df.columns and not c.startswith(("Month_", "Weekday_", "sub_vertical_", "zscore_regime_", "vix_regime_", "adx_trend_"))
    ]
    corrs = df[num_cols].apply(lambda s: s.corr(df[TARGET_BINARY_COL])).dropna()
    top_corr = corrs.reindex(corrs.abs().sort_values(ascending=False).head(15).index).sort_values()
    colors_corr = ["#d62728" if v < 0 else "#2ca02c" for v in top_corr.values]
    axes[0].barh(top_corr.index, top_corr.values, color=colors_corr, alpha=0.85)
    axes[0].axvline(0.0, color="black", lw=0.8)
    axes[0].set_title("Top 15 Feature Correlations with 10d Positive Growth", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Pearson Correlation Coefficient")

    top_imp = df_feat_imp.head(15).iloc[::-1]
    y_pos = np.arange(len(top_imp))
    axes[1].barh(y_pos - 0.18, top_imp["RF_Importance"], height=0.35, label="Random Forest", color="#1f77b4")
    axes[1].barh(y_pos + 0.18, top_imp["XGB_Importance"], height=0.35, label="XGBoost", color="#ff7f0e")
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels(top_imp["Feature"])
    axes[1].set_title("Top 15 Predictive Features (RF & XGBoost Importance)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Relative Feature Importance")
    axes[1].legend(loc="lower right")
    plt.tight_layout()
    p2 = FIGURES_DIR / "02_feature_correlations_and_importance.png"
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    saved_paths.append(p2)

    # 3. Vector Simulation Comparison (Sim1 Raw vs Sim2 Risk-Managed CAGR & Net Revenue)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.8))
    sim_plot = df_sim_summary.sort_values("Sim2_CAGR", ascending=True).copy()
    y_idx = np.arange(len(sim_plot))
    axes[0].barh(y_idx - 0.18, (sim_plot["Sim1_CAGR"] - 1.0) * 100.0, height=0.35, label="Sim1 (Unfiltered 10d Hold)", color="#7f7f7f")
    axes[0].barh(y_idx + 0.18, (sim_plot["Sim2_CAGR"] - 1.0) * 100.0, height=0.35, label="Sim2 (-6% SL / +14% TP Managed)", color="#2ca02c")
    axes[0].set_yticks(y_idx)
    axes[0].set_yticklabels(sim_plot["Strategy_Predictor"])
    axes[0].set_title("Vector Simulation Annualized Return (CAGR %) by Predictor", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Annualized Return (%)")
    axes[0].legend(loc="lower right")

    axes[1].barh(y_idx, sim_plot["Sim2_Sharpe"], color="#1f77b4", alpha=0.88)
    axes[1].set_yticks(y_idx)
    axes[1].set_yticklabels(sim_plot["Strategy_Predictor"])
    axes[1].set_title("Vector Simulation Risk-Adjusted Sharpe Ratio (Sim2)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Annualized Sharpe Ratio")
    plt.tight_layout()
    p3 = FIGURES_DIR / "03_vector_simulation_comparison.png"
    fig.savefig(p3, dpi=150)
    plt.close(fig)
    saved_paths.append(p3)

    # 4. Exact Chronological iterrows() Portfolio Equity Curve & Drawdown vs Benchmarks
    fig, axes = plt.subplots(2, 1, figsize=(14, 7.5), gridspec_kw={"height_ratios": [2.2, 1.0]}, sharex=True)
    axes[0].plot(df_equity["Date"], df_equity["Total_Equity_USD"], label="Cointegration + ML Hybrid Portfolio (Exact Sim3)", color="#2ca02c", lw=2.2)
    axes[0].plot(df_equity["Date"], df_equity["SOXX_Benchmark_USD"], label="SOXX Semiconductor ETF (Buy & Hold)", color="#1f77b4", lw=1.6, ls="--")
    axes[0].plot(df_equity["Date"], df_equity["SP500_Benchmark_USD"], label="S&P 500 Index (Buy & Hold)", color="#7f7f7f", lw=1.5, ls=":")
    axes[0].set_title("Out-of-Sample Exact Portfolio Simulation ($50,000 Initial Capital, Compounding)", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("Portfolio Value ($ USD)")
    axes[0].legend(loc="upper left", fontsize=10)

    for col, lbl, clr in [
        ("Total_Equity_USD", "Strategy Drawdown", "#2ca02c"),
        ("SOXX_Benchmark_USD", "SOXX ETF Drawdown", "#1f77b4"),
        ("SP500_Benchmark_USD", "S&P 500 Drawdown", "#7f7f7f"),
    ]:
        dd = ((df_equity[col] / df_equity[col].cummax()) - 1.0) * 100.0
        axes[1].plot(df_equity["Date"], dd, label=lbl, color=clr, lw=1.3)
    axes[1].fill_between(
        df_equity["Date"],
        ((df_equity["Total_Equity_USD"] / df_equity["Total_Equity_USD"].cummax()) - 1.0) * 100.0,
        0,
        color="#2ca02c",
        alpha=0.15,
    )
    axes[1].set_title("Underwater Drawdown Profile (%)", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("Drawdown (%)")
    axes[1].set_xlabel("Test Period Date")
    axes[1].legend(loc="lower left", fontsize=9)
    plt.tight_layout()
    p4 = FIGURES_DIR / "04_exact_portfolio_equity_curve.png"
    fig.savefig(p4, dpi=150)
    plt.close(fig)
    saved_paths.append(p4)

    # 5. Deep Exploratory Analysis: Sub-Vertical & Spread Z-Score Regime Performance
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    sub_df = deep_analysis["by_subvertical"].sort_values("Managed_10d_Ret_Pct", ascending=True)
    axes[0].barh(sub_df["sub_vertical"], sub_df["Managed_10d_Ret_Pct"], color="#2b5c8f", alpha=0.88)
    axes[0].set_title("Mean 10-Day Trade Return by Semiconductor Sub-Vertical", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Mean Risk-Managed 10-Day Return (%)")

    z_df = deep_analysis["by_zscore_regime"].copy()
    order = ["Deep_Underval", "Mild_Underval", "Equilibrium", "Mild_Overval", "Extended_Overval"]
    z_df["zscore_regime_bin"] = pd.Categorical(z_df["zscore_regime_bin"], categories=order, ordered=True)
    z_df.sort_values("zscore_regime_bin", inplace=True)
    axes[1].bar(z_df["zscore_regime_bin"].astype(str), z_df["Base_Mean_10d_Ret_Pct"], color="#ff7f0e", alpha=0.85)
    axes[1].axhline(0.0, color="black", lw=0.9)
    axes[1].set_title("Mean 10-Day Forward Return Across Pair Spread Z-Score Regimes", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Mean 10-Day Forward Return (%)")
    axes[1].tick_params(axis="x", rotation=20)
    plt.tight_layout()
    p5 = FIGURES_DIR / "05_subvertical_and_zscore_analysis.png"
    fig.savefig(p5, dpi=150)
    plt.close(fig)
    saved_paths.append(p5)

    print(f"[Monitor] Generated {len(saved_paths)} analytical charts in {FIGURES_DIR}.")
    return saved_paths


def build_html_monitoring_dashboard(
    metrics: Dict[str, float],
    df_signals: pd.DataFrame,
    df_sim_summary: pd.DataFrame,
    df_coint_adf: pd.DataFrame,
) -> Path:
    """Render a standalone interactive HTML monitoring dashboard for live trading oversight."""
    active_signals = df_signals[df_signals["Order_Shares"] > 0].copy()
    signals_table_html = df_signals.head(12).to_html(index=False, classes="styled-table", float_format=lambda x: f"{x:.4f}" if abs(x) < 10 else f"{x:.2f}")
    sim_table_html = df_sim_summary.head(10).to_html(index=False, classes="styled-table")
    coint_table_html = df_coint_adf.to_html(index=False, classes="styled-table")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>US & Taiwan Semiconductor Cointegration + ML Trading Monitor</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 24px; background: #f8fafc; color: #0f172a; }}
    h1, h2 {{ color: #1e293b; }}
    .kpi-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 16px; margin-bottom: 28px; }}
    .kpi-card {{ background: white; border-radius: 10px; padding: 16px 20px; box-shadow: 0 1px 4px rgba(0,0,0,0.08); border-left: 4px solid #2563eb; }}
    .kpi-label {{ font-size: 12px; text-transform: uppercase; color: #64748b; font-weight: 600; }}
    .kpi-value {{ font-size: 24px; font-weight: 700; color: #0f172a; margin-top: 6px; }}
    .kpi-sub {{ font-size: 12px; color: #16a34a; margin-top: 4px; }}
    .section-card {{ background: white; border-radius: 10px; padding: 20px; margin-bottom: 24px; box-shadow: 0 1px 4px rgba(0,0,0,0.06); overflow-x: auto; }}
    .styled-table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
    .styled-table th {{ background-color: #1e293b; color: #ffffff; text-align: left; padding: 10px; }}
    .styled-table td {{ padding: 8px 10px; border-bottom: 1px solid #e2e8f0; }}
    .styled-table tr:nth-of-type(even) {{ background-color: #f8fafc; }}
  </style>
</head>
<body>
  <h1>US & Taiwan Semiconductor Supply-Chain Cointegration + ML Strategy Monitor</h1>
  <p><strong>Test Evaluation Window:</strong> {metrics['Test_Period_Start']} to {metrics['Test_Period_End']} ({metrics['Test_Duration_Years']} yrs) | <strong>Active Live Signals:</strong> {len(active_signals)} positions</p>

  <div class="kpi-grid">
    <div class="kpi-card">
      <div class="kpi-label">Strategy Final Equity ($50k Init)</div>
      <div class="kpi-value">${metrics['Strategy_Final_Equity_USD']:,.2f}</div>
      <div class="kpi-sub">Total Return: +{metrics['Strategy_Total_Return_Pct']:.2f}%</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">Strategy Annualized CAGR</div>
      <div class="kpi-value">{metrics['Strategy_Annual_CAGR_Pct']:.2f}%</div>
      <div class="kpi-sub">vs SOXX: {metrics['SOXX_Annual_CAGR_Pct']:.2f}% | S&P 500: {metrics['SP500_Annual_CAGR_Pct']:.2f}%</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">Sharpe / Sortino Ratio</div>
      <div class="kpi-value">{metrics['Strategy_Sharpe_Ratio']:.2f} / {metrics['Strategy_Sortino_Ratio']:.2f}</div>
      <div class="kpi-sub">vs SOXX Sharpe: {metrics['SOXX_Sharpe_Ratio']:.2f} | S&P 500: {metrics['SP500_Sharpe_Ratio']:.2f}</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">Max Drawdown</div>
      <div class="kpi-value">{metrics['Strategy_Max_Drawdown_Pct']:.2f}%</div>
      <div class="kpi-sub">vs SOXX Max DD: {metrics['SOXX_Max_Drawdown_Pct']:.2f}%</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">Win Rate & Executed Trades</div>
      <div class="kpi-value">{metrics['Trade_Win_Rate_Pct']:.1f}%</div>
      <div class="kpi-sub">{metrics['Total_Executed_Trades']} trades (Avg {metrics['Avg_Net_Return_Per_Trade_Pct']:+.2f}%/trade)</div>
    </div>
  </div>

  <div class="section-card">
    <h2>Latest 10-Trading-Day Live Signals & Bracket Order Sizing</h2>
    {signals_table_html}
  </div>

  <div class="section-card">
    <h2>Vector Simulation Strategy Comparison (Out-of-Sample Test Set)</h2>
    {sim_table_html}
  </div>

  <div class="section-card">
    <h2>Supply-Chain Pair Cointegration Augmented Dickey-Fuller (ADF) Stationarity Summary</h2>
    {coint_table_html}
  </div>
</body>
</html>
"""
    DASHBOARD_HTML_PATH.write_text(html, encoding="utf-8")
    print(f"[Monitor] Built interactive HTML monitoring dashboard at {DASHBOARD_HTML_PATH.name}.")
    return DASHBOARD_HTML_PATH


def send_telegram_trade_alert(df_signals: pd.DataFrame, metrics: Dict[str, float]) -> bool:
    """
    Send an automated daily trade alert summary via Telegram Bot API if `TELEGRAM_BOT_TOKEN`
    and `TELEGRAM_CHAT_ID` environment variables are configured.
    """
    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")
    active = df_signals[df_signals["Order_Shares"] > 0]

    lines = [
        "📊 *Semiconductor Cointegration + ML Daily Signal Update*",
        f"• Backtest CAGR: `{metrics['Strategy_Annual_CAGR_Pct']:.2f}%` (Sharpe `{metrics['Strategy_Sharpe_Ratio']:.2f}`)",
        f"• Active 10-Day Signals ({len(active)}):",
    ]
    for _, r in active.iterrows():
        lines.append(
            f"  - *{r['Ticker']}* ({r['Recommended_Action']}): {int(r['Order_Shares'])} sh @ ${r['Close']:.2f} "
            f"| Prob={r['prob_ensemble']:.3f}, Z={r['pair_spread_zscore_20d']:.2f} "
            f"| SL=${r['Stop_Loss_Price']:.2f}, TP=${r['Take_Profit_Price']:.2f}"
        )
    msg_text = "\n".join(lines)

    if not bot_token or not chat_id:
        print("[Monitor] Telegram credentials not set (dry-run); alert message prepared cleanly.")
        return False

    try:
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        resp = requests.post(url, json={"chat_id": chat_id, "text": msg_text, "parse_mode": "Markdown"}, timeout=10)
        return resp.ok
    except Exception as exc:
        print(f"[Monitor] Telegram notification warning: {exc}")
        return False

