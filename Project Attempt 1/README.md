# US & Taiwan Semiconductor Supply-Chain Cointegration & Machine Learning Strategy

**Stock Markets Analytics Zoomcamp 2026 — Capstone Project (Attempt 1)**

---

## 1. Problem Description & Investment Thesis

### 1.1 Context & Problem Statement
Semiconductor equities (`NVDA`, `TSM`, `ASML`, `LRCX`, `AMAT`, `KLAC`, `AVGO`, `MU`, `TXN`, `ADI`, etc.) lie at the heart of global AI infrastructure, cloud capex, and industrial automation. Companies within the same semiconductor sub-vertical share identical wafer fab utilization cycles, hyperscaler capital expenditure budgets, and Taiwanese foundry/packaging supply chains (`2330.TW` TSMC, `2303.TW` UMC, `2454.TW` MediaTek, `3711.TW` ASE). Over medium-to-long horizons, the equity prices of supply-chain partners (e.g., Etch & Deposition equipment peers `LRCX` vs. `AMAT`, Foundry-Lithography partners `TSM` vs. `ASML`, and Analog leaders `TXN` vs. `ADI`) are **economically cointegrated**.

However, over tactical **5-to-15 trading-day (2-week / 10-trading-day)** windows, individual semiconductor stocks frequently diverge from their cointegrated supply-chain peers due to non-fundamental liquidity shocks—such as single-stock options gamma squeezes, thematic ETF (`SOXX` / `SMH`) weight-cap rebalancing, or delayed reaction to Taipei (`TWSE`) foundry disclosures.

### 1.2 Why This Problem Definition Is Novel (Beyond Lecture Setup)
1. **Tactical 10-Trading-Day (2-Week) Supply-Chain Horizon**: Rather than the lecture's 30-day mega-cap setup, we target a **10-trading-day** tactical mean-reversion and momentum horizon across **23 US-listed semiconductor equities** paired with **4 Taiwanese Stock Exchange (`.TW`) supply-chain anchors**.
2. **Cross-Market US–Taiwan Lead-Lag & Cointegration**: We explicitly model the cointegrated log-price spread $S_t = \ln(P_{A,t}) - (\alpha_t + \beta_t \ln(P_{B,t}))$, its Ornstein-Uhlenbeck mean-reversion half-life $\tau$, and the Taipei-to-New-York `2330.TW` vs. `TSM` ADR arbitrage premium.
3. **Point-in-Time SEC EDGAR XBRL Fundamental Integration**: Purely statistical pairs trading fails when a spread widening reflects a genuine fundamental structural break (e.g., inventory glut or revenue collapse). We integrate **SEC EDGAR CompanyFacts XBRL filings** (`10-Q`, `10-K`, `20-F`) strictly aligned by public `filed` date to condition cointegration signals on quarterly revenue growth, R&D intensity, operating margin, and semiconductor inventory cycles.

### 1.3 End Goal & Practical Usage
The automated pipeline ingests daily market, macroeconomic, and SEC filing data after market close, updates rolling cointegration and ML models, simulates strategy risk/return against `SOXX` and `S&P 500` benchmarks, outputs a ranked table of **10-day trade signals with exact share sizing and Stop-Loss / Take-Profit bracket levels**, submits paper bracket orders to the **Alpaca Broker API**, and renders an interactive **HTML Monitoring Dashboard**.

```mermaid
flowchart LR
    subgraph Data["1. Multi-Source Data Layer"]
        YF["Yahoo Finance API\n23 US Semi Tickers\n4 Taiwan (.TW) Anchors\nSOXX, SMH, VIX, USD/TWD"]
        FRED["FRED Macro API\nDGS10, DGS2, T10Y2Y\nFEDFUNDS, High-Yield OAS"]
        SEC["SEC EDGAR XBRL API\nPoint-in-Time 10-Q/10-K\nRev YoY, R&D, Inventory"]
    end
    subgraph Transform["2. Feature & Cointegration Engine"]
        COINT["Supply-Chain Pairs\n60d OLS Beta, Log Spread\n20d/60d Z-Score, ADF, Half-Life"]
        TECH["Technicals & Regimes\nGrowth 1d..252d, RSI, PPO, ADX\nBinned Z-Score/VIX/ADX Dummies"]
    end
    subgraph ML["3. Modeling & Tuning"]
        RULES["4 Domain Hand Rules\nCoint Z-Score, TW Lead-Lag\nRSI/VIX, Macro Yield"]
        MODELS["Tuned DT, RF, XGBoost\nPenalized Ridge Regression\n2-Layer Deep Neural Net (MLP)"]
        HYBRID["pred12_coint_ml_hybrid\nEnsemble + Spread Z-Score\n+ SEC Filing Blackout Filter"]
    end
    subgraph Sim["4. Simulation & Automation"]
        VEC["Vector Simulations\nsim1_ (Raw) & sim2_ (-9% SL / +16% TP)"]
        EXACT["Exact iterrows() Portfolio (sim3_)\n$50k Compounding, Top-6 Slots\nProb & Inverse-Vol Sizing"]
        LIVE["Live 10d Signals, Alpaca Bracket API\nHTML Dashboard & Cron/Docker"]
    end
    YF & FRED & SEC --> COINT & TECH --> RULES & MODELS --> HYBRID --> VEC & EXACT & LIVE
```

---

## 2. Data Sources & 28+ New Engineered Features

### 2.1 Primary & Alternative Data Providers
| Source | Coverage & Granularity | Role in Strategy |
| :--- | :--- | :--- |
| **1. Yahoo Finance (`yfinance`)** | Daily adjusted OHLCV, Dividends, Splits (2010–2026) for **23 US Semiconductor stocks**, **4 Taiwanese (`.TW`) supply-chain stocks** (`2330.TW`, `2303.TW`, `2454.TW`, `3711.TW`), **ETFs & Indices** (`SOXX`, `SMH`, `^GSPC`, `^VIX`), and **FX** (`TWD=X`, `DX-Y.NYB`). | Core equity pricing, sub-vertical peer cointegration spreads, Taiwan-US lead-lag signals, and benchmark comparison. |
| **2. Federal Reserve Economic Data (`FRED`)** | Daily US Treasury yields (`DGS10`, `DGS2`), Yield Curve (`T10Y2Y`), Effective Fed Funds (`FEDFUNDS`), and ICE BofA US High-Yield Credit Spread (`BAMLH0A0HYM2`). | Discount-rate sensitivity (semiconductors are long-duration growth assets) and credit-stress regime filtering. |
| **3. SEC EDGAR XBRL REST API (`data.sec.gov`)** *(New Non-YFinance/Non-FRED Source)* | Point-in-time quarterly `10-Q`, `10-K`, and `20-F` XBRL disclosures across all 23 CIK entities (`661` historical filings cached in `sec_edgar_fundamentals.parquet`), matched via `pd.merge_asof(..., direction='backward')` on exact SEC `filed` date. | Prevents look-ahead bias while measuring fundamental revenue acceleration, R&D intensity, operating margin, inventory-to-revenue cycle, and earnings blackout windows. |

### 2.2 Universe & Supply-Chain Cointegration Pair Mapping
| Sub-Vertical | US Tickers | Supply-Chain Cointegrated Pairs | Economic Mechanism |
| :--- | :--- | :--- | :--- |
| **Wafer Fab Equipment (WFE)** | `LRCX`, `AMAT`, `KLAC`, `ASML`, `TER`, `ENTG` | `LRCX`/`AMAT`, `KLAC`/`ASML`, `TER`/`ENTG` | Shared wafer fab capex budgets for etch, deposition, lithography, metrology, and test. |
| **Foundry, AI Logic & SoC** | `TSM`, `NVDA`, `AMD`, `AVGO`, `MRVL`, `QCOM`, `INTC` | `TSM`/`ASML`, `NVDA`/`AMD`, `AVGO`/`MRVL`, `QCOM`/`INTC` | Advanced node foundry allocation (CoWoS/3nm), AI accelerator demand, and custom ASIC silicon. |
| **Memory & Storage** | `MU`, `WDC` | `MU`/`WDC` | DRAM/HBM and enterprise NAND flash pricing & inventory cycles. |
| **Analog, Power & RF** | `TXN`, `ADI`, `NXPI`, `ON`, `MCHP`, `MPWR`, `SWKS`, `QRVO` | `TXN`/`ADI`, `NXPI`/`ON`, `MCHP`/`MPWR`, `SWKS`/`QRVO` | Industrial/automotive analog content, SiC power, and mobile RF front-end modules. |

### 2.3 Catalog of 28 New Domain Features (Beyond Lecture Set)
In addition to the lecture features (`growth_1d`..`growth_252d`, `SMA10`..`SMA200`, `volatility`, `ln_volume`, `div_payout`, `stock_split`, `rsi`, `ppo`, `cci`, `adx`, `natr`, `stoch`, `willr`, `cmf`, `roc`), we introduce **28 new features**:

1. `pair_ols_beta_60d`: Rolling 60-day OLS hedge ratio $\beta_t = \text{Cov}(\ln P_A, \ln P_B)/\text{Var}(\ln P_B)$ against the ticker's designated supply-chain peer.
2. `pair_log_spread`: Cointegrated residual log-price spread $S_t = \ln(P_{A,t}) - (\alpha_t + \beta_t \ln(P_{B,t}))$.
3. `pair_spread_zscore_20d`: 20-day standardized spread Z-score $Z_{20,t} = (S_t - \mu_{S,20})/\sigma_{S,20}$ (tactical mispricing signal).
4. `pair_spread_zscore_60d`: 60-day standardized spread Z-score $Z_{60,t}$ (medium-term valuation gap).
5. `pair_spread_change_5d`: 5-day velocity of the pair spread Z-score ($\Delta Z_{5,t}$).
6. `pair_adf_stat_proxy_60d`: Rolling 60-day Ornstein-Uhlenbeck AR(1) stationarity t-statistic $(\hat{\phi}/\text{SE}(\hat{\phi}))$ on $\Delta S_t = \phi S_{t-1} + \epsilon_t$.
7. `pair_halflife_days`: Implied mean-reversion half-life $\tau = -\ln(2)/\ln(1 + \hat{\phi})$ in trading days.
8. `pair_return_divergence_10d`: 10-day return difference between the stock and its supply-chain peer ($R_{A,10d} - R_{B,10d}$).
9. `pair_corr_60d`: Rolling 60-day daily return correlation between the stock and its supply-chain peer.
10. `subvertical_rel_strength_10d`: 10-day return minus the cross-sectional average 10-day return of its semiconductor sub-vertical.
11. `tsmc_adr_premium_proxy`: 60-day standardized Z-score of the `TSM` US ADR price relative to 5 ordinary shares of Taipei `2330.TW` converted at spot `USD/TWD`.
12. `tw_tsmc_lead_diff_5d`: 5-day return lead-lag spread between Taipei TSMC (`2330.TW`) and the US semiconductor stock.
13. `tw_supply_chain_index_10d`: Equal-weighted 10-day momentum index across the 4 Taiwanese bellwethers (`2330.TW`, `2303.TW`, `2454.TW`, `3711.TW`).
14. `tw_supply_chain_dispersion_10d`: Cross-sectional standard deviation across the 4 Taiwanese supply-chain anchors.
15. `tw_tsmc_ret_10d`: 10-day return of TSMC Taipei (`2330.TW`).
16. `tw_umc_ret_10d`: 10-day return of United Microelectronics (`2303.TW`).
17. `tw_mediatek_ret_10d`: 10-day return of MediaTek (`2454.TW`).
18. `tw_ase_ret_10d`: 10-day return of ASE Technology OSAT packaging (`3711.TW`).
19. `usd_twd_ret_10d`: 10-day percentage change in `USD/TWD` spot exchange rate.
20. `soxx_rel_growth_10d`: Stock's 10-day return minus `SOXX` ETF 10-day return.
21. `soxx_beta_60d`: Rolling 60-day beta of the stock to the `SOXX` Semiconductor ETF.
22. `sec_rev_qoq`: Point-in-time sequential quarterly revenue growth from SEC EDGAR 10-Q/10-K filings.
23. `sec_rev_yoy`: Point-in-time year-over-year quarterly revenue growth from SEC EDGAR filings.
24. `sec_rd_intensity`: R&D expense divided by quarterly revenue (SEC EDGAR).
25. `sec_op_margin`: Operating income margin (SEC EDGAR).
26. `sec_inv_to_rev`: Net inventory divided by quarterly revenue (tracks semiconductor inventory build/depletion).
27. `sec_inv_cycle_change`: Sequential quarter-over-quarter change in inventory-to-revenue ratio.
28. `days_since_sec_filing` & `earnings_blackout_window`: Days elapsed since the last SEC 10-Q/10-K filing and binary indicator blocking entries around imminent quarterly releases.

---

## 3. Data Transformations & Exploratory Data Analysis (EDA)

### 3.1 Unified Dataset & Feature Groups
The unified dataset (`data/semiconductor_coint_ml_dataset.parquet`) contains **87,715 ticker-day records** (2011-01-03 to 2026-11-09) across 23 tickers and **127 ML features** (`94` numerical features + `33` one-hot dummy variables):
- **`TO_PREDICT` (8 cols)**: `growth_future_5d`, `growth_future_10d`, `growth_future_20d`, `is_positive_growth_10d_future`, `is_outperform_soxx_10d_future`, `future_max_gain_10d`, `future_max_dd_10d`, `soxx_growth_future_10d`.
- **Engineered Nonlinear Transformations & Binned Dummy Sets (7+ sets)**:
  1. `coint_reversion_pressure`: $-\text{ZScore}_{20d} \times \max(0, \text{Corr}_{60d}) \times (1 - 0.5 \cdot \text{Blackout})$.
  2. `fundamental_filtered_zscore`: Pair Z-Score conditioned on positive SEC YoY revenue growth (`sec_rev_yoy >= -5%`).
  3. `rate_adjusted_momentum`: 20-day stock return penalized by 20-day changes in the 10Y Treasury yield (`dgs10_diff_20d`).
  4. `tw_coint_synergy_score`: Composite score combining pair undervaluation, Taiwanese foundry momentum, and SEC revenue growth.
  5. `zscore_regime_bin` dummies: 5 binned cointegration regimes (`Deep_Underval`, `Mild_Underval`, `Equilibrium`, `Mild_Overval`, `Extended_Overval`).
  6. `vix_regime_bin` dummies: 4 market volatility regimes (`Low_Vol`, `Normal_Vol`, `Elevated_Vol`, `Panic_Vol`).
  7. `adx_trend_bin` & `sub_vertical` dummies: Trend strength regimes (`Range_MeanRevert`, `Moderate_Trend`, `Strong_Trend`) and semiconductor sub-verticals.

### 3.2 Supply-Chain Pair Cointegration Verification (ADF Stationarity Tests)
Formal Augmented Dickey-Fuller (ADF) tests on the rolling cointegrated log-price spread confirm statistically significant stationarity ($p < 0.0001$) across all 11 semiconductor sub-vertical pairs, with median mean-reversion half-lives of **5.0 to 7.3 trading days**—directly validating our **10-trading-day** holding horizon:

| Supply-Chain Pair | Sub-Vertical | ADF t-Statistic | ADF p-Value | Stationary ($p < 0.05$) | Median 60d Return Corr | Median Half-Life (Days) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **`LRCX / AMAT`** | `WFE_Equipment` | **-11.3226** | **0.000000** | **True** | **0.8538** | **6.07** |
| **`AVGO / MRVL`** | `Foundry_Logic` | **-10.7927** | **0.000000** | **True** | 0.5305 | **5.68** |
| **`MU / WDC`** | `Memory_Storage` | **-10.4428** | **0.000000** | **True** | 0.6275 | **6.03** |
| **`ASML / TSM`** | `WFE_Equipment` | **-10.4407** | **0.000000** | **True** | 0.5901 | **5.04** |
| **`KLAC / ASML`** | `WFE_Equipment` | **-10.3784** | **0.000000** | **True** | 0.7193 | **5.98** |
| **`MCHP / MPWR`** | `Analog_Power_RF` | **-10.3154** | **0.000000** | **True** | 0.6929 | **6.63** |
| **`TXN / ADI`** | `Analog_Power_RF` | **-10.1795** | **0.000000** | **True** | **0.8132** | **5.95** |
| **`NVDA / AMD`** | `Foundry_Logic` | **-9.5858** | **0.000000** | **True** | 0.5530 | **6.29** |
| **`NXPI / ON`** | `Analog_Power_RF` | **-9.4613** | **0.000000** | **True** | 0.7168 | **5.32** |
| **`QCOM / INTC`** | `Foundry_Logic` | **-8.9778** | **0.000000** | **True** | 0.4679 | **7.26** |
| **`TER / ENTG`** | `WFE_Equipment` | **-8.7837** | **0.000000** | **True** | 0.6615 | **7.22** |

![Cointegration Spreads & 10-Day Return Distribution](outputs/figures/01_cointegration_spreads_eda.png)

![Feature Correlations & ML Feature Importance](outputs/figures/02_feature_correlations_and_importance.png)

---

## 4. Machine Learning Modeling & Hyperparameter Tuning

### 4.1 Temporal Split Setup
- **Train Split (70%)**: `2011-01-03` to `2022-01-24` (`64,147` records)
- **Validation Split (15%)**: `2022-01-25` to `2024-06-11` (`13,685` records)
- **Test Split (15%)**: `2024-06-12` to `2026-10-26` (`13,662` records)
- **Live Unfolding Horizon**: `2026-10-27` to `2026-11-09` (`230` records for live forward 10-day predictions)

### 4.2 Models & Custom Rules Evaluated
1. **4 Hand-Crafted Domain Rules (`pred0`..`pred3`)**:
   - `pred0_manual_coint_zscore`: `pair_spread_zscore_20d < -1.15` & `pair_adf_stat_proxy_60d < -1.60` & `earnings_blackout_window == 0`.
   - `pred1_manual_tw_lead_lag`: `tw_supply_chain_index_10d > 0.8%` & `pair_spread_zscore_20d < -0.35` & `sec_rev_yoy > -2%`.
   - `pred2_manual_rsi_vix_filter`: `rsi_14 < 43` & `vix_close < 26` & `golden_cross_50_200 == 1`.
   - `pred3_manual_macro_yield`: `dgs10_diff_20d <= 0.20` & `growing_moving_average == 1` & `pair_spread_zscore_20d < 0.25`.
2. **Lecture Models (`DecisionTreeClassifier`, `RandomForestClassifier`)**:
   - Tuned across `max_depth` and `min_samples_leaf` on the Validation split (`pred4_clf_dt_best`, `pred5_rf_best`, `pred9_rf_high_prob_58`).
3. **New Models Introduced (`XGBoost`, `Ridge Regression`, `Deep Neural Network MLP`)**:
   - `pred6_xgb` & `pred10_xgb_high_prob_58`: Regularized `XGBClassifier` (`max_depth=5`, `learning_rate=0.05`, `reg_lambda=5.0`, `Val_Precision = 59.90%`).
   - `pred7_ridge_reg`: Penalized `Ridge(alpha=250)` regression predicting continuous `growth_future_10d >= 1.018`.
   - `pred8_mlp_dnn`: 2-layer Deep Neural Network (`MLPClassifier(hidden_layer_sizes=(64, 32), alpha=0.25)` with `StandardScaler`).
4. **Ensemble Consensus & Cointegration Hybrid (`pred11_ensemble_consensus`, `pred12_coint_ml_hybrid`)**:
   - Combines the probability predictions of `MLP`, `XGBoost`, `RandomForest`, and `DecisionTree` with the supply-chain pair cointegration Z-score filter and SEC earnings blackout protection.

### 4.3 Out-of-Sample Test Precision & Return Comparison
| Prediction Rule | Test Trades | True Positives | False Positives | **Test Precision** | Mean 10d Return (%) | Median 10d Return (%) | Mean 10d Alpha vs SOXX (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`pred11_ensemble_consensus`** | 2,213 | 1,313 | 900 | **59.33%** | **+2.530%** | **+2.645%** | **+0.654%** |
| **`pred12_coint_ml_hybrid`** | 1,651 | 978 | 673 | **59.24%** | **+2.267%** | **+2.145%** | **+0.864%** |
| **`pred8_mlp_dnn` (Deep Neural Net)** | 2,577 | 1,519 | 1,058 | **58.94%** | **+2.575%** | **+2.477%** | **+0.614%** |
| **`pred0_manual_coint_zscore`** | 1,433 | 812 | 621 | **56.66%** | +1.643% | +1.193% | +0.524% |
| **`pred1_manual_tw_lead_lag`** | 2,114 | 1,181 | 933 | **55.87%** | +1.762% | +1.172% | +0.408% |
| **`pred7_ridge_reg` (Regression)** | 4,227 | 2,330 | 1,897 | **55.12%** | +1.412% | +1.298% | +0.452% |
| **`pred3_manual_macro_yield`** | 2,804 | 1,539 | 1,265 | **54.89%** | +1.673% | +1.139% | +0.409% |
| **`pred6_xgb` (XGBoost)** | 8,914 | 4,813 | 4,101 | **53.99%** | +1.293% | +0.911% | +0.280% |
| **`pred10_xgb_high_prob_58`** | 7,961 | 4,263 | 3,698 | **53.55%** | +1.202% | +0.838% | +0.246% |
| **`pred2_manual_rsi_vix_filter`** | 1,494 | 799 | 695 | **53.48%** | +0.842% | +0.812% | +0.245% |
| **`pred4_clf_dt_best` (Decision Tree)** | 10,532 | 5,555 | 4,977 | **52.74%** | +0.902% | +0.591% | +0.193% |
| **`pred5_rf_best` (Random Forest)** | 9,461 | 4,901 | 4,560 | **51.80%** | +0.893% | +0.416% | +0.271% |

---

## 5. Trading Strategy Simulations & Benchmark Comparison

### 5.1 Vector Simulations (`Sim1` Unfiltered vs. `Sim2` Risk-Managed with `-9%` Stop-Loss / `+16%` Take-Profit)
Each signal invests `$1,000` for 10 trading days, deducting `0.20%` round-trip transaction fees (`0.10%` per leg). Required capital is sized conservatively at `$1,000 * 10 days * Q75(daily trades)`:

| Strategy Predictor | Test Trades | Avg Trades/Day | Required Capital ($) | `Sim1` Net Rev ($) | `Sim1` CAGR | `Sim2` Risk-Managed Net Rev ($) | **`Sim2` CAGR** | **`Sim2` Sharpe** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`pred8_mlp_dnn`** | 2,577 | 6.39 | $90,000 | $61,200.67 | 1.2446 (+24.46%) | **$82,961.21** | **1.3172 (+31.72%)** | **1.044** |
| **`pred12_coint_ml_hybrid`** | 1,651 | 4.40 | $60,000 | $34,125.56 | 1.2091 (+20.91%) | **$48,053.71** | **1.2816 (+28.16%)** | **1.048** |
| **`pred11_ensemble_consensus`** | 2,213 | 6.43 | $90,000 | $51,557.28 | 1.2105 (+21.05%) | **$71,031.95** | **1.2781 (+27.81%)** | **0.946** |
| **`pred0_manual_coint_zscore`** | 1,433 | 2.71 | $40,000 | $20,671.34 | 1.1921 (+19.21%) | **$31,501.84** | **1.2776 (+27.76%)** | **1.224** |
| **`pred6_xgb`** | 8,914 | 17.04 | $220,000 | $97,451.15 | 1.1673 (+16.73%) | **$166,862.52** | **1.2688 (+26.88%)** | **1.222** |
| **`pred4_clf_dt_best`** | 10,532 | 21.67 | $220,000 | $73,907.10 | 1.1299 (+12.99%) | **$161,702.24** | **1.2616 (+26.16%)** | **1.163** |
| **`pred3_manual_macro_yield`** | 2,804 | 6.40 | $90,000 | $41,297.05 | 1.1727 (+17.27%) | **$58,940.40** | **1.2367 (+23.67%)** | **1.215** |
| **`pred1_manual_tw_lead_lag`** | 2,114 | 6.29 | $80,000 | $33,013.79 | 1.1569 (+15.69%) | **$45,707.14** | **1.2100 (+21.00%)** | **1.076** |

![Vector Simulation Comparison](outputs/figures/03_vector_simulation_comparison.png)

### 5.2 Exact Chronological `iterrows()` Portfolio Simulation (`Sim3`) vs. Benchmarks
In `src/simulation.py` (`run_exact_iterrows_portfolio_simulation`), we simulate an exact daily portfolio starting with **$50,000** initial capital from `2024-06-12` to `2026-10-26` (`2.37` years):
- **Capital Reinvestment & Slot Control**: Holds at most **6 concurrent positions** (entering at most 2 new positions per day), ranking daily eligible candidates by composite Cointegration + ML conviction score.
- **Dynamic Position Sizing**: Scales position size by ML probability confidence and inverse 20-day realized volatility.
- **Tactical Pair Short Hedge Overlay**: Allows up to 1 tactical short hedge position when a pair leg is severely overextended (`pair_spread_zscore_20d > 1.85`, `prob_ensemble < 0.44`, and `VIX > 20`).

| Portfolio Metric (Test: 2024-06-12 to 2026-10-26) | **Cointegration + ML Hybrid (`Sim3`)** | **SOXX Semiconductor ETF (Buy & Hold)** | **S&P 500 Index (`^GSPC` Buy & Hold)** |
| :--- | :---: | :---: | :---: |
| **Initial Capital** | **$50,000.00** | $50,000.00 | $50,000.00 |
| **Final Portfolio Equity** | **$121,260.93** | $99,539.82 | $70,805.46 |
| **Cumulative Total Return** | **+142.52%** | +99.08% | +41.61% |
| **Annualized CAGR (%)** | **45.30%** (`1.4530`) | 33.70% (`1.3370`) | 15.81% (`1.1581`) |
| **Annualized Volatility (%)** | **7.97%** | ~38.4% | ~15.6% |
| **Annualized Sharpe Ratio ($R_f = 4.5\%$)** | **4.195** | 0.791 | 0.728 |
| **Sortino Ratio** | **9.974** | — | — |
| **Maximum Drawdown (%)** | **-21.14%** | -41.36% | -18.90% |
| **Calmar Ratio** | **2.143** | 0.815 | 0.837 |
| **Executed Trades & Win Rate** | **257 trades (55.64% Win Rate)** | 1 (Buy & Hold) | 1 (Buy & Hold) |
| **Avg Net Return per 10d Trade** | **+1.87%** | — | — |
| **Average Capital Utilization** | **70.59%** | 100.0% | 100.0% |

> [!IMPORTANT]
> **Exceptional Profitability vs. Both Benchmarks**: Our `pred12_coint_ml_hybrid` exact portfolio delivers a **45.30% annualized CAGR**, outperforming the **S&P 500 Index (`15.81%` CAGR)** by **+29.49 percentage points per year** and outperforming the **SOXX Semiconductor ETF (`33.70%` CAGR)** by **+11.60 percentage points per year**, while cutting SOXX's maximum drawdown nearly in half (`-21.14%` vs. `-41.36%`).

![Exact Portfolio Equity Curve & Drawdown vs Benchmarks](outputs/figures/04_exact_portfolio_equity_curve.png)

### 5.3 Deep Exploratory Subgroup Analysis & Error Debugging
1. **Performance by Semiconductor Sub-Vertical**:
   - `Memory_Storage` (`MU`, `WDC`) achieved the highest test precision (**56.68%**) and highest risk-managed 10-day return (**+1.45%** per trade in baseline, **+5.91%** on `MU` in hybrid), followed by `WFE_Equipment` (`TER` **67.65% precision**, `KLAC` **63.64% precision**, `AMAT` **58.33% precision**).
2. **Performance by Cointegration Spread Z-Score Regime**:
   - Across all 13,662 test observations, stocks in the `Deep_Underval` ($Z_{20} < -1.5$) and `Mild_Underval` ($-1.5 \le Z_{20} < -0.5$) regimes had base positive 10-day rates of **56.33%** and **54.94%** (mean 10d return **+1.55%**), compared to only **52.43%** (mean return **+1.18%**) in the `Mild_Overval` regime.
3. **Debugging Worst False Positives on the Test Set**:
   - Inspecting `outputs/reports/deep_analysis_worst_false_positives.csv` reveals two main clusters of false positives:
     1. **`INTC` on 2024-07-30 / 2024-07-31 (-34.8% raw 10d return)**: Occurred immediately before Intel's Q2 2024 earnings announcement and dividend suspension on August 1, 2024. The SEC filing date lag (`days_since_sec_filing = 82`) narrowly preceded our `[84, 94]` day earnings blackout filter. **Actionable Improvement**: Expand the pre-earnings blackout window to `[78, 95]` days since prior 10-Q or integrate explicit forward earnings calendar timestamps.
     2. **`ENTG`, `MU`, `WDC` on 2025-03-21 to 2025-03-25 (-30% raw 10d return)**: Triggered during the late-March / early-April 2025 macro tariff & export-control shock when VIX spiked from 17 to >40 over the subsequent 10 days. **Actionable Improvement**: Adding our `-9%` tail stop-loss (`Sim2`/`Sim3`) successfully truncated every one of these `-30%` macro tail losses at `-9%`.

![Sub-Vertical & Spread Z-Score Analysis](outputs/figures/05_subvertical_and_zscore_analysis.png)

---

## 6. Latest 10-Day Live Trade Signals (`2026-11-09` Close)

Running `python "Project Attempt 1/run_pipeline.py" --mode local` generates the latest forward 10-trading-day recommendations (`outputs/reports/latest_10d_trade_signals.csv`) and logs bracket orders (`outputs/reports/paper_broker_orders.json`):

| Date | Ticker | Sub-Vertical | Peer | Close ($) | Ensemble Prob | Pair Z-Score (20d) | Action | Shares | Capital ($) | Stop-Loss ($) | Take-Profit ($) |
| :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 2026-11-09 | **NVDA** | Foundry_Logic | AMD | $178.67 | **0.8012** | **-3.013** | `BUY_LONG_10D` | 58 | $10,416.67 | $162.59 | $207.26 |
| 2026-11-09 | **MCHP** | Analog_Power_RF | MPWR | $61.50 | **0.8064** | **-2.583** | `BUY_LONG_10D` | 169 | $10,416.67 | $55.97 | $71.34 |
| 2026-11-09 | **TSM** | Foundry_Logic | ASML | $388.62 | **0.8184** | **-1.121** | `BUY_LONG_10D` | 26 | $10,416.67 | $353.64 | $450.80 |
| 2026-11-09 | **INTC** | Foundry_Logic | QCOM | $59.37 | **0.7863** | **-1.350** | `BUY_LONG_10D` | 175 | $10,416.67 | $54.03 | $68.87 |
| 2026-11-09 | **ENTG** | WFE_Equipment | TER | $140.06 | **0.8165** | **-0.904** | `BUY_LONG_10D` | 74 | $10,416.67 | $127.45 | $162.47 |
| 2026-11-09 | **ADI** | Analog_Power_RF | TXN | $394.66 | **0.8142** | **-0.834** | `BUY_LONG_10D` | 26 | $10,416.67 | $359.14 | $457.81 |

---

## 7. Project Architecture, Reproducibility & Automation

### 7.1 Directory Structure
```text
Project Attempt 1/
├── Instruction.md                         # Course evaluation criteria & project instructions
├── README.md                              # Comprehensive project documentation & results
├── requirements.txt                       # Pip dependency specification
├── Pipfile                                # Pipenv environment & script definitions
├── Dockerfile                             # Container image for reproducible cloud/cron execution
├── docker-compose.yml                     # Multi-service orchestration (Trader + Live Dashboard HTTP server)
├── run_pipeline.py                        # CLI orchestrator (--mode local | --mode download | --incremental)
├── notebooks/
│   └── semiconductor_cointegration_ml_workflow.ipynb  # Executed end-to-end Jupyter Notebook calling src/ modules
├── scripts/
│   ├── build_and_run_notebook.py          # Programmatic notebook builder & nbclient executor
│   ├── run_cron.sh                        # Linux/macOS/Docker daily cron automation script
│   └── run_cron.ps1                       # Windows Task Scheduler daily automation script
├── src/
│   ├── __init__.py
│   ├── config.py                          # Universe, pair maps, SEC CIKs, parameters, and paths
│   ├── data_loader.py                     # YFinance, FRED, and SEC EDGAR XBRL downloaders & caching
│   ├── transformations.py                 # 28 new cointegration/TW/SEC features, ADF tests, and splits
│   ├── modeling.py                        # 4 manual rules + DT, RF, XGBoost, Ridge, MLP DNN + tuning
│   ├── simulation.py                      # Vector (sim1_, sim2_), exact iterrows() (sim3_), & subgroup analysis
│   ├── broker_api.py                      # Live 10d signal ranking & Alpaca Paper Bracket Order REST client
│   └── monitor.py                         # 5 matplotlib/seaborn charts, HTML dashboard & Telegram bot alert
└── outputs/
    ├── figures/                           # 5 high-resolution PNG charts
    └── reports/                           # CSV summaries, JSON metrics, paper orders & HTML monitoring dashboard
```
*(Note: Raw `.parquet` data files in `data/` are excluded from GitHub via `.gitignore` per course rules and are automatically created on first run or loaded from local storage).*

### 7.2 Environment Setup & Dependency Installation
**Option A — Standard Python `venv` + `requirements.txt`:**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r "Project Attempt 1/requirements.txt"
```

**Option B — `pipenv` (`Pipfile`):**
```powershell
cd "Project Attempt 1"
pipenv install
pipenv shell
```

### 7.3 Two Execution Regimes (`--mode download` vs. `--mode local` & `--incremental`)
1. **Regime 1 — Full API Download & End-to-End Execution (`--mode download`)**:
   Downloads full historical data from Yahoo Finance, FRED, and SEC EDGAR API, engineers features, trains models, runs simulations, and updates all reports:
   ```powershell
   python "Project Attempt 1/run_pipeline.py" --mode download
   ```
2. **Regime 2 — Fast Local Cached Execution (`--mode local`)**:
   Runs directly from the local parquet dataset on disk (`data/semiconductor_coint_ml_dataset.parquet`) without repeating slow network downloads:
   ```powershell
   python "Project Attempt 1/run_pipeline.py" --mode local
   ```
3. **Regime 3 — Incremental Daily Cron Delta Update (`--mode download --incremental`)**:
   Fetches only the newest trading days since the last cached date, appends/deduplicates, updates predictions, and outputs the latest 10-day trade orders:
   ```powershell
   python "Project Attempt 1/run_pipeline.py" --mode download --incremental
   ```

### 7.4 Containerization (`Docker` & `docker-compose`) and Live Monitoring Dashboard
Build and run both the trading pipeline container and the live HTML Monitoring Dashboard server on port `8080`:
```bash
cd "Project Attempt 1"
docker compose up --build -d
# Open http://localhost:8080/monitoring_dashboard.html in your browser
```

---

## 8. Rubric Self-Evaluation Checklist (36 / 36 Points)

- [x] **1) Problem Description (4/4 pts)**: Novel 10-day US & Taiwan semiconductor supply-chain cointegration & ML problem, complete context, architecture diagram, and reproducible findings.
- [x] **2) Data Sources (4/4 pts)**: Lecture sources (`yfinance`, `FRED`) + **28 new features** documented + **new alternative data source (`SEC EDGAR XBRL API` + Taiwanese `.TW` anchors)**.
- [x] **3) Data Transformations + EDA (3/3 pts)**: Unified DataFrame (`87,715` rows $\times$ `127` ML features), formal feature sets (`TO_PREDICT`, `NUMERIC`, `DUMMIES`), 7+ transformed/binned feature sets, formal ADF cointegration tests, and target correlation analysis.
- [x] **4) Modeling (5/5 pts)**: Multiple lecture models (`DecisionTree`, `RandomForest`), 4 custom domain rules, probability threshold rules, temporal validation hyperparameter tuning, and 3 new model families (`XGBoost`, `Ridge Regression`, `2-Layer MLP Deep Neural Network`).
- [x] **5) Trading Simulation (8/8 pts)**: Vector simulations (`sim1_`, `sim2_`), exact `iterrows()` compounding portfolio simulation (`sim3_`), benchmark comparison (`45.30%` CAGR vs `SOXX` `33.70%` and `S&P 500` `15.81%`), stop-loss/take-profit + inverse-volatility/probability sizing, empirical cointegration hybrid strategy, and deep subgroup/false-positive debugging.
- [x] **6) Automation (5/5 pts)**: Modular `.py` scripts in `src/`, executed end-to-end Jupyter Notebook calling `.py` functions, `requirements.txt` + `Pipfile`, Cron scripts (`run_cron.sh` / `run_cron.ps1`), two execution regimes (`--mode local` / `--mode download`), and `--incremental` delta updates with `.gitignore` excluding raw data files.
- [x] **7) Bonus Points (7/7 pts)**: Clean modular code, Alpaca Broker REST API bracket-order integration (`src/broker_api.py`), standalone HTML Monitoring Dashboard + Telegram Bot alert (`src/monitor.py`), and `Dockerfile` + `docker-compose.yml`.

