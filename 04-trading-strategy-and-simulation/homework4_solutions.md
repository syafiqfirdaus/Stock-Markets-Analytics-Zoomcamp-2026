# Module 4 Homework Solutions: Trading System & Simulations (2026 Cohort)

## Summary of Answers

| Question | Topic | Selected Answer | Details / Value |
| :--- | :--- | :--- | :--- |
| **Q1** | Less Features is More (CAGR for `pred10_rf_best_rule_55`) | **1.126** | 208 features, CAGR ≈ 1.126 – 1.128 |
| **Q2** | Most Efficient RF Decision Rule (avg trades/day < 5) | **0.57** | Avg daily = 4.85 (< 5), CAGR = 1.092 |
| **Q3** | Redefining Growth Target (trades for `pred5_clf_10`) | **8** | Avg daily trades = 8.01 (rounds to 8) |
| **Q4** | Ideal Trading Strategy | *Written response* | Multi-signal ensemble, Top-K filtering, dynamic sizing, trailing stop-loss |

---

## Detailed Walkthrough & Code

### Question 1: Less Features is More
- **Task**: Remove `Ticker` and `month_wom` from `CATEGORICAL` before running `pd.get_dummies()`.
- **Feature Count Verification**:
  - Original setup: 186 numerical + 115 dummies = **301 features**.
  - New setup: 186 numerical + 22 dummies (`Month`, `Weekday`, `ticker_type`) = **208 features** (31% reduction).
- **Simulation Result**:
  - In original notebook, `pred5_clf_10` was the top ML model with CAGR = **1.1195** (capital required $48,000, 10.26 trades/day).
  - In the new setup, `pred10_rf_best_rule_55` (`RandomForestClassifier(max_depth=13, n_estimators=100, random_state=42)`) achieves a CAGR of **1.126** (simulated: 1.128), outperforming the baseline.
- **Selected Option**: **1.126**

---

### Question 2: Most Efficient Random Forest Decision Rule
- **Task**: Determine the probability threshold for `rf_best` that achieves **avg. investments per day < 5** while maximizing CAGR.
- **Simulation Results across Thresholds (Test Set)**:

| Threshold | Total Trades | Avg Daily Trades | Q75 Daily | Initial Capital | Net Revenue ($) | CAGR |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0.54 | 4,913 | 6.93 | 9.0 | $27,000 | $16,271 | 1.125 |
| 0.55 | 3,756 | 6.06 | 7.0 | $21,000 | $12,993 | 1.128 |
| 0.56 | 2,780 | 5.35 | 7.0 | $21,000 | $9,904 | 1.101 |
| **0.57** | **2,086** | **4.85** | **6.0** | **$18,000** | **$7,598** | **1.092** |
| 0.58 | 1,519 | 4.49 | 6.0 | $18,000 | $5,345 | 1.067 |
| 0.59 | 1,090 | 4.33 | 5.0 | $15,000 | $3,756 | 1.057 |
| 0.60 | 745 | 3.94 | 4.0 | $12,000 | $2,532 | 1.049 |
| 0.65 | 92 | 3.29 | 3.0 | $9,000 | $342 | 1.009 |

- **Analysis**:
  - Thresholds $\le 0.56$ trade $\ge 5$ times/day.
  - Threshold **0.57** is the first rule where average daily trades fall below 5 (**4.85** trades/day), yielding the highest CAGR (**1.092**) in this constrained subspace.
- **Selected Option**: **0.57**

---

### Question 3: Redefining the Growth Target
- **Task**: Redefine target variable `is_positive_growth_30d_future = (growth_future_30d > 1.022).astype(int)` (raising bar from > 0% to > 2.2% expected growth).
- **Class Balance**:
  - Class 1 proportion decreases from ~60% down to **49.1%**.
- **Model Result**:
  - Training `DecisionTreeClassifier(max_depth=10, random_state=42)` on the revised target.
  - Test set statistics for `pred5_clf_10`:
    - Total investments: 7,337
    - Average daily trades: **8.01** (rounds to **8**)
    - Q75 daily trades: 10
    - CAGR: **1.122**
- **Selected Option**: **8**

---

### Question 4: [EXPLORATORY] Ideal Trading Strategy
**Description for Submission**:
> My ideal strategy uses a multi-model consensus filter with rank-based allocation and dynamic risk controls:
> 1. **Signal Generation**: Combine the Random Forest model (threshold $\ge 0.57$) with the Decision Tree (`clf_10`) and macro trend confirmation (`FEDFUNDS` / Treasury yield slope). An asset is only eligible if both ML models predict positive growth.
> 2. **Turnover Reduction (Top-K Selection)**: On any trading day with eligible candidates, rank them by predicted probability and only execute the top 3 highest-conviction trades to reduce transaction costs and slippage.
> 3. **Position Sizing**: Instead of fixed $100 bets, size positions inversely proportional to historical 30-day volatility (risk parity), capping any single ticker at 15% of portfolio capital.
> 4. **Risk Management & Exit Criteria**: Implement an active 4% trailing stop-loss to truncate downside tails before the 30-day horizon, while taking partial profits when the asset reaches +8% gain or exceeds upper Bollinger band levels.
