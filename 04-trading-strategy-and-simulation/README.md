# Module 4: Trading Strategy and Simulation

This module covers simulated trading strategies on stock price predictions, evaluating financial outcomes (CAGR, capital requirement, gross/net revenue, transaction fees), optimizing decision rules via probability thresholds, and refining growth targets.

The calculations for Homework 4 are fully reproducible via [homework4_solution.py](file:///c:/Users/muhds/.gemini/antigravity-ide/scratch/Stock-Markets-Analytics-Zoomcamp-2026/04-trading-strategy-and-simulation/homework4_solution.py).

---

## Python Environment & Execution

Run the complete solution from the repository root:

```powershell
python .\04-trading-strategy-and-simulation\homework4_solution.py
```

To run individual questions:

```powershell
python .\04-trading-strategy-and-simulation\homework4_solution.py --question 1
python .\04-trading-strategy-and-simulation\homework4_solution.py --question 2
python .\04-trading-strategy-and-simulation\homework4_solution.py --question 3
python .\04-trading-strategy-and-simulation\homework4_solution.py --question 4
```

---

## Homework 4 Answers

Submission form: [https://courses.datatalks.club/sma-zoomcamp-2026/homework/hw04](https://courses.datatalks.club/sma-zoomcamp-2026/homework/hw04)

| Question | Form Options | Submission Answer | Exact Supporting Result |
| :--- | :--- | :---: | :--- |
| **1. CAGR for `pred10_rf_best_rule_55` with fewer dummy variables** | 1.121, 1.126, 1.136, 1.141 | **1.126** | Reducing features from 301 to 208 (excluding `Ticker` and `month_wom`) boosts Random Forest performance. Simulated CAGR reaches **1.126** (exact simulated: 1.128), outperforming the baseline best ML model (`pred5_clf_10` CAGR = 1.1195). |
| **2. Probability threshold for second-best RF rule with avg trades/day < 5** | 0.54, 0.56, 0.57, 0.58 | **0.57** | Threshold **0.57** is the first decision rule achieving average daily investments below 5 (**4.85** trades/day) with the highest CAGR (**1.092**) in this constrained subspace. |
| **3. Number of trades for best predictor `pred5_clf_10` after redefining target** | 6, 7, 8, 9 | **8** | Redefining `is_positive_growth_30d_future` to `growth_future_30d > 1.022` lowers positive frequency to ~49.1%. `pred5_clf_10` (`max_depth=10`) averages **8.01** trades/day on the test set, rounding to **8**. |
| **4. [EXPLORATORY] Ideal Trading Strategy** | Free-form text | *See writeup below* | Multi-model consensus filter, top-3 daily rank allocation, inverse-volatility position sizing, and a 4% trailing stop-loss. |

---

## Detailed Question Analysis

### Question 1: Less Features is More (CAGR for `pred10_rf_best_rule_55`)

- **Objective**: Assess model generalizability when noisy, high-cardinality categorical dummy features (`Ticker` and `month_wom`) are pruned prior to model training.
- **Feature Set Comparison**:
  - *Original setup*: 186 numerical + 115 dummies = **301 total features**.
  - *New setup*: 186 numerical + 22 dummies (`Month`, `Weekday`, `ticker_type`) = **208 total features** (~31% reduction).
- **Simulation Results (TEST Dataset)**:
  - In the original setup, `pred5_clf_10` achieved CAGR = **1.1195** ($48,000 capital, 10.26 trades/day).
  - In the new setup, `RandomForestClassifier(n_estimators=100, max_depth=13, random_state=42)` benefits from lower dimensional noise, and `pred10_rf_best_rule_55` achieves a CAGR of **1.126** (simulated: 1.128, $21,000 capital, 6.06 trades/day).
- **Selected Answer**: **1.126**

---

### Question 2: Most Efficient Random Forest Decision Rule

- **Objective**: Find the threshold among $\{0.54, 0.55, \dots, 0.65\}$ with `sim1_avg_investments_per_day < 5` delivering the highest financial return.
- **Threshold Analysis**:

| Threshold | Total Trades | Avg Daily Trades | Q75 Daily | Initial Capital | Net Revenue ($) | CAGR | Subspace Candidate |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0.54 | 4,913 | 6.93 | 9.0 | $27,000 | $16,271 | 1.125 | No ($\ge 5$) |
| 0.55 | 3,756 | 6.06 | 7.0 | $21,000 | $12,993 | 1.128 | No ($\ge 5$) |
| 0.56 | 2,780 | 5.35 | 7.0 | $21,000 | $9,904 | 1.101 | No ($\ge 5$) |
| **0.57** | **2,086** | **4.85** | **6.0** | **$18,000** | **$7,598** | **1.092** | **Yes (Best CAGR < 5)** |
| 0.58 | 1,519 | 4.49 | 6.0 | $18,000 | $5,345 | 1.067 | Yes (Lower CAGR) |
| 0.59 | 1,090 | 4.33 | 5.0 | $15,000 | $3,756 | 1.057 | Yes (Lower CAGR) |
| 0.60 | 745 | 3.94 | 4.0 | $12,000 | $2,532 | 1.049 | Yes (Lower CAGR) |
| 0.65 | 92 | 3.29 | 3.0 | $9,000 | $342 | 1.009 | Yes (Lower CAGR) |

- **Conclusion**: Threshold **0.57** provides the optimal trade-off between time efficiency (fewer than 5 average daily transactions) and capital growth (CAGR 1.092).
- **Selected Answer**: **0.57**

---

### Question 3: Redefining the Growth Target

- **Objective**: Target stronger upward moves by raising the threshold from nominal positive growth (`growth_future_30d > 1.0`) to meaningful appreciation (`growth_future_30d > 1.022`, or > 2.2%).
- **Impact on Target Distribution**:
  - Class 1 frequency drops from 59.8% to **49.1%**, producing a balanced classification problem.
- **Performance of `pred5_clf_10` (`DecisionTreeClassifier(max_depth=10, random_state=42)`)**:
  - Test set investments: 7,337 (down from 9,861 in the baseline).
  - Average daily investments: **8.01** trades/day, rounded to **8**.
  - Q75 daily trades: 10.
  - Net revenue: $26,380.
  - CAGR: **1.122**.
- **Selected Answer**: **8**

---

### Question 4: [EXPLORATORY] Ideal Trading Strategy

> My ideal strategy uses a multi-model consensus filter with rank-based allocation and dynamic risk controls:
> 1. **Signal Generation**: Combine the Random Forest model (threshold $\ge 0.57$) with the Decision Tree (`clf_10`) and macro trend confirmation (`FEDFUNDS` / Treasury yield slope). An asset is only eligible if both ML models predict positive growth.
> 2. **Turnover Reduction (Top-K Selection)**: On any trading day with eligible candidates, rank them by predicted probability and only execute the top 3 highest-conviction trades to reduce transaction costs and slippage.
> 3. **Position Sizing**: Instead of fixed $100 bets, size positions inversely proportional to historical 30-day volatility (risk parity), capping any single ticker at 15% of portfolio capital.
> 4. **Risk Management & Exit Criteria**: Implement an active 4% trailing stop-loss to truncate downside tails before the 30-day horizon, while taking partial profits when the asset reaches +8% gain or exceeds upper Bollinger band levels.
