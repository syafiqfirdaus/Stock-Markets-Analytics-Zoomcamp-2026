# Module 3: Time Series Modeling

This module covers predictive modeling on financial time series, including engineering seasonal calendar dummy variables, designing heuristic "handcrafted" trading rules from decision boundaries, assessing unique predictive contributions from machine learning models, and hyperparameter tuning of Decision Tree Classifiers.

The calculations for Homework 3 are fully reproducible via [homework3_solution.py](file:///c:/Users/muhds/.gemini/antigravity-ide/scratch/Stock-Markets-Analytics-Zoomcamp-2026/03-modeling/homework3_solution.py).

---

## Python Environment & Execution

Run the complete solution from the repository root:

```powershell
python .\03-modeling\homework3_solution.py
```

To run individual questions:

```powershell
python .\03-modeling\homework3_solution.py --question 1
python .\03-modeling\homework3_solution.py --question 2
python .\03-modeling\homework3_solution.py --question 3
python .\03-modeling\homework3_solution.py --question 4
python .\03-modeling\homework3_solution.py --question 5
```

---

## Homework 3 Answers

Submission form: [https://courses.datatalks.club/sma-zoomcamp-2026/homework/hw03](https://courses.datatalks.club/sma-zoomcamp-2026/homework/hw03)

| Question | Form Options | Submission Answer | Exact Supporting Result |
| :--- | :--- | :---: | :--- |
| **1. Absolute correlation of most correlated `<month>_w<wom>` dummy** | 0.015, 0.025, 0.035, 0.045 | **0.025** | `month_wom_October_w4` has the highest correlation with `is_positive_growth_30d_future` at **0.024584** (absolute value **0.025** when rounded to 3 decimal places). |
| **2. Precision score for best of NEW predictions (`pred3` or `pred4`)** | 0.558, 0.568, 0.578, 0.588, 0.598 | **0.588** | `pred3_manual_dgs10_5` (`(DGS10 <= 4.5) & (DGS5 <= 4)`) achieves **0.58774** (**0.588** rounded) on the TEST dataset (15,910 positive calls, 9,351 TP). |
| **3. Records in TEST where only `pred5_clf_10` is correct and all hand rules incorrect** | 271, 571, 1171, 1571 | **1171** | Decision tree (`max_depth=10, random_state=42`) trained on Train+Validation produces **1,171** uniquely correct positive records on the TEST dataset. |
| **4. Optimal tree depth (1 to 12) for DecisionTreeClassifier** | 4, 6, 8, 12 | **4** | **`max_depth=4`** achieves the highest test precision (**0.62917** / **62.92%**), outperforming deeper trees that suffer from over-fitting. |
| **5. [EXPLORATORY] What data is missing?** | Free-form text | *See writeup below* | Regional non-US macro indicators (RBI, ECB, FX), volatility/fear regimes (VIX, India VIX), fundamental valuation metrics (P/E, FCF), and institutional flow data (FII/DII). |

---

## Detailed Question Analysis

### Question 1: Dummies for Month and Week-of-Month

- **Formula**: `(d.day - 1) // 7 + 1` maps day of month into week index 1 through 5.
- **Combined Feature**: `<Month_Name>_w<Week_Number>` (e.g., `'October_w1'`, `'October_w4'`).
- **Feature Count**: Adding `month_wom` expands categorical dummies from 55 to **115 total dummy variables** (60 new month-week combinations).
- **Target**: Binary future outcome `is_positive_growth_30d_future`.

#### Top 5 Correlated Seasonal Dummy Variables

| Dummy Feature | Linear Correlation | Absolute Correlation (`abs_corr`) | Rounded Answer |
| :--- | :---: | :---: | :---: |
| **`month_wom_October_w4`** | **+0.024584** | **0.024584** | **0.025** |
| `month_wom_November_w3` | +0.023251 | 0.023251 | 0.023 |
| `month_wom_February_w1` | -0.020861 | 0.020861 | 0.021 |
| `month_wom_January_w2` | -0.019846 | 0.019846 | 0.020 |
| `month_wom_January_w5` | -0.019792 | 0.019792 | 0.020 |

---

### Question 2: Define New 'Hand' Rules on Macro and Technical Indicators

Evaluating manual heuristic rules extracted from decision tree nodes:
- `pred3_manual_dgs10_5`: `(DGS10 <= 4.5) & (DGS5 <= 4)`
- `pred4_manual_dgs10_fedfunds`: `(DGS10 > 4) & (FEDFUNDS <= 4.795)`

#### Performance on TEST Dataset (33,163 records)

| Prediction Rule | Positive Predictions | True Positives (TP) | False Positives (FP) | Precision | Rounded (3 dec) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`pred3_manual_dgs10_5`** | **15,910** | **9,351** | **6,559** | **0.58774** | **0.588** |
| `pred4_manual_dgs10_fedfunds` | 15,921 | 8,001 | 7,920 | 0.50254 | 0.503 |
| *Baseline `pred0_manual_cci`* | 921 | 520 | 401 | 0.56460 | 0.565 |
| *Baseline `pred1_manual_prev_g1`* | 19,158 | 11,086 | 8,072 | 0.57866 | 0.579 |
| *Baseline `pred2_manual_prev_g1_and_snp`* | 15,244 | 8,728 | 6,516 | 0.57255 | 0.573 |

The best new hand rule is `pred3_manual_dgs10_5` with precision **0.588**.

---

### Question 3: Unique Correct Predictions from Decision Tree Classifier (`pred5_clf_10`)

- **Model Specification**: `DecisionTreeClassifier(max_depth=10, random_state=42)`
- **Training Set**: Combined `train` + `validation` subsets (169,129 records).
- **Test Set**: 33,163 records.
- **Criteria**:
  - `pred5_clf_10` matches ground truth (`is_positive_growth_30d_future`).
  - ALL hand rule predictions (`pred0` through `pred4`) mismatch the ground truth.
- **Result**: Exactly corresponds to form option **1171**.

---

### Question 4: Hyperparameter Tuning for Decision Tree (Depth 1 to 12)

Iterating `max_depth` from 1 to 12 on Train+Validation and measuring test performance:

| Depth (`max_depth`) | Test Precision | Test Accuracy | Note |
| :---: | :---: | :---: | :--- |
| 1 | 0.57009 | 0.55167 | Underfitting / coarse split |
| 2 | 0.62253 | 0.53982 | Strong gain |
| 3 | 0.57197 | 0.54018 | Suboptimal split threshold |
| **4** | **0.62917** | **0.54519** | **Optimal depth (highest precision)** |
| 5 | 0.59997 | 0.53852 | Onset of complexity degradation |
| 6 | 0.60621 | 0.54063 | Plateau |
| 7 | 0.62107 | 0.52157 | Overfitting secondary branches |
| 8 | 0.62179 | 0.51868 | Overfitting secondary branches |
| 9 | 0.61118 | 0.51268 | Declining generalization |
| 10 | 0.59121 | 0.50614 | Pronounced overfitting |
| 11 | 0.60275 | 0.49968 | Noise memorization |
| 12 | 0.58194 | 0.49208 | Significant degradation |

Optimal depth: **`4`** (Precision **62.92%**).

---

### Question 5: [EXPLORATORY] What data is missing?

#### 1. Cross-Market Regional Macro Indicators
The current dataset applies US macroeconomic series (`FEDFUNDS`, `DGS10`, `CPI Core`) uniformly across all 33 stocks, even though the dataset comprises global equities from the US, India (NSE), and the European Union (EU).
- **Missing Series**:
  - **India**: Reserve Bank of India (RBI) repo rate, India CPI inflation, WPI, and Industrial Production Index (IIP).
  - **Europe**: European Central Bank (ECB) deposit facility rate and main refinancing rate, Eurozone Harmonised Index of Consumer Prices (HICP), and German Bund 10Y yield.
  - **FX Rates & Volatility**: `USD/INR` and `EUR/USD` daily returns and 30-day realized volatility. Currency fluctuations materially impact foreign institutional capital flows and multinational profit margins.
- **Source**: Federal Reserve Bank of St. Louis (FRED), Reserve Bank of India Database on Indian Economy (DBIE), European Central Bank Data Portal.

#### 2. Market Sentiment and Risk-Regime Indicators
Decision trees heavily split on macroeconomic yields because macro regime shifts dictate equity risk appetite. Incorporating forward-looking volatility indicators directly parameterizes market regime:
- **Missing Series**:
  - **Implied Volatility Indices**: `^VIX` (US CBOE VIX), `^INDIAVIX` (NSE India VIX), and `^V2TX` (Euro Stoxx 50 Volatility).
  - **Credit Risk Spreads**: US High Yield Option-Adjusted Spread (OAS) and Investment Grade OAS. Widening credit spreads precede equity market pullbacks and liquidity contractions.
  - **Term Spread Inversion**: 10Y minus 2Y and 10Y minus 3M Treasury yield spreads, which provide leading signals for economic downturns.
- **Source**: Yahoo Finance (`^VIX`, `^INDIAVIX`), FRED (`BAMLH0A0HYM2`, `T10Y2Y`).

#### 3. Fundamental Valuation and Corporate Earnings Events
Pure price momentum and technical indicators fail during fundamental repricing events. Adding quarterly financial health signals anchors model predictions:
- **Missing Series**:
  - **Valuation Ratios**: Trailing P/E, Forward P/E, Enterprise Value to EBITDA (EV/EBITDA), and Free Cash Flow Yield.
  - **Earnings Calendar & Surprise**: Days until next quarterly earnings announcement and most recent EPS surprise percentage (capturing the well-known Post-Earnings Announcement Drift / PEAD anomaly).
- **Source**: SEC EDGAR, Financial Modeling Prep (FMP) API, SimFin.

#### 4. Institutional Order Flow & Market Microstructure
For international and emerging markets like India, institutional money flows dictate directional persistence:
- **Missing Series**:
  - **FII/DII Net Activity**: Daily net buy/sell values by Foreign Institutional Investors (FII) and Domestic Institutional Investors (DII).
  - **Short Interest**: Short interest ratio (days to cover) for US equities.
- **Source**: National Stock Exchange of India (NSE), SEBI daily reports, FINRA.
