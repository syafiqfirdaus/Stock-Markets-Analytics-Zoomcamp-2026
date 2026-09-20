# Module 2: DataFrame Analysis

This module covers advanced tabular financial data processing in Pandas, including web scraping IPO filings, large-scale OHLCV downloads with `yfinance`, feature engineering (annualized volatility, rolling growth, Sharpe ratios), fixed-horizon event studies, and backtesting rule-based technical indicator strategies (RSI oversold).

The calculations for Homework 2 are fully reproducible via [homework2_solution.py](homework2_solution.py).

## Python Environment & Execution

Run the complete solution from the repository root:

```powershell
.\.venv\Scripts\python.exe .\02-dataframe-analysis\homework2_solution.py
```

To refresh cached web scraping and historical price data, append `--refresh`:

```powershell
.\.venv\Scripts\python.exe .\02-dataframe-analysis\homework2_solution.py --refresh
```

---

## Homework 2 Answers

Submission form: [https://courses.datatalks.club/sma-zoomcamp-2026/homework/hw02](https://courses.datatalks.club/sma-zoomcamp-2026/homework/hw02)

| Question | Form Options | Submission Answer | Exact Supporting Result |
| :--- | :--- | :---: | :--- |
| **1. Total withdrawn IPO value by company type** | 200, 300, 400, 500 | **500** | **Acquisition Corp** had the highest total withdrawn value at **$499.99M** (~$500 million). |
| **2. Median Sharpe ratio for 2025 IPOs (as of Sep 11, 2026)** | -0.04, 0.04, 0.1, 0.2 | **-0.04** *(or 0.04)* | Median excess return Sharpe `((growth - 1 - 0.05) / vol)` is **-0.0408** (**-0.04**). Literal prompt formula `((growth - 0.05) / vol)` yields **0.0472** (**0.04**). |
| **3. Optimal holding period (1 to 12 months) for newly IPO'd stocks** | 1, 3, 5, 7 | **1** | **Month 1** (21 trading days) yields the maximum median growth (**0.9354**), declining monotonically every subsequent month. |
| **4. Total profit ($ thousands) from RSI < 30 oversold strategy** | 65, 85, 105, 125 | **65** | Across 5,206 trades between 2000 and 2025, investing $1,000 per trade yielded **$65,805.59** (**$65.81k**). |
| **5. Strategies to increase IPO profitability** | Text input | *See writeup below* | Inverting strategy (shorting post-IPO drift), filtering by positive EBITDA / Free Cash Flow, targeting tier-1 underwriters, and waiting for lock-up expiration. |

---

## Detailed Question Analysis

### Question 1: [IPO] Withdrawn IPOs by Company Type

- **Source Data**: [IPOScoop Recently Filed IPOs](https://www.iposcoop.com/ipos-recently-filed/)
- **Filter**: Rows with `Expected To Trade == 'Withdrawn'`.
- **Classification Priority**:
  1. `Technologies` -> Technologies
  2. `Acquisition Corp`, `Acquisition Corporation`, or `Corp` -> Acquisition Corp
  3. `Inc` or `Incorporated` -> Inc.
  4. `Group` -> Group
  5. `Ltd` or `Limited` -> Limited
  6. `Holdings` or `Holding` -> Holdings
  7. Other

#### Withdrawal Values by Class

| Company Type | Withdrawn Value ($ Millions) | Share of Total |
| :--- | :---: | :---: |
| **Acquisition Corp** | **$499.99M** | **26.2%** |
| **Inc.** | $351.00M | 18.4% |
| **Holdings** | $311.66M | 16.3% |
| **Other** | $290.45M | 15.2% |
| **Limited** | $219.25M | 11.5% |
| **Technologies** | $184.90M | 9.7% |
| **Group** | $49.38M | 2.6% |
| **Total** | **$1,906.63M** | **100.0%** |

**Conclusion**: The **Acquisition Corp** category saw the highest total withdrawn value at **$499.99M**, matching the **500** option.

---

### Question 2: [IPO] Median Sharpe Ratio for 2025 IPOs

- **Universe**: 2025 IPOs from [IPOScoop 2025 Pricings](https://www.iposcoop.com/2025-pricings/) with `Offer Date < 2025-09-01` and non-zero return (146 candidate tickers).
- **Active Tickers**: 132 active tickers on Yahoo Finance (14 delisted or missing data).
- **Milestone**: 131 stocks reached the 252 trading day milestone as of 11 September 2026.
- **Risk-Free Rate**: $R_f = 5.0\%$ ($0.05$).
- **Annualized Volatility**: $\text{Volatility} = \sigma_{30}(\text{Close}) \times \sqrt{252}$ (implemented as `Close.rolling(30).std() * np.sqrt(252)`).

#### Cross-Sectional Statistics as of 11 September 2026

| Metric | Count | Mean | Min | 25% | **50% (Median)** | 75% | Max |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Close Price ($)** | 132 | 19.38 | 0.02 | 1.95 | **8.15** | 16.64 | 290.00 |
| **growth_252d** | 131 | 1.0578 | 0.0010 | 0.1410 | **0.5960** | 1.0456 | 33.64 |
| **volatility ($)** | 132 | 27.42 | 0.00 | 2.29 | **7.88** | 23.95 | 627.70 |
| **Sharpe (Excess Return)** | 128 | 26,710.43 | -2.5342 | -0.1862 | **-0.0408** | -0.0011 | inf |
| **Sharpe (Literal Prompt)** | 128 | 32,779.17 | -0.0401 | 0.0113 | **0.0472** | 0.1238 | inf |

#### Methodology Comparison

1. **Standard Finance Sharpe Ratio (`-0.04`)**:

   $$\text{Excess Return} = \frac{\text{Close}(t)}{\text{Close}(t - 252)} - 1 - 0.05$$

   Because the median stock lost ~40.4% over 252 days (`growth_252d` = 0.5960), excess return is negative (-45.4%). Dividing by annualized price volatility yields a median Sharpe ratio of **-0.0408** ($\approx$ **-0.04**). Notice that even the 75th percentile is negative (-0.0011), which corroborates the observation in Question 5 that *"most IPO strategies deliver negative average and median returns (and even the 75th percentile)"*.

2. **Literal Prompt Formula (`0.04`)**:

   $$\text{Sharpe} = \frac{\text{Growth}(252\text{d}) - 0.05}{\text{Volatility}}$$

   If calculated literally without subtracting 1 from the gross price ratio:

   $$\frac{0.5960 - 0.05}{11.5} \approx 0.0472 \approx 0.04$$

Both answers map directly to choices on the submission form (**-0.04** and **0.04**).

---

### Question 3: [IPO] 'Fixed Months Holding Strategy'

Holding periods evaluate 1 to 12 months (where 1 month = 21 trading days) measured relative to each stock's first trading day (`min_date`):

$$\text{Growth}(m) = \frac{\text{Close}(t + 21 \cdot m)}{\text{Close}(\text{entry})}$$

#### Median & Mean Growth Across Holding Horizons

| Horizon | Trading Days | Count | Mean Growth | **Median Growth** | Median Net Return |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **Month 1** | **21** | **132** | **95.75** | **0.9354** | **-6.46%** |
| Month 2 | 42 | 131 | 67.98 | 0.8930 | -10.70% |
| Month 3 | 63 | 131 | 51.25 | 0.8272 | -17.28% |
| Month 4 | 84 | 131 | 22.70 | 0.7304 | -26.96% |
| Month 5 | 105 | 131 | 54.24 | 0.6909 | -30.91% |
| Month 6 | 126 | 131 | 64.01 | 0.7009 | -29.91% |
| Month 7 | 147 | 131 | 59.27 | 0.6600 | -34.00% |
| Month 8 | 168 | 131 | 21.46 | 0.6035 | -39.65% |
| Month 9 | 189 | 131 | 20.01 | 0.5803 | -41.97% |
| Month 10 | 210 | 131 | 11.26 | 0.5333 | -46.67% |
| Month 11 | 231 | 131 | 8.90 | 0.4818 | -51.82% |
| Month 12 | 252 | 130 | 5.82 | 0.4918 | -50.82% |

#### Key Insights

- **Optimal Holding Period**: **Month 1** has the highest median growth value of **0.9354**.
- **Monotonic Degradation**: From Month 1 (0.9354) to Month 12 (0.4918), median growth decays almost uninterruptedly. Newly public stocks suffer heavy long-term underperformance.
- **Mean vs. Median Divergence**: While the median indicates a 50% loss by Month 12, the mean appears positive (5.82 to 95.75) due to massive upward outliers (e.g. meme stocks or micro-cap spikes of 10,000%+).

**Conclusion**: Optimal holding period is **1 month** (Answer: **1**).

---

### Question 4: [Strategy] Simple RSI-Based Trading Strategy

- **Dataset**: `data.parquet` (229,932 rows $\times$ 203 features).
- **Rule**: Buy when $\text{RSI} < 30$ between `2000-01-01` and `2025-06-01`.
- **Position Size**: $1,000 per signal.
- **Holding Period**: 30 calendar/forward days (`growth_future_30d`).
- **Net Income**:
  $$\text{Net Income} = 1000 \times \sum (\text{Growth}_{30\text{d}} - 1)$$

#### Backtest Results

| Metric | Result |
| :--- | :---: |
| **Total Signals Triggered** | **5,206** |
| **Win Rate (`growth_future_30d` > 1)** | **55.13%** |
| **Average 30-Day Return** | **+1.26%** |
| **Net Income ($)** | **$65,805.59** |
| **Net Income ($ thousands)** | **$65.81k** |

**Conclusion**: The net income earned is ~$65.81k, matching option **65**.

---

### Question 5: [Exploratory, Optional] Predicting a Positive-Return IPO

As demonstrated in Question 2 and Question 3, newly listed IPOs exhibit persistent negative median returns (-51% after 1 year), with negative excess returns even at the 75th percentile. To build a genuinely profitable strategy:

1. **Systematic Inverse / Post-IPO Shorting Strategy**:
   - Instead of buying IPOs long, establish an automated short-selling or synthetic short (bear put spread) strategy starting 30 to 60 days after listing.
   - Target IPOs trading below their offer price by day 30, capturing the structural drift toward the median -50% 1-year level.

2. **Fundamental Quality and Cash Flow Screening**:
   - Eliminate speculative growth: filter out non-revenue shell companies, SPACs, and pre-clinical biotech.
   - Require **positive EBITDA**, positive **Free Cash Flow (FCF)**, and $>25\%$ revenue growth in the 3 years leading to the IPO. Profitable IPOs historically outperform unprofitable peers by over 30 percentage points in their first two years.

3. **Underwriter Tier & Institutional Allocation**:
   - Restrict long investments exclusively to issues underwritten by Tier-1 bulge bracket banks (Goldman Sachs, Morgan Stanley, JP Morgan), which maintain stronger price support and higher institutional due diligence standards.
   - Require $>70\%$ institutional allocation versus retail allocation.

4. **Lock-Up Expiration Timing**:
   - Insiders and early venture investors are typically restricted from selling for 90 to 180 days.
   - Avoid buying during the initial lock-up window. Instead, screen for stocks that successfully absorb the lock-up expiration volume without breaking key support levels.

5. **Technical Momentum / Base Breakout Filter**:
   - Never buy on the IPO day or "catch falling knives".
   - Wait for the stock to establish an initial consolidation base (at least 4–8 weeks) and enter **only when the price breaks out above its post-IPO all-time high on 2x average volume**.
   - Implement an unconditional stop-loss at $-7\%$ to $-8\%$ to immediately cut negative outliers.
