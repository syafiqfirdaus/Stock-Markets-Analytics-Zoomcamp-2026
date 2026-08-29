# Module 1: Introduction and Data Sources

This module explores financial data collection from Wikipedia, Yahoo Finance, and macroeconomic data providers. The quantitative Homework 1 results were calculated on 29 August 2026 with [homework1_solution.py](homework1_solution.py).

## Python environment

From the repository root, create and activate the Python 3.12 environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

The dependency set covers market and macro data (`yfinance`, `pandas-datareader`, `fredapi`, Alpha Vantage, and Finnhub), numerical/statistical analysis, web scraping, technical indicators, visualization, notebooks, and common analytical file formats. The `.venv` directory is intentionally excluded from Git.

Run the complete solution from the repository root:

```powershell
.\.venv\Scripts\python.exe .\01-intro-and-data-sources\homework1_solution.py
```

## Homework 1 answers

| Question | Submission answer | Supporting result |
| --- | --- | --- |
| 1. Highest number of S&P 500 additions since 2020 | **2025** | 18 current constituents were added in 2025, versus 16 in 2024 and 15 each in 2022 and 2023. |
| 2. Non-US indexes beating the S&P 500 YTD | **2** | Japan (27.75%) and Canada (14.06%) exceeded the S&P 500 (11.41%). |
| 3. Median S&P 500 correction drawdown | **8%** | The calculated median is 7.99%, which maps to the 8% choice. |
| 4. Median two-day AMZN return after a positive earnings surprise | **0.35%** | The unrounded median across 20 completed positive-surprise events maps to the 0.35% choice. |

### Q1 - S&P 500 additions

The current constituent table on [Wikipedia](https://en.wikipedia.org/wiki/List_of_S%26P_500_companies) produces these full-year counts:

| Year | Current constituents added |
| ---: | ---: |
| 2020 | 10 |
| 2021 | 10 |
| 2022 | 15 |
| 2023 | 15 |
| 2024 | 16 |
| 2025 | 18 |

Using 21 August 2006 as the fixed 20-year cutoff, **224** current constituents have a known addition date earlier than the cutoff. This is a count of today's constituents, not every historical addition, because that is what Wikipedia's current-members table contains.

### Q2 - global index YTD comparison

The question lists the S&P 500 plus ten non-US indexes. Returns use the first available 2026 close and the final close returned by the homework's exact `end="2026-08-21"` hint. Because Yahoo Finance treats `end` as exclusive, the final observation is 20 August 2026.

| Index | YTD return |
| --- | ---: |
| Nikkei 225 | 27.75% |
| S&P/TSX Composite | 14.06% |
| S&P 500 | 11.41% |
| FTSE 100 | 8.01% |
| DAX | 5.88% |
| Ibovespa | 4.60% |
| S&P/ASX 200 | 4.08% |
| IPC Mexico | 0.32% |
| Hang Seng | -2.43% |
| Shanghai Composite | -2.97% |
| Nifty 50 | -7.32% |

Currency-conversion effects are ignored as instructed.

### Q3 - significant S&P 500 corrections

A completed correction is measured between consecutive all-time-high closes; its trough must be at least 5% below the earlier high. There are 74 completed corrections in the Yahoo Finance history used by the script.

| Percentile | Drawdown | High-to-trough duration |
| --- | ---: | ---: |
| 25th | 6.23% | 22 days |
| Median | **7.99%** | 40.5 days |
| 75th | 14.02% | 86.25 days |

Duration is measured in calendar days, consistent with the example correction dates in the homework brief. An unfinished drawdown after the last all-time high is excluded because it does not yet have a subsequent all-time high.

### Q4 - Amazon earnings surprises

For every AMZN earnings date, the centered two-trading-day return is:

```text
return = Close(day after announcement) / Close(day before announcement) - 1
```

For completed events with a positive reported earnings surprise, the median is **0.35%**. Pearson correlation between surprise magnitude and the two-day return is **0.337** for positive surprises only and **0.222** across all completed surprises. This is a weak-to-moderate positive sample relationship, not evidence that the surprise alone causes or reliably predicts the return.

### Q5 - capstone project idea

**Project: Regime-Aware US–Malaysia Equity Ranking and Portfolio Dashboard**

I want to build an end-to-end system that ranks liquid US and Malaysian stocks by their expected risk-adjusted return over the next 20 trading days while separately estimating downside risk. A reproducible data pipeline will combine price, volume, momentum, volatility, market-regime, macroeconomic, currency, earnings, and valuation features. I will compare interpretable statistical baselines with tree-based machine-learning models, then convert the predictions into a constrained portfolio. Walk-forward testing will compare the strategy with buy-and-hold, equal-weight, and simple momentum benchmarks after transaction costs and will report CAGR, Sharpe ratio, maximum drawdown, turnover, and hit rate. The final dashboard will provide stock rankings, predicted return and downside probability, the current market regime, feature explanations, and portfolio exposure rather than an opaque buy/sell signal.

The minimum viable project will use a manageable universe such as the S&P 100 and FTSE Bursa Malaysia KLCI constituents, daily data, and price/macro features. News sentiment, automated reporting, and an LLM research assistant can be added only after the core pipeline and backtest are reliable.

### Q6 - additional metrics to investigate

Metrics and time series that directly support the project are:

- **Price, momentum, and realized risk:** adjusted OHLCV data from `yfinance` can produce 5/20/60-day returns, RSI, MACD, realized volatility, downside deviation, beta, and drawdown. These form the core stock-level predictors.
- **Market stress and regime:** VIX (`^VIX`), index trend, index realized volatility, and rolling correlations can distinguish bullish, bearish, and stressed regimes. These are available through `yfinance.download()` and can be calculated with rolling pandas operations.
- **Interest rates and the yield curve:** the US 10-year Treasury rate (`DGS10`), federal-funds rate (`DFF`), and 10-year minus 2-year spread (`T10Y2Y`) from [FRED](https://fred.stlouisfed.org/) represent discount-rate pressure and recession expectations. They can be retrieved with `pandas_datareader.data.DataReader(..., "fred")` or `fredapi`.
- **Currency and cross-market conditions:** USD/MYR (`MYR=X`), the US Dollar Index (`DX-Y.NYB`), and relative US–Malaysia index momentum can capture currency exposure and international capital-flow conditions. Yahoo Finance provides these daily series.
- **Breadth and liquidity:** the percentage of constituents above their 50/200-day moving averages, advance-decline ratio, turnover, and Amihud illiquidity can reveal whether a market move is broad and whether a predicted position is realistically tradable. These can be derived from constituent price and volume histories.
- **Earnings and fundamentals:** earnings-surprise magnitude, estimate revisions, valuation multiples, profitability, leverage, and earnings growth help determine whether price momentum is supported by changing business expectations. Yahoo Finance provides a starting point, while Alpha Vantage or Finnhub offer API-based alternatives.

For the first version, I would prioritize OHLCV, VIX, rates, the yield curve, and USD/MYR because they are accessible and consistently timestamped. Fundamentals and sentiment can be later extensions. Every feature will be lagged to the time it was actually public, while index membership will be handled carefully, to prevent look-ahead and survivorship bias.
