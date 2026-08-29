"""Reproducible calculations for Stock Markets Analytics Zoomcamp Homework 1.

Run from the repository root with:
    .venv/Scripts/python.exe 01-intro-and-data-sources/homework1_solution.py

The script intentionally uses unadjusted closing prices because the homework asks
for closing-price growth. Yahoo Finance's ``end`` argument is exclusive.
"""

from __future__ import annotations

from io import StringIO

import pandas as pd
import requests
import yfinance as yf


WIKIPEDIA_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0 Safari/537.36"
)

INDEXES = {
    "United States — S&P 500": "^GSPC",
    "China — Shanghai Composite": "000001.SS",
    "Hong Kong — Hang Seng": "^HSI",
    "Australia — S&P/ASX 200": "^AXJO",
    "India — Nifty 50": "^NSEI",
    "Canada — S&P/TSX Composite": "^GSPTSE",
    "Germany — DAX": "^GDAXI",
    "United Kingdom — FTSE 100": "^FTSE",
    "Japan — Nikkei 225": "^N225",
    "Mexico — IPC": "^MXX",
    "Brazil — Ibovespa": "^BVSP",
}


def get_close(data: pd.DataFrame, ticker: str | None = None) -> pd.Series:
    """Return a clean Close series from either form of yfinance output."""
    close = data["Close"]
    if isinstance(close, pd.DataFrame):
        if ticker is not None and ticker in close.columns:
            close = close[ticker]
        elif close.shape[1] == 1:
            close = close.iloc[:, 0]
        else:
            raise ValueError("Ticker is required for multi-ticker data")
    return close.dropna().astype(float)


def question_1() -> tuple[pd.Series, int, int]:
    response = requests.get(
        WIKIPEDIA_URL,
        headers={"User-Agent": USER_AGENT},
        timeout=30,
    )
    response.raise_for_status()
    companies = pd.read_html(StringIO(response.text), match="Date added")[0]
    companies = companies[["Symbol", "Security", "Date added"]].copy()
    companies["Date added"] = pd.to_datetime(companies["Date added"], errors="coerce")
    companies["Year added"] = companies["Date added"].dt.year.astype("Int64")

    # 2026 is in progress at the homework cutoff, so compare full years through 2025.
    counts = (
        companies.loc[companies["Year added"].between(2020, 2025), "Year added"]
        .value_counts()
        .sort_index()
    )
    winning_year = int(counts.idxmax())
    twenty_year_cutoff = pd.Timestamp("2026-08-21") - pd.DateOffset(years=20)
    older_than_20_years = int((companies["Date added"] < twenty_year_cutoff).sum())
    return counts, winning_year, older_than_20_years


def question_2() -> tuple[pd.Series, int]:
    # end is exclusive: this follows the homework's exact end_date='2026-08-21'
    # hint and therefore uses the latest available close on or before 2026-08-20.
    data = yf.download(
        list(INDEXES.values()),
        start="2026-01-01",
        end="2026-08-21",
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    closes = data["Close"]
    returns_by_ticker = {
        ticker: series.dropna().iloc[-1] / series.dropna().iloc[0] - 1
        for ticker, series in closes.items()
    }
    ticker_to_name = {ticker: name for name, ticker in INDEXES.items()}
    returns = pd.Series(returns_by_ticker).rename(index=ticker_to_name).sort_values(
        ascending=False
    )
    us_return = returns["United States — S&P 500"]
    better_count = int(
        returns.drop(index="United States — S&P 500").gt(us_return).sum()
    )
    return returns * 100, better_count


def question_3() -> tuple[pd.DataFrame, pd.DataFrame]:
    data = yf.download(
        "^GSPC",
        start="1950-01-01",
        end="2026-08-22",
        auto_adjust=False,
        progress=False,
    )
    close = get_close(data, "^GSPC")

    previous_running_high = close.cummax().shift(1)
    record_highs = close[close.gt(previous_running_high) | previous_running_high.isna()]

    corrections: list[dict[str, object]] = []
    for position in range(len(record_highs) - 1):
        high_date = record_highs.index[position]
        next_high_date = record_highs.index[position + 1]
        high = float(record_highs.iloc[position])
        interval = close.loc[high_date:next_high_date]
        low_date = interval.idxmin()
        low = float(interval.loc[low_date])
        drawdown = (high - low) / high * 100
        if drawdown >= 5:
            corrections.append(
                {
                    "High date": high_date.date(),
                    "Low date": low_date.date(),
                    "Drawdown (%)": drawdown,
                    "Duration (days)": (low_date - high_date).days,
                }
            )

    correction_data = pd.DataFrame(corrections)
    percentiles = correction_data[["Drawdown (%)", "Duration (days)"]].quantile(
        [0.25, 0.50, 0.75]
    )
    percentiles.index = ["25th percentile", "Median", "75th percentile"]
    return correction_data, percentiles


def question_4() -> tuple[pd.DataFrame, float, float, float]:
    amazon = yf.Ticker("AMZN")
    earnings = amazon.get_earnings_dates(limit=30)
    if earnings is None or earnings.empty:
        raise RuntimeError("Yahoo Finance returned no AMZN earnings dates")

    # Match the 25-event homework window: 2020-10-29 through the first future
    # earnings date visible at the 2026-08-21 cutoff.
    earnings = earnings.copy()
    earnings.index = earnings.index.tz_localize(None).normalize()
    earnings = earnings.loc[earnings.index >= pd.Timestamp("2020-10-29")].sort_index()
    earnings = earnings.iloc[:25]

    prices = yf.download(
        "AMZN",
        start="2020-10-20",
        end="2026-08-22",
        auto_adjust=False,
        progress=False,
    )
    close = get_close(prices, "AMZN")
    close.index = close.index.tz_localize(None).normalize()

    # At each earnings date (Day 2), measure Close(Day 3)/Close(Day 1)-1.
    centered_two_day_return = close.shift(-1).div(close.shift(1)).sub(1)
    earnings["2-day return"] = centered_two_day_return.reindex(earnings.index)

    surprise_column = next(
        column for column in earnings.columns if str(column).startswith("Surprise")
    )
    completed = earnings.dropna(subset=[surprise_column, "2-day return"])
    positive = completed.loc[completed[surprise_column] > 0].copy()

    median_return_pct = float(positive["2-day return"].median() * 100)
    positive_correlation = float(
        positive[[surprise_column, "2-day return"]].corr().iloc[0, 1]
    )
    all_correlation = float(
        completed[[surprise_column, "2-day return"]].corr().iloc[0, 1]
    )
    return positive, median_return_pct, positive_correlation, all_correlation


def main() -> None:
    counts, winning_year, older_than_20 = question_1()
    print("\nQ1 — additions by full year")
    print(counts.to_string())
    print(f"Answer: {winning_year}")
    print(f"Additional: {older_than_20} current members exceed 20 years")

    returns, better_count = question_2()
    print("\nQ2 — YTD returns (%)")
    print(returns.round(2).to_string())
    print(f"Answer: {better_count} of the 10 non-US indexes beat the S&P 500")

    corrections, percentiles = question_3()
    print("\nQ3 — correction percentiles")
    print(percentiles.round(2).to_string())
    print(f"Answer: {percentiles.loc['Median', 'Drawdown (%)']:.2f}%")
    print(f"Number of completed corrections: {len(corrections)}")

    positive, median_return, positive_corr, all_corr = question_4()
    print("\nQ4 — positive AMZN earnings surprises")
    print(f"Positive completed events: {len(positive)}")
    print(f"Answer: {median_return:.2f}%")
    print(f"Correlation (positive surprises only): {positive_corr:.3f}")
    print(f"Correlation (all completed surprises): {all_corr:.3f}")


if __name__ == "__main__":
    main()
