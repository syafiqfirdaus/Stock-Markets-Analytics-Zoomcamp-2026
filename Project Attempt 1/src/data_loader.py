"""
Data acquisition module for the US & Taiwan Semiconductor Supply-Chain Cointegration & ML Strategy.

Integrates three distinct data sources:
  1. Market OHLCV & Cross-Market Anchors (Yahoo Finance: 23 US Semi tickers, 4 TWSE Foundry/OSAT tickers, ETFs, FX, VIX)
  2. Macroeconomic Interest Rate & Credit Spread Series (FRED: DGS10, DGS2, T10Y2Y, FEDFUNDS, High-Yield OAS)
  3. Alternative Point-in-Time Fundamental Accounting Data (SEC EDGAR XBRL CompanyFacts REST API:
     Quarterly Revenue YoY/QoQ growth, R&D Intensity, Operating Margin, Inventory Turnover)

Supports:
  - Full download mode (`mode='download'`)
  - Local cached mode (`mode='local'`)
  - Incremental delta updates (`incremental=True`)
"""

import io
import time
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import requests
import yfinance as yf

from .config import (
    BENCHMARK_TICKERS,
    COMBINED_DATA_PATH,
    DATA_DIR,
    END_DATE,
    FRED_SERIES,
    SEC_CIK_MAP,
    START_DATE,
    SUB_VERTICAL_MAP,
    SUPPLY_CHAIN_PEER_MAP,
    TW_ANCHOR_TICKERS,
    US_SEMI_TICKERS,
)

SEC_CACHE_PATH = DATA_DIR / "sec_edgar_fundamentals.parquet"
RAW_OHLCV_CACHE_PATH = DATA_DIR / "raw_us_semi_ohlcv.parquet"
RAW_CONTEXT_CACHE_PATH = DATA_DIR / "raw_context_series.parquet"

SEC_HEADERS = {
    "User-Agent": "StockMarketsAnalyticsZoomcamp2026 QuantitativeResearch/1.0 (contact@datatalks.club)",
    "Accept-Encoding": "gzip, deflate",
}


def fetch_us_semi_ohlcv(
    tickers: List[str] = US_SEMI_TICKERS,
    start_date: str = START_DATE,
    end_date: Optional[str] = END_DATE,
) -> pd.DataFrame:
    """Download daily OHLCV, Dividends, and Stock Splits for US Semiconductor equities."""
    frames = []
    print(f"[DataLoader] Fetching daily OHLCV for {len(tickers)} US semiconductor tickers from {start_date}...")
    for ticker in tickers:
        try:
            t_obj = yf.Ticker(ticker)
            hist = t_obj.history(start=start_date, end=end_date, auto_adjust=True)
            if hist.empty:
                print(f"  [WARN] Empty history for {ticker}, skipping.")
                continue
            hist = hist.reset_index()
            hist["Date"] = pd.to_datetime(hist["Date"]).dt.tz_localize(None).dt.normalize()
            hist["Ticker"] = ticker
            hist["sub_vertical"] = SUB_VERTICAL_MAP.get(ticker, "Other")
            hist["peer_ticker"] = SUPPLY_CHAIN_PEER_MAP.get(ticker, "SOXX")
            for col in ["Dividends", "Stock Splits"]:
                if col not in hist.columns:
                    hist[col] = 0.0
            keep_cols = [
                "Date", "Ticker", "sub_vertical", "peer_ticker",
                "Open", "High", "Low", "Close", "Volume", "Dividends", "Stock Splits",
            ]
            frames.append(hist[keep_cols])
        except Exception as exc:
            print(f"  [WARN] Failed fetching {ticker}: {exc}")
    if not frames:
        raise RuntimeError("No US semiconductor OHLCV data could be fetched.")
    df_all = pd.concat(frames, ignore_index=True)
    df_all.sort_values(["Ticker", "Date"], inplace=True)
    df_all.drop_duplicates(subset=["Ticker", "Date"], keep="last", inplace=True)
    df_all.reset_index(drop=True, inplace=True)
    print(f"[DataLoader] Loaded {len(df_all):,} ticker-day records across {df_all['Ticker'].nunique()} tickers.")
    return df_all


def _fetch_single_yf_close(symbol: str, alias: str, start_date: str, end_date: Optional[str]) -> pd.DataFrame:
    """Helper to fetch Close & Volume for a benchmark or Taiwanese anchor ticker."""
    try:
        hist = yf.Ticker(symbol).history(start=start_date, end=end_date, auto_adjust=True)
        if hist.empty:
            return pd.DataFrame(columns=["Date", f"{alias}_close", f"{alias}_ret_1d", f"{alias}_ret_5d", f"{alias}_ret_10d", f"{alias}_ret_20d"])
        hist = hist.reset_index()
        hist["Date"] = pd.to_datetime(hist["Date"]).dt.tz_localize(None).dt.normalize()
        hist = hist.sort_values("Date").drop_duplicates("Date", keep="last")
        out = pd.DataFrame({"Date": hist["Date"]})
        out[f"{alias}_close"] = hist["Close"].astype(float)
        out[f"{alias}_ret_1d"] = out[f"{alias}_close"].pct_change(1)
        out[f"{alias}_ret_5d"] = out[f"{alias}_close"].pct_change(5)
        out[f"{alias}_ret_10d"] = out[f"{alias}_close"].pct_change(10)
        out[f"{alias}_ret_20d"] = out[f"{alias}_close"].pct_change(20)
        return out
    except Exception as exc:
        print(f"  [WARN] Could not fetch context symbol {symbol} ({alias}): {exc}")
        return pd.DataFrame(columns=["Date"])


def fetch_fred_series(series_map: Dict[str, str] = FRED_SERIES, start_date: str = START_DATE) -> pd.DataFrame:
    """
    Fetch daily macroeconomic series directly from FRED CSV endpoint (with pandas_datareader fallback).
    Using direct FRED CSV endpoint avoids pandas 3.0 compatibility issues and requires no API key.
    """
    print(f"[DataLoader] Fetching {len(series_map)} macroeconomic series from FRED...")
    merged = None
    for fred_id, col_name in series_map.items():
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={fred_id}&cosd={start_date}"
        try:
            resp = requests.get(url, timeout=20)
            resp.raise_for_status()
            df_s = pd.read_csv(io.StringIO(resp.text))
            date_col = df_s.columns[0]
            val_col = df_s.columns[1]
            df_s["Date"] = pd.to_datetime(df_s[date_col], errors="coerce").dt.normalize()
            df_s[col_name] = pd.to_numeric(df_s[val_col], errors="coerce")
            df_s = df_s[["Date", col_name]].dropna(subset=["Date"]).sort_values("Date")
            if merged is None:
                merged = df_s
            else:
                merged = pd.merge(merged, df_s, on="Date", how="outer")
        except Exception as exc:
            print(f"  [WARN] Failed to load FRED series {fred_id}: {exc}")
    if merged is None:
        return pd.DataFrame(columns=["Date"] + list(series_map.values()))
    merged.sort_values("Date", inplace=True)
    merged.ffill(inplace=True)
    return merged


def fetch_context_and_macro_data(start_date: str = START_DATE, end_date: Optional[str] = END_DATE) -> pd.DataFrame:
    """Fetch Taiwanese supply-chain anchors, sector ETFs, FX, VIX, and FRED macro series."""
    print("[DataLoader] Fetching Taiwanese supply-chain anchors (TWSE) and market benchmarks...")
    cal_end = pd.Timestamp.today().normalize() if end_date is None else pd.to_datetime(end_date).normalize()
    base_cal = pd.DataFrame({"Date": pd.date_range(start=start_date, end=cal_end, freq="B")})

    # Merge Taiwan supply-chain anchor stocks (2330.TW, 2303.TW, 2454.TW, 3711.TW)
    for sym, alias in TW_ANCHOR_TICKERS.items():
        df_tw = _fetch_single_yf_close(sym, alias, start_date, end_date)
        if not df_tw.empty and "Date" in df_tw.columns:
            base_cal = pd.merge(base_cal, df_tw, on="Date", how="left")

    # Merge Benchmarks (SOXX, SMH, SP500, VIX, USD/TWD, DXY)
    for sym, alias in BENCHMARK_TICKERS.items():
        df_bm = _fetch_single_yf_close(sym, alias, start_date, end_date)
        if not df_bm.empty and "Date" in df_bm.columns:
            base_cal = pd.merge(base_cal, df_bm, on="Date", how="left")

    # Merge FRED Macro series
    df_fred = fetch_fred_series(FRED_SERIES, start_date=start_date)
    if not df_fred.empty:
        base_cal = pd.merge(base_cal, df_fred, on="Date", how="left")

    base_cal.sort_values("Date", inplace=True)
    base_cal.ffill(inplace=True)
    base_cal.bfill(inplace=True)

    # Derived Macro & Cross-Market Regime Features
    if "dgs10" in base_cal.columns:
        base_cal["dgs10_diff_5d"] = base_cal["dgs10"].diff(5)
        base_cal["dgs10_diff_20d"] = base_cal["dgs10"].diff(20)
    if "t10y2y" in base_cal.columns:
        base_cal["yield_curve_inverted"] = (base_cal["t10y2y"] < 0).astype(int)
        base_cal["t10y2y_diff_10d"] = base_cal["t10y2y"].diff(10)
    if "hy_oas_spread" in base_cal.columns:
        base_cal["hy_oas_diff_10d"] = base_cal["hy_oas_spread"].diff(10)
    if "vix_close" in base_cal.columns:
        base_cal["vix_ma20_ratio"] = base_cal["vix_close"] / (base_cal["vix_close"].rolling(20).mean() + 1e-6)
        base_cal["vix_high_regime"] = (base_cal["vix_close"] > 22.0).astype(int)

    # Taiwanese Foundry Aggregate Supply-Chain Momentum Index
    tw_ret_cols = [f"{alias}_ret_10d" for alias in TW_ANCHOR_TICKERS.values() if f"{alias}_ret_10d" in base_cal.columns]
    if tw_ret_cols:
        base_cal["tw_supply_chain_index_10d"] = base_cal[tw_ret_cols].mean(axis=1)
        base_cal["tw_supply_chain_dispersion_10d"] = base_cal[tw_ret_cols].std(axis=1)

    return base_cal


def _extract_sec_concept_series(facts_json: dict, concept_candidates: List[str]) -> pd.DataFrame:
    """Extract quarterly USD accounting concept values keyed by public filing date (`filed`)."""
    taxonomy_dict = facts_json.get("facts", {})
    records = []
    for tax_name in ["us-gaap", "ifrs-full"]:
        tax = taxonomy_dict.get(tax_name, {})
        for concept in concept_candidates:
            if concept not in tax:
                continue
            units = tax[concept].get("units", {})
            for unit_key, items in units.items():
                if unit_key not in ("USD", "TWD", "EUR"):
                    continue
                for item in items:
                    form = item.get("form", "")
                    if form not in ("10-Q", "10-K", "20-F", "6-K"):
                        continue
                    filed = item.get("filed")
                    val = item.get("val")
                    start = item.get("start")
                    end = item.get("end")
                    if filed is None or val is None or end is None:
                        continue
                    # For income statement concepts with a start/end window, keep ~quarterly (~70-105 days) or annual
                    if start is not None:
                        days = (pd.to_datetime(end) - pd.to_datetime(start)).days
                        if days > 120 and form == "10-Q":
                            continue
                    records.append({"filed": pd.to_datetime(filed).normalize(), "end": pd.to_datetime(end).normalize(), "val": float(val)})
            if records:
                break
        if records:
            break
    if not records:
        return pd.DataFrame(columns=["filed", "val"])
    df = pd.DataFrame(records).sort_values(["filed", "end"]).drop_duplicates(subset=["filed"], keep="last")
    return df[["filed", "val"]]


def fetch_sec_edgar_fundamentals(
    cik_map: Dict[str, str] = SEC_CIK_MAP,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Fetch point-in-time quarterly accounting data from the SEC EDGAR XBRL API
    (https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json).

    Features engineered per ticker (strictly aligned by SEC `filed` date to prevent look-ahead bias):
      - `sec_rev_qoq`: Quarterly revenue sequential growth
      - `sec_rev_yoy`: Quarterly revenue year-over-year growth (4 filings lag)
      - `sec_rd_intensity`: R&D expense as % of Revenue
      - `sec_op_margin`: Operating income as % of Revenue
      - `sec_inv_to_rev`: Inventory to quarterly Revenue ratio (semiconductor inventory cycle indicator)
    """
    if use_cache and SEC_CACHE_PATH.exists():
        print(f"[DataLoader] Loading cached SEC EDGAR fundamentals from {SEC_CACHE_PATH.name}...")
        return pd.read_parquet(SEC_CACHE_PATH)

    print(f"[DataLoader] Fetching SEC EDGAR XBRL fundamentals for {len(cik_map)} semiconductor companies...")
    rev_concepts = [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenues",
        "SalesRevenueNet",
        "Revenue",
    ]
    rd_concepts = ["ResearchAndDevelopmentExpense"]
    op_concepts = ["OperatingIncomeLoss", "GrossProfit"]
    inv_concepts = ["InventoryNet", "Inventories"]

    all_rows = []
    for ticker, cik in cik_map.items():
        url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik.zfill(10)}.json"
        try:
            resp = requests.get(url, headers=SEC_HEADERS, timeout=15)
            if resp.status_code != 200:
                continue
            data = resp.json()
            df_rev = _extract_sec_concept_series(data, rev_concepts).rename(columns={"val": "rev"})
            if df_rev.empty:
                continue
            df_rd = _extract_sec_concept_series(data, rd_concepts).rename(columns={"val": "rd"})
            df_op = _extract_sec_concept_series(data, op_concepts).rename(columns={"val": "op_inc"})
            df_inv = _extract_sec_concept_series(data, inv_concepts).rename(columns={"val": "inv"})

            df_m = df_rev.sort_values("filed").copy()
            for sub_df in (df_rd, df_op, df_inv):
                if not sub_df.empty:
                    df_m = pd.merge_asof(
                        df_m.sort_values("filed"),
                        sub_df.sort_values("filed"),
                        on="filed",
                        direction="backward",
                    )
            df_m["Ticker"] = ticker
            df_m["sec_rev_qoq"] = df_m["rev"].pct_change(1).clip(-0.8, 2.0)
            df_m["sec_rev_yoy"] = df_m["rev"].pct_change(4).clip(-0.8, 3.0)
            df_m["sec_rd_intensity"] = (
                (df_m["rd"] / (df_m["rev"].abs() + 1e-6)).clip(0.0, 0.6)
                if "rd" in df_m.columns else np.nan
            )
            df_m["sec_op_margin"] = (
                (df_m["op_inc"] / (df_m["rev"].abs() + 1e-6)).clip(-0.5, 0.8)
                if "op_inc" in df_m.columns else np.nan
            )
            df_m["sec_inv_to_rev"] = (
                (df_m["inv"] / (df_m["rev"].abs() + 1e-6)).clip(0.0, 2.0)
                if "inv" in df_m.columns else np.nan
            )
            df_m["sec_inv_cycle_change"] = df_m["sec_inv_to_rev"].diff(1).clip(-0.5, 0.5)
            keep = [
                "Ticker", "filed",
                "sec_rev_qoq", "sec_rev_yoy", "sec_rd_intensity",
                "sec_op_margin", "sec_inv_to_rev", "sec_inv_cycle_change",
            ]
            all_rows.append(df_m[keep])
            time.sleep(0.12)  # Respect SEC EDGAR 10 req/sec fair access policy
        except Exception as exc:
            print(f"  [WARN] SEC EDGAR extraction warning for {ticker}: {exc}")

    if not all_rows:
        return pd.DataFrame(
            columns=[
                "Ticker", "filed", "sec_rev_qoq", "sec_rev_yoy",
                "sec_rd_intensity", "sec_op_margin", "sec_inv_to_rev", "sec_inv_cycle_change",
            ]
        )

    df_sec = pd.concat(all_rows, ignore_index=True).sort_values(["Ticker", "filed"])
    df_sec.to_parquet(SEC_CACHE_PATH, index=False)
    print(f"[DataLoader] Saved {len(df_sec):,} point-in-time SEC EDGAR filing records to {SEC_CACHE_PATH.name}.")
    return df_sec


def load_or_download_raw_data(
    mode: str = "local",
    incremental: bool = False,
    start_date: str = START_DATE,
    end_date: Optional[str] = END_DATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Orchestrates data loading across the two required execution regimes:
      - `mode='local'`: Reads from local parquet cache if available, otherwise downloads.
      - `mode='download'`: Downloads fresh data from Yahoo Finance, FRED, and SEC EDGAR.
      - `incremental=True`: Appends only newly available trading days onto existing local cache.
    """
    if mode == "local" and not incremental and RAW_OHLCV_CACHE_PATH.exists() and RAW_CONTEXT_CACHE_PATH.exists():
        print("[DataLoader] Regime='local': loading raw OHLCV, context/macro, and SEC EDGAR from local cache...")
        df_ohlcv = pd.read_parquet(RAW_OHLCV_CACHE_PATH)
        df_ctx = pd.read_parquet(RAW_CONTEXT_CACHE_PATH)
        df_sec = fetch_sec_edgar_fundamentals(use_cache=True)
        return df_ohlcv, df_ctx, df_sec

    if incremental and RAW_OHLCV_CACHE_PATH.exists() and RAW_CONTEXT_CACHE_PATH.exists():
        df_old = pd.read_parquet(RAW_OHLCV_CACHE_PATH)
        last_dt = pd.to_datetime(df_old["Date"]).max()
        delta_start = (last_dt - pd.Timedelta(days=45)).strftime("%Y-%m-%d")
        print(f"[DataLoader] Incremental update from {delta_start} (last cached date: {last_dt.strftime('%Y-%m-%d')})...")
        df_new = fetch_us_semi_ohlcv(start_date=delta_start, end_date=end_date)
        df_ohlcv = pd.concat([df_old, df_new], ignore_index=True)
        df_ohlcv.sort_values(["Ticker", "Date"], inplace=True)
        df_ohlcv.drop_duplicates(subset=["Ticker", "Date"], keep="last", inplace=True)
        df_ohlcv.reset_index(drop=True, inplace=True)
        df_ctx = fetch_context_and_macro_data(start_date=start_date, end_date=end_date)
        df_sec = fetch_sec_edgar_fundamentals(use_cache=True)
    else:
        df_ohlcv = fetch_us_semi_ohlcv(start_date=start_date, end_date=end_date)
        df_ctx = fetch_context_and_macro_data(start_date=start_date, end_date=end_date)
        df_sec = fetch_sec_edgar_fundamentals(use_cache=True)

    df_ohlcv.to_parquet(RAW_OHLCV_CACHE_PATH, index=False)
    df_ctx.to_parquet(RAW_CONTEXT_CACHE_PATH, index=False)
    return df_ohlcv, df_ctx, df_sec

