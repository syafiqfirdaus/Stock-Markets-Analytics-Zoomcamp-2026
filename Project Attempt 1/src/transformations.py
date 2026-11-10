"""
Feature engineering, supply-chain cointegration analytics, point-in-time SEC EDGAR merge,
and exploratory data analysis (EDA) transformations for the US & Taiwan Semiconductor strategy.
"""

from typing import Dict, List, Tuple
import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

from .config import (
    COMBINED_DATA_PATH,
    SUPPLY_CHAIN_PEER_MAP,
    TARGET_BINARY_COL,
    TARGET_EXCESS_BINARY_COL,
    TARGET_GROWTH_COL,
)


def _compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    ma_up = up.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()
    ma_down = down.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()
    rs = ma_up / (ma_down + 1e-8)
    return 100.0 - (100.0 / (1.0 + rs))


def _engineer_single_ticker_technicals(g: pd.DataFrame) -> pd.DataFrame:
    """Compute historical growth lags, future growth targets, custom numericals, and technical indicators."""
    g = g.sort_values("Date").copy()
    close = g["Close"].astype(float)
    high = g["High"].astype(float)
    low = g["Low"].astype(float)
    vol = g["Volume"].astype(float)

    # 1. Historical Growth Ratios (Lecture features)
    for d in [1, 3, 5, 10, 20, 30, 60, 120, 252]:
        g[f"growth_{d}d"] = close / close.shift(d)

    # 2. Future Targets (TO_PREDICT)
    g["growth_future_5d"] = close.shift(-5) / close
    g[TARGET_GROWTH_COL] = close.shift(-10) / close
    g["growth_future_20d"] = close.shift(-20) / close
    g[TARGET_BINARY_COL] = (g[TARGET_GROWTH_COL] > 1.0).astype(int)

    # Forward 10-day high/low excursion for realistic stop-loss and take-profit simulation
    fwd_high_10d = high.shift(-1).rolling(window=10, min_periods=1).max().shift(-9)
    fwd_low_10d = low.shift(-1).rolling(window=10, min_periods=1).min().shift(-9)
    g["future_max_gain_10d"] = (fwd_high_10d / close) - 1.0
    g["future_max_dd_10d"] = (fwd_low_10d / close) - 1.0

    # 3. Custom Numerical Features (Lecture + Extended)
    g["SMA10"] = close.rolling(10).mean()
    g["SMA20"] = close.rolling(20).mean()
    g["SMA50"] = close.rolling(50).mean()
    g["SMA200"] = close.rolling(200).mean()
    g["sma10_rel"] = close / (g["SMA10"] + 1e-8) - 1.0
    g["sma20_rel"] = close / (g["SMA20"] + 1e-8) - 1.0
    g["sma50_rel"] = close / (g["SMA50"] + 1e-8) - 1.0
    g["sma200_rel"] = close / (g["SMA200"] + 1e-8) - 1.0
    g["growing_moving_average"] = (g["SMA10"] > g["SMA20"]).astype(int)
    g["golden_cross_50_200"] = (g["SMA50"] > g["SMA200"]).astype(int)
    g["high_minus_low_relative"] = (high - low) / (close + 1e-8)

    ret_1d = close.pct_change(1)
    g["ret_1d"] = ret_1d
    g["volatility_20d"] = ret_1d.rolling(20).std() * np.sqrt(252)
    g["volatility_60d"] = ret_1d.rolling(60).std() * np.sqrt(252)
    g["ln_volume"] = np.log(vol.clip(lower=1.0))
    g["volume_rel_20d"] = vol / (vol.rolling(20).mean() + 1.0)
    g["div_payout"] = (g["Dividends"] > 0).astype(int)
    g["stock_split"] = (g["Stock Splits"] > 0).astype(int)

    roll_max_252 = close.rolling(252, min_periods=20).max()
    roll_min_252 = close.rolling(252, min_periods=20).min()
    g["dist_52w_high"] = (close / (roll_max_252 + 1e-8)) - 1.0
    g["dist_52w_low"] = (close / (roll_min_252 + 1e-8)) - 1.0

    # 4. Technical Indicators (RSI, MACD, Bollinger, Stochastic, CCI, Williams %R, ATR, ADX proxy, OBV)
    g["rsi_14"] = _compute_rsi(close, 14)
    g["rsi_7"] = _compute_rsi(close, 7)

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    macd_sig = macd.ewm(span=9, adjust=False).mean()
    g["ppo"] = (macd / (ema26 + 1e-8)) * 100.0
    g["ppo_signal"] = (macd_sig / (ema26 + 1e-8)) * 100.0
    g["ppo_hist"] = g["ppo"] - g["ppo_signal"]

    bb_std = close.rolling(20).std()
    bb_upper = g["SMA20"] + 2.0 * bb_std
    bb_lower = g["SMA20"] - 2.0 * bb_std
    g["bb_pband"] = (close - bb_lower) / ((bb_upper - bb_lower) + 1e-8)
    g["bb_wband"] = (bb_upper - bb_lower) / (g["SMA20"] + 1e-8)

    low14 = low.rolling(14).min()
    high14 = high.rolling(14).max()
    g["stoch_k"] = 100.0 * (close - low14) / ((high14 - low14) + 1e-8)
    g["stoch_d"] = g["stoch_k"].rolling(3).mean()
    g["willr_14"] = -100.0 * (high14 - close) / ((high14 - low14) + 1e-8)

    tp = (high + low + close) / 3.0
    tp_sma = tp.rolling(20).mean()
    tp_mad = (tp - tp_sma).abs().rolling(20).mean()
    g["cci_20"] = (tp - tp_sma) / (0.015 * tp_mad + 1e-8)

    tr = pd.concat(
        [
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    g["natr_14"] = (tr.rolling(14).mean() / (close + 1e-8)) * 100.0

    # Directional Movement / ADX
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    atr14 = tr.rolling(14).mean() + 1e-8
    plus_di = 100.0 * pd.Series(plus_dm, index=g.index).rolling(14).mean() / atr14
    minus_di = 100.0 * pd.Series(minus_dm, index=g.index).rolling(14).mean() / atr14
    dx = 100.0 * (plus_di - minus_di).abs() / ((plus_di + minus_di) + 1e-8)
    g["adx_14"] = dx.rolling(14).mean()
    g["di_spread_14"] = plus_di - minus_di

    # Money Flow / Volume Flow
    mf_mult = ((close - low) - (high - close)) / ((high - low) + 1e-8)
    mf_vol = mf_mult * vol
    g["cmf_20"] = mf_vol.rolling(20).sum() / (vol.rolling(20).sum() + 1.0)
    g["roc_10"] = close.pct_change(10) * 100.0

    return g


def _engineer_supply_chain_cointegration_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute rolling supply-chain pair cointegration features for each ticker against its
    designated sub-vertical peer (`peer_ticker` in `SUPPLY_CHAIN_PEER_MAP`):
      1. `pair_ols_beta_60d`: Rolling 60-day log-price hedge ratio beta_t
      2. `pair_log_spread`: Cointegrated residual log spread S_t = ln(P_A) - beta_t * ln(P_B)
      3. `pair_spread_zscore_20d`: 20-day standardized spread Z-score (tactical mean-reversion signal)
      4. `pair_spread_zscore_60d`: 60-day standardized spread Z-score (medium-term valuation gap)
      5. `pair_spread_change_5d`: 5-day Z-score change velocity
      6. `pair_adf_stat_proxy_60d`: Rolling 60-day Ornstein-Uhlenbeck / AR(1) stationarity t-statistic
      7. `pair_halflife_days`: Implied mean-reversion half-life in trading days
      8. `pair_return_divergence_10d`: 10-day return divergence vs supply-chain peer
      9. `pair_corr_60d`: 60-day rolling daily return correlation with supply-chain peer
      10. `subvertical_rel_strength_10d`: 10-day return relative to semiconductor sub-vertical peers
    """
    print("[Transformations] Computing supply-chain cointegration & pair spread Z-score features...")
    # Pivot log close and 1d returns by Ticker
    piv_close = df.pivot(index="Date", columns="Ticker", values="Close").sort_index().ffill()
    piv_log = np.log(piv_close.clip(lower=1e-4))
    piv_ret1 = piv_close.pct_change(1)
    piv_ret10 = piv_close.pct_change(10)

    coint_frames = []
    for ticker in piv_log.columns:
        peer = SUPPLY_CHAIN_PEER_MAP.get(ticker)
        if peer not in piv_log.columns:
            continue
        y = piv_log[ticker]
        x = piv_log[peer]

        # Rolling 60-day OLS hedge ratio: Cov(y, x) / Var(x)
        cov_60 = y.rolling(60, min_periods=30).cov(x)
        var_60 = x.rolling(60, min_periods=30).var()
        beta_60 = (cov_60 / (var_60 + 1e-8)).clip(0.1, 3.0).fillna(1.0)
        alpha_60 = y.rolling(60, min_periods=30).mean() - beta_60 * x.rolling(60, min_periods=30).mean()

        spread = y - (alpha_60 + beta_60 * x)
        s_mean_20 = spread.rolling(20, min_periods=10).mean()
        s_std_20 = spread.rolling(20, min_periods=10).std()
        z_20 = ((spread - s_mean_20) / (s_std_20 + 1e-6)).clip(-5.0, 5.0)

        s_mean_60 = spread.rolling(60, min_periods=20).mean()
        s_std_60 = spread.rolling(60, min_periods=20).std()
        z_60 = ((spread - s_mean_60) / (s_std_60 + 1e-6)).clip(-5.0, 5.0)

        # Ornstein-Uhlenbeck AR(1) / ADF stationarity proxy on spread: dS_t = phi * S_{t-1} + eps_t
        ds = spread.diff(1)
        s_lag = spread.shift(1)
        cov_ds_s = ds.rolling(60, min_periods=30).cov(s_lag)
        var_s = s_lag.rolling(60, min_periods=30).var()
        phi = (cov_ds_s / (var_s + 1e-8)).clip(-0.95, 0.05)
        # Standard error of phi in rolling window
        resid_var = (ds - phi * s_lag).rolling(60, min_periods=30).var()
        se_phi = np.sqrt((resid_var / (58.0 * var_s + 1e-8)).clip(lower=1e-10))
        adf_t_proxy = (phi / (se_phi + 1e-8)).clip(-8.0, 2.0)

        # Implied Half-life of mean reversion: tau = -ln(2) / ln(1 + phi)
        rho = (1.0 + phi).clip(0.05, 0.995)
        halflife = (-np.log(2.0) / np.log(rho)).clip(1.0, 120.0)

        corr_60 = piv_ret1[ticker].rolling(60, min_periods=20).corr(piv_ret1[peer]).fillna(0.5)
        ret_div_10 = (piv_ret10[ticker] - piv_ret10[peer]).clip(-1.0, 1.0)

        sub_df = pd.DataFrame(
            {
                "Date": piv_log.index,
                "Ticker": ticker,
                "pair_ols_beta_60d": beta_60.values,
                "pair_log_spread": spread.values,
                "pair_spread_zscore_20d": z_20.values,
                "pair_spread_zscore_60d": z_60.values,
                "pair_spread_change_5d": z_20.diff(5).values,
                "pair_adf_stat_proxy_60d": adf_t_proxy.values,
                "pair_halflife_days": halflife.values,
                "pair_return_divergence_10d": ret_div_10.values,
                "pair_corr_60d": corr_60.values,
            }
        )
        coint_frames.append(sub_df)

    df_coint = pd.concat(coint_frames, ignore_index=True)
    df = pd.merge(df, df_coint, on=["Date", "Ticker"], how="left")

    # Sub-vertical relative strength (10d return minus sub-vertical daily cross-sectional mean)
    sub_mean_10d = df.groupby(["Date", "sub_vertical"])["growth_10d"].transform("mean")
    df["subvertical_rel_strength_10d"] = df["growth_10d"] - sub_mean_10d
    return df


def run_pair_cointegration_adf_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run formal Augmented Dickey-Fuller (ADF) tests on the log-price ratios and rolling OLS spreads
    of each semiconductor supply-chain pair for Exploratory Data Analysis (EDA).
    """
    piv_close = df.pivot(index="Date", columns="Ticker", values="Close").sort_index().ffill()
    seen_pairs = set()
    rows = []
    for t_a, t_b in SUPPLY_CHAIN_PEER_MAP.items():
        pair_key = tuple(sorted([t_a, t_b]))
        if pair_key in seen_pairs or t_a not in piv_close.columns or t_b not in piv_close.columns:
            continue
        seen_pairs.add(pair_key)
        sub_s = df[(df["Ticker"] == t_a) & df["pair_log_spread"].notna()]["pair_log_spread"]
        if len(sub_s) < 252:
            continue
        adf_res = adfuller(sub_s.values, maxlag=10, autolag="AIC")
        corr_val = df[df["Ticker"] == t_a]["pair_corr_60d"].median()
        hl_val = df[df["Ticker"] == t_a]["pair_halflife_days"].median()
        rows.append(
            {
                "Pair": f"{t_a} / {t_b}",
                "Sub_Vertical": df[df["Ticker"] == t_a]["sub_vertical"].iloc[0],
                "ADF_Statistic": round(float(adf_res[0]), 4),
                "ADF_p_value": round(float(adf_res[1]), 6),
                "Is_Stationary_p05": bool(adf_res[1] < 0.05),
                "Median_60d_Return_Corr": round(float(corr_val), 4),
                "Median_HalfLife_Days": round(float(hl_val), 2),
                "Observations": int(len(sub_s)),
            }
        )
    return pd.DataFrame(rows).sort_values("ADF_Statistic").reset_index(drop=True)


def build_unified_dataset(
    df_ohlcv: pd.DataFrame,
    df_ctx: pd.DataFrame,
    df_sec: pd.DataFrame,
    save_parquet: bool = True,
) -> pd.DataFrame:
    """
    Merge all data sources (US Semi OHLCV, Taiwan TWSE anchors, FRED macro, SEC EDGAR fundamentals),
    engineer 25+ new domain features, binned regimes, and seasonal dummies, and output a unified table.
    """
    print("[Transformations] Engineering per-ticker technicals and historical/future growth rates...")
    ticker_groups = []
    for _, grp in df_ohlcv.groupby("Ticker", sort=False):
        ticker_groups.append(_engineer_single_ticker_technicals(grp))
    df = pd.concat(ticker_groups, ignore_index=True)

    # Compute Supply-Chain Cointegration Features
    df = _engineer_supply_chain_cointegration_features(df)

    # Merge Taiwanese Supply-Chain Anchors, Benchmarks, and FRED Macro Series
    print("[Transformations] Merging Taiwanese TWSE supply-chain anchors, benchmarks, and FRED macro series...")
    df["Date"] = pd.to_datetime(df["Date"]).astype("datetime64[ns]")
    df_ctx = df_ctx.copy()
    df_ctx["Date"] = pd.to_datetime(df_ctx["Date"]).astype("datetime64[ns]")
    df = pd.merge(df, df_ctx, on="Date", how="left").copy()

    # Future 10-day SOXX benchmark return to define excess-return binary target
    soxx_future = (
        df_ctx[["Date", "soxx_etf_close"]]
        .drop_duplicates("Date")
        .sort_values("Date")
        .copy()
    )
    soxx_future["soxx_growth_future_10d"] = soxx_future["soxx_etf_close"].shift(-10) / soxx_future["soxx_etf_close"]
    df = pd.merge(df, soxx_future[["Date", "soxx_growth_future_10d"]], on="Date", how="left").copy()
    df[TARGET_EXCESS_BINARY_COL] = (df[TARGET_GROWTH_COL] > df["soxx_growth_future_10d"]).astype(int)

    # Cross-market Taiwan vs US ADR / Foundry Features
    # 1 share of TSM ADR = 5 ordinary shares of 2330.TW converted via USD/TWD
    if {"tw_tsmc_close", "usd_twd_close"}.issubset(df.columns):
        tsm_close_map = df[df["Ticker"] == "TSM"].set_index("Date")["Close"]
        tsm_us_close_ref = df["Date"].map(tsm_close_map)
        implied_tw_usd = (df["tw_tsmc_close"] * 5.0) / (df["usd_twd_close"].replace(0, np.nan) + 1e-6)
        raw_adr_prem = (tsm_us_close_ref / (implied_tw_usd + 1e-6)) - 1.0
        df["tsmc_adr_premium_proxy"] = (
            (raw_adr_prem - raw_adr_prem.rolling(60, min_periods=10).mean())
            / (raw_adr_prem.rolling(60, min_periods=10).std() + 1e-6)
        ).clip(-4.0, 4.0)
    else:
        df["tsmc_adr_premium_proxy"] = 0.0

    df["tw_tsmc_lead_diff_5d"] = (df["tw_tsmc_ret_5d"] - (df["growth_5d"] - 1.0)).clip(-0.5, 0.5)
    df["soxx_rel_growth_10d"] = (df["growth_10d"] - 1.0) - df["soxx_etf_ret_10d"]

    # Rolling 60-day Beta to SOXX Semiconductor Index
    cov_soxx = df.groupby("Ticker")["ret_1d"].transform(
        lambda s: s.rolling(60, min_periods=20).cov(df.loc[s.index, "soxx_etf_ret_1d"])
    )
    var_soxx = df.groupby("Ticker")["soxx_etf_ret_1d"].transform(
        lambda s: s.rolling(60, min_periods=20).var()
    )
    df["soxx_beta_60d"] = (cov_soxx / (var_soxx + 1e-8)).clip(0.2, 2.5).fillna(1.0)
    df = df.copy()

    # Point-in-Time Merge of SEC EDGAR Fundamental Features (via `merge_asof` on `filed` <= `Date`)
    print("[Transformations] Merging point-in-time SEC EDGAR quarterly fundamentals...")
    if not df_sec.empty:
        df_sec_sorted = df_sec.copy()
        df_sec_sorted["filed"] = pd.to_datetime(df_sec_sorted["filed"]).astype("datetime64[ns]")
        df_sec_sorted.sort_values("filed", inplace=True)
        df_sec_sorted["filed_dt"] = df_sec_sorted["filed"]
        df["Date"] = pd.to_datetime(df["Date"]).astype("datetime64[ns]")
        df = df.sort_values("Date").copy()
        df = pd.merge_asof(
            df,
            df_sec_sorted,
            left_on="Date",
            right_on="filed",
            by="Ticker",
            direction="backward",
        ).copy()
        df["days_since_sec_filing"] = (df["Date"] - df["filed_dt"]).dt.days.clip(0, 180).fillna(45.0)
        df["earnings_blackout_window"] = (
            ((df["days_since_sec_filing"] >= 84) & (df["days_since_sec_filing"] <= 94))
            | (df["days_since_sec_filing"] <= 2)
        ).astype(int)
        df.drop(columns=["filed", "filed_dt"], errors="ignore", inplace=True)
    else:
        for c in ["sec_rev_qoq", "sec_rev_yoy", "sec_rd_intensity", "sec_op_margin", "sec_inv_to_rev", "sec_inv_cycle_change"]:
            df[c] = 0.0
        df["days_since_sec_filing"] = 45.0
        df["earnings_blackout_window"] = 0

    for c, default_val in [
        ("sec_rev_qoq", 0.02),
        ("sec_rev_yoy", 0.08),
        ("sec_rd_intensity", 0.14),
        ("sec_op_margin", 0.22),
        ("sec_inv_to_rev", 0.35),
        ("sec_inv_cycle_change", 0.0),
    ]:
        df[c] = df.groupby("Ticker")[c].ffill().fillna(default_val)

    # 5+ Engineered Nonlinear Transformations & Binned Categorical Regimes (Rubric Section 3)
    print("[Transformations] Creating manual interaction transformations, binned regime variables, and calendar dummies...")
    df["coint_reversion_pressure"] = (
        -df["pair_spread_zscore_20d"]
        * df["pair_corr_60d"].clip(0.0, 1.0)
        * (1.0 - 0.5 * df["earnings_blackout_window"])
    )
    df["fundamental_filtered_zscore"] = -df["pair_spread_zscore_20d"] * np.where(df["sec_rev_yoy"] >= -0.05, 1.0, -0.5)
    df["rate_adjusted_momentum"] = (df["growth_20d"] - 1.0) - 0.02 * df["dgs10_diff_20d"].fillna(0.0)
    df["tw_coint_synergy_score"] = (
        -0.5 * df["pair_spread_zscore_20d"]
        + 5.0 * df["tw_supply_chain_index_10d"].fillna(0.0)
        + 2.0 * df["sec_rev_yoy"].fillna(0.0)
    )

    # Binned Regime Variables (Converted to one-hot dummies for ML)
    df["zscore_regime_bin"] = pd.cut(
        df["pair_spread_zscore_20d"].fillna(0.0),
        bins=[-np.inf, -1.5, -0.5, 0.5, 1.5, np.inf],
        labels=["Deep_Underval", "Mild_Underval", "Equilibrium", "Mild_Overval", "Extended_Overval"],
    ).astype(str)

    df["vix_regime_bin"] = pd.cut(
        df["vix_close"].fillna(18.0),
        bins=[-np.inf, 15.0, 22.0, 30.0, np.inf],
        labels=["Low_Vol", "Normal_Vol", "Elevated_Vol", "Panic_Vol"],
    ).astype(str)

    df["adx_trend_bin"] = pd.cut(
        df["adx_14"].fillna(20.0),
        bins=[-np.inf, 18.0, 28.0, np.inf],
        labels=["Range_MeanRevert", "Moderate_Trend", "Strong_Trend"],
    ).astype(str)

    # Calendar / Seasonal Attributes (including Module 3 `month_wom`)
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.strftime("%B")
    df["Weekday"] = df["Date"].dt.day_name()
    wom = ((df["Date"].dt.day - 1) // 7 + 1).astype(str)
    df["month_wom"] = df["Month"] + "_w" + wom

    # Filter out warmup window prior to 2011-01-01 so 252-day lags are populated
    df = df[df["Date"] >= "2011-01-01"].sort_values(["Date", "Ticker"]).reset_index(drop=True)

    # Add Temporal Split Labels (70% Train, 15% Validation, 15% Test on historical rows with known 10d target,
    # plus 'live' for the latest 10 trading days where future target is still unfolding!)
    has_target = df[TARGET_GROWTH_COL].notna()
    min_dt = df.loc[has_target, "Date"].min()
    max_dt = df.loc[has_target, "Date"].max()
    total_days = (max_dt - min_dt).days
    train_end = min_dt + pd.Timedelta(days=int(total_days * 0.70))
    val_end = train_end + pd.Timedelta(days=int(total_days * 0.15))

    conditions = [
        ~has_target,
        df["Date"] <= train_end,
        df["Date"] <= val_end,
    ]
    choices = ["live", "train", "validation"]
    df["split"] = np.select(conditions, choices, default="test")

    if save_parquet:
        df.to_parquet(COMBINED_DATA_PATH, index=False)
        print(
            f"[Transformations] Saved unified dataset to {COMBINED_DATA_PATH.name} "
            f"({len(df):,} rows x {df.shape[1]} columns; train_end={train_end.date()}, val_end={val_end.date()})."
        )
    return df


def get_feature_sets(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    """
    Define feature groups (`TO_PREDICT`, `NUMERIC`, `CATEGORICAL`, `DUMMIES`, `ALL_FEATURES`)
    and append one-hot dummy variables to the dataframe.
    """
    to_predict = [
        "growth_future_5d",
        TARGET_GROWTH_COL,
        "growth_future_20d",
        TARGET_BINARY_COL,
        TARGET_EXCESS_BINARY_COL,
        "future_max_gain_10d",
        "future_max_dd_10d",
        "soxx_growth_future_10d",
    ]

    growth_features = [f"growth_{d}d" for d in [1, 3, 5, 10, 20, 30, 60, 120, 252]]

    custom_numerical = [
        "sma10_rel", "sma20_rel", "sma50_rel", "sma200_rel",
        "growing_moving_average", "golden_cross_50_200",
        "high_minus_low_relative", "volatility_20d", "volatility_60d",
        "ln_volume", "volume_rel_20d", "div_payout", "stock_split",
        "dist_52w_high", "dist_52w_low",
    ]

    technical_indicators = [
        "rsi_14", "rsi_7", "ppo", "ppo_signal", "ppo_hist",
        "bb_pband", "bb_wband", "stoch_k", "stoch_d", "willr_14",
        "cci_20", "natr_14", "adx_14", "di_spread_14", "cmf_20", "roc_10",
    ]

    # 28 New Supply-Chain Cointegration, Taiwan Anchor, SEC EDGAR & Nonlinear Interaction Features
    new_domain_features = [
        # Supply-chain pair cointegration (10)
        "pair_ols_beta_60d",
        "pair_log_spread",
        "pair_spread_zscore_20d",
        "pair_spread_zscore_60d",
        "pair_spread_change_5d",
        "pair_adf_stat_proxy_60d",
        "pair_halflife_days",
        "pair_return_divergence_10d",
        "pair_corr_60d",
        "subvertical_rel_strength_10d",
        # Taiwanese foundry / cross-market lead-lag (8)
        "tsmc_adr_premium_proxy",
        "tw_tsmc_lead_diff_5d",
        "tw_supply_chain_index_10d",
        "tw_supply_chain_dispersion_10d",
        "tw_tsmc_ret_10d",
        "tw_umc_ret_10d",
        "tw_mediatek_ret_10d",
        "tw_ase_ret_10d",
        "usd_twd_ret_10d",
        "soxx_rel_growth_10d",
        "soxx_beta_60d",
        # SEC EDGAR Point-in-time Fundamentals (8)
        "sec_rev_qoq",
        "sec_rev_yoy",
        "sec_rd_intensity",
        "sec_op_margin",
        "sec_inv_to_rev",
        "sec_inv_cycle_change",
        "days_since_sec_filing",
        "earnings_blackout_window",
        # Engineered Nonlinear Interactions (4)
        "coint_reversion_pressure",
        "fundamental_filtered_zscore",
        "rate_adjusted_momentum",
        "tw_coint_synergy_score",
    ]

    macro_benchmark_features = [
        "soxx_etf_ret_5d", "soxx_etf_ret_10d", "soxx_etf_ret_20d",
        "smh_etf_ret_10d", "sp500_ret_10d", "sp500_ret_20d",
        "vix_close", "vix_ret_5d", "vix_ma20_ratio", "vix_high_regime",
        "dxy_ret_10d",
        "dgs10", "dgs2", "t10y2y", "fedfunds", "hy_oas_spread",
        "dgs10_diff_5d", "dgs10_diff_20d", "yield_curve_inverted",
        "t10y2y_diff_10d", "hy_oas_diff_10d",
    ]

    numeric_cols = [
        c for c in (growth_features + custom_numerical + technical_indicators + new_domain_features + macro_benchmark_features)
        if c in df.columns
    ]

    # Prune high-cardinality noisy dummies per Homework 4 Q1 finding ("Less features is more"):
    # Keep informative regime bins, sub_vertical, Month, and Weekday as primary dummy variables
    categorical_cols = [
        "sub_vertical",
        "zscore_regime_bin",
        "vix_regime_bin",
        "adx_trend_bin",
        "Month",
        "Weekday",
    ]

    df_dummies = pd.get_dummies(df[categorical_cols], prefix=categorical_cols, dtype=int)
    dummy_cols = list(df_dummies.columns)

    df_out = pd.concat([df, df_dummies], axis=1)

    feature_sets = {
        "TO_PREDICT": [c for c in to_predict if c in df_out.columns],
        "GROWTH": [c for c in growth_features if c in df_out.columns],
        "CUSTOM_NUMERICAL": [c for c in custom_numerical if c in df_out.columns],
        "TECHNICAL_INDICATORS": [c for c in technical_indicators if c in df_out.columns],
        "NEW_DOMAIN_FEATURES": [c for c in new_domain_features if c in df_out.columns],
        "MACRO": [c for c in macro_benchmark_features if c in df_out.columns],
        "NUMERIC": numeric_cols,
        "CATEGORICAL": categorical_cols,
        "DUMMIES": dummy_cols,
        "ALL_FEATURES": numeric_cols + dummy_cols,
    }
    return df_out, feature_sets
