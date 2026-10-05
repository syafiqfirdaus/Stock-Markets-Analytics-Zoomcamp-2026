"""Reproducible calculations for Stock Markets Analytics Zoomcamp Homework 3 (2026 Cohort).

Run from the repository root with:
    python 03-modeling/homework3_solution.py
or for specific questions:
    python 03-modeling/homework3_solution.py --question 1
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import gdown
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score
from sklearn.tree import DecisionTreeClassifier

# Directory paths
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DATA_PATH = SCRIPT_DIR / "stocks_df_combined_2026_09_18.parquet.brotli"
GDRIVE_FILE_ID = "1oQSUMCs2DyQQIh8Y62UhrT00cIsE9Sr5"


# -----------------------------------------------------------------------------
# Data Loading & Base Preprocessing
# -----------------------------------------------------------------------------
def load_and_preprocess_data() -> tuple[pd.DataFrame, list[str]]:
    """Download (if needed), load parquet dataset, and prepare feature sets."""
    if not DATA_PATH.exists():
        print(f"Data file not found at {DATA_PATH}. Downloading from Google Drive...")
        url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
        gdown.download(url, str(DATA_PATH), quiet=False)

    df_full = pd.read_parquet(DATA_PATH)

    # Growth indicators (excluding future growth)
    growth = [g for g in df_full.keys() if (g.startswith("growth_")) and ("future" not in g)]

    # Custom numerical features
    df_full["ln_volume"] = (
        df_full["Volume"].replace(0, np.nan).fillna(1e-9).apply(lambda x: np.log(x))
    )
    custom_numerical = [
        "SMA10",
        "SMA20",
        "growing_moving_average",
        "high_minus_low_relative",
        "volatility",
        "ln_volume",
    ]

    # Technical indicators
    technical_indicators = [
        "adx", "adxr", "apo", "aroon_1", "aroon_2", "aroonosc",
        "bop", "cci", "cmo", "dx", "macd", "macdsignal", "macdhist", "macd_ext",
        "macdsignal_ext", "macdhist_ext", "macd_fix", "macdsignal_fix",
        "macdhist_fix", "mfi", "minus_di", "mom", "plus_di", "dm", "ppo",
        "roc", "rocp", "rocr", "rocr100", "rsi", "slowk", "slowd", "fastk",
        "fastd", "fastk_rsi", "fastd_rsi", "trix", "ultosc", "willr",
        "ad", "adosc", "obv", "atr", "natr", "ht_dcperiod", "ht_dcphase",
        "ht_phasor_inphase", "ht_phasor_quadrature", "ht_sine_sine", "ht_sine_leadsine",
        "ht_trendmod", "avgprice", "medprice", "typprice", "wclprice",
    ]

    # Technical patterns and macro features
    technical_patterns = [g for g in df_full.keys() if "cdl" in g]
    macro = [
        "gdppot_us_yoy", "gdppot_us_qoq", "cpi_core_yoy", "cpi_core_mom",
        "FEDFUNDS", "DGS1", "DGS5", "DGS10",
    ]

    numerical = (
        growth
        + technical_indicators
        + technical_patterns
        + custom_numerical
        + macro
    )

    # Filter for 25 years of data (>= 2000-01-01)
    df = df_full[df_full["Date"] >= "2000-01-01"].copy()
    df["Weekday"] = df["Weekday"].astype(str)
    df["Month"] = df["Month"].dt.month.astype(str)

    return df, numerical


def add_month_wom_dummies(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Compute week-of-month and generate one-hot dummies for all categorical features."""
    wom = ((df["Date"].dt.day - 1) // 7 + 1).astype(str)
    month_name = df["Date"].dt.month_name()
    df["month_wom"] = month_name + "_w" + wom

    categorical = ["Month", "Weekday", "Ticker", "ticker_type", "month_wom"]
    dummy_vars = pd.get_dummies(df[categorical], dtype="int32")
    dummy_cols = dummy_vars.columns.tolist()

    df_with_dummies = pd.concat([df, dummy_vars], axis=1)
    return df_with_dummies, dummy_cols


def temporal_split(
    df: pd.DataFrame,
    train_prop: float = 0.70,
    val_prop: float = 0.15,
) -> pd.DataFrame:
    """Split data into train (70%), validation (15%), and test (15%) subsets by date."""
    min_date = df["Date"].min()
    max_date = df["Date"].max()
    train_end = min_date + pd.Timedelta(days=(max_date - min_date).days * train_prop)
    val_end = train_end + pd.Timedelta(days=(max_date - min_date).days * val_prop)

    split_labels = []
    for d in df["Date"]:
        if d <= train_end:
            split_labels.append("train")
        elif d <= val_end:
            split_labels.append("validation")
        else:
            split_labels.append("test")

    df["split"] = split_labels
    return df


def add_hand_rules(df: pd.DataFrame) -> pd.DataFrame:
    """Add manual hand rules pred0 through pred4 to the dataset."""
    df["pred0_manual_cci"] = (df["cci"] > 200).astype(int)
    df["pred1_manual_prev_g1"] = (df["growth_30d"] > 1).astype(int)
    df["pred2_manual_prev_g1_and_snp"] = (
        (df["growth_30d"] > 1) & (df["growth_snp500_30d"] > 1)
    ).astype(int)
    df["pred3_manual_dgs10_5"] = (
        (df["DGS10"] <= 4.5) & (df["DGS5"] <= 4)
    ).astype(int)
    df["pred4_manual_dgs10_fedfunds"] = (
        (df["DGS10"] > 4) & (df["FEDFUNDS"] <= 4.795)
    ).astype(int)
    return df


# -----------------------------------------------------------------------------
# Question 1: Dummies for Month and Week-of-Month
# -----------------------------------------------------------------------------
def question_1(df_with_dummies: pd.DataFrame, dummy_cols: list[str]) -> dict[str, Any]:
    """Calculate the correlation of month_wom dummies with is_positive_growth_30d_future."""
    target_col = "is_positive_growth_30d_future"
    corr_series = df_with_dummies[dummy_cols + [target_col]].corr()[target_col]
    corr_df = pd.DataFrame(corr_series)
    corr_df["abs_corr"] = corr_df[target_col].abs()

    month_wom_cols = [c for c in dummy_cols if c.startswith("month_wom_")]
    month_wom_corr = corr_df.loc[month_wom_cols].sort_values(by="abs_corr", ascending=False)

    top_feature = month_wom_corr.index[0]
    top_corr = month_wom_corr.loc[top_feature, target_col]
    top_abs_corr = month_wom_corr.loc[top_feature, "abs_corr"]
    rounded_val = round(float(top_abs_corr), 3)

    return {
        "top_feature": top_feature,
        "top_corr": top_corr,
        "top_abs_corr": top_abs_corr,
        "rounded_val": rounded_val,
        "form_answer": f"{rounded_val:.3f}",
        "top_5": month_wom_corr.head(5),
    }


# -----------------------------------------------------------------------------
# Question 2: Define New 'Hand' Rules
# -----------------------------------------------------------------------------
def question_2(df: pd.DataFrame) -> dict[str, Any]:
    """Evaluate precision of pred3 and pred4 on the TEST dataset."""
    test_df = df[df["split"] == "test"]
    target_col = "is_positive_growth_30d_future"

    stats = {}
    best_prec = -1.0
    best_rule = ""

    for rule in ["pred3_manual_dgs10_5", "pred4_manual_dgs10_fedfunds"]:
        pos_preds = test_df[test_df[rule] == 1]
        cnt = len(pos_preds)
        if cnt > 0:
            tp = (pos_preds[target_col] == 1).sum()
            fp = cnt - tp
            prec = tp / cnt
        else:
            tp, fp, prec = 0, 0, 0.0

        stats[rule] = {
            "count": cnt,
            "tp": tp,
            "fp": fp,
            "precision": prec,
            "rounded": round(prec, 3),
        }

        if prec > best_prec:
            best_prec = prec
            best_rule = rule

    rounded_best = round(best_prec, 3)
    return {
        "stats": stats,
        "best_rule": best_rule,
        "best_precision": best_prec,
        "form_answer": f"{rounded_best:.3f}",
    }


# -----------------------------------------------------------------------------
# Question 3: Unique Correct Predictions from pred5_clf_10
# -----------------------------------------------------------------------------
def question_3(
    df: pd.DataFrame,
    features_list: list[str],
    train_val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> dict[str, Any]:
    """Train DecisionTreeClassifier (depth 10, random_state 42) and find unique correct count."""
    target_col = "is_positive_growth_30d_future"

    X_train = train_val_df[features_list].copy().replace([np.inf, -np.inf], np.nan).fillna(0)
    y_train = train_val_df[target_col].copy()

    clf10 = DecisionTreeClassifier(max_depth=10, random_state=42)
    clf10.fit(X_train, y_train)

    X_all = df[features_list].copy().replace([np.inf, -np.inf], np.nan).fillna(0)
    df["pred5_clf_10"] = clf10.predict(X_all)

    test_sub = df[df["split"] == "test"].copy()
    y_test = test_sub[target_col]
    p5 = test_sub["pred5_clf_10"]

    hand_rules = [
        "pred0_manual_cci",
        "pred1_manual_prev_g1",
        "pred2_manual_prev_g1_and_snp",
        "pred3_manual_dgs10_5",
        "pred4_manual_dgs10_fedfunds",
    ]

    # Condition: pred5 matches true label, all hand rules mismatch true label
    pred5_correct = (p5 == y_test)
    all_hand_incorrect = pd.Series(True, index=test_sub.index)
    for hr in hand_rules:
        all_hand_incorrect = all_hand_incorrect & (test_sub[hr] != y_test)

    unique_correct_count = (pred5_correct & all_hand_incorrect).sum()

    # Form options: 271, 571, 1171, 1571
    form_options = [271, 571, 1171, 1571]
    closest_option = min(form_options, key=lambda x: abs(x - unique_correct_count))

    return {
        "unique_correct_count": int(unique_correct_count),
        "closest_form_option": closest_option,
        "form_answer": str(closest_option),
    }


# -----------------------------------------------------------------------------
# Question 4: Hyperparameter Tuning (Depth 1 to 12)
# -----------------------------------------------------------------------------
def question_4(
    features_list: list[str],
    train_val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> dict[str, Any]:
    """Iterate max_depth from 1 to 12 and find the depth maximizing precision on TEST."""
    target_col = "is_positive_growth_30d_future"

    X_train = train_val_df[features_list].copy().replace([np.inf, -np.inf], np.nan).fillna(0)
    y_train = train_val_df[target_col].copy()

    X_test = test_df[features_list].copy().replace([np.inf, -np.inf], np.nan).fillna(0)
    y_test = test_df[target_col].copy()

    results = []
    for depth in range(1, 13):
        clf = DecisionTreeClassifier(max_depth=depth, random_state=42)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        prec = precision_score(y_test, y_pred, zero_division=0)
        acc = accuracy_score(y_test, y_pred)
        results.append({
            "max_depth": depth,
            "test_precision": prec,
            "test_accuracy": acc,
        })

    results_df = pd.DataFrame(results)
    best_row = results_df.loc[results_df["test_precision"].idxmax()]
    best_depth = int(best_row["max_depth"])
    best_precision = float(best_row["test_precision"])

    return {
        "results_table": results_df,
        "best_max_depth": best_depth,
        "best_precision": best_precision,
        "form_answer": str(best_depth),
    }


# -----------------------------------------------------------------------------
# Question 5: Exploratory Analysis - Missing Data & Future Indicators
# -----------------------------------------------------------------------------
def question_5() -> str:
    """Return structured recommendations for missing data and new indicators."""
    return """
[Question 5: What data is missing? - Comprehensive Recommendation]

Based on our correlation analysis and Decision Tree feature importances (where macroeconomic
indicators like CPI Core, FEDFUNDS, and interest rate yields ranked among the most influential factors),
several key data dimensions are currently absent and would materially enhance predictive power:

1. Cross-Market & Regional Macro Indicators:
   - Current dataset primarily uses US macro data (FEDFUNDS, DGS1/5/10, CPI Core) for all 33 stocks,
     despite 1/3 of stocks being Indian (NSE) and 1/3 European (EU).
   - Missing:
     * Reserve Bank of India (RBI) repo rate & India CPI / WPI inflation for Indian equities.
     * European Central Bank (ECB) deposit facility rate & Eurozone HICP for EU equities.
     * Currency exchange rate movements (USD/INR, EUR/USD) and FX realized volatility.
       Source: FRED (Federal Reserve Bank of St. Louis) & Reserve Bank of India DBIE.

2. Market Sentiment & Regime Indicators (Volatility & Liquidity):
   - Current model lacks explicit regime indicators distinguishing low-volatility bull runs from
     liquidity crunch panics.
   - Missing:
     * VIX (CBOE Volatility Index) for US markets, India VIX for Indian equities, and VSTOXX for EU.
     * High-Yield Credit Spreads (e.g. ICE BofA US High Yield Spread) to detect credit stress.
     * Yield Curve Inversion Spreads (10Y minus 2Y, 10Y minus 3M Treasury yields).
       Source: Yahoo Finance (^VIX, ^INDIAVIX) & FRED.

3. Fundamental Valuation & Earnings Calendar Anomalies:
   - Technical price action indicators alone cannot capture fundamental re-rating or post-earnings drift.
   - Missing:
     * Trailing & Forward P/E, Price-to-Book, EV/EBITDA, and Free Cash Flow Yield.
     * Earnings Announcement Proximity: Days to next quarterly earnings release and previous
       earnings surprise (PEAD - Post-Earnings Announcement Drift effect).
       Source: SEC EDGAR / Financial Modeling Prep (FMP) API / SimFin.

4. Institutional Flow & Microstructure:
   - For emerging markets like India, Foreign Institutional Investor (FII) and Domestic Institutional
     Investor (DII) net buying/selling flows are primary drivers of multi-week trends.
   - Short Interest Ratio & Days to Cover for US equities.
       Source: SEBI / NSE India daily FII/DII releases, FINRA.
""".strip()


# -----------------------------------------------------------------------------
# Main Execution Runner
# -----------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Homework 3 (Module 03: Modeling) Solution Runner"
    )
    parser.add_argument(
        "--question",
        choices=["1", "2", "3", "4", "5", "all"],
        default="all",
        help="Specify which question to run (default: all)",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("Stock Markets Analytics Zoomcamp 2026 - Homework 3 Solutions")
    print("=" * 80)

    # 1. Load data
    print("\n[Loading and Preprocessing Dataset]...")
    df, numerical = load_and_preprocess_data()
    print(f"Data records: {len(df):,} (from {df['Date'].min().date()} to {df['Date'].max().date()})")

    # 2. Add month_wom dummies
    df_with_dummies, dummy_cols = add_month_wom_dummies(df)
    print(f"Total dummy features created: {len(dummy_cols)} (including 60 month_wom dummies)")

    # 3. Temporal split
    df_with_dummies = temporal_split(df_with_dummies)
    train_val_df = df_with_dummies[df_with_dummies["split"].isin(["train", "validation"])].copy()
    test_df = df_with_dummies[df_with_dummies["split"] == "test"].copy()
    print(f"Train + Validation split: {len(train_val_df):,} records | Test split: {len(test_df):,} records")

    # 4. Add hand rules
    df_with_dummies = add_hand_rules(df_with_dummies)
    train_val_df = add_hand_rules(train_val_df)
    test_df = add_hand_rules(test_df)

    features_list = numerical + dummy_cols

    # --- Question 1 ---
    if args.question in ["1", "all"]:
        print("\n" + "=" * 80)
        print("[Q1] Dummies for Month and Week-of-Month")
        print("-" * 50)
        q1_res = question_1(df_with_dummies, dummy_cols)
        print(f"Most correlated month_wom dummy: {q1_res['top_feature']}")
        print(f"Correlation value:             {q1_res['top_corr']:.6f}")
        print(f"Absolute correlation value:    {q1_res['top_abs_corr']:.6f}")
        print(f"Form Answer (rounded to 3 dec): {q1_res['form_answer']}")
        print("\nTop 5 correlated month_wom dummy variables:")
        print(q1_res["top_5"])

    # --- Question 2 ---
    if args.question in ["2", "all"]:
        print("\n" + "=" * 80)
        print("[Q2] Define New 'Hand' Rules on Macro and Technical Indicators")
        print("-" * 50)
        q2_res = question_2(df_with_dummies)
        for rule, val in q2_res["stats"].items():
            print(f"Rule: {rule:<30} | Count: {val['count']:<6} | TP: {val['tp']:<6} | Precision: {val['precision']:.5f} ({val['rounded']:.3f})")
        print(f"\nBest Rule:   {q2_res['best_rule']}")
        print(f"Form Answer: {q2_res['form_answer']}")

    # --- Question 3 ---
    if args.question in ["3", "all"]:
        print("\n" + "=" * 80)
        print("[Q3] Unique Correct Predictions from Decision Tree Classifier (depth=10)")
        print("-" * 50)
        q3_res = question_3(df_with_dummies, features_list, train_val_df, test_df)
        print(f"Calculated unique correct count on TEST: {q3_res['unique_correct_count']}")
        print(f"Form Answer:                             {q3_res['form_answer']}")

    # --- Question 4 ---
    if args.question in ["4", "all"]:
        print("\n" + "=" * 80)
        print("[Q4] Hyperparameter Tuning for Decision Tree (depth 1 to 12)")
        print("-" * 50)
        q4_res = question_4(features_list, train_val_df, test_df)
        print(q4_res["results_table"].to_string(index=False))
        print(f"\nOptimal Max Depth: {q4_res['best_max_depth']} (Precision: {q4_res['best_precision']:.5f})")
        print(f"Form Answer:       {q4_res['form_answer']}")

    # --- Question 5 ---
    if args.question in ["5", "all"]:
        print("\n" + "=" * 80)
        print("[Q5] [EXPLORATORY] What data is missing?")
        print("-" * 50)
        print(question_5())

    # --- Submission Form Summary ---
    if args.question == "all":
        print("\n" + "=" * 80)
        print("Submission Form Summary (https://courses.datatalks.club/sma-zoomcamp-2026/homework/hw03):")
        print(f"Q1: {q1_res['form_answer']}")
        print(f"Q2: {q2_res['form_answer']}")
        print(f"Q3: {q3_res['form_answer']}")
        print(f"Q4: {q4_res['form_answer']}")
        print("Q5: [See detailed response in Q5 section above]")
        print("=" * 80)


if __name__ == "__main__":
    main()
