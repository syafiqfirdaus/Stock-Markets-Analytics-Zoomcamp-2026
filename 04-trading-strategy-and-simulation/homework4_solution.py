"""
Homework 4 Solution Script - Stock Markets Analytics Zoomcamp 2026
Module 4: Trading Strategy and Simulation

Reproduces answers for all questions in Homework 4:
  - Question 1: Feature reduction CAGR for pred10_rf_best_rule_55
  - Question 2: Probability threshold for second-best RF rule with avg trades/day < 5
  - Question 3: Redefined growth target and trades for pred5_clf_10
  - Question 4: Ideal trading strategy write-up
"""

import argparse
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

DATA_PATH = "03-modeling/stocks_df_combined_2026_09_18.parquet.brotli"


def load_base_data():
    df_full = pd.read_parquet(DATA_PATH)

    GROWTH = [g for g in df_full.keys() if (g.find('growth_') == 0) and (g.find('future') < 0)]
    OHLCV = ['Open', 'High', 'Low', 'Close_x', 'Volume']
    CATEGORICAL = ['Month', 'Weekday', 'Ticker', 'ticker_type']
    TO_PREDICT = [g for g in df_full.keys() if (g.find('future') >= 0)]
    TO_DROP = ['Year', 'Date', 'index_x', 'index_y', 'index', 'Quarter', 'Close_y', 'Dividends', 'Stock Splits'] + CATEGORICAL + OHLCV

    df_full['ln_volume'] = df_full.Volume.apply(lambda x: np.log(x + 1e-6))
    df_full['div_payout'] = (df_full.Dividends > 0).astype(int)
    df_full['stock_split'] = (df_full['Stock Splits'] > 0).astype(int)

    CUSTOM_NUMERICAL = ['SMA10', 'SMA20', 'growing_moving_average', 'high_minus_low_relative', 'volatility', 'ln_volume', 'div_payout', 'stock_split']
    TECHNICAL_INDICATORS = [
        'adx', 'adxr', 'apo', 'aroon_1', 'aroon_2', 'aroonosc',
        'bop', 'cci', 'cmo', 'dx', 'macd', 'macdsignal', 'macdhist', 'macd_ext',
        'macdsignal_ext', 'macdhist_ext', 'macd_fix', 'macdsignal_fix',
        'macdhist_fix', 'mfi', 'minus_di', 'mom', 'plus_di', 'dm', 'ppo',
        'roc', 'rocp', 'rocr', 'rocr100', 'rsi', 'slowk', 'slowd', 'fastk',
        'fastd', 'fastk_rsi', 'fastd_rsi', 'trix', 'ultosc', 'willr',
        'ad', 'adosc', 'obv', 'atr', 'natr', 'ht_dcperiod', 'ht_dcphase',
        'ht_phasor_inphase', 'ht_phasor_quadrature', 'ht_sine_sine', 'ht_sine_leadsine',
        'ht_trendmod', 'avgprice', 'medprice', 'typprice', 'wclprice'
    ]
    TECHNICAL_PATTERNS = [g for g in df_full.keys() if g.find('cdl') >= 0]
    MACRO = ['gdppot_us_yoy', 'gdppot_us_qoq', 'cpi_core_yoy', 'cpi_core_mom', 'FEDFUNDS', 'DGS1', 'DGS5', 'DGS10']
    NUMERICAL = GROWTH + TECHNICAL_INDICATORS + TECHNICAL_PATTERNS + CUSTOM_NUMERICAL + MACRO

    df = df_full[df_full.Date >= '2000-01-01'].copy()
    df['Month'] = pd.to_datetime(df['Month'], format='%m').dt.strftime('%B')
    df['Weekday'] = df['Weekday'].astype('string')
    df['wom'] = df.Date.apply(lambda d: (d.day - 1) // 7 + 1).astype(str)
    df['month_wom'] = df.Month + '_w' + df.wom
    del df['wom']

    return df, CATEGORICAL, NUMERICAL


def temporal_split(df_in, min_date, max_date, train_prop=0.7, val_prop=0.15, test_prop=0.15):
    train_end = min_date + pd.Timedelta(days=(max_date - min_date).days * train_prop)
    val_end = train_end + pd.Timedelta(days=(max_date - min_date).days * val_prop)
    split_labels = []
    for d in df_in['Date']:
        if d <= train_end:
            split_labels.append('train')
        elif d <= val_end:
            split_labels.append('validation')
        else:
            split_labels.append('test')
    df_in['split'] = split_labels
    return df_in


def clean_dataframe(df_in: pd.DataFrame):
    df_in.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_in.fillna(0, inplace=True)
    return df_in


def simulate_prediction(df, pred_col, pred_prefix=None):
    if pred_prefix is None:
        pred_prefix = pred_col.split('_')[0]

    gross_col = 'sim1_gross_rev_' + pred_prefix
    fees_col = 'sim1_fees_' + pred_prefix
    net_col = 'sim1_net_rev_' + pred_prefix

    df[gross_col] = df[pred_col] * 100 * (df['growth_future_30d'] - 1)
    df[fees_col] = -df[pred_col] * 100 * 0.002
    df[net_col] = df[gross_col] + df[fees_col]

    flt = (df.split == 'test') & (df[pred_col] == 1)
    sim1_count = len(df[flt])
    if sim1_count == 0:
        return {'count': 0, 'avg_daily': 0, 'q75_daily': 0, 'capital': 0, 'net_rev': 0, 'CAGR': np.nan}

    gross = df[flt][gross_col].sum()
    fees = df[flt][fees_col].sum()
    net = df[flt][net_col].sum()

    daily_counts = df[flt].groupby('Date')[pred_col].count()
    avg_daily = daily_counts.mean()
    q75_daily = daily_counts.quantile(0.75)
    capital = 100 * 30 * q75_daily
    cagr = ((capital + net) / capital) ** (1 / 4)

    return {
        'count': sim1_count,
        'avg_daily': avg_daily,
        'q75_daily': q75_daily,
        'capital': capital,
        'gross_rev': gross,
        'fees': fees,
        'net_rev': net,
        'CAGR': cagr
    }


def run_question_1():
    print("\n" + "=" * 60)
    print("QUESTION 1: Less Features is More (CAGR for pred10_rf_best_rule_55)")
    print("=" * 60)

    df, CATEGORICAL, NUMERICAL = load_base_data()

    # Reduced CATEGORICAL (removing Ticker and month_wom)
    cat_for_dummies = [c for c in CATEGORICAL if c not in ['Ticker', 'ticker']]
    dummy_variables = pd.get_dummies(df[cat_for_dummies], dtype='int32')
    DUMMIES = dummy_variables.keys().to_list()
    features_list = NUMERICAL + DUMMIES

    print(f"Total features in new setup: {len(features_list)} (Numerical: {len(NUMERICAL)}, Dummies: {len(DUMMIES)})")

    df_with_dummies = pd.concat([df, dummy_variables], axis=1)
    df_with_dummies = temporal_split(df_with_dummies, min_date=df_with_dummies.Date.min(), max_date=df_with_dummies.Date.max())
    new_df = df_with_dummies.copy()

    to_predict = 'is_positive_growth_30d_future'
    train_valid_df = new_df[new_df.split.isin(['train', 'validation'])].copy(deep=True)
    X_train_valid = clean_dataframe(train_valid_df[features_list + [to_predict]])
    X_all = clean_dataframe(new_df[features_list + [to_predict]].copy(deep=True))
    y_train_valid = X_train_valid[to_predict]
    del X_train_valid[to_predict]
    del X_all[to_predict]

    print("Fitting RandomForestClassifier(n_estimators=100, max_depth=13, random_state=42)...")
    rf_best = RandomForestClassifier(n_estimators=100, max_depth=13, random_state=42, n_jobs=-1)
    rf_best.fit(X_train_valid, y_train_valid)

    y_pred_all = rf_best.predict_proba(X_all)
    y_pred_all_class1 = np.array([k[1] for k in y_pred_all])
    new_df['pred10_rf_best_rule_55'] = (y_pred_all_class1 >= 0.55).astype(int)

    res = simulate_prediction(new_df, 'pred10_rf_best_rule_55', pred_prefix='pred10')
    print(f"Results for pred10_rf_best_rule_55:")
    print(f"  Count Investments: {res['count']}")
    print(f"  Avg Investments / Day: {res['avg_daily']:.3f}")
    print(f"  Q75 Investments / Day: {res['q75_daily']}")
    print(f"  Capital Required: ${res['capital']:,.0f}")
    print(f"  Net Revenue: ${res['net_rev']:,.2f}")
    print(f"  CAGR: {res['CAGR']:.6f} (Closest Option: 1.126)")
    return res


def run_question_2():
    print("\n" + "=" * 60)
    print("QUESTION 2: Most Efficient Random Forest Decision Rule")
    print("=" * 60)

    df, CATEGORICAL, NUMERICAL = load_base_data()
    cat_for_dummies = [c for c in CATEGORICAL if c not in ['Ticker', 'ticker']]
    dummy_variables = pd.get_dummies(df[cat_for_dummies], dtype='int32')
    DUMMIES = dummy_variables.keys().to_list()
    features_list = NUMERICAL + DUMMIES

    df_with_dummies = pd.concat([df, dummy_variables], axis=1)
    df_with_dummies = temporal_split(df_with_dummies, min_date=df_with_dummies.Date.min(), max_date=df_with_dummies.Date.max())
    new_df = df_with_dummies.copy()

    to_predict = 'is_positive_growth_30d_future'
    train_valid_df = new_df[new_df.split.isin(['train', 'validation'])].copy(deep=True)
    X_train_valid = clean_dataframe(train_valid_df[features_list + [to_predict]])
    X_all = clean_dataframe(new_df[features_list + [to_predict]].copy(deep=True))
    y_train_valid = X_train_valid[to_predict]
    del X_train_valid[to_predict]
    del X_all[to_predict]

    print("Fitting RandomForestClassifier...")
    rf_best = RandomForestClassifier(n_estimators=100, max_depth=13, random_state=42, n_jobs=-1)
    rf_best.fit(X_train_valid, y_train_valid)

    y_pred_all = rf_best.predict_proba(X_all)
    y_pred_all_class1 = np.array([k[1] for k in y_pred_all])

    thresholds = [0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.60]
    print(f"\nEvaluating probability thresholds {thresholds}:")
    print(f"{'Threshold':<10} {'Trades':<8} {'Avg Daily':<12} {'Q75 Daily':<10} {'Capital':<10} {'CAGR':<10} {'Notes'}")
    print("-" * 75)

    best_subspace_thresh = None
    best_subspace_cagr = -1

    for t in thresholds:
        col = f'pred_rf_{int(round(t*100))}'
        new_df[col] = (y_pred_all_class1 >= t).astype(int)
        res = simulate_prediction(new_df, col, pred_prefix=f't{int(round(t*100))}')
        is_subspace = res['avg_daily'] < 5
        note = "Avg Daily < 5 [CANDIDATE]" if is_subspace else "Avg Daily >= 5"
        print(f"{t:<10.2f} {res['count']:<8} {res['avg_daily']:<12.2f} {res['q75_daily']:<10.1f} ${res['capital']:<9.0f} {res['CAGR']:<10.4f} {note}")

        if is_subspace and res['CAGR'] > best_subspace_cagr:
            best_subspace_cagr = res['CAGR']
            best_subspace_thresh = t

    print(f"\nWinning Decision Rule with Avg Daily < 5: Threshold = {best_subspace_thresh} (CAGR = {best_subspace_cagr:.4f})")
    return best_subspace_thresh


def run_question_3():
    print("\n" + "=" * 60)
    print("QUESTION 3: Redefining Growth Target (trades for pred5_clf_10)")
    print("=" * 60)

    df, CATEGORICAL, NUMERICAL = load_base_data()

    # Redefine target variable: > 1.022 (2.2% growth)
    df['is_positive_growth_30d_future'] = (df['growth_future_30d'] > 1.022).astype(int)
    pos_pct = df['is_positive_growth_30d_future'].mean() * 100
    print(f"Redefined Target: Positive growth > 2.2%. Class 1 frequency: {pos_pct:.2f}% (down from ~60%)")

    # Original full dummies setup as per fresh copy of Module 04
    cat_for_dummies = CATEGORICAL + ['month_wom']
    dummy_variables = pd.get_dummies(df[cat_for_dummies], dtype='int32')
    DUMMIES = dummy_variables.keys().to_list()
    features_list = NUMERICAL + DUMMIES

    df_with_dummies = pd.concat([df, dummy_variables], axis=1)
    df_with_dummies = temporal_split(df_with_dummies, min_date=df_with_dummies.Date.min(), max_date=df_with_dummies.Date.max())
    new_df = df_with_dummies.copy()

    to_predict = 'is_positive_growth_30d_future'
    train_valid_df = new_df[new_df.split.isin(['train', 'validation'])].copy(deep=True)
    X_train_valid = clean_dataframe(train_valid_df[features_list + [to_predict]])
    X_all = clean_dataframe(new_df[features_list + [to_predict]].copy(deep=True))
    y_train_valid = X_train_valid[to_predict]
    del X_train_valid[to_predict]
    del X_all[to_predict]

    print("Fitting DecisionTreeClassifier(max_depth=10, random_state=42)...")
    clf_10 = DecisionTreeClassifier(max_depth=10, random_state=42)
    clf_10.fit(X_train_valid, y_train_valid)

    new_df['pred5_clf_10'] = clf_10.predict(X_all)
    res = simulate_prediction(new_df, 'pred5_clf_10', pred_prefix='pred5')

    print(f"Results for pred5_clf_10:")
    print(f"  Count Investments on Test: {res['count']}")
    print(f"  Avg Investments / Day: {res['avg_daily']:.4f} -> Rounded: {round(res['avg_daily'])}")
    print(f"  Q75 Investments / Day: {res['q75_daily']}")
    print(f"  CAGR: {res['CAGR']:.4f}")
    return round(res['avg_daily'])


def run_question_4():
    print("\n" + "=" * 60)
    print("QUESTION 4: [EXPLORATORY] Ideal Trading Strategy")
    print("=" * 60)
    strategy = (
        "My ideal strategy uses a multi-model consensus filter with rank-based allocation and dynamic risk controls:\n"
        "1. Signal Selection: Combine the Random Forest model (threshold >= 0.57) with Decision Tree (clf_10) and macro trend confirmation (FEDFUNDS / Treasury yield slope). An asset is only eligible if both ML models predict positive growth.\n"
        "2. Turnover Reduction (Top-K Selection): On any trading day with eligible candidates, rank them by predicted probability and only execute the top 3 highest-conviction trades to reduce transaction costs and slippage.\n"
        "3. Position Sizing: Size positions inversely proportional to historical 30-day volatility (risk parity), capping any single ticker at 15% of portfolio capital.\n"
        "4. Risk Management & Exit Criteria: Implement an active 4% trailing stop-loss to truncate downside tails before the 30-day horizon, while taking partial profits when the asset reaches +8% gain or exceeds upper Bollinger band levels."
    )
    print(strategy)
    return strategy


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Homework 4 Solution Runner")
    parser.add_argument("--question", type=int, choices=[1, 2, 3, 4], help="Run a specific question (1, 2, 3, or 4)")
    args = parser.parse_args()

    if args.question == 1:
        run_question_1()
    elif args.question == 2:
        run_question_2()
    elif args.question == 3:
        run_question_3()
    elif args.question == 4:
        run_question_4()
    else:
        run_question_1()
        run_question_2()
        run_question_3()
        run_question_4()
