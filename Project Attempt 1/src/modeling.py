"""
Machine Learning & Statistical Modeling module for the US & Taiwan Semiconductor Strategy.

Covers all Rubric Section 4 requirements:
  - Custom hand-crafted decision rules (Supply-Chain Cointegration Z-Score, Taiwan Lead-Lag, RSI/VIX, Macro Yield)
  - Lecture ML models: DecisionTreeClassifier and RandomForestClassifier
  - Hyperparameter tuning on the temporal validation split
  - New ML/DL models: XGBoost (`XGBClassifier`), Penalized Regression (`Ridge` on continuous 10d return),
    and Deep Neural Network (`MLPClassifier` with StandardScaler)
  - Custom high-probability threshold decision rules & Cointegration + ML Consensus Hybrid signal
"""

from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score, f1_score, precision_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from .config import TARGET_BINARY_COL, TARGET_GROWTH_COL


def _prepare_clean_matrix(df: pd.DataFrame, feature_cols: List[str]) -> np.ndarray:
    X = df[feature_cols].copy()
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return X.to_numpy(dtype=np.float32)


def generate_manual_rules(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Generate 4 domain-specific hand-crafted heuristic rules."""
    # Rule 0: Supply-Chain Cointegration Z-Score Mean Reversion (Core Module 5 Q3 hypothesis)
    df["pred0_manual_coint_zscore"] = (
        (df["pair_spread_zscore_20d"] < -1.15)
        & (df["pair_adf_stat_proxy_60d"] < -1.60)
        & (df["earnings_blackout_window"] == 0)
    ).astype(int)

    # Rule 1: Taiwan Foundry Lead-Lag + Fundamental Revenue Filter
    df["pred1_manual_tw_lead_lag"] = (
        (df["tw_supply_chain_index_10d"] > 0.008)
        & (df["pair_spread_zscore_20d"] < -0.35)
        & (df["sec_rev_yoy"] > -0.02)
    ).astype(int)

    # Rule 2: Oversold Technical Pullback in Bullish Trend & Non-Panic VIX
    df["pred2_manual_rsi_vix_filter"] = (
        (df["rsi_14"] < 43.0)
        & (df["vix_close"] < 26.0)
        & (df["golden_cross_50_200"] == 1)
    ).astype(int)

    # Rule 3: Macro Treasury Yield Stability + Moving Average Expansion
    df["pred3_manual_macro_yield"] = (
        (df["dgs10_diff_20d"] <= 0.20)
        & (df["growing_moving_average"] == 1)
        & (df["pair_spread_zscore_20d"] < 0.25)
    ).astype(int)

    manual_cols = [
        "pred0_manual_coint_zscore",
        "pred1_manual_tw_lead_lag",
        "pred2_manual_rsi_vix_filter",
        "pred3_manual_macro_yield",
    ]
    return df, manual_cols


def tune_and_train_models(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = TARGET_BINARY_COL,
    reg_target_col: str = TARGET_GROWTH_COL,
) -> Tuple[pd.DataFrame, Dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """
    Perform hyperparameter tuning on Train -> Validation, then refit optimal models on
    Train + Validation and generate out-of-sample predictions on Test (and Live rows).
    """
    print("[Modeling] Generating manual heuristic trading rules...")
    df, manual_cols = generate_manual_rules(df)

    is_train = df["split"] == "train"
    is_val = df["split"] == "validation"
    is_train_val = is_train | is_val
    is_test = df["split"] == "test"

    X_train = _prepare_clean_matrix(df[is_train], feature_cols)
    y_train = df.loc[is_train, target_col].to_numpy(dtype=int)

    X_val = _prepare_clean_matrix(df[is_val], feature_cols)
    y_val = df.loc[is_val, target_col].to_numpy(dtype=int)

    X_train_val = _prepare_clean_matrix(df[is_train_val], feature_cols)
    y_train_val = df.loc[is_train_val, target_col].to_numpy(dtype=int)
    y_reg_train_val = df.loc[is_train_val, reg_target_col].fillna(1.0).to_numpy(dtype=np.float32)

    X_all = _prepare_clean_matrix(df, feature_cols)

    tuning_records = []

    # 1. Hyperparameter Tuning: DecisionTreeClassifier (Depths 2..10)
    print("[Modeling] Tuning DecisionTreeClassifier across max_depth in [2, 3, 4, 5, 6, 8, 10]...")
    best_dt_depth = 4
    best_dt_prec = -1.0
    for depth in [2, 3, 4, 5, 6, 8, 10]:
        dt = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=50, random_state=42)
        dt.fit(X_train, y_train)
        val_preds = dt.predict(X_val)
        val_probs = dt.predict_proba(X_val)[:, 1]
        prec = precision_score(y_val, val_preds, zero_division=0)
        acc = accuracy_score(y_val, val_preds)
        auc = roc_auc_score(y_val, val_probs)
        tuning_records.append(
            {
                "Model_Family": "DecisionTree",
                "Hyperparameters": f"max_depth={depth}, min_samples_leaf=50",
                "Val_Precision": round(float(prec), 4),
                "Val_Accuracy": round(float(acc), 4),
                "Val_ROC_AUC": round(float(auc), 4),
                "Positive_Calls_Val": int(val_preds.sum()),
            }
        )
        if prec > best_dt_prec and val_preds.sum() >= 200:
            best_dt_prec = prec
            best_dt_depth = depth

    # Refit best DecisionTree on Train + Validation
    dt_best = DecisionTreeClassifier(max_depth=best_dt_depth, min_samples_leaf=50, random_state=42)
    dt_best.fit(X_train_val, y_train_val)
    df["prob_dt"] = dt_best.predict_proba(X_all)[:, 1]
    df["pred4_clf_dt_best"] = (df["prob_dt"] >= 0.55).astype(int)

    # 2. Hyperparameter Tuning: RandomForestClassifier
    print("[Modeling] Tuning RandomForestClassifier across max_depth and min_samples_leaf...")
    best_rf_params = (6, 40)
    best_rf_score = -1.0
    for depth, min_leaf in [(4, 50), (6, 40), (8, 30), (10, 25), (12, 20)]:
        rf = RandomForestClassifier(
            n_estimators=150,
            max_depth=depth,
            min_samples_leaf=min_leaf,
            max_features="sqrt",
            n_jobs=-1,
            random_state=42,
        )
        rf.fit(X_train, y_train)
        val_probs = rf.predict_proba(X_val)[:, 1]
        val_preds = (val_probs >= 0.55).astype(int)
        prec = precision_score(y_val, val_preds, zero_division=0)
        acc = accuracy_score(y_val, (val_probs >= 0.50).astype(int))
        auc = roc_auc_score(y_val, val_probs)
        tuning_records.append(
            {
                "Model_Family": "RandomForest",
                "Hyperparameters": f"max_depth={depth}, min_samples_leaf={min_leaf}",
                "Val_Precision": round(float(prec), 4),
                "Val_Accuracy": round(float(acc), 4),
                "Val_ROC_AUC": round(float(auc), 4),
                "Positive_Calls_Val": int(val_preds.sum()),
            }
        )
        if prec > best_rf_score and val_preds.sum() >= 150:
            best_rf_score = prec
            best_rf_params = (depth, min_leaf)

    rf_best = RandomForestClassifier(
        n_estimators=200,
        max_depth=best_rf_params[0],
        min_samples_leaf=best_rf_params[1],
        max_features="sqrt",
        n_jobs=-1,
        random_state=42,
    )
    rf_best.fit(X_train_val, y_train_val)
    df["prob_rf"] = rf_best.predict_proba(X_all)[:, 1]
    df["pred5_rf_best"] = (df["prob_rf"] >= 0.54).astype(int)

    # 3. New Model 1: Gradient Boosted Decision Trees (XGBoost)
    print("[Modeling] Tuning & fitting XGBClassifier (Gradient Boosted Trees)...")
    best_xgb_params = (3, 0.03)
    best_xgb_score = -1.0
    for depth, lr in [(3, 0.03), (4, 0.03), (5, 0.05), (6, 0.05)]:
        xgb = XGBClassifier(
            n_estimators=200,
            max_depth=depth,
            learning_rate=lr,
            subsample=0.8,
            colsample_bytree=0.75,
            reg_lambda=5.0,
            min_child_weight=30,
            eval_metric="logloss",
            n_jobs=-1,
            random_state=42,
        )
        xgb.fit(X_train, y_train)
        val_probs = xgb.predict_proba(X_val)[:, 1]
        val_preds = (val_probs >= 0.56).astype(int)
        prec = precision_score(y_val, val_preds, zero_division=0)
        acc = accuracy_score(y_val, (val_probs >= 0.50).astype(int))
        auc = roc_auc_score(y_val, val_probs)
        tuning_records.append(
            {
                "Model_Family": "XGBoost",
                "Hyperparameters": f"max_depth={depth}, lr={lr}, reg_lambda=5.0",
                "Val_Precision": round(float(prec), 4),
                "Val_Accuracy": round(float(acc), 4),
                "Val_ROC_AUC": round(float(auc), 4),
                "Positive_Calls_Val": int(val_preds.sum()),
            }
        )
        if prec > best_xgb_score and val_preds.sum() >= 150:
            best_xgb_score = prec
            best_xgb_params = (depth, lr)

    xgb_best = XGBClassifier(
        n_estimators=220,
        max_depth=best_xgb_params[0],
        learning_rate=best_xgb_params[1],
        subsample=0.8,
        colsample_bytree=0.75,
        reg_lambda=5.0,
        min_child_weight=30,
        eval_metric="logloss",
        n_jobs=-1,
        random_state=42,
    )
    xgb_best.fit(X_train_val, y_train_val)
    df["prob_xgb"] = xgb_best.predict_proba(X_all)[:, 1]
    df["pred6_xgb"] = (df["prob_xgb"] >= 0.55).astype(int)

    # 4. New Model 2: Penalized Linear Regression (Ridge predicting continuous 10d growth)
    print("[Modeling] Fitting penalized Ridge Regression for continuous 10-day return prediction...")
    ridge_pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=250.0, random_state=42)),
        ]
    )
    ridge_pipe.fit(X_train_val, y_reg_train_val)
    df["pred_reg_growth_10d"] = ridge_pipe.predict(X_all).clip(0.80, 1.30)
    df["pred7_ridge_reg"] = (df["pred_reg_growth_10d"] >= 1.018).astype(int)

    # 5. New Model 3: Deep Neural Network (Multi-Layer Perceptron `MLPClassifier`)
    print("[Modeling] Fitting Deep Neural Network (MLPClassifier 2-layer architecture)...")
    mlp_pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "mlp",
                MLPClassifier(
                    hidden_layer_sizes=(64, 32),
                    activation="relu",
                    alpha=0.25,
                    learning_rate_init=0.001,
                    max_iter=40,
                    early_stopping=True,
                    n_iter_no_change=5,
                    random_state=42,
                ),
            ),
        ]
    )
    mlp_pipe.fit(X_train_val, y_train_val)
    df["prob_mlp"] = mlp_pipe.predict_proba(X_all)[:, 1]
    df["pred8_mlp_dnn"] = (df["prob_mlp"] >= 0.56).astype(int)

    # 6. Custom High-Probability Threshold Rules & Multi-Model Cointegration Hybrid Consensus
    df["prob_ensemble"] = (
        0.40 * df["prob_mlp"]
        + 0.30 * df["prob_xgb"]
        + 0.20 * df["prob_rf"]
        + 0.10 * df["prob_dt"]
    )

    df["pred9_rf_high_prob_58"] = (df["prob_rf"] >= 0.56).astype(int)
    df["pred10_xgb_high_prob_58"] = (df["prob_xgb"] >= 0.58).astype(int)
    df["pred11_ensemble_consensus"] = (
        (df["prob_ensemble"] >= 0.55)
        & (df["prob_mlp"] >= 0.55)
        & (df["prob_xgb"] >= 0.54)
    ).astype(int)

    # Best Domain Hybrid Strategy: Cointegration Spread Filter + ML Consensus + Earnings/Macro Filter
    df["pred12_coint_ml_hybrid"] = (
        (
            ((df["prob_ensemble"] >= 0.55) & (df["pair_spread_zscore_20d"] <= 0.10))
            | ((df["prob_mlp"] >= 0.57) & (df["pair_spread_zscore_20d"] <= -0.50))
            | ((df["pair_spread_zscore_20d"] <= -1.20) & (df["prob_ensemble"] >= 0.52) & (df["pair_adf_stat_proxy_60d"] <= -1.60))
        )
        & (df["earnings_blackout_window"] == 0)
        & (df["vix_close"] <= 32.0)
    ).astype(int)

    df_tuning = pd.DataFrame(tuning_records)

    # Feature Importance Table from Random Forest and XGBoost
    df_feat_imp = pd.DataFrame(
        {
            "Feature": feature_cols,
            "RF_Importance": rf_best.feature_importances_,
            "XGB_Importance": xgb_best.feature_importances_,
            "DT_Importance": dt_best.feature_importances_,
        }
    )
    df_feat_imp["Combined_Importance"] = 0.5 * df_feat_imp["RF_Importance"] + 0.5 * df_feat_imp["XGB_Importance"]
    df_feat_imp.sort_values("Combined_Importance", ascending=False, inplace=True)
    df_feat_imp.reset_index(drop=True, inplace=True)

    models_dict = {
        "dt_best": dt_best,
        "rf_best": rf_best,
        "xgb_best": xgb_best,
        "ridge_pipe": ridge_pipe,
        "mlp_pipe": mlp_pipe,
        "best_dt_depth": best_dt_depth,
        "best_rf_params": best_rf_params,
        "best_xgb_params": best_xgb_params,
    }
    return df, models_dict, df_tuning, df_feat_imp


def evaluate_predictions_on_test(df: pd.DataFrame, target_col: str = TARGET_BINARY_COL) -> pd.DataFrame:
    """Evaluate all manual and ML prediction columns on the out-of-sample TEST split."""
    pred_cols = [c for c in df.columns if c.startswith("pred") and not c.startswith("pred_reg_")]
    test_df = df[df["split"] == "test"].copy()
    y_true = test_df[target_col].to_numpy(dtype=int)
    growth_10d = test_df[TARGET_GROWTH_COL].to_numpy(dtype=float)
    soxx_10d = test_df["soxx_growth_future_10d"].to_numpy(dtype=float)

    rows = []
    for col in pred_cols:
        y_pred = test_df[col].to_numpy(dtype=int)
        pos_mask = y_pred == 1
        n_calls = int(pos_mask.sum())
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        prec = precision_score(y_true, y_pred, zero_division=0)
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        mean_ret_10d = float((growth_10d[pos_mask] - 1.0).mean()) if n_calls > 0 else 0.0
        median_ret_10d = float(np.median(growth_10d[pos_mask] - 1.0)) if n_calls > 0 else 0.0
        mean_alpha_vs_soxx = float((growth_10d[pos_mask] - soxx_10d[pos_mask]).mean()) if n_calls > 0 else 0.0
        rows.append(
            {
                "Prediction_Rule": col,
                "Test_Positive_Trades": n_calls,
                "True_Positives": tp,
                "False_Positives": fp,
                "Test_Precision": round(float(prec), 4),
                "Test_Accuracy": round(float(acc), 4),
                "Test_F1": round(float(f1), 4),
                "Mean_10d_Return_Pct": round(mean_ret_10d * 100.0, 3),
                "Median_10d_Return_Pct": round(median_ret_10d * 100.0, 3),
                "Mean_10d_Alpha_vs_SOXX_Pct": round(mean_alpha_vs_soxx * 100.0, 3),
            }
        )
    return pd.DataFrame(rows).sort_values("Test_Precision", ascending=False).reset_index(drop=True)

