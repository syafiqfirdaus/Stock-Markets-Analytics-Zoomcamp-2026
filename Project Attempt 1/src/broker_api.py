"""
Broker API Execution & Live Signal Generator Module (Rubric Section 6 & Section 7 Bonus).

Generates live trade recommendations for the latest available market date (where future 10-day
returns are still unfolding) and submits bracket orders (Market Entry + Stop-Loss + Take-Profit)
via the Alpaca Paper Trading REST API (`https://paper-api.alpaca.markets/v2/orders`) or logs them
to a local auditable paper-trading order ledger when run in dry-run mode.
"""

import json
import os
from datetime import datetime
from typing import Dict, List
import numpy as np
import pandas as pd
import requests

from .config import (
    LATEST_SIGNALS_PATH,
    MAX_CONCURRENT_POSITIONS,
    REPORTS_DIR,
    STOP_LOSS_PCT,
    TAKE_PROFIT_PCT,
)

BROKER_LEDGER_PATH = REPORTS_DIR / "paper_broker_orders.json"
ALPACA_PAPER_BASE_URL = "https://paper-api.alpaca.markets"


def generate_latest_live_signals(
    df: pd.DataFrame,
    portfolio_capital: float = 50000.0,
    top_k: int = MAX_CONCURRENT_POSITIONS,
) -> pd.DataFrame:
    """
    Extract the latest trading date from the dataset (`split == 'live'`) and rank all 23
    US semiconductor equities by their Cointegration + ML Ensemble conviction score.
    """
    latest_date = df["Date"].max()
    live_slice = df[df["Date"] == latest_date].copy()
    live_slice["conviction_score"] = (
        live_slice["prob_ensemble"]
        - 0.08 * live_slice["pair_spread_zscore_20d"].clip(-3.0, 3.0)
        + 0.10 * live_slice["tw_supply_chain_index_10d"].fillna(0.0)
    )
    live_slice.sort_values("conviction_score", ascending=False, inplace=True)

    # Compute recommended position size & stop-loss / take-profit price levels
    slot_base = portfolio_capital / float(top_k)
    actions = []
    allocs = []
    shares_list = []
    sl_prices = []
    tp_prices = []

    active_buys = 0
    for _, row in live_slice.iterrows():
        close_px = float(row["Close"])
        is_hybrid_buy = int(row.get("pred12_coint_ml_hybrid", 0)) == 1
        is_ens_buy = int(row.get("pred11_ensemble_consensus", 0)) == 1
        is_short_hedge = (float(row["pair_spread_zscore_20d"]) > 1.85) and (float(row["prob_ensemble"]) < 0.44)

        if (is_hybrid_buy or is_ens_buy) and active_buys < top_k:
            action = "BUY_LONG_10D"
            prob_mult = float(np.clip((float(row["prob_ensemble"]) - 0.50) / 0.12, 0.75, 1.25))
            alloc = round(min(portfolio_capital * 0.22, slot_base * prob_mult), 2)
            qty = max(int(alloc // close_px), 1)
            sl_px = round(close_px * (1.0 + STOP_LOSS_PCT), 2)
            tp_px = round(close_px * (1.0 + TAKE_PROFIT_PCT), 2)
            active_buys += 1
        elif is_short_hedge and active_buys < top_k:
            action = "SHORT_PAIR_HEDGE_10D"
            alloc = round(slot_base * 0.65, 2)
            qty = max(int(alloc // close_px), 1)
            sl_px = round(close_px * (1.0 - STOP_LOSS_PCT), 2)
            tp_px = round(close_px * (1.0 - TAKE_PROFIT_PCT), 2)
            active_buys += 1
        else:
            action = "HOLD_WAIT"
            alloc = 0.0
            qty = 0
            sl_px = round(close_px * (1.0 + STOP_LOSS_PCT), 2)
            tp_px = round(close_px * (1.0 + TAKE_PROFIT_PCT), 2)

        actions.append(action)
        allocs.append(alloc)
        shares_list.append(qty)
        sl_prices.append(sl_px)
        tp_prices.append(tp_px)

    live_slice["Recommended_Action"] = actions
    live_slice["Allocated_Capital_USD"] = allocs
    live_slice["Order_Shares"] = shares_list
    live_slice["Stop_Loss_Price"] = sl_prices
    live_slice["Take_Profit_Price"] = tp_prices

    out_cols = [
        "Date", "Ticker", "sub_vertical", "peer_ticker", "Close",
        "prob_rf", "prob_xgb", "prob_mlp", "prob_ensemble",
        "pred_reg_growth_10d", "pair_spread_zscore_20d", "pair_adf_stat_proxy_60d",
        "tw_supply_chain_index_10d", "sec_rev_yoy", "conviction_score",
        "Recommended_Action", "Allocated_Capital_USD", "Order_Shares",
        "Stop_Loss_Price", "Take_Profit_Price",
    ]
    df_signals = live_slice[[c for c in out_cols if c in live_slice.columns]].reset_index(drop=True)
    df_signals.to_csv(LATEST_SIGNALS_PATH, index=False)
    print(f"[BrokerAPI] Saved latest 10-day trade rankings ({latest_date.date()}) to {LATEST_SIGNALS_PATH.name}.")
    return df_signals


def submit_orders_to_broker(
    df_signals: pd.DataFrame,
    dry_run: bool = True,
) -> List[Dict]:
    """
    Submit actionable orders (`BUY_LONG_10D` / `SHORT_PAIR_HEDGE_10D`) to Alpaca Markets REST API
    (`POST /v2/orders` with `order_class='bracket'`), or record simulated paper bracket orders locally.
    """
    api_key = os.environ.get("ALPACA_API_KEY", "")
    api_secret = os.environ.get("ALPACA_SECRET_KEY", "")
    active_orders = df_signals[df_signals["Order_Shares"] > 0].copy()

    submitted_payloads = []
    for _, row in active_orders.iterrows():
        side = "buy" if row["Recommended_Action"] == "BUY_LONG_10D" else "sell"
        order_payload = {
            "timestamp_utc": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "signal_date": str(pd.to_datetime(row["Date"]).date()),
            "symbol": str(row["Ticker"]),
            "peer_hedge_reference": str(row["peer_ticker"]),
            "qty": int(row["Order_Shares"]),
            "side": side,
            "type": "market",
            "time_in_force": "gtc",
            "order_class": "bracket",
            "reference_close_price": round(float(row["Close"]), 2),
            "notional_usd": round(float(row["Allocated_Capital_USD"]), 2),
            "take_profit": {"limit_price": str(row["Take_Profit_Price"])},
            "stop_loss": {"stop_price": str(row["Stop_Loss_Price"])},
            "ml_ensemble_prob": round(float(row["prob_ensemble"]), 4),
            "pair_spread_zscore_20d": round(float(row["pair_spread_zscore_20d"]), 3),
        }

        if not dry_run and api_key and api_secret:
            try:
                resp = requests.post(
                    f"{ALPACA_PAPER_BASE_URL}/v2/orders",
                    headers={
                        "APCA-API-KEY-ID": api_key,
                        "APCA-API-SECRET-KEY": api_secret,
                        "Content-Type": "application/json",
                    },
                    json={
                        "symbol": order_payload["symbol"],
                        "qty": str(order_payload["qty"]),
                        "side": order_payload["side"],
                        "type": "market",
                        "time_in_force": "gtc",
                        "order_class": "bracket",
                        "take_profit": order_payload["take_profit"],
                        "stop_loss": order_payload["stop_loss"],
                    },
                    timeout=10,
                )
                order_payload["broker_status"] = f"ALPACA_HTTP_{resp.status_code}"
                order_payload["broker_response"] = resp.json() if resp.ok else resp.text
            except Exception as exc:
                order_payload["broker_status"] = f"ALPACA_ERROR: {exc}"
        else:
            order_payload["broker_status"] = "SIMULATED_BRACKET_ORDER_LOGGED"

        submitted_payloads.append(order_payload)

    with open(BROKER_LEDGER_PATH, "w", encoding="utf-8") as f:
        json.dump(submitted_payloads, f, indent=2)

    print(f"[BrokerAPI] Logged {len(submitted_payloads)} bracket order(s) to {BROKER_LEDGER_PATH.name}.")
    return submitted_payloads

