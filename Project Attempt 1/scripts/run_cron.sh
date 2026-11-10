#!/usr/bin/env bash
# ==============================================================================
# Daily Cron Execution Script (Linux / macOS / Docker)
# Schedule example in crontab (runs Mon-Fri at 21:30 UTC after US market close):
#   30 21 * * 1-5 /app/scripts/run_cron.sh >> /var/log/semi_coint_cron.log 2>&1
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"
echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] Starting daily incremental Semiconductor Cointegration + ML workflow..."

python run_pipeline.py --mode download --incremental

echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] Daily workflow completed successfully. Latest signals saved to outputs/reports/latest_10d_trade_signals.csv"

