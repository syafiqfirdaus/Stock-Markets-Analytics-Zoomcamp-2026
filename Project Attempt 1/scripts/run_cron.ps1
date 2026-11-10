# ==============================================================================
# Daily Scheduled Task Script (Windows PowerShell)
# Register with Windows Task Scheduler to run Mon-Fri at 17:00 EST / 05:00 MYT:
#   schtasks /Create /SC WEEKLY /D MON,TUE,WED,THU,FRI /TN "SemiCointMLTrader" `
#     /TR "powershell.exe -ExecutionPolicy Bypass -File '...\Project Attempt 1\scripts\run_cron.ps1'" /ST 05:00
# ==============================================================================
$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir

Set-Location $ProjectRoot
$Timestamp = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
Write-Host "[$Timestamp] Starting daily incremental Semiconductor Cointegration + ML workflow..."

python .\run_pipeline.py --mode download --incremental

$EndTimestamp = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
Write-Host "[$EndTimestamp] Workflow finished. Latest trade rankings written to outputs\reports\latest_10d_trade_signals.csv"

