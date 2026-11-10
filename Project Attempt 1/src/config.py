"""
Configuration constants, universes, supply-chain pair definitions, and paths
for the US & Taiwan Semiconductor Supply-Chain Cointegration & ML Strategy.
"""

from pathlib import Path

# Base Paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
REPORTS_DIR = OUTPUT_DIR / "reports"
MODELS_DIR = OUTPUT_DIR / "models"

for _d in (DATA_DIR, OUTPUT_DIR, FIGURES_DIR, REPORTS_DIR, MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

COMBINED_DATA_PATH = DATA_DIR / "semiconductor_coint_ml_dataset.parquet"
PREDICTIONS_DATA_PATH = DATA_DIR / "semiconductor_predictions_test.parquet"
LATEST_SIGNALS_PATH = REPORTS_DIR / "latest_10d_trade_signals.csv"

# Date range for historical extraction
START_DATE = "2010-01-01"
END_DATE = None  # None fetches up to latest available market close

# Primary US-Listed Semiconductor Universe (23 liquid tickers across 4 sub-verticals)
US_SEMI_TICKERS = [
    # Wafer Fab Equipment (WFE) & Materials
    "LRCX", "AMAT", "KLAC", "ASML", "TER", "ENTG",
    # Foundry, AI Logic & Networking
    "TSM", "NVDA", "AMD", "AVGO", "MRVL", "QCOM", "INTC",
    # Memory & Storage
    "MU", "WDC",
    # Analog, Power & RF
    "TXN", "ADI", "NXPI", "ON", "MCHP", "MPWR", "SWKS", "QRVO",
]

# Sub-vertical mapping for categorical dummy features & subgroup analysis
SUB_VERTICAL_MAP = {
    "LRCX": "WFE_Equipment",
    "AMAT": "WFE_Equipment",
    "KLAC": "WFE_Equipment",
    "ASML": "WFE_Equipment",
    "TER": "WFE_Equipment",
    "ENTG": "WFE_Equipment",
    "TSM": "Foundry_Logic",
    "NVDA": "Foundry_Logic",
    "AMD": "Foundry_Logic",
    "AVGO": "Foundry_Logic",
    "MRVL": "Foundry_Logic",
    "QCOM": "Foundry_Logic",
    "INTC": "Foundry_Logic",
    "MU": "Memory_Storage",
    "WDC": "Memory_Storage",
    "TXN": "Analog_Power_RF",
    "ADI": "Analog_Power_RF",
    "NXPI": "Analog_Power_RF",
    "ON": "Analog_Power_RF",
    "MCHP": "Analog_Power_RF",
    "MPWR": "Analog_Power_RF",
    "SWKS": "Analog_Power_RF",
    "QRVO": "Analog_Power_RF",
}

# Economically Coupled Supply-Chain Cointegration Peer Map (Ticker -> Primary Peer)
SUPPLY_CHAIN_PEER_MAP = {
    # Etch & Deposition Wafer Fab Equipment peers
    "LRCX": "AMAT",
    "AMAT": "LRCX",
    # Lithography & Metrology/Inspection peers
    "KLAC": "ASML",
    "ASML": "TSM",
    # Test & Wafer Materials peers
    "TER": "ENTG",
    "ENTG": "TER",
    # Foundry & EUV Lithography partner
    "TSM": "ASML",
    # AI Compute & GPU Duopoly
    "NVDA": "AMD",
    "AMD": "NVDA",
    # Custom AI ASIC & High-Speed Networking Silicon
    "AVGO": "MRVL",
    "MRVL": "AVGO",
    # Client / Mobile SoC & Compute
    "QCOM": "INTC",
    "INTC": "QCOM",
    # Memory & Enterprise Storage
    "MU": "WDC",
    "WDC": "MU",
    # Core Industrial & Automotive Analog Duopoly
    "TXN": "ADI",
    "ADI": "TXN",
    # Automotive Power & SiC Microcontrollers
    "NXPI": "ON",
    "ON": "NXPI",
    # Embedded Microcontrollers & Monolithic Power
    "MCHP": "MPWR",
    "MPWR": "MCHP",
    # RF Front-End Filter Duopoly
    "SWKS": "QRVO",
    "QRVO": "SWKS",
}

# Taiwanese Supply-Chain Anchor Equities (TWSE) & FX for Lead-Lag / ADR Arbitrage
TW_ANCHOR_TICKERS = {
    "2330.TW": "tw_tsmc",      # TSMC Taipei Ordinary Shares (5 shares = 1 TSM ADR)
    "2303.TW": "tw_umc",       # United Microelectronics Corp (Wafer Foundry)
    "2454.TW": "tw_mediatek",  # MediaTek Inc (Fabless SoC Bellwether)
    "3711.TW": "tw_ase",       # ASE Technology Holding (OSAT Packaging/Testing)
}

# Benchmarks & Market Regime Tickers (Yahoo Finance)
BENCHMARK_TICKERS = {
    "SOXX": "soxx_etf",        # iShares Semiconductor ETF (Primary Sector Benchmark)
    "SMH": "smh_etf",          # VanEck Semiconductor ETF
    "^GSPC": "sp500",          # S&P 500 Index
    "^VIX": "vix",             # CBOE Volatility Index
    "TWD=X": "usd_twd",        # USD/TWD Spot Exchange Rate
    "DX-Y.NYB": "dxy",         # US Dollar Index
}

# Macroeconomic Series from FRED (Federal Reserve Economic Data)
FRED_SERIES = {
    "DGS10": "dgs10",                  # 10-Year US Treasury Constant Maturity Rate
    "DGS2": "dgs2",                    # 2-Year US Treasury Constant Maturity Rate
    "T10Y2Y": "t10y2y",                # 10-Year minus 2-Year Treasury Yield Spread
    "FEDFUNDS": "fedfunds",            # Effective Federal Funds Rate
    "BAMLH0A0HYM2": "hy_oas_spread",   # ICE BofA US High Yield Index Option-Adjusted Spread
}

# SEC EDGAR CIK Mapping for Point-in-Time Quarterly Fundamental Accounting Features
# (Alternative Free Data Source outside Yahoo Finance / FRED)
SEC_CIK_MAP = {
    "LRCX": "0000707549",
    "AMAT": "0000006951",
    "KLAC": "0000319201",
    "ASML": "0000937966",
    "TER":  "0000097210",
    "ENTG": "0001101302",
    "TSM":  "0001046179",
    "NVDA": "0001045810",
    "AMD":  "0000002488",
    "AVGO": "0001730168",
    "MRVL": "0001835632",
    "QCOM": "0000804328",
    "INTC": "0000050863",
    "MU":   "0000723125",
    "WDC":  "0000106040",
    "TXN":  "0000097476",
    "ADI":  "0000006281",
    "NXPI": "0001413447",
    "ON":   "0001097864",
    "MCHP": "0000827054",
    "MPWR": "0001280452",
    "SWKS": "0000004127",
    "QRVO": "0001604778",
}

# Prediction Horizon & Simulation Parameters
TARGET_HORIZON_DAYS = 10  # 10 trading days (2 calendar weeks, matching 5-15 day thesis)
TARGET_GROWTH_COL = "growth_future_10d"
TARGET_BINARY_COL = "is_positive_growth_10d_future"
TARGET_EXCESS_BINARY_COL = "is_outperform_soxx_10d_future"

# Trading Simulation Constants
POSITION_SIZE_USD = 1000.0        # Base bet per signal in vector simulation ($1,000)
TRANSACTION_COST_RATE = 0.0010    # 10 bps (0.10%) one-way (0.20% round-trip slippage + commission)
INITIAL_PORTFOLIO_CAPITAL = 50000.0
MAX_CONCURRENT_POSITIONS = 6      # Maximum simultaneous holdings in exact portfolio simulation
STOP_LOSS_PCT = -0.09             # -9% tail-risk stop loss (calibrated for semiconductor volatility)
TAKE_PROFIT_PCT = 0.16            # +16% take profit target on 10d swing horizon

