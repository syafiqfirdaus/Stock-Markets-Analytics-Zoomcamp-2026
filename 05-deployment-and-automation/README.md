# Module 5: Deployment, Automation, and Stock Research Formulation

This module focuses on formulating rigorous, testable stock research hypotheses suitable for a capstone project or automated investment pipeline—connecting an economic opportunity, a clear investment thesis, observable primary/alternative data sources, a concrete signal/model test, and realistic validation benchmarks and failure criteria.

The complete written proposals for Homework 5 are documented in [homework5_solutions.md](file:///c:/Users/muhds/.gemini/antigravity-ide/scratch/Stock-Markets-Analytics-Zoomcamp-2026/05-deployment-and-automation/homework5_solutions.md) and can also be printed from the CLI via [homework5_solution.py](file:///c:/Users/muhds/.gemini/antigravity-ide/scratch/Stock-Markets-Analytics-Zoomcamp-2026/05-deployment-and-automation/homework5_solution.py).

---

## Python Environment & Execution

Print all three structured research proposals from the repository root:

```powershell
python .\05-deployment-and-automation\homework5_solution.py
```

To print individual questions:

```powershell
python .\05-deployment-and-automation\homework5_solution.py --question 1
python .\05-deployment-and-automation\homework5_solution.py --question 2
python .\05-deployment-and-automation\homework5_solution.py --question 3
```

---

## Homework 5 Overview

Submission form: [https://courses.datatalks.club/sma-zoomcamp-2026/homework/hw05](https://courses.datatalks.club/sma-zoomcamp-2026/homework/hw05)

| Question | Research Idea Title | Market & Horizon | Economic Mechanism | Key Data Sources |
| :--- | :--- | :--- | :--- | :--- |
| **Q1** | **High-Frequency Export Volume & FX Margin Pass-Through in Bursa Malaysia Mid-Cap Plantation Equities** | Bursa Malaysia (`.KL`) mid-cap planters; **20–40 trading days** | Information diffusion lag in under-covered local mid-caps following 10-day physical cargo shipment estimates, `USD/MYR` shifts, and tiered monthly export duty changes | `yfinance` (`.KL` equities, `MYR=X`), MPOB monthly stock/export statistics, Royal Malaysian Customs export duty gazettes, Intertek/AmSpec cargo surveyor reports |
| **Q2** | **Pre-Earnings Steam Concurrent Player Velocity vs. Consensus Revenue in Listed Game Publishers** | US, European & Japanese pure-play gaming publishers; **15–30 trading days pre-earnings** | Divergence between continuous digital consumption telemetry (post-launch retention slopes & DLC player-hours) and static sell-side revenue estimates | `yfinance` (OHLCV & earnings dates), Steam Web API (`ISteamUserStats`), SteamDB / SteamCharts historical concurrent player series |
| **Q3** | **Supply-Chain Cointegration & Fundamental-Filtered Pairs Trading in US Semiconductor Equities** | NASDAQ/NYSE semiconductor value-chain pairs (`LRCX`/`AMAT`, `TSM`/`ASML`, `TXN`/`ADI`); **5–15 trading days** | Short-term liquidity/ETF-flow spread divergences between cointegrated supply-chain peers, filtered by monthly Taiwanese foundry revenue disclosures | `yfinance` daily adjusted prices, FRED macro series (`DGS10`, `T10Y2Y`), TWSE Monthly Revenue MOPS disclosures (`2330.TW`) |

---

## Detailed Proposals (Submission-Ready Free Text)

See [homework5_solutions.md](file:///c:/Users/muhds/.gemini/antigravity-ide/scratch/Stock-Markets-Analytics-Zoomcamp-2026/05-deployment-and-automation/homework5_solutions.md) for the full 6-point breakdown (`Market and scope`, `Investment thesis`, `Potential edge`, `Data and sources`, `Research approach`, `Validation and risks`) for each of the three questions.

