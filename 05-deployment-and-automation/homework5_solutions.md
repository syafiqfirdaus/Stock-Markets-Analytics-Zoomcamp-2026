# Module 5 Homework Solutions: Deployment, Automation & Stock Research Proposals (2026 Cohort)

## Summary of Research Ideas

| Question | Research Theme | Market & Horizon | Core Economic Mechanism | Primary Data Sources |
| :--- | :--- | :--- | :--- | :--- |
| **Q1** | **Local-Market Export & Margin Pass-Through** | Bursa Malaysia (`.KL`) Mid-Cap Plantation & Oleochemical Stocks (20–40 trading days) | Slow analyst revision and retail underreaction in under-covered local mid-caps to 10-day physical cargo shipment data, USD/MYR FX shifts, and tiered export duty changes | `yfinance` (`.KL` equities, `MYR=X`), MPOB monthly statistics, Royal Malaysian Customs export duty gazettes, Intertek/AmSpec cargo surveyor reports |
| **Q2** | **Alternative Consumer Telemetry in Gaming** | US, European & Japanese Listed Game Publishers (15–30 trading days pre-earnings) | Divergence between real-time Steam concurrent player hours / DLC retention velocity and static sell-side quarterly revenue consensus | `yfinance` (OHLCV & earnings calendar), Steam Web API (`ISteamUserStats`), SteamDB / SteamSpy concurrent user (CCU) and top-seller histories |
| **Q3** | **Supply-Chain Cointegration & Pairs Mean Reversion** | US-Listed Semiconductor Value-Chain Pairs on NASDAQ/NYSE (5–15 trading days) | Temporary ETF/options-driven price divergence between economically locked supply-chain partners (`LRCX` vs. `AMAT`, `TSM` vs. `ASML`) filtered by monthly Taiwanese revenue releases | `yfinance` daily adjusted prices & volume, FRED rates (`T10Y2Y`), TWSE monthly revenue disclosures (released by the 10th of each month) |

---

## Question 1: [Exploratory] Research Idea Number One

### Title: High-Frequency Export Volume and FX Margin Pass-Through in Under-Covered Bursa Malaysia Mid-Cap Plantation Equities

1. **Market and scope.**
   - **Universe:** Mid-cap upstream and integrated palm oil plantation companies listed on **Bursa Malaysia** (Kuala Lumpur Stock Exchange, `.KL` tickers such as Sarawak Oil Palms `5126.KL`, Hap Seng Plantations `5138.KL`, Ta Ann Holdings `5012.KL`, and United Plantations `2089.KL`), excluding mega-cap index heavyweights (`5285.KL` SD Guthrie, `1961.KL` IOI Corp) that are heavily tracked by global institutions.
   - **Investment horizon:** Medium-term swing horizon of **20 to 40 trading days** (approximately 1 to 2 months), timed around monthly physical shipment cycles and ahead of quarterly corporate financial disclosures.

2. **Investment thesis.**
   - **Expected outcome & mechanism:** Palm oil is priced globally in sensitivity to Crude Palm Oil (FCPO) futures and exported primarily in USD, while local Malaysian mid-cap producers incur operating costs (labor, domestic transport, milling) predominantly in Malaysian Ringgit (MYR) and pay a tiered monthly government export duty. When 10-day physical export volumes accelerate alongside a weakening MYR (higher `USD/MYR`) or a favorable shift in the monthly gazetted export tax bracket, the realized local-currency operating margins of pure-play mid-cap producers expand significantly. Because mid-cap plantation stocks have sparse sell-side coverage, their share prices adjust with a measurable lag relative to large-cap peers and FCPO futures, creating a predictable drift over the subsequent 20–40 trading days.

3. **Potential edge.**
   - **Why it is overlooked:** International quantitative funds focus on liquid US/European equities or trade front-month FCPO futures directly, while local retail investors on Bursa Malaysia primarily react after quarterly earnings reports are filed on Bursa Announcements.
   - **Information advantage:** By combining mid-month physical export shipment estimates, the Royal Malaysian Customs monthly reference price formula, and spot `USD/MYR` into a real-time "effective netback margin proxy" per metric ton, we can quantify revenue and margin expansion weeks before quarterly income statements are published.

4. **Data and sources.**
   - **Sources checked:**
     - **Yahoo Finance (`yfinance`):** Verified availability of daily OHLCV and dividend-adjusted prices for Bursa Malaysia plantation equities (`5126.KL`, `5138.KL`, `5012.KL`, `2089.KL`), the Bursa Malaysia Plantation Index, and spot FX (`MYR=X`).
     - **Malaysian Palm Oil Board (MPOB) Portal:** Verified public monthly disclosures of national crude palm oil production, closing inventory stocks, and export tonnage (published around the 10th of each month).
     - **Royal Malaysian Customs / MPOB Export Duty Gazette:** Verified the public tiered tax schedule (8% to 10% brackets above RM 2,250/MT reference price) released monthly.
   - **Sources still to verify:**
     - **Cargo Surveyor 10-Day Export Releases (Intertek Testing Services / AmSpec Agri Malaysia):** Need to verify whether clean, point-in-time historical archives of the 1st–10th, 1st–15th, and 1st–20th monthly shipment estimates can be systematically scraped from local financial news wires (e.g., *The Edge Malaysia* or *Bernama*) without look-ahead bias.
     - **Company-Specific Production Disclosures:** Need to verify the exact timestamp consistency of monthly fresh fruit bunch (FFB) production filings submitted by individual listed planters to Bursa Malaysia.

5. **Research approach.**
   - **Signal & model:** Construct a composite monthly/bi-weekly **Netback Margin & Volume Momentum Signal**:
     1. Compute the 20-day percentage change in MYR-denominated net CPO price (spot CPO price converted to MYR, net of the gazetted monthly export duty rate).
     2. Combine with the year-over-year and month-over-month inventory drawdown ratio from MPOB data (export growth minus production growth).
     3. Measure the "catch-up spread": the residual return of each mid-cap stock over the past 10 trading days regressed against the large-cap plantation index and MYR-CPO price changes.
   - **Decision rule:** Go long the top tercile of mid-cap planters exhibiting positive netback margin momentum (> +5% over 20 days) but negative or flat residual price reaction (underreaction), holding for 20–40 trading days or exiting early if the stock converges to its historical beta-implied fair value.

6. **Validation and risks.**
   - **Benchmark:** Equal-weighted buy-and-hold portfolio of the Bursa Malaysia Plantation Index constituents and a simple 20-day price momentum baseline on the same tickers.
   - **Rejection criteria:** Reject the thesis if the out-of-sample Information Coefficient (rank correlation between the netback margin signal and 30-day forward residual returns) is statistically indistinguishable from zero ($p > 0.05$) or if net CAGR fails to beat the sector benchmark after deducting realistic round-trip transaction costs of **0.45% to 0.60%** (brokerage, stamp duty, clearing fees, and bid-ask spread).
   - **Key limitations & risks:**
     - *Liquidity constraints:* Several mid-cap `.KL` plantation stocks trade under RM 1–2 million in daily volume; position sizes must be capped at < 5% of 20-day Average Daily Volume (ADV).
     - *Corporate hedging & forward sales:* Some producers lock in forward CPO sales prices 3–6 months ahead, which can mute their short-term sensitivity to spot netback margin spikes.
     - *Weather & regulatory shocks:* Sudden Indonesian export bans/levy changes or El Niño yield disruptions can override local historical relationships.

---

## Question 2: [Exploratory] Research Idea Number Two

### Title: Pre-Earnings Steam Concurrent Player Velocity vs. Consensus Revenue Expectations in Listed Video Game Publishers

1. **Market and scope.**
   - **Universe:** Publicly traded pure-play PC/console video game publishers across US, European, and Japanese exchanges whose revenues are meaningfully driven by digital PC releases, live-service expansions, and DLCs—specifically tickers such as Take-Two (`TTWO`), Electronic Arts (`EA`), CD Projekt (`CDR.WA`), Paradox Interactive (`PDX.ST`), Embracer (`EMBRAC-B.ST`), Ubisoft (`UBI.PA`), Capcom (`9697.T`), and Koei Tecmo (`3635.T`). Mega-conglomerates where gaming is a minor fraction of total revenue (e.g., `MSFT`, `SONY`, `TCEHY`) are excluded.
   - **Investment horizon:** Event-driven window of **15 to 30 trading days** leading up to each company's quarterly earnings announcement, extending through **2 trading days post-earnings** (to capture both pre-earnings positioning drift and the earnings-day surprise reaction).

2. **Investment thesis.**
   - **Expected outcome & mechanism:** For digital-first game publishers, high-margin digital sales and live-service microtransactions depend directly on active player engagement and new-release retention during the fiscal quarter. While sell-side equity analysts update their revenue models sporadically based on management guidance and initial launch-week press releases, daily player telemetry (concurrent active users, player retention decay curves, and top-seller chart residency) is observable continuously throughout the quarter. When a publisher's aggregate fiscal-quarter player-hours or post-patch concurrent user (CCU) recovery significantly exceeds its historical decay baseline (or conversely, when a flagship release suffers rapid 14-day player churn), quarterly revenue and forward guidance systematically beat (or miss) consensus expectations.

3. **Potential edge.**
   - **Why it is overlooked:** Mainstream market participants over-index on raw launch-day peak player counts (which are widely reported in gaming media and quickly priced in) but fail to systematically track **post-launch retention slopes (days 7 to 60)** and **catalog/DLC reactivation surges** across a publisher's entire portfolio of live-service titles.
   - **Information advantage:** Aggregating daily player-hour integrals across all AppIDs owned by a listed parent company and weighting them by game price tier and DLC release dates provides a bottom-up proxy for digital bookings before the fiscal quarter closes.

4. **Data and sources.**
   - **Sources checked:**
     - **Yahoo Finance (`yfinance`):** Verified daily OHLCV price series, market capitalization, and historical earnings announcement dates for US (`EA`, `TTWO`), European (`CDR.WA`, `PDX.ST`, `UBI.PA`), and Japanese (`9697.T`) gaming tickers.
     - **Steam Web API (`ISteamUserStats/GetNumberOfCurrentPlayers` & `ISteamApps/GetAppList`):** Verified free programmatic access to real-time concurrent player counts per Steam AppID.
     - **SteamDB / SteamCharts Public Historical Series:** Verified availability of historical daily peak and average concurrent player counts for individual PC titles.
   - **Sources still to verify:**
     - **Parent-Company-to-AppID Mapping Table:** Need to build and verify a clean point-in-time relational mapping linking every subsidiary studio and Steam AppID (including paid DLC AppIDs) to its publicly traded parent ticker, accounting for M&A acquisition dates to prevent look-ahead bias.
     - **Historical Steam Global Top Sellers Rank:** Need to verify whether daily historical snapshots of Steam's "Top Sellers by Revenue" chart can be retrieved cleanly via the Steam API or archive snapshots to weight high-ARPU (average revenue per user) titles against free-to-play titles.

5. **Research approach.**
   - **Signal & model:** For each publisher $i$ in fiscal quarter $q$ (measured 15 trading days prior to the earnings release date):
     1. Calculate **Portfolio Player-Hours Growth ($\Delta \text{PH}_{i,q}$)**: the percentage change in total integrated concurrent player-hours across all owned AppIDs during the completed fiscal quarter relative to the prior quarter and the same quarter last year, adjusted for base-game price.
     2. Calculate **Retention Quality Ratio**: for titles or expansions launched during the quarter, compute the ratio of Day-30 average CCU to Day-3 peak CCU (distinguishing durable monetization hits from hyped launches with high refund/churn rates).
     3. Standardize these metrics against the stock's 60-day pre-earnings price return to identify **Telemetry-Price Divergence** (e.g., $\Delta \text{PH}_{i,q}$ in the top quartile while the stock price has underperformed the sector over the past 60 days).
   - **Decision rule:** Enter a long position 15 trading days before earnings when Telemetry-Price Divergence exceeds $+1.0$ standard deviation (strong player engagement growth not yet reflected in pre-earnings stock drift), and exit 2 trading days after the earnings release.

6. **Validation and risks.**
   - **Benchmark:** Equal-weighted basket of the listed gaming universe held over the same 17-day event windows, plus a naive pre-earnings momentum strategy (buying stocks with the highest 30-day return prior to earnings).
   - **Rejection criteria:** Reject the hypothesis if the signal fails to predict positive Earnings Announcement Return (EAR) or pre-earnings drift with a hit rate > 55% across at least 60 historical publisher-quarter events, or if a single outlier event (such as *Cyberpunk 2077*'s launch) accounts for the majority of backtest alpha.
   - **Key limitations & risks:**
     - *Platform representativeness bias:* Steam only captures PC player activity; console-heavy franchises (e.g., Take-Two's console sports titles or Nintendo titles) have weaker correlation with Steam CCU, requiring the filter to weight publishers by PC revenue share.
     - *Monetization mismatch:* Concurrent player spikes driven by deep 80% discount sales or "free weekend" promotions generate far less revenue per player-hour than full-price launches or paid expansion packs.
     - *Cross-market execution & FX costs:* Trading across US, Polish (`WSE`), Swedish (`OMX`), and Japanese (`TSE`) exchanges introduces multi-currency conversion fees and non-overlapping market hours around earnings announcements.

---

## Question 3: [Exploratory] Research Idea Number Three

### Title: Supply-Chain Cointegration and Fundamental-Filtered Pairs Trading in US-Listed Semiconductor Equities

1. **Market and scope.**
   - **Universe:** Liquid US-listed semiconductor stocks and ADRs on NASDAQ and NYSE (`TSM`, `ASML`, `AMAT`, `LRCX`, `KLAC`, `NVDA`, `AMD`, `AVGO`, `MU`, `TXN`, `ADI`), grouped into economically tightly coupled sub-vertical pairs—specifically Wafer Fab Equipment peers (`LRCX` vs. `AMAT` / `KLAC`), Foundry-Lithography partners (`TSM` vs. `ASML`), and Analog/Industrial chip peers (`TXN` vs. `ADI`).
   - **Investment horizon:** Short-term tactical mean-reversion horizon of **5 to 15 trading days** per trade.

2. **Investment thesis.**
   - **Expected outcome & mechanism:** Companies within the same semiconductor sub-vertical share identical end-market wafer demand, hyperscaler capital expenditure cycles, and fab utilization rates. Over medium-to-long horizons, their cash flows and equity valuations are cointegrated. However, over 1-to-3-week windows, their relative stock prices frequently diverge due to non-fundamental flows—such as retail options gamma squeezes on single tickers, thematic ETF rebalancing (`SMH` vs. `SOXX` weight caps), or isolated analyst rating changes. When a pair's log-price spread diverges beyond 2 standard deviations *without* an underlying divergence in fundamental supply-chain run-rates, the spread reliably mean-reverts over the next 5–15 trading days as institutional relative-value desks arbitrage the valuation gap.

3. **Potential edge.**
   - **Why it is overlooked:** Purely statistical pairs-trading strategies suffer heavy drawdowns when a spread widening is actually caused by a permanent fundamental regime break (e.g., one company gaining structural node share while the other loses out). Conversely, fundamental semiconductor investors operate on multi-quarter horizons and ignore 10-day relative mispricings.
   - **Information advantage:** Conditioning a statistical cointegration z-score on **high-frequency Taiwan supply-chain revenue disclosures** (published monthly by the 10th of every calendar month by TSMC, UMC, ASE, and MediaTek) and avoiding earnings windows allows us to distinguish temporary liquidity-driven divergences from genuine fundamental structural breaks.

4. **Data and sources.**
   - **Sources checked:**
     - **Yahoo Finance (`yfinance`):** Verified daily adjusted close prices, intraday high/low, trading volume, and historical dividend/split adjustments from 2010–2026 for all target semiconductor tickers (`TSM`, `ASML`, `LRCX`, `AMAT`, `KLAC`, `TXN`, `ADI`) and sector ETFs (`SOXX`, `SMH`).
     - **FRED (`pandas_datareader`):** Verified daily 10-Year Treasury Yield (`DGS10`), 10Y-2Y Yield Curve (`T10Y2Y`), and CBOE Volatility Index (`VIXCLS`) to control for macro duration sensitivity across high-multiple vs. value semiconductor names.
   - **Sources still to verify:**
     - **Taiwan Stock Exchange (TWSE) Monthly Revenue Market Observation Post System (MOPS):** Need to verify programmatic scraping or API extraction of exact release dates and timestamps for monthly consolidated revenue reports of `2330.TW` (TSMC) and key Taiwanese foundry/OSAT suppliers to build a point-in-time fundamental regime filter.
     - **Point-in-Time Historical Earnings Calendar:** Need to verify exact after-hours vs. pre-market earnings release dates for each US ticker to strictly block new pair entries within 5 days before an earnings announcement.

5. **Research approach.**
   - **Signal & model:**
     1. **Pair Selection & Hedge Ratio:** On a rolling 120-trading-day training window, test pre-defined economic pairs (e.g., `LRCX`/`AMAT`, `TXN`/`ADI`, `TSM`/`ASML`) for cointegration using the Augmented Dickey-Fuller (ADF) test ($p < 0.05$) and estimate the dynamic hedge ratio $\beta_t$ via a Kalman Filter (or rolling 60-day OLS on log prices).
     2. **Spread Z-Score:** Compute the standardized spread $Z_t = \frac{S_t - \mu_{S,20}}{\sigma_{S,20}}$, where $S_t = \ln(P_{A,t}) - \beta_t \ln(P_{B,t})$.
     3. **Fundamental Regime Filter:** Disable mean-reversion entries if either leg has an earnings release within the next 5 trading days, or if monthly TWSE foundry revenue growth shows an extreme structural shock (> 2$\sigma$ surprise) that disproportionately favors one leg of the pair.
   - **Decision rule:** Enter a dollar-neutral (or beta-neutral) long/short pair position when $|Z_t| > 2.0$ and ADF $p < 0.05$. Exit when the spread reverts to $|Z_t| < 0.5$, when holding time reaches 15 trading days (time stop), or if the spread widens beyond $|Z_t| > 3.2$ (structural break stop-loss).

6. **Validation and risks.**
   - **Benchmark:**
     1. Unfiltered textbook Bollinger-band pairs strategy on the exact same pairs (without the earnings blackout and TWSE monthly revenue regime filter) to isolate the value added by the fundamental filter.
     2. Risk-free rate (`SOFR` / 3-month T-bill yield) and `SOXX` ETF buy-and-hold Sharpe ratio.
   - **Rejection criteria:** Reject the strategy if the out-of-sample annualized Sharpe ratio (net of 5 bps per leg slippage/commissions and short borrow costs) is below **1.0**, if the fundamental filter fails to reduce maximum drawdown by at least 20% compared to the unfiltered baseline, or if more than 25% of trades hit the $|Z_t| > 3.2$ stop-loss (indicating frequent cointegration breakdown).
   - **Key limitations & risks:**
     - *Cointegration instability (Regime shift):* AI product cycles can cause permanent valuation decoupling between historically linked peers (e.g., high-bandwidth memory or advanced packaging winners vs. legacy nodes).
     - *Short-selling and margin constraints:* Executing a two-legged long/short strategy requires margin account borrowing and incurs short-financing fees, reducing net returns compared to long-only strategies.
     - *Execution synchrony:* Entering and exiting both legs simultaneously at daily close requires limit/VWAP execution to avoid paying double bid-ask spread during volatile market closes.

