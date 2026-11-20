# MORESHWAR 3.1 (PROMPT 1/5) — DATA INTEGRITY, DIVIDEND & LOOK-AHEAD FORENSIC FINDINGS

- **Dataset Version Audited:** `m3_20261119T191710Z` (`data/m3_universe/nse_pit_universe.db`)
- **Panel Dimensions:** `1,454` trading sessions (`2021-01-01` to `2026-11-19`) × `2,832` operating NSE equities

## 1. Survivorship Bias & Delisted Security Verification (`VERIFIED CLEAN`)

- **Security Lifecycle Distribution in Panel (`2,832` equities):** `{"INACTIVE_OR_DELISTED_MERGED": 225, "ACTIVE": 2606, "DELISTED": 1}`
- **Post-Delisting Eligibility Violations (`dt >= delisting_date` and `is_pit_elig == True`):** **`0`**
- **Verification (`panel_loader.py` lines 190–202):** Every daily bar explicitly checks `listing_date <= trade_date < delisting_date`, `series in ('EQ', 'BE')`, `close >= INR 10.0`, and causal trailing 20D/60D liquidity (`adv_20d_inr >= INR 5 Cr`, `trading_freq_60d >= 0.80`). Distressed and delisted stocks (e.g., `HDFC`, `MINDTREE`, `LTI`, `DHFL`, `SINTEX`, `FUTURECONSUMER`, `ZEEL`, `YESBANK`) enter and exit the eligible universe causally.

## 2. CRITICAL FINDING #1: Omitted Cash Dividends in Portfolio & Strategy NAV Ledgers (`VERIFIED LEAKAGE`)

- **Code Location:** [`builder.py`](file:///home/antigravity/Parth2004124/Finsistant-version-2/finsistant_v2/m3/builder.py#L275-L276) lines 275–276, [`strategy_base.py`](file:///home/antigravity/Parth2004124/Finsistant-version-2/finsistant_v2/m3/strategy_base.py#L280-L370), and [`portfolio_construction.py`](file:///home/antigravity/Parth2004124/Finsistant-version-2/finsistant_v2/m3/portfolio_construction.py#L390-L520).
- **Exact Mechanism:**
  1. In `builder.py` line 275 (`if parsed_ca['action_type'] in ('SPLIT', 'BONUS')`), `adj_open`, `adj_high`, `adj_low`, and `adj_close` are backward-adjusted **only for stock splits and bonuses** — **NOT for cash dividends (`DIVIDEND`)**.
  2. When a stock held by a Moreshwar 3.0 strategy goes **ex-dividend** on date `ti`, its market price drops by roughly the cash dividend amount per share (and sometimes triggers an ATR stop-loss on the ex-dividend drop), **but neither `simulate_independent_strategy_m3` nor `simulate_shared_capital_portfolio_m3` credits `qty * div_inr_ex[ti, col]` to the portfolio cash balance**.
  3. Simultaneously, the Prompt 9 benchmark comparison ([`independent_audit_p9.py`](file:///home/antigravity/Parth2004124/Finsistant-version-2/finsistant_v2/m3/independent_audit_p9.py#L67) line 67) adds `+1.25% p.a.` of reinvested dividends to Nifty 50 (`b_tri_rets = b_rets + 0.0125 / 252.0`). Thus Moreshwar 3.0 compares **Price-Return-Only strategies (with 0% dividend credit) against a Total-Return Index (TRI) benchmark**.
- **Measured Omitted Cash Dividend Leakage on Actual Held Positions (`entry_date < ex_date <= exit_date`):**
  - **Family E (`STRAT_E_DEFENSIVE_YIELD_LOWVOL`):** **`INR 10,760.65`** across `54` trades (`+10.76 pp` omitted return -> lifts Family E from `+26.15%` to **`+36.91%`**).
  - **Shared Portfolio (`PORT_RISK_BUDGETED`):** **`INR 6,104.43`** across `55` trades (`+6.10 pp` omitted return -> lifts uninvested portfolio from `+7.43%` to **`+13.54%`**).
  - **Family A (`STRAT_A_CS_MOMENTUM`):** **`INR 8,121.40`** across `52` trades (`+8.12 pp` omitted return).

## 3. CRITICAL FINDING #2: Family E `DIV_YIELD_TTM_PIT` Timing vs. Split-Unadjusted Numerator (`VERIFIED DEFECT`)

- **Publication / Ex-Date Timing (`VERIFIED NO LOOK-AHEAD`):** In [`panel_loader.py`](file:///home/antigravity/Parth2004124/Finsistant-version-2/finsistant_v2/m3/panel_loader.py#L133-L140) lines 133–140, `div_inr_ex[ti, sj]` is populated strictly on `ex_date` (`ti`). Under SEBI/NSE listing regulations, corporate dividend announcements occur 10–45 calendar days *before* `ex_date`. Therefore, summing trailing 252-day ex-dividend amounts `div_252 = _rolling_sum(div_ex, 252)` at day `ti` close uses only dividends that have already gone ex-dividend (`<= ti`) — **zero future dividend look-ahead exists**.
- **Split/Bonus Numerator Distortion (`VERIFIED BUG` in [`factors.py`](file:///home/antigravity/Parth2004124/Finsistant-version-2/finsistant_v2/m3/factors.py#L284-L285) lines 284–285):**
  - `factors.py` computes `div_yield_ttm = div_252 / raw_c`, where `div_252` sums raw unadjusted `cash_dividend_inr` per share over the past 252 trading days and divides by today's unadjusted close `raw_c`.
  - Whenever a stock executes a **stock split or bonus issue** inside the trailing 252-day window, `raw_c` drops by the split/bonus factor (e.g., 5× or 10×), while pre-split dividends inside `div_252` remain stated per *pre-split* share. This artificially inflates `DIV_YIELD_TTM_PIT` by the split factor for up to 252 trading days after a split/bonus!
  - **Empirical Scope:** Found **`1704`** PIT-eligible stock-days where a split/bonus inside the trailing 252-day window inflated `DIV_YIELD_TTM_PIT` above 8%.

| Date | Symbol | Raw Close (`INR`) | Reported `DIV_YIELD_TTM_PIT` | True Split-Adjusted TTM Yield | Inflation Factor |
| :---: | :--- | :---: | :---: | :---: | :---: |
| `2022-08-17` | `RECLTD` | 105.50 | **16.12%** | 12.09% | `1.33x` |
| `2024-06-21` | `BPCL` | 307.60 | **8.13%** | 4.06% | `2.00x` |
| `2026-01-16` | `BESTAGRO` | 29.65 | **10.12%** | 6.75% | `1.50x` |
| `2025-02-03` | `HINDPETRO` | 324.90 | **8.00%** | 6.46% | `1.24x` |
| `2023-05-18` | `PDSL` | 326.80 | **8.06%** | 2.22% | `3.62x` |
| `2022-09-06` | `GAIL` | 93.05 | **10.75%** | 7.16% | `1.50x` |
| `2022-02-22` | `HGS` | 1310.10 | **15.88%** | 9.01% | `1.76x` |
| `2025-02-14` | `BANCOINDIA` | 319.70 | **9.70%** | 6.57% | `1.48x` |
| `2022-06-30` | `IOC` | 74.25 | **14.14%** | 9.43% | `1.50x` |
| `2025-06-16` | `BAJFINANCE` | 938.00 | **9.81%** | 1.96% | `5.00x` |
| `2023-08-30` | `VISAKAIND` | 94.70 | **8.03%** | 2.11% | `3.80x` |
| `2022-06-20` | `KKCL` | 235.15 | **8.08%** | 4.68% | `1.73x` |

## 4. CRITICAL FINDING #3: Static Sector Snapshot Collapses 73.62% of Universe into One 'Unclassified' Bucket (`VERIFIED BOTTLENECK`)

- **Code Location:** [`builder.py`](file:///home/antigravity/Parth2004124/Finsistant-version-2/finsistant_v2/m3/builder.py#L208-L224) lines 208–224 & [`portfolio_construction.py`](file:///home/antigravity/Parth2004124/Finsistant-version-2/finsistant_v2/m3/portfolio_construction.py#L659) line 659.
- **Exact Scope:** Sector labels were populated only from current NSE index CSV snapshots (`Nifty Total Market` ~750 constituents). Consequently, **`2,085` of `2,832` panel equities (`73.62%`)** — representing **`34.91%` of all PIT-eligible stock-days** — carry the identical fallback sector label `"Unclassified (Historical/Non-Total-Market)"`.
- **Portfolio Impact:**
  1. In `STRAT_D_SECTOR_ROTATION` (`strategy_families.py` line 337), all `Unclassified` stocks are excluded completely (`if 'Unclassified' in s_nm: continue`).
  2. In `PORT_RISK_BUDGETED` (`portfolio_construction.py` line 659) and standalone strategies (`strategy_base.py` line 421), the sector concentration cap (`max_sector_positions = 2`) treats all 2,085 `Unclassified` companies as a **single sector**, blocking any 3rd candidate from that entire pool (`SECTOR_CAP_BLOCKED = 114` times in `PORT_RISK_BUDGETED`).
  3. For the 747 classified stocks, sector labels reflect end-of-sample index snapshot membership (minor static metadata look-ahead for companies that changed industry classification).

## 5. Calendar Alignment & Evaluation Horizon Mismatch (`VERIFIED DISCREPANCY`)

- **Prompt 3 & Prompt 5 Evaluation Start:** `EVAL_START_DATE = "2021-07-01"` (`1,332` sessions through `2026-11-19`).
- **Prompt 4 & Prompt 7 Evaluation Start:** `MIN_REGIME_EVAL_DATE = "2022-01-03"` (`1,206` sessions through `2026-11-19`).
- **Consequence:** Prompt 7's `CONSTRAINED_ADAPTIVE_ORCHESTRATOR` (`+14.83%`) skips the `2021-07-01 -> 2021-12-31` period (126 sessions where `PORT_RISK_BUDGETED` lost `-3.2%`), making direct full-period return comparisons between Prompt 5 (`2021-07-01` start, Nifty 50 Price Return `+62.72%`) and Prompt 7 (`2022-01-03` start, Nifty 50 Price Return `+47.02%`) apples-to-oranges unless aligned to identical start dates.
