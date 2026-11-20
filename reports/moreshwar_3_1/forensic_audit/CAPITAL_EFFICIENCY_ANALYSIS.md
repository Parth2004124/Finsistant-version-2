# MORESHWAR 3.1 (PROMPT 1/5) — CAPITAL EFFICIENCY & 55.0% IDLE CASH FORENSIC ANALYSIS

## 1. Executive Summary of the 55.0% Average Idle Cash Finding

In `PORT_RISK_BUDGETED` (`2021-07-01 -> 2026-11-19`), average deployed equity exposure is **`44.98%`** and average idle cash is **`55.02%`** (of which `1.09%` is in T+1 unsettled exit proceeds). The portfolio holds an average of **`6.96` open positions** out of `10` allowed slots.

**Crucial Forensic Conclusion:** The 55.02% cash reserve is **NOT primarily an intentional defensive regime call** — across all 5 market regimes, the code's intentional regime cash buffer (`eff_reg_cash_pct = min(25%, 0.40 * reg_cash_pct)` in `portfolio_construction.py` line 616) is only **`5.0%` in Strong/Weak Bull, `6.0%` in Sideways, `14.0%` in Weak Bear, and `24.0%` in Stress Bear**. The remaining **`37.6 pp to 60.3 pp` of idle cash is unintentional structural leakage** caused by four interacting code mechanics.

## 2. Four Structural Causes of Unintentional Idle Cash

### Cause 1: Intra-Cycle Stop-Out Vacated Slots Wait Up to 20 Trading Days to Refill (`-15.10 pp` Utilization Drop)
- **Code Location:** [`portfolio_construction.py`](file:///home/antigravity/Parth2004124/Finsistant-version-2/finsistant_v2/m3/portfolio_construction.py#L390-L532) lines 390–483 (daily stop-loss exits) vs. line 529 (`is_rebal_day = (step_k % 20 == 0)`).
- **Mechanism:** Stop-loss (`STOP_LOSS`) and breakeven-stop (`STOP_LOSS_BE`) exits execute **every single day** (accounting for **161 of 238 total exits = 67.6%**), but the portfolio **only scans for new entry candidates once every 20 trading days (~1 calendar month)**.
- **Measured Intra-Cycle Decay across 67 Rebalance Cycles:**
  - **Day 1 After Rebalance (`step_k % 20 == 1`):** Average Deployed Exposure = **`51.84%`** (`8.13` open positions).
  - **Day 10 Mid-Cycle (`step_k % 20 == 10`):** Average Deployed Exposure = **`45.01%`**.
  - **Day 19 Before Next Rebalance (`step_k % 20 == 19`):** Average Deployed Exposure drops to **`36.74%`** (`5.64` open positions) — an intra-cycle bleed of **`-15.10 pp`**!

### Cause 2: Multiplicative Stacking of Five Sizing & Exposure Haircuts on Rebalance Day
- **Code Location:** [`portfolio_construction.py`](file:///home/antigravity/Parth2004124/Finsistant-version-2/finsistant_v2/m3/portfolio_construction.py#L615-L691) lines 615–691.
- Even on Day 1 immediately after a rebalance, average deployed capital is only **`51.84%`** (never approaching the 90–95% ceiling) because 5 haircuts stack multiplicatively:
  1. `eff_max_exp_pct = min(95%, 100% - max(5%, 0.40 * reg_cash_pct)) * dd_mult * vol_scale`.
  2. `base_slot_inr = min(13% * nav, max_exp_amt / 8)` -> caps base slot at `11.875% * dd_mult` (`~9.7%` when drawdown governor `dd_mult = 0.82` is active).
  3. `RISK_BUDGETED` takes `min(base_slot_inr * erc_mult, stop_cap_inr)` where `erc_mult` is penalized by `1 / (1 + 0.35 * avg_corr)`.
  4. Integer share truncation `qty = int(alloc_amt // spot)` leaves `INR 500–3,500` uninvested per slot on higher-priced NSE stocks (`ULTRACEMCO`, `MARUTI`, `BAJFINANCE`).
  5. Daily two-way turnover throttle (`max_daily_turnover_pct = 28.0%`, line 272) caps rebalance-day sells + buys at 28% of NAV (`TURNOVER_THROTTLE_BLOCKED = 25` times).

### Cause 3: 'Unclassified' Sector Concentration Bottleneck (`SECTOR_CAP_BLOCKED = 114`)
- Because `2,085` of `2,832` panel equities (`73.62%`) share the single fallback sector string `"Unclassified (Historical/Non-Total-Market)"`, `max_sector_positions = 2` blocks any 3rd candidate from that entire pool.

### Cause 4: Breakeven Stop Ratchet (`+3.0 ATR -> entry_px * 1.0065`) Prematurely Ejects Winners into Cash
- **Code Location:** [`portfolio_construction.py`](file:///home/antigravity/Parth2004124/Finsistant-version-2/finsistant_v2/m3/portfolio_construction.py#L488-L492) lines 488–492.
- **60 out of 238 exits (25.2%)** are `STOP_LOSS_BE` or `STOP_LOSS_BE_GAP` (locked at `+0.65%` above entry price after a `3.0 ATR` move), ejecting trending positions back into idle cash during normal pullbacks.

## 3. Capital Utilization by Market Regime (`2021-07-01 -> 2026-11-19`)

| Causal Market Regime | Trading Days | % of Sample | Policy Cash Reserve | Code Effective Cash Reserve | Actual Avg Deployed % | Actual Avg Idle Cash % | Unexplained Idle Cash Gap |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `SIDEWAYS_CHOP` | 318 | 23.9% | 15.0% | 6.0% | **33.7%** | **66.3%** | `+60.3 pp` |
| `WEAK_BULL` | 427 | 32.1% | 5.0% | 5.0% | **50.9%** | **49.1%** | `+44.1 pp` |
| `STRESS_BEAR` | 272 | 20.4% | 60.0% | 24.0% | **35.9%** | **64.1%** | `+40.1 pp` |
| `STRONG_BULL` | 285 | 21.4% | 0.0% | 5.0% | **57.4%** | **42.6%** | `+37.6 pp` |
| `WEAK_BEAR` | 30 | 2.3% | 35.0% | 14.0% | **45.1%** | **54.9%** | `+40.9 pp` |

## 4. Counterfactual Idle-Cash Deployment Evaluation (Including Drawdown & Tail Risk)

| Idle-Cash Treatment Counterfactual | Total Return | CAGR | Max Drawdown | Annualized Vol | Sharpe (vs 6.5% RF) | Daily 95% CVaR (Tail Risk) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Uninvested Idle Cash (0% Yield — M3.0 Baseline)** | `+7.43%` | `+1.37%` | `16.50%` | `8.64%` | `-0.55` | `-1.38%` |
| **2. 6.5% Risk-Free Liquid Cash Sweep** | `+29.76%` | `+5.06%` | `12.75%` | `8.64%` | `-0.14` | `-1.37%` |
| **3. Regime-Gated Nifty BeES Core (Bull/Sideways) + Liquid BeES (Bear)** | **`+39.72%`** | **`+6.54%`** | `18.02%` | `11.94%` | `+0.05` | `-1.81%` |
| **4. Always-Invested Nifty BeES Core on Idle Cash (5% Buffer)** | **`+71.31%`** | **`+10.73%`** | `16.76%` | `13.66%` | `+0.34` | `-2.07%` |
| *Reference: Executable 100% Nifty BeES ETF* | *`+75.10%`* | *`+11.19%`* | *`16.35%`* | *`13.72%`* | *`+0.37`* | *—* |

- **Risk-Adjusted Verdict on Idle Cash:** Sweeping idle cash into **Always-Invested Nifty BeES** lifts portfolio return from `+7.43%` to **`+71.31%`** (`+63.88 pp`, Sharpe `+0.34` vs `-0.55`), while increasing max drawdown only marginally (`16.50%` -> `16.76%`) and widening daily 95% CVaR (`-1.38%` -> `-2.07%`). However, even with 95% total equity exposure, it still trails 100% executable Nifty BeES (`+75.10%`, Max DD `16.35%`, Sharpe `+0.37`) by `-3.79 pp` because the **45% active stock-picking sleeve lags Nifty 50 before dividends are credited and after churn costs**.
