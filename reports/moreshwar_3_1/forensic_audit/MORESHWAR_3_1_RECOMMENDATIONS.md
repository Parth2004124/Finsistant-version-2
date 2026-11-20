# MORESHWAR 3.1 (PROMPT 1/5) — PRIORITIZED ENGINEERING & RESEARCH RECOMMENDATIONS

Based strictly on code-verified forensic findings from Moreshwar 3.0 (`v3.0.0-rc1`), the following five engineering improvements are ordered by quantified performance-leakage impact for Moreshwar 3.1:

## Priority 1: Credit Point-in-Time Ex-Date Cash Dividends, Protect Stops from Ex-Div Drops & Fix Split-Adjusted `DIV_YIELD_TTM_PIT`
- **Forensic Evidence:**
  1. `builder.py` (line 275) adjusts `adj_close` only for `SPLIT` and `BONUS`, while `strategy_base.py` and `portfolio_construction.py` never credit `qty * div_inr_ex[ti, col]` on ex-dividend dates. This omits **`INR 10,760.65` (`+10.76 pp`)** across `54` trades in `STRAT_E_DEFENSIVE_YIELD_LOWVOL` (`+26.15% -> +36.91%`) and **`INR 6,104.43` (`+6.10 pp`)** across `55` trades in `PORT_RISK_BUDGETED` (`+7.43% -> +13.54%`), while benchmarks receive TRI dividend accrual.
  2. Worse still, large ex-dividend open price drops falsely trigger stop-losses on the exact ex-dividend date (e.g., `IDFC 2023-02-13` Rs 11.00 div, `COALINDIA 2022-11-15` Rs 15.00 div, `VEDL 2022-05-06` Rs 31.50 div, `OFSS 2023-05-09` Rs 225.00 div), turning winning total-return trades into recorded stop-loss losses.
  3. `factors.py` (line 285) divides unadjusted 252D cash dividends by post-split `raw_c`, artificially inflating `DIV_YIELD_TTM_PIT` by up to `5.0x` across **`1,704` PIT-eligible stock-days** (e.g., `BAJFINANCE`, `PDSL`, `VISAKAIND`, `BPCL`).
- **Moreshwar 3.1 Action:** Credit split-adjusted cash dividends `pos['qty'] * (div_inr_ex[ti, col] / adj_factor)` directly to `settled_cash` on `ex_date`, lower open position stop prices `sl_px` by the ex-dividend cash drop on `ex_date`, and compute `DIV_YIELD_TTM_PIT` strictly on split-adjusted dividend per share divided by `adj_close`.

## Priority 2: Eliminate Stop-Loss Whipsaw Churn (`STOP_LOSS_BE`) & Right-Size Position Lots to Crush Flat CDSL DP Fee Drag
- **Forensic Evidence:**
  - In `PORT_RISK_BUDGETED` (`238` trades), **67.6%** of trades exit via stop-loss or breakeven stop (`STOP_LOSS_BE` at `entry_px * 1.0065` after a `3.0 ATR` move), cutting right-tail winners early and capping average holding period at **37.8 trading days** (`32.91x` turnover, **`INR 9,915.45`** total cost).
  - Flat `INR 15.93` CDSL DP sell charges consume **`INR 3,791.34` (`38.2%` of ALL portfolio transaction costs)** because 10 tiny slots average only `INR 6,883` per trade.
- **Moreshwar 3.1 Action:** Remove the tight `+0.65%` breakeven stop trigger (`breakeven_trigger_atr`) that chokes off multi-month trends on normal pullbacks; replace hard 20-day rank-drop liquidation with hysteresis rank buffers (enter top-10, hold while in top-40 and `close > SMA100`); and consolidate position count on `INR 100k` accounts from 10 tiny lots to 6–7 meaningful lots (`INR 14,000–16,000` each).

## Priority 3: Replace Uninvested Idle Cash & 20-Day Slot Starvation with an Executable Core-Satellite Nifty BeES / Liquid BeES Architecture
- **Forensic Evidence:**
  - `PORT_RISK_BUDGETED` leaves **`55.02%`** of capital idle because mid-cycle stop-outs wait up to 20 trading days for `step_k % 20 == 0` to refill (`-15.10 pp` intra-cycle decay from Day 1 `51.84%` to Day 19 `36.74%`), compounded by 5 stacked sizing haircuts.
  - Sweeping idle cash into Nifty BeES lifts full return from `+7.43%` to **`+71.31%`** (Sharpe `+0.34`) with almost identical max drawdown (`16.76%` vs `16.50%`).
- **Moreshwar 3.1 Action:** Build a unified **Executable Core-Satellite Engine** where unallocated capital resides in **Nifty BeES** (during Bull/Sideways) or **Liquid BeES** (during Bear/Stress), and active sleeves only displace the core when their expected alpha clears explicit transaction-cost hurdles.

## Priority 4: Resolve the `73.62%` 'Unclassified' Sector Bottleneck & Prune Negative-Expectancy Sleeves
- **Forensic Evidence:**
  1. `2,085 / 2,832` panel equities (`73.62%`, or `34.91%` of PIT-eligible stock-days) share one `"Unclassified"` label, triggering `114` false sector-cap rejections.
  2. Standalone **`STRAT_C_TREND_PULLBACK`** lost **`-12.29%`** (`Max DD 39.85%`, cost `INR 20,514.74`), and **`STRAT_D_SECTOR_ROTATION`** returned only **`+5.56%`** (`Max DD 36.69%`); inside `PORT_RISK_BUDGETED`, **`STRAT_B_VCP_BREAKOUT`** detracted **`-5.51%`** (`INR -5,508.38`).
- **Moreshwar 3.1 Action:** Backfill complete NSE industry/sector classifications across all 2,832 equities, remove `STRAT_C` and `STRAT_B` from core allocation, and focus active sleeves on high-capacity, lower-turnover Quality/Low-Vol Yield (`Family E`) + 52W-High Momentum (`Family A`).

## Priority 5: Unify Cost Accounting with Prompt 8 Historical Tariffs, ADV Impact & Aligned Evaluation Windows
- **Forensic Evidence:**
  1. `strategy_base.py` and `portfolio_construction.py` call Moreshwar 2.2's static `compute_repaired_delivery_costs` (`7.5 bps` flat slippage, `0.00297%` flat NSE exchange fee) instead of Prompt 8's `calculate_historical_indian_costs` and square-root ADV market-impact model.
  2. Prompt 7's `simulate_strategy_orchestration_m3` blends daily percentage returns starting `2022-01-03` (`1,206` days) rather than discrete share-lot execution starting `2021-07-01` (`1,332` days).
- **Moreshwar 3.1 Action:** Wire Prompt 8's historical tariff and square-root ADV market-impact engine directly into the shared-capital simulator and enforce identical canonical evaluation dates across all modules.
