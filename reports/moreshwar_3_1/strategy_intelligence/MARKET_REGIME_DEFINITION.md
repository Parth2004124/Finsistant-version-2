# MORESHWAR 3.1 (PROMPT 4/5) — MARKET REGIME DEFINITION & CAUSAL FEATURE SPECIFICATION

## 1. Point-in-Time Data Audit & Excluded Variables

- **100%-Coverage Tradable Regime Data (`2021-01-01 -> 2026-11-19`, `1,454` sessions):** Built exclusively on real NSE `NIFTYBEES` (`INF204KB14I2`) OHLCV bars and the Point-in-Time eligible NSE equity universe (`is_pit_elig[ti]`, NEVER surviving stocks) + 4-tier sector hierarchy.
- **Partial-Coverage External Index Series (`2021-01-01 -> 2025-10-10`, `1,181` sessions):** `India VIX`, `Nifty GS 10Yr`, and `NIFTY50 EQL Wgt` end on `2025-10-10` in `daily_index_bars`. They are recorded strictly as **diagnostic columns** in `REGIME_CLASSIFICATION_HISTORY.csv` (`data_availability_status` explicitly flags post-`2025-10-10` dates) and are NOT fabricated or forward-filled into the tradable regime classifier.
- **Excluded Macroeconomic & Fundamental Variables:** RBI policy repo announcements, CPI inflation prints, USD/INR FX rates, and quarterly corporate earnings/balance-sheet statements lack historical point-in-time publication timestamps in the local database and are **explicitly excluded** to prevent lookahead leakage.
- **Clearly Labeled Liquidity Proxies:** No historical order-book bid-ask spread or market-depth data exists. All liquidity/stress indicators (`etf_hl_spread_proxy_10d`, `mkt_median_hl_proxy`, `etf_vol_ratio_5_20`, `mkt_gap_freq_pct`) are explicitly labeled OHLCV proxies.

## 2. Model A: Transparent 6-State Causal Rule-Based Hierarchy (3-Day Confirmation Hysteresis)

| Priority | State | Transparent Causal Definition (Strictly `<= ti`) | Eval Days (`2021-07 -> 2026-11`) |
| :---: | :--- | :--- | :---: |
| 1 | `HIGH_VOL_STRESS` | `vol_20d_ann >= 18.5%` OR (`vol_pctile_252d >= 0.85` AND `dist_252h <= -5%`) OR `dist_252h <= -10.5%` (Immediate transition for capital protection) | `285` |
| 2 | `BEAR_DETERIORATING` | (`px < SMA200` AND (`pct_ab50 < 45%` OR `ret_60d < -3%`)) OR (`px < SMA50 - 2.5%` AND `pct_ab50 < 35%`) | `78` |
| 3 | `RECOVERY_IMPROVING` | (`dist_252h <= -3.5%` OR `px_vs_sma200 < +1.5%`) AND `breadth_delta_20d >= +8 pp` AND `ret_20d >= +1.2%` | `99` |
| 4 | `BULL_BROAD` | `px >= SMA50` AND `sma50_slope_20d >= 0` AND `pct_ab50 >= 55%` AND `sector_participation >= 55%` | `485` |
| 5 | `BULL_NARROW` | `px >= SMA50` AND (`ret_20d > 0.5%` OR `px_vs_sma200 >= 2%`) AND (`pct_ab50 < 55%` OR `ew_vs_nifty_20d < -1.5%`) | `127` |
| 6 | `SIDEWAYS_RANGE` | All other range-bound / choppy consolidation sessions | `258` |

## 3. Model B: Statistical 4-State K-Means Classifier (Strictly Train-Frozen Per Fold)

- Fitted on 6 standardized causal features (`nb_ret_60d`, `px_vs_sma50_pct`, `pct_ab50`, `breadth_delta_20d`, `vol_20d_ann`, `ew_vs_nifty_20d`) using **training-slice mean/std and centroids ONLY**, then frozen across outer evaluation folds (`STAT_BULL_BROAD`, `STAT_BULL_NARROW_CHOP`, `STAT_RECOVERY`, `STAT_STRESS_BEAR`).
- Compared against Simple Trend Baseline (`NIFTYBEES >= SMA200`: `BULL_TREND` vs `BEAR_TREND`).
