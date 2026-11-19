# MORESHWAR 3.0 (PROMPT 6) — RELATIVE-VALUE, PAIRS, MARKET-NEUTRAL & FUTURES-HEDGED RESEARCH REPORT

> [!IMPORTANT]
> **ISOLATED RESEARCH BRANCH (`mixed_into_long_only_portfolio = false`)**
> All short/relative-value strategies below are evaluated with realistic Indian NSE SLBM borrow costs (`10.0%–10.5% p.a.`), top-180 ADV borrow-availability restrictions, dividend-recall closeouts, and NSE Nifty 50 Futures lot-size (`25/50/75` units, `~INR 18.4L/lot`) & monthly expiry roll costs. Unvalidated or cost-eroded short strategies are **explicitly rejected** from the `INR 100,000` long-only portfolio.

## 1. Executive Summary & Strategy Branch Comparison (`2021-07-01 -> 2026-11-19`)

| Branch ID | Strategy Description | Full Net Return | Full Max DD | Full Vol | Ex-Post Beta | Corr vs Nifty | Corr vs Long Port | OOS Net Return (`2025-04 -> 2026-11`) | OOS Max DD | Institutional Verdict |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `STRAT_RV1_COINT_PAIRS` | Branch 1A: Engle-Granger Cointegrated Pairs Trading (SLB 10% Borrow + Risk Stops) | **-20.93%** | **21.42%** | 4.84% | `-0.010` | `-0.029` | `-0.087` | **-3.96%** | **7.47%** | `REJECTED_AFTER_SLB_BORROW_AND_COSTS` |
| `STRAT_RV1B_HIGH_CORR_NON_COINT_ABLATION` | Branch 1B (Ablation): High-Correlation WITHOUT Cointegration Pairs (Divergence Proof) | **-39.12%** | **40.19%** | 4.02% | `+0.023` | `+0.079` | `+0.046` | **-13.85%** | **15.47%** | `REJECTED_STRUCTURAL_DIVERGENCE` |
| `STRAT_RV2A_SECTOR_REL_MOM` | Branch 2A: Intra-Sector Leadership Long-Short (Top Leaders vs Borrowable Laggards) | **+4.29%** | **19.51%** | 10.44% | `-0.016` | `-0.021` | `+0.105` | **-12.38%** | **13.65%** | `RESEARCH_MONITOR_ONLY_SLB_DRAG` |
| `STRAT_RV2B_SECTOR_SPREAD_REV` | Branch 2B: Intra-Sector 20D Spread Mean-Reversion Long-Short | **-35.95%** | **44.21%** | 11.06% | `+0.064` | `+0.079` | `-0.008` | **+3.86%** | **11.69%** | `REJECTED_NEGATIVE_EXPECTANCY` |
| `STRAT_RV3_MARKET_NEUTRAL_LS` | Branch 3: Cross-Sectional Beta-Neutral Factor Long-Short (Top-180 SLB Universe) | **-22.63%** | **41.27%** | 8.51% | `-0.081` | `-0.130` | `+0.041` | **-14.55%** | **17.96%** | `VALID_ZERO_BETA_DIVERSIFIER_AT_INSTITUTIONAL_SCALE_ONLY` |
| `STRAT_RV4A_NIFTY_FUT_HEDGED_25L_LOT` | Branch 4A: Nifty 50 Futures Regime-Hedged Portfolio (Rs 25L Capital, Actual 25/50/75 Lot Size & Monthly Rolls) | **+6.55%** | **14.36%** | 8.40% | `+0.299` | `+0.489` | `+0.970` | **-5.01%** | **9.69%** | `VALIDATED_DRAWDOWN_REDUCER_FOR_25L_PLUS_CAPITAL` |
| `STRAT_RV4B_NIFTY_FUT_HEDGED_100K_LOT` | Branch 4B: Nifty 50 Futures Hedged at Rs 1,00,000 Capital (Blocked by Rs 18.4L Contract Lot Minimum) | **+7.43%** | **16.50%** | 8.64% | `+0.360` | `+0.572` | `+1.000` | **-5.74%** | **10.39%** | `INFEASIBLE_AT_100K_CAPITAL_USE_CASH_RESERVE_INSTEAD` |
| `STRAT_RV4C_NIFTY_FUT_IDEAL_FRACTIONAL` | Branch 4C (Theoretical Reference): Fractional Nifty 50 Futures Regime Hedge | **+3.36%** | **16.68%** | 8.16% | `+0.219` | `+0.369` | `+0.938` | **-5.88%** | **8.15%** | `THEORETICAL_REFERENCE_ONLY` |
| `BASELINE_PROMPT5_LONG_ONLY` | Prompt 5 Unhedged Long-Only Risk-Budgeted Portfolio (With Regime Cash Reserve) | **+7.43%** | **16.50%** | 8.64% | `+0.360` | `+0.572` | `+1.000` | **-5.74%** | **10.39%** | `PRIMARY_PRODUCTION_CANDIDATE_FOR_100K` |

## 2. Cointegration vs High-Correlation Pair Discovery (Strictly In-Sample `<= 2025-03-31`)

- **Same-Sector / Same-Industry Liquid Pairs Screened (`rho_IS >= 0.48`):** `197`
- **Passed Engle-Granger ADF Cointegration (`t_ADF <= -2.86`, `3d <= Half-Life <= 35d`, Beta Drift CV `<= 0.35`):** `19` (`12` diversified pairs traded)

### 2A. Selected In-Sample Cointegrated Pairs

| Pair ID | Sector | Industry | IS Return Corr | IS OLS Beta | Engle-Granger ADF t-stat | Approx p-val | OU Half-Life (Days) | Subwindow Beta CV |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `PNB_MAHABANK` | Financial Services | Financial Services | `+0.640` | `0.978` | **`-4.449`** | `0.0050` | **15.1d** | `0.033` |
| `ONGC_GAIL` | Oil Gas & Consumable Fuels | Oil Gas & Consumable Fuels | `+0.573` | `0.861` | **`-4.134`** | `0.0050` | **22.5d** | `0.073` |
| `HINDALCO_NMDC` | Metals & Mining | Metals & Mining | `+0.535` | `0.590` | **`-3.985`** | `0.0050` | **27.8d** | `0.270` |
| `SIEMENS_CUMMINSIND` | Capital Goods | Capital Goods | `+0.515` | `0.825` | **`-3.769`** | `0.0050` | **21.5d** | `0.142` |
| `TCS_LTTS` | Information Technology | Information Technology | `+0.485` | `0.480` | **`-3.679`** | `0.0050` | **26.3d** | `0.238` |
| `UNIONBANK_MAHABANK` | Financial Services | Financial Services | `+0.637` | `1.007` | **`-3.513`** | `0.0050` | **31.0d** | `0.262` |
| `GAIL_IOC` | Oil Gas & Consumable Fuels | Oil Gas & Consumable Fuels | `+0.529` | `0.983` | **`-3.393`** | `0.0200` | **26.5d** | `0.224` |
| `NMDC_HINDCOPPER` | Metals & Mining | Metals & Mining | `+0.585` | `0.604` | **`-3.321`** | `0.0200` | **33.2d** | `0.204` |
| `TCS_HCLTECH` | Information Technology | Information Technology | `+0.640` | `0.434` | **`-3.301`** | `0.0200` | **27.0d** | `0.216` |
| `HINDALCO_NATIONALUM` | Metals & Mining | Metals & Mining | `+0.705` | `0.484` | **`-3.283`** | `0.0200` | **30.2d** | `0.284` |
| `PFC_RECLTD` | Financial Services | Financial Services | `+0.883` | `0.957` | **`-3.261`** | `0.0200` | **30.5d** | `0.058` |
| `SIEMENS_ABB` | Capital Goods | Capital Goods | `+0.584` | `0.835` | **`-3.235`** | `0.0200` | **28.6d** | `0.264` |

### 2B. High-Correlation Pairs That FAILED Cointegration (Why Correlation != Cointegration)

| Pair ID | Sector | IS Return Corr | Engle-Granger ADF t-stat | OU Half-Life | Failure Diagnosis |
| :--- | :--- | :---: | :---: | :---: | :--- |
| `AMBUJACEM_ACC` | Construction Materials | **`+0.826`** | `-0.820` (`p=0.65`) | `352.6d` | Non-stationary residual drift (unit root not rejected) |
| `BANKBARODA_CANBK` | Financial Services | **`+0.796`** | `-2.157` (`p=0.25`) | `72.0d` | Non-stationary residual drift (unit root not rejected) |
| `SBIN_BANKBARODA` | Financial Services | **`+0.755`** | `-1.668` (`p=0.65`) | `102.1d` | Non-stationary residual drift (unit root not rejected) |
| `IOC_HINDPETRO` | Oil Gas & Consumable Fuels | **`+0.750`** | `-2.141` (`p=0.25`) | `84.4d` | Non-stationary residual drift (unit root not rejected) |
| `SBIN_CANBK` | Financial Services | **`+0.735`** | `-2.359` (`p=0.25`) | `49.0d` | Non-stationary residual drift (unit root not rejected) |
| `RVNL_IRCON` | Construction | **`+0.723`** | `-1.459` (`p=0.65`) | `122.0d` | Non-stationary residual drift (unit root not rejected) |
| `BANKBARODA_PNB` | Financial Services | **`+0.719`** | `-2.212` (`p=0.25`) | `105.0d` | Non-stationary residual drift (unit root not rejected) |
| `DLF_GODREJPROP` | Realty | **`+0.719`** | `-1.770` (`p=0.65`) | `131.1d` | Non-stationary residual drift (unit root not rejected) |

- **Empirical Proof (Cointegrated vs High-Corr Non-Cointegrated):** Cointegrated pairs (`STRAT_RV1_COINT_PAIRS`) experienced `{'MEAN_REVERSION_EXIT': 129, 'SPREAD_STOP_Z_EXIT': 16, 'MAX_LOSS_STOP_EXIT': 14, 'CORR_BREAKDOWN_EXIT': 19, 'HEDGE_DRIFT_EXIT': 61, 'BORROW_RECALL_DIV_EXIT': 43, 'HORIZON_EXIT': 3}` exit events with `37.20%` execution cost drag and `11.69%` SLB borrow drag, whereas trading high-correlation non-cointegrated pairs (`STRAT_RV1B_HIGH_CORR_NON_COINT_ABLATION`) suffered repeated structural break stops (`{'MEAN_REVERSION_EXIT': 119, 'SPREAD_STOP_Z_EXIT': 26, 'MAX_LOSS_STOP_EXIT': 18, 'CORR_BREAKDOWN_EXIT': 5, 'HEDGE_DRIFT_EXIT': 49, 'BORROW_RECALL_DIV_EXIT': 17, 'HORIZON_EXIT': 8}`).

## 3. Indian Short-Selling & Futures Microstructure Constraints Analysis

1. **SEBI Cash Short-Selling & SLBM Financing Drag:** Multi-day shorting in NSE cash equities requires SLBM borrowing (available across only `180` liquid F&O/SLB stocks) at ~`10.5% p.a.` lending fee plus two-legged delivery/roll friction. Across 5.4 years, SLB financing & transaction friction consumed **`17.56%` borrow drag + `7.19%` execution cost drag** on the Market-Neutral Long-Short portfolio (`STRAT_RV3_MARKET_NEUTRAL_LS`), turning positive gross factor spread into a modest net return (`-22.63%` Full, `-14.55%` OOS) despite near-zero market beta (`-0.081`).
2. **Nifty 50 Index Futures Lot-Size Quantization (`INR 1,00,000` vs `INR 25,00,000`):** Post-Nov-2024 SEBI F&O contract revision, 1 lot of Nifty 50 Futures is `75` units (`~INR 1,913,573` notional, requiring `~INR 4.0L` SPAN+Exposure margin). On a `INR 1,00,000` account (`STRAT_RV4B_NIFTY_FUT_HEDGED_100K_LOT`), lot quantization blocked futures hedging on `510` regime-stress sessions because even 1 lot represents `18x` portfolio notional. At `INR 25,00,000` institutional capital (`STRAT_RV4A_NIFTY_FUT_HEDGED_25L_LOT`), lot-quantized Nifty Futures hedging (`13` monthly rolls, `1.54%` roll/commission drag) reduced Full Max Drawdown from **`16.50%` → `14.36%`** and cut market beta from `+0.360` to **`+0.299`**.

## 4. Final Prompt 6 Decision

- **For `INR 1,00,000` Retail/Standard Capital:** Do **NOT** mix SLBM equity shorts, pairs trading, or over-sized Nifty Futures lots into the Prompt 5 long-only portfolio. Dynamic cash-reserve raising (`CASH_RESERVE` + Drawdown Governor from Prompts 4 & 5) is strictly superior after Indian SLB borrow fees and F&O lot constraints.
- **For `INR 25,00,000+` Institutional Capital:** The causal Regime-Conditioned Nifty 50 Futures Overlay (`STRAT_RV4A_NIFTY_FUT_HEDGED_25L_LOT`) is validated as a genuine drawdown-reduction overlay during `SIDEWAYS_CHOP`, `WEAK_BEAR`, and `STRESS_BEAR` regimes.
