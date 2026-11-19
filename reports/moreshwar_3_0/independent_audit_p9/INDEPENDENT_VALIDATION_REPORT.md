# MORESHWAR 3.0 (PROMPT 9) — INDEPENDENT ROBUSTNESS, STATISTICAL & STRESS VALIDATION REPORT

> [!CAUTION]
> **FINAL INDEPENDENT AUDIT VERDICT: `NO-GO` FOR LIVE CAPITAL DEPLOYMENT (`approved_for_live_capital = false`)**
> **Verdict Code:** `NO_GO_INSUFFICIENT_NET_RISK_ADJUSTED_EDGE_OVER_PASSIVE_BENCHMARKS`
>
> As required by the Prompt 9 independent auditor mandate, zero parameters were tuned to force a `GO` decision. While Moreshwar 3.0 successfully eliminated survivorship bias, look-ahead bias, and accounting errors across 4,416 NSE securities (`3.50M` validated daily bars), it **does not demonstrate a statistically significant, cost-resilient net alpha edge over passive Indian benchmarks or 6.5% risk-free cash**.

## 1. Primary Reasons for `NO-GO` Live Deployment Verdict

- 1. Negative Active Risk Premium vs Risk-Free Cash & Passive Benchmarks: Over 5.38 years (2021-07 -> 2026-11), active equity selection returned +7.43% uninvested (+29.76% with 6.5% idle cash sweep), underperforming 6.5% risk-free cash (+39.50%), the utilisation-matched 45% Nifty 50 TRI + 55% Cash benchmark (+56.82%, 6.32% DD), and Nifty 50 TRI (+73.82%).
- 2. Statistical Insignificance After Multiple-Testing Correction: Trade expectancy t-stat is +0.478 (p = 0.3165), and Deflated Sharpe Ratio probability across 185 total research trials is 0.0018 (failing the >= 0.95 institutional gate).
- 3. Outlier Concentration Fragility: Removing the top 5 winning trades out of 238 trades turns net uninvested return from +7.43% to -9.32%.
- 4. Walk-Forward Inconsistency & Holdout Contamination: Only 4 of 6 walk-forward sub-windows produced positive uninvested net returns, and the 2025-04 -> 2026-11 OOS period was repeatedly inspected across Prompts 2-7.

## 2. Reproducibility & Four-Ledger Reconciliation (`reproducibility_and_failure_registry.json`)

- **Initial Capital:** `INR 100,000.00` → **Final NAV:** `INR 107,434.41` (`Net Gain: INR +7,434.41`)
- **Sum of Closed Trade Net PnLs:** `INR +7,434.37` → **Trade-to-NAV Ledger Discrepancy:** **`INR +0.04`**
- **Daily Cash-to-Position Mark-to-Market Discrepancy:** **`INR +0.00`**
- **Minimum Daily Settled Cash:** `INR 8,397.07` (`Zero Overdraft: True`)
- **Maximum Gross Exposure:** `92.79%` (`Zero Leverage: True`)
- **Idle Cash Accounting Audit Finding:** Uninvested cash net return is +7.43% (CAGR +1.37%), whereas sweeping the 55.0% average idle cash into 6.5% overnight liquid yield produces +29.76%. Thus +22.33 pp comes from passive risk-free yield on idle cash rather than active equity alpha.

## 3. Consistent Multi-Benchmark Comparison (`2021-07-01 -> 2026-11-19`)

| Portfolio / Benchmark | Total Return | CAGR | Max Drawdown | Basis & Dividend / Cash Assumption |
| :--- | :---: | :---: | :---: | :--- |
| `M3_PORT_RISK_BUDGETED_UNINVESTED_CASH` | **+7.43%** | +1.37% | **16.50%** | Active equity trades only (idle cash at 0%) |
| `M3_PORT_RISK_BUDGETED_WITH_65_CASH_SWEEP` | **+29.76%** | +5.05% | **16.50%** | 45% active equity + 55% idle cash swept at 6.5% p.a. |
| `BENCH_1_NIFTY_50_PRICE` | **+62.72%** | +9.65% | **16.54%** | Nifty 50 Price Index (ex-dividends) |
| `BENCH_2_NIFTY_50_TRI` | **+73.82%** | +11.03% | **16.54%** | Nifty 50 Total Return Index (+1.25% p.a. dividend yield) |
| `BENCH_3_PIT_NSE_BROAD_UNIVERSE_TRI` | **+103.20%** | +14.36% | **26.62%** | Equal-weighted PIT eligible NSE liquid equity universe TRI (Nifty 500 proxy) |
| `BENCH_4_RISK_FREE_CASH_65` | **+39.50%** | +6.50% | **0.00%** | 6.50% p.a. Indian Sovereign T-Bill / Liquid Benchmark |
| `BENCH_5_PASSIVE_45_NIFTY_TRI_55_CASH` | **+56.82%** | +8.88% | **6.32%** | Risk-equivalent passive 45% Nifty 50 TRI + 55% 6.5% Cash (matches strategy's 45% equity utilisation) |

## 4. Six-Window Chronological Walk-Forward & Terminal Holdout Audit

| Walk-Forward Fold | Role | Window | Uninvested Net Return | w/ 6.5% Cash Sweep | Max DD | Net PF | Trades | Nifty 50 Return | Excess vs Nifty |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `WF1_2021H2_2022H1` | `IN_SAMPLE_TRAIN_1` | `2021-07-01 -> 2022-06-30` | **-5.13%** | -0.11% | 8.56% | 0.45 | 30 | +0.64% | -5.77 pp |
| `WF2_2022H2_2023H1` | `IN_SAMPLE_TRAIN_2` | `2022-07-01 -> 2023-06-30` | **+0.91%** | +3.74% | 10.96% | 1.07 | 51 | +21.82% | -20.91 pp |
| `WF3_2023H2_2024H1` | `IN_SAMPLE_TRAIN_3` | `2023-07-01 -> 2024-06-30` | **+7.58%** | +10.72% | 7.44% | 1.45 | 54 | +24.26% | -16.68 pp |
| `WF4_2024H2_2025Q1` | `IN_SAMPLE_VALIDATION` | `2024-07-01 -> 2025-03-31` | **-8.00%** | -5.52% | 14.61% | 0.49 | 44 | -2.58% | -5.42 pp |
| `WF5_2025Q2_2026M05` | `OUT_OF_SAMPLE_TEST` | `2025-04-01 -> 2026-05-18` | **+4.60%** | +8.15% | 7.94% | 1.25 | 63 | +2.09% | +2.51 pp |
| `WF6_2026M05_2026M11_HOLDOUT` | `TERMINAL_6M_HOLDOUT` | `2026-05-19 -> 2026-11-19` | **+4.56%** | +6.25% | 8.15% | 1.46 | 31 | +8.03% | -3.47 pp |

## 5. Statistical Analysis, 20D Block Bootstrap, Monte Carlo & Deflated Sharpe

- **20-Day Circular Block Bootstrap (`2,000` resamples):**
  - Uninvested Net Return 95% CI: **`[-25.57%, +6.91%, +62.76%]`** (`P(>0%) = 63.5%`)
  - Cash-Sweep Net Return 95% CI: **`[-9.64%, +29.19%, +95.88%]`**
  - Max Drawdown 95% CI: **`[9.35%, 18.08%, 34.77%]`**
  - Probability of Beating 6.5% Risk-Free Cash: **`35.8%`** | Probability of Beating Nifty 50 TRI: **`14.1%`**
- **Monte Carlo Trade-Sequence Permutation (`2,000` shuffles of `238` trades):**
  - Trade Expectancy t-stat: **`+0.478`** (`p = 0.3165`)
  - MC Max Drawdown Distribution: `P50 = 11.16%` | `P95 = 17.06%` | `P99 = 19.93%` (`P(DD > 20%) = 0.9%`)
- **Higher Moments & Tail Risk:** Skewness `-1.676` | Excess Kurtosis `+20.317` | Daily VaR(95%) `-0.87%` | Daily CVaR(95%) `-1.38%`
- **Multiple-Testing & Deflated Sharpe Ratio (`N = 185` trials):** Observed Sharpe (vs 6.5% RF) = `-0.552` | Expected Null Max Sharpe = `+1.480` | **DSR Probability = `0.0018`** (`Passes 95% Gate: False`)

## 6. Concentration & Outlier Removal Fragility Audit

- **Baseline Uninvested Return:** **`+7.43%`**
- **Remove Largest Winning Stock (`M&M`, `INR +4,446.84`):** Remaining Return = **`+2.99%`**
- **Remove Top-5 Winning Trades (`INR +16,751.12`):** Remaining Return = **`-9.32%`** (`Turns Negative: True`)
- **Remove Best Calendar Month (`2023-12`, `+9.11%`):** Remaining Return = **`-1.54%`**
- **Remove Best Calendar Year (`2023`, `+17.27%`):** Remaining Return = **`-8.39%`**
- **Remove Best Strategy Family (`STRAT_A_CS_MOMENTUM`, `INR +10,632.04`):** Remaining Return = **`-3.20%`**

## 7. Stress Testing Suite (Costs, Slippage, Delay, Liquidity, Gaps & Vol Spikes)

| Stress Scenario | Description | Uninvested Net Return | w/ Cash Sweep | Max DD | Cost Drag % |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `STRESS_1_COST_1_25X` | 1.25x Transaction & Slippage Friction (~11.5 bps eff. slippage) | **+6.00%** | +28.08% | 16.42% | 11.10% |
| `STRESS_2_COST_1_50X` | 1.50x Transaction & Slippage Friction (~15.5 bps eff. slippage) | **+4.93%** | +26.77% | 16.52% | 12.43% |
| `STRESS_3_COST_2_00X` | 2.00x Transaction & Slippage Friction (~23.5 bps eff. slippage) | **+2.50%** | +24.78% | 16.82% | 13.99% |
| `STRESS_4_SLIPPAGE_20BPS` | 20.0 bps One-Way Execution Slippage | **+1.35%** | +23.34% | 18.30% | 13.08% |
| `STRESS_5_DELAYED_EXEC_T2` | Delayed Execution & T+2 Settlement Lag | **+5.74%** | +27.79% | 16.39% | 10.61% |
| `STRESS_6_LIQUIDITY_CRUNCH` | Liquidity Deterioration (0.35% Volume Cap + 18.75 bps Slippage) | **+5.10%** | +27.06% | 15.80% | 13.30% |
| `STRESS_7_MARKET_GAP_1_5X` | 1.5x Magnification of All Adverse Overnight Stop-Gap Losses (`16` gap trades) | **+2.42%** | — | — | — |
| `STRESS_8_VOLATILITY_SPIKE_REGIMES` | Performance During High-Volatility Market Spikes (Nifty 20D Vol >= 18%) (`182` sessions) | **-6.36%** | — | — | — |

## 8. Capital Scale Sensitivity (`INR 50k` to `INR 50L`) & Parameter Stability

| Capital Tier | Uninvested Net Return | w/ Cash Sweep | Max Drawdown | Cost Drag % | Liquidity Partial Fills |
| :--- | :---: | :---: | :---: | :---: | :---: |
| INR 50,000 (Small Retail - High Fixed DP Charge Impact) | **+7.51%** | +31.58% | 17.55% | 12.82% | `0` |
| INR 1,00,000 (Baseline Research Capital) | **+7.43%** | +29.76% | 16.50% | 9.83% | `0` |
| INR 10,00,000 (HNI Scale) | **+1.58%** | +24.70% | 19.96% | 5.75% | `0` |
| INR 50,00,000 (Institutional Scale - Liquidity Cap Binding) | **-0.04%** | +23.13% | 18.57% | 5.24% | `1` |

## 9. Eight-Category Forensic Bias & Failure Registry

| Audit Category | Status | Forensic Evidence |
| :--- | :---: | :--- |
| **1. Survivorship Bias** | `PASSED_ELIMINATED` | Prompt 1 database retains 4,416 NSE securities including 240+ delisted/merged/suspended equities (e.g. HDFC, MINDTREE, PEL, IDFC, TATACOFFEE) up to their actual delisting date. |
| **2. Look-Ahead Bias** | `PASSED_ELIMINATED` | Verified programmatically across all 31 factors, 5 strategy families, 5 regime detectors, and pair discovery via future-price corruption invariance tests (t close -> t+1 open execution). |
| **3. Data Leakage (Static Sector & Pre-2024 MCAP)** | `DISCLOSED_MINOR_RESIDUAL` | Pre-2024 official NSE MCAP is properly left NULL (never backfilled), but sector labels in security_master come from current NSE classification snapshots (`known_limitations.json`). |
| **4. Multiple-Testing & Researcher Selection Bias (OOS Holdout Exposure)** | `FAILED_CONTAMINATED` | 185+ factor, strategy, sizing, and orchestrator configurations were evaluated across Prompts 2-7 with repeated inspection of the 2025-04 -> 2026-11 OOS window. Deflated Sharpe Ratio probability is well below 0.95. |
| **5. Benchmark & Risk-Free Rate Hurdle** | `FAILED_ECONOMIC_HURDLE` | Active equity trading returned +7.43% uninvested (+29.76% with 6.5% idle cash sweep), trailing the 6.5% risk-free cash benchmark (+39.50%), the utilisation-matched 45/55 passive Nifty TRI + Cash benchmark (+56.82%), and Nifty 50 TRI (+73.82%). |
| **6. Outlier Concentration & Cost Fragility** | `FAILED_FRAGILITY_GATE` | Removing the Top-5 winning trades drops uninvested return from +7.43% to -9.32%. Under 2.0x transaction costs, uninvested return drops to +2.50%. |
| **7. Accounting & Four-Ledger Reconciliation** | `PASSED_EXACT` | Trade-to-NAV discrepancy = INR +0.04; daily cash-to-position discrepancy = INR +0.00; T+1 cash settlement enforced with 0 overdrafts. |
| **8. Execution & Broker Isolation Realism** | `PASSED_PAPER_ISOLATED` | Prompt 8 historical pre/post-Oct-2024 Zerodha/NSE/SEBI tariffs, DP charges (Rs 15.93), square-root market impact, volume participation caps (1%), and paper-only broker safety gates verified. |
