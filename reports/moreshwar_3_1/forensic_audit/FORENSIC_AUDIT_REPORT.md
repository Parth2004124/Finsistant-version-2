# MORESHWAR 3.1 (PROMPT 1/5) — INDEPENDENT FORENSIC PERFORMANCE AUDIT REPORT

- **Audit Timestamp:** `2026-11-20T03:32:39.698799+00:00`
- **Audited Baseline:** Moreshwar 3.0 (`v3.0.0-rc1`, commit `e6af98f`, Dataset `m3_20261119T191710Z`, Seed `20261119`)
- **Initial Capital:** `INR 100,000.00` (Zero parameter modifications; 100% preservation of Moreshwar 3.0 code and reports)

> [!IMPORTANT]
> **EXECUTIVE FORENSIC VERDICT**
> 1. **Baseline Reproducibility:** **3 of 5** headline figures (`Shared Portfolio + Cash Sweep = +29.76%`, `Adaptive Orchestrator = +14.83%`, `Shared Portfolio without Cash Sweep = +7.43%`) reproduce to **`0.00 pp`** from the frozen Moreshwar 3.0 code and JSON reports. The other **2 standalone figures** (`Family E = +45.47%` and `Family A = +11.98%`) and the **`INR 23,663` transaction cost figure** quoted in the post-compaction chat reply were **conversational summary transcription errors**: the actual frozen Prompt 3 code and [`strategy_families_report.json`](file:///home/antigravity/Parth2004124/Finsistant-version-2/reports/moreshwar_3_0/strategies/strategy_families_report.json) produce **`+26.15%` for Family E** (`-19.32 pp` vs chat), **`+10.97%` for Family A** (`-1.01 pp` vs chat), and **`INR 9,825.77`** reported (`INR 9,915.45` trade-reconstructed) total transaction costs for `PORT_RISK_BUDGETED` (`INR 13,556.29` for Family E; `INR 18,676.26` for Family A; `INR 20,514.74` for Family C).
> 2. **Does Family E Outperform Passive Benchmarks Under Forensic Scrutiny?** **NO.** At its true code-reproduced return of **`+26.15%`** (or **`+36.91%`** after crediting **`INR 10,760.65`** of omitted cash dividends across `54` ex-div holds), Family E **still underperforms** 100% Risk-Free Cash (`+39.46%` / `+40.95%`), executable 45/55 Nifty BeES + Liquid BeES (`+54.93%`), and executable 100% Nifty BeES (`+75.10%`), while enduring a `22.58%` max drawdown and a **`580`-trading-day (~2.3-year) drawdown recovery duration**. Moreover, `1,704` stock-days of `DIV_YIELD_TTM_PIT` had split-unadjusted numerator inflation (though zero future dividend look-ahead).
> 3. **Three Largest Verified Sources of Performance Leakage:** (1) **`55.02%` Unintentional Idle Cash** (`-22.33 pp` vs cash sweep, `-63.88 pp` vs Nifty BeES core overlay) driven by daily stop-outs waiting up to 20 trading days to refill (`-15.10 pp` intra-cycle decay) and 5 stacked sizing haircuts; (2) **Stop-Loss Whipsaw Churn (`67.6%` stop-out rate) & Small-Lot CDSL DP Fee Drag** (`INR 9,915.45` total cost = `9.92%` of capital in `PORT_RISK_BUDGETED`, of which flat `INR 15.93` CDSL DP charges are **`38.2%`**); and (3) **Omitted Ex-Date Cash Dividends + Ex-Dividend Stop-Loss Whipsaws** (`INR 6,104.43` / `+6.10 pp` omitted in `PORT_RISK_BUDGETED`; `INR 10,760.65` / `+10.76 pp` omitted in `Family E`, plus 4 false ex-dividend-day stop-outs each in Portfolio and Family E).

---

## 1. Repository & Execution Pipeline Inspection

| Pipeline Stage | Source File(s) | Forensic Verification & Identified Architectural Gaps |
| :--- | :--- | :--- |
| **1. PIT Data & Corporate Actions** | [`builder.py`](file:///home/antigravity/Parth2004124/Finsistant-version-2/finsistant_v2/m3/builder.py), [`panel_loader.py`](file:///home/antigravity/Parth2004124/Finsistant-version-2/finsistant_v2/m3/panel_loader.py) | `1,454` sessions (`2021-01-01 -> 2026-11-19`), `2,832` active/delisted equities (`0` post-delist leaks). **Gap 1:** `adj_close` adjusts only for `SPLIT`/`BONUS` (line 275), omitting cash dividends. **Gap 2:** Static NSE index snapshots leave `2,085 / 2,832` stocks (`73.62%`) in a single `"Unclassified"` sector bucket. |
| **2. Factor Discovery** | [`factors.py`](file:///home/antigravity/Parth2004124/Finsistant-version-2/finsistant_v2/m3/factors.py) | Strictly causal `[T x N]` matrices (`<= t`). **Gap 3:** Line 285 computes `DIV_YIELD_TTM_PIT = _rolling_sum(div_ex, 252) / raw_c` without scaling pre-split dividends by split factors (`1,704` split-distorted stock-days). |
| **3. Strategy Families (A..E)** | [`strategy_base.py`](file:///home/antigravity/Parth2004124/Finsistant-version-2/finsistant_v2/m3/strategy_base.py), [`strategy_families.py`](file:///home/antigravity/Parth2004124/Finsistant-version-2/finsistant_v2/m3/strategy_families.py) | Signal at `t` close -> fill at `t+1` open; exact `INR 0.00` cash-to-NAV reconciliation. **Gap 4:** Uses static M2.2 `compute_repaired_delivery_costs` (`7.5 bps` flat slippage) instead of Prompt 8 historical tariffs & square-root impact; never credits `div_inr_ex` on held positions or adjusts stops on ex-dividend dates. |
| **4. Shared-Capital Portfolio** | [`portfolio_construction.py`](file:///home/antigravity/Parth2004124/Finsistant-version-2/finsistant_v2/m3/portfolio_construction.py) | Enforces `1.0x` max gross exposure, T+1 cash settlement, 60D correlation gate (`rho <= 0.72`). **Gap 5:** Daily stop-outs (`67.6%` of exits) cannot re-enter until `step_k % 20 == 0`, causing `55.02%` average idle cash. |
| **5. Adaptive Orchestrator** | [`strategy_orchestrator.py`](file:///home/antigravity/Parth2004124/Finsistant-version-2/finsistant_v2/m3/strategy_orchestrator.py) | **Gap 6:** Simulates a linear weight blend of 5 standalone `INR 100k` return streams (`lines 622-623`) starting `2022-01-03` (`1,206` days) rather than discrete share execution starting `2021-07-01` (`1,332` days). |
| **6. Benchmark & Audit** | [`independent_audit_p9.py`](file:///home/antigravity/Parth2004124/Finsistant-version-2/finsistant_v2/m3/independent_audit_p9.py) | **Gap 7:** Compares dividend-excluded strategy NAVs against dividend-included Nifty 50 TRI (`+1.25% p.a.`) and zero-cost daily-rebalanced 45/55 Nifty TRI + Cash. |

---

## 2. Baseline Reproduction & Discrepancy Forensics

Full details exported to [`BASELINE_REPRODUCTION.csv`](file:///home/antigravity/Parth2004124/Finsistant-version-2/reports/moreshwar_3_1/forensic_audit/BASELINE_REPRODUCTION.csv).

| Strategy | Chat Reported Full | Frozen JSON Report Full | Reproduced Full (`2021-07 -> 2026-11`) | Diff vs Chat | Diff vs Frozen JSON | Reproduced Code OOS (`2025-04 -> 2026-11`) | Reproduced Prompt Window (`2024-07 -> 2026-03` Standalone / Slice) | Verdict |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Family E (STRAT_E_DEFENSIVE_YIELD_LOWVOL)** | `+45.47%` | `+26.15%` | **`+26.15%`** | `-19.32 pp` | **`+0.00 pp`** | `+13.31%` | `+1.05%` / `-10.66%` | `MATCHES_FROZEN_CODE_DISCREPANCY_VS_CHAT_SUMMARY` |
| **Shared Portfolio with Cash Sweep (PORT_RISK_BUDGETED + 6.5% Sweep)** | `+29.76%` | `+29.76%` | **`+29.76%`** | `+0.00 pp` | **`+0.00 pp`** | `+21.44%` | `-4.71%` / `-6.81%` | `EXACT_MATCH` |
| **Adaptive Orchestrator (CONSTRAINED_ADAPTIVE_ORCHESTRATOR)** | `+14.83%` | `+14.83%` | **`+14.83%`** | `+0.00 pp` | **`+0.00 pp`** | `+4.28%` | `-5.15%` / `-9.51%` | `EXACT_MATCH` |
| **Shared Portfolio without Cash Sweep (PORT_RISK_BUDGETED Uninvested)** | `+7.43%` | `+7.43%` | **`+7.43%`** | `+0.00 pp` | **`+0.00 pp`** | `+15.70%` | `-11.22%` / `-6.81%` | `EXACT_MATCH` |
| **Family A (STRAT_A_CS_MOMENTUM)** | `+11.98%` | `+10.97%` | **`+10.97%`** | `-1.01 pp` | **`+0.00 pp`** | `+15.41%` | `-18.36%` / `-13.42%` | `MATCHES_FROZEN_CODE_DISCREPANCY_VS_CHAT_SUMMARY` |

---

## 3. Transaction Cost Forensics

Full breakdown across all 6 engines exported to [`TRANSACTION_COST_BREAKDOWN.csv`](file:///home/antigravity/Parth2004124/Finsistant-version-2/reports/moreshwar_3_1/forensic_audit/TRANSACTION_COST_BREAKDOWN.csv) and trade-by-trade attribution to [`TRADE_LEVEL_ATTRIBUTION.csv`](file:///home/antigravity/Parth2004124/Finsistant-version-2/reports/moreshwar_3_1/forensic_audit/TRADE_LEVEL_ATTRIBUTION.csv).

### 3.1 Investigation of the Reported `INR 23,663` Cost Burden vs. Exact Reconstructed Costs
- In the frozen Prompt 5 report ([`portfolio_construction_report.json`](file:///home/antigravity/Parth2004124/Finsistant-version-2/reports/moreshwar_3_0/portfolio_construction/portfolio_construction_report.json) line 412), `PORT_RISK_BUDGETED` executed **`238` trades** with **`32.43x`** reported turnover and incurred **`INR 9,825.77`** (`9.83%` of starting capital; trade-level exact reconstruction = **`INR 9,915.45`**, `0.9%` variance solely due to 2-decimal `net_return_pct` rounding when inverting `qty` from the closed-trade ledger).
- The `INR 23,663` number quoted in the earlier chat reply does **not** exist in any Moreshwar 3.0 code or JSON artifact (`grep` confirmed 0 occurrences across the entire repo); actual high-turnover standalone sleeve costs are **`INR 18,676.26` (`STRAT_A`)**, **`INR 14,347.99` (`STRAT_B`)**, **`INR 20,514.74` (`STRAT_C`)**, **`INR 17,313.85` (`STRAT_D`)**, and **`INR 13,556.29` (`STRAT_E`)**.

### 3.2 Component-Level Statutory & Slippage Breakdown (`INR 100,000` Starting Capital)

| Metric / Cost Component | Shared Portfolio (`PORT_RISK_BUDGETED`) | Family E (`STRAT_E`) | Family A (`STRAT_A`) | Family B (`STRAT_B`) | Family C (`STRAT_C`) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Total Closed Trades** | `238` | `215` | `303` | `245` | `315` |
| **Buy Executed Turnover (`INR`)** | `1,638,037.46` | `2,704,584.37` | `3,708,969.95` | `2,794,914.73` | `4,162,108.33` |
| **Sell Executed Turnover (`INR`)** | `1,652,919.05` | `2,740,206.33` | `3,733,036.26` | `2,817,971.05` | `4,164,088.44` |
| **Total Two-Way Turnover (`INR` / Multiple)** | **`3,290,956.51` (`32.91x`)** | **`5,444,790.69` (`54.45x`)** | **`7,442,006.21` (`74.42x`)** | **`5,612,885.78` (`56.13x`)** | **`8,326,196.77` (`83.26x`)** |
| **1. Delivery STT (`0.10%` Buy + `0.10%` Sell)** | `INR 3,290.96` | `INR 5,444.79` | `INR 7,442.01` | `INR 5,612.89` | `INR 8,326.20` |
| **2. Stamp Duty (`0.015%` Buy Only)** | `INR 245.71` | `INR 405.69` | `INR 556.35` | `INR 419.24` | `INR 624.32` |
| **3. NSE Exchange Txn Charge (`0.00297%`)** | `INR 97.74` | `INR 161.71` | `INR 221.03` | `INR 166.70` | `INR 247.29` |
| **4. SEBI Turnover Fee (`0.0001%`)** | `INR 3.29` | `INR 5.44` | `INR 7.44` | `INR 5.61` | `INR 8.33` |
| **5. GST (`18%` on Exchange + SEBI + Brok)** | `INR 18.19` | `INR 30.09` | `INR 41.13` | `INR 31.02` | `INR 46.01` |
| **6. CDSL DP Charge (`INR 15.93` per Scrip Sell Day)** | **`INR 3,791.34`** | **`INR 3,424.95`** | **`INR 4,826.79`** | **`INR 3,902.85`** | **`INR 5,017.95`** |
| **7. Zerodha Equity Delivery Brokerage** | `INR 0.00` | `INR 0.00` | `INR 0.00` | `INR 0.00` | `INR 0.00` |
| **Total Statutory & DP Charges (`INR`)** | **`INR 7,447.22`** | **`INR 9,472.67`** | **`INR 13,094.74`** | **`INR 10,138.31`** | **`INR 14,270.09`** |
| **8. Execution Slippage (`7.5 bps` One-Way)** | **`INR 2,468.23`** | **`INR 4,083.62`** | **`INR 5,581.52`** | **`INR 4,209.68`** | **`INR 6,244.65`** |
| **Total All Transaction Costs (`INR`)** | **`INR 9,915.45`** | **`INR 13,556.29`** | **`INR 18,676.26`** | **`INR 14,347.99`** | **`INR 20,514.74`** |
| **1. Gross Strategy Return (Before All Costs)** | **`+17.35%`** | **`+39.71%`** | **`+29.65%`** | **`+27.27%`** | **`+8.22%`** |
| **2. Return After Statutory Charges Only** | `+9.90%` | `+30.23%` | `+16.55%` | `+17.13%` | `-6.05%` |
| **3. Return After Slippage Only** | `+14.88%` | `+35.62%` | `+24.07%` | `+23.06%` | `+1.98%` |
| **4. Net Return After All Costs** | **`+7.43%`** | **`+26.15%`** | **`+10.97%`** | **`+12.92%`** | **`-12.29%`** |
| **5. Omitted Ex-Date Cash Dividends (`INR`)** | **`INR 6,104.43`** | **`INR 10,760.65`** | **`INR 8,121.40`** | **`INR 3,242.65`** | **`INR 4,700.63`** |
| **6. Net Return If Omitted Dividends Credited** | **`+13.54%`** | **`+36.91%`** | **`+19.09%`** | **`+16.16%`** | **`-7.59%`** |
| **7. Average Cost per Round-Trip Trade (`INR`)** | `INR 41.66` | `INR 63.05` | `INR 61.64` | `INR 58.56` | `INR 65.13` |
| **8. Cost as % of One-Way Turnover (`bps`)** | `0.3013%` (`30.13 bps`) | `0.2490%` (`24.90 bps`) | `0.2510%` (`25.10 bps`) | `0.2556%` (`25.56 bps`) | `0.2464%` (`24.64 bps`) |
| **9. Cost as % of Starting Capital** | **`9.92%`** | **`13.56%`** | **`18.68%`** | **`14.35%`** | **`20.51%`** |

### 3.3 Forensic Verification of Tax Base & Cost Deduction Mechanics
- **Deducted Exactly Once (`VERIFIED`):** In `compute_repaired_delivery_costs` ([`moreshwar_v2_2.py`](file:///home/antigravity/Parth2004124/Finsistant-version-2/finsistant_v2/moreshwar_v2_2.py#L137-L170)), `exec_notional = raw_notional * (1 ± 0.00075)` embeds the `7.5 bps` slippage directly into `tot_buy = exec_notional + regulatory_total` and `net_proc = exec_notional - regulatory_total`. `slippage_cost` is tracked separately in `total_cost` for reporting and is **never double-deducted** from `cash` (`reconciliation_diff = INR 0.00`).
- **Taxable Base (`VERIFIED`):** STT (`0.10%` buy & sell), Stamp Duty (`0.015%` buy-only), Exchange Txn (`0.00297%`), and SEBI fees (`0.0001%`) are assessed on `exec_notional`, and 18% GST is assessed strictly on `brokerage + exchange_txn + sebi_fee` (and included in `INR 15.93 = 13.50 * 1.18` CDSL DP fee).
- **Architectural Discrepancy vs. Prompt 8 Tariff Engine:** `strategy_base.py` and `portfolio_construction.py` did not wire in Prompt 8's `HistoricalTariffSchedule` (which charges `0.00325%` NSE exchange txn fee prior to `2024-10-01`), understating pre-Oct-2024 exchange + GST fees by `INR 6.27` on `PORT_RISK_BUDGETED`, and used a flat `7.5 bps` slippage instead of Prompt 8's ADV square-root market-impact curve.

---

## 4. Outlier Dependence & Profit Concentration Counterfactuals

| Diagnostic Counterfactual | `PORT_RISK_BUDGETED` (Uninvested `+7.43%`) | `STRAT_E_DEFENSIVE_YIELD_LOWVOL` (`+26.15%`) | `STRAT_A_CS_MOMENTUM` (`+10.97%`) |
| :--- | :---: | :---: | :---: |
| **Baseline Net Profit / Return** | `INR +7,434.37` (`+7.43%`) | `INR +26,149.29` (`+26.15%`) | `INR +10,971.57` (`+10.97%`) |
| **Remove Top-1 Winning Trade** | **`+3.41%`** (Removed `INR 4,024.77`) | **`+19.58%`** (Removed `INR 6,572.22`) | **`+5.92%`** (Removed `INR 5,046.58`) |
| **Remove Top-3 Winning Trades** | **`-3.27%`** (Removed `INR 10,701.74`) | **`+8.88%`** (Removed `INR 17,272.56`) | **`-3.18%`** (Removed `INR 14,150.21`) |
| **Remove Top-5 Winning Trades** | **`-9.32%`** (Removed `INR 16,751.12`) | **`-0.26%`** (Removed `INR 26,411.31`) | **`-11.53%`** (Removed `INR 22,506.57`) |
| **Remove Top-10 Winning Trades** | **`-21.74%`** (Removed `INR 29,172.97`) | **`-17.66%`** (Removed `INR 43,810.79`) | **`-29.34%`** (Removed `INR 40,311.17`) |
| **Remove Largest Winning Stock** | **`+2.99%`** (`M&M`: `INR 4,446.84`) | **`+19.52%`** (`ASHOKLEY`: `INR 6,629.03`) | **`+3.81%`** (`M&M`: `INR 7,157.77`) |
| **Remove Best Calendar Month** | **`-1.54%`** (`2023-12`: `+9.11%`) | **`+17.94%`** (`2024-04`: `+6.96%`) | **`-0.04%`** (`2022-08`: `+11.02%`) |
| **Remove Best Calendar Quarter** | **`-4.70%`** (`2023-Q4`: `+12.73%`) | **`+13.11%`** (`2024-Q2`: `+11.53%`) | **`-8.07%`** (`2022-Q3`: `+20.71%`) |
| **Remove Best Calendar Year** | **`-8.39%`** (`2023`: `+17.27%`) | **`+12.03%`** (`2023`: `+12.61%`) | **`-1.26%`** (`2023`: `+12.39%`) |

- **Outlier Forensic Verdict:** In `PORT_RISK_BUDGETED`, the top 3 winning trades alone (`INR 10,701.74`) exceed 100% of total net profit (`INR 7,434.37`), so removing the top 3 trades flips the portfolio to **`-3.27%`** (and removing top 5 flips it to **`-9.32%`**). Even in `STRAT_E_DEFENSIVE_YIELD_LOWVOL`, removing the top 5 trades (`INR 26,411.31`) flips net return from `+26.15%` to **`-0.26%`**. Why? Because the remaining trades suffer from a truncated right tail (`STOP_LOSS_BE` and periodic `REBALANCE_ROTATE` cutting winners early) combined with `INR 9,915.45`–`INR 13,556.29` in delivery friction.

---

## 5. Benchmark Reconstruction & Fairness Audit (`2021-07-01 -> 2026-11-19` Full Window)

Complete benchmark and strategy metrics across all 3 windows are saved in [`BENCHMARK_RECONSTRUCTION.csv`](file:///home/antigravity/Parth2004124/Finsistant-version-2/reports/moreshwar_3_1/forensic_audit/BENCHMARK_RECONSTRUCTION.csv) and [`STRATEGY_PERFORMANCE_METRICS.csv`](file:///home/antigravity/Parth2004124/Finsistant-version-2/reports/moreshwar_3_1/forensic_audit/STRATEGY_PERFORMANCE_METRICS.csv).

| Entity / Benchmark | Type | Total Return | CAGR | Ann Vol | Max DD | Sharpe (6.5% RF) | Sortino | Calmar | Exposure % | Max Recovery Days | Rolling 12M Median |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Benchmark A1: 100% Nifty 50 Price Return Index (Theoretical)** | `BENCHMARK_THEORETICAL` | **`+62.72%`** | `+9.66%` | `13.73%` | `17.23%` | `+0.27` | `+0.26` | `0.56` | `100.0%` | `314d` | `+7.65%` |
| **Benchmark A2: 100% Nifty 50 TRI (Prompt 9 +1.25% p.a. Dividend Accrual)** | `BENCHMARK_THEORETICAL` | **`+73.82%`** | `+11.03%` | `13.73%` | `16.54%` | `+0.36` | `+0.35` | `0.67` | `100.0%` | `265d` | `+9.01%` |
| **Benchmark A3: 100% Nifty 50 TRI (Empirical Top-50 Ex-Date Dividends)** | `BENCHMARK_THEORETICAL` | **`+75.47%`** | `+11.23%` | `13.72%` | `16.32%` | `+0.37` | `+0.36` | `0.69` | `100.0%` | `264d` | `+9.17%` |
| **Benchmark A4: Executable 100% Nifty BeES ETF (0.04% TER + Entry Friction)** | `BENCHMARK_EXECUTABLE` | **`+75.10%`** | `+11.19%` | `13.72%` | `16.35%` | `+0.37` | `+0.36` | `0.68` | `100.0%` | `264d` | `+9.13%` |
| **Benchmark B1: 45% Nifty 50 TRI + 55% Cash (Prompt 9 Zero-Cost Daily Rebal)** | `BENCHMARK_THEORETICAL` | **`+56.82%`** | `+8.89%` | `6.18%` | `6.32%` | `+0.36` | `+0.35` | `1.41` | `45.0%` | `206d` | `+7.98%` |
| **Benchmark B2: Executable 45% Nifty BeES + 55% Liquid BeES (Annual Rebal + TER)** | `BENCHMARK_EXECUTABLE` | **`+54.93%`** | `+8.64%` | `6.23%` | `6.34%` | `+0.32` | `+0.31` | `1.36` | `45.0%` | `206d` | `+7.74%` |
| **Benchmark C1: 100% Risk-Free Cash (6.5% Linear Daily Accrual r=0.065/252)** | `BENCHMARK_THEORETICAL` | **`+40.95%`** | `+6.72%` | `0.00%` | `0.00%` | `+0.00` | `+0.00` | `99.00` | `0.0%` | `0d` | `+6.72%` |
| **Benchmark C2: 100% Risk-Free Cash (6.5% Effective Annual Rate (1.065)^(t/252))** | `BENCHMARK_THEORETICAL` | **`+39.46%`** | `+6.50%` | `0.00%` | `0.00%` | `+0.00` | `+0.00` | `99.00` | `0.0%` | `0d` | `+6.50%` |
| **M3.0 Shared Portfolio (PORT_RISK_BUDGETED, Uninvested Idle Cash)** | `STRATEGY_PORTFOLIO` | **`+7.43%`** | `+1.37%` | `8.64%` | `16.50%` | `-0.55` | `-0.52` | `0.08` | `45.0%` | `531d` | `+0.15%` |
| **M3.0 Shared Portfolio (PORT_RISK_BUDGETED + 6.5% Idle Cash Sweep)** | `STRATEGY_PORTFOLIO` | **`+29.76%`** | `+5.06%` | `8.64%` | `12.75%` | `-0.14` | `-0.12` | `0.40` | `45.0%` | `531d` | `+3.75%` |
| **M3.0 Adaptive Strategy Orchestrator (Prompt 7 Sleeve Return Blend)** | `STRATEGY_ORCHESTRATOR` | **`+14.83%`** | `+2.93%` | `7.99%` | `14.03%` | `-0.41` | `-0.35` | `0.21` | `72.7%` | `530d` | `+1.72%` |
| **Family E: Defensive PIT Dividend-Yield & Low-Volatility Compounder** | `STRATEGY_STANDALONE` | **`+26.15%`** | `+4.50%` | `10.45%` | `22.58%` | `-0.15` | `-0.15` | `0.20` | `63.0%` | `580d` | `+5.16%` |
| **Family A: Cross-Sectional 52W-High & Validated Factor Momentum** | `STRATEGY_STANDALONE` | **`+10.97%`** | `+1.99%` | `12.45%` | `24.89%` | `-0.30` | `-0.29` | `0.08` | `62.2%` | `530d` | `+4.23%` |
| **Family B: Volatility-Contraction & Volume/Delivery Breakout** | `STRATEGY_STANDALONE` | **`+12.92%`** | `+2.33%` | `13.95%` | `25.51%` | `-0.23` | `-0.22` | `0.09` | `67.4%` | `620d` | `-0.86%` |
| **Family C: Primary-Uptrend ATR Pullback Mean-Reversion** | `STRATEGY_STANDALONE` | **`-12.29%`** | `-2.45%` | `11.18%` | `39.85%` | `-0.75` | `-0.70` | `-0.06` | `67.9%` | `530d` | `+7.26%` |
| **Family D: Causal PIT Sector-First Leadership Rotation** | `STRATEGY_STANDALONE` | **`+5.56%`** | `+1.03%` | `14.55%` | `36.69%` | `-0.30` | `-0.31` | `0.03` | `50.8%` | `688d` | `-4.98%` |

### Is the Moreshwar 3.0 Benchmark Comparison Fair?
1. **Major Dividend Asymmetry (`UNFAIR TO STRATEGIES` by `+6.10 pp` in Portfolio and `+10.76 pp` in Family E):** Prompt 9 credits `+1.25% p.a.` continuous dividends to Nifty 50 (`+73.82%` synthetic TRI vs `+62.72%` Price Return), while crediting **0% cash dividends** to Moreshwar 3.0 equity holdings across ex-dividend dates (and even allowing ex-dividend price drops to trigger false stop-losses on `IDFC`, `COALINDIA`, `VEDL`, and `OFSS`). Empirical top-50 ex-date dividend reinvestment yields **`+75.47%`** (`+75.10%` executable Nifty BeES after TER & entry cost), showing Prompt 9's synthetic `+1.25% p.a.` Nifty TRI assumption was actually conservative by `1.28 pp` vs executable Nifty BeES.
2. **Executable vs. Theoretical 45/55 Benchmark (`+1.89 pp` Friction Gap):** Prompt 9's zero-cost daily-rebalanced `45% Nifty TRI + 55% Cash` (`+56.82%`) overstates an executable annually-rebalanced `45% Nifty BeES + 55% Liquid BeES` portfolio (`+54.93%`) by `1.89 pp` due to TER, entry friction, and effective-annual cash compounding.
3. **Bottom Line on Benchmark Fairness:** Even after fully correcting the dividend omission AND comparing against strictly executable ETFs (`+75.10%` Nifty BeES; `+54.93%` 45/55 Nifty+Liquid BeES), **Moreshwar 3.0 (`+35.86%` w/ cash sweep + omitted dividends; Family E `+36.91%` w/ omitted dividends) still underperforms both the 45/55 passive benchmark by `18–19 pp` and 100% Nifty BeES by `38–39 pp`**.
