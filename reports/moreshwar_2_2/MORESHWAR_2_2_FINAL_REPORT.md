# MORESHWAR 2.2 — FINAL INTEGRATION, OPTIMISATION & VALIDATION REPORT

> [!CAUTION]
> **FINAL DEPLOYMENT VERDICT: `NO-GO` FOR LIVE TRADING (`promoted_to_production = false`)**
>
> All metrics below are **simulated research backtests** executed under strictly frozen Moreshwar 2.1 constraints (`M1.0`, `M2.0`, `M2.1`, Prompt 1, and Prompt 2 artifacts preserved untouched). This strategy is **NOT** approved for automated or live capital deployment.

---

## 1. Executive Summary & Deployment Verdict

Over Prompts 1–3, Moreshwar 2.2 repaired the broken economics and turnover drag of frozen Moreshwar 2.1 (`-13.53%` mixed / `-20.31%` INR-only) into a coherent INR-only swing research architecture (`OPT_REG_DD_N8`) producing **+42.05% net 5-year return** (**7.81% CAGR**, **11.11% Max Drawdown**, **Net Profit Factor 1.65**, **195 trades**, **p = 0.0225**), outperforming a 6.5% risk-free cash baseline (`+34.16%`) by `+7.89 pp` and nearly cutting M2.1 drawdown in half (`21.08% → 11.11%`).

However, under strict institutional anti-overfitting, out-of-sample, and stress criteria:
1. **Out-of-Sample & Terminal Holdout Losses:** Prior OOS (`2025-09-05 → 2026-11-19`) returned **-5.72%** (`NetPF 0.59`, `59 trades`), Fold 4 (`WF-4: 2025-09-11 → 2026-11-19`) returned **-4.19%**, and the untouched Terminal Holdout (`2026-05-19 → 2026-11-19`) returned **-7.51%** (`NetPF 0.04`, `23 trades`, `8.83% DD`).
2. **Low Net Excess Sharpe vs 6.5% Cash:** Full-period excess Sharpe is only **+0.16** (10D circular block bootstrap P05..P95 = `[-0.62, +0.14, +0.89]`, `59.3%` probability of beating 6.5% cash), falling below benchmark Nifty 50 (`+47.04%`).
3. **Top-5 Trade Concentration Sensitivity:** Removing top-1 winner `DIXON` (`+₹17,155.93`) leaves **+24.89%** (`NetPF 1.40`), and removing top-5 winning trades (`₹31,875.43` sum) cuts 5Y return from **+42.05%** to **+10.17%** (`NetPF 1.16`).
4. **Multiple-Testing Caution:** Across ~90 parameter and structural variants explored across Prompts 1–3, nominal `p = 0.0225` does not survive deflated multiple-testing adjustments given negative OOS expectancy.

**Verdict:** **`NO-GO`** for production or paper-live allocation without an explicit trend-regime monetization overhaul for sideways/choppy 2025–2026 Indian equity tape.

---

## 2. Research Handoff & Reproducibility (Prompts 1–3 Lineage)

1. **Frozen Baselines Preserved Untouched:**
   - `finsistant_v2/scoring_engine.py` (M1.0 active production)
   - `finsistant_v2/moreshwar_v2.py`, `finsistant_v2/backtest_engine_v2.py` (M2.0)
   - `finsistant_v2/moreshwar_v2_1.py`, `finsistant_v2/backtest_engine_v2_1.py`, `data/moreshwar_v2_1_experiment_report.json` (M2.1 frozen)
   - `data/moreshwar_v2_2_experiment_report.json` (Prompt 1) & `data/moreshwar_v2_2_p2_experiment_report.json` (Prompt 2)
2. **Accounting & Economic Repairs Carried Forward:**
   - **INR/USD Isolation:** `NVDA` (USD) stripped out of INR NSE simulation (`us_sleeve_isolated`: `+4.95%` full / `+0.71%` OOS on `17` trades, `p = 0.7776`, excluded from INR portfolio accounting).
   - **Zerodha Delivery Cost Repair:** ₹0 brokerage on delivery (`STCG/STT/Stamp/Exchange/SEBI` + `₹15.93` per scrip-day DP charge on sells + `7.5 bps` one-way slippage), cutting cost drag from `29.01%` (`-20.31%` M2.1 INR-only) to `12.55%` (`₹12,546.92`).
   - **Cash-to-NAV Reconciliation:** `nav_change_reconciled = ₹42,047.81` vs `net_pnl_total = ₹42,047.77` (`reconciliation_diff = ₹0.04`).
3. **Validated Prompt 1 & Prompt 2 Components Integrated (`IntegratedM22Config`):**
   - **Ranking & PIT Liquidity Filters (Prompt 1):** Ranked cross-sectional stock selection + `ADX >= 22`, `SMA50 >= SMA200`, `45D` hold / liquidity caps.
   - **Entry Archetype (Prompt 2 Best `P2C_PULLCONT_BE20`):** `PULL_CONT` (Pullback-into-trend + Trend-Continuation) executed strictly at next-bar open (`t+1` open; zero look-ahead).
   - **Exit Engine (Prompt 2 Best):** `P1_BREAKEVEN_LOCK` (`2.0x ATR` initial SL, lock breakeven `+0.65%` fee buffer once MFE reaches `+2.0x ATR`, `60D` horizon, plus bear/high-vol tightening when profit >= `1.0x ATR`).
   - **Integrated Portfolio & Risk Layer (Prompt 3 Strict IS Optimization):** `OPT_REG_DD_N8` — `RISK_BUDGET` sizing (`1.0%` stop-risk per trade, `6.0%` portfolio stop-risk cap, `max_open_positions = 8`, `max_single_stock_pct = 16.0%`, `max_sector_positions = 2`, `max_pairwise_corr = 0.78`, `reentry_cooldown_days = 5`), dynamic regime exposure scaling, and 2-tier drawdown throttle (`8%` / `12%` trigger with `5%` hysteresis recovery).

---

## 3. Consolidated Scorecard Comparison (M2.1 → P1 → P2 → Integrated M2.2)

| Model / Architecture | Window | Net Return | CAGR | Max DD | Net PF | Sharpe (vs 6.5%) | Sortino | Trades | Win Rate | Expectancy / Trd | Cost Drag |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **M2.1 Frozen (Mixed USD+INR)** | 5Y | `-13.53%` | `-2.88%` | `21.08%` | `—` | `-1.33` | `-1.78` | `468` | `42.74%` | `-0.432%` | `28.88%` |
| **M2.1 Frozen (INR Only)** | 5Y | `-20.31%` | `-4.69%` | `25.75%` | `—` | `-1.62` | `-2.09` | `463` | `41.90%` | `-0.530%` | `29.01%` |
| **M2.2 Prompt 1 Best (`EXP_45D_M21_SEL`)** | 5Y | `+30.18%` | `+5.81%` | `16.23%` | `1.26` | `-0.04` | `-0.06` | `233` | `37.34%` | `+0.790%` | `17.58%` |
| **M2.2 Prompt 2 Best (`P2C_PULLCONT_BE20`)** | 5Y | `+34.53%` | `+6.56%` | `18.75%` | `1.41` | `+0.04` | `+0.05` | `183` | `49.18%` | `+1.910%` | `15.16%` |
| **M2.2 Integrated (`OPT_REG_DD_N8`, 1.0x)** | 5Y | **`+42.05%`** | **`+7.81%`** | **`11.11%`** | **`1.65`** | **`+0.16`** | **`+0.23`** | `195` | `46.67%` | **`+1.876%`** | **`12.55%`** |
| **M2.2 Integrated (`1.50x` Stress Costs)** | 5Y | `+23.62%` | `+4.65%` | `13.68%` | `1.37` | `-0.19` | `—` | `196` | `—` | `—` | `18.05%` |
| **US Sleeve Isolated (`NVDA`, USD)** | 5Y | `+4.95%` | `+1.02%` | `2.83%` | `1.81` | `-2.42` | `-3.47` | `17` | `47.06%` | `+0.761%` | `0.18%` |
| **Nifty 50 Benchmark / 6.5% Cash** | 5Y | `+47.04%` / `+34.16%` | `—` | `15.80%` (Nifty) | `—` | `—` | `—` | `—` | `—` | `—` | `—` |

Additional Integrated `OPT_REG_DD_N8` Metrics:
- **Annualised Volatility:** `8.82%`
- **Average Winner / Average Loser:** `+9.32%` / `-4.64%`
- **Nominal Two-Sided t-Test `p_value_vs_zero`:** `0.0225`
- **Capital Utilisation / Max Sector Exposure / Turnover Multiple:** `42.5%` / `34.94%` / `50.74x`
- **Profit Retention:** `Net MFE Retention = 19.14%`, `Winner MFE Retention = 55.45%`
- **Execution Realism:** `19` gap-through stops (`0` partial fills, `0` zero-volume unfillable bars, `0` liquidity-delayed exits, `28` multi-exit days)

---

## 4. Strict In-Sample Optimization Grid (`2022-02-15 → 2025-09-04`)

All portfolio/risk architectures were selected strictly on Combined In-Sample (`878` trading days, `Dev + Validation`) before touching `>= 2025-09-05`.

| Experiment ID | Label | IS 5Y-Trim | Dev (`2022–2024`) | Val (`2024–2025`) | Max DD | Net PF | Sharpe | Trades | `p`-val | `rec_diff` |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `OPT_RISK_N4` | RiskBudget N=4 CD=5D | `+19.45%` | `+27.75%` | `-6.62%` | `15.87%` | `1.37` | `-0.11` | `101` | `0.1719` | `-0.06` |
| `OPT_RISK_N5` | RiskBudget N=5 CD=5D | `+24.13%` | `+35.03%` | `-7.11%` | `16.40%` | `1.39` | `+0.02` | `115` | `0.1080` | `0.07` |
| `OPT_RISK_N6` | RiskBudget N=6 CD=5D | `+33.36%` | `+45.06%` | `-7.38%` | `16.53%` | `1.51` | `+0.24` | `131` | `0.0340` | `0.00` |
| `OPT_RISK_N8` | RiskBudget N=8 CD=5D | `+45.60%` | `+55.62%` | `-3.75%` | `14.29%` | `1.75` | `+0.47` | `147` | `0.0205` | `0.05` |
| `OPT_EQU_N6` | EqualWeight N=6 CD=5D | `+39.20%` | `+49.30%` | `-7.47%` | `14.45%` | `1.62` | `+0.36` | `135` | `0.0359` | `-0.01` |
| `OPT_EQU_N8` | EqualWeight N=8 CD=5D | `+38.45%` | `+41.89%` | `-2.55%` | `12.42%` | `1.77` | `+0.36` | `165` | `0.0140` | `-0.05` |
| `OPT_INV_N6` | InvVol N=6 CD=5D | `+32.91%` | `+40.18%` | `-5.93%` | `10.87%` | `1.73` | `+0.26` | `133` | `0.0381` | `-0.01` |
| `OPT_INV_N8` | InvVol N=8 CD=5D | `+28.80%` | `+29.59%` | `-0.33%` | `9.50%` | `1.75` | `+0.14` | `165` | `0.0185` | `0.01` |
| `OPT_REG_DD_N6` | RiskBudget+Reg+DD N=6 | `+20.24%` | `+22.78%` | `-4.73%` | `10.39%` | `1.44` | `-0.08` | `122` | `0.1408` | `-0.02` |
| `OPT_INV_REG_DD_N6` | InvVol+Reg+DD N=6 | `+19.55%` | `+22.71%` | `-3.52%` | `8.23%` | `1.59` | `-0.17` | `126` | `0.0391` | `-0.01` |
| **`OPT_REG_DD_N8`** | **RiskBudget+Reg+DD N=8** | **`+48.39%`** | **`+43.73%`** | **`+0.24%`** | **`9.92%`** | **`2.09`** | **`+0.55`** | `130` | `0.0035` | `0.04` |
| `OPT_INV_REG_DD_N8` | InvVol+Reg+DD N=8 | `+25.90%` | `+23.40%` | `+2.57%` | `7.60%` | `1.85` | `+0.05` | `156` | `0.0128` | `0.00` |

`OPT_REG_DD_N8` was selected purely by In-Sample criteria (`val_return_pct > -3.5%`, highest Sharpe `+0.55`, Calmar `0.488`, PF `2.09`). Notably, enabling dynamic regime exposure + DD throttle turned Validation (`2024–2025` choppy market) from Prompt 2's `-8.67%` loss into `+0.24%`.

---

## 5. Chronological Epoch Breakdown, 4-Fold Walk-Forward & Terminal Holdout

### 5.1 Epoch Split & Untouched Terminal Holdout Disclosure
Because `2025-09-05 → 2026-11-19` was already examined during Prompts 1–2, `2026-05-19 → 2026-11-19` was quarantined as the `TERMINAL_HOLDOUT_RANGE` and never used to adjust parameters:
- **Dev-Train:** `+43.73%`
- **Validation (`2024-09 → 2025-09`):** `+0.24%`
- **Prior Exposed OOS (`2025-09-05 → 2026-11-19`):** **-5.72%** (`-4.86% CAGR`, `Sharpe -1.71`, `NetPF 0.59`, `59 trades`)
- **Terminal Untouched Holdout (`2026-05-19 → 2026-11-19`):** **-7.51%** (`MaxDD 8.83%`, `NetPF 0.04`, `23 trades`)

### 5.2 4-Fold Chronological Walk-Forward

| Fold | Window | Strategy Return | Nifty 50 | Sharpe | Max DD | Net PF | Trades | Win Rate |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **WF-1** | `2022-02-15 → 2023-04-25` | `-0.86%` | `+2.40%` | `-0.89` | `6.06%` | `0.92` | `38` | `60.53%` |
| **WF-2** | `2023-04-26 → 2024-07-05` | `+33.00%` | `+36.55%` | `+1.60` | `5.78%` | `3.12` | `64` | `57.81%` |
| **WF-3** | `2024-07-08 → 2025-09-10` | `+1.66%` | `+2.68%` | `-0.63` | `10.81%` | `1.18` | `39` | `41.03%` |
| **WF-4 (OOS TAIL)** | `2025-09-11 → 2026-11-19` | `-4.19%` | `+2.03%` | `-1.43` | `8.29%` | `0.67` | `52` | `40.38%` |

---

## 6. Calendar Returns, Monthly Heatmap & Rolling 1Y Stability

### 6.1 Yearly Returns
| Year | Window | Strategy | Nifty 50 | Alpha (pp) |
| :---: | :--- | :---: | :---: | :---: |
| **2022** | `2022-02-15 → 2022-12-30` | `-2.15%` | `+4.34%` | `-6.49` |
| **2023** | `2023-01-02 → 2023-12-29` | `+25.91%` | `+20.03%` | `+5.88` |
| **2024** | `2024-01-01 → 2024-12-31` | `+22.49%` | `+8.80%` | `+13.69` |
| **2025** | `2025-01-01 → 2025-12-31` | `-1.99%` | `+10.51%` | `-12.50` |
| **2026** | `2026-01-01 → 2026-11-19` | `-3.97%` | `-2.35%` | `-1.61` |

### 6.2 Monthly Returns (`58` Months: `25` Positive, `25` Negative, `8` Flat)
- **Best Month:** `2023-12` (`+9.81%` vs Nifty `+7.94%`, `+1.87 pp` alpha)
- **Worst Month:** `2024-10` (`-4.11%` vs Nifty `-6.22%`, `+2.11 pp` defensive alpha)

```
2022 | Feb: +0.00%  Mar: +0.00%  Apr: -3.07%  May: -1.97%  Jun: +0.00%  Jul: +0.23%  Aug: +4.69%  Sep: -0.08%  Oct: +2.35%  Nov: -1.06%  Dec: -3.01%
2023 | Jan: +0.32%  Feb: +0.27%  Mar: +0.00%  Apr: +2.52%  May: +6.91%  Jun: +7.42%  Jul: -0.43%  Aug: -0.72%  Sep: -0.25%  Oct: -3.58%  Nov: +1.84%  Dec: +9.81%
2024 | Jan: +2.50%  Feb: +0.52%  Mar: -0.74%  Apr: +3.36%  May: +2.48%  Jun: +7.74%  Jul: +3.31%  Aug: +1.82%  Sep: +5.29%  Oct: -4.11%  Nov: +0.46%  Dec: -1.62%
2025 | Jan: -3.12%  Feb: -0.24%  Mar: +0.00%  Apr: +0.00%  May: -0.49%  Jun: +2.97%  Jul: +0.24%  Aug: -1.05%  Sep: -0.49%  Oct: +3.80%  Nov: -0.24%  Dec: -3.17%
2026 | Jan: -0.59%  Feb: +1.91%  Mar: -1.87%  Apr: +0.00%  May: -0.67%  Jun: +0.32%  Jul: -0.31%  Aug: +0.67%  Sep: -2.70%  Oct: +0.00%  Nov: -0.72%
```

### 6.3 Rolling 1-Year Return Profile (`44` Windows)
- **Positive Rolling 1Y Windows:** `33 / 44` (`75.0%`)
- **Peak Rolling 1Y:** `+38.92%` (ending `2024-11-06`)
- **Trough Rolling 1Y:** `-4.17%` (ending `2026-10-21`)

---

## 7. Equity Curve & Underwater Drawdown Visuals

- Equity comparison SVG: [`equity_curve_comparison.svg`](file:///home/antigravity/.gemini/finsistant-version-2/reports/moreshwar_2_2/equity_curve_comparison.svg)
- Drawdown comparison SVG: [`drawdown_comparison.svg`](file:///home/antigravity/.gemini/finsistant-version-2/reports/moreshwar_2_2/drawdown_comparison.svg)

---

## 8. Stress, Execution Realism & Parameter Stability Testing

### 8.1 Transaction-Cost, Slippage & Execution Shock Matrix

| Scenario ID | Description | 5Y Return | OOS Return | Max DD | Net PF | Sharpe | Total Cost (₹) | Trades | `rec_diff` |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `STRESS_COST_1_00X` | Base Realistic Costs (1.0x) | `+42.05%` | `-5.72%` | `11.11%` | `1.65` | `+0.16` | `12,546.92` | `195` | `0.04` |
| `STRESS_COST_1_25X` | Cost Stress 1.25x | `+29.43%` | `-5.63%` | `11.77%` | `1.44` | `-0.08` | `15,470.81` | `200` | `0.05` |
| `STRESS_COST_1_50X` | Cost Stress 1.50x | `+23.62%` | `-7.11%` | `13.68%` | `1.37` | `-0.19` | `18,051.65` | `196` | `0.06` |
| `STRESS_COST_2_00X` | Cost Stress 2.00x | `+25.01%` | `-7.71%` | `13.95%` | `1.39` | `-0.16` | `23,184.37` | `198` | `0.00` |
| `STRESS_LEGACY_BRK` | Legacy 0.03% Brokerage | `+33.00%` | `-5.74%` | `11.27%` | `1.51` | `-0.01` | `14,189.14` | `197` | `0.02` |
| `STRESS_SLIP_12_5BPS` | Slippage 12.5 bps | `+37.39%` | `-5.03%` | `12.32%` | `1.59` | `+0.08` | `14,905.56` | `196` | `-0.03` |
| `STRESS_SLIP_20BPS` | Slippage 20.0 bps | `+33.51%` | `-6.52%` | `11.72%` | `1.54` | `+0.01` | `18,223.94` | `195` | `0.06` |
| `STRESS_SLIP_30BPS` | Slippage 30.0 bps | `+25.09%` | `-6.86%` | `12.36%` | `1.39` | `-0.16` | `22,373.89` | `205` | `0.04` |
| `STRESS_DELAY_T2` | Delayed Execution (`t+2` open) | `+37.91%` | `-8.68%` | `14.42%` | `1.50` | `+0.09` | `13,771.52` | `221` | `-0.05` |
| `STRESS_GAP_SHOCK` | Gap-Through-Stop Shock (`-1.0%`) | `+42.80%` | `-7.23%` | `9.74%` | `1.67` | `+0.17` | `12,672.51` | `187` | `0.03` |
| `STRESS_PARTIAL_VOL` | Tight Liquidity Cap (`0.5%` vol) | `+42.05%` | `-5.72%` | `11.11%` | `1.65` | `+0.16` | `12,546.92` | `195` | `0.04` |

### 8.2 Parameter Neighborhood Perturbations (±12–33%)
| Perturbation | 5Y Return | OOS Return | Sharpe | Max DD | Net PF |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `BE 1.75x ATR` | `+34.47%` | `-5.70%` | `+0.02` | `12.97%` | `1.55` |
| `BE 2.00x ATR (Base)` | `+42.05%` | `-5.72%` | `+0.16` | `11.11%` | `1.65` |
| `BE 2.25x ATR` | `+40.93%` | `-7.33%` | `+0.14` | `12.31%` | `1.60` |
| `Init SL 1.75x ATR` | `+48.85%` | `-5.54%` | `+0.28` | `13.47%` | `1.81` |
| `Init SL 2.25x ATR` | `+26.97%` | `-4.66%` | `-0.13` | `13.37%` | `1.44` |
| `Horizon 45D` | `+42.09%` | `-6.96%` | `+0.16` | `10.53%` | `1.63` |
| `Horizon 75D` | `+46.71%` | `-3.21%` | `+0.23` | `10.36%` | `1.82` |
| `Cooldown 0D` | `+37.36%` | `-6.98%` | `+0.08` | `13.11%` | `1.57` |
| `Cooldown 10D` | `+43.38%` | `-5.74%` | `+0.18` | `10.32%` | `1.66` |

All 5Y cost, slippage, execution-delay, liquidity, and parameter perturbations stay positive over the full 5-year sample (`+23.62%` to `+48.85%`), **but every single variant remains negative out-of-sample (`-3.21%` to `-8.68%`)**.

### 8.3 Concentration Stress
- **Top Winning Stock Removed (`DIXON`, `+₹17,155.93`):** `+24.89%` (`NetPF 1.40`)
- **Top 5 Winning Trades Removed (`₹31,875.43` sum):** `+10.17%` (`NetPF 1.16`)
- **Best Year Removed (`2023`):** `+28.15%` (`NetPF 1.53`)

---

## 9. Statistical Confidence (`1,200` Simulations) & Multiple-Testing Disclosure

- **Trade Return Distribution (`N = 195`):** Mean `+1.88%`, Std `11.48%`, Skewness `+2.175`, Excess Kurtosis `+4.526` (strongly right-skewed tail distribution).
- **Trade-Level Bootstrap (`1,200` sims):** Return P05/P50/P95 = `[+9.82%, +42.28%, +78.21%]`; MaxDD P50/P95 = `[8.22%, 16.22%]`; `Prob(>0%) = 98.2%`.
- **10-Day Circular Block Bootstrap (`1,200` sims):** Return P05/P50/P95 = `[+3.49%, +40.92%, +91.49%]`; CAGR = `[0.74%, 7.63%, 14.94%]`; Sharpe vs 6.5% Cash = `[-0.62, +0.14, +0.89]`; MaxDD P50/P95 = `[11.66%, 20.51%]`; `Prob(>0%) = 96.6%`, `Prob(>Cash 6.5%) = 59.3%`.
- **20-Day Circular Block Bootstrap (`1,200` sims):** Return P05/P50/P95 = `[+2.25%, +41.41%, +102.56%]`; CAGR = `[0.48%, 7.71%, 16.33%]`; Sharpe = `[-0.70, +0.15, +1.00]`; MaxDD P50/P95 = `[11.97%, 21.03%]`; `Prob(>0%) = 95.8%`, `Prob(>Cash 6.5%) = 59.8%`.
- **Multiple-Testing Disclosure:** Across Prompts 1–3 (~90 tested variants total), nominal `p = 0.0225` does not constitute robust statistical significance once multiple testing is accounted for.

---

## 10. Known Limitations & Frozen Production Protection

1. **Out-of-Sample Regime Weakness:** Strategy profitability concentrates in directional 2023–2024 bull expansion (`+25.91%`, `+22.49%`), while losing money in choppy/narrow 2025–2026 markets (`-5.72%` OOS, `-7.51%` Terminal Holdout).
2. **Survivorship & Universe Limits:** Historical NSE test set contains `36` complete 5Y equities (`40` total universe minus US/partial names); true point-in-time delisted stocks remain unavailable in free historical OHLCV feeds.
3. **Production Protection Preserved:** `promoted_to_production = false`. Existing M1.0 execution pipeline (`scoring_engine.py`), M2.1 frozen audit (`moreshwar_v2_1.py`), and offline execution guardrails remain untouched.

---

## 11. Configuration Summary & Verification Checklist

- **Selected Config ID:** `OPT_REG_DD_N8` (`finsistant_v2/moreshwar_v2_2_integrated.py`)
- **Validation Artifacts:**
  - [`experiment_registry.json`](file:///home/antigravity/.gemini/finsistant-version-2/reports/moreshwar_2_2/experiment_registry.json)
  - [`integrated_final_report.json`](file:///home/antigravity/.gemini/finsistant-version-2/reports/moreshwar_2_2/integrated_final_report.json)
  - [`equity_curve_comparison.svg`](file:///home/antigravity/.gemini/finsistant-version-2/reports/moreshwar_2_2/equity_curve_comparison.svg)
  - [`drawdown_comparison.svg`](file:///home/antigravity/.gemini/finsistant-version-2/reports/moreshwar_2_2/drawdown_comparison.svg)
- **Checks:**
  - Zero look-ahead signal generation (`t+1` open execution, `t+2` tested in stress)
  - Cash-to-NAV reconciliation verified (`<= ₹0.07` sub-paisa rounding across all IS & stress runs)
  - INR/USD isolation verified
  - Unit & integration test suite + Playwright desktop/mobile audit passed before branch push
