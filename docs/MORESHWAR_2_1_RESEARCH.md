# Finsistant V2 — Moreshwar Mathematical Model 2.1: Robustness, Statistical Validation & Intelligence Upgrade Report

**Status:** `RESEARCH_ONLY_NOT_PROMOTED` (`promoted_to_production = False` — **Moreshwar 1.0 remains active in production pending user review; cloud performs scanning & research only, while broker execution stays isolated on local laptop.**)  
**Final Research Classification:** **`1. Research failure: No reliable edge demonstrated`**  
**Evaluation Dataset:** `2022-01-24` to `2026-11-19` (`1,252` aligned trading days · `40` multi-sector instruments + `^NSEI` benchmark · Untouched OOS tail `2025-09-05 → 2026-11-19`)  
**Execution & Cost Model:** `INDIAN_EQUITY_DELIVERY` (Next-bar open fill, `0.10%` buy & sell STT, `min(₹20, 0.03%)` brokerage, `0.00297%` NSE txn, `0.0001%` SEBI, `0.015%` buy stamp, `18%` GST, `₹15.93` sell DP charge, `7.5 bps` liquid / `20 bps` `<₹2.5Cr/day` illiquid slippage, gap-through-stop fills at open, `1%` daily volume cap, zero-volume halt gating, minimum viable delivery notional `₹1,500`)

---

## 1. Preserved Versions & Separation Guarantee
- **Moreshwar 1.0 (`finsistant_v2/scoring_engine.py`):** Completely untouched (`calculate_moreshwar_levels`).
- **Moreshwar 2.0 (`finsistant_v2/moreshwar_v2.py`, `finsistant_v2/backtest_engine_v2.py`, `data/moreshwar_v2_experiment_report.json`):** Completely untouched.
- **Moreshwar 2.1 (`finsistant_v2/moreshwar_v2_1.py`, `finsistant_v2/backtest_engine_v2_1.py`, `finsistant_v2/historical_data.py`, `data/moreshwar_v2_1_experiment_report.json`):** Isolated versioned modules and read-only research telemetry.

---

## 2. Historical Data Expansion & Honest Survivorship Audit (`historical_data.py`)
- **Benchmark `^NSEI` (Nifty 50):** `2021-11-22 → 2026-11-19` (`1,235` trading days / ~5 full years) covering post-2021 bull continuation, 2022 bear/choppy correction, 2023–2024 strong bull expansion, 2024–2025 sharp pullback & sideways volatility, and 2025–2026 OOS rotation.
- **Universe Breakdown (`40 / 40` loaded):**
  - `36` complete 5-year active Indian large/mid/small-cap equities + `NVDA` (`1,254` bars, USD isolated).
  - `3` distressed / insolvency survivorship controls (`YESBANK`, `JETAIRWAYS` with `501` zero-volume suspended days, `RCOM` with `294` zero-volume days, plus `ZEEL` drawdown stress test).
  - `1` partial post-demerger listing (`TMCV.NS` for `TATAMOTORS`, `253` bars, `79.6%` missing vs 5Y benchmark — transparently tagged, never fabricated).
- **Delisted Ticker Probes (Never Fabricated):** Real HTTP probes against `DHFL.NS`, `RELCAPITAL.NS`, `HDFC.NS`, `MINDTREE.NS`, and `SINTEX.NS` returned `5/5` HTTP `404` responses (Yahoo Finance purges delisted/merged tickers from public chart feeds). Documented openly in `data_quality_report.delisted_probes`.

---

## 3. Multi-Strategy 5-Year Comparison Matrix (Identical Window & Realistic Indian Delivery Costs)

| Strategy / Benchmark | Tot% | CAGR% | Win% (95% CI) | PF | Sharpe | Sortino | MaxDD% | Recov | Exp%/Trd (p vs 0) | Trades | Hold(D) | Util% | Cost% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Moreshwar 1.0 (Baseline)** | -41.50% | -10.23% | 36.36% `[33.52–39.31]` | 0.83 | -1.11 | -1.47 | 47.54% | -0.87 | -0.284% (`0.0886`) | 1,056 | 8.6 | 85.3% | 63.25% |
| **Moreshwar 2.0 (Clean Causal)** | -16.44% | -3.55% | 43.57% `[40.14–47.06]` | 0.91 | -0.93 | -1.24 | 25.30% | -0.65 | -0.264% (`0.1963`) | 785 | 9.6 | 60.1% | 44.77% |
| **Moreshwar 2.1 (6-Regime + Port Risk)** | **-13.53%** | **-2.88%** | 42.74% `[38.33–47.26]` | 0.88 | -1.33 | -1.78 | **21.08%** | -0.64 | -0.432% (`0.0712`) | 468 | 9.9 | 38.9% | **28.88%** |
| **Moreshwar 2.1 (Permissive Control)** | -16.47% | -3.56% | 42.26% `[38.29–46.34]` | 0.86 | -1.44 | -1.90 | 20.79% | -0.79 | -0.330% (`0.1602`) | 575 | 9.6 | 40.9% | 31.25% |
| **Moreshwar 2.1 + Static Fund (Leak Audit)** | -14.66% | -3.14% | 43.91% `[39.52–48.40]` | 0.88 | -1.13 | -1.56 | 20.29% | -0.72 | -0.232% (`0.3491`) | 476 | 9.7 | 42.9% | 29.29% |
| **Simple SMA-20 Strategy** | +3.92% | +0.78% | 45.75% `[42.63–48.90]` | 1.01 | -0.33 | -0.46 | 36.65% | 0.11 | -0.064% (`0.7388`) | 964 | 11.6 | 90.5% | 63.10% |
| **Fixed 5% Stop / 10% Target** | -5.71% | -1.18% | 45.00% `[41.91–48.13]` | 0.98 | -0.47 | -0.65 | 39.09% | -0.15 | +0.074% (`0.6847`) | 980 | 10.5 | 89.1% | 65.08% |
| **Nifty 50 (`^NSEI`) Buy & Hold** | **+48.78%** | **+8.78%** | 100.0% | — | **+0.21** | **+0.29** | **15.77%** | **3.09** | — | 1 | 1190 | 100% | 0% |
| **Risk-Free Cash (6.5% India T-Bill)** | **+38.11%** | **+6.50%** | 100.0% | — | 0.00 | 0.00 | **0.00%** | — | — | 0 | — | 0% | 0% |

---

## 4. Walk-Forward Validation, Untouched OOS & Monte Carlo Robustness

### Rolling 4-Fold Walk-Forward (Zero Tuning on Test Periods)
- **WF-1 (`2022-01-24 → 2023-04-06`, Nifty `+2.62%`):** M1.0 `-17.72%` (DD `26.65%`) · M2.0 `-12.31%` (DD `20.29%`) · **M2.1 `-7.00%` (`Sharpe -1.95`, DD `7.20%`, Win `39.56%`, `n=91`)**
- **WF-2 (`2023-04-10 → 2024-06-21`, Nifty `+33.35%`):** M1.0 `+22.44%` (DD `9.48%`) · M2.0 `+13.77%` (DD `11.19%`) · **M2.1 `+9.68%` (`Sharpe +0.16`, DD `8.73%`, Win `48.80%`, `n=166`)**
- **WF-3 (`2024-06-24 → 2025-09-04`, Nifty `+5.08%`):** M1.0 `-26.39%` (DD `29.11%`) · M2.0 `-14.27%` (DD `19.07%`) · **M2.1 `-9.60%` (`Sharpe -2.15`, DD `13.00%`, Win `35.51%`, `n=107`)**
- **WF-4 UNTOUCHED OOS (`2025-09-05 → 2026-11-19`, Nifty `+3.13%`):**
  - Moreshwar 2.1: **`-4.77%`** (`CAGR -3.86%`, `Sharpe -1.74`, `DD 13.04%`, `Win 38.94%`, `n=113`) vs M2.0 `-15.99%` (`DD 21.28%`) vs M1.0 `-22.86%` (`DD 30.14%`) vs SMA-20 `-8.14%` vs Fixed 5/10 `-5.47%` vs Nifty `+3.13%` vs Cash `+8.41%`.
- **Temporal Stability Check:** 1st-half avg `+1.34%` vs 2nd-half avg `-7.18%` (`deterioration_detected = True`).

### Monte Carlo Simulation (`1,200` Runs Each — Trade Bootstrap vs 10-Day Block Bootstrap)
| Model / Resampling Method | P5 Return | Median Return | P95 Return | Median MaxDD | P95 MaxDD | P(Ret > 0) | P(MaxDD ≥ 25%) | P(MaxDD ≥ 50%) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **M2.1 Trade Bootstrap (1,200 sims)** | -30.72% | -13.13% | +4.87% | 19.33% | 33.53% | 11.3% | 26.33% | **0.00%** |
| **M2.1 10-Day Block Bootstrap (1,200 sims)** | -33.20% | -14.23% | +12.10% | 22.29% | 36.95% | 19.0% | 39.00% | **0.00%** |
| **M1.0 Trade Bootstrap (1,200 sims)** | -72.05% | -41.85% | -9.41% | 47.44% | 74.50% | 2.0% | 92.58% | **44.33%** |
| **M1.0 10-Day Block Bootstrap (1,200 sims)** | -67.04% | -40.92% | +4.61% | 50.85% | 70.03% | 6.3% | 97.58% | **52.08%** |

*Limitation:* Trade bootstrap assumes i.i.d. PnLs (breaks multi-week regime clustering); circular 10-day block bootstrap preserves local serial correlation but cannot synthesize tail crashes worse than observed inside `2022–2026`.

---

## 5. Regime, Portfolio-Risk & Root-Cause Strategy Investigation

### 6-State Causal Nifty Regime Decomposition (`Moreshwar 2.1`)
- **`STRONG_BULLISH` (257 days, `162` trades):** `46.91%` win, **`+0.045%` expectancy/trade**, **`PF 1.09`**, **`Sharpe +2.61`**, `MaxDD 7.13%`.
- **`WEAK_BULLISH` (373 days, `244` trades):** `43.44%` win, `-0.382%` expectancy, `PF 0.84`, `Sharpe -0.96`, `MaxDD 13.63%`.
- **`SIDEWAYS` (54 days, `62` trades):** `29.03%` win, `-1.873%` expectancy, `PF 0.41`, `Sharpe -3.84`, `MaxDD 8.23%`.
- **`WEAK_BEARISH` (237d) / `STRONG_BEARISH` (143d) / `HIGH_VOLATILITY` (188d):** `0` trades entered (cleanly gated).

### Portfolio-Level Risk Upgrades (`Moreshwar 1.0 → Moreshwar 2.1`)
- **Max Simultaneous Positions:** `12` → **`6`** (avg `3.50` vs `6.67`)
- **Max Sector Concentration:** `71.44%` → **`36.20%`** (avg `13.79%` vs `35.82%`)
- **Peak Aggregate SL Exposure:** `13.53%` → **`5.82%`** (avg `1.82%` vs `5.25%`)
- **Single-Day / 5-Day Tail Loss:** `-4.38% / -7.32%` → **`-2.24% / -3.90%`**
- **Nifty Decline Days Exposure (121 drop days):** `-1.324%` avg (`71.5%` deployed) → **`-0.459%` avg (`30.66%` deployed)**

### Forensic Investigation Findings
1. **ATR Stops Work (`50.9% → 40.2%` SL rate):** Replacing `100 - X` fundamental compression eliminates noise whip-saws (`M2.0: 41.8%`, `M2.1: 40.2%`).
2. **Regime Filtering Works (`+2.94 pp` vs Permissive Control):** Blocking bearish/high-vol Nifty days improved 5Y return from `-16.47%` to `-13.53%`, and `M2.1` loses money primarily in `WEAK_BULLISH` (`-0.382%/trd`) and `SIDEWAYS` (`-1.873%/trd`) rather than `STRONG_BULLISH` (`+0.045%/trd`).
3. **E-Ref Static Leakage Proven (`r = +0.161`):** Today's fundamental scores correlate at Pearson `r = +0.161` with prior 4-year returns (`2021–2025`), proving why using today's financial ratios in a multi-year backtest leaks future winners/losers.
4. **Excessive Selectivity & Friction Drag (`-28.88%` cost drag vs `-13.53%` total return):** Counterfactual evaluation of `13,359` rejected setups averaged `-0.038%` (`44.79%` win) vs `468` accepted trades `-0.432%` (`42.74%` win); `UNFAVORABLE_RR` skips (`-0.521%`) and `ZERO_SHARES` skips (`-2.462%`) helped avoid bad trades, whereas gross technical edge before statutory STT/DP/slippage is insufficient to overcome `~₹28,880` (`28.88%`) in Indian delivery trading costs across `468` short-horizon (`9.9D`) trades.

---

## 6. Unresolved Risks & Production Stance
- **Why Production Promotion Is Rejected:** Even after state-of-the-art ATR sizing, 6-regime gating, pairwise correlation caps, and cutting M1.0's 50%-ruin risk from `52.08%` down to `0.00%`, **Moreshwar 2.1 still loses `-13.53%` net of real Indian delivery costs (`p = 0.0712`)**, trailing both Nifty 50 (`+48.78%`) and 6.5% Cash (`+38.11%`).
- **Recommendation:** Do **not** deploy Moreshwar 2.1 for automated capital allocation. Preserve it strictly as a research benchmark alongside Moreshwar 1.0/2.0 while collecting clean forward point-in-time fundamental audit snapshots and testing longer-horizon / strong-bull-only hypotheses on truly fresh future data.
