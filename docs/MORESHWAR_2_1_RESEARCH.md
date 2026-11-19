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
| **Moreshwar 1.0 (Baseline)** | -41.56% | -10.25% | 36.45% `[33.56–39.44]` | 0.82 | -1.10 | -1.46 | 47.89% | -0.87 | -0.290% (`0.0812`) | 1,026 | 8.8 | 85.2% | 61.69% |
| **Moreshwar 2.0 (Clean Causal)** | -14.42% | -3.09% | 43.95% `[40.43–47.54]` | 0.92 | -0.90 | -1.20 | 27.84% | -0.52 | -0.118% (`0.5789`) | 744 | 9.8 | 60.2% | 45.00% |
| **Moreshwar 2.1 (6-Regime + Port Risk)** | **-10.54%** | **-2.22%** | 42.27% `[37.87–46.81]` | 0.90 | -1.24 | -1.68 | **21.03%** | -0.50 | -0.386% (`0.1084`) | 466 | 10.0 | 38.4% | **29.13%** |
| **Moreshwar 2.1 (Permissive Control)** | -17.43% | -3.78% | 41.64% `[37.67–45.71]` | 0.86 | -1.45 | -1.92 | 22.18% | -0.79 | -0.297% (`0.2056`) | 574 | 9.6 | 40.7% | 31.27% |
| **Moreshwar 2.1 + Static Fund (Leak Audit)** | -14.11% | -3.02% | 43.74% `[39.33–48.25]` | 0.88 | -1.12 | -1.54 | 20.20% | -0.70 | -0.245% (`0.3246`) | 471 | 9.8 | 42.8% | 29.20% |
| **Simple SMA-20 Strategy** | +2.79% | +0.56% | 48.11% `[44.91–51.33]` | 1.01 | -0.33 | -0.46 | 35.74% | 0.08 | +0.120% (`0.5436`) | 927 | 11.4 | 90.2% | 61.83% |
| **Fixed 5% Stop / 10% Target** | +19.77% | +3.70% | 43.27% `[40.24–46.36]` | 1.07 | -0.13 | -0.18 | 26.07% | 0.76 | +0.020% (`0.9125`) | 1,003 | 10.4 | 89.2% | 71.31% |
| **Nifty 50 (`^NSEI`) Buy & Hold** | **+48.78%** | **+8.78%** | 100.0% | — | **+0.21** | **+0.29** | **15.77%** | **3.09** | — | 1 | 1190 | 100% | 0% |
| **Risk-Free Cash (6.5% India T-Bill)** | **+38.11%** | **+6.50%** | 100.0% | — | 0.00 | 0.00 | **0.00%** | — | — | 0 | — | 0% | 0% |

---

## 4. Walk-Forward Validation, Untouched OOS & Monte Carlo Robustness

### Rolling 4-Fold Walk-Forward (Zero Tuning on Test Periods)
- **WF-1 (`2022-01-24 → 2023-04-06`, Nifty `+2.62%`):** M1.0 `-19.63%` (DD `26.63%`) · M2.0 `-9.59%` (DD `20.28%`) · **M2.1 `-6.77%` (`Sharpe -1.92`, DD `6.97%`, Win `39.56%`, `n=91`)**
- **WF-2 (`2023-04-10 → 2024-06-21`, Nifty `+33.35%`):** M1.0 `+25.06%` (DD `11.69%`) · M2.0 `+16.01%` (DD `11.78%`) · **M2.1 `+9.95%` (`Sharpe +0.18`, DD `8.50%`, Win `48.80%`, `n=166`)**
- **WF-3 (`2024-06-24 → 2025-09-04`, Nifty `+5.08%`):** M1.0 `-22.46%` (DD `25.82%`) · M2.0 `-11.43%` (DD `19.35%`) · **M2.1 `-7.80%` (`Sharpe -1.92`, DD `13.00%`, Win `37.38%`, `n=107`)**
- **WF-4 UNTOUCHED OOS (`2025-09-05 → 2026-11-19`, Nifty `+3.13%`):**
  - Moreshwar 2.1: **`-4.99%`** (`CAGR -4.04%`, `Sharpe -1.78`, `DD 13.19%`, `Win 37.17%`, `n=113`) vs M2.0 `-13.88%` (`DD 20.70%`) vs M1.0 `-23.44%` (`DD 30.87%`) vs SMA-20 `-8.78%` vs Fixed 5/10 `-15.32%` vs Nifty `+3.13%` vs Cash `+8.41%`.
- **Temporal Stability Check:** 1st-half avg `+1.59%` vs 2nd-half avg `-6.39%` (`deterioration_detected = True`).

### Monte Carlo Simulation (`1,200` Runs Each — Trade Bootstrap vs 10-Day Block Bootstrap)
| Model / Resampling Method | P5 Return | Median Return | P95 Return | Median MaxDD | P95 MaxDD | P(Ret > 0) | P(MaxDD ≥ 25%) | P(MaxDD ≥ 50%) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **M2.1 Trade Bootstrap (1,200 sims)** | -28.40% | -9.69% | +8.82% | 17.21% | 32.07% | 19.0% | 17.17% | **0.08%** |
| **M2.1 10-Day Block Bootstrap (1,200 sims)** | -31.05% | -10.91% | +16.23% | 20.98% | 35.28% | 25.8% | 34.83% | **0.00%** |
| **M1.0 Trade Bootstrap (1,200 sims)** | -73.95% | -41.30% | -13.00% | 46.91% | 76.39% | 1.0% | 93.75% | **43.00%** |
| **M1.0 10-Day Block Bootstrap (1,200 sims)** | -67.31% | -42.18% | +2.20% | 50.89% | 70.15% | 5.9% | 97.00% | **53.42%** |

*Limitation:* Trade bootstrap assumes i.i.d. PnLs (breaks multi-week regime clustering); circular 10-day block bootstrap preserves local serial correlation but cannot synthesize tail crashes worse than observed inside `2022–2026`.

---

## 5. Regime, Portfolio-Risk & Root-Cause Strategy Investigation

### 6-State Causal Nifty Regime Decomposition (`Moreshwar 2.1`)
- **`STRONG_BULLISH` (257 days, `162` trades):** `48.15%` win, **`+0.236%` expectancy/trade**, **`PF 1.15`**, **`Sharpe +3.00`**, `MaxDD 5.98%`.
- **`WEAK_BULLISH` (373 days, `243` trades):** `42.39%` win, `-0.358%` expectancy, `PF 0.87`, `Sharpe -1.07`, `MaxDD 12.99%`.
- **`SIDEWAYS` (54 days, `61` trades):** `26.23%` win, `-2.153%` expectancy, `PF 0.34`, `Sharpe -4.13`, `MaxDD 8.76%`.
- **`WEAK_BEARISH` (237d) / `STRONG_BEARISH` (143d) / `HIGH_VOLATILITY` (188d):** `0` trades entered (cleanly gated).

### Portfolio-Level Risk Upgrades (`Moreshwar 1.0 → Moreshwar 2.1`)
- **Max Simultaneous Positions:** `12` → **`6`** (avg `3.5` vs `6.67`)
- **Max Sector Concentration:** `71.44%` → **`35.48%`** (avg `13.42%` vs `35.82%`)
- **Peak Aggregate SL Exposure:** `13.53%` → **`5.71%`** (avg `1.82%` vs `5.25%`)
- **Single-Day / 5-Day Tail Loss:** `-4.38% / -7.32%` → **`-2.24% / -3.91%`**
- **Nifty Decline Days Exposure (121 drop days):** `-1.324%` avg (`71.5%` deployed) → **`-0.463%` avg (`30.66%` deployed)**

### Forensic Investigation Findings
1. **ATR Stops Work (`50.0% → 39.3%` SL rate):** Replacing `100 - X` fundamental compression eliminates noise whip-saws (`M2.0: 40.5%`, `M2.1: 39.3%`).
2. **Regime Filtering Works (`+6.89 pp` vs Permissive Control):** Blocking bearish/high-vol Nifty days improved 5Y return from `-17.43%` to `-10.54%`, and `M2.1` loses money primarily in `WEAK_BULLISH` (`-0.358%/trd`) and `SIDEWAYS` (`-2.153%/trd`) rather than `STRONG_BULLISH` (`+0.236%/trd`).
3. **E-Ref Static Leakage Proven (`r = +0.162`):** Today's fundamental scores correlate at Pearson `r = +0.162` with prior 4-year returns (`2021–2025`), proving why using today's financial ratios in a multi-year backtest leaks future winners/losers.
4. **Excessive Selectivity & Friction Drag (`-29.13%` cost drag vs `-10.54%` total return):** Counterfactual evaluation of `13,383` rejected setups averaged `-0.025%` (`44.92%` win) vs `466` accepted trades `-0.386%` (`42.27%` win); `UNFAVORABLE_RR` skips (`-0.420%`) helped avoid bad trades, whereas gross technical edge before statutory STT/DP/slippage is insufficient to overcome `~₹29,130` (`29.13%`) in Indian delivery trading costs across `466` short-horizon (`10D`) trades.

---

## 6. Unresolved Risks & Production Stance
- **Why Production Promotion Is Rejected:** Even after state-of-the-art ATR sizing, 6-regime gating, pairwise correlation caps, and cutting M1.0's 50%-ruin risk from `53.42%` down to `0.00%`, **Moreshwar 2.1 still loses `-10.54%` net of real Indian delivery costs (`p = 0.1084`)**, trailing both Nifty 50 (`+48.78%`) and 6.5% Cash (`+38.11%`).
- **Recommendation:** Do **not** deploy Moreshwar 2.1 for automated capital allocation. Preserve it strictly as a research benchmark alongside Moreshwar 1.0/2.0 while collecting clean forward point-in-time fundamental audit snapshots and testing longer-horizon / strong-bull-only hypotheses on truly fresh future data.
