# Finsistant V2 — Moreshwar Mathematical Model 2.0: Quantitative Research & Validation Report

**Status:** `SHADOW_RESEARCH_ONLY` (**Moreshwar 1.0 remains active in live production pending user review.**)  
**Evaluation Window:** `2026-01-21` to `2026-11-18` (214 post-warmup daily trading bars across 10 live assets + Nifty 50 `^NSEI`)  
**Data Sources:** Official NSE MCP (`cm-market` + daily `bhavcopy`) + Yahoo Finance Live Fundamentals  
**Execution & Friction Model:** Next-bar open fill + 5 bps entry slippage + 5 bps exit slippage + 25 bps statutory/brokerage costs (`0.35%` round-trip), strict zero overlapping same-symbol positions.

---

## Phase 1: Audit of Moreshwar 1.0 (`calculate_moreshwar_levels`)

Inspected modules: `finsistant_v2/scoring_engine.py`, `finsistant_v2/backtest_engine.py`, `finsistant_v2/quant_engine.py`, `finsistant_v2/agents.py`, `server.py`.

### Documented Moreshwar 1.0 Formulation
```text
X = round((f_score_total + p_score_total) / 2.0, 1)
downside_pts = 100.0 - X
scale = max(1.0, round(spot_price / 1000.0, 2))
target = round(spot_price + X * scale, 2)
stop_loss = round(max(spot_price * 0.85, spot_price - downside_pts * scale), 2)
entry_price = round(spot_price if is_holding else max(spot_price * 0.90, spot_price - (downside_pts * 0.65 * scale)), 2)
```

### Quantified Structural Weaknesses Identified
1. **Fundamental-to-Stop Compression (`downside_pts = 100 - X`):** Higher fundamental scores (`X = 80..95`) artificially compress stop-loss distance to `1.5%–2.0%` regardless of actual price volatility (daily ATR often `1.8%–3.2%`), resulting in a **63.1% stop-loss whip-saw rate** in Model A (`-7.76%` total, `14.24%` max drawdown).
2. **`scale = max(1.0, spot / 1000)` Distortion on Lower-Priced Stocks:** Any ₹100 stock gets `scale = 1.0`, forcing the `-15%` stop floor (`₹85`) and an unrealistic `+80%` target (`₹180`) on a 14-day swing horizon, while a ₹3,500 stock gets `scale = 3.5` and tight percentage bands.
3. **Static Snapshot Leakage in Earlier Backtests:** Live current financial ratios (`PE`, `ROE`, `Sales Growth`, `Profit Growth`) were evaluated across past historical dates because Yahoo/NSE feeds do not expose point-in-time historical quarterly fundamental time series.
4. **Regime Blindness:** 40 out of 84 Model A trades (`47.6%`) entered during `BEARISH` Nifty 50 regimes, generating **-₹5,008** in losses (`64.5%` of total Model A portfolio drawdown).

*Moreshwar 1.0 (`calculate_moreshwar_levels`) remains completely intact as the reproducible baseline and active production engine.*

---

## Phase 2 & Phase 3: Moreshwar 2.0 Architecture (`finsistant_v2/moreshwar_v2.py`)

### A. Confidence Adjustment (Preserving Anti-Certainty Principle)
- `base_quality_conf = min(0.85, 0.32 + 0.50 * sqrt(avg_q / 100.0))` (concave diminishing returns; a `100/100` fundamental+porter score caps strictly at `<= 0.85`, NEVER `1.00`).
- **Explicit Penalties:**
  - Unavailable point-in-time historical fundamentals (`historical_unavailable`): neutral quality base `0.66` minus `-0.08` unavailability haircut (`0.58`).
  - Missing live ratios: `-0.04` per missing field (capped at `-0.18`).
  - Contradictory technical signals: below 20D SMA (`-0.12`), bearish tech (`-0.14`) / neutral tech (`-0.04`), downward polynomial curve (`-0.09`), overbought RSI ≥ 74 (`-0.08`).
  - Hostile Nifty regime: `NEUTRAL` (`-0.04`), `HIGH_VOLATILITY` (`-0.08`), `BEARISH` (`-0.14`).

### B. ATR(14) Volatility-Normalised Stops
- Strictly causal Wilder ATR(14) computed from trailing OHLC bars.
- `Stop Distance = clamp(ATR(14) * dyn_mult [with structural support alignment], 2.5%, 10.0%)`.
- Eliminates fundamental-score compression: an identical price chart yields identical ATR stop geometry regardless of whether fundamental score is `55` or `99`.

### C. Volatility-Adjusted Risk-Budget Position Sizing
- `Risk Budget = Available Capital × 1.25% × Confidence/High-Vol Throttle`
- `Qty = floor(min(Risk Budget / (Entry - Stop), Single Stock Cap 20% / Entry, Remaining Portfolio Cap 85% / Entry))`
- Enforces integer share quantities and minimum 20-day average daily turnover liquidity threshold.

### D. Statistically Realistic Horizon Targets
- `Expected Move = ATR(14) × sqrt(14) × 1.05`, bounded by 20-day structural resistance breakout headroom and `22%` ceiling.
- Automatically rejects setups whose realistic ATR R:R fails `min_rr_ratio = 1.50`.

### E. Strictly Causal Market Regime Filter (`classify_market_regime`)
- Evaluates Nifty 50 (`^NSEI`) strictly using `bars[date <= decision_date]`:
  - `HIGH_VOLATILITY`: 20D annualized vol ≥ `18.0%`, OR Nifty ATR(14) ≥ `1.65%`, OR 5-day drop ≤ `-3.5%`
  - `BEARISH`: `close < SMA(50)` and (`close < SMA(20)` or falling `SMA(20)`)
  - `BULLISH`: `close >= SMA(20) >= SMA(50)` with rising `SMA(20)`
  - `NEUTRAL`: transitional/mixed structure

---

## Phase 4–7: Experimental Results & Model A–F Isolation Comparison

All models executed under identical daily portfolio cash constraints (`₹100,000`), `0.35%` round-trip cost, next-bar open execution, and max `14D` swing horizon via `python3 scripts/run_moreshwar_v2_experiments.py`:

| Model Variant | Tot% | Ann% | Win% | PF | Sharpe | MaxDD% | Trades | SL% | TP% | Util% | Rej / RegBlk | Integrity Tag |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| **Model A: Original Baseline (Moreshwar 1.0)** | -7.76% | -9.08% | 32.1% | 0.82 | -1.18 | 14.24% | 84 | 63.1% | 15.5% | 54.3% | 31 / 0 | Static Fund Warn |
| **Model B: ATR Stops Only** | -8.82% | -10.30% | 38.6% | 0.79 | -1.16 | 15.11% | 70 | 42.9% | 18.6% | 62.9% | 43 / 0 | Static Fund Warn |
| **Model C: ATR Stops + Vol Sizing** | -4.79% | -5.62% | 38.9% | 0.85 | -1.07 | 11.35% | 72 | 41.7% | 18.1% | 49.1% | 11 / 0 | Static Fund Warn |
| **Model D: ATR + Sizing + Nifty Regime** | **+0.58%** | **+0.68%** | **43.5%** | **1.03** | **-0.58** | **7.25%** | 46 | **30.4%** | 15.2% | 37.1% | 153 / 140 | Static Fund Warn |
| **Model E: Complete Moreshwar 2.0 (Clean Causal)** | -6.14% | -7.19% | 31.2% | 0.72 | -1.60 | 11.39% | 64 | 46.9% | 17.2% | 34.8% | 237 / 47 | **CAUSAL CLEAN** |
| **Model E-Ref: Moreshwar 2.0 + Static Fund Warn** | -0.99% | -1.17% | 42.3% | 0.95 | -0.85 | 8.65% | 52 | 34.6% | 19.2% | 34.7% | 136 / 136 | Static Fund Warn |
| **Model F: Fixed 5% / 10% Benchmark** | -5.07% | -5.94% | 44.6% | 0.87 | -0.82 | 13.44% | 65 | 41.5% | 20.0% | 63.0% | 36 / 0 | Static Fund Warn |
| **Simple SMA-20 + Fixed 1:2 Benchmark** | -5.36% | -6.28% | 36.0% | 0.89 | -0.71 | 15.85% | 89 | 47.2% | 18.0% | 75.1% | 157 / 0 | CAUSAL CLEAN |
| **Nifty 50 (`^NSEI`) Buy & Hold** | +0.28% | +0.35% | 100.0% | N/A | -0.32 | 14.34% | 1 | — | — | 100.0% | — | INDEX BENCHMARK |

### Out-of-Sample & Walk-Forward Stability (Phase 6)
- **Strict Out-of-Sample Tail (`2026-09-04` to `2026-11-18`, 54 unseen bars):**
  - `Complete Moreshwar 2.0 (Clean Causal)`: **`+4.92%`** (`0.70%` MaxDD, **`70.0%` Win**, `10` trades)
  - `Moreshwar 1.0 Baseline`: `+4.40%` (`3.40%` MaxDD, `48.0%` Win, `25` trades)
  - `Nifty 50 (^NSEI)`: `+5.57%` (`6.97%` MaxDD)
- **Rolling Walk-Forward Folds:**
  - `WF-1 (Jan–May)`: M1 `-8.14% (DD 8.86%)` vs M2 Clean `-4.15% (DD 5.14%)` -> M2 saved `3.99%`
  - `WF-2 (May–Aug)`: M1 `+0.32% (DD 5.70%)` vs M2 Clean `-3.61% (DD 6.61%)`
  - `WF-3 (Aug–Nov)`: M1 `-1.03% (DD 7.80%)` vs M2 Clean **`+1.50% (DD 3.19%, 47.1% Win)`**

---

## Phase 10: Honest Conclusions, Limitations & Recommendation

### What Genuinely Improved
1. **Premature Stop-Out Reduction:** ATR(14) normalization reduced stop-loss trigger frequency from `63.1%` (M1) down to `30.4%–42.9%`.
2. **Drawdown & Bear-Regime Protection:** Blocking entries during `BEARISH` Nifty 50 regimes eliminated Model A's entire `-₹5,008` bear-market bleed (`Model D`: **`-7.76% -> +0.58%`**, MaxDD halved `14.24% -> 7.25%`).
3. **Out-of-Sample Capital Efficiency:** On the unseen tail (`Sep–Nov 2026`), Clean Causal M2 captured `88%` of Nifty's `+5.57%` rally (`+4.92%`) with **nearly 10x lower drawdown** (`0.70%` vs `6.97%`).

### Honest Weaknesses & Why Moreshwar 2.0 Is NOT Auto-Promoted Yet
1. **Risk Math Alone Cannot Replace Quality Selection (`Model E Clean = -6.14%`):** When today's static fundamentals are stripped for strict historical causality, `Model E Clean` takes pure technical signals on structurally weak names (`RELIANCE` 1/9 wins `-₹4,534`, `ZEEL` 1/7 wins `-₹1,342`), losing `-6.14%` (`PF 0.72`, `Sharpe -1.60`).
2. **Missing Point-in-Time Historical Fundamentals:** Neither NSE MCP nor Yahoo Finance free endpoints expose historical quarterly financial ratios (`ROE`, `OPM`, `Sales/Profit Growth` as of `2026-03-01`). Bridging `Model D` (`+0.58%`) and `Model E Clean` (`-6.14%`) cleanly requires ingesting point-in-time historical fundamental snapshots as the permanent audit ledger accumulates forward over time.
3. **Risk-Free Benchmark Hurdle (`6.5%` India T-Bill):** Even Model D (`+0.68%` ann, `Sharpe -0.58`) remains below cash yields across this 214-bar choppy sideways sample.

### Recommended Next Step
Keep **Moreshwar 1.0 active in production** while running **Moreshwar 2.0 in Shadow Mode** (12-field explainability live on Tab 4) until forward point-in-time fundamental snapshots + out-of-sample decision logs reach 30+ settled records and user approves promotion.
