# MORESHWAR 3.0 (PROMPT 7) — ADAPTIVE STRATEGY ORCHESTRATION, SUSPENSION & EXPLAINABILITY REPORT

- **Evaluation Window (Full):** `2022-01-03 -> 2026-11-19 (1206d)` | **IS:** `2022-01-03 -> 2025-03-31` | **OOS:** `2025-04-01 -> 2026-11-19`
- **Registered Strategy Engines:** `9` (`5` Prompt 3 Equity Sleeves + `4` Prompt 6 Relative-Value / Hedged Branches)
- **Primary Central Intelligence Layer:** `CONSTRAINED_ADAPTIVE_ORCHESTRATOR` (Causal Regime Gate + Risk-Budgeted Weights + Sleeve Suspension/Recovery State Machine + 4% Deadband & EWMA Smoothing)

## 1. Authoritative Strategy Registry (`strategy_registry.json`)

| Strategy ID | Version | Source | Capital / Market Eligibility | Validated Regimes | IS Return (`<=2025-03`) | IS PF | Throttle / Suspend DD | Validation Status |
| :--- | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :--- |
| `STRAT_A_CS_MOMENTUM` | `3.0.0-p3` | `PROMPT_3` | `NSE_CASH_EQUITY_CNC_LONG_ONLY` (`INR 100,000`) | `STRONG_BULL,WEAK_BULL,SIDEWAYS_CHOP` | -3.85% | 1.01 | `14.0% / 21.0%` | **`VALIDATED_CORE_LONG`** |
| `STRAT_B_VCP_BREAKOUT` | `3.0.0-p3` | `PROMPT_3` | `NSE_CASH_EQUITY_CNC_LONG_ONLY` (`INR 100,000`) | `STRONG_BULL,WEAK_BULL` | +4.61% | 1.07 | `13.5% / 20.0%` | **`VALIDATED_CORE_LONG`** |
| `STRAT_C_TREND_PULLBACK` | `3.0.0-p3` | `PROMPT_3` | `NSE_CASH_EQUITY_CNC_LONG_ONLY` (`INR 100,000`) | `WEAK_BULL,SIDEWAYS_CHOP` | -14.46% | 0.89 | `10.0% / 16.0%` | **`CONDITIONAL_MEAN_REV_LONG`** |
| `STRAT_D_SECTOR_ROTATION` | `3.0.0-p3` | `PROMPT_3` | `NSE_CASH_EQUITY_CNC_LONG_ONLY` (`INR 100,000`) | `STRONG_BULL,WEAK_BULL,SIDEWAYS_CHOP` | +4.74% | 1.05 | `14.0% / 21.0%` | **`VALIDATED_CORE_LONG`** |
| `STRAT_E_DEFENSIVE_YIELD_LOWVOL` | `3.0.0-p3` | `PROMPT_3` | `NSE_CASH_EQUITY_CNC_LONG_ONLY` (`INR 100,000`) | `STRONG_BULL,WEAK_BULL,SIDEWAYS_CHOP,WEAK_BEAR,STRESS_BEAR` | +11.33% | 1.18 | `12.5% / 18.5%` | **`VALIDATED_CORE_LONG`** |
| `STRAT_RV1_COINT_PAIRS` | `3.0.0-p6` | `PROMPT_6` | `NSE_SLBM_SHORT_OR_SSF` (`INR 2,500,000`) | `SIDEWAYS_CHOP,WEAK_BEAR` | -17.67% | 0.74 | `8.0% / 12.0%` | **`REJECTED_SLBM_DRAG`** |
| `STRAT_RV2A_SECTOR_REL_MOM` | `3.0.0-p6` | `PROMPT_6` | `NSE_SLBM_SHORT_OR_SSF` (`INR 2,500,000`) | `STRONG_BULL,WEAK_BULL,SIDEWAYS_CHOP` | +19.02% | 1.15 | `10.0% / 15.0%` | **`REJECTED_SLBM_DRAG`** |
| `STRAT_RV3_MARKET_NEUTRAL_LS` | `3.0.0-p6` | `PROMPT_6` | `NSE_SLBM_SHORT_OR_SSF` (`INR 2,500,000`) | `SIDEWAYS_CHOP,WEAK_BEAR,STRESS_BEAR` | -9.45% | 0.88 | `10.0% / 15.0%` | **`REJECTED_SLBM_DRAG`** |
| `STRAT_RV4A_NIFTY_FUT_HEDGED_25L_LOT` | `3.0.0-p6` | `PROMPT_6` | `NSE_FNO_INDEX_FUTURES_25L` (`INR 2,500,000`) | `SIDEWAYS_CHOP,WEAK_BEAR,STRESS_BEAR` | +12.18% | 1.14 | `10.0% / 15.0%` | **`ISOLATED_25L_FUTURES_HEDGE`** |

## 2. Five Allocation Architectures + Stock-Level Shared Capital Comparison

| Allocator Architecture | Full Net Return | Full CAGR | Full Max DD | Full Calmar | OOS Net Return (`2025-04 -> 2026-11`) | OOS Max DD | Switches | Switch Cost Drag | Suspensions / Recoveries |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `STATIC` | **+17.04%** | +3.34% | **20.25%** | +0.17 | **+8.18%** | **10.33%** | 1 | 0.03% | `11 / 11` |
| `REGIME_BASED` | **+12.88%** | +2.56% | **13.90%** | +0.18 | **+3.62%** | **9.08%** | 40 | 2.07% | `11 / 11` |
| `VOLATILITY_ADJUSTED` | **+18.20%** | +3.56% | **18.50%** | +0.19 | **+8.21%** | **9.83%** | 81 | 0.41% | `11 / 11` |
| `RISK_BUDGETED` | **+18.14%** | +3.54% | **18.46%** | +0.19 | **+8.20%** | **9.84%** | 81 | 0.43% | `11 / 11` |
| `CONSTRAINED_ADAPTIVE_ORCHESTRATOR` | **+14.83%** | +2.93% | **14.03%** | +0.21 | **+3.84%** | **7.47%** | 61 | 1.57% | `11 / 11` |
| `STOCK_LEVEL_SHARED_CAPITAL_ORCHESTRATED` | **+8.48%** | +1.72% | **15.77%** | +0.11 | **-3.46%** | **6.29%** | 81 | 9.21% | `11 / 11` |

## 3. Orchestration Stress & Failure Validation Suite

1. **Single-Strategy Structural Failure Injection (`-26%` 25-day collapse injected into `STRAT_A` in OOS):**
   - `CONSTRAINED_ADAPTIVE_ORCHESTRATOR` automatically triggered `13` sleeve suspensions, limiting Max Drawdown to **`14.03%`** (Full Return **`+14.42%`**) vs **`22.69%`** Max Drawdown (`+9.47%` Return) on the unprotected `STATIC` portfolio (**`+8.66 pp` drawdown reduction**).
2. **1-Day Delayed Signal Execution (`t` close applied at `t+2` open):** Full Net Return **`+12.25%`** vs `+14.83%` baseline (`-2.58 pp` drift, Max DD `14.60%`), confirming low sensitivity to 1-day execution latency.
3. **Correlation Convergence Stress:** Full Net Return **`+11.31%`**, Max Drawdown **`13.98%`**.

## 4. Sample Explainability Audit Logs (`allocation_explanations_log.json`)

| Decision Date | Causal Regime | Port DD % | Expected Gross Exp % | Cash Reserve % | Active Target Weights | Suspended / Throttled Sleeves |
| :--- | :---: | :---: | :---: | :---: | :--- | :--- |
| `2026-07-24` | `SIDEWAYS_CHOP` | 10.35% | **69.0%** | **31.0%** | `A_CS_MOMENTUM:16%, B_VCP_BREAKOUT:8%, D_SECTOR_ROTATION:9%, E_DEFENSIVE_YIELD_LOWVOL:35%, CASH_RESERVE:31%` | `C_TREND_PULLBACK(SUSPENDED)` |
| `2026-08-14` | `SIDEWAYS_CHOP` | 8.39% | **69.0%** | **31.0%** | `A_CS_MOMENTUM:16%, B_VCP_BREAKOUT:8%, D_SECTOR_ROTATION:9%, E_DEFENSIVE_YIELD_LOWVOL:35%, CASH_RESERVE:31%` | `C_TREND_PULLBACK(SUSPENDED)` |
| `2026-09-04` | `SIDEWAYS_CHOP` | 8.88% | **69.0%** | **31.0%** | `A_CS_MOMENTUM:16%, B_VCP_BREAKOUT:8%, D_SECTOR_ROTATION:9%, E_DEFENSIVE_YIELD_LOWVOL:35%, CASH_RESERVE:31%` | `C_TREND_PULLBACK(SUSPENDED)` |
| `2026-09-28` | `STRESS_BEAR` | 10.04% | **46.4%** | **53.6%** | `E_DEFENSIVE_YIELD_LOWVOL:46%, CASH_RESERVE:54%` | `C_TREND_PULLBACK(SUSPENDED)` |
| `2026-10-21` | `STRESS_BEAR` | 10.31% | **46.4%** | **53.6%** | `E_DEFENSIVE_YIELD_LOWVOL:46%, CASH_RESERVE:54%` | `C_TREND_PULLBACK(SUSPENDED)` |
| `2026-11-12` | `WEAK_BULL` | 8.54% | **68.0%** | **32.0%** | `A_CS_MOMENTUM:9%, B_VCP_BREAKOUT:9%, D_SECTOR_ROTATION:11%, E_DEFENSIVE_YIELD_LOWVOL:39%, CASH_RESERVE:32%` | `A_CS_MOMENTUM(THROTTLED_50PCT), C_TREND_PULLBACK(SUSPENDED)` |
