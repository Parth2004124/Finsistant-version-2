# MORESHWAR 3.0 (PROMPT 4) — CAUSAL MARKET REGIME INTELLIGENCE & ADAPTIVE STRATEGY ALLOCATION REPORT

- **Evaluation Window:** `2022-01-03 -> 2026-11-19 (1206d)` (In-Sample `<= 2025-03-31` | Out-of-Sample `>= 2025-04-01`)
- **Regime Models Compared:** `5` strictly causal architectures (`RULE_5CORE_HYST`, `HMM_3_CAUSAL`, `KMEANS_4`, `VOL_STATE`, `CUSUM_CP`, plus unsmoothed `RULE_9FINE` ablation)
- **Switching Friction:** `15.0 bps` one-way turnover cost charged on every regime allocation change (`69` transitions, `3.01%` cumulative drag)

## 1. Out-of-Sample (`2025-04-01 -> 2026-11-19`) & Full-Window Regime Detector Stability Comparison

| Regime Model | Full Transitions/Yr | Full Avg Duration | Full False Flips (<=2d) | OOS Transitions/Yr | OOS Avg Duration | OOS False Flip Rate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `RULE_5CORE_HYST` | 14.21 | 17.5d | 0 (0.0%) | **11.82** | **20.2d** | **0.0%** |
| `RULE_9FINE` | 48.48 | 5.2d | 128 (55.2%) | **42.31** | **5.9d** | **58.8%** |
| `KMEANS_4` | 35.73 | 7.0d | 72 (42.1%) | **35.47** | **7.0d** | **47.4%** |
| `HMM_3_CAUSAL` | 11.70 | 21.2d | 3 (5.4%) | **11.20** | **21.3d** | **0.0%** |
| `CUSUM_CP` | 3.97 | 60.3d | 1 (5.3%) | **4.98** | **45.0d** | **0.0%** |
| `VOL_STATE` | 16.72 | 14.9d | 25 (31.2%) | **18.67** | **13.1d** | **50.0%** |

## 2. Causal Core Regime Distribution & Next-Day Nifty 50 Behavior (`RULE_5CORE_HYST` Full Window)

| Core Regime | Days | % of Time | Next-1D Nifty Ann. Return | Next-1D Nifty Ann. Vol | Target Equity Exposure | Target Cash Buffer |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `SIDEWAYS_CHOP` | 219 | 18.2% | -4.85% | 12.20% | **85%** | **15%** |
| `STRESS_BEAR` | 261 | 21.6% | +31.33% | 17.55% | **40%** | **60%** |
| `STRONG_BULL` | 285 | 23.6% | +1.31% | 12.08% | **100%** | **0%** |
| `WEAK_BEAR` | 30 | 2.5% | -32.75% | 19.96% | **65%** | **35%** |
| `WEAK_BULL` | 410 | 34.0% | +9.67% | 12.40% | **95%** | **5%** |

## 3. Adaptive Regime-Allocated Meta-Portfolio vs Static Equal-Weight & Single-Family Baselines

| Portfolio / Strategy | Full Net Return | Full CAGR | Full Max DD | Full Sharpe | Full Calmar | OOS Net Return | OOS Max DD | OOS Sharpe | OOS Excess vs Nifty |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `M3_RULE5_REGIME_ADAPTIVE` | **+15.23%** | +3.01% | **14.52%** | **-0.36** | +0.21 | **+2.76%** | **8.30%** | **-0.65** | **-5.72 pp** |
| `M3_HMM3_CAUSAL_ADAPTIVE` | **+15.99%** | +3.15% | **15.96%** | **-0.36** | +0.20 | **+1.53%** | **7.90%** | **-0.73** | **-6.95 pp** |
| `M3_STATIC_EQUAL_WEIGHT_5FAM` | **+9.72%** | +1.96% | **21.77%** | **-0.39** | +0.09 | **+1.43%** | **11.59%** | **-0.55** | **-7.05 pp** |
| `STRAT_A_CS_MOMENTUM` | **+10.97%** | +2.20% | **24.89%** | **-0.27** | +0.09 | **-2.19%** | **17.35%** | **-0.59** | **-10.67 pp** |
| `STRAT_B_VCP_BREAKOUT` | **+12.92%** | +2.57% | **25.51%** | **-0.20** | +0.10 | **+16.21%** | **10.08%** | **+0.27** | **+7.73 pp** |
| `STRAT_C_TREND_PULLBACK` | **-12.29%** | -2.70% | **39.85%** | **-0.73** | -0.07 | **-28.10%** | **32.43%** | **-2.00** | **-36.58 pp** |
| `STRAT_D_SECTOR_ROTATION` | **+5.56%** | +1.14% | **36.69%** | **-0.27** | +0.03 | **+11.49%** | **16.46%** | **+0.09** | **+3.01 pp** |
| `STRAT_E_DEFENSIVE_YIELD_LOWVOL` | **+26.15%** | +4.97% | **22.58%** | **-0.09** | +0.22 | **+14.53%** | **11.13%** | **+0.24** | **+6.05 pp** |

## 4. Regime-Conditioned Strategy Allocation Matrix

| Core Regime | `A_CS_MOMENTUM` | `B_VCP_BREAKOUT` | `C_TREND_PULLBACK` | `D_SECTOR_ROTATION` | `E_DEFENSIVE_YIELD` | `CASH_RESERVE (6.5%)` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `STRONG_BULL` | 30% | 25% | 15% | 20% | 10% | **0%** |
| `WEAK_BULL` | 25% | 15% | 20% | 20% | 15% | **5%** |
| `SIDEWAYS_CHOP` | 10% | 5% | 25% | 15% | 30% | **15%** |
| `WEAK_BEAR` | 5% | 0% | 15% | 10% | 35% | **35%** |
| `STRESS_BEAR` | 0% | 0% | 10% | 0% | 30% | **60%** |
