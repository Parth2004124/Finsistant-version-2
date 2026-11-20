# MORESHWAR 3.1 (PROMPT 4/5) — ADVISORY RISK MONITORING & FAILURE DETECTION RULES

> [!CAUTION]
> **ADVISORY ONLY:** All monitoring triggers produce human-review alerts and paper-trading state flags. Live trading and broker order placement remain **STRICTLY DISABLED**.

## 1. Predefined Evidence-Based Thresholds (`regime_monitoring.py`)

| Condition Monitored | Warning Threshold | Suspension / Halt Advisory | Rationale (Separating Noise from Decay) |
| :--- | :--- | :--- | :--- |
| **Normal vs Unexpected Drawdown** | `12.0%` (`NORMAL_DRAWDOWN_WATCH`) / `16.0%` (`WARNING`) | `>= 21.0%` (`SUSPEND_TACTICAL_ADVISORY`) | Historical primary DD is `13.82%`; never suspend inside normal `<=15%` pullbacks. |
| **Declining Family E Profit Factor** | Rolling 25-trade `PF < 0.95` (requires `>= 15` closed trades) | Rolling 25-trade `PF < 0.70` (`>= 15` closed trades) | Prevents false suspension after 2-4 normal losing trades. |
| **Sudden Volatility Expansion** | `vol_20d / vol_60d >= 1.45` OR `vol_pctile >= 0.90` | Advisory review + stress-state verification | Detects regime transition into `HIGH_VOL_STRESS`. |
| **Deteriorating Breadth** | `pct_ab50 < 35%` AND `breadth_delta_20d <= -15 pp` | Advisory review | Flags broad internal market breakdown. |
| **Rising Turnover / Cost Drag** | Annual turnover `> 10.0x` OR cost `> 12%` gross profit | Advisory execution audit | Protects net alpha from friction creep. |
| **Sector Concentration / Excess Cash** | Sector `> 28%` OR uninvested cash `> 12%` for `>= 5d` | Advisory rebalance review | Prevents silent idle cash or concentration drift. |
| **Regime Instability** | `>= 4` confirmed regime flips in 20 sessions | Freeze adaptive switching -> fallback to static | Prevents classifier whipsaw costs. |
| **Data / Accounting Failure** | Missing ETF bar, NaN surge, or `|recon_diff| > INR 5` | Immediate `DATA_EXECUTION_FAILURE_HALT` | Hard stop on corrupted feeds or broken ledgers. |

## 2. Historical Replay Audit (`2021-07 -> 2026-11`, `1,332` Sessions)

- **Status Distribution:** `{'NORMAL_OPERATING': 993, 'EDGE_DETERIORATION_WARNING': 339}`
- **Alert Frequency by Code:** `{'DETERIORATING_MARKET_BREADTH': 151, 'SUDDEN_VOLATILITY_EXPANSION': 133, 'REGIME_CLASSIFIER_INSTABILITY': 81, 'FAMILY_E_PF_WARNING': 31}`
