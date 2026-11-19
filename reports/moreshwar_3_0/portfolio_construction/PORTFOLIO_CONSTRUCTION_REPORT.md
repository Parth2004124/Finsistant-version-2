# MORESHWAR 3.0 (PROMPT 5) — PORTFOLIO CONSTRUCTION, RISK BUDGETING & CAPITAL ALLOCATION REPORT

- **Evaluation Window (Full):** `2021-07-01 -> 2026-11-19` | **IS:** `2021-07-01 -> 2025-03-31` | **OOS:** `2025-04-01 -> 2026-11-19`
- **Shared Capital Pool:** `INR 100,000` (`1.0x` max gross exposure, **zero leverage**, T+1 NSE exit cash settlement, Zerodha Delivery + 7.5 bps slippage)
- **Primary Architecture:** `PORT_RISK_BUDGETED` (Marginal Risk-Budgeted & Correlation-Penalized Sizing + Causal Regime Budgets + Drawdown Governor)

## 1. Position Sizing Schemes Head-to-Head Comparison (Shared Capital Pool)

| Sizing Method | Full Return (Uninvested Cash) | Full Return (w/ 6.5% Idle Cash Sweep) | Full Max DD | Full Net PF | Trades | Cost Drag | OOS Return (`2025-04 -> 2026-11`) | OOS Max DD | OOS Net PF | Reconciliation |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `EQUAL_WEIGHT` | **+9.32%** | **+31.76%** | **18.96%** | 1.12 | 218 | 9.56% | **+13.96%** | **9.45%** | 1.52 | `+0.00` |
| `VOLATILITY_ADJUSTED` | **+6.76%** | **+30.29%** | **16.35%** | 1.10 | 241 | 9.47% | **+16.04%** | **6.81%** | 1.63 | `+0.00` |
| `RISK_BUDGETED` | **+7.43%** | **+29.76%** | **16.50%** | 1.10 | 238 | 9.83% | **+15.70%** | **7.94%** | 1.57 | `+0.00` |
| `MAX_LOSS_BASED` | **+5.97%** | **+27.35%** | **20.73%** | 1.08 | 178 | 9.10% | **+3.27%** | **10.77%** | 1.13 | `+0.00` |
| `CONFIDENCE_VALIDATED` | **+7.36%** | **+28.82%** | **15.90%** | 1.10 | 224 | 9.82% | **+11.73%** | **7.87%** | 1.44 | `+0.00` |

## 2. Portfolio-Level Results vs Standalone Strategy Families (Prompt 3)

| Portfolio / Standalone Strategy | Full Net Return | Full Max Drawdown | Full Net PF | OOS Net Return | OOS Max Drawdown | Capital Pool |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **`PORT_RISK_BUDGETED` (Shared Capital)** | **+7.43%** (`+29.76%` w/ sweep) | **16.50%** | **1.10** | **+15.70%** | **7.94%** | Shared `INR 100k` |
| **`PORT_CONFIDENCE_VALIDATED` (Shared Capital)** | **+7.36%** (`+28.82%` w/ sweep) | **15.90%** | **1.10** | **+11.73%** | **7.87%** | Shared `INR 100k` |
| `STRAT_A_CS_MOMENTUM` (Standalone) | +10.97% | 24.89% | 1.08 | +15.41% | 14.26% | Isolated `INR 100k` |
| `STRAT_B_VCP_BREAKOUT` (Standalone) | +12.92% | 25.51% | 1.09 | +7.95% | 16.70% | Isolated `INR 100k` |
| `STRAT_C_TREND_PULLBACK` (Standalone) | -12.29% | 39.85% | 0.92 | +2.54% | 12.58% | Isolated `INR 100k` |
| `STRAT_D_SECTOR_ROTATION` (Standalone) | +5.56% | 36.69% | 1.04 | +0.78% | 14.87% | Isolated `INR 100k` |
| `STRAT_E_DEFENSIVE_YIELD_LOWVOL` (Standalone) | +26.15% | 22.58% | 1.25 | +13.31% | 11.01% | Isolated `INR 100k` |

## 3. Strategy Sleeve Contribution & Multi-Strategy Consensus Analysis (`PORT_RISK_BUDGETED`)

| Strategy Sleeve | Closed Trades | Winning Trades | Win Rate % | Consensus Multi-Family Trades | Net PnL (INR) | Return Contribution |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `STRAT_A_CS_MOMENTUM` | 86 | 36 | 41.9% | 59 | `INR +10,632.04` | **+10.63%** |
| `STRAT_B_VCP_BREAKOUT` | 23 | 10 | 43.5% | 15 | `INR -5,508.38` | **-5.51%** |
| `STRAT_C_TREND_PULLBACK` | 1 | 0 | 0.0% | 1 | `INR -8.52` | **-0.01%** |
| `STRAT_D_SECTOR_ROTATION` | 17 | 7 | 41.2% | 15 | `INR -96.55` | **-0.10%** |
| `STRAT_E_DEFENSIVE_YIELD_LOWVOL` | 111 | 54 | 48.6% | 36 | `INR +2,415.78` | **+2.42%** |

## 4. Capital Utilisation, Signal Conflicts & Allocation Bottlenecks

- **Average Deployed Capital Utilisation:** `45.0%` | **Average Idle Cash Reserve:** `55.0%` (intentionally preserved during Bear/Stress regimes & drawdown governor scaling)
- **Average Raw Strategy Opportunities per Rebalance:** `131.5` candidates across Families A..E
- **Average Portfolio Diversification Ratio:** `1.889` | **Average Ex-Ante Portfolio Vol:** `12.14%` (Target `<= 15.0%`)

| Constraint / Gatekeeper Event | Count | Role in Portfolio Protection |
| :--- | :---: | :--- |
| `SIGNAL_CONFLICTS_MERGED` | `684` | Hard pre-trade / risk-budget constraint enforcement |
| `SECTOR_CAP_BLOCKED` | `114` | Hard pre-trade / risk-budget constraint enforcement |
| `CORRELATION_CAP_BLOCKED` | `14` | Hard pre-trade / risk-budget constraint enforcement |
| `MAX_POSITIONS_BLOCKED` | `58` | Hard pre-trade / risk-budget constraint enforcement |
| `TURNOVER_THROTTLE_BLOCKED` | `25` | Hard pre-trade / risk-budget constraint enforcement |
| `CASH_SETTLEMENT_BLOCKED` | `0` | Hard pre-trade / risk-budget constraint enforcement |
| `DRAWDOWN_GOVERNOR_ACTIVE_DAYS` | `30` | Hard pre-trade / risk-budget constraint enforcement |
| `VOL_TARGET_SCALED_DAYS` | `1` | Hard pre-trade / risk-budget constraint enforcement |
| `LIQUIDITY_PARTIAL_FILLS` | `0` | Hard pre-trade / risk-budget constraint enforcement |
| `DRIFT_SKIPPED_REBALANCES` | `4` | Hard pre-trade / risk-budget constraint enforcement |

## 5. Rebalancing Frequency, Drift Band & T+1 Settlement Ablation

| Rebalance Configuration | Full Net Return | w/ Cash Sweep | Max Drawdown | Net PF | Trades | Cost Drag % | Turnover X |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `REBAL_10D_DRIFT_35_T1` | **-10.27%** | +9.82% | 19.64% | 0.86 | 281 | 9.89% | 29.09x |
| `REBAL_15D_DRIFT_35_T1` | **-12.10%** | +8.77% | 20.63% | 0.83 | 260 | 9.12% | 26.74x |
| `REBAL_20D_DRIFT_35_T1` | **+7.43%** | +29.76% | 16.50% | 1.10 | 238 | 9.83% | 32.43x |
| `REBAL_30D_DRIFT_50_T1` | **+18.66%** | +47.69% | 13.84% | 1.29 | 194 | 8.34% | 28.23x |
| `REBAL_20D_DRIFT_35_T0` | **+7.43%** | +29.76% | 16.50% | 1.10 | 238 | 9.83% | 32.43x |

## 6. Institutional Stress Test Suite

1. **Correlation Spike Stress:** Max concurrent positions `12` → worst-case simultaneous stop-loss impact **`-10.62%`** (Pairwise 60D correlation gate (rho <= 0.72) + Drawdown Governor (0.65x/0.45x scaling)).
2. **Sector Crash Stress (`-20%` Sector Shock):** Capped at **`-5.60%`** portfolio NAV impact due to hard `28%` sector limit.
3. **Consecutive Losses Stress:** Max consecutive losing trades `16` (`-8.98%` cumulative streak loss).
4. **Liquidity Deterioration Stress (OOS `2.5x` Slippage `18.75 bps` + `0.5%` Volume Cap):** OOS Net Return **`+14.65%`**, OOS Max DD **`8.17%`**, Cost Drag `5.56%`.
