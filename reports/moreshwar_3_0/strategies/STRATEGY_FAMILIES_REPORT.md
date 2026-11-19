# MORESHWAR 3.0 (PROMPT 3) — INDEPENDENT MULTI-STRATEGY FAMILIES REPORT

- **Evaluation Window (Full):** `2021-07-01 -> 2026-11-19`
- **In-Sample Window:** `2021-07-01 -> 2025-03-31` | **Out-of-Sample Window:** `2025-04-01 -> 2026-11-19`
- **Initial Capital:** `INR 100,000` per strategy (Isolated Stateless Harness, Zerodha Delivery + 7.5 bps Slippage)
- **Equal-Weight 5-Family Baseline:** Full Return **`+9.72%`** | Max Drawdown **`21.77%`**

## 1. Full-Window Strategy Family Comparison (`2021-07-01 -> 2026-11-19`)

| Strategy ID | Family | Net Return | CAGR | Max DD | Sharpe | Sortino | Net PF | Trades | Win % | Cost Drag | Util % | Reconciliation |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `STRAT_A_CS_MOMENTUM` | `A_CS_MOMENTUM` | **+10.97%** | +1.99% | 24.89% | -0.30 | -0.29 | 1.08 | 303 | 46.2% | 18.68% | 62.2% | `+0.00` |
| `STRAT_B_VCP_BREAKOUT` | `B_VCP_BREAKOUT` | **+12.92%** | +2.33% | 25.51% | -0.23 | -0.22 | 1.09 | 245 | 46.9% | 14.35% | 67.4% | `+0.00` |
| `STRAT_C_TREND_PULLBACK` | `C_TREND_PULLBACK` | **-12.29%** | -2.45% | 39.85% | -0.75 | -0.70 | 0.92 | 315 | 47.6% | 20.52% | 67.9% | `+0.00` |
| `STRAT_D_SECTOR_ROTATION` | `D_SECTOR_ROTATION` | **+5.56%** | +1.03% | 36.69% | -0.30 | -0.31 | 1.04 | 321 | 48.3% | 17.31% | 50.8% | `+0.00` |
| `STRAT_E_DEFENSIVE_YIELD_LOWVOL` | `E_DEFENSIVE_YIELD_LOWVOL` | **+26.15%** | +4.49% | 22.58% | -0.15 | -0.15 | 1.25 | 215 | 51.6% | 13.56% | 63.0% | `+0.00` |

## 2. In-Sample (`2021-07 -> 2025-03`) vs Out-of-Sample (`2025-04 -> 2026-11`) Breakdown

| Strategy ID | IS Net Return | IS CAGR | IS Max DD | IS Sharpe | IS PF | OOS Net Return | OOS CAGR | OOS Max DD | OOS Sharpe | OOS PF | OOS Excess vs Nifty |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `STRAT_A_CS_MOMENTUM` | +13.15% | +3.41% | 22.76% | -0.19 | 1.13 | **+15.41%** | +9.33% | 14.26% | +0.24 | 1.35 | **+5.27 pp** |
| `STRAT_B_VCP_BREAKOUT` | -3.09% | -0.85% | 25.51% | -0.46 | 0.97 | **+7.95%** | +4.88% | 16.70% | -0.06 | 1.17 | **-2.19 pp** |
| `STRAT_C_TREND_PULLBACK` | +21.69% | +5.48% | 15.80% | -0.06 | 1.24 | **+2.54%** | +1.57% | 12.58% | -0.36 | 1.07 | **-7.60 pp** |
| `STRAT_D_SECTOR_ROTATION` | -5.46% | -1.52% | 26.19% | -0.45 | 0.95 | **+0.78%** | +0.48% | 14.87% | -0.32 | 1.02 | **-9.36 pp** |
| `STRAT_E_DEFENSIVE_YIELD_LOWVOL` | +9.98% | +2.62% | 22.58% | -0.31 | 1.13 | **+13.31%** | +8.08% | 11.01% | +0.17 | 1.39 | **+3.17 pp** |

## 3. Daily Return Correlation Matrix Across Strategy Families

| Strategy | `STRAT_A_CS_MOMENTUM` | `STRAT_B_VCP_BREAKOUT` | `STRAT_C_TREND_PULLBACK` | `STRAT_D_SECTOR_ROTATION` | `STRAT_E_DEFENSIVE_YIELD_LOWVOL` |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `STRAT_A_CS_MOMENTUM` | +1.000 | +0.598 | +0.567 | +0.470 | +0.546 |
| `STRAT_B_VCP_BREAKOUT` | +0.598 | +1.000 | +0.541 | +0.526 | +0.534 |
| `STRAT_C_TREND_PULLBACK` | +0.567 | +0.541 | +1.000 | +0.395 | +0.577 |
| `STRAT_D_SECTOR_ROTATION` | +0.470 | +0.526 | +0.395 | +1.000 | +0.460 |
| `STRAT_E_DEFENSIVE_YIELD_LOWVOL` | +0.546 | +0.534 | +0.577 | +0.460 | +1.000 |
