# MORESHWAR 3.0 (PROMPT 8) — SECURITY, TARIFF VALIDATION & EXECUTION QUALITY REPORT

- **Report Version:** `3.0.0-p8-execution-intelligence`
- **Tariff Version:** `2024-10-nse-zerodha-v1`
- **Generated At (UTC):** `2026-11-19T19:53:27.284621+00:00`
- **All Security Invariants Passed:** **`True`**

## 1. Historical Zerodha / NSE / SEBI Statutory Tariff Schedule Validation

| Segment / Levy | Pre-Oct-2024 (`<= 2024-09-30`) | Post-Oct-2024 (`>= 2024-10-01`) | Incidence |
| :--- | :---: | :---: | :--- |
| Equity Delivery (`CNC`) Brokerage | `INR 0` (`ZERODHA_FREE`) | `INR 0` (`ZERODHA_FREE`) | Buy & Sell |
| Equity Delivery (`CNC`) STT | `0.1000%` | `0.1000%` | Buy & Sell |
| Equity Delivery (`CNC`) Stamp Duty | `0.0150%` | `0.0150%` | Buy only (`0` on Sell) |
| Equity Delivery (`CNC`) DP Charge | `INR 15.93` | `INR 15.93` | Sell >= 1 qty, once/scrip/day (`Buy = INR 0.00`) |
| NSE Cash Exchange Txn Fee (`CNC`/`MIS`) | `0.00325%` | `0.00297%` | Buy & Sell |
| Equity Intraday (`MIS`) Brokerage | `min(INR 20, 0.03%) per order` | `min(INR 20, 0.03%) per order` | Per executed order |
| Equity Intraday (`MIS`) STT | `0.0250%` | `0.0250%` | Sell only |
| Equity Intraday (`MIS`) Stamp Duty | `0.0030%` | `0.0030%` | Buy only |
| Futures (`NRML`) Brokerage | `min(INR 20, 0.03%) per order` | `min(INR 20, 0.03%) per order` | Per executed order |
| Futures (`NRML`) STT | `0.0125%` | `0.0200%` | Sell only |
| Futures (`NRML`) NSE Exchange Fee | `0.00173%` | `0.00173%` | Buy & Sell |
| Futures (`NRML`) Stamp Duty | `0.0020%` | `0.0020%` | Buy only |
| SEBI Turnover Fee | `0.0001%` | `0.0001%` | Buy & Sell |
| GST | `18.0%` | `18.0%` | On `(Brokerage + Exchange + SEBI)` |

### Pre-Oct-2024 vs Post-Oct-2024 Round-Trip Statutory Benchmarks (Zero Slippage)

| Product | Notional (INR) | Pre-Oct-24 RT (INR) | Pre-Oct-24 (bps) | Post-Oct-24 RT (INR) | Post-Oct-24 (bps) | Delta (INR) | Delta (bps) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `DELIVERY` | 100,000 | 238.83 | 23.88 | 238.17 | 23.82 | -0.66 | -0.06 |
| `INTRADAY` | 100,000 | 83.10 | 8.31 | 82.44 | 8.24 | -0.66 | -0.07 |
| `FUTURES` | 100,000 | 66.02 | 6.60 | 73.52 | 7.35 | +7.50 | +0.75 |
| `DELIVERY` | 1,000,000 | 2,244.99 | 22.45 | 2,238.39 | 22.38 | -6.60 | -0.07 |
| `INTRADAY` | 1,000,000 | 406.26 | 4.06 | 399.66 | 4.00 | -6.60 | -0.06 |
| `FUTURES` | 1,000,000 | 235.38 | 2.35 | 310.38 | 3.10 | +75.00 | +0.75 |

## 2. Equity Delivery (`CNC`) Round-Trip Cost & Square-Root Market Impact Matrix

| Order Size | ADV Tier | % ADV | Half-Spread (bps) | Impact (bps) | 1-Way Slip (bps) | RT Statutory (INR) | RT Slip (INR) | RT Total (INR) | RT Total (bps) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| INR 10k | INR 5 Cr | 0.020% | 5.00 | 0.64 | 5.64 | 38.15 | 11.28 | 49.43 | **49.43** |
| INR 10k | INR 25 Cr | 0.004% | 5.00 | 0.28 | 5.28 | 38.15 | 10.56 | 48.71 | **48.71** |
| INR 10k | INR 100 Cr | 0.001% | 5.00 | 0.14 | 5.14 | 38.15 | 10.28 | 48.43 | **48.43** |
| INR 10k | INR 500 Cr | 0.000% | 5.00 | 0.06 | 5.06 | 38.15 | 10.12 | 48.27 | **48.27** |
| INR 50k | INR 5 Cr | 0.100% | 5.00 | 1.42 | 6.42 | 127.06 | 64.20 | 191.26 | **38.25** |
| INR 50k | INR 25 Cr | 0.020% | 5.00 | 0.64 | 5.64 | 127.06 | 56.40 | 183.46 | **36.69** |
| INR 50k | INR 100 Cr | 0.005% | 5.00 | 0.32 | 5.32 | 127.06 | 53.20 | 180.26 | **36.05** |
| INR 50k | INR 500 Cr | 0.001% | 5.00 | 0.14 | 5.14 | 127.06 | 51.40 | 178.46 | **35.69** |
| INR 2L | INR 5 Cr | 0.400% | 5.00 | 2.85 | 7.85 | 460.44 | 314.00 | 774.44 | **38.72** |
| INR 2L | INR 25 Cr | 0.080% | 5.00 | 1.27 | 6.27 | 460.44 | 250.80 | 711.24 | **35.56** |
| INR 2L | INR 100 Cr | 0.020% | 5.00 | 0.64 | 5.64 | 460.44 | 225.60 | 686.04 | **34.30** |
| INR 2L | INR 500 Cr | 0.004% | 5.00 | 0.28 | 5.28 | 460.43 | 211.20 | 671.63 | **33.58** |
| INR 10L | INR 5 Cr | 2.000% | 5.00 | 6.36 | 11.36 | 2,238.55 | 2,272.00 | 4,510.55 | **45.11** |
| INR 10L | INR 25 Cr | 0.400% | 5.00 | 2.85 | 7.85 | 2,238.50 | 1,570.00 | 3,808.50 | **38.09** |
| INR 10L | INR 100 Cr | 0.100% | 5.00 | 1.42 | 6.42 | 2,238.48 | 1,284.00 | 3,522.48 | **35.22** |
| INR 10L | INR 500 Cr | 0.020% | 5.00 | 0.64 | 5.64 | 2,238.47 | 1,128.00 | 3,366.47 | **33.66** |
| INR 50L | INR 5 Cr | 10.000% | 5.00 | 14.23 | 19.23 | 11,129.63 | 19,230.00 | 30,359.63 | **60.72** |
| INR 50L | INR 25 Cr | 2.000% | 5.00 | 6.36 | 11.36 | 11,129.04 | 11,360.00 | 22,489.04 | **44.98** |
| INR 50L | INR 100 Cr | 0.500% | 5.00 | 3.18 | 8.18 | 11,128.80 | 8,180.00 | 19,308.80 | **38.62** |
| INR 50L | INR 500 Cr | 0.100% | 5.00 | 1.42 | 6.42 | 11,128.67 | 6,420.00 | 17,548.67 | **35.10** |

## 3. Realistic OHLCV Bar Execution Simulation Scenarios

| Scenario | Status | Reason | Req Qty | Filled Qty | Raw Price | Exec Fill Price | Slip (bps) | Gap Fill | Total Cost (INR) |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | :---: | ---: |
| `MARKET_BUY_FULL` | `FILLED` | `FILLED_FULL` | 200 | 200 | 1000.00 | 1001.07 | 10.68 | `False` | 451.09 |
| `LIMIT_BUY_PASSIVE_FILLED` | `FILLED` | `FILLED_FULL` | 200 | 200 | 990.00 | 990.00 | 0.00 | `False` | 234.87 |
| `LIMIT_BUY_PARTIAL_VOL_CAP` | `PARTIAL_FILL` | `PARTIAL_VOL_CAP` | 1000 | 500 | 995.00 | 995.00 | 0.00 | `False` | 590.15 |
| `LIMIT_BUY_UNFILLED_NOT_REACHED` | `UNFILLED` | `BUY_LIMIT_NOT_REACHED` | — | 0 | — | — | — | `False` | 0.00 |
| `SL_M_SELL_INTRADAY_TRIGGER` | `FILLED` | `FILLED_FULL` | 200 | 200 | 990.00 | 988.94 | 10.67 | `False` | 432.15 |
| `SL_M_SELL_GAP_THROUGH_OPEN` | `FILLED` | `GAP_THROUGH_OPEN_FILL` | 200 | 200 | 940.00 | 939.20 | 8.54 | `True` | 371.13 |
| `SL_L_SELL_GAP_BEYOND_LIMIT_UNFILLED` | `UNFILLED` | `SL_LIMIT_GAP_THROUGH_UNFILLED` | — | 0 | — | — | — | `False` | 0.00 |
| `ADV_5PCT_THROTTLE_AND_VOL_CAP` | `PARTIAL_FILL` | `PARTIAL_VOL_CAP` | 2500 | 2000 | 1000.00 | 1001.52 | 15.22 | `False` | 5,420.06 |
| `DUST_ORDER_REJECTED` | `REJECTED` | `DUST_ORDER_BELOW_MINIMUM` | — | 0 | — | — | — | `False` | 0.00 |

## 4. Six-Stage Execution Isolation & `ZerodhaBrokerAdapter` Security Audit

| Stage | Responsibility | Broker Capability |
| :--- | :--- | :--- |
| `1. Signal` | Causal cross-sectional alpha & regime-aware target scores at close t (no look-ahead). | `NONE` |
| `2. PortfolioAllocation` | Weight sizing, sector/single-name caps, and turnover-buffered target weights. | `NONE` |
| `3. RiskGate` | Drawdown circuit breakers, gross exposure cap, PIT liquidity & ADV participation checks. | `NONE` |
| `4. PaperOrderPlanner` | Converts delta weights into validated OrderTicket objects (symbol, side, qty, order_type, product). | `OrderConstructor (pure schema validation only)` |
| `5. ExecutionSimulator` | Next-bar open + spread + sqrt market impact + historical Zerodha/SEBI statutory cost simulation. | `OFFLINE_SIMULATION_ONLY` |
| `6. BrokerAdapter` | Six-port ZerodhaBrokerAdapter enforcing OrderApprovalGate and live_execution_enabled=False lock. | `PAPER_SIMULATION / READ_ONLY_MASKED` |

### Verified Security Controls

- **Default `live_execution_enabled`:** `True` (value is `False`)
- **Default `paper_mode`:** `True` (value is `True`)
- **OrderApprovalGate blocks unapproved orders:** `True` (`Order rejected by OrderApprovalGate: ticket was not explicitly approved.`)
- **Dry-run paper submission mode:** `PAPER_SIMULATION` (order id `PAPER-789d5da0ddddf579`)
- **Explicit live-mode override blocked:** `True` (`Live Kite Connect order placement is intentionally blocked in research environment; use dedicated human-approved paper gateway.`)
- **Zero network/HTTP/SDK imports in engine:** `True` (detected: `[]`)
- **Zero credential leakage in payloads/logs:** `True` (markers matched: `[]`)
