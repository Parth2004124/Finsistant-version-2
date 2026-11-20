# MORESHWAR 3.1 (PROMPT 5/5) — FINAL INTEGRATION, PROSPECTIVE PAPER-TRADING & RELEASE CANDIDATE AUDIT

> [!IMPORTANT]
> **RELEASE CANDIDATE:** `Moreshwar 3.1 (v3.1.0-rc1)` — **PAPER TRADING ONLY — LIVE TRADING DISABLED**
> - **Frozen Strategy Policy:** `G_SIMPLE_TREND_BASELINE` (`NIFTYBEES >= SMA200`, 3-session confirmation)
> - **Frozen Execution Profile:** `EXP_E_COMBINED_EXEC_INTEL` (`STRAT_E_DEFENSIVE_YIELD_LOWVOL` only; `STRAT_A_CS_MOMENTUM` permanently suspended & hard-blocked)
> - **Frozen Config Canonical SHA256:** `47a30f4bbd004c5f238cf97c549b4d41a10232c4cdf2a4966efecf75bf2d255a` (expected prefix `47a30f4bbd004c5f` — **VERIFIED**)
> - **Frozen Config Raw File SHA256:** `658d17894618de9dc974b8748e1b2d25ba2e8020d5fc4dc99f9855b1a835f068` (**VERIFIED**)
> - **Live Trading / Broker Placement / Auto-Capital Deployment:** **`false` / `false` / `false`** (hard-enforced at config, engine, orchestrator, scheduler, HTTP API, and UI layers)
> - **Prospective Unseen Sessions Completed:** **`0 / 60` minimum (`0 / 120` target)** as of `2026-11-19` — strictly separated from historical backtests (`2021-07 -> 2026-11-19`).

---

## 1. Absolute Preservation & Zero-Regression Verification

- **Moreshwar 3.0 (`finsistant_v2/m3/`, `reports/moreshwar_3_0/`)**: **100% untouched** (`git status --short` confirms zero edits).
- **Moreshwar 3.1 Prompts 1–4 Modules & Frozen Reports**: **100% preserved** (`forensic_audit`, `capital_allocation`, `execution_intelligence`, `strategy_intelligence`).
- **Only Existing Files Modified (Strictly Integration & UI Mirroring)**:
  1. `finsistant_v2/m3_1/__init__.py` — exported new Prompt 5 integration symbols cleanly without altering any Prompt 1–4 export.
  2. `server.py` — added `GET /api/m3_1/dashboard`, `GET /api/m3_1/reports`, paper-only `POST /api/m3_1/control` (HTTP `403` fail-closed on prohibited live/strategy actions), and `"moreshwar_v3_1"` in `_build_full_state_payload()`.
  3. `index.html`, `app.js` (mirrored byte-for-byte into `web/` and `mobile-app/app/src/main/assets/`) — added dedicated **Section 3.1 (`intelSectionV31`)** while keeping M3.0 (`intelSectionV30`) 100% intact.

---

## 2. Frozen Configuration & Instrument Mapping Disclosure

Authoritative source: `reports/moreshwar_3_1/strategy_intelligence/FROZEN_PROSPECTIVE_PAPER_TRADING_CONFIG.json`

| Parameter | Frozen Authoritative Value | Verification Status |
| :--- | :--- | :---: |
| Canonical Payload SHA256 Prefix | `47a30f4bbd004c5f` | **MATCH** |
| Selected Regime Policy | `G_SIMPLE_TREND_BASELINE` (`NIFTYBEES >= SMA200`, `3`-session confirm) | **VERIFIED** |
| Bull Allocation (`BULL_TREND`) | `60% Core NIFTYBEES` / `30% Tactical Family E` / `10% Liquid LIQUIDBEES` / `0% Gold` | **VERIFIED** |
| Bear Allocation (`BEAR_TREND`) | `45% Core NIFTYBEES` / `10% Tactical Family E (New Entries HALTED)` / `45% Liquid LIQUIDBEES` / `0% Gold` | **VERIFIED** |
| Execution Profile | `EXP_E_COMBINED_EXEC_INTEL` (`FIXED_200` EV hurdle, `NO_PROACTIVE_ROTATION`, `C_VOL_ADJUSTED_TRAILING` stop, `NEXT_SESSION_OPEN` fill, `B_VOLATILITY_ADJUSTED` sizing, `8.5%` core band, `INR 5,000` liquid sweep floor, `T+1` settlement, dividend accounting) | **VERIFIED** |
| Statutory Tariff Schedule | `m3_1_india_statutory_2021_2026_v1` (Oct 1, 2024 switch: NSE CM `0.00325% -> 0.00297%`, CDSL DP `INR 15.93 -> INR 15.34` once per sell-scrip-day) | **VERIFIED** |
| Admitted Tactical Sleeve | `STRAT_E_DEFENSIVE_YIELD_LOWVOL` ONLY | **VERIFIED** |
| Suspended Sleeves (Hard-Blocked) | `STRAT_A_CS_MOMENTUM` (Family A) | **BLOCKED** |
| `live_trading_enabled` / `broker_order_placement_enabled` / `automatic_capital_deployment_enabled` | `false` / `false` / `false` | **ENFORCED** |

> [!NOTE]
> **Instrument Mapping Honesty Note:** Section 2.1 of the Prompt 5 brief listed "Gold 30%/10%" while explicitly directing *"The actual instrument mappings must come from the frozen configuration and existing allocation engine. Do not invent substitute instruments."* In the frozen configuration, `60/30/10` (Bull) and `45/10/45` (Bear) map strictly to **Core ETF (`NIFTYBEES`) / Tactical Equity (`STRAT_E_DEFENSIVE_YIELD_LOWVOL`) / Liquid ETF (`LIQUIDBEES`)**. Gold exposure is explicitly reported as **`0.0%`** across the risk monitor, reports, and dashboard; no substitute Gold instrument was invented.

---

## 3. Integrated Moreshwar 3.1 Architecture

Four modular Prompt 5 production files were added under `finsistant_v2/m3_1/`:

1. **`finsistant_v2/m3_1/prospective_paper_trading_engine.py`**
   - Immutable `VerifiedFrozenConfig` + `load_and_verify_frozen_config` (verifies canonical & raw SHA256 before every session).
   - `PaperOnlyExecutionGuard` (multi-layer fail-closed guard against live trading, broker adapters, capital deployment, suspended Family A, and SAFE_MODE bypass).
   - `ProspectivePaperTradingEngine`: SQLite WAL-backed durable ledger (`data/m3_1_paper_trading/prospective_paper_ledger.db`) with atomic transactions, idempotent `session_checkpoints`, 1-to-1 unique `transaction_cost_ledger`, deduplicated `dividend_ledger`, `settlement_ledger`, four-ledger reconciliation (`<= INR 0.01`), and trigger-protected append-only hash-chained `decision_audit_log`.
   - Enforces strict separation across `HISTORICAL_RESEARCH`, `HISTORICAL_VALIDATION`, and `PROSPECTIVE_PAPER` (`trading_date > 2026-11-19`).
2. **`finsistant_v2/m3_1/paper_trading_risk_monitor.py`**
   - Independent Section 5.1 portfolio risk metrics with explicit sample-size sufficiency flags (`INSUFFICIENT_SAMPLE` for CAGR `< 60` sessions and rolling volatility/Sharpe/Sortino `< 20` sessions).
   - Complete Section 5.2 alert catalog + Section 5.3 automatic `SAFE_MODE` fail-safe evaluation reusing Prompt 4's `regime_monitoring.py`.
3. **`finsistant_v2/m3_1/paper_trading_orchestrator.py`**
   - Central integration pipeline: PIT data validation (`validate_market_data_snapshot`) -> causal snapshot builder (`build_pit_snapshot_from_context`) -> `ProspectivePaperTradingEngine` -> `PaperTradingRiskMonitor` -> 12-file daily report exporter + append-only `daily_audit_trail.jsonl` -> strictly separated historical-vs-prospective dashboard payload -> paper-only control dispatcher.
4. **`finsistant_v2/m3_1/paper_trading_scheduler.py`**
   - Single-worker `fcntl.flock` lock-protected daily NSE trading-session workflow (`IDLE | WAITING_FOR_DATA | PROCESSING | COMPLETED | FAILED | SAFE_MODE`).
   - NSE weekend & holiday calendar awareness (`is_eligible_nse_trading_day`), explicit missed-session recording (`MISSED` — zero fabrication), and duplicate-invocation idempotency.

---

## 4. Data Boundaries & Point-in-Time Integrity

| Dataset / Feed | Verified Boundary | Engine Handling |
| :--- | :--- | :--- |
| NSE PIT `daily_bars` (2,832 equities + `NIFTYBEES` + `LIQUIDBEES`) | `2021-01-01 -> 2026-11-19` (`1,454` sessions) | Authoritative historical archive (`<= 2026-11-19`); strictly excluded from prospective counts |
| `daily_index_bars` (`India VIX`, `Nifty GS 10Yr`, sector indices) | Ends `2025-10-10` | Explicitly tagged `PARTIAL_DIAGNOSTIC_ONLY`; never fabricated or forward-filled |
| Intraday tick / minute data | Unavailable | `NEXT_SESSION_OPEN` daily bars only; no intraday approximations |
| PIT macro timestamps & corporate financials | Unavailable | Excluded cleanly |

---

## 5. Daily Reporting & Audit Trail Deliverables (Section 8)

All 12 required outputs (plus append-only `daily_audit_trail.jsonl`) are generated in `reports/moreshwar_3_1/prospective_paper_trading/`:

1. `paper_trading_daily_report.json`
2. `paper_trading_daily_report.csv`
3. `portfolio_snapshot.json`
4. `open_positions.csv`
5. `simulated_orders.csv`
6. `transaction_cost_ledger.csv`
7. `dividend_ledger.csv`
8. `settlement_ledger.csv`
9. `risk_alerts.json`
10. `data_quality_report.json`
11. `configuration_integrity.json`
12. `prospective_validation_progress.json`
13. `daily_audit_trail.jsonl` (hash-chained append-only audit log)

---

## 6. End-to-End Verification & Regression Results (Section 9)

| Verification Gate | Test Suite / Scope | Result |
| :--- | :--- | :---: |
| 9.1 Frozen Config Hash, Immutability & Tamper Rejection | `tests/m3_1/test_prompt5_integration.py` | **PASS** |
| 9.2 Causality, 3-Day Confirmation, Next-Open Fill & Prospective Boundary | `tests/m3_1/test_prompt5_integration.py` | **PASS** |
| 9.3 Realistic Execution, Once-Only Costs, Dividends, T+1 Settlement & Rejections | `tests/m3_1/test_prompt5_integration.py` | **PASS** |
| 9.4 Four-Ledger Accounting Reconciliation (`<= INR 0.01`, observed `INR 0.0000`) | `tests/m3_1/test_prompt5_integration.py` | **PASS** |
| 9.5 Restart, Crash Recovery, Duplicate Scheduler Idempotency, Missed Sessions & Append-Only Audit Triggers | `tests/m3_1/test_prompt5_integration.py` | **PASS** |
| 9.6 Hard Fail-Closed Safety (All 7 Prohibited Live/Config/Family-A/SAFE_MODE Bypass Attempts) | `tests/m3_1/test_prompt5_integration.py` | **PASS** |
| 9.7 Live HTTP API (`GET /api/m3_1/dashboard`, `POST /api/m3_1/control` 200/403) & Root/Web/Mobile UI Parity | `tests/m3_1/test_prompt5_integration.py` | **PASS** |
| 9.8 Independent Risk Monitor SQLite Corruption Detection & Oct-2024 Tariff Transition | `tests/m3_1/test_prompt5_integration.py` | **PASS** |
| Full Repository Regression Suite (M2 + M3.0 + M3.1 Prompts 1–5 + Release Audit) | **20 test suites / 102 tests** | **102 / 102 PASS (0 FAIL)** |

---

## 7. Release Candidate Readiness Decision

- **Operational Infrastructure Status:** **READY FOR PROSPECTIVE PAPER TRADING (`v3.1.0-rc1`)**
- **Live Capital Deployment Status:** **NOT AUTHORIZED — STRICTLY DISABLED**
- **Mandatory Next Gate:** Accumulate a minimum of **`60` unseen prospective NSE trading sessions** (target **`120` sessions**) strictly after `2026-11-19` with unmodified frozen configuration `47a30f4bbd004c5f`, verified `<= INR 0.01` daily reconciliation, and clean risk-monitor health before any future governance review.
