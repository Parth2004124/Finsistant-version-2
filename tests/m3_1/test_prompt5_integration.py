"""Comprehensive End-to-End Verification Suite for Moreshwar 3.1 (Prompt 5/5).

Covers all Section 9 mandatory release gates:
- 9.1 Frozen configuration integrity & immutability tests
- 9.2 Point-in-time causality, 3-session confirmation, & prospective boundary tests
- 9.3 Realistic execution, once-only cost accounting, sizing, T+1 settlement, dividends, & rejections
- 9.4 Four-ledger accounting reconciliation (<= INR 0.01) across full lifecycle
- 9.5 Persistence, restart recovery, partial checkpoint rollback, duplicate scheduler idempotency, & audit chain
- 9.6 Hard fail-closed safety tests (live trading, broker placement, capital deployment, live adapters, config tampering, Family A, SAFE_MODE bypass)
- 9.7 Dashboard HTTP API & UI parity tests (strict historical vs prospective separation, paper-only controls, 403 fail-closed)
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from http.server import HTTPServer
import json
from pathlib import Path
import sqlite3
import threading
import urllib.error
import urllib.request

import pytest

from finsistant_v2.m3_1.paper_trading_orchestrator import (
    DEFAULT_PROSPECTIVE_REPORTS_DIR,
    PaperTradingOrchestrator,
    validate_market_data_snapshot,
)
from finsistant_v2.m3_1.paper_trading_risk_monitor import PaperTradingRiskMonitor
from finsistant_v2.m3_1.paper_trading_scheduler import (
    PaperTradingScheduler,
    is_eligible_nse_trading_day,
)
from finsistant_v2.m3_1.prospective_paper_trading_engine import (
    ConfigIntegrityError,
    DEFAULT_FROZEN_CONFIG_PATH,
    EXPECTED_CONFIG_SHA256_PREFIX,
    HISTORICAL_DB_END_DATE,
    INITIAL_PAPER_CAPITAL_INR,
    PaperOnlyExecutionGuard,
    PaperTradingSafetyError,
    ProspectivePaperTradingEngine,
    RECONCILIATION_TOLERANCE_INR,
    compute_config_hashes,
    load_and_verify_frozen_config,
)
import server as finsistant_server


# ---------------------------------------------------------------------------
# Deterministic Market Snapshot Fixtures (Zero Lookahead)
# ---------------------------------------------------------------------------
def _make_market_snapshot(
    *,
    trading_date: str,
    nb_price: float = 280.0,
    nb_sma_base: float = 260.0,
    lb_price: float = 1000.0,
    eq_price_a: float = 500.0,
    eq_low_a: float | None = None,
    eq_div_a: float = 0.0,
    eq_a_valid: bool = True,
    eq_price_b: float = 400.0,
    include_candidates: bool = True,
    candidate_score: float = 0.92,
    strategy_id: str = "STRAT_E_DEFENSIVE_YIELD_LOWVOL",
    nb_valid: bool = True,
) -> dict:
    # Build 200-bar strictly causal NIFTYBEES history ending at `nb_price`
    nb_hist = [nb_sma_base] * 199 + [nb_price]
    low_a = eq_low_a if eq_low_a is not None else eq_price_a * 0.992
    eq_bars = {
        "SEC_SYN_A": {
            "sec_id": "SEC_SYN_A",
            "symbol": "ITC",
            "sector": "Fast Moving Consumer Goods",
            "industry": "Diversified FMCG",
            "is_unknown_sector": False,
            "open": eq_price_a,
            "high": eq_price_a * 1.012,
            "low": low_a,
            "close": eq_price_a * 1.004 if low_a >= eq_price_a * 0.95 else low_a,
            "prev_close": eq_price_a,
            "volume": 2_500_000.0,
            "adv_20d_inr": 900_000_000.0,
            "vol_20d": 0.014,
            "hl_range_pct": 1.6,
            "atr_14": eq_price_a * 0.018,
            "sma_50": eq_price_a * 0.97,
            "low_20_prev": eq_price_a * 0.94,
            "div_per_adj_share": eq_div_a,
            "div_yield_ttm": 0.032,
            "valid": eq_a_valid,
        },
        "SEC_SYN_B": {
            "sec_id": "SEC_SYN_B",
            "symbol": "POWERGRID",
            "sector": "Power",
            "industry": "Power Transmission",
            "is_unknown_sector": False,
            "open": eq_price_b,
            "high": eq_price_b * 1.01,
            "low": eq_price_b * 0.99,
            "close": eq_price_b * 1.003,
            "prev_close": eq_price_b,
            "volume": 3_000_000.0,
            "adv_20d_inr": 850_000_000.0,
            "vol_20d": 0.015,
            "hl_range_pct": 1.8,
            "atr_14": eq_price_b * 0.019,
            "sma_50": eq_price_b * 0.97,
            "low_20_prev": eq_price_b * 0.94,
            "div_per_adj_share": 0.0,
            "div_yield_ttm": 0.036,
            "valid": True,
        },
    }
    cands = (
        [
            {"sec_id": "SEC_SYN_A", "symbol": "ITC", "strategy_id": strategy_id, "score": candidate_score, "stop_ratio": 0.08},
            {"sec_id": "SEC_SYN_B", "symbol": "POWERGRID", "strategy_id": strategy_id, "score": candidate_score - 0.02, "stop_ratio": 0.08},
        ]
        if include_candidates
        else []
    )
    return {
        "trading_date": trading_date,
        "data_timestamp_utc": f"{trading_date}T16:15:00+05:30",
        "niftybees_bar": {
            "open": nb_price,
            "high": nb_price * 1.006,
            "low": nb_price * 0.994,
            "close": nb_price,
            "volume": 5_000_000.0,
            "valid": nb_valid,
        },
        "liquidbees_bar": {
            "open": lb_price,
            "high": lb_price,
            "low": lb_price,
            "close": lb_price,
            "volume": 1_000_000.0,
            "valid": True,
        },
        "niftybees_close_history": nb_hist,
        "equity_bars": eq_bars,
        "family_e_candidates": cands,
        "diagnostic_index_available": False,
    }


# ===========================================================================
# 9.1 FROZEN CONFIGURATION TESTS
# ===========================================================================
def test_9_1_frozen_configuration_hash_immutability_and_parameters(tmp_path: Path) -> None:
    hashes = compute_config_hashes(DEFAULT_FROZEN_CONFIG_PATH)
    assert hashes["canonical_prefix_16"] == EXPECTED_CONFIG_SHA256_PREFIX
    assert hashes["stored_prefix"] == EXPECTED_CONFIG_SHA256_PREFIX

    cfg = load_and_verify_frozen_config(DEFAULT_FROZEN_CONFIG_PATH)
    assert cfg.canonical_sha256.startswith(EXPECTED_CONFIG_SHA256_PREFIX)
    assert cfg.live_trading_enabled is False
    assert cfg.broker_order_placement_enabled is False
    assert cfg.automatic_capital_deployment_enabled is False

    # Frozen strategy & allocation parameters
    assert cfg.regime_policy.policy_id == "G_SIMPLE_TREND_BASELINE"
    assert cfg.regime_policy.allocation_confirm_days == 3
    rules = {r[0]: r[1:] for r in cfg.regime_policy.regime_rules}
    assert rules["BULL_TREND"][:4] == (60.0, 30.0, 10.0, True)
    assert rules["BEAR_TREND"][:4] == (45.0, 10.0, 45.0, False)

    # Frozen execution intelligence parameters
    ec = cfg.exec_cfg
    assert ec.config_id == "EXP_E_COMBINED_EXEC_INTEL"
    assert ec.active_sleeves == ("STRAT_E_DEFENSIVE_YIELD_LOWVOL",)
    assert ec.hurdle_mode == "FIXED_200"
    assert ec.replacement_policy == "NO_PROACTIVE_ROTATION"
    assert ec.stop_model == "C_VOL_ADJUSTED_TRAILING"
    assert ec.execution_mode == "NEXT_SESSION_OPEN"
    assert ec.sizing_mode == "B_VOLATILITY_ADJUSTED"
    assert ec.core_rebal_band_pct == 8.5
    assert ec.liquid_sweep_min_cash_inr == 5000.0
    assert ec.settlement_lag_days == 1
    assert ec.enable_dividend_accounting is True

    # Immutability checks (cannot mutate VerifiedFrozenConfig or raw_config)
    with pytest.raises((FrozenInstanceError, AttributeError)):
        cfg.live_trading_enabled = True  # type: ignore[misc]
    with pytest.raises(TypeError):
        cfg.raw_config["live_trading_enabled"] = True  # type: ignore[index]

    # Tampered copy on disk MUST fail closed
    tampered = tmp_path / "tampered_config.json"
    raw = json.loads(DEFAULT_FROZEN_CONFIG_PATH.read_text(encoding="utf-8"))
    raw["execution_config"]["core_rebal_band_pct"] = 5.0
    tampered.write_text(json.dumps(raw, indent=2), encoding="utf-8")
    with pytest.raises(ConfigIntegrityError, match="CONFIG_HASH_MISMATCH"):
        load_and_verify_frozen_config(tampered)


# ===========================================================================
# 9.2 CAUSALITY & PROSPECTIVE BOUNDARY TESTS
# ===========================================================================
def test_9_2_causality_3day_confirmation_next_open_and_prospective_boundary(tmp_path: Path) -> None:
    eng = ProspectivePaperTradingEngine(
        db_path=tmp_path / "causality.db",
        prospective_start_date=HISTORICAL_DB_END_DATE,
    )

    # 1. Historical dates CANNOT be labelled PROSPECTIVE_PAPER
    with pytest.raises(PaperTradingSafetyError, match="PROSPECTIVE_BOUNDARY_VIOLATION"):
        eng.process_trading_session(
            trading_date="2026-11-19",
            market_snapshot=_make_market_snapshot(trading_date="2026-11-19"),
            observation_mode="PROSPECTIVE_PAPER",
        )

    # 2. Historical validation dry-run NEVER counts toward prospective sessions!
    r_hist = eng.process_trading_session(
        trading_date="2026-11-18",
        market_snapshot=_make_market_snapshot(trading_date="2026-11-18"),
        observation_mode="HISTORICAL_VALIDATION",
    )
    assert r_hist["status"] == "COMPLETED"
    snap0 = eng.get_full_ledger_snapshot()
    assert snap0["prospective_validation"]["completed_unseen_sessions"] == 0

    # 3. Signal generated at Day 1 close does NOT fill on Day 1 — fills strictly at Day 2 NEXT_SESSION_OPEN!
    eng2 = ProspectivePaperTradingEngine(db_path=tmp_path / "causality2.db", prospective_start_date="2026-11-19")
    res_d1 = eng2.process_trading_session(
        trading_date="2026-11-20",
        market_snapshot=_make_market_snapshot(trading_date="2026-11-20", nb_price=280.0, nb_sma_base=260.0),
    )
    assert res_d1["observation_mode"] == "PROSPECTIVE_PAPER"
    assert len(res_d1["orders"]) == 0  # Zero same-day fills!
    assert any(o["kind"] == "TACTICAL_BUY" for o in res_d1["pending_next_session_orders"])
    assert any(o["kind"] == "CORE_REBALANCE" for o in res_d1["pending_next_session_orders"])

    # Day 2 open executes Day 1 queued orders at Day 2 open price (not Day 1 price!)
    res_d2 = eng2.process_trading_session(
        trading_date="2026-11-23",
        market_snapshot=_make_market_snapshot(trading_date="2026-11-23", nb_price=285.0, nb_sma_base=260.0, eq_price_a=515.0),
    )
    filled_buys = [o for o in res_d2["orders"] if o["instrument_class"] == "DELIVERY" and o["side"] == "BUY"]
    assert len(filled_buys) >= 1
    assert abs(filled_buys[0]["expected_price"] - 515.0) < 0.01

    # 4. 3-session Bear confirmation hysteresis: Days 1 & 2 below SMA200 stay BULL_TREND; Day 3 confirms BEAR_TREND (45/10/45)!
    r_b1 = eng2.process_trading_session(
        trading_date="2026-11-25",
        market_snapshot=_make_market_snapshot(trading_date="2026-11-25", nb_price=240.0, nb_sma_base=270.0),
    )
    assert r_b1["nav_snapshot"]["raw_regime"] == "BEAR_TREND"
    assert r_b1["nav_snapshot"]["confirmed_regime"] == "BULL_TREND"
    assert r_b1["nav_snapshot"]["target_core_pct"] == 60.0

    r_b2 = eng2.process_trading_session(
        trading_date="2026-11-26",
        market_snapshot=_make_market_snapshot(trading_date="2026-11-26", nb_price=238.0, nb_sma_base=270.0),
    )
    assert r_b2["nav_snapshot"]["confirmed_regime"] == "BULL_TREND"
    assert r_b2["nav_snapshot"]["regime_confirm_streak"] == 2

    r_b3 = eng2.process_trading_session(
        trading_date="2026-11-27",
        market_snapshot=_make_market_snapshot(trading_date="2026-11-27", nb_price=236.0, nb_sma_base=270.0),
    )
    assert r_b3["nav_snapshot"]["confirmed_regime"] == "BEAR_TREND"
    assert r_b3["nav_snapshot"]["target_core_pct"] == 45.0
    assert r_b3["nav_snapshot"]["target_tactical_pct"] == 10.0
    assert r_b3["nav_snapshot"]["target_liquid_pct"] == 45.0
    assert r_b3["nav_snapshot"]["allow_new_entries"] == 0

    # 5. Missing/invalid market data triggers SAFE_MODE and creates NO artificial signals
    r_bad = eng2.process_trading_session(
        trading_date="2026-11-30",
        market_snapshot=_make_market_snapshot(trading_date="2026-11-30", nb_valid=False),
    )
    assert r_bad["status"] == "SAFE_MODE_INVALID_DATA"
    assert eng2.get_engine_state()["engine_state"] == "SAFE_MODE"


# ===========================================================================
# 9.3 & 9.4 EXECUTION, DIVIDENDS, SETTLEMENT, REJECTIONS & FOUR-LEDGER ACCOUNTING
# ===========================================================================
def test_9_3_and_9_4_execution_dividends_settlement_and_exact_reconciliation(tmp_path: Path) -> None:
    orch = PaperTradingOrchestrator(
        db_path=tmp_path / "acct.db",
        reports_dir=tmp_path / "reports",
        prospective_start_date="2026-11-19",
    )

    # Day 1: Queue Core + Family E buys + Liquid sweep
    c1 = orch.run_daily_cycle(
        trading_date="2026-11-20",
        market_snapshot=_make_market_snapshot(trading_date="2026-11-20", eq_price_a=500.0, eq_price_b=400.0),
    )
    assert c1["reconciliation"]["passed"] is True
    assert c1["reconciliation"]["reconciliation_diff_inr"] <= RECONCILIATION_TOLERANCE_INR

    # Day 2: Execute fills (Core NIFTYBEES + ITC + POWERGRID + LIQUIDBEES)
    c2 = orch.run_daily_cycle(
        trading_date="2026-11-23",
        market_snapshot=_make_market_snapshot(trading_date="2026-11-23", eq_price_a=502.0, eq_price_b=402.0),
    )
    assert c2["reconciliation"]["passed"] is True
    assert c2["reconciliation"]["reconciliation_diff_inr"] <= RECONCILIATION_TOLERANCE_INR

    # Day 3: Ex-dividend on ITC (INR 7.50/share) + overnight LIQUIDBEES yield accrual
    c3 = orch.run_daily_cycle(
        trading_date="2026-11-25",
        market_snapshot=_make_market_snapshot(trading_date="2026-11-25", eq_price_a=506.0, eq_div_a=7.50, eq_price_b=405.0),
    )
    assert c3["reconciliation"]["passed"] is True
    assert c3["reconciliation"]["cum_dividends_inr"] > 0.0
    assert c3["reconciliation"]["reconciliation_diff_inr"] <= RECONCILIATION_TOLERANCE_INR

    # Day 4: Severe stop-loss hit on ITC -> creates T+1 PENDING settlement + DP charge once
    c4 = orch.run_daily_cycle(
        trading_date="2026-11-26",
        market_snapshot=_make_market_snapshot(trading_date="2026-11-26", eq_price_a=490.0, eq_low_a=420.0, eq_price_b=408.0),
    )
    assert c4["reconciliation"]["passed"] is True
    assert c4["reconciliation"]["unsettled_cash_inr"] > 0.0  # T+1 settlement pending!
    assert c4["reconciliation"]["reconciliation_diff_inr"] <= RECONCILIATION_TOLERANCE_INR

    # Day 5: T+1 settlement releases into settled_cash; verify exact 1-to-1 cost ledger & <= 0.01 INR diff
    c5 = orch.run_daily_cycle(
        trading_date="2026-11-27",
        market_snapshot=_make_market_snapshot(trading_date="2026-11-27", eq_price_a=430.0, eq_price_b=410.0, include_candidates=False),
    )
    assert c5["reconciliation"]["passed"] is True
    assert c5["reconciliation"]["unsettled_cash_inr"] == 0.0
    assert c5["reconciliation"]["reconciliation_diff_inr"] <= RECONCILIATION_TOLERANCE_INR

    snap = orch.engine.get_full_ledger_snapshot()
    filled_orders = [o for o in snap["simulated_orders"] if o["execution_status"] in ("FILLED", "PARTIAL")]
    assert len(filled_orders) == len(snap["transaction_cost_ledger"])
    assert len({c["order_id"] for c in snap["transaction_cost_ledger"]}) == len(filled_orders)

    # Verify all 12 Section 8 daily report files exist on disk!
    for fname in (
        "paper_trading_daily_report.json",
        "paper_trading_daily_report.csv",
        "portfolio_snapshot.json",
        "open_positions.csv",
        "simulated_orders.csv",
        "transaction_cost_ledger.csv",
        "dividend_ledger.csv",
        "settlement_ledger.csv",
        "risk_alerts.json",
        "data_quality_report.json",
        "configuration_integrity.json",
        "prospective_validation_progress.json",
    ):
        assert (tmp_path / "reports" / fname).exists(), f"Missing Section 8 report: {fname}"

    # Failed execution (missing bar on next open) creates ZERO phantom positions!
    eng_fail = ProspectivePaperTradingEngine(db_path=tmp_path / "fail_exec.db", prospective_start_date="2026-11-19")
    eng_fail.process_trading_session(
        trading_date="2026-11-20",
        market_snapshot=_make_market_snapshot(trading_date="2026-11-20"),
    )
    r_fail = eng_fail.process_trading_session(
        trading_date="2026-11-23",
        market_snapshot=_make_market_snapshot(trading_date="2026-11-23", eq_a_valid=False),
    )
    failed_a = [o for o in r_fail["orders"] if o["sec_id"] == "SEC_SYN_A" and o["execution_status"] == "FAILED"]
    assert len(failed_a) == 1
    assert failed_a[0]["total_cost_inr"] == 0.0
    held_sids = {p["sec_id"] for p in eng_fail.get_full_ledger_snapshot()["open_tactical_positions"]}
    assert "SEC_SYN_A" not in held_sids


# ===========================================================================
# 9.5 PERSISTENCE, RESTART, CRASH RECOVERY & DUPLICATE SCHEDULER IDEMPOTENCY
# ===========================================================================
def test_9_5_persistence_restart_crash_recovery_and_scheduler_idempotency(tmp_path: Path) -> None:
    db_file = tmp_path / "persist.db"
    rep_dir = tmp_path / "reports"

    orch1 = PaperTradingOrchestrator(db_path=db_file, reports_dir=rep_dir, prospective_start_date="2026-11-19")
    sched1 = PaperTradingScheduler(orchestrator=orch1, state_dir=tmp_path / "sched")

    # Weekend & NSE holiday cleanly skipped
    ok_sat, r_sat = is_eligible_nse_trading_day("2026-11-21")
    assert ok_sat is False and r_sat == "WEEKEND_NON_TRADING_DAY"
    ok_hol, r_hol = is_eligible_nse_trading_day("2026-11-24")  # Gurunanak Jayanti
    assert ok_hol is False and r_hol == "NSE_MARKET_HOLIDAY"

    # Delayed data -> WAITING_FOR_DATA (does not fabricate)
    res_wait = sched1.run_scheduled_session(trading_date="2026-11-20", market_data_provider=lambda _d: None)
    assert res_wait["scheduler_status"] == "WAITING_FOR_DATA"

    # Run trading session via scheduler
    snap_d1 = _make_market_snapshot(trading_date="2026-11-20")
    res_run1 = sched1.run_scheduled_session(trading_date="2026-11-20", market_data_provider=lambda _d: snap_d1)
    assert res_run1["scheduler_status"] == "COMPLETED"
    nav_after_1 = res_run1["cycle_result"]["reconciliation"]["nav_from_assets_inr"]

    # Duplicate scheduler invocation on SAME date -> idempotent no-op!
    res_dup = sched1.run_scheduled_session(trading_date="2026-11-20", market_data_provider=lambda _d: snap_d1)
    assert res_dup["cycle_result"]["execution_result"]["idempotent_skip"] is True
    assert res_dup["cycle_result"]["reconciliation"]["nav_from_assets_inr"] == nav_after_1

    # Simulated Application Restart: fresh Orchestrator & Scheduler from same SQLite DB
    orch2 = PaperTradingOrchestrator(db_path=db_file, reports_dir=rep_dir, prospective_start_date="2026-11-19")
    assert orch2.engine.reconcile_portfolio()["passed"] is True
    assert orch2.engine.verify_audit_chain()["valid"] is True
    assert orch2.engine.get_full_ledger_snapshot()["prospective_validation"]["completed_unseen_sessions"] == 1

    # Simulate interrupted DB write after STARTED checkpoint -> enters SAFE_MODE on next attempt -> recovers cleanly
    with pytest.raises(RuntimeError, match="SIMULATED_DB_WRITE_INTERRUPTION"):
        orch2.engine.process_trading_session(
            trading_date="2026-11-23",
            market_snapshot=_make_market_snapshot(trading_date="2026-11-23"),
            simulate_db_failure_after_checkpoint=True,
        )
    res_after_crash = orch2.engine.process_trading_session(
        trading_date="2026-11-23",
        market_snapshot=_make_market_snapshot(trading_date="2026-11-23"),
    )
    assert res_after_crash["status"] == "SAFE_MODE_INCOMPLETE_CHECKPOINT"
    assert orch2.engine.get_engine_state()["engine_state"] == "SAFE_MODE"

    # Cannot bypass SAFE_MODE directly
    with pytest.raises(PaperTradingSafetyError, match="HARD_SAFE_MODE_GUARD"):
        orch2.engine.set_operational_state("RUNNING")

    # Verified recovery succeeds once reconciled and allows clean execution of 2026-11-23
    # (Reconciliation initially flags the incomplete checkpoint; mark checkpoint FAILED then recover)
    conn = sqlite3.connect(str(db_file))
    conn.execute("UPDATE session_checkpoints SET phase = 'FAILED' WHERE trading_date = '2026-11-23'")
    conn.commit()
    conn.close()
    rec = orch2.engine.recover_from_safe_mode(operator_note="verified_crash_recovery")
    assert rec["recovered"] is True
    res_clean = orch2.engine.process_trading_session(
        trading_date="2026-11-23",
        market_snapshot=_make_market_snapshot(trading_date="2026-11-23"),
    )
    assert res_clean["status"] == "COMPLETED"
    assert orch2.engine.reconcile_portfolio()["passed"] is True

    # Append-only audit log SQLite triggers prohibit UPDATE / DELETE
    conn = sqlite3.connect(str(db_file))
    with pytest.raises(sqlite3.DatabaseError, match="APPEND_ONLY_AUDIT_LOG"):
        conn.execute("DELETE FROM decision_audit_log WHERE log_id = 1")
    with pytest.raises(sqlite3.DatabaseError, match="APPEND_ONLY_AUDIT_LOG"):
        conn.execute("UPDATE decision_audit_log SET event_type = 'TAMPERED' WHERE log_id = 1")
    conn.close()

    # Missed session is recorded explicitly without fabricating fills
    sched2 = PaperTradingScheduler(orchestrator=orch2, state_dir=tmp_path / "sched")
    res_miss = sched2.run_scheduled_session(
        trading_date="2026-11-25",
        market_data_provider=lambda _d: None,
        mark_missed_if_unavailable=True,
    )
    assert res_miss["missed_recorded"] is True
    assert res_miss["scheduler_status"] == "FAILED"


# ===========================================================================
# 9.6 HARD FAIL-CLOSED SAFETY TESTS
# ===========================================================================
def test_9_6_all_prohibited_operations_fail_closed(tmp_path: Path) -> None:
    guard = PaperOnlyExecutionGuard()

    # 1. Enable live trading
    with pytest.raises(PaperTradingSafetyError, match="HARD_PAPER_ONLY_GUARD"):
        ProspectivePaperTradingEngine(db_path=tmp_path / "s1.db", live_trading_enabled=True)

    # 2. Activate broker order placement
    with pytest.raises(PaperTradingSafetyError, match="HARD_PAPER_ONLY_GUARD"):
        ProspectivePaperTradingEngine(db_path=tmp_path / "s2.db", broker_order_placement_enabled=True)
    with pytest.raises(PaperTradingSafetyError, match="HARD_PAPER_ONLY_GUARD"):
        guard.submit_live_order({"symbol": "ITC", "qty": 10})

    # 3. Activate automatic capital deployment
    with pytest.raises(PaperTradingSafetyError, match="HARD_PAPER_ONLY_GUARD"):
        PaperTradingOrchestrator(db_path=tmp_path / "s3.db", automatic_capital_deployment_enabled=True)
    with pytest.raises(PaperTradingSafetyError, match="HARD_PAPER_ONLY_GUARD"):
        guard.deploy_live_capital(100_000.0)

    # 4. Load live execution adapter
    with pytest.raises(PaperTradingSafetyError, match="HARD_PAPER_ONLY_GUARD"):
        ProspectivePaperTradingEngine(db_path=tmp_path / "s4.db", broker_adapter=object())
    with pytest.raises(PaperTradingSafetyError, match="HARD_PAPER_ONLY_GUARD"):
        guard.load_live_execution_adapter("zerodha_kite")

    # 5. Override frozen configuration
    orch = PaperTradingOrchestrator(db_path=tmp_path / "s5.db", reports_dir=tmp_path / "rep5")
    with pytest.raises(PaperTradingSafetyError, match="HARD_PAPER_ONLY_GUARD"):
        orch.execute_paper_control("override_config")

    # 6. Activate suspended Family A
    with pytest.raises(PaperTradingSafetyError, match="HARD_STRATEGY_GUARD"):
        orch.engine.process_trading_session(
            trading_date="2026-11-20",
            market_snapshot=_make_market_snapshot(
                trading_date="2026-11-20",
                strategy_id="STRAT_A_CS_MOMENTUM",
            ),
        )
    with pytest.raises(PaperTradingSafetyError, match="HARD_STRATEGY_GUARD"):
        orch.execute_paper_control("start_monitoring", payload={"strategy_ids": ["STRAT_A_CS_MOMENTUM"]})

    # 7. Bypass critical risk halt (SAFE_MODE)
    orch.engine.enter_safe_mode("CRITICAL_TEST_HALT")
    with pytest.raises(PaperTradingSafetyError, match="HARD_SAFE_MODE_GUARD"):
        orch.execute_paper_control("start_monitoring")


# ===========================================================================
# 9.7 DASHBOARD HTTP API & UI PARITY TESTS
# ===========================================================================
def test_9_7_dashboard_api_separation_controls_and_ui_parity(tmp_path: Path) -> None:
    # Bind isolated temp orchestrator/scheduler into server globals for test HTTP server
    prev_orch = finsistant_server.M3_1_ORCHESTRATOR
    prev_sched = finsistant_server.M3_1_SCHEDULER
    test_orch = PaperTradingOrchestrator(
        db_path=tmp_path / "api_test.db",
        reports_dir=tmp_path / "api_reports",
        prospective_start_date="2026-11-19",
    )
    test_sched = PaperTradingScheduler(orchestrator=test_orch, state_dir=tmp_path / "api_sched")
    finsistant_server.M3_1_ORCHESTRATOR = test_orch
    finsistant_server.M3_1_SCHEDULER = test_sched

    srv = HTTPServer(("127.0.0.1", 0), finsistant_server.FinsistantV2HTTPHandler)
    port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        # 1. GET /api/m3_1/dashboard
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/m3_1/dashboard", timeout=5) as resp:
            assert resp.status == 200
            dash = json.loads(resp.read().decode("utf-8"))

        assert dash["banner"] == "PAPER TRADING ONLY — LIVE TRADING DISABLED"
        assert dash["safety_posture"]["live_trading_enabled"] is False
        assert dash["configuration_integrity"]["verified"] is True
        assert dash["configuration_integrity"]["canonical_prefix_16"] == EXPECTED_CONFIG_SHA256_PREFIX

        # Historical vs Prospective strict separation
        assert dash["historical_research_baseline"]["data_classification"] == "HISTORICAL_BACKTEST_RESEARCH_ONLY"
        assert dash["historical_research_baseline"]["counts_toward_prospective_validation"] is False
        assert dash["prospective_paper_trading"]["data_classification"] == "PROSPECTIVE_PAPER_TRADING_ONLY"
        assert dash["prospective_paper_trading"]["validation_tracker"]["completed_unseen_sessions"] == 0
        assert dash["prospective_paper_trading"]["portfolio_overview"]["current_nav_inr"] == INITIAL_PAPER_CAPITAL_INR
        assert dash["operational_controls"]["live_order_button_present"] is False

        # 2. Paper-only control POST /api/m3_1/control
        req_ok = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/m3_1/control",
            data=json.dumps({"action": "pause_entries"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req_ok, timeout=5) as resp:
            assert resp.status == 200
            ctrl_res = json.loads(resp.read().decode("utf-8"))
        assert ctrl_res["engine_state"]["engine_state"] == "PAUSED"
        assert ctrl_res["live_trading_enabled"] is False

        # 3. Live-trading control attempt MUST fail closed with HTTP 403
        req_bad = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/m3_1/control",
            data=json.dumps({"action": "start_monitoring", "live_trading_enabled": True}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(req_bad, timeout=5)
        assert exc_info.value.code == 403
    finally:
        srv.shutdown()
        srv.server_close()
        finsistant_server.M3_1_ORCHESTRATOR = prev_orch
        finsistant_server.M3_1_SCHEDULER = prev_sched

    # 4. Verify HTML/JS UI parity & paper-only banner across root, web/, and mobile-app/
    root = Path(__file__).resolve().parent.parent.parent
    r_html = (root / "index.html").read_text(encoding="utf-8")
    w_html = (root / "web" / "index.html").read_text(encoding="utf-8")
    m_html = (root / "mobile-app" / "app" / "src" / "main" / "assets" / "index.html").read_text(encoding="utf-8")
    assert r_html == w_html == m_html
    assert "intelSectionV31" in r_html
    assert "PAPER TRADING ONLY — LIVE TRADING DISABLED" in r_html
    # Confirm M3.0 section remains intact
    assert "intelSectionV30" in r_html

    r_js = (root / "app.js").read_text(encoding="utf-8")
    w_js = (root / "web" / "app.js").read_text(encoding="utf-8")
    m_js = (root / "mobile-app" / "app" / "src" / "main" / "assets" / "app.js").read_text(encoding="utf-8")
    assert r_js == w_js == m_js
    assert "function renderMoreshwarV31Dashboard(" in r_js
    assert "function renderMoreshwarV30Dashboard(" in r_js

    # Confirm all 12 production prospective reports exist in reports/moreshwar_3_1/prospective_paper_trading/
    for fname in (
        "paper_trading_daily_report.json",
        "paper_trading_daily_report.csv",
        "portfolio_snapshot.json",
        "open_positions.csv",
        "simulated_orders.csv",
        "transaction_cost_ledger.csv",
        "dividend_ledger.csv",
        "settlement_ledger.csv",
        "risk_alerts.json",
        "data_quality_report.json",
        "configuration_integrity.json",
        "prospective_validation_progress.json",
    ):
        assert (DEFAULT_PROSPECTIVE_REPORTS_DIR / fname).exists()


def test_9_8_risk_monitor_corruption_detection_and_oct2024_tariff_transition(tmp_path: Path) -> None:
    """Verify independent risk monitor catches SQLite cash corruption -> SAFE_MODE & verifies Oct-2024 tariffs."""
    db_file = tmp_path / "corrupt.db"
    orch = PaperTradingOrchestrator(db_path=db_file, reports_dir=tmp_path / "rep", prospective_start_date="2026-11-19")

    # Verify exact Oct 1, 2024 Indian statutory + CDSL DP tariff transition in loaded cost engine
    ce = orch.engine.cost_engine
    assert ce.applicable_exchange_rate("2024-09-30") == 0.0000325
    assert ce.applicable_exchange_rate("2024-10-01") == 0.0000297
    assert ce.applicable_dp_charge_inr("2024-09-30") == 15.93
    assert ce.applicable_dp_charge_inr("2024-10-01") == 15.34

    # Verify insufficient sample-size labelling on fresh portfolio
    snap0 = orch.engine.get_full_ledger_snapshot()
    rm0 = orch.risk_monitor.evaluate_risk_and_alerts(snap0, config_verified=True)
    assert rm0["overall_health"] == "HEALTHY"
    assert rm0["should_enter_safe_mode"] is False
    assert rm0["metrics"]["cagr_pct"] is None
    assert "INSUFFICIENT_SAMPLE" in rm0["metrics"]["cagr_sample_status"]
    assert rm0["metrics"]["rolling_sharpe"] is None
    assert rm0["metrics"]["gold_exposure_pct"] == 0.0

    # Inject a Rs 250.00 silent cash corruption directly into SQLite -> reconciliation & risk monitor MUST trigger SAFE_MODE!
    conn = sqlite3.connect(str(db_file))
    conn.execute("UPDATE portfolio_state SET settled_cash_inr = settled_cash_inr - 250.0 WHERE singleton_id = 1")
    conn.commit()
    conn.close()

    snap_bad = orch.engine.get_full_ledger_snapshot()
    assert snap_bad["reconciliation"]["passed"] is False
    assert snap_bad["reconciliation"]["reconciliation_diff_inr"] >= 249.99

    rm_bad = orch.risk_monitor.evaluate_risk_and_alerts(snap_bad, config_verified=True)
    assert rm_bad["overall_health"] == "CRITICAL_SAFE_MODE"
    assert rm_bad["should_enter_safe_mode"] is True
    assert any(a["code"] == "PORTFOLIO_RECONCILIATION_FAILURE" for a in rm_bad["alerts"])

