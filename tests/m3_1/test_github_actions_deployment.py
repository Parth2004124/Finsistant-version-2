"""Validation Suite for Moreshwar 3.1 GitHub Actions Unattended Deployment Adaptation.

Tests all 10 mandatory deployment & resilience scenarios:
1. Successful daily execution (causal Family E bar pipeline + DB locking + reconciliation)
2. Duplicate workflow invocation (idempotent skip, zero duplicate orders/dividends/costs)
3. Runner termination (unreleased crash lock + incomplete checkpoint preserves last valid state)
4. Database reconnection & SQLite integrity check across fresh adapter instances
5. Retry after partial execution (stale crash lock fails closed until validated recovery)
6. Missed trading session (explicit MISSED checkpoint, zero fabricated fills)
7. Missing / delayed market data (WAITING_FOR_DATA, zero fabricated fills)
8. Recovery after workflow failure (validated lock/checkpoint clear + SAFE_MODE recovery)
9. Next-session-open execution timing (pending fills execute before new daily signals)
10. Persistent reconciliation (<= INR 0.01), secret redaction & GitHub Actions YAML security audit
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
import json
from pathlib import Path
import re
import sqlite3

import numpy as np
import pytest

from finsistant_v2.m3_1.live_nse_eod_feed import compute_family_e_candidates_from_bars
from finsistant_v2.m3_1.paper_trading_orchestrator import PaperTradingOrchestrator
from finsistant_v2.m3_1.paper_trading_scheduler import PaperTradingScheduler
from finsistant_v2.m3_1.persistent_storage_adapter import (
    DEFAULT_STORAGE_CONFIG_PATH,
    DatabaseExecutionLock,
    DatabaseLockError,
    PersistentStorageAdapter,
    redact_sensitive_text,
)
from finsistant_v2.m3_1.prospective_paper_trading_engine import (
    RECONCILIATION_TOLERANCE_INR,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "moreshwar-3-1-paper-trading.yml"


def _build_offline_eod_snapshot(target_date: str, *, px_shift: float = 0.0) -> dict:
    """Build deterministic causal snapshot using `compute_family_e_candidates_from_bars` (zero network)."""
    n_days = 260
    end_d = date.fromisoformat(target_date)
    # Build n_days weekday dates ending exactly on target_date
    dates: list[str] = []
    cur = end_d
    while len(dates) < n_days:
        if cur.weekday() < 5:
            dates.append(cur.isoformat())
        cur -= timedelta(days=1)
    dates.reverse()

    nb_closes = np.linspace(245.0, 280.0 + px_shift, n_days)
    nb_rets = np.diff(nb_closes) / nb_closes[:-1]

    eq_histories = {}
    sectors = [
        ("SEC_GA_1", "ITC", "Fast Moving Consumer Goods", "Diversified FMCG", 450.0, 14.0, 0.008),
        ("SEC_GA_2", "POWERGRID", "Power", "Power Transmission", 320.0, 12.0, 0.009),
        ("SEC_GA_3", "COALINDIA", "Oil Gas & Consumable Fuels", "Coal", 480.0, 32.0, 0.006),
        ("SEC_GA_4", "NTPC", "Power", "Power Generation", 380.0, 18.0, 0.008),
        ("SEC_GA_5", "INFY", "Information Technology", "IT Services", 1600.0, 40.0, 0.012),
        ("SEC_GA_6", "WIPRO", "Information Technology", "IT Services", 540.0, 2.0, 0.028),
        ("SEC_GA_7", "TATASTEEL", "Metals & Mining", "Steel", 160.0, 1.0, 0.032),
        ("SEC_GA_8", "DLF", "Realty", "Realty", 820.0, 2.0, 0.035),
    ]
    rng = np.random.default_rng(42)
    for sid, sym, sec, ind, base_px, annual_div, vol in sectors:
        noise = rng.normal(0.0004, vol, size=n_days)
        c = base_px * np.cumprod(1.0 + noise) + px_shift
        # Keep calm dividend payers near 52W high so Family E approves them cleanly
        if annual_div >= 10.0:
            c = np.linspace(base_px * 0.90, base_px * 1.02 + px_shift, n_days) + rng.normal(0.0, base_px * 0.0018, size=n_days)
        o = c * 0.999
        h = c * 1.006
        l = c * 0.994
        v = np.full(n_days, 3_500_000.0)
        divs = np.zeros(n_days)
        divs[-40] = annual_div
        eq_histories[sid] = {
            "symbol": sym,
            "sector": sec,
            "industry": ind,
            "is_unknown_sector": False,
            "dates": dates,
            "open": o,
            "high": h,
            "low": l,
            "close": c,
            "volume": v,
            "dividends": divs,
        }

    eq_bars, cands, corr_map = compute_family_e_candidates_from_bars(
        target_date=target_date,
        nb_returns=nb_rets,
        equity_histories=eq_histories,
    )
    nb_c = float(nb_closes[-1])
    return {
        "trading_date": target_date,
        "data_timestamp_utc": f"{target_date}T17:00:00+05:30",
        "niftybees_bar": {"open": nb_c, "high": nb_c * 1.005, "low": nb_c * 0.995, "close": nb_c, "volume": 4_000_000.0, "valid": True},
        "liquidbees_bar": {"open": 1000.0, "high": 1000.0, "low": 1000.0, "close": 1000.0, "volume": 500_000.0, "valid": True},
        "niftybees_close_history": nb_closes[-220:].tolist(),
        "equity_bars": eq_bars,
        "family_e_candidates": cands,
        "pairwise_corr_map": corr_map,
        "diagnostic_index_available": False,
    }


def test_ga_1_2_9_10_daily_execution_duplicate_next_open_and_reconciliation(tmp_path: Path) -> None:
    db_file = tmp_path / "gha_paper.db"
    rep_dir = tmp_path / "reports"
    sched_dir = tmp_path / "sched"

    storage = PersistentStorageAdapter(db_path=db_file, database_url="")
    assert storage.backend_mode == "DURABLE_GIT_LEDGER"

    # Day 1 (2026-11-20): pull -> lock -> run -> reconcile -> release -> push
    storage.pull_durable_ledger(owner_run_id="gha-1001")
    storage.db_lock.acquire(owner_run_id="gha-1001", trading_date="2026-11-20")
    orch = PaperTradingOrchestrator(db_path=db_file, reports_dir=rep_dir, prospective_start_date="2026-11-19")
    sched = PaperTradingScheduler(orchestrator=orch, state_dir=sched_dir)

    snap_d1 = _build_offline_eod_snapshot("2026-11-20", px_shift=0.0)
    assert len(snap_d1["family_e_candidates"]) >= 2

    res_d1 = sched.run_scheduled_session(trading_date="2026-11-20", market_data_provider=lambda _d: snap_d1)
    assert res_d1["scheduler_status"] == "COMPLETED"
    # Day 1 generated pending orders for NEXT_SESSION_OPEN — ZERO same-day fills!
    assert len(res_d1["cycle_result"]["execution_result"]["orders"]) == 0
    assert len(res_d1["cycle_result"]["execution_result"]["pending_next_session_orders"]) >= 2
    assert res_d1["cycle_result"]["reconciliation"]["reconciliation_diff_inr"] <= RECONCILIATION_TOLERANCE_INR
    storage.db_lock.release(owner_run_id="gha-1001")
    push1 = storage.push_durable_ledger(owner_run_id="gha-1001")
    assert len(push1["ledger_sha256"]) == 64

    # Duplicate workflow invocation on SAME date (2026-11-20) -> idempotent skip!
    storage.db_lock.acquire(owner_run_id="gha-1002", trading_date="2026-11-20")
    res_dup = sched.run_scheduled_session(trading_date="2026-11-20", market_data_provider=lambda _d: snap_d1)
    assert res_dup["cycle_result"]["execution_result"]["idempotent_skip"] is True
    storage.db_lock.release(owner_run_id="gha-1002")

    # Day 2 (2026-11-23): fresh reconnected adapter & engine executes Day 1 pending orders at Day 2 open BEFORE Day 2 signals!
    storage2 = PersistentStorageAdapter(db_path=db_file, database_url="")
    pull2 = storage2.pull_durable_ledger(owner_run_id="gha-1003")
    assert pull2["sqlite_integrity_ok"] is True
    storage2.db_lock.acquire(owner_run_id="gha-1003", trading_date="2026-11-23")
    orch2 = PaperTradingOrchestrator(db_path=db_file, reports_dir=rep_dir, prospective_start_date="2026-11-19")
    sched2 = PaperTradingScheduler(orchestrator=orch2, state_dir=sched_dir)

    snap_d2 = _build_offline_eod_snapshot("2026-11-23", px_shift=4.0)
    res_d2 = sched2.run_scheduled_session(trading_date="2026-11-23", market_data_provider=lambda _d: snap_d2)
    assert res_d2["scheduler_status"] == "COMPLETED"
    orders_d2 = res_d2["cycle_result"]["execution_result"]["orders"]
    assert any(o["instrument_class"] == "DELIVERY" and o["side"] == "BUY" and o["execution_status"] == "FILLED" for o in orders_d2)
    assert any(o["symbol"] == "NIFTYBEES" and o["side"] == "BUY" and o["execution_status"] == "FILLED" for o in orders_d2)
    assert res_d2["cycle_result"]["reconciliation"]["passed"] is True
    assert res_d2["cycle_result"]["reconciliation"]["reconciliation_diff_inr"] <= RECONCILIATION_TOLERANCE_INR
    storage2.db_lock.release(owner_run_id="gha-1003")
    storage2.push_durable_ledger(owner_run_id="gha-1003")

    full = orch2.engine.get_full_ledger_snapshot()
    assert full["prospective_validation"]["completed_unseen_sessions"] == 2
    assert full["audit_chain"]["valid"] is True


def test_ga_3_4_5_6_7_8_crash_lock_missing_data_missed_session_and_recovery(tmp_path: Path) -> None:
    db_file = tmp_path / "crash_test.db"
    rep_dir = tmp_path / "rep"
    sched_dir = tmp_path / "sched"

    orch = PaperTradingOrchestrator(db_path=db_file, reports_dir=rep_dir, prospective_start_date="2026-11-19")
    sched = PaperTradingScheduler(orchestrator=orch, state_dir=sched_dir)
    lock = DatabaseExecutionLock(db_path=db_file, lease_ttl_seconds=600)

    # 7. Missing market data -> WAITING_FOR_DATA (zero fabricated fills)
    r_wait = sched.run_scheduled_session(trading_date="2026-11-20", market_data_provider=lambda _d: None)
    assert r_wait["scheduler_status"] == "WAITING_FOR_DATA"
    assert orch.engine.get_full_ledger_snapshot()["prospective_validation"]["completed_unseen_sessions"] == 0

    # 6. Missed trading session -> explicit MISSED checkpoint, zero fabricated fills
    r_miss = sched.run_scheduled_session(
        trading_date="2026-11-20", market_data_provider=lambda _d: None, mark_missed_if_unavailable=True
    )
    assert r_miss["missed_recorded"] is True
    assert orch.engine.get_full_ledger_snapshot()["prospective_validation"]["completed_unseen_sessions"] == 0

    # 3 & 5. Active lock conflict + simulated runner termination (stale expired crash lock + partial checkpoint)
    lock.acquire(owner_run_id="gha-crash-1", trading_date="2026-11-23")
    with pytest.raises(DatabaseLockError, match="ACTIVE_EXECUTION_LOCK_CONFLICT"):
        lock.acquire(owner_run_id="gha-overlap-2", trading_date="2026-11-23")

    # Expire lease without releasing -> simulates sudden GitHub runner kill mid-execution!
    past_iso = (datetime.now(timezone.utc) - timedelta(minutes=45)).isoformat()
    conn = sqlite3.connect(str(db_file))
    conn.execute("UPDATE distributed_execution_locks SET expires_at_utc = ? WHERE owner_run_id = 'gha-crash-1'", (past_iso,))
    conn.commit()
    conn.close()

    with pytest.raises(DatabaseLockError, match="STALE_CRASH_LOCK_DETECTED"):
        lock.acquire(owner_run_id="gha-retry-3", trading_date="2026-11-23")

    # Simulate partial checkpoint left behind on runner termination
    with pytest.raises(RuntimeError, match="SIMULATED_DB_WRITE_INTERRUPTION"):
        orch.engine.process_trading_session(
            trading_date="2026-11-23",
            market_snapshot=_build_offline_eod_snapshot("2026-11-23"),
            simulate_db_failure_after_checkpoint=True,
        )
    # Next attempt detects incomplete checkpoint and activates SAFE_MODE preserving last valid INR 100,000 state!
    r_safe = orch.engine.process_trading_session(
        trading_date="2026-11-23",
        market_snapshot=_build_offline_eod_snapshot("2026-11-23"),
    )
    assert r_safe["status"] == "SAFE_MODE_INCOMPLETE_CHECKPOINT"
    assert orch.engine.get_engine_state()["engine_state"] == "SAFE_MODE"
    assert orch.engine.reconcile_portfolio()["nav_from_assets_inr"] == 100_000.0

    # 8. Validated recovery clears stale lock & incomplete checkpoint and resumes cleanly
    lock.force_clear_stale_lock_for_recovery(operator_note="test_recovery")
    conn = sqlite3.connect(str(db_file))
    conn.execute("UPDATE session_checkpoints SET phase = 'FAILED' WHERE trading_date = '2026-11-23'")
    conn.commit()
    conn.close()
    rec = orch.engine.recover_from_safe_mode(operator_note="test_recovery")
    assert rec["recovered"] is True
    lock.acquire(owner_run_id="gha-clean-4", trading_date="2026-11-23")
    r_ok = sched.run_scheduled_session(
        trading_date="2026-11-23",
        market_data_provider=lambda d: _build_offline_eod_snapshot(d),
    )
    lock.release(owner_run_id="gha-clean-4")
    assert r_ok["scheduler_status"] == "COMPLETED"
    assert orch.engine.reconcile_portfolio()["passed"] is True


def test_ga_security_audit_workflow_yaml_and_credential_redaction() -> None:
    # 1. Credential redaction check
    raw = "Connection failed to postgresql://user_admin:SuperSecretPass123!@db.example.com:5432/m31"
    redacted = redact_sensitive_text(raw)
    assert "SuperSecretPass123!" not in redacted
    assert "user_admin" not in redacted
    assert "***:***@db.example.com:5432/m31" in redacted

    # 2. Persistent storage config audit
    assert DEFAULT_STORAGE_CONFIG_PATH.exists()
    scfg = json.loads(DEFAULT_STORAGE_CONFIG_PATH.read_text(encoding="utf-8"))
    pol = scfg["authoritative_storage_policy"]
    assert pol["ephemeral_runner_filesystem_permitted"] is False
    assert pol["github_actions_cache_permitted"] is False
    assert pol["github_artifacts_as_authoritative_ledger_permitted"] is False
    assert scfg["safety_invariants"]["live_trading_enabled"] is False

    # 3. GitHub Actions workflow YAML static security & compliance audit
    assert WORKFLOW_PATH.exists()
    yml = WORKFLOW_PATH.read_text(encoding="utf-8")

    # Must have scheduled post-close IST crons, workflow_dispatch, single-flight cancel-in-progress: false
    assert "schedule:" in yml
    assert "workflow_dispatch:" in yml
    assert "TZ: Asia/Kolkata" in yml
    assert "cancel-in-progress: false" in yml
    assert "permissions: {}" in yml

    # Hard safety flags strictly false & ZERO Zerodha/Kite/broker secrets or SDKs!
    assert 'LIVE_TRADING_ENABLED: "false"' in yml
    assert 'BROKER_ORDER_PLACEMENT_ENABLED: "false"' in yml
    assert 'AUTOMATIC_CAPITAL_DEPLOYMENT_ENABLED: "false"' in yml
    for forbidden in ("KITE_API_KEY", "KITE_API_SECRET", "KITE_ACCESS_TOKEN", "kiteconnect", "actions/cache"):
        assert forbidden not in yml, f"Forbidden token found in paper-trading workflow: {forbidden}"

    # EVERY third-party action `uses:` MUST be pinned to a 40-char immutable commit SHA!
    uses_lines = re.findall(r"uses:\s*([^\s#]+)", yml)
    assert len(uses_lines) >= 3
    sha_re = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+@[0-9a-f]{40}$")
    for ref in uses_lines:
        assert sha_re.match(ref), f"GitHub Action not pinned to 40-char immutable SHA: {ref}"

    # Failure notification present
    assert "if: failure()" in yml
    assert "gh issue" in yml
    assert "m3_1_notify.py" in yml


def test_ga_smart_ntfy_and_email_notifier_logic() -> None:
    import importlib.util

    spec = importlib.util.spec_from_file_location("m3_1_notify", REPO_ROOT / "scripts" / "m3_1_notify.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    assert mod.resolve_ntfy_url("", "my-secret-topic") == "https://ntfy.sh/my-secret-topic"
    assert mod.resolve_ntfy_url("https://ntfy.example.com/custom", "ignored") == "https://ntfy.example.com/custom"

    def _ctx(dt: str, completed: int, state: str = "RUNNING") -> dict:
        return {
            "daily": {
                "trading_date": dt,
                "engine_status": state,
                "portfolio_nav": {"current_nav_inr": 101_500.0, "cumulative_return_pct": 1.5, "benchmark_niftybees_nav": 100_800.0, "initial_capital_inr": 100_000.0},
                "reconciliation_result": {"passed": True, "reconciliation_diff_inr": 0.0},
                "regime_determination": {"confirmed_regime": "BULL_TREND"},
            },
            "progress": {"completed_unseen_sessions": completed, "minimum_required_sessions": 60, "target_sessions": 120},
            "alerts": {"overall_health": "HEALTHY", "alert_counts": {"CRITICAL": 0}},
        }

    # 1. Duplicate/idempotent retry (newly_completed=False) -> NO spam!
    d_dup = mod.decide_and_format_notification(_ctx("2026-11-23", 2), session_newly_completed=False)
    assert d_dup["should_send"] is False

    # 2. Normal mid-week non-milestone session (Mon, day 3) -> NO spam!
    d_mid = mod.decide_and_format_notification(_ctx("2026-11-25", 3), session_newly_completed=True)
    assert d_mid["should_send"] is False

    # 3. Friday weekly digest (Fri, day 4) -> sends weekly check-in!
    d_fri = mod.decide_and_format_notification(_ctx("2026-11-27", 4), session_newly_completed=True)
    assert d_fri["should_send"] is True and d_fri["reason"] == "WEEKLY_FRIDAY_DIGEST"

    # 4. Day 60 interim review gate -> high-priority milestone alert!
    d_60 = mod.decide_and_format_notification(_ctx("2027-02-18", 60), session_newly_completed=True)
    assert d_60["should_send"] is True and d_60["priority"] == "high" and "60-SESSION" in d_60["title"]

    # 5. SAFE_MODE or failure -> ALWAYS high-priority alert!
    d_fail = mod.decide_and_format_notification(_ctx("2026-11-25", 3, state="SAFE_MODE"), session_newly_completed=False)
    assert d_fail["should_send"] is True and d_fail["priority"] == "high"


def test_full_pit_universe_no_subset_and_real_data_ui_guards():
    """Guard against silent universe subset truncation and verify real-data UI parity."""
    uni_path = REPO_ROOT / "data" / "m3_1_paper_trading" / "nse_prospective_universe.json"
    uni = json.loads(uni_path.read_text(encoding="utf-8"))
    equities = uni.get("equities", [])
    assert len(equities) >= 1000, f"Universe must include full PIT-eligible NSE equities, got only {len(equities)}"
    assert uni.get("count") == len(equities)
    tiers = {e.get("pit_tier") for e in equities}
    assert "TIER_A_LARGE_CORE" in tiers and "TIER_B_BROAD_LIQUID" in tiers

    # UI mirrors must be byte-for-byte identical and load real snapshot + CSV equity curve
    root_js = (REPO_ROOT / "app.js").read_text(encoding="utf-8")
    assert root_js == (REPO_ROOT / "web" / "app.js").read_text(encoding="utf-8")
    assert root_js == (REPO_ROOT / "mobile-app" / "app" / "src" / "main" / "assets" / "app.js").read_text(encoding="utf-8")
    assert "portfolio_snapshot.json" in root_js
    assert "paper_trading_daily_report.csv" in root_js

