#!/usr/bin/env python3
"""Moreshwar 3.2 — orchestrator-driven prospective paper trading CLI (paper-only).

  python3 scripts/moreshwar_3_2_paper_ops.py run-session --date YYYY-MM-DD|auto [--mark-missed]
  python3 scripts/moreshwar_3_2_paper_ops.py reconcile
  python3 scripts/moreshwar_3_2_paper_ops.py status

The multi-agent orchestrator scans the FULL PIT-eligible NSE universe; its approved picks are the
paper trades; matured real outcomes feed back into the agents. ZERO broker credentials, ZERO live orders.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import uuid

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from finsistant_v2.m3_1.live_nse_eod_feed import current_ist_date_str
from finsistant_v2.m3_1.paper_trading_scheduler import (
    DEFAULT_NSE_HOLIDAYS,
    fetch_official_nse_holidays,
    is_eligible_nse_trading_day,
    resolve_pending_auto_sessions,
)
from finsistant_v2.m3_1.persistent_storage_adapter import (
    PersistentStorageAdapter,
    redact_sensitive_text,
)
from finsistant_v2.m3_1.prospective_paper_trading_engine import HISTORICAL_DB_END_DATE
from finsistant_v2.m3_2.orchestrated_session import (
    M32_DB_PATH,
    M32_LEARNING_DB_PATH,
    OrchestratedSessionRunner,
)

M32_LOCK_NAME = "m3_2_daily_paper_session"
M32_POSTGRES_ADVISORY_LOCK_KEY = 32032032
M32_LEARNING_LOCK_NAME = "m3_2_learning_ledger"
M32_LEARNING_ADVISORY_LOCK_KEY = 32032033


def _build_m32_adapters() -> tuple[PersistentStorageAdapter, PersistentStorageAdapter]:
    db_url = os.getenv("M32_DATABASE_URL", "").strip() or os.getenv("M31_DATABASE_URL", "").strip()
    paper_adapter = PersistentStorageAdapter(
        db_path=M32_DB_PATH,
        database_url=db_url,
        lock_name=M32_LOCK_NAME,
        advisory_lock_key=M32_POSTGRES_ADVISORY_LOCK_KEY,
        ledger_id="m3_2_primary",
    )
    learn_adapter = PersistentStorageAdapter(
        db_path=M32_LEARNING_DB_PATH,
        database_url=db_url,
        lock_name=M32_LEARNING_LOCK_NAME,
        advisory_lock_key=M32_LEARNING_ADVISORY_LOCK_KEY,
        ledger_id="m3_2_learning",
    )
    return paper_adapter, learn_adapter


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="command", required=True)
    rs = sub.add_parser("run-session")
    rs.add_argument("--date", default="auto")
    rs.add_argument("--mark-missed", action="store_true")
    rs.add_argument("--offline-holidays", action="store_true", help="Only for offline deterministic test runs.")
    sub.add_parser("reconcile")
    sub.add_parser("status")
    args = ap.parse_args()

    owner_run_id = f"m3_2_{os.getenv('GITHUB_RUN_ID', 'local')}_{uuid.uuid4().hex[:8]}"
    paper_adapter, learn_adapter = _build_m32_adapters()

    if args.command == "run-session":
        raw_date = str(args.date).strip()
        is_auto = raw_date.lower() == "auto"
        target_dt = current_ist_date_str() if is_auto else raw_date
        if target_dt <= HISTORICAL_DB_END_DATE:
            print(json.dumps({"trading_date": target_dt, "skipped": True, "reason": "HISTORICAL_CUTOFF_GUARD"}, indent=2))
            return 0

        nse_holidays = DEFAULT_NSE_HOLIDAYS if args.offline_holidays else fetch_official_nse_holidays()

        lock_held = False
        try:
            paper_adapter.pull_durable_ledger(owner_run_id=owner_run_id)
            learn_adapter.pull_durable_ledger(owner_run_id=owner_run_id)
            paper_adapter.db_lock.acquire(owner_run_id=owner_run_id, trading_date=target_dt)
            lock_held = True

            runner = OrchestratedSessionRunner()
            if is_auto:
                dates_to_run = resolve_pending_auto_sessions(runner.orch, target_dt, holidays=nse_holidays)
                if not dates_to_run:
                    _, why = is_eligible_nse_trading_day(target_dt, holidays=nse_holidays)
                    print(json.dumps({"trading_date": target_dt, "skipped": True, "reason": why}, indent=2))
                    return 0
            else:
                ok, why = is_eligible_nse_trading_day(target_dt, holidays=nse_holidays)
                if not ok:
                    print(json.dumps({"trading_date": target_dt, "skipped": True, "reason": why}, indent=2))
                    return 0
                dates_to_run = [target_dt]

            session_outputs: list[dict] = []
            overall_ok = True
            for dt in dates_to_run:
                if dt <= HISTORICAL_DB_END_DATE:
                    continue
                res = runner.run_session(dt, mark_missed=bool(args.mark_missed))
                recon = runner.orch.engine.reconcile_portfolio()
                chain = runner.orch.engine.verify_audit_chain()
                scan = runner.last_scan or {}
                status = res.get("scheduler_status") or res.get("status")
                session_outputs.append({
                    "trading_date": dt,
                    "scheduler_status": status,
                    "orchestrator_stages": scan.get("stages"),
                    "weights": scan.get("weights"),
                    "paper_candidates": [c["symbol"] for c in scan.get("candidates", [])],
                    "exit_signals": [e["symbol"] for e in (runner.last_exit_signals or [])],
                    "reconciliation_passed": recon["passed"],
                    "reconciliation_diff_inr": recon["reconciliation_diff_inr"],
                    "audit_chain_valid": chain["valid"],
                })
                if not (recon["passed"] and chain["valid"]):
                    overall_ok = False
                    break
                if res.get("waiting") or status in ("WAITING_FOR_DATA", "SAFE_MODE", "FAILED"):
                    if status == "FAILED":
                        overall_ok = False
                    break

            if overall_ok:
                paper_adapter.push_durable_ledger(owner_run_id=owner_run_id)
                learn_adapter.push_durable_ledger(owner_run_id=owner_run_id)
            else:
                paper_adapter.close_without_push()
                learn_adapter.close_without_push()

            out = session_outputs[-1] if len(session_outputs) == 1 else {"sessions": session_outputs}
            print(json.dumps(out, indent=2, default=str))
            return 0 if overall_ok else 1
        except Exception as exc:
            paper_adapter.close_without_push()
            learn_adapter.close_without_push()
            print(json.dumps({"error": redact_sensitive_text(str(exc)), "status": "FAILED"}, indent=2), file=sys.stderr)
            raise
        finally:
            if lock_held:
                try:
                    paper_adapter.db_lock.release(owner_run_id=owner_run_id)
                except Exception:
                    pass

    if args.command == "reconcile":
        lock_held = False
        try:
            paper_adapter.pull_durable_ledger(owner_run_id=owner_run_id)
            paper_adapter.db_lock.acquire(owner_run_id=owner_run_id, trading_date="reconcile")
            lock_held = True
            runner = OrchestratedSessionRunner()
            recon = runner.orch.execute_paper_control("run_reconciliation")
            chain = runner.orch.engine.verify_audit_chain()
            ok = recon["status"] == "PASS" and chain["valid"]
            if ok:
                paper_adapter.push_durable_ledger(owner_run_id=owner_run_id)
            else:
                paper_adapter.close_without_push()
            print(json.dumps({"reconciliation": recon, "audit_chain": chain}, indent=2, default=str))
            return 0 if ok else 1
        except Exception:
            paper_adapter.close_without_push()
            raise
        finally:
            if lock_held:
                try:
                    paper_adapter.db_lock.release(owner_run_id=owner_run_id)
                except Exception:
                    pass

    runner = OrchestratedSessionRunner()
    snap = runner.orch.engine.get_full_ledger_snapshot()
    print(json.dumps({
        "engine_state": snap["engine_state"],
        "nav": snap["portfolio_state"].get("last_nav_inr"),
        "pending": snap["pending_next_session_orders"],
        "weights": runner.ledger.current_weights(),
        "storage": paper_adapter.describe_storage(),
    }, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
