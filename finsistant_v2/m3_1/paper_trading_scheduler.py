"""Moreshwar 3.1 (Prompt 5/5) — Daily NSE Session Paper-Trading Scheduler & State Machine.

Coordinates the deterministic daily workflow (Section 7) without uncontrolled background threads
or overlapping worker invocations:
1. Identify eligible NSE trading sessions (excludes weekends & NSE holidays).
2. Check/wait for completed daily market data (transitions to `WAITING_FOR_DATA` if delayed).
3. Validate data freshness & completeness (never fabricates missing sessions).
4. Execute `PaperTradingOrchestrator.run_daily_cycle(...)`.
5. Persist outputs, run four-ledger reconciliation, update risk metrics & alerts, export daily reports.
6. Record missed NSE sessions explicitly as `MISSED`.
7. Enforce single-worker file lock + idempotency so duplicate invocations are harmless no-ops.

Explicit Scheduler Status FSM:
    IDLE | WAITING_FOR_DATA | PROCESSING | COMPLETED | FAILED | SAFE_MODE
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
import fcntl
import json
from pathlib import Path
from typing import Any, Callable

from finsistant_v2.m3_1.paper_trading_orchestrator import PaperTradingOrchestrator
from finsistant_v2.m3_1.prospective_paper_trading_engine import (
    DEFAULT_PAPER_DATA_DIR,
    PaperOnlyExecutionGuard,
)

SCHEDULER_STATUSES = frozenset({
    "IDLE",
    "WAITING_FOR_DATA",
    "PROCESSING",
    "COMPLETED",
    "FAILED",
    "SAFE_MODE",
})

# Standard NSE Equity Market Holidays (extensible via constructor for 2026–2027 prospective sessions)
DEFAULT_NSE_HOLIDAYS: frozenset[str] = frozenset({
    "2026-01-26",  # Republic Day
    "2026-03-03",  # Holi
    "2026-03-26",  # Shri Ram Navami
    "2026-03-31",  # Mahavir Jayanti
    "2026-04-03",  # Good Friday
    "2026-04-14",  # Dr. Baba Saheb Ambedkar Jayanti
    "2026-05-01",  # Maharashtra Day
    "2026-05-28",  # Bakri Id
    "2026-06-26",  # Muharram
    "2026-09-14",  # Ganesh Chaturthi
    "2026-10-02",  # Mahatma Gandhi Jayanti
    "2026-10-20",  # Dussehra
    "2026-11-10",  # Diwali Balipratipada
    "2026-11-24",  # Gurunanak Jayanti
    "2026-12-25",  # Christmas
    "2027-01-26",  # Republic Day
    "2027-03-22",  # Holi
    "2027-03-26",  # Good Friday
    "2027-04-14",  # Ambedkar Jayanti
    "2027-05-01",  # Maharashtra Day
    "2027-08-15",  # Independence Day
    "2027-10-02",  # Gandhi Jayanti
    "2027-12-25",  # Christmas
})


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def is_eligible_nse_trading_day(
    trading_date: str | date,
    *,
    holidays: frozenset[str] = DEFAULT_NSE_HOLIDAYS,
) -> tuple[bool, str]:
    """Return `(True, 'ELIGIBLE_NSE_SESSION')` if Mon–Fri and not an NSE holiday."""
    dt_obj = date.fromisoformat(str(trading_date)) if isinstance(trading_date, str) else trading_date
    dt_str = dt_obj.isoformat()
    if dt_obj.weekday() >= 5:
        return False, "WEEKEND_NON_TRADING_DAY"
    if dt_str in holidays:
        return False, "NSE_MARKET_HOLIDAY"
    return True, "ELIGIBLE_NSE_SESSION"


def list_eligible_nse_sessions_between(
    start_date_exclusive: str,
    end_date_inclusive: str,
    *,
    holidays: frozenset[str] = DEFAULT_NSE_HOLIDAYS,
) -> list[str]:
    """Enumerate all eligible NSE sessions strictly after `start_date_exclusive` through `end_date_inclusive`."""
    cur = date.fromisoformat(start_date_exclusive) + timedelta(days=1)
    end = date.fromisoformat(end_date_inclusive)
    out: list[str] = []
    while cur <= end:
        ok, _ = is_eligible_nse_trading_day(cur, holidays=holidays)
        if ok:
            out.append(cur.isoformat())
        cur += timedelta(days=1)
    return out


class PaperTradingScheduler:
    """Single-worker, lock-protected, idempotent daily NSE paper-trading scheduler."""

    def __init__(
        self,
        *,
        orchestrator: PaperTradingOrchestrator,
        state_dir: Path | None = None,
        holidays: frozenset[str] = DEFAULT_NSE_HOLIDAYS,
    ) -> None:
        self.safety_guard = PaperOnlyExecutionGuard()
        self.safety_guard.assert_paper_only()
        self.orchestrator = orchestrator
        self.state_dir = Path(state_dir) if state_dir is not None else (orchestrator.engine.db_path.parent)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.lock_path = self.state_dir / "scheduler.lock"
        self.status_path = self.state_dir / "scheduler_status.json"
        self.holidays = holidays
        self._status: dict[str, Any] = self._load_or_init_status()

    def _load_or_init_status(self) -> dict[str, Any]:
        if self.status_path.exists():
            try:
                data = json.loads(self.status_path.read_text(encoding="utf-8"))
                if data.get("status") in SCHEDULER_STATUSES:
                    return data
            except Exception:
                pass
        eng = self.orchestrator.engine.get_engine_state()
        init_status = "SAFE_MODE" if eng["engine_state"] == "SAFE_MODE" else "IDLE"
        st = {
            "status": init_status,
            "last_checked_date": eng["prospective_start_date"],
            "last_completed_date": "",
            "last_reason": "Scheduler initialized cleanly (paper-only).",
            "missed_sessions": [],
            "updated_at_utc": _utc_now_iso(),
            "live_trading_enabled": False,
        }
        self._persist_status(st)
        return st

    def _persist_status(self, st: dict[str, Any]) -> None:
        st["updated_at_utc"] = _utc_now_iso()
        st["live_trading_enabled"] = False
        self.status_path.write_text(json.dumps(st, indent=2) + "\n", encoding="utf-8")
        self._status = st

    def get_status(self) -> dict[str, Any]:
        # Reflect engine SAFE_MODE immediately even if triggered externally
        eng = self.orchestrator.engine.get_engine_state()
        if eng["engine_state"] == "SAFE_MODE" and self._status.get("status") != "SAFE_MODE":
            self._status["status"] = "SAFE_MODE"
            self._status["last_reason"] = eng.get("safe_mode_reason") or "ENGINE_IN_SAFE_MODE"
            self._persist_status(self._status)
        return dict(self._status)

    def run_scheduled_session(
        self,
        *,
        trading_date: str,
        market_data_provider: Callable[[str], dict[str, Any] | None],
        observation_mode: str | None = None,
        mark_missed_if_unavailable: bool = False,
        export_reports: bool = True,
    ) -> dict[str, Any]:
        """Run one locked, idempotent scheduled session check & execution cycle."""
        dt = str(trading_date)

        # Acquire non-blocking single-worker lock so overlapping workers NEVER run concurrently
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_fh = self.lock_path.open("w", encoding="utf-8")
        try:
            try:
                fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return {
                    "trading_date": dt,
                    "scheduler_status": self.get_status()["status"],
                    "skipped": True,
                    "reason": "CONCURRENT_WORKER_LOCK_HELD",
                }

            # Step 1: Check NSE trading calendar
            eligible, cal_reason = is_eligible_nse_trading_day(dt, holidays=self.holidays)
            if not eligible:
                st = self.get_status()
                st["last_checked_date"] = dt
                st["last_reason"] = f"Skipped {dt}: {cal_reason}"
                if st["status"] not in ("SAFE_MODE",):
                    st["status"] = "IDLE"
                self._persist_status(st)
                return {
                    "trading_date": dt,
                    "scheduler_status": st["status"],
                    "skipped": True,
                    "reason": cal_reason,
                }

            # Check engine state before processing
            eng = self.orchestrator.engine.get_engine_state()
            if eng["engine_state"] == "SAFE_MODE":
                st = self.get_status()
                st["status"] = "SAFE_MODE"
                st["last_checked_date"] = dt
                st["last_reason"] = eng.get("safe_mode_reason") or "SAFE_MODE_ACTIVE"
                self._persist_status(st)
                return {
                    "trading_date": dt,
                    "scheduler_status": "SAFE_MODE",
                    "skipped": True,
                    "reason": st["last_reason"],
                }

            # Step 2: Check for completed daily market data
            st = self.get_status()
            st["status"] = "WAITING_FOR_DATA"
            st["last_checked_date"] = dt
            st["last_reason"] = f"Querying daily market data for {dt}"
            self._persist_status(st)

            snapshot = market_data_provider(dt)
            if snapshot is None:
                if mark_missed_if_unavailable:
                    self.orchestrator.engine.record_missed_session(
                        dt, reason="MARKET_DATA_UNAVAILABLE_AT_CUTOFF", observation_mode=observation_mode
                    )
                    missed = list(st.get("missed_sessions", []))
                    if dt not in missed:
                        missed.append(dt)
                    st["missed_sessions"] = missed
                    st["status"] = "FAILED"
                    st["last_reason"] = f"Session {dt} recorded as MISSED (market data unavailable; NOT fabricated)."
                    self._persist_status(st)
                    return {
                        "trading_date": dt,
                        "scheduler_status": "FAILED",
                        "missed_recorded": True,
                        "reason": "MARKET_DATA_UNAVAILABLE_AT_CUTOFF",
                    }
                st["status"] = "WAITING_FOR_DATA"
                st["last_reason"] = f"Daily market data for {dt} not yet available; waiting."
                self._persist_status(st)
                return {
                    "trading_date": dt,
                    "scheduler_status": "WAITING_FOR_DATA",
                    "waiting": True,
                    "reason": "MARKET_DATA_PENDING",
                }

            # Step 3–10: Run orchestrator cycle (validate -> execute -> reconcile -> monitor -> reports)
            st["status"] = "PROCESSING"
            st["last_reason"] = f"Processing paper-trading session {dt}"
            self._persist_status(st)

            try:
                res = self.orchestrator.run_daily_cycle(
                    trading_date=dt,
                    market_snapshot=snapshot,
                    observation_mode=observation_mode,
                    export_reports=export_reports,
                )
            except Exception as exc:
                st["status"] = "SAFE_MODE" if self.orchestrator.engine.get_engine_state()["engine_state"] == "SAFE_MODE" else "FAILED"
                st["last_reason"] = f"Cycle error on {dt}: {exc}"
                self._persist_status(st)
                raise

            eng_after = res["engine_state"]["engine_state"]
            if eng_after == "SAFE_MODE":
                st["status"] = "SAFE_MODE"
                st["last_reason"] = res["engine_state"].get("safe_mode_reason") or f"Entered SAFE_MODE on {dt}"
            else:
                st["status"] = "COMPLETED"
                st["last_completed_date"] = dt
                st["last_reason"] = (
                    f"Idempotent skip on {dt} (already completed)"
                    if res["execution_result"].get("idempotent_skip")
                    else f"Completed session {dt} cleanly (recon_diff={res['reconciliation']['reconciliation_diff_inr']:.4f} INR)"
                )
            self._persist_status(st)
            return {
                "trading_date": dt,
                "scheduler_status": st["status"],
                "cycle_result": res,
            }
        finally:
            try:
                fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)
            except Exception:
                pass
            lock_fh.close()
