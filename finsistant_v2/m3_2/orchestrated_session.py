"""Moreshwar 3.2 — orchestrator-driven daily paper-trading session runner (paper-only).

Wires the multi-agent orchestrator INTO the 80-session prospective paper-trading loop:

    real EOD universe (~1,182 NSE) -> orchestrator scan (Tech -> shortlist -> Fund+Exec) -> paper engine
            ^                                                                                  |
            +--- every passed/failed closed trade + matured 5D outcomes teach the agents <-----+
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from finsistant_v2.m3_1.live_nse_eod_feed import fetch_completed_nse_eod_snapshot
from finsistant_v2.m3_1.paper_trading_orchestrator import PaperTradingOrchestrator
from finsistant_v2.m3_1.paper_trading_scheduler import PaperTradingScheduler
from finsistant_v2.m3_1.prospective_paper_trading_engine import REPO_ROOT

from .learning_ledger import OrchestratorLearningLedger
from .orchestrator_scanner import (
    MISSION_TARGET_SESSIONS,
    ORCH_STRATEGY_ID,
    SCANNER_VERSION,
    OrchestratorScanConfig,
    OrchestratorUniverseScanner,
)

M32_DATA_DIR = REPO_ROOT / "data" / "m3_2_paper_trading"
M32_DB_PATH = M32_DATA_DIR / "prospective_paper_ledger.db"
M32_LEARNING_DB_PATH = M32_DATA_DIR / "orchestrator_learning.db"
M32_REPORTS_DIR = REPO_ROOT / "reports" / "moreshwar_3_2"
M32_PAPER_REPORTS_DIR = M32_REPORTS_DIR / "prospective_paper_trading"
M32_SCAN_DIR = M32_REPORTS_DIR / "orchestrator_scan"
RELEASE_VERSION_M3_2 = "3.2.0"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def pairwise_corr_for_candidates(
    candidates: list[dict[str, Any]], equity_histories: dict[str, dict[str, Any]]
) -> dict[tuple[str, str], float]:
    """Causal 60D return correlations among the orchestrator's candidates (real histories)."""
    rets: dict[str, np.ndarray] = {}
    for c in candidates:
        h = equity_histories.get(str(c["sec_id"]))
        if not h or len(h["close"]) < 61:
            continue
        arr = np.asarray(h["close"][-61:], dtype=np.float64)
        r = np.diff(arr) / np.maximum(1e-6, arr[:-1])
        if float(np.std(r)) > 1e-8:
            rets[str(c["sec_id"])] = r
    ids = list(rets)
    out: dict[tuple[str, str], float] = {}
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            v = float(np.corrcoef(rets[ids[i]], rets[ids[j]])[0, 1])
            if np.isfinite(v):
                out[(ids[i], ids[j])] = round(v, 4)
    return out


class OrchestratedSessionRunner:
    def __init__(
        self,
        *,
        db_path: Path = M32_DB_PATH,
        learning_db_path: Path = M32_LEARNING_DB_PATH,
        reports_dir: Path = M32_PAPER_REPORTS_DIR,
        scan_dir: Path = M32_SCAN_DIR,
        scanner: OrchestratorUniverseScanner | None = None,
    ) -> None:
        self.orch = PaperTradingOrchestrator(
            db_path=db_path,
            reports_dir=reports_dir,
            extra_admitted_strategy_ids=(ORCH_STRATEGY_ID,),
        )
        self.ledger = OrchestratorLearningLedger(learning_db_path)
        self.scanner = scanner or OrchestratorUniverseScanner(OrchestratorScanConfig())
        self.scan_dir = Path(scan_dir)
        self.reports_dir = Path(reports_dir)
        self.last_scan: dict[str, Any] | None = None

    # -- provider handed to PaperTradingScheduler -------------------------------------------------
    def market_data_provider(self, trading_date: str) -> dict[str, Any] | None:
        self.last_scan = None
        ledger_snap = self.orch.engine.get_full_ledger_snapshot()
        snap_open = ledger_snap.get("open_tactical_positions", [])
        extra_open = [
            {
                "sec_id": str(p["sec_id"]),
                "symbol": str(p["symbol"]),
                "sector": str(p["sector"]),
                "industry": str(p["industry"]),
            }
            for p in snap_open
        ]
        snap = fetch_completed_nse_eod_snapshot(trading_date, extra_open_symbols=extra_open, include_histories=True)
        if snap is None:
            return None
        histories = snap.pop("equity_histories")
        nb_dates = snap.pop("niftybees_dates")
        nb_closes = snap.pop("niftybees_closes_full")

        # Learn from prior closed trades + newly matured 5D/10D/20D real outcomes BEFORE scanning today
        new_lessons = self.ledger.sync_closed_trades(self.orch.engine.db_path)
        matured = self.ledger.mature_outcomes(
            today=trading_date, equity_histories=histories, nb_dates=nb_dates, nb_closes=nb_closes
        )
        upd = self.ledger.update_weights(trading_date)

        nav = float(ledger_snap.get("portfolio_state", {}).get("last_nav_inr", 100_000.0) or 100_000.0)
        scan = self.scanner.scan(
            trading_date=trading_date,
            equity_histories=histories,
            eq_bars=snap["equity_bars"],
            learned_params=upd,
            capital_inr=nav,
        )
        scan["outcomes_matured_this_run"] = matured
        scan["trade_lessons_synced_pre_scan"] = new_lessons
        self.last_scan = scan

        snap["family_e_candidates"] = scan["candidates"]  # engine key; candidates are orchestrator picks
        snap["candidate_source"] = "ORCHESTRATOR_AGENT_CONSENSUS"
        snap["pairwise_corr_map"] = pairwise_corr_for_candidates(scan["candidates"], histories)
        return snap

    # -- full lifecycle -----------------------------------------------------------------------------
    def run_session(self, trading_date: str, *, mark_missed: bool = False) -> dict[str, Any]:
        sched = PaperTradingScheduler(orchestrator=self.orch)
        res = sched.run_scheduled_session(
            trading_date=trading_date,
            market_data_provider=self.market_data_provider,
            observation_mode="PROSPECTIVE_PAPER",
            mark_missed_if_unavailable=mark_missed,
        )
        self.finalize_reports(trading_date)
        return res

    def finalize_reports(self, trading_date: str) -> None:
        paper_snap = self.orch.engine.get_full_ledger_snapshot()
        # Sync any trades closed during today's session immediately so post-mortems & weights reflect today
        post_lessons = self.ledger.sync_closed_trades(self.orch.engine.db_path)
        if post_lessons > 0:
            self.ledger.update_weights(trading_date)

        scan = self.last_scan
        if scan is not None:
            pend = paper_snap.get("pending_next_session_orders", []) or []
            queued = {str(p.get("symbol")) for p in pend if p.get("kind") == "TACTICAL_BUY"}
            self.ledger.record_scan(scan, queued)

            for r in scan["shortlist"]:
                r["queued_for_next_open"] = r["symbol"] in queued
            public = {k: v for k, v in scan.items() if k != "universe_tech"}
            public["universe_tech_top100"] = scan["universe_tech"][:100]
            public["config_sha256"] = hashlib.sha256(json.dumps(scan["config"], sort_keys=True).encode()).hexdigest()
            public["queued_for_next_open"] = sorted(queued)
            public["trade_lessons_synced_post_session"] = post_lessons
            self.scan_dir.mkdir(parents=True, exist_ok=True)
            text = json.dumps(public, indent=2, default=str)
            (self.scan_dir / "latest_scan.json").write_text(text, encoding="utf-8")
            (self.scan_dir / f"scan_{trading_date}.json").write_text(text, encoding="utf-8")

            daily = self.reports_dir / "paper_trading_daily_report.json"
            if daily.exists():
                try:
                    d = json.loads(daily.read_text(encoding="utf-8"))
                    d["release_version"] = RELEASE_VERSION_M3_2
                    d["mission_target_sessions"] = MISSION_TARGET_SESSIONS
                    d["candidate_source"] = "ORCHESTRATOR_AGENT_CONSENSUS (Moreshwar 3.2)"
                    d["orchestrator_scan_summary"] = public["stages"]
                    daily.write_text(json.dumps(d, indent=2, default=str), encoding="utf-8")
                except Exception:
                    pass

        self.ledger.export_learning_json(
            self.scan_dir / "learning_report.json",
            paper_snapshot=paper_snap,
            extra={"scanner_version": SCANNER_VERSION},
        )

    def build_dashboard_payload(self) -> dict[str, Any]:
        """Build complete Moreshwar 3.2 dashboard payload combining paper ledger + orchestrator scan + learning."""
        base = self.orch.build_dashboard_payload()
        latest_scan = None
        scan_path = self.scan_dir / "latest_scan.json"
        if scan_path.exists():
            try:
                latest_scan = json.loads(scan_path.read_text(encoding="utf-8"))
            except Exception:
                latest_scan = None
        learn_path = self.scan_dir / "learning_report.json"
        learning_doc = None
        if learn_path.exists():
            try:
                learning_doc = json.loads(learn_path.read_text(encoding="utf-8"))
            except Exception:
                learning_doc = None
        if learning_doc is None:
            learning_doc = {
                "mission": self.ledger.build_mission_status(self.orch.engine.get_full_ledger_snapshot()),
                "weights": self.ledger.current_params(),
                "weight_history": [],
                "learning": self.ledger.learning_report(),
            }
        base["release_version"] = RELEASE_VERSION_M3_2
        base["candidate_source"] = "ORCHESTRATOR_AGENT_CONSENSUS"
        base["mission"] = learning_doc.get("mission")
        base["orchestrator_scan"] = latest_scan
        base["orchestrator_learning"] = learning_doc
        return base
