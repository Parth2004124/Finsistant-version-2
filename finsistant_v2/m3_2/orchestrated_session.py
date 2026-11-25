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

from finsistant_v2.m3_1.adaptive_strategy_engine import AdaptiveRegimePolicy
from finsistant_v2.m3_1.live_nse_eod_feed import fetch_completed_nse_eod_snapshot
from finsistant_v2.m3_1.paper_trading_orchestrator import PaperTradingOrchestrator
from finsistant_v2.m3_1.paper_trading_scheduler import PaperTradingScheduler
from finsistant_v2.m3_1.prospective_paper_trading_engine import REPO_ROOT
from finsistant_v2.trade_automation import (
    EMSOrderBlotter,
    RuleBuilderEngine,
)

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
MIN_EOD_UNIVERSE_BARS = 800

# MOVIQ 3.2 aims to beat NIFTYBEES significantly via active AI stock picks,
# so it holds 0% passive NIFTYBEES and deploys 90% into Tactical AI picks in BULL_TREND
# (keeping MOVIQ 3.1 frozen at 60/30/10 as the conservative baseline).
M32_ALPHA_REGIME_POLICY = AdaptiveRegimePolicy(
    policy_id="M32_ACTIVE_ALPHA_0_90_10",
    policy_name="MOVIQ 3.2 Active Alpha (BULL: 0% NIFTYBEES / 90% AI Picks / 10% Liquid)",
    regime_source="SIMPLE_TREND",
    regime_rules=(
        ("BULL_TREND", 0.0, 90.0, 10.0, True, 0.0, 1.0),
        ("BEAR_TREND", 0.0, 30.0, 70.0, False, 80.0, 0.7),
    ),
    allocation_confirm_days=3,
)
M32_EXEC_CFG_OVERRIDE: dict[str, Any] = {
    "core_target_pct": 0.0,
    "tactical_target_pct": 90.0,
    "liquid_target_pct": 10.0,
    "sweep_unused_tactical_to_core": False,
    "max_tactical_positions": 8,
    "max_single_stock_pct": 15.0,
    "hurdle_mode": "DYNAMIC_VOL",
    "sizing_mode": "F_HYBRID_RISK_ALPHA",
}


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
            regime_policy_override=M32_ALPHA_REGIME_POLICY,
            exec_cfg_override=M32_EXEC_CFG_OVERRIDE,
        )
        self.ledger = OrchestratorLearningLedger(learning_db_path)
        self.scanner = scanner or OrchestratorUniverseScanner(OrchestratorScanConfig())
        self.rbld = RuleBuilderEngine()
        self.scan_dir = Path(scan_dir)
        self.reports_dir = Path(reports_dir)
        self.last_scan: dict[str, Any] | None = None
        self.last_exit_signals: list[dict[str, Any]] = []
        self.last_blotter: dict[str, Any] | None = None

    # -- provider handed to PaperTradingScheduler -------------------------------------------------
    def market_data_provider(self, trading_date: str) -> dict[str, Any] | None:
        self.last_scan = None
        self.last_exit_signals = []
        self.last_blotter = None
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
        snap = fetch_completed_nse_eod_snapshot(
            trading_date,
            extra_open_symbols=extra_open,
            include_histories=True,
            prefer_nse_bhavcopy=True,
        )
        if snap is None:
            return None
        eq_count = len(snap.get("equity_bars") or {})
        if eq_count < MIN_EOD_UNIVERSE_BARS:
            raise RuntimeError(
                f"INCOMPLETE_EOD_UNIVERSE on {trading_date}: only {eq_count} equity bars loaded "
                f"(minimum required {MIN_EOD_UNIVERSE_BARS}); failing upfront instead of scanning a broken subset."
            )
        histories = snap.pop("equity_histories")
        nb_dates = snap.pop("niftybees_dates")
        nb_closes = snap.pop("niftybees_closes_full")

        # Learn from prior closed trades + newly matured 5D/10D/20D real outcomes BEFORE scanning today
        new_lessons = self.ledger.sync_closed_trades(self.orch.engine.db_path)
        matured = self.ledger.mature_outcomes(
            today=trading_date, equity_histories=histories, nb_dates=nb_dates, nb_closes=nb_closes
        )
        upd = self.ledger.update_weights(trading_date)
        btca_summary = self.ledger.btca_report()

        nav = float(ledger_snap.get("portfolio_state", {}).get("last_nav_inr", 100_000.0) or 100_000.0)
        dd_pct = float((ledger_snap.get("risk_metrics") or {}).get("max_drawdown_pct") or 0.0)
        recon_diff = float((ledger_snap.get("reconciliation") or {}).get("reconciliation_diff_inr") or 0.0)
        eng_state = str((ledger_snap.get("engine_state") or {}).get("engine_state") or "RUNNING")
        desk_halted, ks_reasons = self.rbld.evaluate_desk_kill_switch(
            engine_state=eng_state,
            drawdown_pct=dd_pct,
            recon_diff_inr=recon_diff,
            btca_drift_breached=bool(btca_summary.get("kill_switch_drift_breached")),
        )
        blotter = EMSOrderBlotter(desk_halted=desk_halted, kill_switch_reasons=ks_reasons)

        exit_signals = self.scanner.evaluate_open_holdings(
            snap_open,
            learned_params=upd,
            capital_inr=nav,
        )
        self.last_exit_signals = list(exit_signals)
        for ex in exit_signals:
            bar = snap["equity_bars"].get(str(ex["sec_id"]), {})
            ex_px = float(bar.get("close") or 0.0)
            ex_verdict = self.rbld.evaluate_order(
                symbol=str(ex["symbol"]),
                side="SELL",
                arrival_price=ex_px,
                order_notional_inr=ex_px * 1.0,
                nav_inr=nav,
                adv_20d_inr=float(bar.get("adv_20d_inr") or 80_000_000.0),
                hl_range_pct=float(bar.get("hl_range_pct") or 2.0),
                daily_vol_frac=float(bar.get("vol_20d") or 0.018),
                bar_valid=bool(bar.get("valid", True)),
                desk_halted=desk_halted,
                desk_halt_reasons=ks_reasons,
                prefer_algo_wheel=True,
                require_min_adv=False,
            )
            blotter.record_compliance_result(ex_verdict, qty=0)

        scan = self.scanner.scan(
            trading_date=trading_date,
            equity_histories=histories,
            eq_bars=snap["equity_bars"],
            learned_params=upd,
            capital_inr=nav,
        )
        # Record any scanner symbol failures explicitly into EMS exception blotter
        for row in scan.get("shortlist") or []:
            if str(row.get("status") or "") in ("FAILED_DATA_UNAVAILABLE", "EXCLUDED_PRICE_DISCREPANCY"):
                blotter.record_data_exception(
                    str(row.get("symbol") or ""),
                    str(row.get("error") or row.get("status")),
                    source="ORCHESTRATOR_SHORTLIST",
                )

        shortlist_by_sid = {str(r.get("sec_id")): r for r in (scan.get("shortlist") or [])}
        open_sec_exp: dict[str, float] = {}
        for p in snap_open:
            sn = str(p.get("sector") or "UNKNOWN_SECTOR")
            open_sec_exp[sn] = open_sec_exp.get(sn, 0.0) + float(p.get("qty") or 0) * float(p.get("last_price") or 0.0)
        base_slot_inr = (nav * 0.90) / max(1, int(M32_EXEC_CFG_OVERRIDE["max_tactical_positions"]))
        tca_adder = float(btca_summary.get("tca_hurdle_adder_bps") or 0.0)

        approved_candidates: list[dict[str, Any]] = []
        for cand in scan.get("candidates") or []:
            sid = str(cand["sec_id"])
            bar = snap["equity_bars"].get(sid, {})
            sl_row = shortlist_by_sid.get(sid, {})
            c_px = float(bar.get("close") or sl_row.get("eod_close") or 0.0)
            sec_nm = str(bar.get("sector") or sl_row.get("sector") or "UNKNOWN_SECTOR")
            est_notional = min(base_slot_inr, nav * (float(M32_EXEC_CFG_OVERRIDE["max_single_stock_pct"]) / 100.0))
            sec_after_pct = ((open_sec_exp.get(sec_nm, 0.0) + est_notional) / max(1.0, nav)) * 100.0
            verdict = self.rbld.evaluate_order(
                symbol=str(cand["symbol"]),
                side="BUY",
                arrival_price=c_px,
                order_notional_inr=est_notional,
                nav_inr=nav,
                adv_20d_inr=float(bar.get("adv_20d_inr") or 0.0),
                hl_range_pct=float(bar.get("hl_range_pct") or 2.0),
                daily_vol_frac=float(bar.get("vol_20d") or 0.018),
                fundamentals_status="live" if sl_row.get("status") == "AGENTS_OK" else "unavailable",
                bar_valid=bool(bar.get("valid", True)),
                price_discrepancy_pct=float(sl_row.get("price_discrepancy_pct") or 0.0),
                sector_exposure_after_pct=sec_after_pct,
                open_positions_count=len(snap_open),
                max_positions=int(M32_EXEC_CFG_OVERRIDE["max_tactical_positions"]),
                desk_halted=desk_halted,
                desk_halt_reasons=ks_reasons,
                prefer_algo_wheel=True,
                require_min_adv=True,
            )
            blotter.record_compliance_result(verdict, qty=int(est_notional // max(1e-6, c_px)) if c_px > 0 else 0)
            if verdict["compliance_passed"]:
                enriched_cand = dict(cand)
                enriched_cand["execution_mode"] = verdict["selected_algo"]
                enriched_cand["pre_trade_cost_bps"] = verdict["pre_trade_tca"]["est_total_cost_bps"]
                enriched_cand["matched_rules"] = verdict["matched_rules"]
                enriched_cand["tca_hurdle_adder_bps"] = tca_adder
                approved_candidates.append(enriched_cand)

        self.last_blotter = blotter.to_payload(rules=self.rbld.get_rules(), btca_summary=btca_summary)
        scan["candidates"] = approved_candidates
        scan["outcomes_matured_this_run"] = matured
        scan["trade_lessons_synced_pre_scan"] = new_lessons
        scan["exit_signals"] = list(exit_signals)
        scan["ems_blotter"] = self.last_blotter
        self.last_scan = scan

        snap["family_e_candidates"] = approved_candidates
        snap["candidate_source"] = "ORCHESTRATOR_AGENT_CONSENSUS"
        snap["tactical_exit_signals"] = exit_signals
        snap["pairwise_corr_map"] = pairwise_corr_for_candidates(approved_candidates, histories)
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
        exec_orders = ((res.get("execution_result") or {}).get("orders") or [])
        self.finalize_reports(trading_date, session_orders=exec_orders)
        return res

    def finalize_reports(self, trading_date: str, *, session_orders: list[dict[str, Any]] | None = None) -> None:
        paper_snap = self.orch.engine.get_full_ledger_snapshot()
        # Sync Post-Trade BTCA execution analytics & closed-trade post-mortems immediately
        if session_orders:
            self.ledger.sync_btca_executions(session_orders)
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
            public["exit_signals"] = list(self.last_exit_signals)
            public["ems_blotter"] = self.last_blotter
            public["allocation_policy"] = "0% NIFTYBEES / 90% Tactical AI Picks / 10% Liquid (BULL_TREND)"
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
                    d["candidate_source"] = "ORCHESTRATOR_AGENT_CONSENSUS (MOVIQ 3.2)"
                    d["allocation_policy"] = public["allocation_policy"]
                    d["exit_signals"] = list(self.last_exit_signals)
                    d["ems_blotter"] = self.last_blotter
                    d["orchestrator_scan_summary"] = public["stages"]
                    daily.write_text(json.dumps(d, indent=2, default=str), encoding="utf-8")
                except Exception:
                    pass

        self.ledger.export_learning_json(
            self.scan_dir / "learning_report.json",
            paper_snapshot=paper_snap,
            extra={
                "scanner_version": SCANNER_VERSION,
                "allocation_policy": "0% NIFTYBEES / 90% Tactical AI Picks / 10% Liquid (BULL_TREND)",
            },
        )

    def build_dashboard_payload(self) -> dict[str, Any]:
        """Build complete MOVIQ 3.2 dashboard payload combining paper ledger + orchestrator scan + learning + BTCA."""
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
                "btca": self.ledger.btca_report(),
            }
        base["release_version"] = RELEASE_VERSION_M3_2
        base["candidate_source"] = "ORCHESTRATOR_AGENT_CONSENSUS"
        base["mission"] = learning_doc.get("mission")
        base["orchestrator_scan"] = latest_scan
        base["orchestrator_learning"] = learning_doc
        base["ems_blotter"] = (latest_scan or {}).get("ems_blotter") or self.last_blotter
        base["btca"] = learning_doc.get("btca") or self.ledger.btca_report()
        return base
