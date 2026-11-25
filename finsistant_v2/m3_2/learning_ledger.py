"""Moreshwar 3.2 — durable learning ledger that makes the orchestrator better with EVERY passed & failed trade.

Mission: 80-session paper-trading mock test targeting maximum risk-adjusted profit before real money
is ever considered. Every session the ledger:
  1. Persists the full-universe TechBot scores and the FundBot/ExecBot shortlist (`scan_universe`,
     `scan_shortlist`).
  2. Syncs every CLOSED paper trade from `prospective_paper_ledger.db` into `trade_lessons`,
     joining it with the exact agent scores & indicators from the signal session, classifying it as
     PASSED (net realized P&L + dividends > 0) or FAILED (<= 0), and recording a concrete post-mortem.
  3. Matures 1/5/10/20-trading-day real forward excess returns vs NIFTYBEES across BOTH taken picks
     and skipped/shortlisted stocks (`outcomes`).
  4. Immediately updates bounded, audit-logged orchestrator parameters via smooth Bayesian shrinkage
     (starting from the very first closed trade or matured session — no artificial 20-session freeze):
       - `w_tech` / `w_fund`  (agent conviction blend, bounded [0.25, 0.75], max step +/-0.03/session)
       - `min_conviction`     (entry conviction bar, bounded [0.56, 0.76], max step +/-0.02/session)
       - `rsi_overbought_penalty` (bounded [0.00, 0.12], penalizes RSI > 72 if overbought entries fail)
       - `sector_adjustments` (bounded [-0.05, +0.05] per sector, Bayesian shrunk by n / (n + 4))
  5. Evaluates the 80-Session Mission & Real-Money Readiness Gate (all paper-only locks stay hard-on).
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from typing import Any

import numpy as np

from finsistant_v2.adversarial_prosecutor import (
    ATTACK_MULTIPLIER_MAX,
    ATTACK_MULTIPLIER_MIN,
    DEFAULT_MIN_SURVIVAL_SCORE,
)
from finsistant_v2.trade_automation import (
    compute_post_trade_tca,
    summarize_btca_and_feedback,
)

from .orchestrator_scanner import (
    MISSION_TARGET_SESSIONS,
    PRIOR_MIN_CONVICTION,
    PRIOR_RSI_OVERBOUGHT_PENALTY,
    PRIOR_W_FUND,
    PRIOR_W_TECH,
)

HORIZONS = (1, 5, 10, 20)
LEARNING_HORIZON = 5
MIN_UNIVERSE_OBS_PER_SESSION = 30
MIN_SHORTLIST_OBS_PER_SESSION = 5

W_TECH_MIN, W_TECH_MAX = 0.25, 0.75
MIN_CONV_FLOOR, MIN_CONV_CEIL = 0.56, 0.76
RSI_PEN_MIN, RSI_PEN_MAX = 0.0, 0.12
SECTOR_ADJ_LIMIT = 0.05
PRIOR_STOP_TIGHTEN_MULT = 1.0
STOP_MULT_MIN, STOP_MULT_MAX = 0.80, 1.10
SURVIVAL_FLOOR, SURVIVAL_CEIL = 48.0, 74.0

MAX_WEIGHT_STEP = 0.03
MAX_CONV_STEP = 0.02
MAX_RSI_PEN_STEP = 0.02
MAX_STOP_MULT_STEP = 0.05
MAX_SURVIVAL_STEP = 4.0
SHRINKAGE_K_TRADES = 6.0
SHRINKAGE_K_SESSIONS = 8.0


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(v)))


def _rank_avg(a: np.ndarray) -> np.ndarray:
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=np.float64)
    sorted_a = a[order]
    i, n = 0, len(a)
    while i < n:
        j = i
        while j + 1 < n and sorted_a[j + 1] == sorted_a[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def spearman_ic(x: list[float], y: list[float]) -> float | None:
    if len(x) < 5 or len(x) != len(y):
        return None
    xa, ya = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    ok = np.isfinite(xa) & np.isfinite(ya)
    xa, ya = xa[ok], ya[ok]
    if len(xa) < 5:
        return None
    rx, ry = _rank_avg(xa), _rank_avg(ya)
    if np.std(rx) < 1e-12 or np.std(ry) < 1e-12:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


class OrchestratorLearningLedger:
    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        conn = self._connect()
        try:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS scan_universe (
                    trading_date TEXT NOT NULL, sec_id TEXT NOT NULL, symbol TEXT NOT NULL,
                    tech_conf REAL NOT NULL, rank_key REAL NOT NULL, close REAL NOT NULL,
                    rsi_14 REAL, sector TEXT,
                    PRIMARY KEY (trading_date, sec_id)
                );
                CREATE TABLE IF NOT EXISTS scan_shortlist (
                    trading_date TEXT NOT NULL, sec_id TEXT NOT NULL, symbol TEXT NOT NULL,
                    tech_rank INTEGER, status TEXT NOT NULL, tech_conf REAL, fund_conf REAL,
                    fundamental_total REAL, porters_total REAL, conviction REAL, exec_action TEXT,
                    eod_close REAL, queued INTEGER NOT NULL DEFAULT 0,
                    rsi_14 REAL, sector TEXT,
                    prosecution_verdict TEXT, survival_score REAL, attack_ids_json TEXT,
                    PRIMARY KEY (trading_date, sec_id)
                );
                CREATE TABLE IF NOT EXISTS outcomes (
                    trading_date TEXT NOT NULL, sec_id TEXT NOT NULL, horizon INTEGER NOT NULL,
                    fwd_ret REAL NOT NULL, bench_ret REAL NOT NULL, excess REAL NOT NULL,
                    matured_on TEXT NOT NULL,
                    PRIMARY KEY (trading_date, sec_id, horizon)
                );
                CREATE TABLE IF NOT EXISTS trade_lessons (
                    position_id TEXT PRIMARY KEY,
                    sec_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    sector TEXT,
                    signal_date TEXT,
                    entry_date TEXT NOT NULL,
                    exit_date TEXT NOT NULL,
                    bars_held INTEGER NOT NULL,
                    exit_reason TEXT,
                    cost_basis_inr REAL NOT NULL,
                    realized_pnl_inr REAL NOT NULL,
                    dividends_inr REAL NOT NULL DEFAULT 0.0,
                    total_pnl_inr REAL NOT NULL,
                    return_pct REAL NOT NULL,
                    mae_pct REAL,
                    mfe_pct REAL,
                    outcome TEXT NOT NULL,
                    tech_conf REAL,
                    fund_conf REAL,
                    conviction REAL,
                    rsi_14 REAL,
                    lesson_summary TEXT NOT NULL,
                    recorded_at_utc TEXT NOT NULL,
                    mfe_giveback_pct REAL,
                    pillar_attribution TEXT,
                    rupees_at_risk_inr REAL,
                    pnl_per_1k_inr REAL,
                    rupee_days REAL,
                    survival_score REAL,
                    attack_ids_json TEXT
                );
                CREATE TABLE IF NOT EXISTS weight_history (
                    trading_date TEXT PRIMARY KEY,
                    w_tech REAL NOT NULL,
                    w_fund REAL NOT NULL,
                    min_conviction REAL NOT NULL DEFAULT 0.60,
                    rsi_overbought_penalty REAL NOT NULL DEFAULT 0.0,
                    sector_adjustments_json TEXT NOT NULL DEFAULT '{}',
                    stop_tighten_mult REAL NOT NULL DEFAULT 1.0,
                    attack_multipliers_json TEXT NOT NULL DEFAULT '{}',
                    min_survival_score REAL NOT NULL DEFAULT 52.0,
                    total_rupees_invested_inr REAL NOT NULL DEFAULT 0.0,
                    rupee_roic_pct REAL,
                    status TEXT NOT NULL,
                    matured_sessions INTEGER NOT NULL,
                    closed_trades INTEGER NOT NULL DEFAULT 0,
                    passed_trades INTEGER NOT NULL DEFAULT 0,
                    failed_trades INTEGER NOT NULL DEFAULT 0,
                    ic_tech REAL,
                    ic_fund REAL,
                    note TEXT,
                    recorded_at_utc TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS btca_executions (
                    order_id TEXT PRIMARY KEY,
                    trading_date TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    algo_strategy TEXT NOT NULL,
                    arrival_price REAL,
                    bar_open REAL,
                    actual_exec_price REAL,
                    req_qty REAL NOT NULL,
                    filled_qty REAL NOT NULL,
                    fill_rate_pct REAL NOT NULL,
                    delay_cost_bps REAL,
                    execution_slippage_bps REAL,
                    statutory_cost_bps REAL,
                    implementation_shortfall_bps REAL,
                    slippage_surprise_bps REAL,
                    execution_status TEXT NOT NULL,
                    recorded_at_utc TEXT NOT NULL
                );
                """
            )
            # Safe schema migration for any existing v3.2 alpha DB created before extra columns
            self._ensure_columns(conn, "scan_universe", [("rsi_14", "REAL"), ("sector", "TEXT")])
            self._ensure_columns(
                conn,
                "scan_shortlist",
                [
                    ("rsi_14", "REAL"),
                    ("sector", "TEXT"),
                    ("prosecution_verdict", "TEXT"),
                    ("survival_score", "REAL"),
                    ("attack_ids_json", "TEXT"),
                ],
            )
            self._ensure_columns(
                conn,
                "trade_lessons",
                [
                    ("mfe_giveback_pct", "REAL"),
                    ("pillar_attribution", "TEXT"),
                    ("rupees_at_risk_inr", "REAL"),
                    ("pnl_per_1k_inr", "REAL"),
                    ("rupee_days", "REAL"),
                    ("survival_score", "REAL"),
                    ("attack_ids_json", "TEXT"),
                ],
            )
            self._ensure_columns(
                conn,
                "weight_history",
                [
                    ("min_conviction", "REAL NOT NULL DEFAULT 0.60"),
                    ("rsi_overbought_penalty", "REAL NOT NULL DEFAULT 0.0"),
                    ("sector_adjustments_json", "TEXT NOT NULL DEFAULT '{}'"),
                    ("stop_tighten_mult", "REAL NOT NULL DEFAULT 1.0"),
                    ("attack_multipliers_json", "TEXT NOT NULL DEFAULT '{}'"),
                    ("min_survival_score", "REAL NOT NULL DEFAULT 52.0"),
                    ("total_rupees_invested_inr", "REAL NOT NULL DEFAULT 0.0"),
                    ("rupee_roic_pct", "REAL"),
                    ("closed_trades", "INTEGER NOT NULL DEFAULT 0"),
                    ("passed_trades", "INTEGER NOT NULL DEFAULT 0"),
                    ("failed_trades", "INTEGER NOT NULL DEFAULT 0"),
                ],
            )
            conn.commit()
        finally:
            conn.close()

    @staticmethod
    def _ensure_columns(conn: sqlite3.Connection, table: str, cols: list[tuple[str, str]]) -> None:
        existing = {r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()}
        for col, col_def in cols:
            if col not in existing:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {col_def}")

    # ------------------------------------------------------------------
    # 1. Persist each session's full-universe scan & shortlist
    # ------------------------------------------------------------------
    def record_scan(self, scan: dict[str, Any], queued_symbols: set[str] | None = None) -> None:
        dt = str(scan["trading_date"])
        queued = {str(s) for s in (queued_symbols or set())}
        conn = self._connect()
        try:
            conn.execute("DELETE FROM scan_universe WHERE trading_date=?", (dt,))
            conn.execute("DELETE FROM scan_shortlist WHERE trading_date=?", (dt,))
            conn.executemany(
                "INSERT INTO scan_universe (trading_date, sec_id, symbol, tech_conf, rank_key, close, rsi_14, sector) "
                "VALUES (?,?,?,?,?,?,?,?)",
                [
                    (dt, r["sec_id"], r["symbol"], r["tech_conf"], r["rank_key"], r["close"], r.get("rsi_14"), r.get("sector"))
                    for r in scan["universe_tech"]
                ],
            )
            conn.executemany(
                "INSERT INTO scan_shortlist (trading_date, sec_id, symbol, tech_rank, status, tech_conf, fund_conf, "
                "fundamental_total, porters_total, conviction, exec_action, eod_close, queued, rsi_14, sector, "
                "prosecution_verdict, survival_score, attack_ids_json) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                [
                    (
                        dt,
                        r["sec_id"],
                        r["symbol"],
                        r.get("tech_rank"),
                        r["status"],
                        r.get("tech_conf"),
                        r.get("fund_conf"),
                        r.get("fundamental_total"),
                        r.get("porters_total"),
                        r.get("conviction"),
                        r.get("exec_action"),
                        r.get("eod_close"),
                        1 if r["symbol"] in queued else 0,
                        r.get("rsi_14"),
                        r.get("sector"),
                        r.get("prosecution_verdict"),
                        r.get("survival_score"),
                        json.dumps(r.get("attack_ids") or []),
                    )
                    for r in scan["shortlist"]
                ],
            )
            conn.commit()
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # 2. Sync EVERY closed paper trade into per-rupee post-mortem lessons
    # ------------------------------------------------------------------
    def sync_closed_trades(self, paper_db_path: Path) -> int:
        paper_db_path = Path(paper_db_path)
        if not paper_db_path.exists():
            return 0
        pconn = sqlite3.connect(str(paper_db_path), timeout=30.0)
        pconn.row_factory = sqlite3.Row
        try:
            closed = pconn.execute(
                "SELECT * FROM positions WHERE status = 'CLOSED' ORDER BY exit_date ASC, position_id ASC"
            ).fetchall()
        except sqlite3.OperationalError:
            return 0
        finally:
            pconn.close()

        if not closed:
            return 0

        conn = self._connect()
        added = 0
        try:
            existing = {r[0] for r in conn.execute("SELECT position_id FROM trade_lessons").fetchall()}
            for pos in closed:
                pid = str(pos["position_id"])
                if pid in existing:
                    continue
                sec_id = str(pos["sec_id"])
                sym = str(pos["symbol"])
                entry_dt = str(pos["entry_date"])
                exit_dt = str(pos["exit_date"] or entry_dt)
                cost_basis = max(1e-6, float(pos["cost_basis_inr"] or 0.0))
                realized = float(pos["realized_pnl_inr"] or 0.0)
                divs = float(pos["dividends_received_inr"] or 0.0)
                total_pnl = realized + divs
                ret_pct = (total_pnl / cost_basis) * 100.0
                bars_held = int(pos["bars_held"] or 0)
                exit_reason = str(pos["exit_reason"] or "CLOSED")
                mae = float(pos["mae_pct"]) if pos["mae_pct"] is not None else None
                mfe = float(pos["mfe_pct"]) if pos["mfe_pct"] is not None else None
                outcome = "PASSED" if total_pnl > 0.0 else "FAILED"

                # Per-rupee capital attribution metrics
                rupees_at_risk = round(cost_basis * abs(min(0.0, float(mae or 0.0))) / 100.0, 2)
                pnl_per_1k = round((total_pnl / max(1.0, cost_basis)) * 1000.0, 2)
                rupee_days = round(cost_basis * max(1, bars_held), 2)

                # Join with exact shortlist / universe agent scores & adversarial objections on or before entry_date
                sl = conn.execute(
                    "SELECT trading_date, tech_conf, fund_conf, conviction, rsi_14, sector, survival_score, attack_ids_json "
                    "FROM scan_shortlist WHERE sec_id=? AND trading_date < ? ORDER BY trading_date DESC LIMIT 1",
                    (sec_id, entry_dt),
                ).fetchone()
                if sl is None:
                    sl = conn.execute(
                        "SELECT trading_date, tech_conf, NULL AS fund_conf, NULL AS conviction, rsi_14, sector, "
                        "NULL AS survival_score, '[]' AS attack_ids_json "
                        "FROM scan_universe WHERE sec_id=? AND trading_date <= ? ORDER BY trading_date DESC LIMIT 1",
                        (sec_id, entry_dt),
                    ).fetchone()

                sig_dt = str(sl["trading_date"]) if sl else None
                t_c = float(sl["tech_conf"]) if (sl and sl["tech_conf"] is not None) else None
                f_c = float(sl["fund_conf"]) if (sl and sl["fund_conf"] is not None) else None
                conv = float(sl["conviction"]) if (sl and sl["conviction"] is not None) else None
                rsi = float(sl["rsi_14"]) if (sl and sl["rsi_14"] is not None) else None
                surv_score = float(sl["survival_score"]) if (sl and sl["survival_score"] is not None) else None
                try:
                    attack_ids = json.loads((sl["attack_ids_json"] if sl else None) or "[]")
                    if not isinstance(attack_ids, list):
                        attack_ids = []
                except Exception:
                    attack_ids = []
                attack_ids_json = json.dumps(attack_ids)
                sector = str(pos["sector"] or (sl["sector"] if sl else "UNKNOWN_SECTOR"))

                # Concrete deterministic per-rupee post-mortem lesson
                parts = [
                    f"{outcome} ({ret_pct:+.2f}% / ₹{total_pnl:+,.2f} on ₹{cost_basis:,.0f} = ₹{pnl_per_1k:+.2f}/₹1k in {bars_held}d via {exit_reason})"
                ]
                if t_c is not None and f_c is not None:
                    parts.append(f"Tech={t_c:.2f}, Fund={f_c:.2f}, Conv={conv:.2f}" if conv is not None else f"Tech={t_c:.2f}, Fund={f_c:.2f}")
                if attack_ids:
                    parts.append(
                        f"Objections [{', '.join(attack_ids)}] "
                        + ("predicted ₹ loss -> escalating prosecutor multiplier" if outcome == "FAILED" else "survived cleanly")
                    )
                if outcome == "FAILED":
                    if rsi is not None and rsi > 72.0:
                        parts.append(f"Overbought entry RSI={rsi:.1f} -> reinforces RSI guard")
                    if t_c is not None and f_c is not None and abs(t_c - f_c) >= 0.10:
                        weaker = "TechBot" if t_c > f_c else "FundBot"
                        parts.append(f"{weaker} over-scored ₹{cost_basis:,.0f} losing trade -> rupee-weighted penalty")
                    if conv is not None and conv < 0.68:
                        parts.append("Low-conviction entry failed -> nudges min_conviction upward")
                else:
                    if t_c is not None and f_c is not None and abs(t_c - f_c) >= 0.08:
                        stronger = "TechBot" if t_c > f_c else "FundBot"
                        parts.append(f"{stronger} led ₹{total_pnl:+,.0f} winning trade -> reinforcing weight")
                mfe_giveback = round(max(0.0, float(mfe) - ret_pct), 3) if mfe is not None else None
                if mfe_giveback is not None and mfe_giveback > 4.0:
                    parts.append(f"Peak MFE +{mfe:.1f}% vs exit {ret_pct:+.1f}% (gave back {mfe_giveback:.1f}%)")
                if t_c is not None and f_c is not None:
                    if outcome == "PASSED":
                        pillar_attr = "TECH_LED" if t_c >= f_c + 0.05 else ("FUND_LED" if f_c >= t_c + 0.05 else "BALANCED_WIN")
                    else:
                        pillar_attr = "TECH_OVERCONFIDENT" if t_c >= f_c + 0.05 else ("FUND_VALUE_TRAP" if f_c >= t_c + 0.05 else "JOINT_MISS")
                else:
                    pillar_attr = "TECH_ONLY" if t_c is not None else "UNATTRIBUTED"

                lesson_txt = " · ".join(parts)
                conn.execute(
                    "INSERT INTO trade_lessons ("
                    "position_id, sec_id, symbol, sector, signal_date, entry_date, exit_date, bars_held, exit_reason, "
                    "cost_basis_inr, realized_pnl_inr, dividends_inr, total_pnl_inr, return_pct, mae_pct, mfe_pct, "
                    "outcome, tech_conf, fund_conf, conviction, rsi_14, lesson_summary, recorded_at_utc, "
                    "mfe_giveback_pct, pillar_attribution, rupees_at_risk_inr, pnl_per_1k_inr, rupee_days, survival_score, attack_ids_json"
                    ") VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        pid,
                        sec_id,
                        sym,
                        sector,
                        sig_dt,
                        entry_dt,
                        exit_dt,
                        bars_held,
                        exit_reason,
                        round(cost_basis, 2),
                        round(realized, 2),
                        round(divs, 2),
                        round(total_pnl, 2),
                        round(ret_pct, 3),
                        round(mae, 3) if mae is not None else None,
                        round(mfe, 3) if mfe is not None else None,
                        outcome,
                        t_c,
                        f_c,
                        conv,
                        rsi,
                        lesson_txt,
                        _utc_now_iso(),
                        mfe_giveback,
                        pillar_attr,
                        rupees_at_risk,
                        pnl_per_1k,
                        rupee_days,
                        surv_score,
                        attack_ids_json,
                    ),
                )
                added += 1
            conn.commit()
        finally:
            conn.close()
        return added

    # ------------------------------------------------------------------
    # 3. Mature real forward returns across BOTH taken & skipped stocks
    # ------------------------------------------------------------------
    def mature_outcomes(
        self,
        *,
        today: str,
        equity_histories: dict[str, dict[str, Any]],
        nb_dates: list[str],
        nb_closes: list[float],
    ) -> int:
        conn = self._connect()
        inserted = 0
        try:
            nb_idx = {d: i for i, d in enumerate(nb_dates)}
            scan_dates = [r[0] for r in conn.execute("SELECT DISTINCT trading_date FROM scan_universe WHERE trading_date < ?", (today,))]
            sec_date_idx: dict[str, dict[str, int]] = {}
            for dt in scan_dates:
                if dt not in nb_idx:
                    continue
                rows = conn.execute("SELECT sec_id FROM scan_universe WHERE trading_date=?", (dt,)).fetchall()
                done = {
                    (r[0], r[1])
                    for r in conn.execute("SELECT sec_id, horizon FROM outcomes WHERE trading_date=?", (dt,))
                }
                batch: list[tuple[Any, ...]] = []
                for (sid,) in rows:
                    h = equity_histories.get(sid)
                    if not h:
                        continue
                    if sid not in sec_date_idx:
                        sec_date_idx[sid] = {d: i for i, d in enumerate(h["dates"])}
                    idx_map = sec_date_idx[sid]
                    if dt not in idx_map:
                        continue
                    i0 = idx_map[dt]
                    c = h["close"]
                    for hz in HORIZONS:
                        if (sid, hz) in done:
                            continue
                        i1 = i0 + hz
                        j0 = nb_idx[dt]
                        j1 = j0 + hz
                        if i1 >= len(c) or j1 >= len(nb_closes):
                            continue
                        if float(c[i0]) <= 0 or float(nb_closes[j0]) <= 0:
                            continue
                        fwd = float(c[i1]) / float(c[i0]) - 1.0
                        bench = float(nb_closes[j1]) / float(nb_closes[j0]) - 1.0
                        batch.append((dt, sid, hz, fwd, bench, fwd - bench, today))
                if batch:
                    conn.executemany("INSERT OR IGNORE INTO outcomes VALUES (?,?,?,?,?,?,?)", batch)
                    inserted += len(batch)
            conn.commit()
        finally:
            conn.close()
        return inserted

    # ------------------------------------------------------------------
    # 4. Compute full learning diagnostics (closed trades + matured IC)
    # ------------------------------------------------------------------
    def closed_trade_summary(self) -> dict[str, Any]:
        conn = self._connect()
        try:
            rows = [dict(r) for r in conn.execute("SELECT * FROM trade_lessons ORDER BY exit_date DESC, position_id DESC").fetchall()]
        finally:
            conn.close()

        passed = [r for r in rows if r["outcome"] == "PASSED"]
        failed = [r for r in rows if r["outcome"] == "FAILED"]
        gross_win = sum(float(r["total_pnl_inr"]) for r in passed)
        gross_loss = abs(sum(float(r["total_pnl_inr"]) for r in failed))
        pf = round(gross_win / gross_loss, 3) if gross_loss > 0 else (None if not passed else 9.99)
        total_pnl = sum(float(r["total_pnl_inr"]) for r in rows)
        total_invested = sum(float(r.get("cost_basis_inr") or 0.0) for r in rows)
        rupee_roic = round((total_pnl / total_invested) * 100.0, 3) if total_invested > 0 else None
        pnl_per_1k = round((total_pnl / total_invested) * 1000.0, 2) if total_invested > 0 else None
        total_at_risk = sum(
            float(r["rupees_at_risk_inr"])
            if r.get("rupees_at_risk_inr") is not None
            else (float(r.get("cost_basis_inr") or 0.0) * abs(min(0.0, float(r.get("mae_pct") or 0.0))) / 100.0)
            for r in rows
        )
        total_rupee_days = sum(
            float(r["rupee_days"])
            if r.get("rupee_days") is not None
            else float(r.get("cost_basis_inr") or 0.0) * max(1, int(r.get("bars_held") or 1))
            for r in rows
        )

        # Track real rupees gained/lost per adversarial objection raised at entry
        atk_map: dict[str, dict[str, Any]] = {}
        for r in rows:
            try:
                atk_ids = json.loads(r.get("attack_ids_json") or "[]")
            except Exception:
                atk_ids = []
            if not isinstance(atk_ids, list):
                atk_ids = []
            cb = float(r.get("cost_basis_inr") or 0.0)
            pnl = float(r.get("total_pnl_inr") or 0.0)
            for aid in atk_ids:
                bucket = atk_map.setdefault(
                    str(aid),
                    {"attack_id": str(aid), "trades": 0, "failed_trades": 0, "rupees_invested_inr": 0.0, "net_rupee_pnl_inr": 0.0, "rupees_lost_inr": 0.0},
                )
                bucket["trades"] += 1
                bucket["rupees_invested_inr"] = round(bucket["rupees_invested_inr"] + cb, 2)
                bucket["net_rupee_pnl_inr"] = round(bucket["net_rupee_pnl_inr"] + pnl, 2)
                if pnl <= 0:
                    bucket["failed_trades"] += 1
                    bucket["rupees_lost_inr"] = round(bucket["rupees_lost_inr"] + abs(pnl), 2)
        for b in atk_map.values():
            inv = float(b["rupees_invested_inr"] or 0.0)
            b["roic_pct"] = round((float(b["net_rupee_pnl_inr"]) / inv) * 100.0, 2) if inv > 0 else 0.0
        atk_list = sorted(atk_map.values(), key=lambda x: (x["net_rupee_pnl_inr"], -x["rupees_lost_inr"]))

        return {
            "total_closed_trades": len(rows),
            "passed_trades": len(passed),
            "failed_trades": len(failed),
            "win_rate_pct": round((len(passed) / len(rows)) * 100.0, 2) if rows else None,
            "profit_factor": pf,
            "total_rupees_invested_inr": round(total_invested, 2),
            "total_realized_pnl_inr": round(total_pnl, 2),
            "rupee_roic_pct": rupee_roic,
            "pnl_per_1000_rupees_inr": pnl_per_1k,
            "total_rupees_at_risk_inr": round(total_at_risk, 2),
            "total_rupee_days": round(total_rupee_days, 2),
            "expectancy_per_trade_inr": round(total_pnl / len(rows), 2) if rows else None,
            "avg_win_pct": round(float(np.mean([r["return_pct"] for r in passed])), 2) if passed else None,
            "avg_loss_pct": round(float(np.mean([r["return_pct"] for r in failed])), 2) if failed else None,
            "attack_rupee_attribution": atk_list,
            "recent_lessons": rows[:25],
        }

    def learning_report(self, horizon: int = LEARNING_HORIZON) -> dict[str, Any]:
        conn = self._connect()
        try:
            dates = [
                r[0]
                for r in conn.execute(
                    "SELECT DISTINCT trading_date FROM outcomes WHERE horizon=? ORDER BY trading_date", (horizon,)
                )
            ]
            per_session: list[dict[str, Any]] = []
            ic_t_all: list[float] = []
            ic_f_all: list[float] = []
            pick_excess: list[float] = []
            shortlist_excess: list[float] = []
            universe_excess: list[float] = []
            for dt in dates:
                u = conn.execute(
                    "SELECT u.tech_conf, u.rank_key, o.excess FROM scan_universe u JOIN outcomes o "
                    "ON o.trading_date=u.trading_date AND o.sec_id=u.sec_id AND o.horizon=? WHERE u.trading_date=?",
                    (horizon, dt),
                ).fetchall()
                s = conn.execute(
                    "SELECT s.fund_conf, s.conviction, s.queued, o.excess FROM scan_shortlist s JOIN outcomes o "
                    "ON o.trading_date=s.trading_date AND o.sec_id=s.sec_id AND o.horizon=? "
                    "WHERE s.trading_date=? AND s.status='AGENTS_OK'",
                    (horizon, dt),
                ).fetchall()
                ic_t = None
                if len(u) >= MIN_UNIVERSE_OBS_PER_SESSION:
                    composite = [float(r["tech_conf"]) * 1000.0 + float(np.clip(r["rank_key"], -50, 50)) for r in u]
                    ic_t = spearman_ic(composite, [float(r["excess"]) for r in u])
                ic_f = None
                if len(s) >= MIN_SHORTLIST_OBS_PER_SESSION:
                    ic_f = spearman_ic([float(r["fund_conf"]) for r in s], [float(r["excess"]) for r in s])
                if ic_t is not None:
                    ic_t_all.append(ic_t)
                if ic_f is not None:
                    ic_f_all.append(ic_f)
                universe_excess.extend(float(r["excess"]) for r in u)
                shortlist_excess.extend(float(r["excess"]) for r in s)
                pick_excess.extend(float(r["excess"]) for r in s if int(r["queued"]) == 1)
                per_session.append(
                    {
                        "trading_date": dt,
                        "n_universe": len(u),
                        "n_shortlist": len(s),
                        "ic_tech": None if ic_t is None else round(ic_t, 4),
                        "ic_fund": None if ic_f is None else round(ic_f, 4),
                    }
                )

            def _mean_pct(v: list[float]) -> float | None:
                return round(float(np.mean(v)) * 100.0, 4) if v else None

            both = [r for r in per_session if r["ic_tech"] is not None and r["ic_fund"] is not None]
            return {
                "horizon_trading_days": horizon,
                "matured_sessions": len(per_session),
                "sessions_with_both_ics": len(both),
                "mean_ic_tech": None if not ic_t_all else round(float(np.mean(ic_t_all)), 4),
                "mean_ic_fund": None if not ic_f_all else round(float(np.mean(ic_f_all)), 4),
                "mean_excess_vs_niftybees_pct": {
                    "universe_all_eligible": _mean_pct(universe_excess),
                    "agent_shortlist": _mean_pct(shortlist_excess),
                    "queued_paper_picks": _mean_pct(pick_excess),
                },
                "observations": {
                    "universe": len(universe_excess),
                    "shortlist": len(shortlist_excess),
                    "queued_picks": len(pick_excess),
                },
                "closed_trades": self.closed_trade_summary(),
                "per_session": per_session[-80:],
            }
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # 5. Current & Adaptive Parameters (learns from EVERY passed/failed trade)
    # ------------------------------------------------------------------
    def current_params(self) -> dict[str, Any]:
        conn = self._connect()
        try:
            r = conn.execute("SELECT * FROM weight_history ORDER BY trading_date DESC LIMIT 1").fetchone()
        finally:
            conn.close()
        if r is None:
            return {
                "w_tech": PRIOR_W_TECH,
                "w_fund": PRIOR_W_FUND,
                "min_conviction": PRIOR_MIN_CONVICTION,
                "rsi_overbought_penalty": PRIOR_RSI_OVERBOUGHT_PENALTY,
                "sector_adjustments": {},
                "stop_tighten_mult": PRIOR_STOP_TIGHTEN_MULT,
                "attack_multipliers": {},
                "min_survival_score": DEFAULT_MIN_SURVIVAL_SCORE,
                "total_rupees_invested_inr": 0.0,
                "rupee_roic_pct": None,
                "tca_hurdle_adder_bps": float(self.btca_report().get("tca_hurdle_adder_bps") or 0.0),
                "status": "SESSION 1/80 · PRIOR_FIXED (awaiting first closed trades & matured outcomes)",
            }
        try:
            sec_adj = json.loads(r["sector_adjustments_json"] or "{}")
        except Exception:
            sec_adj = {}
        r_keys = r.keys() if hasattr(r, "keys") else ()
        try:
            atk_mult = json.loads(r["attack_multipliers_json"] or "{}") if "attack_multipliers_json" in r_keys else {}
        except Exception:
            atk_mult = {}
        stop_mult = (
            float(r["stop_tighten_mult"])
            if ("stop_tighten_mult" in r_keys and r["stop_tighten_mult"] is not None)
            else PRIOR_STOP_TIGHTEN_MULT
        )
        min_surv = (
            float(r["min_survival_score"])
            if ("min_survival_score" in r_keys and r["min_survival_score"] is not None)
            else DEFAULT_MIN_SURVIVAL_SCORE
        )
        tot_inv = float(r["total_rupees_invested_inr"] or 0.0) if "total_rupees_invested_inr" in r_keys else 0.0
        r_roic = float(r["rupee_roic_pct"]) if ("rupee_roic_pct" in r_keys and r["rupee_roic_pct"] is not None) else None
        return {
            "w_tech": float(r["w_tech"]),
            "w_fund": float(r["w_fund"]),
            "min_conviction": float(r["min_conviction"]),
            "rsi_overbought_penalty": float(r["rsi_overbought_penalty"]),
            "sector_adjustments": sec_adj if isinstance(sec_adj, dict) else {},
            "stop_tighten_mult": _clamp(stop_mult, STOP_MULT_MIN, STOP_MULT_MAX),
            "attack_multipliers": atk_mult if isinstance(atk_mult, dict) else {},
            "min_survival_score": _clamp(min_surv, SURVIVAL_FLOOR, SURVIVAL_CEIL),
            "total_rupees_invested_inr": round(tot_inv, 2),
            "rupee_roic_pct": r_roic,
            "tca_hurdle_adder_bps": float(self.btca_report().get("tca_hurdle_adder_bps") or 0.0),
            "status": str(r["status"]),
        }

    def current_weights(self) -> tuple[tuple[float, float], str]:
        p = self.current_params()
        return (p["w_tech"], p["w_fund"]), p["status"]

    def update_weights(self, trading_date: str) -> dict[str, Any]:
        """Update agent weights, conviction bar, RSI guard, sector bias, stops, and adversarial rules from every rupee invested."""
        report = self.learning_report()
        prev = self.current_params()
        w_t_prev = float(prev["w_tech"])
        min_conv_prev = float(prev["min_conviction"])
        rsi_pen_prev = float(prev["rsi_overbought_penalty"])
        stop_mult_prev = float(prev.get("stop_tighten_mult", PRIOR_STOP_TIGHTEN_MULT))
        min_surv_prev = float(prev.get("min_survival_score", DEFAULT_MIN_SURVIVAL_SCORE))
        prev_atk_mult = dict(prev.get("attack_multipliers") or {})

        conn = self._connect()
        try:
            lessons = [dict(r) for r in conn.execute("SELECT * FROM trade_lessons ORDER BY exit_date ASC").fetchall()]
            shortlist_obs = [
                dict(r)
                for r in conn.execute(
                    "SELECT s.sector, s.rsi_14, s.conviction, o.excess FROM scan_shortlist s "
                    "JOIN outcomes o ON o.trading_date=s.trading_date AND o.sec_id=s.sec_id AND o.horizon=? "
                    "WHERE s.status='AGENTS_OK'",
                    (LEARNING_HORIZON,),
                ).fetchall()
            ]
        finally:
            conn.close()

        n_trades = len(lessons)
        n_pass = sum(1 for r in lessons if r["outcome"] == "PASSED")
        n_fail = n_trades - n_pass
        n_ic = int(report["sessions_with_both_ics"])
        ic_t, ic_f = report["mean_ic_tech"], report["mean_ic_fund"]
        ct_sum = report.get("closed_trades") or {}
        total_rupees_invested = float(ct_sum.get("total_rupees_invested_inr") or 0.0)
        net_rupee_pnl = float(ct_sum.get("total_realized_pnl_inr") or 0.0)
        rupee_roic_pct = ct_sum.get("rupee_roic_pct")

        notes: list[str] = []

        # (A) Rupee-weighted Agent blend weight (w_tech vs w_fund) + matured IC
        trade_delta = 0.0
        if n_trades > 0:
            num, den, count_t = 0.0, 0.0, 0.0
            for r in lessons:
                if r["tech_conf"] is None or r["fund_conf"] is None:
                    continue
                rupee_w = max(100.0, float(r.get("cost_basis_inr") or 0.0))
                ret_clipped = _clamp(float(r["return_pct"]) / 10.0, -1.5, 1.5)
                diff_agent = float(r["tech_conf"]) - float(r["fund_conf"])
                num += rupee_w * ret_clipped * diff_agent
                den += rupee_w
                count_t += 1.0
            if den > 0 and count_t > 0:
                shrink_t = count_t / (count_t + SHRINKAGE_K_TRADES)
                trade_delta = _clamp((num / den) * 1.2 * shrink_t, -0.20, 0.20)
                notes.append(f"rupee_attr_delta={trade_delta:+.3f} (₹{den:,.0f} across n={int(count_t)})")

        ic_delta = 0.0
        if n_ic > 0 and ic_t is not None and ic_f is not None:
            shrink_s = n_ic / (n_ic + SHRINKAGE_K_SESSIONS)
            ic_delta = _clamp(1.8 * (ic_t - ic_f) * shrink_s, -0.20, 0.20)
            notes.append(f"ic_delta={ic_delta:+.3f} (ic_t={ic_t:+.3f}, ic_f={ic_f:+.3f}, n={n_ic})")

        if n_trades == 0 and n_ic == 0:
            new_t = w_t_prev
            new_min_conv = min_conv_prev
            new_rsi_pen = rsi_pen_prev
            new_stop_mult = stop_mult_prev
            new_min_surv = min_surv_prev
            new_atk_mult = prev_atk_mult
            sector_adj: dict[str, float] = {}
            status = "OBSERVING (0 closed trades, ₹0 closed basis) — adversarial priors active"
            notes.append("No closed trades or matured 5D outcomes yet; priors held.")
        else:
            combined_delta = (0.60 * trade_delta + 0.40 * ic_delta) if (n_trades > 0 and n_ic > 0) else (trade_delta or ic_delta)
            target_t = _clamp(PRIOR_W_TECH + combined_delta, W_TECH_MIN, W_TECH_MAX)
            step_t = _clamp(target_t - w_t_prev, -MAX_WEIGHT_STEP, MAX_WEIGHT_STEP)
            new_t = _clamp(w_t_prev + step_t, W_TECH_MIN, W_TECH_MAX)

            # (B) Adaptive min_conviction bar from closed trades & rupee ROIC
            target_conv = PRIOR_MIN_CONVICTION
            if n_trades > 0:
                low_c = [r for r in lessons if r["conviction"] is not None and float(r["conviction"]) < 0.70]
                hi_c = [r for r in lessons if r["conviction"] is not None and float(r["conviction"]) >= 0.70]
                low_wr = (sum(1 for r in low_c if r["outcome"] == "PASSED") / len(low_c)) if low_c else None
                hi_wr = (sum(1 for r in hi_c if r["outcome"] == "PASSED") / len(hi_c)) if hi_c else None
                overall_wr = n_pass / n_trades
                shrink_c = n_trades / (n_trades + SHRINKAGE_K_TRADES)
                if low_wr is not None and low_wr < 0.45 and (hi_wr is None or hi_wr > low_wr + 0.10):
                    target_conv = PRIOR_MIN_CONVICTION + 0.12 * shrink_c
                    notes.append(f"low_conv_wr={low_wr:.0%} -> tightening min_conviction")
                elif overall_wr >= 0.60 and (ct_sum.get("profit_factor") or 0.0) >= 1.4 and net_rupee_pnl > 0:
                    target_conv = PRIOR_MIN_CONVICTION - 0.04 * shrink_c
                    notes.append(f"strong_wr={overall_wr:.0%} -> deploying conviction bar")
                elif overall_wr < 0.45 or (total_rupees_invested > 0 and net_rupee_pnl < 0):
                    target_conv = PRIOR_MIN_CONVICTION + 0.10 * shrink_c
                    notes.append(f"net_rupee_loss (₹{net_rupee_pnl:+,.0f}) -> tightening min_conviction")
            target_conv = _clamp(target_conv, MIN_CONV_FLOOR, MIN_CONV_CEIL)
            new_min_conv = _clamp(min_conv_prev + _clamp(target_conv - min_conv_prev, -MAX_CONV_STEP, MAX_CONV_STEP), MIN_CONV_FLOOR, MIN_CONV_CEIL)

            # (C) Overbought RSI (>72) guard weighted by actual rupees deployed + shortlist outcomes
            ob_vals: list[tuple[float, float]] = [
                (_clamp(float(r["return_pct"]) / 100.0, -0.25, 0.25), max(100.0, float(r.get("cost_basis_inr") or 0.0)))
                for r in lessons
                if r["rsi_14"] is not None and float(r["rsi_14"]) > 72.0
            ] + [
                (_clamp(float(r["excess"]), -0.25, 0.25), 5000.0)
                for r in shortlist_obs
                if r["rsi_14"] is not None and float(r["rsi_14"]) > 72.0
            ]
            norm_vals: list[tuple[float, float]] = [
                (_clamp(float(r["return_pct"]) / 100.0, -0.25, 0.25), max(100.0, float(r.get("cost_basis_inr") or 0.0)))
                for r in lessons
                if r["rsi_14"] is not None and float(r["rsi_14"]) <= 72.0
            ] + [
                (_clamp(float(r["excess"]), -0.25, 0.25), 5000.0)
                for r in shortlist_obs
                if r["rsi_14"] is not None and float(r["rsi_14"]) <= 72.0
            ]
            target_rsi_pen = PRIOR_RSI_OVERBOUGHT_PENALTY
            if len(ob_vals) >= 2 and norm_vals:
                ob_w = sum(v * w for v, w in ob_vals) / max(1.0, sum(w for _, w in ob_vals))
                norm_w = sum(v * w for v, w in norm_vals) / max(1.0, sum(w for _, w in norm_vals))
                gap = norm_w - ob_w
                if gap > 0.005:
                    shrink_r = len(ob_vals) / (len(ob_vals) + 5.0)
                    target_rsi_pen = _clamp(gap * 2.5 * shrink_r, RSI_PEN_MIN, RSI_PEN_MAX)
                    notes.append(f"rsi>72 underperformed by {gap*100:.2f}% -> pen={target_rsi_pen:.3f}")
            new_rsi_pen = _clamp(rsi_pen_prev + _clamp(target_rsi_pen - rsi_pen_prev, -MAX_RSI_PEN_STEP, MAX_RSI_PEN_STEP), RSI_PEN_MIN, RSI_PEN_MAX)

            # (D) Per-sector rupee-weighted Bayesian edge calibration
            sec_buckets: dict[str, list[tuple[float, float]]] = {}
            for r in lessons:
                sec = str(r.get("sector") or "")
                if sec and sec != "UNKNOWN_SECTOR":
                    sec_buckets.setdefault(sec, []).append(
                        (_clamp(float(r["return_pct"]) / 100.0, -0.15, 0.15), max(100.0, float(r.get("cost_basis_inr") or 0.0)))
                    )
            for r in shortlist_obs:
                sec = str(r.get("sector") or "")
                if sec and sec != "UNKNOWN_SECTOR":
                    sec_buckets.setdefault(sec, []).append((_clamp(float(r["excess"]), -0.15, 0.15) * 0.5, 4000.0))
            sector_adj = {}
            for sec, vw_list in sec_buckets.items():
                if len(vw_list) >= 2:
                    mean_v = sum(v * w for v, w in vw_list) / max(1.0, sum(w for _, w in vw_list))
                    shrink = len(vw_list) / (len(vw_list) + 4.0)
                    adj = round(_clamp(mean_v * 0.8 * shrink, -SECTOR_ADJ_LIMIT, SECTOR_ADJ_LIMIT), 4)
                    if abs(adj) >= 0.003:
                        sector_adj[sec] = adj

            # (E) Stop-loss multiplier [0.80, 1.10] calibrated from MFE giveback & stop-out rate (step ±0.05)
            target_stop_mult = PRIOR_STOP_TIGHTEN_MULT
            if n_trades > 0:
                givebacks = [
                    float(r["mfe_giveback_pct"])
                    if r.get("mfe_giveback_pct") is not None
                    else max(0.0, float(r["mfe_pct"]) - float(r["return_pct"]))
                    for r in lessons
                    if r.get("mfe_giveback_pct") is not None or r.get("mfe_pct") is not None
                ]
                avg_giveback = float(np.mean(givebacks)) if givebacks else 0.0
                sl_fails = [r for r in lessons if r["outcome"] == "FAILED" and "STOP" in str(r.get("exit_reason") or "").upper()]
                sl_fail_rate = len(sl_fails) / n_trades
                shrink_m = n_trades / (n_trades + SHRINKAGE_K_TRADES)
                if avg_giveback >= 5.0 or sl_fail_rate >= 0.45:
                    tighten_delta = min(0.20, (avg_giveback * 0.018 + sl_fail_rate * 0.15) * shrink_m)
                    target_stop_mult = PRIOR_STOP_TIGHTEN_MULT - tighten_delta
                    notes.append(f"mfe_giveback={avg_giveback:.1f}% sl_fail={sl_fail_rate:.0%} -> tightening stop_mult")
                elif n_pass >= 4 and (n_pass / n_trades) >= 0.60 and avg_giveback < 3.0:
                    target_stop_mult = PRIOR_STOP_TIGHTEN_MULT + 0.08 * shrink_m
                    notes.append(f"clean_wins={n_pass}/{n_trades} low_giveback={avg_giveback:.1f}% -> easing stop_mult")
            target_stop_mult = _clamp(target_stop_mult, STOP_MULT_MIN, STOP_MULT_MAX)
            new_stop_mult = _clamp(
                stop_mult_prev + _clamp(target_stop_mult - stop_mult_prev, -MAX_STOP_MULT_STEP, MAX_STOP_MULT_STEP),
                STOP_MULT_MIN,
                STOP_MULT_MAX,
            )

            # (F) Per-rupee Adversarial Attack Escalation & Survival Bar Hardening
            new_atk_mult: dict[str, float] = {}
            for row in ct_sum.get("attack_rupee_attribution") or []:
                aid = str(row["attack_id"])
                inv = float(row.get("rupees_invested_inr") or 0.0)
                pnl = float(row.get("net_rupee_pnl_inr") or 0.0)
                t_cnt = int(row.get("trades") or 0)
                fail_cnt = int(row.get("failed_trades") or 0)
                if inv <= 0 or t_cnt == 0:
                    continue
                shrink_a = t_cnt / (t_cnt + 3.0)
                if pnl < 0:
                    # Objection was flagged, we invested rupees anyway, and LOST -> escalate damage multiplier permanently
                    loss_ratio = min(0.25, abs(pnl) / inv)
                    mult = _clamp(1.0 + shrink_a * (0.20 + loss_ratio * 4.5), 1.05, ATTACK_MULTIPLIER_MAX)
                    new_atk_mult[aid] = round(mult, 4)
                    notes.append(f"₹{abs(pnl):,.0f} lost under {aid} ({fail_cnt}/{t_cnt} failed) -> x{mult:.2f}")
                elif t_cnt >= 3 and fail_cnt == 0 and (pnl / inv) >= 0.04:
                    mult = _clamp(1.0 - 0.18 * shrink_a, ATTACK_MULTIPLIER_MIN, 1.0)
                    new_atk_mult[aid] = round(mult, 4)

            target_surv = DEFAULT_MIN_SURVIVAL_SCORE
            marginal = [r for r in lessons if r.get("survival_score") is not None and float(r["survival_score"]) < 62.0]
            if marginal:
                marg_pnl = sum(float(r.get("total_pnl_inr") or 0.0) for r in marginal)
                marg_inv = sum(float(r.get("cost_basis_inr") or 0.0) for r in marginal)
                if marg_inv > 0 and marg_pnl < 0:
                    shrink_ms = len(marginal) / (len(marginal) + 3.0)
                    target_surv = DEFAULT_MIN_SURVIVAL_SCORE + min(18.0, (abs(marg_pnl) / marg_inv) * 100.0 * 1.2 * shrink_ms + 3.0)
                    notes.append(f"marginal survival lost ₹{abs(marg_pnl):,.0f} -> raising min_survival_score")
            elif total_rupees_invested > 0 and net_rupee_pnl < 0:
                target_surv = DEFAULT_MIN_SURVIVAL_SCORE + 5.0 * (n_trades / (n_trades + 4.0))
            target_surv = _clamp(target_surv, SURVIVAL_FLOOR, SURVIVAL_CEIL)
            new_min_surv = round(
                _clamp(min_surv_prev + _clamp(target_surv - min_surv_prev, -MAX_SURVIVAL_STEP, MAX_SURVIVAL_STEP), SURVIVAL_FLOOR, SURVIVAL_CEIL),
                2,
            )

            roic_str = f" · ROIC {rupee_roic_pct:+.2f}% on ₹{total_rupees_invested:,.0f}" if rupee_roic_pct is not None else ""
            status = f"LEARNING ACTIVE ({n_pass} passed / {n_fail} failed · {n_ic} matured sessions{roic_str})"

        new_f = round(1.0 - new_t, 4)
        new_t = round(new_t, 4)
        new_min_conv = round(new_min_conv, 4)
        new_rsi_pen = round(new_rsi_pen, 4)
        new_stop_mult = round(new_stop_mult, 4)
        note_str = "; ".join(notes) if notes else "Weights updated cleanly."

        conn = self._connect()
        try:
            conn.execute(
                "INSERT OR REPLACE INTO weight_history "
                "(trading_date, w_tech, w_fund, min_conviction, rsi_overbought_penalty, sector_adjustments_json, stop_tighten_mult, "
                "attack_multipliers_json, min_survival_score, total_rupees_invested_inr, rupee_roic_pct, "
                "status, matured_sessions, closed_trades, passed_trades, failed_trades, ic_tech, ic_fund, note, recorded_at_utc) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    str(trading_date),
                    new_t,
                    new_f,
                    new_min_conv,
                    new_rsi_pen,
                    json.dumps(sector_adj, sort_keys=True),
                    new_stop_mult,
                    json.dumps(new_atk_mult, sort_keys=True),
                    new_min_surv,
                    round(total_rupees_invested, 2),
                    rupee_roic_pct,
                    status,
                    n_ic,
                    n_trades,
                    n_pass,
                    n_fail,
                    ic_t,
                    ic_f,
                    note_str,
                    _utc_now_iso(),
                ),
            )
            conn.commit()
        finally:
            conn.close()

        return {
            "w_tech": new_t,
            "w_fund": new_f,
            "min_conviction": new_min_conv,
            "rsi_overbought_penalty": new_rsi_pen,
            "sector_adjustments": sector_adj,
            "stop_tighten_mult": new_stop_mult,
            "attack_multipliers": new_atk_mult,
            "min_survival_score": new_min_surv,
            "total_rupees_invested_inr": round(total_rupees_invested, 2),
            "rupee_roic_pct": rupee_roic_pct,
            "status": status,
            "note": note_str,
            "learning": report,
        }

    # ------------------------------------------------------------------
    # 6. 80-Session Mission & Real-Money Readiness Gate
    # ------------------------------------------------------------------
    def build_mission_status(self, paper_snapshot: dict[str, Any] | None = None) -> dict[str, Any]:
        snap = paper_snapshot or {}
        rm = snap.get("risk_metrics") or {}
        pv = snap.get("prospective_validation") or {}
        ps = snap.get("portfolio_state") or {}
        recon = snap.get("reconciliation") or {}
        eng = snap.get("engine_state") or {}
        prosp_navs = snap.get("prospective_daily_navs") or snap.get("daily_nav_ledger") or []
        ct = self.closed_trade_summary()

        sessions_done = int(
            pv.get("completed_unseen_sessions")
            or rm.get("sample_sessions")
            or len(prosp_navs)
            or 0
        )
        init_cap = float(rm.get("initial_capital_inr") or ps.get("initial_capital_inr") or 100_000.0)
        last_row = prosp_navs[-1] if prosp_navs else {}
        nav = float(rm.get("current_nav_inr") or last_row.get("nav_inr") or ps.get("last_nav_inr") or init_cap)
        cum_ret_pct = float(
            rm.get("cumulative_return_pct")
            if rm.get("cumulative_return_pct") is not None
            else (last_row.get("cumulative_return_pct") if last_row.get("cumulative_return_pct") is not None else ((nav / max(1.0, init_cap)) - 1.0) * 100.0)
        )
        cum_pnl_inr = round(nav - init_cap, 2)
        max_dd_pct = float(
            rm.get("max_drawdown_pct")
            if rm.get("max_drawdown_pct") is not None
            else max([float(r.get("drawdown_pct") or 0.0) for r in prosp_navs], default=0.0)
        )
        recon_pass = bool(recon.get("passed", True))
        eng_state = str(eng.get("engine_state") or "RUNNING")

        alpha_vs_nb = None
        if prosp_navs:
            nb_nav = float(last_row.get("benchmark_niftybees_nav") or init_cap)
            nb_ret = ((nb_nav / max(1.0, init_cap)) - 1.0) * 100.0
            alpha_vs_nb = round(cum_ret_pct - nb_ret, 2)

        gates = {
            "sessions_80_completed": sessions_done >= MISSION_TARGET_SESSIONS,
            "positive_net_profit": cum_ret_pct > 0.0,
            "beats_niftybees_benchmark": (alpha_vs_nb is not None and alpha_vs_nb > 0.0),
            "profit_factor_ge_1_25": (ct["total_closed_trades"] >= 10 and (ct["profit_factor"] or 0.0) >= 1.25),
            "max_drawdown_le_12pct": max_dd_pct <= 12.0,
            "reconciliation_clean": recon_pass and eng_state != "SAFE_MODE",
        }
        passed_gates = sum(1 for v in gates.values() if v)
        return {
            "mission_name": "80-Session Maximum-Profit Orchestrator Mock Test",
            "target_sessions": MISSION_TARGET_SESSIONS,
            "completed_sessions": sessions_done,
            "remaining_sessions": max(0, MISSION_TARGET_SESSIONS - sessions_done),
            "progress_pct": round(min(100.0, (sessions_done / float(MISSION_TARGET_SESSIONS)) * 100.0), 1),
            "current_nav_inr": round(nav, 2),
            "cumulative_profit_inr": cum_pnl_inr,
            "cumulative_return_pct": round(cum_ret_pct, 2),
            "alpha_vs_niftybees_pct": alpha_vs_nb,
            "max_drawdown_pct": round(max_dd_pct, 2),
            "passed_trades": ct["passed_trades"],
            "failed_trades": ct["failed_trades"],
            "win_rate_pct": ct["win_rate_pct"],
            "profit_factor": ct["profit_factor"],
            "real_money_ready": all(gates.values()),
            "gates_passed": passed_gates,
            "gates_total": len(gates),
            "graduation_gates": gates,
            "live_trading_enabled": False,
        }

    def sync_btca_executions(self, paper_orders: list[dict[str, Any]]) -> int:
        """Persist Post-Trade BTCA Implementation Shortfall metrics for tactical equity executions."""
        if not paper_orders:
            return 0
        added = 0
        conn = self._connect()
        try:
            existing = {r[0] for r in conn.execute("SELECT order_id FROM btca_executions").fetchall()}
            for o in paper_orders:
                if str(o.get("instrument_class") or "").upper() != "DELIVERY":
                    continue
                oid = str(o.get("order_id") or "")
                if not oid or oid in existing:
                    continue
                arr_px = float(o.get("arrival_price") or o.get("expected_price") or 0.0)
                bar_open = float(o.get("bar_open") or o.get("expected_price") or 0.0)
                tca = compute_post_trade_tca(
                    order_id=oid,
                    symbol=str(o.get("symbol") or ""),
                    side=str(o.get("side") or "BUY"),
                    trading_date=str(o.get("trading_date") or ""),
                    arrival_price=arr_px,
                    bar_open=bar_open,
                    actual_exec_price=float(o.get("actual_exec_price") or 0.0),
                    req_qty=float(o.get("requested_qty") or 0.0),
                    filled_qty=float(o.get("filled_qty") or 0.0),
                    gross_value_inr=float(o.get("gross_value_inr") or 0.0),
                    total_cost_inr=float(o.get("total_cost_inr") or 0.0),
                    slippage_inr=float(o.get("slippage_inr") or 0.0),
                    algo_strategy=str(o.get("algo_strategy") or "NEXT_SESSION_OPEN"),
                    pre_trade_est_cost_bps=o.get("pre_trade_est_cost_bps"),
                )
                conn.execute(
                    "INSERT INTO btca_executions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        tca["order_id"],
                        tca["trading_date"],
                        tca["symbol"],
                        tca["side"],
                        tca["algo_strategy"],
                        tca["arrival_price"],
                        tca["bar_open"],
                        tca["actual_exec_price"],
                        float(o.get("requested_qty") or 0.0),
                        float(o.get("filled_qty") or 0.0),
                        tca["fill_rate_pct"],
                        tca["delay_cost_bps"],
                        tca["execution_slippage_bps"],
                        tca["statutory_cost_bps"],
                        tca["implementation_shortfall_bps"],
                        tca["slippage_surprise_bps"],
                        str(o.get("execution_status") or "UNKNOWN"),
                        _utc_now_iso(),
                    ),
                )
                added += 1
            conn.commit()
        finally:
            conn.close()
        return added

    def btca_report(self) -> dict[str, Any]:
        """Return Bloomberg BTCA-style Post-Trade Implementation Shortfall summary and bounded hurdle feedback."""
        conn = self._connect()
        try:
            rows = [dict(r) for r in conn.execute("SELECT * FROM btca_executions ORDER BY trading_date DESC, order_id DESC").fetchall()]
        finally:
            conn.close()
        summary = summarize_btca_and_feedback(rows)
        summary["recent_executions"] = rows[:15]
        return summary

    def export_learning_json(
        self,
        path: Path,
        *,
        paper_snapshot: dict[str, Any] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        params = self.current_params()
        conn = self._connect()
        try:
            history = [dict(r) for r in conn.execute("SELECT * FROM weight_history ORDER BY trading_date")]
        finally:
            conn.close()
        payload = {
            "generated_at_utc": _utc_now_iso(),
            "mission": self.build_mission_status(paper_snapshot),
            "weights": params,
            "weight_history": history[-80:],
            "learning": self.learning_report(),
            "btca": self.btca_report(),
            **(extra or {}),
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        return payload
