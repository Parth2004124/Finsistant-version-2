#!/usr/bin/env python3
"""Unified Backend API & Web UI Server for Finsistant Version 2 (Live Data Only).

Zero simulation policy:
- Uses live Yahoo Finance OHLCV candles & fundamentals for all stocks and charts.
- Uses Zerodha Kite Connect v3 when configured; otherwise starts with 0 holdings.
- Runs the 3-Agent Orchestrator (TechBot_1, FundBot_1, ExecBot_1) on live market data.
"""

from __future__ import annotations

from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import mimetypes
import os
from pathlib import Path
import time
from typing import Any
from urllib.parse import parse_qs, urlparse

from finsistant_v2 import (
    ASSET_UNIVERSE,
    AgentRunner,
    ExecutionAgent,
    FinsistantSettings,
    FundamentalAnalyst,
    StockyAssistant,
    TechnicalAnalyst,
    analyze_trade_with_gemini,
    apply_trade_recalibration,
    ask_gemini_stocky,
    auto_fetch_kite_access_token,
    build_audit_dashboard_view,
    build_data_integrity_report,
    critique_suggested_trades,
    exchange_request_token,
    fetch_benchmark_candles,
    fetch_kite_portfolio,
    fetch_live_market_snapshot,
    get_asset_full_analysis,
    get_kite_auth_status,
    load_audit_ledger,
    load_settings,
    record_broker_execution,
    record_scan_decisions,
    run_backtest_and_attribution,
)
from finsistant_v2.audit_ledger import DEFAULT_LEDGER_PATH
from finsistant_v2.kite_auth import write_daily_token_file

BASE_DIR = Path(__file__).resolve().parent
WEB_DIR = BASE_DIR / "web"
LEDGER_PATH: Path = DEFAULT_LEDGER_PATH

current_settings: FinsistantSettings = load_settings(BASE_DIR / ".env")
tech_agent = TechnicalAnalyst(name="TechBot_1")
fund_agent = FundamentalAnalyst(name="FundBot_1")
exec_agent = ExecutionAgent(name="ExecBot_1")
runner = AgentRunner(agents=[tech_agent, fund_agent, exec_agent], settings=current_settings)
stocky = StockyAssistant()

# Unified Live State (0 simulated holdings; live Kite holdings when connected)
PORTFOLIO_STATE: dict[str, dict[str, Any]] = {}
ANALYSIS_CACHE: dict[str, dict[str, Any]] = {}
FULL_HISTORY_CACHE: dict[str, dict[str, Any]] = {}
BENCHMARK_CACHE: dict[str, Any] = {}
PIPELINE_CACHE: dict[str, dict[str, Any]] = {}
GEMINI_CRITIQUES: dict[str, dict[str, Any]] = {}
TRADE_QUEUE: list[dict[str, Any]] = []
EXECUTED_LOG: list[dict[str, Any]] = []
KITE_META: dict[str, Any] = {"connected": False, "status": "Not initialized", "holdings": [], "fetched_at": None}
BACKTEST_CACHE: dict[str, Any] = {}
AUDIT_LEDGER_CACHE: dict[str, Any] = {}
PROVENANCE_CACHE: dict[str, Any] = {}
AUTO_RULES: list[dict[str, Any]] = [
    {"rule_id": "rule-1", "name": "Strict Breakout Rules (Score > 80 & R:R > 1.8)", "min_score": 80, "active": True},
    {"rule_id": "rule-2", "name": "Moat Compounders (Fundamental >= 75 & Porter >= 75)", "min_score": 75, "active": True},
    {"rule_id": "rule-3", "name": "Auto-Exit Capital Traps (Fundamental <= 40)", "min_score": 40, "active": True},
]
CASH_BALANCE: float = 0.0
KITE_STATUS: str = "0 holdings (Kite access token not configured)"
LAST_SCAN_AT: str = ""


def _load_extra_watchlist_symbols() -> list[str]:
    cfg_path = BASE_DIR / "watchlist_config.json"
    if not cfg_path.exists():
        return []
    try:
        data = json.loads(cfg_path.read_text(encoding="utf-8"))
        return [str(s).upper().strip() for s in (data.get("extra_symbols") or []) if str(s).strip()]
    except Exception:
        return []


def _refresh_analytics_and_audit(force_new_for_symbol: str | None = None, force_benchmark: bool = False) -> None:
    """Recompute backtest attribution, audit ledger persistence, and data provenance."""
    global BACKTEST_CACHE, AUDIT_LEDGER_CACHE, PROVENANCE_CACHE
    if force_benchmark or not BENCHMARK_CACHE.get("candles"):
        try:
            BENCHMARK_CACHE.clear()
            BENCHMARK_CACHE.update(fetch_benchmark_candles("^NSEI", force_refresh=force_benchmark))
        except Exception:
            pass

    AUDIT_LEDGER_CACHE = record_scan_decisions(
        assets=ANALYSIS_CACHE,
        pipelines=PIPELINE_CACHE,
        trade_queue=TRADE_QUEUE,
        critiques=GEMINI_CRITIQUES,
        ledger_path=LEDGER_PATH,
        force_new_for_symbol=force_new_for_symbol,
    )
    history_src = FULL_HISTORY_CACHE if FULL_HISTORY_CACHE else ANALYSIS_CACHE
    BACKTEST_CACHE = run_backtest_and_attribution(
        universe_history=history_src,
        benchmark_data=BENCHMARK_CACHE,
        executed_orders=EXECUTED_LOG,
        audit_records=AUDIT_LEDGER_CACHE.get("records"),
    )
    PROVENANCE_CACHE = build_data_integrity_report(
        assets=ANALYSIS_CACHE,
        critiques=GEMINI_CRITIQUES,
        kite_holdings=KITE_META.get("holdings"),
        kite_status_meta=KITE_META,
        executed_orders=EXECUTED_LOG,
        benchmark_meta=BENCHMARK_CACHE,
        ledger_meta=AUDIT_LEDGER_CACHE,
    )


def _initialize_platform_state(force_refresh: bool = False, force_token_refresh: bool = False) -> None:
    """Seed watchlist from live Yahoo Finance and load real Zerodha Kite holdings (0 if none)."""
    global CASH_BALANCE, KITE_STATUS
    PORTFOLIO_STATE.clear()
    ANALYSIS_CACHE.clear()
    FULL_HISTORY_CACHE.clear()
    PIPELINE_CACHE.clear()
    GEMINI_CRITIQUES.clear()
    TRADE_QUEUE.clear()

    kite_data = fetch_kite_portfolio(force_token_refresh=force_token_refresh)
    KITE_META.clear()
    KITE_META.update(kite_data)
    KITE_STATUS = kite_data["status"]
    CASH_BALANCE = float(kite_data.get("cash_balance", 0.0))
    kite_holdings_map = {h["symbol"]: h for h in kite_data.get("holdings", [])}

    extra_symbols = _load_extra_watchlist_symbols()
    symbols = list(dict.fromkeys(list(ASSET_UNIVERSE.keys()) + extra_symbols + list(kite_holdings_map.keys())))
    for sym in symbols:
        kh = kite_holdings_map.get(sym)
        qty = int(kh["qty"]) if kh else 0
        avg = float(kh["avg_price"]) if kh else 0.0
        snap = get_asset_full_analysis(sym, is_holding=(qty > 0), force_refresh=force_refresh)
        ANALYSIS_CACHE[sym] = snap
        try:
            raw_hist = fetch_live_market_snapshot(sym, force_refresh=False)
            FULL_HISTORY_CACHE[sym] = {
                **raw_hist,
                "fundamental_score": snap["fundamental_score"],
                "porters_score": snap["porters_score"],
            }
        except Exception:
            FULL_HISTORY_CACHE[sym] = snap
        PORTFOLIO_STATE[sym] = {
            "symbol": sym,
            "yahoo_symbol": snap.get("yahoo_symbol", sym),
            "name": snap["name"],
            "exchange": snap["exchange"],
            "currency": snap.get("currency", "INR"),
            "sector": snap["sector"],
            "asset_type": snap["asset_type"],
            "qty": qty,
            "avg_price": avg if qty > 0 else snap["price"],
            "ltp": snap["price"],
            "change_pct": snap["change_pct"],
            "fundamental_score": snap["fundamental_score"]["total"],
            "porters_score": snap["porters_score"]["total"],
            "decision": snap["decision_matrix"]["action"],
            "moreshwar": snap["moreshwar_levels"],
        }

    _scan_and_populate_queue(force_refresh=False)


def _scan_and_populate_queue(force_refresh: bool = True) -> list[dict[str, Any]]:
    """Run TechSight + StockSight V2 + 3-Agent Orchestrator + Gemini Critical Analyst scan."""
    global LAST_SCAN_AT
    TRADE_QUEUE.clear()
    sizing_capital = CASH_BALANCE if CASH_BALANCE > 0 else 100000.0

    for sym in list(PORTFOLIO_STATE.keys()):
        p_item = PORTFOLIO_STATE[sym]
        is_held = int(p_item.get("qty", 0)) > 0
        snap = get_asset_full_analysis(sym, is_holding=is_held, force_refresh=force_refresh)
        ANALYSIS_CACHE[sym] = snap
        try:
            raw_hist = fetch_live_market_snapshot(sym, force_refresh=False)
            FULL_HISTORY_CACHE[sym] = {
                **raw_hist,
                "fundamental_score": snap["fundamental_score"],
                "porters_score": snap["porters_score"],
            }
        except Exception:
            FULL_HISTORY_CACHE[sym] = snap

        p_item["ltp"] = snap["price"]
        p_item["change_pct"] = snap["change_pct"]
        p_item["fundamental_score"] = snap["fundamental_score"]["total"]
        p_item["porters_score"] = snap["porters_score"]["total"]
        p_item["decision"] = snap["decision_matrix"]["action"]
        p_item["moreshwar"] = snap["moreshwar_levels"]
        if not is_held:
            p_item["avg_price"] = snap["price"]

        run = runner.run_pipeline(
            symbol=sym,
            timeframe="1D",
            capital=sizing_capital,
            active_trades=len([t for t in EXECUTED_LOG if t.get("side") == "BUY"]),
            is_holding=is_held,
        )
        PIPELINE_CACHE[sym] = run.model_dump()

        f_tot = snap["fundamental_score"]["total"]
        p_tot = snap["porters_score"]["total"]
        m = snap["moreshwar_levels"]
        dec = snap["decision_matrix"]

        sip_qualified = (
            dec["action"] == "SIP ONLY"
            and f_tot >= 70
            and p_tot >= 70
            and float(snap.get("price") or 0.0) >= float(snap.get("sma_20") or 0.0)
            and snap.get("regression_curve") == "Upward Curve"
        )
        if dec["action"] in ("EXIT", "REDUCE") and is_held:
            tx_type = "SELL"
        elif (
            (dec["action"] in ("BUY NOW", "ADD") or sip_qualified)
            and run.final_action == "BUY"
            and float(m.get("rr_ratio") or 0.0) >= 1.5
            and run.conviction_score >= 0.65
        ):
            tx_type = "BUY"
        else:
            continue

        TRADE_QUEUE.append(
            {
                "id": f"pick-{sym.lower()}-{int(time.time()) % 10000}",
                "symbol": sym,
                "name": snap["name"],
                "currency": snap.get("currency", "INR"),
                "setup_type": snap["setup_type"],
                "transaction_type": tx_type,
                "status": "PENDING",
                "technical_score": int(round(run.messages[0].confidence * 100)),
                "fundamental_score": f_tot,
                "porters_score": p_tot,
                "confidence": round(run.conviction_score * 100, 1),
                "decision_badge": dec["action"],
                "entry": snap["price"],
                "target": m["target"],
                "stop_loss": m["stop_loss"],
                "rr_ratio": m["rr_ratio"],
                "qty": p_item.get("qty", 1) if tx_type == "SELL" else max(1, run.quantity),
                "horizon_days": "5D" if tx_type == "SELL" else "14D",
                "rationale": f"{run.messages[1].summary} {run.messages[2].summary}",
            }
        )

    gemini_key = os.getenv("GEMINI_API_KEY", "").strip() or runner.settings.gemini_api_key
    critiques = critique_suggested_trades(
        assets=ANALYSIS_CACHE,
        pipelines=PIPELINE_CACHE,
        trade_queue=TRADE_QUEUE,
        api_key=gemini_key,
    )
    GEMINI_CRITIQUES.clear()
    GEMINI_CRITIQUES.update(critiques)

    pick_by_sym = {t["symbol"]: t for t in TRADE_QUEUE}
    for sym, snap in ANALYSIS_CACHE.items():
        crit = GEMINI_CRITIQUES.get(sym)
        p_item = PORTFOLIO_STATE.get(sym, {})
        is_held = int(p_item.get("qty", 0)) > 0
        pick = pick_by_sym.get(sym)
        pipe = PIPELINE_CACHE.get(sym)

        recal = apply_trade_recalibration(
            symbol=sym,
            analysis=snap,
            pipeline=pipe,
            trade_pick=pick,
            critique=crit,
            is_holding=is_held,
        )
        if recal.get("applied") and p_item:
            p_item["decision"] = snap["decision_matrix"]["action"]
            p_item["moreshwar"] = snap["moreshwar_levels"]

        # If holding is flagged EXIT WARNING by Gemini critic and not yet queued as SELL, escalate it
        if (
            recal.get("applied")
            and is_held
            and not pick
            and (crit or {}).get("stance") == "EXIT WARNING"
        ):
            m = snap["moreshwar_levels"]
            TRADE_QUEUE.append(
                {
                    "id": f"pick-{sym.lower()}-exit-{int(time.time()) % 10000}",
                    "symbol": sym,
                    "name": snap["name"],
                    "currency": snap.get("currency", "INR"),
                    "setup_type": snap["setup_type"],
                    "transaction_type": "SELL",
                    "status": "PENDING",
                    "technical_score": int(round((pipe or {}).get("messages", [{}])[0].get("confidence", 0.5) * 100)),
                    "fundamental_score": snap["fundamental_score"]["total"],
                    "porters_score": snap["porters_score"]["total"],
                    "confidence": recal["after"]["conviction"],
                    "decision_badge": "EXIT NOW",
                    "entry": recal["after"]["entry"],
                    "target": recal["after"]["target"],
                    "stop_loss": recal["after"]["stop_loss"],
                    "rr_ratio": recal["after"]["rr_ratio"],
                    "qty": int(p_item.get("qty", 1)),
                    "horizon_days": "5D",
                    "rationale": crit.get("rationale", ""),
                    "recalibrated": True,
                    "recalibration_summary": recal.get("summary", ""),
                }
            )

    for pick in TRADE_QUEUE:
        c = GEMINI_CRITIQUES.get(pick["symbol"])
        if c and c.get("status") == "live" and c.get("rationale"):
            pick["gemini_stance"] = c.get("stance", "")
            pick["gemini_risk_rating"] = c.get("risk_rating", "")
            pick["gemini_rationale"] = c.get("rationale", "")

    LAST_SCAN_AT = datetime.now(timezone.utc).isoformat()
    _refresh_analytics_and_audit(force_benchmark=force_refresh)
    return TRADE_QUEUE


def _build_full_state_payload(active_symbol: str = "RELIANCE") -> dict[str, Any]:
    holdings = []
    watchlist = []
    invested_total = 0.0
    current_total = 0.0

    for sym, item in PORTFOLIO_STATE.items():
        qty = int(item["qty"])
        avg = float(item["avg_price"])
        ltp = float(item["ltp"])
        inv = round(qty * avg, 2)
        cur = round(qty * ltp, 2)
        pnl = round(cur - inv, 2)
        net_chg = round(((ltp - avg) / max(1.0, avg)) * 100.0, 2) if qty > 0 else item["change_pct"]

        enriched = {
            **item,
            "invested": inv,
            "current": cur,
            "pnl": pnl,
            "net_chg": net_chg,
        }
        if qty > 0:
            invested_total += inv
            current_total += cur
            holdings.append(enriched)
        else:
            watchlist.append(enriched)

    if active_symbol not in PORTFOLIO_STATE and PORTFOLIO_STATE:
        active_symbol = next(iter(PORTFOLIO_STATE.keys()))

    is_held = PORTFOLIO_STATE.get(active_symbol, {}).get("qty", 0) > 0
    active_analysis = ANALYSIS_CACHE.get(active_symbol) or get_asset_full_analysis(active_symbol, is_holding=is_held)
    ANALYSIS_CACHE[active_symbol] = active_analysis

    active_pipeline = PIPELINE_CACHE.get(active_symbol)
    if not active_pipeline:
        active_pipeline = runner.run_pipeline(
            symbol=active_symbol,
            timeframe="1D",
            capital=CASH_BALANCE if CASH_BALANCE > 0 else 100000.0,
            active_trades=len([t for t in EXECUTED_LOG if t.get("side") == "BUY"]),
            is_holding=is_held,
        ).model_dump()
        PIPELINE_CACHE[active_symbol] = active_pipeline

    weighted_health = (
        round(
            sum(h["fundamental_score"] * h["current"] for h in holdings) / max(1.0, current_total),
            1,
        )
        if holdings
        else None
    )

    def _mask_val(k: str, v: Any) -> Any:
        is_secret = any(word in k for word in ("key", "token", "password", "secret", "totp"))
        return "***" if is_secret and v else v

    active_critique = GEMINI_CRITIQUES.get(active_symbol)
    if not active_critique:
        active_pick = next((t for t in TRADE_QUEUE if t.get("symbol") == active_symbol), None)
        active_critique = analyze_trade_with_gemini(
            active_symbol,
            active_analysis,
            pipeline=active_pipeline,
            trade_pick=active_pick,
            api_key=os.getenv("GEMINI_API_KEY", "").strip() or runner.settings.gemini_api_key,
        )
        apply_trade_recalibration(
            symbol=active_symbol,
            analysis=active_analysis,
            pipeline=active_pipeline,
            trade_pick=active_pick,
            critique=active_critique,
            is_holding=is_held,
        )
        GEMINI_CRITIQUES[active_symbol] = active_critique

    live_critiques = [c for c in GEMINI_CRITIQUES.values() if c.get("status") == "live"]
    recal_count = len([c for c in live_critiques if (c.get("recalibration") or {}).get("applied")])
    first_err = next((c.get("error") for c in GEMINI_CRITIQUES.values() if c.get("error")), None)
    gemini_status = (
        f"Gemini Live ({live_critiques[0].get('model', 'gemini-2.5-flash')} · {len(live_critiques)}/{len(GEMINI_CRITIQUES)} analyzed · {recal_count} recalibrated)"
        if live_critiques
        else f"Gemini Unavailable ({first_err or 'GEMINI_API_KEY not configured'})"
    )

    if not PROVENANCE_CACHE or not BACKTEST_CACHE or not AUDIT_LEDGER_CACHE:
        _refresh_analytics_and_audit()

    return {
        "data_mode": "LIVE_ONLY",
        "data_source": "Yahoo Finance Live Market Data + 3-Agent Orchestrator",
        "kite_status": KITE_STATUS,
        "kite_auth": get_kite_auth_status(),
        "gemini_status": gemini_status,
        "generated_at": LAST_SCAN_AT or datetime.now(timezone.utc).isoformat(),
        "portfolio": {
            "stats": {
                "invested": round(invested_total, 2),
                "current": round(current_total, 2),
                "pnl": round(current_total - invested_total, 2),
                "pnl_pct": round(((current_total - invested_total) / max(1.0, invested_total)) * 100.0, 2) if invested_total > 0 else 0.0,
                "balance": round(CASH_BALANCE, 2),
                "health_score": weighted_health,
            },
            "holdings": holdings,
            "watchlist": watchlist,
        },
        "queue": {
            "buys": [t for t in TRADE_QUEUE if t["status"] == "PENDING" and t["transaction_type"] == "BUY"],
            "sells": [t for t in TRADE_QUEUE if t["status"] == "PENDING" and t["transaction_type"] == "SELL"],
            "deferred": [t for t in TRADE_QUEUE if t["status"] == "DEFERRED"],
            "executed": EXECUTED_LOG[:15],
        },
        "assets": ANALYSIS_CACHE,
        "pipelines": PIPELINE_CACHE,
        "gemini_critiques": GEMINI_CRITIQUES,
        "provenance": PROVENANCE_CACHE,
        "backtest": BACKTEST_CACHE,
        "audit_trail": build_audit_dashboard_view(AUDIT_LEDGER_CACHE or load_audit_ledger(LEDGER_PATH)),
        "rules": AUTO_RULES,
        "active_analysis": active_analysis,
        "active_pipeline": active_pipeline,
        "active_critique": active_critique,
        "health": runner.health_check(),
        "config": {
            k: _mask_val(k, v)
            for k, v in runner.settings.model_dump().items()
        },
    }


class FinsistantV2HTTPHandler(BaseHTTPRequestHandler):
    """HTTP & REST API Handler for Finsistant Version 2."""

    def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        raw = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(raw)

    def _serve_file(self, file_path: Path) -> None:
        if not file_path.exists() or not file_path.is_file():
            self.send_error(404, "File not found")
            return
        mime, _ = mimetypes.guess_type(str(file_path))
        data = file_path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mime or "application/octet-stream")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/state":
            try:
                if not PORTFOLIO_STATE:
                    _initialize_platform_state(force_refresh=False)
                self._send_json(_build_full_state_payload("RELIANCE"))
            except Exception as exc:
                self._send_json({"error": str(exc), "data_mode": "LIVE_ONLY"}, status=502)
            return
        if path == "/api/provenance":
            if not PORTFOLIO_STATE:
                _initialize_platform_state(force_refresh=False)
            if not PROVENANCE_CACHE:
                _refresh_analytics_and_audit()
            self._send_json(PROVENANCE_CACHE)
            return
        if path == "/api/backtest":
            if not PORTFOLIO_STATE:
                _initialize_platform_state(force_refresh=False)
            if not BACKTEST_CACHE:
                _refresh_analytics_and_audit()
            self._send_json(BACKTEST_CACHE)
            return
        if path == "/api/audit":
            ledger = AUDIT_LEDGER_CACHE or load_audit_ledger(LEDGER_PATH)
            self._send_json(build_audit_dashboard_view(ledger))
            return
        if path == "/api/kite/status":
            self._send_json({"kite_status": KITE_STATUS, "kite_auth": get_kite_auth_status()})
            return
        if path == "/api/kite/callback":
            params = parse_qs(parsed.query)
            req_token = (params.get("request_token") or [""])[0].strip()
            api_key = os.getenv("KITE_API_KEY", "").strip()
            api_secret = os.getenv("KITE_API_SECRET", "").strip()
            if not req_token:
                self._send_json({"error": "Missing request_token query parameter"}, status=400)
                return
            try:
                exchange_request_token(api_key, api_secret, req_token)
                _initialize_platform_state(force_refresh=False)
                self._send_json(
                    {
                        "status": "success",
                        "message": "Kite request_token exchanged and daily access_token activated.",
                        "kite_status": KITE_STATUS,
                        "kite_auth": get_kite_auth_status(),
                    }
                )
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=400)
            return
        if path.startswith("/api/analyze/"):
            sym = path.split("/api/analyze/", 1)[1].upper().strip()
            try:
                self._send_json(_build_full_state_payload(sym))
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=400)
            return
        if path == "/api/health":
            self._send_json(runner.health_check())
            return

        if path in ("/", "/index.html"):
            self._serve_file(WEB_DIR / "index.html")
            return
        if path.startswith("/assets/"):
            rel = path.split("/assets/", 1)[1]
            self._serve_file(WEB_DIR / rel)
            return

        candidate = WEB_DIR / path.lstrip("/")
        if candidate.exists():
            self._serve_file(candidate)
            return
        self.send_error(404, "Not Found")

    def do_POST(self) -> None:  # noqa: N802
        global CASH_BALANCE
        parsed = urlparse(self.path)
        path = parsed.path
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length).decode("utf-8")) if length > 0 else {}

        if path == "/api/scan":
            try:
                if not PORTFOLIO_STATE:
                    _initialize_platform_state(force_refresh=True)
                else:
                    _scan_and_populate_queue(force_refresh=True)
                sym = str(body.get("symbol", "RELIANCE")).upper()
                self._send_json(_build_full_state_payload(sym))
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=502)
            return

        if path == "/api/recalibrate":
            sym = str(body.get("symbol", "RELIANCE")).upper().strip()
            focus_note = str(body.get("focus_note", "")).strip()
            try:
                if not PORTFOLIO_STATE:
                    _initialize_platform_state(force_refresh=False)
                snap = ANALYSIS_CACHE.get(sym)
                if not snap:
                    self._send_json({"error": f"Unknown symbol: {sym}"}, status=404)
                    return
                pipe = PIPELINE_CACHE.get(sym)
                pick = next((t for t in TRADE_QUEUE if t.get("symbol") == sym), None)
                p_item = PORTFOLIO_STATE.get(sym, {})
                is_held = int(p_item.get("qty", 0)) > 0
                gemini_key = os.getenv("GEMINI_API_KEY", "").strip() or runner.settings.gemini_api_key
                crit = analyze_trade_with_gemini(
                    sym, snap, pipeline=pipe, trade_pick=pick, api_key=gemini_key, focus_note=focus_note
                )
                GEMINI_CRITIQUES[sym] = crit
                recal = apply_trade_recalibration(
                    symbol=sym, analysis=snap, pipeline=pipe, trade_pick=pick, critique=crit, is_holding=is_held
                )
                if recal.get("applied") and p_item:
                    p_item["decision"] = snap["decision_matrix"]["action"]
                    p_item["moreshwar"] = snap["moreshwar_levels"]
                if pick and crit.get("status") == "live":
                    pick["gemini_stance"] = crit.get("stance", "")
                    pick["gemini_risk_rating"] = crit.get("risk_rating", "")
                    pick["gemini_rationale"] = crit.get("rationale", "")
                _refresh_analytics_and_audit(force_new_for_symbol=sym)
                self._send_json(_build_full_state_payload(sym))
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=502)
            return

        if path == "/api/approve":
            trade_ids = set(body.get("trade_ids", []))
            executed_any = False
            for t in TRADE_QUEUE:
                if t["id"] in trade_ids and t["status"] == "PENDING":
                    t["status"] = "EXECUTED"
                    sym = t["symbol"]
                    qty = int(t["qty"])
                    price = float(t["entry"])
                    order_id = f"kite-{int(time.time()*1000)%1000000:06d}"
                    if t["transaction_type"] == "BUY":
                        cost = qty * price
                        CASH_BALANCE = max(0.0, CASH_BALANCE - cost)
                        p = PORTFOLIO_STATE.get(sym)
                        if p:
                            old_q = int(p["qty"])
                            old_avg = float(p["avg_price"])
                            new_q = old_q + qty
                            p["avg_price"] = round(((old_q * old_avg) + cost) / max(1, new_q), 2)
                            p["qty"] = new_q
                    else:
                        p = PORTFOLIO_STATE.get(sym)
                        if p and int(p["qty"]) > 0:
                            CASH_BALANCE += int(p["qty"]) * price
                            p["qty"] = 0
                    EXECUTED_LOG.insert(
                        0,
                        {
                            "order_id": order_id,
                            "symbol": sym,
                            "side": t["transaction_type"],
                            "qty": qty,
                            "price": price,
                            "timestamp": time.strftime("%H:%M:%S"),
                        },
                    )
                    record_broker_execution(sym, t["transaction_type"], qty, price, order_id, ledger_path=LEDGER_PATH)
                    executed_any = True
            if executed_any:
                _refresh_analytics_and_audit()
            self._send_json(_build_full_state_payload(str(body.get("active_symbol", "RELIANCE"))))
            return

        if path == "/api/portfolio/update":
            sym = str(body.get("symbol", "RELIANCE")).upper().strip()
            qty = max(0, int(body.get("qty", 0)))
            avg = float(body.get("avg_price", 0.0))
            try:
                snap = get_asset_full_analysis(sym, is_holding=(qty > 0))
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=400)
                return
            ANALYSIS_CACHE[sym] = snap
            if sym not in PORTFOLIO_STATE:
                PORTFOLIO_STATE[sym] = {
                    "symbol": sym,
                    "yahoo_symbol": snap.get("yahoo_symbol", sym),
                    "name": snap["name"],
                    "exchange": snap["exchange"],
                    "currency": snap.get("currency", "INR"),
                    "sector": snap["sector"],
                    "asset_type": snap["asset_type"],
                    "qty": qty,
                    "avg_price": avg if avg > 0 else snap["price"],
                    "ltp": snap["price"],
                    "change_pct": snap["change_pct"],
                    "fundamental_score": snap["fundamental_score"]["total"],
                    "porters_score": snap["porters_score"]["total"],
                    "decision": snap["decision_matrix"]["action"],
                    "moreshwar": snap["moreshwar_levels"],
                }
            else:
                PORTFOLIO_STATE[sym]["qty"] = qty
                if avg > 0:
                    PORTFOLIO_STATE[sym]["avg_price"] = avg
                PORTFOLIO_STATE[sym]["decision"] = snap["decision_matrix"]["action"]
                PORTFOLIO_STATE[sym]["moreshwar"] = snap["moreshwar_levels"]
            _scan_and_populate_queue(force_refresh=False)
            self._send_json(_build_full_state_payload(sym))
            return

        if path == "/api/rules/toggle":
            rule_id = str(body.get("rule_id", ""))
            for r in AUTO_RULES:
                if r["rule_id"] == rule_id:
                    r["active"] = not r["active"]
            self._send_json({"rules": AUTO_RULES})
            return

        if path == "/api/chat":
            msg = str(body.get("message", ""))
            holdings = [v for v in PORTFOLIO_STATE.values() if int(v["qty"]) > 0]
            try:
                res = stocky.process_query(msg, holdings=holdings, queue=TRADE_QUEUE)
                gemini_key = os.getenv("GEMINI_API_KEY", "").strip() or runner.settings.gemini_api_key
                is_default_help = "Try asking:" in str(res.get("reply", ""))
                if gemini_key and (is_default_help or any(k in msg.lower() for k in ("why", "risk", "critic", "should", "rationale", "gemini"))):
                    ctx_lines = [
                        f"Kite Status: {KITE_STATUS}",
                        "Top Live Assets & Critiques:",
                    ]
                    for sym, a in list(ANALYSIS_CACHE.items())[:10]:
                        c = GEMINI_CRITIQUES.get(sym, {})
                        ctx_lines.append(
                            f"- {sym}: Spot {a.get('price')}, Fund {a.get('fundamental_score', {}).get('total')}/100, "
                            f"Porter {a.get('porters_score', {}).get('total')}/100, Matrix {a.get('decision_matrix', {}).get('action')}, "
                            f"Gemini Stance={c.get('stance', 'N/A')}"
                        )
                    try:
                        gem_reply = ask_gemini_stocky(msg, "\n".join(ctx_lines), api_key=gemini_key)
                        if gem_reply:
                            res["reply"] = gem_reply if is_default_help else f"{res['reply']}\n\n🧠 Gemini Critical Note:\n{gem_reply}"
                    except Exception:
                        pass
                self._send_json(res)
            except Exception as exc:
                self._send_json({"reply": f"❌ Live lookup error: {exc}"})
            return

        if path == "/api/kite/auto-login":
            try:
                sess = auto_fetch_kite_access_token()
                _initialize_platform_state(force_refresh=False)
                payload = _build_full_state_payload(str(body.get("active_symbol", "RELIANCE")).upper())
                payload["kite_session"] = {
                    "status": "success",
                    "source": sess.get("source", "auto_totp_login"),
                    "user_id": sess.get("user_id", ""),
                    "login_time": sess.get("login_time", ""),
                }
                self._send_json(payload)
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=400)
            return

        if path == "/api/kite/exchange-token":
            req_token = str(body.get("request_token", "")).strip()
            api_key = str(body.get("api_key", "")).strip() or os.getenv("KITE_API_KEY", "").strip()
            api_secret = str(body.get("api_secret", "")).strip() or os.getenv("KITE_API_SECRET", "").strip()
            if api_key and api_key != "***":
                os.environ["KITE_API_KEY"] = api_key
            if api_secret and api_secret != "***":
                os.environ["KITE_API_SECRET"] = api_secret
            try:
                sess = exchange_request_token(api_key, api_secret, req_token)
                _initialize_platform_state(force_refresh=False)
                payload = _build_full_state_payload(str(body.get("active_symbol", "RELIANCE")).upper())
                payload["kite_session"] = {
                    "status": "success",
                    "source": "request_token_exchange",
                    "user_id": sess.get("user_id", ""),
                }
                self._send_json(payload)
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=400)
            return

        if path == "/api/update_token":
            expected_secret = os.getenv("EXECUTION_PASSWORD", "") or os.getenv("FINSISTANT_API_KEY", "")
            provided_secret = str(body.get("secret", "")).strip()
            if expected_secret and provided_secret != expected_secret:
                self._send_json({"status": "error", "detail": "Unauthorized"}, status=401)
                return
            new_token = str(body.get("token", "")).strip()
            if not new_token:
                self._send_json({"status": "error", "detail": "No token provided"}, status=400)
                return
            try:
                write_daily_token_file(new_token)
                _initialize_platform_state(force_refresh=False)
                self._send_json(
                    {
                        "status": "success",
                        "message": "Token updated successfully",
                        "kite_status": KITE_STATUS,
                        "kite_auth": get_kite_auth_status(),
                    }
                )
            except Exception as exc:
                self._send_json({"status": "error", "detail": str(exc)}, status=400)
            return

        if path == "/api/config":
            def _pick_secret(field: str, env_key: str, current_val: str) -> str:
                raw = str(body.get(field, "")).strip()
                if raw and raw != "***":
                    os.environ[env_key] = raw
                    return raw
                return current_val

            kite_key = _pick_secret("kite_api_key", "KITE_API_KEY", runner.settings.kite_api_key)
            kite_sec = _pick_secret("kite_api_secret", "KITE_API_SECRET", runner.settings.kite_api_secret)
            kite_uid = _pick_secret("kite_user_id", "KITE_USER_ID", runner.settings.kite_user_id)
            kite_pwd = _pick_secret("kite_password", "KITE_PASSWORD", runner.settings.kite_password)
            kite_totp = _pick_secret("kite_totp_secret", "KITE_TOTP_SECRET", runner.settings.kite_totp_secret)
            kite_tok = _pick_secret("kite_access_token", "KITE_ACCESS_TOKEN", runner.settings.kite_access_token)
            gemini_key = _pick_secret("gemini_api_key", "GEMINI_API_KEY", runner.settings.gemini_api_key)

            new_settings = FinsistantSettings(
                finsistant_api_key=str(body.get("finsistant_api_key", runner.settings.finsistant_api_key)),
                finsistant_environment=body.get("finsistant_environment", "production"),
                agent_mode=body.get("agent_mode", "paper_trading"),
                default_currency=str(body.get("default_currency", "INR")).upper().strip() or "INR",
                max_concurrent_trades=max(1, min(50, int(body.get("max_concurrent_trades", 5)))),
                execution_password=str(body.get("execution_password", runner.settings.execution_password)),
                kite_api_key=kite_key,
                kite_api_secret=kite_sec,
                kite_user_id=kite_uid,
                kite_password=kite_pwd,
                kite_totp_secret=kite_totp,
                kite_access_token=kite_tok,
                gemini_api_key=gemini_key,
            )
            runner.settings = new_settings
            _initialize_platform_state(force_refresh=False)
            self._send_json(
                {
                    "status": "updated",
                    "kite_status": KITE_STATUS,
                    "kite_auth": get_kite_auth_status(),
                    "config": {
                        k: ("***" if any(w in k for w in ("key", "token", "password", "secret", "totp")) and v else v)
                        for k, v in new_settings.model_dump().items()
                    },
                    "health": runner.health_check(),
                }
            )
            return

        self.send_error(404, "Not Found")


def run_server(host: str = "0.0.0.0", port: int = 8000) -> None:
    _initialize_platform_state(force_refresh=True)
    server = HTTPServer((host, port), FinsistantV2HTTPHandler)
    print(f"Finsistant Version 2 (Live Data Only) running at http://{host}:{port}")
    server.serve_forever()


if __name__ == "__main__":
    run_server()
