#!/usr/bin/env python3
"""Unified Backend API & Web UI Server for Finsistant Version 2 (All-in-One Platform).

Merges:
1. StockSight Version 2 (Portfolio & Watchlist Manager, 4-Pillar Fundamental Engine, Porter's 5 Forces, Moreshwar Levels, Decision Matrix)
2. Finsistant (Execution Terminal, Top Picks Trade Queue, Sell Warnings, Auto-Execution Rules, Polynomial Regression, Stocky AI)
3. Multi-Agent Trading Orchestrator (AgentRunner: TechBot_1, FundBot_1, ExecBot_1)
"""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import mimetypes
from pathlib import Path
import time
from typing import Any
from urllib.parse import urlparse

from finsistant_v2 import (
    ASSET_UNIVERSE,
    AgentRunner,
    ExecutionAgent,
    FinsistantSettings,
    FundamentalAnalyst,
    StockyAssistant,
    TechnicalAnalyst,
    get_asset_full_analysis,
    load_settings,
)

BASE_DIR = Path(__file__).resolve().parent
WEB_DIR = BASE_DIR / "web"

current_settings: FinsistantSettings = load_settings(BASE_DIR / ".env")
tech_agent = TechnicalAnalyst(name="TechBot_1")
fund_agent = FundamentalAnalyst(name="FundBot_1")
exec_agent = ExecutionAgent(name="ExecBot_1")
runner = AgentRunner(agents=[tech_agent, fund_agent, exec_agent], settings=current_settings)
stocky = StockyAssistant()

# Unified In-Memory State (Holdings + Watchlist + Trade Queue + Execution History + Auto Rules)
PORTFOLIO_STATE: dict[str, dict[str, Any]] = {}
TRADE_QUEUE: list[dict[str, Any]] = []
EXECUTED_LOG: list[dict[str, Any]] = []
AUTO_RULES: list[dict[str, Any]] = [
    {"rule_id": "rule-1", "name": "Strict Breakout Rules (Score > 80 & R:R > 1.8)", "min_score": 80, "active": True},
    {"rule_id": "rule-2", "name": "Moat Compounders (Fundamental >= 75 & Porter >= 75)", "min_score": 75, "active": True},
    {"rule_id": "rule-3", "name": "Auto-Exit Capital Traps (Fundamental <= 40)", "min_score": 40, "active": True},
]
CASH_BALANCE: float = 485000.0


def _initialize_platform_state() -> None:
    """Seed portfolio holdings, watchlist, and Finsistant pending trade queue."""
    PORTFOLIO_STATE.clear()
    TRADE_QUEUE.clear()

    for sym, meta in ASSET_UNIVERSE.items():
        qty = int(meta.get("default_qty", 0))
        avg = float(meta.get("default_avg", 0.0))
        snap = get_asset_full_analysis(sym, is_holding=(qty > 0))
        PORTFOLIO_STATE[sym] = {
            "symbol": sym,
            "name": snap["name"],
            "exchange": snap["exchange"],
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

    _scan_and_populate_queue()


def _scan_and_populate_queue() -> list[dict[str, Any]]:
    """Run TechSight + StockSight V2 + Multi-Agent scanner across the asset universe."""
    TRADE_QUEUE.clear()
    for sym in ASSET_UNIVERSE:
        p_item = PORTFOLIO_STATE.get(sym, {})
        is_held = int(p_item.get("qty", 0)) > 0
        snap = get_asset_full_analysis(sym, is_holding=is_held)
        run = runner.run_pipeline(
            symbol=sym,
            timeframe="1D",
            capital=CASH_BALANCE,
            active_trades=2,
            is_holding=is_held,
        )
        f_tot = snap["fundamental_score"]["total"]
        p_tot = snap["porters_score"]["total"]
        m = snap["moreshwar_levels"]
        dec = snap["decision_matrix"]

        if dec["action"] in ("EXIT", "REDUCE") and is_held:
            tx_type = "SELL"
        elif dec["action"] in ("BUY NOW", "ADD", "SIP ONLY") or f_tot >= 70:
            tx_type = "BUY"
        else:
            continue

        TRADE_QUEUE.append(
            {
                "id": f"pick-{sym.lower()}-{int(time.time()) % 10000}",
                "symbol": sym,
                "name": snap["name"],
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
                "qty": p_item.get("qty", 10) if tx_type == "SELL" else max(1, run.quantity),
                "horizon_days": "5D" if tx_type == "SELL" else "14D",
                "rationale": f"{run.messages[1].summary} {run.messages[2].summary}",
            }
        )
    return TRADE_QUEUE


def _build_full_state_payload(active_symbol: str = "RELIANCE") -> dict[str, Any]:
    global CASH_BALANCE
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

    is_held = PORTFOLIO_STATE.get(active_symbol, {}).get("qty", 0) > 0
    active_analysis = get_asset_full_analysis(active_symbol, is_holding=is_held)
    active_pipeline = runner.run_pipeline(
        symbol=active_symbol,
        timeframe="1D",
        capital=CASH_BALANCE,
        active_trades=len([t for t in EXECUTED_LOG if t.get("side") == "BUY"]),
        is_holding=is_held,
    )

    weighted_health = (
        round(
            sum(h["fundamental_score"] * h["current"] for h in holdings) / max(1.0, current_total),
            1,
        )
        if holdings
        else 78.0
    )

    return {
        "portfolio": {
            "stats": {
                "invested": round(invested_total, 2),
                "current": round(current_total, 2),
                "pnl": round(current_total - invested_total, 2),
                "pnl_pct": round(((current_total - invested_total) / max(1.0, invested_total)) * 100.0, 2),
                "balance": round(CASH_BALANCE, 2),
                "health_score": weighted_health,
            },
            "holdings": holdings,
            "watchlist": watchlist,
        },
        "queue": {
            "buys": [t for t in TRADE_QUEUE if t["status"] == "PENDING" and t["transaction_type"] == "BUY"],
            "sells": [t for t in TRADE_QUEUE if t["status"] == "PENDING" and t["transaction_type"] == "SELL"],
            "executed": EXECUTED_LOG[:15],
        },
        "rules": AUTO_RULES,
        "active_analysis": active_analysis,
        "active_pipeline": active_pipeline.model_dump(),
        "health": runner.health_check(),
        "config": runner.settings.model_dump(),
    }


_initialize_platform_state()


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
            self._send_json(_build_full_state_payload("RELIANCE"))
            return
        if path.startswith("/api/analyze/"):
            sym = path.split("/api/analyze/", 1)[1].upper().strip()
            self._send_json(_build_full_state_payload(sym))
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
            _scan_and_populate_queue()
            sym = str(body.get("symbol", "RELIANCE")).upper()
            self._send_json(_build_full_state_payload(sym))
            return

        if path == "/api/approve":
            trade_ids = set(body.get("trade_ids", []))
            for t in TRADE_QUEUE:
                if t["id"] in trade_ids and t["status"] == "PENDING":
                    t["status"] = "EXECUTED"
                    sym = t["symbol"]
                    qty = int(t["qty"])
                    price = float(t["entry"])
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
                            "order_id": f"kite-{int(time.time()*1000)%1000000:06d}",
                            "symbol": sym,
                            "side": t["transaction_type"],
                            "qty": qty,
                            "price": price,
                            "timestamp": time.strftime("%H:%M:%S"),
                        },
                    )
            self._send_json(_build_full_state_payload(str(body.get("active_symbol", "RELIANCE"))))
            return

        if path == "/api/portfolio/update":
            sym = str(body.get("symbol", "RELIANCE")).upper().strip()
            qty = max(0, int(body.get("qty", 0)))
            avg = float(body.get("avg_price", 0.0))
            if sym not in PORTFOLIO_STATE:
                snap = get_asset_full_analysis(sym, is_holding=(qty > 0))
                PORTFOLIO_STATE[sym] = {
                    "symbol": sym,
                    "name": snap["name"],
                    "exchange": snap["exchange"],
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
                snap = get_asset_full_analysis(sym, is_holding=(qty > 0))
                PORTFOLIO_STATE[sym]["decision"] = snap["decision_matrix"]["action"]
                PORTFOLIO_STATE[sym]["moreshwar"] = snap["moreshwar_levels"]
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
            res = stocky.process_query(msg, holdings=holdings, queue=TRADE_QUEUE)
            self._send_json(res)
            return

        if path == "/api/config":
            new_settings = FinsistantSettings(
                finsistant_api_key=str(body.get("finsistant_api_key", runner.settings.finsistant_api_key)),
                finsistant_environment=body.get("finsistant_environment", "production"),
                agent_mode=body.get("agent_mode", "paper_trading"),
                default_currency=str(body.get("default_currency", "INR")).upper().strip() or "INR",
                max_concurrent_trades=max(1, min(50, int(body.get("max_concurrent_trades", 5)))),
                execution_password=str(body.get("execution_password", runner.settings.execution_password)),
                kite_api_key=str(body.get("kite_api_key", runner.settings.kite_api_key)),
                gemini_api_key=runner.settings.gemini_api_key,
            )
            runner.settings = new_settings
            self._send_json(
                {
                    "status": "updated",
                    "config": new_settings.model_dump(),
                    "health": runner.health_check(),
                }
            )
            return

        self.send_error(404, "Not Found")


def run_server(host: str = "0.0.0.0", port: int = 8000) -> None:
    server = HTTPServer((host, port), FinsistantV2HTTPHandler)
    print(f"Finsistant Version 2 (All-in-One Platform) running at http://{host}:{port}")
    server.serve_forever()


if __name__ == "__main__":
    run_server()
