"""Explainability & Permanent Decision Audit Trail for Finsistant Version 2.

Preserves an immutable, git-committable JSON ledger (`data/decision_audit_ledger.json`)
recording every stock recommendation's complete lifecycle:
  ORIGINAL DECISION -> CRITICAL REVIEW -> FINAL DECISION -> ACTUAL OUTCOME

Survives GitHub Actions 15-minute cloud scans and local executions with zero
dependency on the user's laptop.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

DEFAULT_LEDGER_PATH = Path(__file__).resolve().parent.parent / "data" / "decision_audit_ledger.json"

ENGINE_VERSIONS: dict[str, str] = {
    "platform_version": "Finsistant-V2.3-Audit",
    "scoring_model": "StockSight-V2.1",
    "quant_model": "PolyReg2-RSI14-MACD-v2",
    "moreshwar_model": "Moreshwar-Dynamic-V2",
    "orchestrator_model": "AgentRunner-3Bot-v2",
}


def load_audit_ledger(ledger_path: Path = DEFAULT_LEDGER_PATH) -> dict[str, Any]:
    if not ledger_path.exists():
        return {"version": 1, "updated_at": None, "records": []}
    try:
        data = json.loads(ledger_path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and isinstance(data.get("records"), list):
            return data
    except Exception:
        pass
    return {"version": 1, "updated_at": None, "records": []}


def save_audit_ledger(ledger: dict[str, Any], ledger_path: Path = DEFAULT_LEDGER_PATH) -> Path:
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    ledger["updated_at"] = datetime.now(timezone.utc).isoformat()
    ledger["total_records"] = len(ledger.get("records") or [])
    ledger_path.write_text(json.dumps(ledger, indent=2), encoding="utf-8")
    return ledger_path


def _pct_delta(before: float, after: float) -> float:
    if not before:
        return 0.0
    return round(((after - before) / abs(before)) * 100.0, 2)


def _extract_reasons(critique: dict[str, Any] | None, recal: dict[str, Any] | None) -> list[str]:
    reasons: list[str] = []
    if not critique or critique.get("status") != "live":
        return ["No live Gemini CRO recalibration applied (raw quantitative levels retained)."]
    cal = critique.get("calibration") or {}
    if cal.get("entry_discount_pct", 0) > 0:
        reasons.append(f"Applied {cal['entry_discount_pct']}% pullback discount to avoid chasing extended spot.")
    if cal.get("stop_tighten_pct", 0) > 0:
        reasons.append(f"Tightened stop-loss distance by {cal['stop_tighten_pct']}% to constrain downside MAE.")
    if cal.get("target_haircut_pct", 0) > 0:
        reasons.append(f"Trimmed target upside by {cal['target_haircut_pct']}% for conservative R:R alignment.")
    mult = float(cal.get("size_multiplier", 1.0))
    if mult == 0.0:
        reasons.append("Zero-sized (DEFERRED) due to CRO critical objections / unfavorable risk.")
    elif mult != 1.0:
        reasons.append(f"Adjusted position size multiplier to {mult:.2f}x based on conviction & risk.")
    if critique.get("critical_risks"):
        reasons.append(f"Objection: {critique['critical_risks']}")
    return reasons or [recal.get("summary") or "Reviewed by Gemini CRO."]


def _build_record(
    symbol: str,
    analysis: dict[str, Any],
    pipeline: dict[str, Any] | None,
    pick: dict[str, Any] | None,
    critique: dict[str, Any] | None,
    force_new_stamp: str = "",
) -> dict[str, Any]:
    now_iso = datetime.now(timezone.utc).isoformat()
    mkt_date = str(analysis.get("last_candle_date") or now_iso[:10])
    mkt_ts = str(analysis.get("market_timestamp") or f"{mkt_date}T10:00:00+00:00")
    ret_ts = str(analysis.get("fetched_at") or now_iso)

    raw_m = analysis.get("raw_moreshwar_levels") or analysis.get("moreshwar_levels") or {}
    raw_d = analysis.get("raw_decision_matrix") or analysis.get("decision_matrix") or {}
    cur_m = analysis.get("moreshwar_levels") or {}
    cur_d = analysis.get("decision_matrix") or {}
    recal = (critique or {}).get("recalibration") or {}

    before = recal.get("before") or (pick or {}).get("original_plan") or {
        "action": str(raw_d.get("action") or "MONITOR"),
        "entry": float(raw_m.get("entry_price") or analysis.get("price") or 0.0),
        "target": float(raw_m.get("target") or 0.0),
        "stop_loss": float(raw_m.get("stop_loss") or 0.0),
        "rr_ratio": float(raw_m.get("rr_ratio") or 1.0),
        "qty": int((pipeline or {}).get("quantity") or (pick or {}).get("qty") or 0),
        "conviction": round(float((pipeline or {}).get("conviction_score") or 0.5) * 100.0, 1),
    }
    after = recal.get("after") or {
        "action": str((pick or {}).get("decision_badge") or cur_d.get("action") or before["action"]),
        "entry": float((pick or {}).get("entry") or cur_m.get("entry_price") or before["entry"]),
        "target": float((pick or {}).get("target") or cur_m.get("target") or before["target"]),
        "stop_loss": float((pick or {}).get("stop_loss") or cur_m.get("stop_loss") or before["stop_loss"]),
        "rr_ratio": float((pick or {}).get("rr_ratio") or cur_m.get("rr_ratio") or before["rr_ratio"]),
        "qty": int((pick or {}).get("qty") if pick is not None else before["qty"]),
        "conviction": round(float((pick or {}).get("confidence") or before["conviction"]), 1),
        "status": str((pick or {}).get("status") or "PENDING"),
    }

    msgs = (pipeline or {}).get("messages") or []
    agent_assessments = {
        m.get("agent_name", f"Agent_{idx}"): {
            "role": m.get("agent_role", ""),
            "signal": m.get("signal", "NEUTRAL"),
            "confidence_pct": round(float(m.get("confidence", 0.0)) * 100.0, 1),
            "summary": m.get("summary", ""),
        }
        for idx, m in enumerate(msgs)
    }

    crit_block = {
        "status": (critique or {}).get("status") or "unavailable",
        "stance": (critique or {}).get("stance") or "PENDING",
        "risk_rating": (critique or {}).get("risk_rating") or "UNKNOWN",
        "bull_case": (critique or {}).get("bull_case") or "",
        "objections_raised": (critique or {}).get("critical_risks") or "",
        "rationale": (critique or {}).get("rationale") or "",
        "model": (critique or {}).get("model") or None,
        "generated_at": (critique or {}).get("generated_at") or now_iso,
    }

    deltas = {
        "applied": bool(recal.get("applied")),
        "entry_delta_pct": _pct_delta(float(before["entry"]), float(after["entry"])),
        "target_delta_pct": _pct_delta(float(before["target"]), float(after["target"])),
        "stop_delta_pct": _pct_delta(float(before["stop_loss"]), float(after["stop_loss"])),
        "qty_delta": int(after["qty"]) - int(before["qty"]),
        "conviction_delta": round(float(after["conviction"]) - float(before["conviction"]), 1),
        "reasons": _extract_reasons(critique, recal),
        "summary": recal.get("summary") or "Initial quantitative levels active.",
    }

    sig_payload = "|".join([
        symbol,
        mkt_date,
        str(before.get("action")),
        str(after.get("action")),
        str(after.get("entry")),
        str(after.get("target")),
        str(after.get("stop_loss")),
        str(after.get("qty")),
        str(crit_block.get("stance")),
        force_new_stamp,
    ])
    sig_hash = hashlib.sha256(sig_payload.encode("utf-8")).hexdigest()[:8]
    audit_id = f"fnv2-aud-{mkt_date.replace('-', '')}-{symbol}-{sig_hash}"

    f = analysis.get("fundamental_score") or {}
    p = analysis.get("porters_score") or {}
    tech_conf_100 = agent_assessments.get("TechBot_1", {}).get("confidence_pct", 50.0)

    is_deferred = after.get("status") == "DEFERRED" or "AVOID" in str(after.get("action", "")).upper()

    return {
        "audit_id": audit_id,
        "signature": hashlib.sha256(sig_payload.encode("utf-8")).hexdigest()[:16],
        "symbol": symbol,
        "name": analysis.get("name", symbol),
        "currency": analysis.get("currency", "INR"),
        "created_at": now_iso,
        "last_seen_at": now_iso,
        "market_date": mkt_date,
        "market_timestamp": mkt_ts,
        "retrieval_timestamp": ret_ts,
        "versions": {
            **ENGINE_VERSIONS,
            "critic_model": crit_block["model"] or "none",
        },
        "scores": {
            "fundamental_total": f.get("total", 0),
            "business": f.get("business", 0),
            "moat": f.get("moat", 0),
            "management": f.get("management", 0),
            "risk": f.get("risk", 0),
            "porters_total": p.get("total", 0),
            "technical_score": tech_conf_100,
            "composite_score": raw_m.get("x_composite_score") or cur_m.get("x_composite_score") or 0,
        },
        "agent_assessments": agent_assessments,
        "market_snapshot": {
            "spot_price": float(analysis.get("price") or 0.0),
            "change_pct": float(analysis.get("change_pct") or 0.0),
            "rsi_14": float(analysis.get("rsi_14") or 50.0),
            "sma_20": float(analysis.get("sma_20") or 0.0),
            "projected_30d_target": float(analysis.get("projected_30d_target") or 0.0),
            "regression_curve": analysis.get("regression_curve", ""),
            "data_source": analysis.get("data_source", "Yahoo Finance Live"),
        },
        "original_decision": before,
        "critical_review": crit_block,
        "final_decision": after,
        "recalibration_deltas": deltas,
        "outcome": {
            "status": "DEFERRED_MONITORED" if is_deferred else "OPEN",
            "broker_executed": False,
            "order_id": None,
            "broker_fill_price": None,
            "broker_timestamp": None,
            "latest_mark_price": float(analysis.get("price") or 0.0),
            "mark_date": mkt_date,
            "realized_return_pct": None,
            "unrealized_return_pct": 0.0,
            "counterfactual_return_pct": 0.0 if is_deferred else None,
            "mfe_pct": 0.0,
            "mae_pct": 0.0,
            "holding_days": 0,
            "marked_at": now_iso,
        },
    }


def _update_outcome_from_bars(rec: dict[str, Any], analysis: dict[str, Any]) -> None:
    """Update historical audit record outcome deterministically from subsequent real daily bars."""
    candles = analysis.get("candles") or []
    rec_date = str(rec.get("market_date") or "")
    out = rec.get("outcome") or {}
    final_d = rec.get("final_decision") or {}
    spot_now = float(analysis.get("price") or 0.0)
    last_bar_date = str(analysis.get("last_candle_date") or (candles[-1]["date"] if candles else rec_date))

    if out.get("status") in ("TARGET_HIT", "STOP_HIT", "HORIZON_CLOSED") and out.get("realized_return_pct") is not None:
        return

    entry_ref = float(out.get("broker_fill_price") or final_d.get("entry") or rec.get("market_snapshot", {}).get("spot_price") or 0.0)
    if entry_ref <= 0:
        return

    fwd_bars = [c for c in candles if str(c.get("date", "")) > rec_date]
    now_iso = datetime.now(timezone.utc).isoformat()
    if not fwd_bars:
        if spot_now > 0:
            pct = round(((spot_now - entry_ref) / entry_ref) * 100.0, 2)
            out["latest_mark_price"] = spot_now
            out["mark_date"] = last_bar_date
            if final_d.get("status") == "DEFERRED":
                out["counterfactual_return_pct"] = pct
            else:
                out["unrealized_return_pct"] = pct
            out["marked_at"] = now_iso
        rec["outcome"] = out
        return

    tp = float(final_d.get("target") or 0.0)
    sl = float(final_d.get("stop_loss") or 0.0)
    mfe = float(out.get("mfe_pct") or 0.0)
    mae = float(out.get("mae_pct") or 0.0)

    is_deferred = final_d.get("status") == "DEFERRED" or "AVOID" in str(final_d.get("action", "")).upper()
    if is_deferred:
        latest_c = float(fwd_bars[-1]["close"])
        out["latest_mark_price"] = latest_c
        out["mark_date"] = fwd_bars[-1]["date"]
        out["holding_days"] = len(fwd_bars)
        out["counterfactual_return_pct"] = round(((latest_c - entry_ref) / entry_ref) * 100.0, 2)
        out["status"] = "DEFERRED_COUNTERFACTUAL"
        out["marked_at"] = now_iso
        rec["outcome"] = out
        return

    for idx, bar in enumerate(fwd_bars[:14]):
        hi = float(bar.get("high") or bar.get("close") or entry_ref)
        lo = float(bar.get("low") or bar.get("close") or entry_ref)
        cl = float(bar.get("close") or entry_ref)
        mfe = max(mfe, round(((hi - entry_ref) / entry_ref) * 100.0, 2))
        mae = min(mae, round(((lo - entry_ref) / entry_ref) * 100.0, 2))
        out["mfe_pct"] = mfe
        out["mae_pct"] = mae
        out["holding_days"] = idx + 1
        out["latest_mark_price"] = cl
        out["mark_date"] = bar["date"]

        if sl > 0 and lo <= sl:
            out["status"] = "STOP_HIT"
            out["realized_return_pct"] = round(((sl - entry_ref) / entry_ref) * 100.0 - 0.35, 2)
            out["marked_at"] = now_iso
            rec["outcome"] = out
            return
        if tp > 0 and hi >= tp:
            out["status"] = "TARGET_HIT"
            out["realized_return_pct"] = round(((tp - entry_ref) / entry_ref) * 100.0 - 0.35, 2)
            out["marked_at"] = now_iso
            rec["outcome"] = out
            return

    latest_c = float(fwd_bars[min(13, len(fwd_bars) - 1)]["close"])
    pct = round(((latest_c - entry_ref) / entry_ref) * 100.0, 2)
    out["unrealized_return_pct"] = pct
    if len(fwd_bars) >= 14:
        out["status"] = "HORIZON_CLOSED"
        out["realized_return_pct"] = round(pct - 0.35, 2)
    else:
        out["status"] = "BROKER_EXECUTED" if out.get("broker_executed") else "OPEN"
    out["marked_at"] = now_iso
    rec["outcome"] = out


def record_scan_decisions(
    assets: dict[str, dict[str, Any]],
    pipelines: dict[str, dict[str, Any]],
    trade_queue: list[dict[str, Any]],
    critiques: dict[str, dict[str, Any]],
    ledger_path: Path = DEFAULT_LEDGER_PATH,
    force_new_for_symbol: str | None = None,
    max_records: int = 500,
) -> dict[str, Any]:
    """Persist scan decisions, mark existing records to market, and return ledger summary."""
    ledger = load_audit_ledger(ledger_path)
    records: list[dict[str, Any]] = ledger.get("records") or []

    # 1. Update forward outcomes for any historical record whose symbol is in current live assets
    for rec in records:
        sym = str(rec.get("symbol", "")).upper()
        if sym in assets:
            _update_outcome_from_bars(rec, assets[sym])

    # 2. Index latest record per (symbol, market_date) for clean deduplication
    latest_by_sym_date: dict[tuple[str, str], dict[str, Any]] = {}
    for rec in records:
        key = (str(rec.get("symbol", "")).upper(), str(rec.get("market_date", "")))
        if key not in latest_by_sym_date:
            latest_by_sym_date[key] = rec

    pick_by_sym = {str(p.get("symbol", "")).upper(): p for p in trade_queue if p.get("symbol")}
    now_stamp = datetime.now(timezone.utc).strftime("%H%M%S")

    new_count = 0
    for sym, analysis in assets.items():
        force_stamp = now_stamp if (force_new_for_symbol and sym.upper() == force_new_for_symbol.upper()) else ""
        rec = _build_record(
            symbol=sym,
            analysis=analysis,
            pipeline=pipelines.get(sym),
            pick=pick_by_sym.get(sym.upper()),
            critique=critiques.get(sym),
            force_new_stamp=force_stamp,
        )
        existing = latest_by_sym_date.get((sym.upper(), rec["market_date"]))
        if existing and existing.get("signature") == rec["signature"] and not force_stamp:
            existing["last_seen_at"] = rec["last_seen_at"]
            existing["outcome"]["latest_mark_price"] = rec["outcome"]["latest_mark_price"]
        else:
            records.insert(0, rec)
            latest_by_sym_date[(sym.upper(), rec["market_date"])] = rec
            new_count += 1

    records.sort(key=lambda r: str(r.get("created_at", "")), reverse=True)
    ledger["records"] = records[:max_records]
    ledger["last_scan_new_records"] = new_count
    save_audit_ledger(ledger, ledger_path)
    return ledger


def record_broker_execution(
    symbol: str,
    side: str,
    qty: int,
    fill_price: float,
    order_id: str,
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> dict[str, Any]:
    """Attach a real broker-confirmed execution outcome to the latest audit record for `symbol`."""
    ledger = load_audit_ledger(ledger_path)
    records = ledger.get("records") or []
    now_iso = datetime.now(timezone.utc).isoformat()
    target_rec = next((r for r in records if str(r.get("symbol", "")).upper() == symbol.upper()), None)
    if target_rec:
        out = target_rec.get("outcome") or {}
        out["broker_executed"] = True
        out["order_id"] = order_id
        out["broker_side"] = side
        out["broker_qty"] = qty
        out["broker_fill_price"] = round(float(fill_price), 2)
        out["broker_timestamp"] = now_iso
        out["status"] = "BROKER_EXECUTED"
        target_rec["final_decision"]["status"] = "EXECUTED"
        target_rec["outcome"] = out
        save_audit_ledger(ledger, ledger_path)
    return ledger


def build_audit_dashboard_view(ledger: dict[str, Any], limit: int = 80) -> dict[str, Any]:
    """Build compact, UI-ready audit ledger view for Web, Mobile APK, and /api/audit."""
    records = (ledger.get("records") or [])[:limit]
    return {
        "updated_at": ledger.get("updated_at"),
        "total_records": len(ledger.get("records") or []),
        "last_market_timestamp": records[0].get("market_timestamp") if records else None,
        "engine_versions": ENGINE_VERSIONS,
        "records": records,
    }
