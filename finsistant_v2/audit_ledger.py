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
    "moreshwar_model": "MOVIQ-Dynamic-V2",
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


def _eval_plan_forward(
    fwd_bars: list[dict[str, Any]],
    entry_ref: float,
    tp: float,
    sl: float,
    weight: float = 1.0,
) -> tuple[str, float, float, float, int]:
    """Deterministically evaluate a decision plan over up to 14 forward daily bars."""
    if not fwd_bars or entry_ref <= 0:
        return "OPEN", 0.0, 0.0, 0.0, 0
    mfe = 0.0
    mae = 0.0
    for idx, bar in enumerate(fwd_bars[:14]):
        hi = float(bar.get("high") or bar.get("close") or entry_ref)
        lo = float(bar.get("low") or bar.get("close") or entry_ref)
        mfe = max(mfe, round(((hi - entry_ref) / entry_ref) * 100.0, 2))
        mae = min(mae, round(((lo - entry_ref) / entry_ref) * 100.0, 2))
        if sl > 0 and lo <= sl:
            ret = round((((sl - entry_ref) / entry_ref) * 100.0 - 0.35) * weight, 2)
            return "STOP_HIT", ret, mfe, mae, idx + 1
        if tp > 0 and hi >= tp:
            ret = round((((tp - entry_ref) / entry_ref) * 100.0 - 0.35) * weight, 2)
            return "TARGET_HIT", ret, mfe, mae, idx + 1
    last_bar = fwd_bars[min(13, len(fwd_bars) - 1)]
    cl = float(last_bar.get("close") or entry_ref)
    raw_ret = ((cl - entry_ref) / entry_ref) * 100.0
    if len(fwd_bars) >= 14:
        return "HORIZON_CLOSED", round((raw_ret - 0.35) * weight, 2), mfe, mae, 14
    return "OPEN", round(raw_ret * weight, 2), mfe, mae, len(fwd_bars)


def _classify_cro_verdict(
    recal_applied: bool,
    is_deferred: bool,
    settled: bool,
    orig_ret: float | None,
    final_ret: float | None,
) -> str:
    if not recal_applied:
        return "NO_INTERVENTION"
    if not settled or orig_ret is None:
        return "PENDING_FORWARD_BARS"
    if is_deferred:
        return "PREVENTED_BAD_TRADE" if orig_ret < 0 else "UNNECESSARY_BLOCK"
    f_val = float(final_ret if final_ret is not None else orig_ret)
    if orig_ret < 0 and f_val > orig_ret + 0.05:
        return "PREVENTED_BAD_TRADE"
    if orig_ret >= 0 and f_val < orig_ret - 0.05:
        return "UNNECESSARY_DOWNGRADE"
    if orig_ret >= 0 and f_val >= orig_ret - 0.05:
        return "PRESERVED_OR_IMPROVED_WINNER"
    if f_val > orig_ret + 0.05:
        return "BENEFICIAL_DOWNGRADE"
    return "NEUTRAL_OR_TIGHTENED"


def _update_outcome_from_bars(rec: dict[str, Any], analysis: dict[str, Any]) -> None:
    """Update historical audit record outcome & CRO counterfactuals deterministically from real daily bars."""
    candles = analysis.get("candles") or []
    rec_date = str(rec.get("market_date") or "")
    out = rec.get("outcome") or {}
    orig_d = rec.get("original_decision") or {}
    final_d = rec.get("final_decision") or {}
    deltas = rec.get("recalibration_deltas") or {}
    spot_now = float(analysis.get("price") or 0.0)
    last_bar_date = str(analysis.get("last_candle_date") or (candles[-1]["date"] if candles else rec_date))

    if out.get("status") in ("TARGET_HIT", "STOP_HIT", "HORIZON_CLOSED") and out.get("realized_return_pct") is not None and out.get("cro_verdict"):
        return

    entry_ref = float(out.get("broker_fill_price") or final_d.get("entry") or rec.get("market_snapshot", {}).get("spot_price") or 0.0)
    if entry_ref <= 0:
        return

    orig_entry = float(orig_d.get("entry") or rec.get("market_snapshot", {}).get("spot_price") or entry_ref)
    orig_tp = float(orig_d.get("target") or 0.0)
    orig_sl = float(orig_d.get("stop_loss") or 0.0)
    orig_qty = max(1, int(orig_d.get("qty") or 1))
    fin_qty = int(final_d.get("qty") if final_d.get("qty") is not None else orig_qty)
    size_weight = min(1.25, max(0.0, fin_qty / orig_qty)) if deltas.get("applied") else 1.0

    fwd_bars = [c for c in candles if str(c.get("date", "")) > rec_date]
    now_iso = datetime.now(timezone.utc).isoformat()
    is_deferred = final_d.get("status") == "DEFERRED" or "AVOID" in str(final_d.get("action", "")).upper()

    if not fwd_bars:
        if spot_now > 0:
            pct = round(((spot_now - entry_ref) / entry_ref) * 100.0, 2)
            orig_pct = round(((spot_now - orig_entry) / max(1e-6, orig_entry)) * 100.0, 2)
            out["latest_mark_price"] = spot_now
            out["mark_date"] = last_bar_date
            out["original_counterfactual_pct"] = orig_pct
            out["original_counterfactual_status"] = "OPEN"
            if is_deferred:
                out["counterfactual_return_pct"] = orig_pct
            else:
                out["unrealized_return_pct"] = pct
            out["cro_verdict"] = _classify_cro_verdict(bool(deltas.get("applied")), is_deferred, False, orig_pct, pct)
            out["marked_at"] = now_iso
        rec["outcome"] = out
        return

    orig_status, orig_ret, _, _, _ = _eval_plan_forward(fwd_bars, orig_entry, orig_tp, orig_sl, weight=1.0)
    out["original_counterfactual_status"] = orig_status
    out["original_counterfactual_pct"] = orig_ret
    orig_settled = orig_status in ("TARGET_HIT", "STOP_HIT", "HORIZON_CLOSED")

    if is_deferred:
        last_idx_d = min(13, len(fwd_bars) - 1)
        latest_c = float(fwd_bars[last_idx_d]["close"])
        mark_dt = fwd_bars[last_idx_d]["date"]
        out["latest_mark_price"] = latest_c
        out["mark_date"] = mark_dt
        out["holding_days"] = min(14, len(fwd_bars))
        out["counterfactual_return_pct"] = orig_ret
        out["status"] = "DEFERRED_COUNTERFACTUAL" if orig_settled else "DEFERRED_MONITORED"
        out["cro_verdict"] = _classify_cro_verdict(True, True, orig_settled, orig_ret, 0.0)
        out["marked_at"] = now_iso
        rec["outcome"] = out
        return

    tp = float(final_d.get("target") or 0.0)
    sl = float(final_d.get("stop_loss") or 0.0)
    fin_status, fin_ret, mfe, mae, hold_days = _eval_plan_forward(fwd_bars, entry_ref, tp, sl, weight=1.0)
    weighted_fin_ret = round(fin_ret * size_weight, 2)

    last_idx = min(max(0, hold_days - 1), len(fwd_bars) - 1)
    out["mfe_pct"] = max(float(out.get("mfe_pct") or 0.0), mfe)
    out["mae_pct"] = min(float(out.get("mae_pct") or 0.0), mae)
    out["holding_days"] = hold_days
    out["latest_mark_price"] = float(fwd_bars[last_idx].get("close") or entry_ref)
    out["mark_date"] = fwd_bars[last_idx]["date"]

    fin_settled = fin_status in ("TARGET_HIT", "STOP_HIT", "HORIZON_CLOSED")
    if fin_settled:
        out["status"] = fin_status
        out["realized_return_pct"] = fin_ret
    else:
        out["unrealized_return_pct"] = fin_ret
        out["status"] = "BROKER_EXECUTED" if out.get("broker_executed") else "OPEN"

    out["cro_verdict"] = _classify_cro_verdict(
        bool(deltas.get("applied")),
        False,
        fin_settled and orig_settled,
        orig_ret,
        weighted_fin_ret,
    )
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


def summarize_live_ledger_quality(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize real forward-accumulated ledger settlements & live CRO confusion outcomes."""
    settled = [
        r for r in records
        if (r.get("outcome") or {}).get("status") in ("TARGET_HIT", "STOP_HIT", "HORIZON_CLOSED", "DEFERRED_COUNTERFACTUAL")
    ]
    verdict_counts: dict[str, int] = {
        "PREVENTED_BAD_TRADE": 0,
        "BENEFICIAL_DOWNGRADE": 0,
        "PRESERVED_OR_IMPROVED_WINNER": 0,
        "UNNECESSARY_DOWNGRADE": 0,
        "UNNECESSARY_BLOCK": 0,
        "PENDING_FORWARD_BARS": 0,
    }
    for r in records:
        v = str((r.get("outcome") or {}).get("cro_verdict") or "PENDING_FORWARD_BARS")
        if v in verdict_counts:
            verdict_counts[v] += 1

    distinct_dates = sorted({str(r.get("market_date", "")) for r in records if r.get("market_date")})
    settled_cro_total = (
        verdict_counts["PREVENTED_BAD_TRADE"]
        + verdict_counts["BENEFICIAL_DOWNGRADE"]
        + verdict_counts["PRESERVED_OR_IMPROVED_WINNER"]
        + verdict_counts["UNNECESSARY_DOWNGRADE"]
        + verdict_counts["UNNECESSARY_BLOCK"]
    )
    prevented_total = verdict_counts["PREVENTED_BAD_TRADE"] + verdict_counts["BENEFICIAL_DOWNGRADE"]
    unnecessary_total = verdict_counts["UNNECESSARY_DOWNGRADE"] + verdict_counts["UNNECESSARY_BLOCK"]

    return {
        "total_records": len(records),
        "distinct_market_dates": len(distinct_dates),
        "date_span": [distinct_dates[0], distinct_dates[-1]] if distinct_dates else [],
        "settled_count": len(settled),
        "open_count": len(records) - len(settled),
        "min_recommended_settled": 30,
        "sufficient_live_sample": len(settled) >= 30,
        "cro_settled_interventions": settled_cro_total,
        "cro_prevented_bad_trades": prevented_total,
        "cro_unnecessary_downgrades": unnecessary_total,
        "cro_preserved_winners": verdict_counts["PRESERVED_OR_IMPROVED_WINNER"],
        "verdict_breakdown": verdict_counts,
    }


def build_audit_dashboard_view(ledger: dict[str, Any], limit: int = 80) -> dict[str, Any]:
    """Build compact, UI-ready audit ledger view for Web, Mobile APK, and /api/audit."""
    all_records = ledger.get("records") or []
    records = all_records[:limit]
    return {
        "updated_at": ledger.get("updated_at"),
        "total_records": len(all_records),
        "last_market_timestamp": records[0].get("market_timestamp") if records else None,
        "engine_versions": ENGINE_VERSIONS,
        "accumulation_summary": summarize_live_ledger_quality(all_records),
        "records": records,
    }
