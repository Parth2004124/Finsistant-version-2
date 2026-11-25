"""Data Provenance & Reliability Layer for Finsistant Version 2.

Categorizes every displayed financial metric across six explicit origins:
1. YAHOO_LIVE_FEED          — Yahoo Finance live quote feed
2. HISTORICAL_MARKET_DATA   — Historical OHLCV market data bars
3. BROKER_API               — Zerodha Kite Connect v3 live portfolio/order API
4. CACHED_DATABASE          — Cached records / Git ledger / offline cache
5. INTERNAL_CALCULATION     — Deterministic indicators, StockSight, MOVIQ
6. AI_ANALYTICAL_ESTIMATE   — Gemini Critical Analyst estimates & recalibrations

Strictly separates `market_timestamp` from `retrieval_timestamp`, flags stale/missing/proxied
fields, detects cross-source price discrepancies, and distinguishes analytical estimates from
broker-confirmed execution prices.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

SOURCE_TAXONOMY: dict[str, dict[str, str]] = {
    "NSE_OFFICIAL_MCP": {
        "code": "NSE_OFFICIAL_MCP",
        "label": "Official NSE India MCP (cm-market & bhavcopy)",
        "short": "NSE MCP",
        "tone": "emerald",
    },
    "YAHOO_LIVE_FEED": {
        "code": "YAHOO_LIVE_FEED",
        "label": "Yahoo Finance Live Feed (US Equities & Ratios)",
        "short": "YF LIVE",
        "tone": "cyan",
    },
    "HISTORICAL_MARKET_DATA": {
        "code": "HISTORICAL_MARKET_DATA",
        "label": "Historical Market Data",
        "short": "HIST BAR",
        "tone": "amber",
    },
    "BROKER_API": {
        "code": "BROKER_API",
        "label": "Broker API (Zerodha Kite)",
        "short": "BROKER",
        "tone": "emerald",
    },
    "CACHED_DATABASE": {
        "code": "CACHED_DATABASE",
        "label": "Cached Database Records",
        "short": "CACHED",
        "tone": "amber",
    },
    "INTERNAL_CALCULATION": {
        "code": "INTERNAL_CALCULATION",
        "label": "Internally Calculated Indicators",
        "short": "CALC",
        "tone": "cyan",
    },
    "AI_ANALYTICAL_ESTIMATE": {
        "code": "AI_ANALYTICAL_ESTIMATE",
        "label": "AI-Generated Analytical Estimates",
        "short": "AI EST",
        "tone": "crimson",
    },
}

PRICE_CLASSIFICATIONS: dict[str, str] = {
    "MARKET_FEED": "Market Quote (Exchange/Yahoo Feed)",
    "HISTORICAL_EOD": "Historical End-of-Day Bar",
    "ANALYTICAL_ESTIMATE": "Analytical Model Price (Not Executed)",
    "BROKER_CONFIRMED": "Broker-Confirmed Execution / Holding Price",
}


def _parse_iso(ts: str | None) -> datetime | None:
    if not ts:
        return None
    raw = str(ts).strip()
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(raw)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def compute_age_hours(ts: str | None, now: datetime | None = None) -> float | None:
    dt = _parse_iso(ts)
    if dt is None:
        return None
    ref = now or datetime.now(timezone.utc)
    return round(max(0.0, (ref - dt).total_seconds() / 3600.0), 2)


def classify_freshness(
    market_age_hours: float | None,
    retrieval_age_hours: float | None,
    cache_hit: bool = False,
    market_ts_kind: str = "end_of_day_bar",
    unavailable: bool = False,
) -> tuple[str, bool, bool]:
    """Return (freshness_label, is_stale, is_delayed)."""
    if unavailable or market_age_hours is None:
        return "UNAVAILABLE", True, True
    # Beyond 96h (> 4 calendar days) covers weekends + market holidays before flagging STALE
    if market_age_hours > 96.0 or (retrieval_age_hours is not None and retrieval_age_hours > 24.0):
        return "STALE", True, True
    if cache_hit:
        return "CACHED", False, True
    if market_ts_kind == "end_of_day_bar" or market_age_hours > 6.0:
        return "DELAYED / EOD", False, True
    return "LIVE", False, False


def build_asset_provenance(
    symbol: str,
    analysis: dict[str, Any],
    critique: dict[str, Any] | None = None,
    kite_holding: dict[str, Any] | None = None,
    kite_status: dict[str, Any] | None = None,
    executed_orders: list[dict[str, Any]] | None = None,
    now_utc: datetime | None = None,
) -> dict[str, Any]:
    """Build granular provenance, freshness, and discrepancy telemetry for one symbol."""
    ref_now = now_utc or datetime.now(timezone.utc)
    retrieval_ts = analysis.get("fetched_at") or ref_now.isoformat()
    market_ts = analysis.get("market_timestamp") or (
        f"{analysis.get('last_candle_date', '')}T10:00:00+00:00" if analysis.get("last_candle_date") else None
    )
    market_kind = str(analysis.get("market_timestamp_kind") or "end_of_day_bar")
    # `cache_hit` in live_data is only a <=60s in-process memo of a fetch made during the SAME
    # scan run (the runner is ephemeral, nothing persists between runs). It is a live source
    # fetch, not a "cached database" record, so it must not downgrade provenance to CACHED_DATABASE.
    memo_hit = bool(analysis.get("cache_hit", False))
    memo_age_s = float(analysis.get("cache_age_seconds") or 0.0)
    cache_hit = memo_hit and memo_age_s > 120.0

    mkt_age_h = compute_age_hours(market_ts, ref_now)
    ret_age_h = compute_age_hours(retrieval_ts, ref_now)
    spot_freshness, spot_stale, spot_delayed = classify_freshness(
        mkt_age_h, ret_age_h, cache_hit=cache_hit, market_ts_kind=market_kind, unavailable=not analysis.get("price")
    )

    spot = float(analysis.get("price") or 0.0)
    last_close = float(analysis.get("last_candle_close") or 0.0)
    missing_fields = list(analysis.get("missing_fields") or [])
    proxy_fields = dict(analysis.get("proxy_fields") or {})
    discrepancies: list[dict[str, Any]] = []

    # 1. Cross-check Spot vs Last Daily Bar Close
    if spot > 0 and last_close > 0:
        bar_gap_pct = round(((spot - last_close) / last_close) * 100.0, 2)
        if abs(bar_gap_pct) >= 1.5:
            discrepancies.append({
                "type": "SPOT_VS_EOD_CLOSE",
                "severity": "INFO" if market_kind == "exchange_quote" else "WARNING",
                "source_a": f"Live Quote ({spot})",
                "source_b": f"Last OHLCV Close ({last_close})",
                "delta_pct": bar_gap_pct,
                "message": f"Quote diverges {bar_gap_pct:+.2f}% from last daily bar close ({analysis.get('last_candle_date')}).",
            })

    # 2. Cross-check Primary Feed (NSE MCP / Yahoo) vs Secondary Spot (Yahoo Finance) if available
    sec_spot = analysis.get("secondary_spot_price")
    if sec_spot is not None and float(sec_spot or 0.0) > 0 and spot > 0:
        sec_val = float(sec_spot)
        sec_gap_pct = round(((sec_val - spot) / spot) * 100.0, 2)
        if abs(sec_gap_pct) >= 0.75:
            discrepancies.append({
                "type": "NSE_MCP_VS_YAHOO",
                "severity": "WARNING" if abs(sec_gap_pct) < 2.5 else "CRITICAL",
                "source_a": f"NSE MCP Spot ({spot})",
                "source_b": f"{analysis.get('secondary_source') or 'Yahoo Spot'} ({sec_val})",
                "delta_pct": sec_gap_pct,
                "message": f"Secondary feed diverges {sec_gap_pct:+.2f}% from Official NSE MCP quote.",
            })

    # 3. Cross-check Market Feed Spot vs Zerodha Kite Broker LTP if held
    kite_ltp = None
    kite_avg = None
    kite_ts = (kite_status or {}).get("fetched_at")
    if kite_holding and float(kite_holding.get("ltp") or 0.0) > 0:
        kite_ltp = float(kite_holding["ltp"])
        kite_avg = float(kite_holding.get("avg_price") or 0.0)
        if spot > 0:
            broker_gap_pct = round(((kite_ltp - spot) / spot) * 100.0, 2)
            if abs(broker_gap_pct) >= 0.75:
                feed_name = "NSE MCP" if analysis.get("price_source_category") == "NSE_OFFICIAL_MCP" else "Yahoo"
                discrepancies.append({
                    "type": "FEED_VS_BROKER_LTP",
                    "severity": "WARNING" if abs(broker_gap_pct) < 2.5 else "CRITICAL",
                    "source_a": f"{feed_name} Spot ({spot})",
                    "source_b": f"Kite Broker LTP ({kite_ltp})",
                    "delta_pct": broker_gap_pct,
                    "message": f"Broker LTP diverges {broker_gap_pct:+.2f}% from {feed_name} quote — verify before routing orders.",
                })

    # Latest broker execution price (strictly from real order executions, never Moreshwar)
    latest_fill = next(
        (o for o in (executed_orders or []) if str(o.get("symbol", "")).upper() == symbol.upper()),
        None,
    )

    crit_status = (critique or {}).get("status")
    crit_ts = (critique or {}).get("generated_at")
    crit_model = (critique or {}).get("model") or "gemini-2.5-flash"

    # Build per-metric provenance dictionary
    raw_spot_cat = str(analysis.get("price_source_category") or "YAHOO_LIVE_FEED")
    if raw_spot_cat not in SOURCE_TAXONOMY:
        raw_spot_cat = "YAHOO_LIVE_FEED"
    spot_cat = "CACHED_DATABASE" if cache_hit else raw_spot_cat

    raw_ohlcv_cat = str(analysis.get("ohlcv_source_category") or "HISTORICAL_MARKET_DATA")
    if raw_ohlcv_cat not in SOURCE_TAXONOMY:
        raw_ohlcv_cat = "HISTORICAL_MARKET_DATA"

    raw_fund_cat = str(analysis.get("fundamental_source_category") or "YAHOO_LIVE_FEED")
    if cache_hit:
        fund_cat = "CACHED_DATABASE"
    elif raw_fund_cat in SOURCE_TAXONOMY:
        fund_cat = raw_fund_cat
    else:
        fund_cat = "YAHOO_LIVE_FEED"

    fund_flags: list[str] = []
    if missing_fields:
        fund_flags.append(f"Missing fields defaulted/omitted: {', '.join(missing_fields)}")
    if proxy_fields:
        fund_flags.extend(f"{k}: {v}" for k, v in proxy_fields.items())

    corp_adj = analysis.get("nse_corporate_adjustments") or []
    ohlcv_note = f"Covers bars ending {analysis.get('last_candle_date', 'N/A')}."
    if raw_ohlcv_cat == "NSE_OFFICIAL_MCP":
        ohlcv_note += f" (NSE Bhavcopy MCP · {len(corp_adj)} split/bonus adjustments applied)"

    metrics = {
        "spot_price": {
            "label": "Spot Price",
            "value": spot,
            "source_category": spot_cat,
            "source_label": SOURCE_TAXONOMY[spot_cat]["label"],
            "price_classification": "MARKET_FEED",
            "market_timestamp": market_ts,
            "market_timestamp_kind": market_kind,
            "retrieval_timestamp": retrieval_ts,
            "market_age_hours": mkt_age_h,
            "retrieval_age_hours": ret_age_h,
            "freshness": spot_freshness,
            "is_stale": spot_stale,
            "is_delayed": spot_delayed,
            "notes": (
                f"NSE Exchange quote IST {analysis.get('nse_raw_ist_timestamp')} vs retrieval UTC"
                if analysis.get("nse_raw_ist_timestamp")
                else ("Exchange quote timestamp differs from retrieval timestamp" if market_kind == "exchange_quote" else "Based on latest daily close bar")
            ),
        },
        "ohlcv_history": {
            "label": "1Y Daily OHLCV History",
            "value": f"{analysis.get('history_bars_count', len(analysis.get('candles') or []))} bars",
            "source_category": raw_ohlcv_cat,
            "source_label": SOURCE_TAXONOMY[raw_ohlcv_cat]["label"],
            "price_classification": "HISTORICAL_EOD",
            "market_timestamp": f"{analysis.get('last_candle_date', '')}T10:00:00+00:00" if analysis.get("last_candle_date") else market_ts,
            "retrieval_timestamp": retrieval_ts,
            "freshness": "STALE" if spot_stale else "HISTORICAL",
            "is_stale": spot_stale,
            "is_delayed": True,
            "notes": ohlcv_note,
        },
        "fundamental_ratios": {
            "label": "Fundamental Ratios (P/E, ROE, OPM, Growth)",
            "value": f"{len(missing_fields)} missing / {len(proxy_fields)} proxied",
            "source_category": fund_cat,
            "source_label": SOURCE_TAXONOMY[fund_cat]["label"],
            "price_classification": "COMPUTED_METRIC",
            "market_timestamp": market_ts,
            "retrieval_timestamp": retrieval_ts,
            "freshness": "UNAVAILABLE" if raw_fund_cat == "UNAVAILABLE" else ("PARTIAL" if missing_fields else spot_freshness),
            "is_stale": spot_stale or raw_fund_cat == "UNAVAILABLE",
            "is_delayed": True,
            "warnings": fund_flags,
        },
        "technical_indicators": {
            "label": "RSI-14, MACD, SMA-20, EMA-12/26, Degree-2 Poly Reg",
            "value": f"RSI {analysis.get('rsi_14')} | 30D Reg {analysis.get('projected_30d_target')}",
            "source_category": "INTERNAL_CALCULATION",
            "source_label": SOURCE_TAXONOMY["INTERNAL_CALCULATION"]["label"],
            "price_classification": "ANALYTICAL_ESTIMATE",
            "market_timestamp": market_ts,
            "retrieval_timestamp": retrieval_ts,
            "freshness": "COMPUTED",
            "is_stale": spot_stale,
            "is_delayed": spot_delayed,
            "notes": "Computed locally by quant_engine.py from real historical bars.",
        },
        "stocksight_and_moreshwar": {
            "label": "StockSight V2 Score & MOVIQ Entry/TP/SL",
            "value": f"Fund {analysis.get('fundamental_score', {}).get('total')}/100 | Porter {analysis.get('porters_score', {}).get('total')}/100",
            "source_category": "INTERNAL_CALCULATION",
            "source_label": SOURCE_TAXONOMY["INTERNAL_CALCULATION"]["label"],
            "price_classification": "ANALYTICAL_ESTIMATE",
            "market_timestamp": market_ts,
            "retrieval_timestamp": retrieval_ts,
            "freshness": "COMPUTED",
            "is_stale": spot_stale,
            "is_delayed": spot_delayed,
            "notes": "Analytical model targets — NOT broker-confirmed execution prices.",
        },
        "gemini_cro": {
            "label": "Gemini Critical Analyst & Recalibration",
            "value": (critique or {}).get("stance") or "UNAVAILABLE",
            "source_category": "AI_ANALYTICAL_ESTIMATE",
            "source_label": SOURCE_TAXONOMY["AI_ANALYTICAL_ESTIMATE"]["label"],
            "price_classification": "ANALYTICAL_ESTIMATE",
            "market_timestamp": market_ts,
            "retrieval_timestamp": crit_ts or retrieval_ts,
            "freshness": "LIVE" if crit_status == "live" else "UNAVAILABLE",
            "is_stale": crit_status != "live",
            "is_delayed": crit_status != "live",
            "notes": f"Model: {crit_model}" if crit_status == "live" else ((critique or {}).get("error") or "AI key missing/error"),
        },
        "broker_execution": {
            "label": "Zerodha Kite Holding / Execution Price",
            "value": latest_fill.get("price") if latest_fill else (kite_ltp if kite_ltp is not None else "NO_BROKER_FILL"),
            "avg_cost": kite_avg,
            "source_category": "BROKER_API",
            "source_label": SOURCE_TAXONOMY["BROKER_API"]["label"],
            "price_classification": "BROKER_CONFIRMED" if (latest_fill or kite_ltp is not None) else "UNAVAILABLE",
            "market_timestamp": kite_ts,
            "retrieval_timestamp": kite_ts or retrieval_ts,
            "freshness": "LIVE" if (kite_status or {}).get("connected") else "DISCONNECTED",
            "is_stale": not bool((kite_status or {}).get("connected")),
            "is_delayed": not bool((kite_status or {}).get("connected")),
            "notes": (
                f"Confirmed fill {latest_fill.get('order_id')} @ {latest_fill.get('price')}"
                if latest_fill
                else ("Live Kite holding LTP" if kite_ltp is not None else "No real broker fill recorded (simulated fills excluded)")
            ),
        },
    }

    overall_status = "DISCREPANCY" if any(d["severity"] in ("WARNING", "CRITICAL") for d in discrepancies) else (
        "STALE" if spot_stale else ("PARTIAL" if missing_fields else ("DELAYED / EOD" if spot_delayed else "VERIFIED LIVE"))
    )

    return {
        "symbol": symbol,
        "overall_status": overall_status,
        "freshness": spot_freshness,
        "is_stale": spot_stale,
        "is_delayed": spot_delayed,
        "market_timestamp": market_ts,
        "market_timestamp_kind": market_kind,
        "retrieval_timestamp": retrieval_ts,
        "market_age_hours": mkt_age_h,
        "retrieval_age_hours": ret_age_h,
        "cache_hit": cache_hit,
        "missing_fields": missing_fields,
        "proxy_fields": proxy_fields,
        "discrepancies": discrepancies,
        "metrics": metrics,
    }


def build_data_integrity_report(
    assets: dict[str, dict[str, Any]],
    critiques: dict[str, dict[str, Any]],
    kite_holdings: list[dict[str, Any]] | None = None,
    kite_status_meta: dict[str, Any] | None = None,
    executed_orders: list[dict[str, Any]] | None = None,
    benchmark_meta: dict[str, Any] | None = None,
    ledger_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compile system-wide Data Integrity Monitor report and per-asset provenance."""
    now_utc = datetime.now(timezone.utc)
    kite_map = {str(h.get("symbol", "")).upper(): h for h in (kite_holdings or [])}
    per_asset: dict[str, dict[str, Any]] = {}
    all_discrepancies: list[dict[str, Any]] = []
    stale_symbols: list[str] = []
    delayed_symbols: list[str] = []
    partial_symbols: list[str] = []

    for sym, snap in assets.items():
        prov = build_asset_provenance(
            symbol=sym,
            analysis=snap,
            critique=critiques.get(sym),
            kite_holding=kite_map.get(sym.upper()),
            kite_status=kite_status_meta,
            executed_orders=executed_orders,
            now_utc=now_utc,
        )
        per_asset[sym] = prov
        if prov["is_stale"]:
            stale_symbols.append(sym)
        if prov["is_delayed"]:
            delayed_symbols.append(sym)
        if prov["missing_fields"]:
            partial_symbols.append(sym)
        for d in prov["discrepancies"]:
            all_discrepancies.append({"symbol": sym, **d})

    live_crit = sum(1 for c in critiques.values() if c.get("status") == "live")
    nse_mcp_symbols = [
        sym for sym, snap in assets.items()
        if str(snap.get("price_source_category") or "") == "NSE_OFFICIAL_MCP"
    ]
    yahoo_price_symbols = [
        sym for sym, snap in assets.items()
        if str(snap.get("price_source_category") or "YAHOO_LIVE_FEED") == "YAHOO_LIVE_FEED"
    ]
    bench_ok = bool(benchmark_meta and benchmark_meta.get("candles"))
    bench_mkt_ts = benchmark_meta.get("market_timestamp") if bench_ok else None
    bench_ret_ts = benchmark_meta.get("fetched_at") if bench_ok else None
    bench_src = benchmark_meta.get("data_source") or "Benchmark"

    feeds_summary = [
        {
            "category": SOURCE_TAXONOMY["NSE_OFFICIAL_MCP"]["label"],
            "code": "NSE_OFFICIAL_MCP",
            "status": "ACTIVE" if nse_mcp_symbols else "FALLBACK / OFF",
            "market_timestamp": next((per_asset[s]["market_timestamp"] for s in nse_mcp_symbols if per_asset[s]["market_timestamp"]), None),
            "retrieval_timestamp": next((per_asset[s]["retrieval_timestamp"] for s in nse_mcp_symbols if per_asset[s]["retrieval_timestamp"]), now_utc.isoformat()),
            "detail": (
                f"{len(nse_mcp_symbols)}/{len(assets)} symbols live via cm-market & bhavcopy"
                + (f" · Benchmark: {bench_src}" if benchmark_meta and benchmark_meta.get("source_category") == "NSE_OFFICIAL_MCP" else "")
            ),
        },
        {
            "category": SOURCE_TAXONOMY["YAHOO_LIVE_FEED"]["label"],
            "code": "YAHOO_LIVE_FEED",
            "status": "STALE" if stale_symbols else "ACTIVE",
            "market_timestamp": next((per_asset[s]["market_timestamp"] for s in yahoo_price_symbols if per_asset[s]["market_timestamp"]), next((p["market_timestamp"] for p in per_asset.values() if p["market_timestamp"]), None)),
            "retrieval_timestamp": next((p["retrieval_timestamp"] for p in per_asset.values() if p["retrieval_timestamp"]), now_utc.isoformat()),
            "detail": f"{len(yahoo_price_symbols)} global spot + company ratios · {len(stale_symbols)} stale · {len(partial_symbols)} partial",
        },
        {
            "category": SOURCE_TAXONOMY["HISTORICAL_MARKET_DATA"]["label"],
            "code": "HISTORICAL_MARKET_DATA",
            "status": "ACTIVE" if bench_ok else "DEGRADED",
            "market_timestamp": bench_mkt_ts,
            "retrieval_timestamp": bench_ret_ts,
            "detail": (
                f"1Y OHLCV bars + Nifty 50 ({len(benchmark_meta.get('candles', []))} bars · {bench_src})"
                if bench_ok
                else "Equity OHLCV bars active; Nifty 50 benchmark unavailable"
            ),
        },
        {
            "category": SOURCE_TAXONOMY["BROKER_API"]["label"],
            "code": "BROKER_API",
            "status": "CONNECTED" if (kite_status_meta or {}).get("connected") else "IDLE / READ-ONLY",
            "market_timestamp": (kite_status_meta or {}).get("fetched_at"),
            "retrieval_timestamp": (kite_status_meta or {}).get("fetched_at") or now_utc.isoformat(),
            "detail": (kite_status_meta or {}).get("status") or "Zerodha Kite Connect isolated to execution server",
        },
        {
            "category": SOURCE_TAXONOMY["CACHED_DATABASE"]["label"],
            "code": "CACHED_DATABASE",
            "status": "PERSISTENT",
            "market_timestamp": (ledger_meta or {}).get("last_market_timestamp"),
            "retrieval_timestamp": (ledger_meta or {}).get("updated_at") or now_utc.isoformat(),
            "detail": f"Audit Ledger: {(ledger_meta or {}).get('total_records', 0)} records · Cache hits: {sum(1 for p in per_asset.values() if p['cache_hit'])}",
        },
        {
            "category": SOURCE_TAXONOMY["INTERNAL_CALCULATION"]["label"],
            "code": "INTERNAL_CALCULATION",
            "status": "DETERMINISTIC",
            "market_timestamp": next((p["market_timestamp"] for p in per_asset.values() if p["market_timestamp"]), None),
            "retrieval_timestamp": now_utc.isoformat(),
            "detail": "StockSight V2, Porter's 5 Forces, MOVIQ Levels, Poly-Reg2, Backtester (ROCE proxied via ROE/ROA)",
        },
        {
            "category": SOURCE_TAXONOMY["AI_ANALYTICAL_ESTIMATE"]["label"],
            "code": "AI_ANALYTICAL_ESTIMATE",
            "status": "LIVE" if live_crit > 0 else "UNAVAILABLE",
            "market_timestamp": next((p["market_timestamp"] for p in per_asset.values() if p["market_timestamp"]), None),
            "retrieval_timestamp": next((c.get("generated_at") for c in critiques.values() if c.get("generated_at")), now_utc.isoformat()),
            "detail": f"Gemini CRO audited {live_crit}/{len(assets)} assets (analytical estimate only, never simulated fills)",
        },
    ]

    return {
        "generated_at": now_utc.isoformat(),
        "taxonomy": SOURCE_TAXONOMY,
        "price_classifications": PRICE_CLASSIFICATIONS,
        "summary": {
            "total_assets": len(assets),
            "nse_mcp_count": len(nse_mcp_symbols),
            "stale_count": len(stale_symbols),
            "delayed_count": len(delayed_symbols),
            "partial_fundamentals_count": len(partial_symbols),
            "discrepancy_count": len(all_discrepancies),
            "broker_connected": bool((kite_status_meta or {}).get("connected")),
            "gemini_live_count": live_crit,
        },
        "stale_symbols": stale_symbols,
        "delayed_symbols": delayed_symbols,
        "partial_symbols": partial_symbols,
        "discrepancies": all_discrepancies,
        "feeds": feeds_summary,
        "assets": per_asset,
    }
