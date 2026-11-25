"""Official NSE India MCP Client for Finsistant Version 2.

Connects to NSE's official Streamable-HTTP Model Context Protocol endpoints:
- CM Market Live (`cm-market-mcp`): https://mcp.nseindia.in/cmmkt/mcp
  Near real-time cash market stock quotes & live index quotes (1-3 min exchange delay).
- NSE Bhavcopy (`nse-bhavcopy-redis-mcp`): https://mcp.nseindia.in/bhavcopy/cm/mcp
  5-year daily EOD OHLCV equity history, 12-month index history, and corporate actions.

Scope Note:
- Official NSE MCP covers NSE-listed Indian equities & NSE indices only.
- It does NOT cover non-Indian equities (e.g. NVDA) or company-level fundamental ratios
  (P/E, ROE, OPM, Sales/Profit Growth). Those are explicitly attributed in provenance.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import http.cookiejar
import json
import threading
from typing import Any
import urllib.error
import urllib.request

CM_MARKET_MCP_URL = "https://mcp.nseindia.in/cmmkt/mcp"
BHAVCOPY_MCP_URL = "https://mcp.nseindia.in/bhavcopy/cm/mcp"

_IST = timezone(timedelta(hours=5, minutes=30))


def _parse_mcp_response(raw_body: str) -> dict[str, Any]:
    text = raw_body.strip()
    if text.startswith("{"):
        return json.loads(text)
    for line in text.splitlines():
        if line.startswith("data:"):
            part = line[5:].strip()
            if part:
                return json.loads(part)
    raise ValueError(f"Unrecognized NSE MCP payload: {text[:240]}")


def _parse_ist_timestamp(raw_ts: str | None) -> str | None:
    """Convert NSE IST timestamps ('2026-11-19 14:39:52' or '19-Nov-2026 14:39:53') to ISO-8601."""
    if not raw_ts:
        return None
    s = str(raw_ts).strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%d-%b-%Y %H:%M:%S", "%d-%m-%Y %H:%M:%S"):
        try:
            dt = datetime.strptime(s, fmt).replace(tzinfo=_IST)
            return dt.astimezone(timezone.utc).isoformat()
        except ValueError:
            continue
    return None


class _McpEndpointSession:
    """Thread-safe persistent Streamable-HTTP MCP client for one NSE endpoint."""

    def __init__(self, url: str, name: str) -> None:
        self.url = url
        self.name = name
        self._lock = threading.Lock()
        self._cookie_jar = http.cookiejar.CookieJar()
        self._opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(self._cookie_jar))
        self._session_id: str | None = None
        self._session_ts: float = 0.0
        self._circuit_open_until: float = 0.0
        self._circuit_reason: str = ""
        self._req_id = 0

    def _post(self, body: dict[str, Any], include_session: bool = True, timeout: float = 7.0) -> tuple[Any, dict[str, Any]]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "User-Agent": "Mozilla/5.0 (compatible; FinsistantV2-MCP/2.0)",
        }
        if include_session and self._session_id:
            headers["Mcp-Session-Id"] = self._session_id
        req = urllib.request.Request(
            self.url,
            data=json.dumps(body).encode("utf-8"),
            method="POST",
            headers=headers,
        )
        with self._opener.open(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", "replace")
            parsed = _parse_mcp_response(raw) if raw.strip() else {}
            return resp.headers, parsed

    def _ensure_session(self) -> None:
        now = datetime.now(timezone.utc).timestamp()
        if self._session_id and (now - self._session_ts) < 900.0:
            return
        self._cookie_jar.clear()
        self._req_id += 1
        hdrs, _ = self._post(
            {
                "jsonrpc": "2.0",
                "id": self._req_id,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "finsistant-v2", "version": "2.0.0"},
                },
            },
            include_session=False,
        )
        sid = hdrs.get("Mcp-Session-Id")
        if not sid:
            raise RuntimeError(f"NSE MCP '{self.name}' initialize did not return Mcp-Session-Id.")
        self._session_id = sid
        self._session_ts = now
        try:
            self._post({"jsonrpc": "2.0", "method": "notifications/initialized"}, include_session=True, timeout=4.0)
        except Exception:
            pass

    def call_tool(self, tool_name: str, arguments: dict[str, Any], timeout: float = 8.0) -> dict[str, Any]:
        import time as _time

        with self._lock:
            now_mono = _time.monotonic()
            if now_mono < self._circuit_open_until:
                raise RuntimeError(f"NSE MCP '{self.name}' cooldown active ({self._circuit_reason})")

            last_err: Exception | None = None
            max_attempts = 2
            transport_failure = False
            for attempt in range(max_attempts):
                try:
                    self._ensure_session()
                    self._req_id += 1
                    _, payload = self._post(
                        {
                            "jsonrpc": "2.0",
                            "id": self._req_id,
                            "method": "tools/call",
                            "params": {"name": tool_name, "arguments": arguments},
                        },
                        include_session=True,
                        timeout=timeout,
                    )
                    if "error" in payload:
                        raise RuntimeError(f"NSE MCP error: {payload['error']}")
                    res = payload.get("result") or {}
                    content = res.get("content") or []
                    txt = content[0].get("text", "") if content and isinstance(content[0], dict) else ""
                    data = json.loads(txt) if txt.strip().startswith("{") else {"raw": txt}
                    if res.get("isError"):
                        raise ValueError(f"NSE MCP tool error ({tool_name}): {data}")
                    self._circuit_open_until = 0.0
                    return data
                except ValueError as exc:
                    # Symbol/argument-level error from responding endpoint — do not trip endpoint circuit breaker
                    raise exc
                except (urllib.error.HTTPError, urllib.error.URLError, RuntimeError, TimeoutError, OSError) as exc:
                    last_err = exc
                    transport_failure = True
                    self._session_id = None
                    if attempt < max_attempts - 1:
                        _time.sleep(0.8)
                        continue
            if transport_failure:
                self._circuit_open_until = _time.monotonic() + 300.0
                self._circuit_reason = str(last_err)[:120]
            raise last_err or RuntimeError(f"NSE MCP tool call failed: {tool_name}")


_CM_CLIENT = _McpEndpointSession(CM_MARKET_MCP_URL, "cm-market-mcp")
_BHAV_CLIENT = _McpEndpointSession(BHAVCOPY_MCP_URL, "nse-bhavcopy-mcp")


def fetch_nse_stock_quote(nse_symbol: str) -> dict[str, Any] | None:
    """Fetch near-real-time quote for an NSE cash-market symbol via cm-market-mcp."""
    sym = nse_symbol.upper().strip()
    if sym.endswith(".NS"):
        sym = sym[:-3]
    data = _CM_CLIENT.call_tool("cm_get_stock_quote", {"symbol": sym})
    if not isinstance(data, dict) or data.get("error") or not data.get("stock"):
        return None
    stock = data["stock"]
    updated_at = data.get("updatedAt")
    mkt_ts = _parse_ist_timestamp(stock.get("latestTimestamp")) or updated_at
    return {
        "symbol": sym,
        "spot": float(stock.get("lastTradedPrice") or 0.0),
        "open": float(stock.get("openPrice") or 0.0),
        "high": float(stock.get("highPrice") or 0.0),
        "low": float(stock.get("lowPrice") or 0.0),
        "prev_close": float(stock.get("preClosePrice") or 0.0),
        "change_pct": round(float(stock.get("perChange") or 0.0), 2),
        "volume": int(stock.get("volume") or 0),
        "week52_high": float(stock.get("fiftyTwoWeekHigh") or 0.0),
        "week52_low": float(stock.get("fiftyTwoWeekLow") or 0.0),
        "market_timestamp": mkt_ts,
        "updated_at": updated_at,
        "raw_ist_timestamp": stock.get("latestTimestamp"),
    }


def fetch_nse_corporate_actions(nse_symbol: str, months_back: int = 13) -> list[dict[str, Any]]:
    sym = nse_symbol.upper().strip()
    if sym.endswith(".NS"):
        sym = sym[:-3]
    today = datetime.now(_IST).date()
    from_date = (today - timedelta(days=31 * months_back)).isoformat()
    try:
        res = _BHAV_CLIENT.call_tool(
            "get_corporate_actions",
            {"symbol": sym, "fromDate": from_date, "toDate": today.isoformat()},
        )
        return list(res.get("actions") or [])
    except Exception:
        return []


def fetch_nse_stock_history_1y(nse_symbol: str, target_bars: int = 248) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Fetch ~1Y daily OHLCV bars from nse-bhavcopy-mcp (4 x 3M chunks) + corporate action adjustment checks."""
    sym = nse_symbol.upper().strip()
    if sym.endswith(".NS"):
        sym = sym[:-3]

    by_date: dict[str, dict[str, Any]] = {}
    end_date = "today"
    for _ in range(5):
        chunk = _BHAV_CLIENT.call_tool(
            "get_stock_history",
            {"symbol": sym, "months": 3, "endDate": end_date},
        )
        rows = chunk.get("data") or []
        for row in rows:
            dt_str = str(row.get("date") or "")[:10]
            if not dt_str:
                continue
            o = row.get("open")
            h = row.get("high")
            l = row.get("low")
            c = row.get("close")
            if None in (o, h, l, c):
                continue
            by_date[dt_str] = {
                "date": dt_str,
                "open": round(float(o), 2),
                "high": round(float(h), 2),
                "low": round(float(l), 2),
                "close": round(float(c), 2),
                "volume": int(row.get("volume") or 0),
            }
        nxt = chunk.get("next_end_date")
        if len(by_date) >= target_bars or not nxt or nxt == end_date:
            break
        end_date = str(nxt)

    candles = [by_date[d] for d in sorted(by_date.keys())]
    if len(candles) < 20:
        raise RuntimeError(f"NSE Bhavcopy returned insufficient candles ({len(candles)}) for {sym}.")

    # Apply split/bonus adjustment factor from NSE corporate actions if any factor != 1.0
    actions = fetch_nse_corporate_actions(sym, months_back=13)
    applied_adjustments: list[dict[str, Any]] = []
    for act in actions:
        atype = str(act.get("actionType") or "").upper()
        ex_date = str(act.get("exDate") or "")[:10]
        factor = float(act.get("adjustmentFactor") or 1.0)
        if atype in ("SPLIT", "BONUS") and ex_date and factor > 0 and abs(factor - 1.0) > 1e-4:
            # Adjust bars strictly prior to exDate so historical prices align with post-split/bonus quotes
            adjusted_bars = 0
            for bar in candles:
                if bar["date"] < ex_date:
                    bar["open"] = round(bar["open"] / factor, 2)
                    bar["high"] = round(bar["high"] / factor, 2)
                    bar["low"] = round(bar["low"] / factor, 2)
                    bar["close"] = round(bar["close"] / factor, 2)
                    adjusted_bars += 1
            if adjusted_bars:
                applied_adjustments.append({
                    "exDate": ex_date,
                    "actionType": atype,
                    "factor": factor,
                    "purpose": act.get("purpose"),
                    "adjusted_bars": adjusted_bars,
                })

    return candles[-260:], applied_adjustments


def fetch_nse_index_1y(index_name: str = "Nifty 50") -> dict[str, Any]:
    """Fetch 12M daily OHLCV + live quote for an NSE index via official NSE MCP."""
    hist = _BHAV_CLIENT.call_tool(
        "get_index_history",
        {"indexName": index_name, "months": 12, "endDate": "today"},
    )
    rows = hist.get("data") or []
    candles: list[dict[str, Any]] = []
    for row in rows:
        dt_str = str(row.get("date") or "")[:10]
        o, h, l, c = row.get("open"), row.get("high"), row.get("low"), row.get("close")
        if not dt_str or None in (o, h, l, c):
            continue
        candles.append({
            "date": dt_str,
            "open": round(float(o), 2),
            "high": round(float(h), 2),
            "low": round(float(l), 2),
            "close": round(float(c), 2),
            "volume": int(row.get("volume") or 0),
        })
    if len(candles) < 20:
        raise RuntimeError(f"NSE Bhavcopy returned insufficient index candles ({len(candles)}) for {index_name}.")

    live_quote = None
    try:
        q = _CM_CLIENT.call_tool("cm_get_index_quote", {"indexName": index_name})
        if isinstance(q, dict) and q.get("index"):
            live_quote = q
    except Exception:
        live_quote = None

    mkt_ts = (
        _parse_ist_timestamp((live_quote or {}).get("nseTimestamp"))
        or (live_quote or {}).get("updatedAt")
        or f"{candles[-1]['date']}T10:00:00+00:00"
    )
    return {
        "symbol": "NIFTY50" if index_name.lower() == "nifty 50" else index_name.upper(),
        "index_name": index_name,
        "candles": candles,
        "live_quote": (live_quote or {}).get("index"),
        "market_timestamp": mkt_ts,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "data_source": "Official NSE MCP (bhavcopy + cm-market)",
    }
