"""Live Market Data & Zerodha Kite Connector for Finsistant Version 2.

Zero simulation policy:
- All OHLCV candles, spot prices, 1D changes, and fundamental metrics are fetched live
  from Yahoo Finance endpoints (no random walks, no synthetic fallback prices).
- Live portfolio holdings and cash margins are fetched from Zerodha Kite Connect v3
  when KITE_API_KEY and KITE_ACCESS_TOKEN are configured; otherwise returns 0 holdings.
"""

from __future__ import annotations

from datetime import datetime, timezone
import http.cookiejar
import json
import os
import time
from typing import Any
import urllib.error
import urllib.parse
import urllib.request

DEFAULT_SYMBOL_CONFIG: dict[str, dict[str, str]] = {
    "RELIANCE": {"yahoo": "RELIANCE.NS", "exchange": "NSE", "setup_type": "Institutional Breakout"},
    "TCS": {"yahoo": "TCS.NS", "exchange": "NSE", "setup_type": "Moat Accumulation"},
    "HDFCBANK": {"yahoo": "HDFCBANK.NS", "exchange": "NSE", "setup_type": "High Quality Value"},
    "ICICIBANK": {"yahoo": "ICICIBANK.NS", "exchange": "NSE", "setup_type": "Momentum Leader"},
    "INFY": {"yahoo": "INFY.NS", "exchange": "NSE", "setup_type": "Channel Breakout"},
    "TATAMOTORS": {"yahoo": "TMCV.NS", "exchange": "NSE", "setup_type": "Deep Value Turnaround"},
    "NVDA": {"yahoo": "NVDA", "exchange": "NASDAQ", "setup_type": "AI Momentum Continuation"},
    "ZEEL": {"yahoo": "ZEEL.NS", "exchange": "NSE", "setup_type": "Risk Breakdown Warning"},
    "ITC": {"yahoo": "ITC.NS", "exchange": "NSE", "setup_type": "Dividend Compounder"},
    "SBIN": {"yahoo": "SBIN.NS", "exchange": "NSE", "setup_type": "PSU Banking Leader"},
}

_UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
_COOKIE_JAR = http.cookiejar.CookieJar()
_OPENER = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(_COOKIE_JAR))
_OPENER.addheaders = [("User-Agent", _UA)]
_CACHED_CRUMB: str | None = None
_CRUMB_TS: float = 0.0
_LIVE_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}
_CACHE_TTL_SEC = 60.0


def _get_yahoo_crumb() -> str:
    global _CACHED_CRUMB, _CRUMB_TS
    if _CACHED_CRUMB and (time.time() - _CRUMB_TS) < 1800:
        return _CACHED_CRUMB
    try:
        _OPENER.open("https://fc.yahoo.com", timeout=8)
    except Exception:
        pass
    with _OPENER.open("https://query1.finance.yahoo.com/v1/test/getcrumb", timeout=8) as resp:
        crumb = resp.read().decode("utf-8").strip()
    if not crumb:
        raise RuntimeError("Failed to obtain Yahoo Finance session crumb.")
    _CACHED_CRUMB = crumb
    _CRUMB_TS = time.time()
    return crumb


def _fetch_json(url: str, retries: int = 3) -> dict[str, Any]:
    last_err: Exception | None = None
    hosts = ["query1.finance.yahoo.com", "query2.finance.yahoo.com"]
    for attempt in range(retries):
        target = url
        if attempt % 2 == 1 and "query1.finance.yahoo.com" in target:
            target = target.replace("query1.finance.yahoo.com", hosts[1])
        elif attempt % 2 == 1 and "query2.finance.yahoo.com" in target:
            target = target.replace("query2.finance.yahoo.com", hosts[0])
        try:
            with _OPENER.open(target, timeout=12) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            last_err = exc
            if exc.code == 429:
                time.sleep(1.5 * (attempt + 1))
                continue
            raise
        except Exception as exc:
            last_err = exc
            time.sleep(1.0 * (attempt + 1))
    raise last_err or RuntimeError(f"Failed fetching {url}")


def _fetch_via_yfinance(sym: str, yahoo_sym: str, exchange_hint: str) -> dict[str, Any] | None:
    """Use yfinance (with curl_cffi browser TLS fingerprint) if installed."""
    try:
        import yfinance as yf  # type: ignore
    except ImportError:
        return None

    ticker = yf.Ticker(yahoo_sym)
    hist = ticker.history(period="1y", interval="1d", auto_adjust=False)
    if hist is None or hist.empty:
        return None

    candles: list[dict[str, Any]] = []
    for idx, row in hist.iterrows():
        dt_str = idx.date().isoformat() if hasattr(idx, "date") else str(idx)[:10]
        o, h, l, c = row.get("Open"), row.get("High"), row.get("Low"), row.get("Close")
        if any(v is None or v != v for v in (o, h, l, c)):
            continue
        candles.append(
            {
                "date": dt_str,
                "open": round(float(o), 2),
                "high": round(float(h), 2),
                "low": round(float(l), 2),
                "close": round(float(c), 2),
                "volume": int(row.get("Volume") or 0),
            }
        )
    if len(candles) < 20:
        return None

    info = {}
    try:
        info = ticker.info or {}
    except Exception:
        info = {}

    spot = float(info.get("regularMarketPrice") or info.get("currentPrice") or candles[-1]["close"])
    prev_bar = float(candles[-2]["close"]) if len(candles) >= 2 else spot
    change_pct = round(((spot - prev_bar) / max(1e-6, prev_bar)) * 100.0, 2)
    ret_1y = round(((spot - float(candles[0]["close"])) / max(1e-6, float(candles[0]["close"]))) * 100.0, 2)

    currency = str(info.get("currency") or ("INR" if yahoo_sym.endswith(".NS") else "USD"))
    exchange = str(info.get("exchange") or exchange_hint)
    if exchange == "NSI":
        exchange = "NSE"

    pe = float(info.get("trailingPE") or info.get("forwardPE") or 0.0)
    roe_raw = info.get("returnOnEquity")
    roa_raw = info.get("returnOnAssets")
    roe = round(float(roe_raw) * 100.0, 2) if roe_raw is not None else 0.0
    roce = round(((float(roe_raw or 0.0) * 0.7) + (float(roa_raw or 0.0) * 0.3)) * 100.0, 2) if (roe_raw is not None or roa_raw is not None) else roe
    opm = round(float(info.get("operatingMargins") or 0.0) * 100.0, 2)
    sales_growth = round(float(info.get("revenueGrowth") or 0.0) * 100.0, 2)
    profit_growth = round(float(info.get("earningsGrowth") or info.get("earningsQuarterlyGrowth") or 0.0) * 100.0, 2)
    mcap_raw = float(info.get("marketCap") or 0.0)
    mcap_cr = round(mcap_raw / 1e7, 2) if currency == "INR" else round(mcap_raw / 1e6, 2)
    beta = round(float(info.get("beta") or 1.0), 2)
    de_raw = info.get("debtToEquity")
    debt_to_equity = round(float(de_raw) / 100.0, 2) if de_raw is not None else 0.0

    return {
        "symbol": sym,
        "yahoo_symbol": yahoo_sym,
        "name": str(info.get("longName") or info.get("shortName") or sym),
        "exchange": exchange,
        "currency": currency,
        "sector": str(info.get("sector") or info.get("industry") or "Equity"),
        "asset_type": "STOCK",
        "setup_type": DEFAULT_SYMBOL_CONFIG.get(sym, {}).get("setup_type", "Live Market Scan"),
        "price": round(spot, 2),
        "change_pct": change_pct,
        "pe": round(pe, 2),
        "roe": roe,
        "roce": roce,
        "opm": opm,
        "sales_growth": sales_growth,
        "profit_growth": profit_growth,
        "mcap_cr": mcap_cr,
        "beta": beta,
        "ret_1y": ret_1y,
        "debt_to_equity": debt_to_equity,
        "candles": candles,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "data_source": "Yahoo Finance Live (yfinance)",
    }


def resolve_yahoo_symbol(symbol: str) -> tuple[str, str]:
    sym = symbol.upper().strip()
    if sym in DEFAULT_SYMBOL_CONFIG:
        cfg = DEFAULT_SYMBOL_CONFIG[sym]
        return cfg["yahoo"], cfg["exchange"]
    if "." in sym:
        ex = "NSE" if sym.endswith(".NS") else ("BSE" if sym.endswith(".BO") else "GLOBAL")
        return sym, ex
    # Try NSE first, then raw symbol (e.g., US tickers like AAPL, MSFT)
    for candidate, ex in ((f"{sym}.NS", "NSE"), (sym, "GLOBAL")):
        try:
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{urllib.parse.quote(candidate)}?range=5d&interval=1d"
            data = _fetch_json(url, retries=2)
            res = data.get("chart", {}).get("result")
            if res and res[0].get("meta", {}).get("regularMarketPrice") is not None:
                return candidate, ex
        except Exception:
            continue
    raise RuntimeError(f"Symbol '{sym}' could not be resolved on live Yahoo Finance market feed.")


def _raw_num(val: Any) -> float | None:
    if isinstance(val, dict):
        v = val.get("raw")
        return float(v) if v is not None else None
    if isinstance(val, (int, float)):
        return float(val)
    return None


def fetch_live_market_snapshot(symbol: str, force_refresh: bool = False) -> dict[str, Any]:
    """Fetch real OHLCV bars and live fundamentals from Yahoo Finance. Zero simulation."""
    sym = symbol.upper().strip()
    now = time.time()
    if not force_refresh and sym in _LIVE_CACHE:
        ts, cached = _LIVE_CACHE[sym]
        if (now - ts) < _CACHE_TTL_SEC:
            return cached

    yahoo_sym, exchange_hint = resolve_yahoo_symbol(sym)

    # Prefer yfinance if installed (handles Yahoo session/TLS cleanly on cloud runners)
    try:
        yf_snap = _fetch_via_yfinance(sym, yahoo_sym, exchange_hint)
        if yf_snap:
            _LIVE_CACHE[sym] = (now, yf_snap)
            return yf_snap
    except Exception:
        pass

    time.sleep(0.25)
    chart_url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{urllib.parse.quote(yahoo_sym)}"
        "?range=1y&interval=1d"
    )
    chart_json = _fetch_json(chart_url)
    chart_res = (chart_json.get("chart", {}) or {}).get("result") or []
    if not chart_res:
        raise RuntimeError(f"Live chart data unavailable for {sym} ({yahoo_sym}).")

    result0 = chart_res[0]
    meta = result0.get("meta") or {}
    timestamps = result0.get("timestamp") or []
    quote0 = ((result0.get("indicators") or {}).get("quote") or [{}])[0]
    opens = quote0.get("open") or []
    highs = quote0.get("high") or []
    lows = quote0.get("low") or []
    closes = quote0.get("close") or []
    volumes = quote0.get("volume") or []

    candles: list[dict[str, Any]] = []
    for i, ts_val in enumerate(timestamps):
        o = opens[i] if i < len(opens) else None
        h = highs[i] if i < len(highs) else None
        l = lows[i] if i < len(lows) else None
        c = closes[i] if i < len(closes) else None
        v = volumes[i] if i < len(volumes) else 0
        if None in (o, h, l, c):
            continue
        dt_str = datetime.fromtimestamp(int(ts_val), tz=timezone.utc).date().isoformat()
        candles.append(
            {
                "date": dt_str,
                "open": round(float(o), 2),
                "high": round(float(h), 2),
                "low": round(float(l), 2),
                "close": round(float(c), 2),
                "volume": int(v or 0),
            }
        )

    if len(candles) < 20:
        raise RuntimeError(f"Insufficient live OHLCV candles ({len(candles)}) returned for {sym} ({yahoo_sym}).")

    spot = float(meta.get("regularMarketPrice") or candles[-1]["close"])
    prev_close = float(meta.get("chartPreviousClose") or meta.get("previousClose") or candles[-2]["close"])
    # Prefer 1D delta between last two bars or regularMarketChangePercent
    prev_bar_close = float(candles[-2]["close"]) if len(candles) >= 2 else prev_close
    change_pct = round(((spot - prev_bar_close) / max(1e-6, prev_bar_close)) * 100.0, 2)
    ret_1y = round(((spot - float(candles[0]["close"])) / max(1e-6, float(candles[0]["close"]))) * 100.0, 2)

    # Fetch live fundamentals via quoteSummary
    crumb = _get_yahoo_crumb()
    qs_url = (
        f"https://query2.finance.yahoo.com/v10/finance/quoteSummary/{urllib.parse.quote(yahoo_sym)}"
        f"?modules=price,summaryDetail,defaultKeyStatistics,financialData,assetProfile&crumb={urllib.parse.quote(crumb)}"
    )
    qs_json = _fetch_json(qs_url)
    qs_res = ((qs_json.get("quoteSummary") or {}).get("result") or [{}])[0]

    price_mod = qs_res.get("price") or {}
    summary_mod = qs_res.get("summaryDetail") or {}
    stats_mod = qs_res.get("defaultKeyStatistics") or {}
    fin_mod = qs_res.get("financialData") or {}
    prof_mod = qs_res.get("assetProfile") or {}

    if _raw_num(price_mod.get("regularMarketChangePercent")) is not None:
        change_pct = round(float(_raw_num(price_mod.get("regularMarketChangePercent")) or 0.0) * 100.0, 2)

    currency = str(meta.get("currency") or price_mod.get("currency") or ("INR" if yahoo_sym.endswith(".NS") else "USD"))
    exchange = str(meta.get("fullExchangeName") or meta.get("exchangeName") or exchange_hint)
    if exchange == "NSI":
        exchange = "NSE"

    long_name = str(price_mod.get("longName") or price_mod.get("shortName") or meta.get("longName") or meta.get("shortName") or sym)
    sector = str(prof_mod.get("sector") or prof_mod.get("industry") or "Equity")

    pe = _raw_num(summary_mod.get("trailingPE")) or _raw_num(stats_mod.get("forwardPE")) or 0.0
    roe_raw = _raw_num(fin_mod.get("returnOnEquity"))
    roa_raw = _raw_num(fin_mod.get("returnOnAssets"))
    roe = round(roe_raw * 100.0, 2) if roe_raw is not None else 0.0
    # Yahoo Finance does not directly publish ROCE; use ROE/ROA blend when ROA is available
    roce = round(((roe_raw or 0.0) * 0.7 + (roa_raw or 0.0) * 0.3) * 100.0, 2) if (roe_raw is not None or roa_raw is not None) else roe
    opm_raw = _raw_num(fin_mod.get("operatingMargins"))
    opm = round(opm_raw * 100.0, 2) if opm_raw is not None else 0.0
    rev_g_raw = _raw_num(fin_mod.get("revenueGrowth"))
    sales_growth = round(rev_g_raw * 100.0, 2) if rev_g_raw is not None else 0.0
    earn_g_raw = _raw_num(fin_mod.get("earningsGrowth")) or _raw_num(fin_mod.get("earningsQuarterlyGrowth"))
    profit_growth = round(earn_g_raw * 100.0, 2) if earn_g_raw is not None else 0.0

    mcap_raw = _raw_num(summary_mod.get("marketCap")) or _raw_num(price_mod.get("marketCap"))
    if mcap_raw is None:
        shares = _raw_num(stats_mod.get("sharesOutstanding")) or _raw_num(stats_mod.get("impliedSharesOutstanding"))
        if shares:
            mcap_raw = shares * spot
    mcap_cr = round((mcap_raw or 0.0) / 1e7, 2) if currency == "INR" else round((mcap_raw or 0.0) / 1e6, 2)

    beta = round(_raw_num(stats_mod.get("beta")) or _raw_num(summary_mod.get("beta")) or 1.0, 2)
    de_raw = _raw_num(fin_mod.get("debtToEquity"))
    debt_to_equity = round((de_raw / 100.0), 2) if de_raw is not None else 0.0

    setup_type = DEFAULT_SYMBOL_CONFIG.get(sym, {}).get("setup_type", "Live Market Scan")

    snapshot = {
        "symbol": sym,
        "yahoo_symbol": yahoo_sym,
        "name": long_name,
        "exchange": exchange,
        "currency": currency,
        "sector": sector,
        "asset_type": "STOCK",
        "setup_type": setup_type,
        "price": round(spot, 2),
        "change_pct": change_pct,
        "pe": round(pe, 2),
        "roe": roe,
        "roce": roce,
        "opm": opm,
        "sales_growth": sales_growth,
        "profit_growth": profit_growth,
        "mcap_cr": mcap_cr,
        "beta": beta,
        "ret_1y": ret_1y,
        "debt_to_equity": debt_to_equity,
        "candles": candles,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "data_source": "Yahoo Finance Live",
    }
    _LIVE_CACHE[sym] = (now, snapshot)
    return snapshot


def fetch_kite_portfolio(force_token_refresh: bool = False) -> dict[str, Any]:
    """Fetch live holdings and cash margins from Zerodha Kite Connect v3.

    Uses Finsistant's daily Kite token resolver (`KITE_ACCESS_TOKEN`, daily `token.txt`,
    `KITE_REQUEST_TOKEN` exchange, or automated TOTP login when credentials exist).
    Never returns simulated/fake holdings.
    """
    from .kite_auth import clear_daily_token_file, resolve_kite_session

    session = resolve_kite_session(force_refresh=force_token_refresh)
    if not session["ready"]:
        return {
            "connected": False,
            "status": session["message"],
            "token_source": session["source"],
            "holdings": [],
            "cash_balance": 0.0,
        }

    def _query_kite(api_key: str, access_token: str) -> tuple[list[dict[str, Any]], float]:
        headers = {
            "X-Kite-Version": "3",
            "Authorization": f"token {api_key}:{access_token}",
        }
        req_h = urllib.request.Request("https://api.kite.trade/portfolio/holdings", headers=headers)
        with urllib.request.urlopen(req_h, timeout=10) as resp:
            h_payload = json.loads(resp.read().decode("utf-8"))
        raw_holdings = h_payload.get("data") or []

        cash_balance = 0.0
        try:
            req_m = urllib.request.Request("https://api.kite.trade/user/margins", headers=headers)
            with urllib.request.urlopen(req_m, timeout=10) as resp_m:
                m_payload = json.loads(resp_m.read().decode("utf-8"))
                eq = (m_payload.get("data") or {}).get("equity") or {}
                cash_balance = float((eq.get("available") or {}).get("cash") or 0.0)
        except Exception:
            cash_balance = 0.0

        holdings: list[dict[str, Any]] = []
        for item in raw_holdings:
            qty = int(item.get("quantity") or 0) + int(item.get("t1_quantity") or 0)
            if qty <= 0:
                continue
            holdings.append(
                {
                    "symbol": str(item.get("tradingsymbol") or "").upper(),
                    "qty": qty,
                    "avg_price": float(item.get("average_price") or 0.0),
                    "ltp": float(item.get("last_price") or 0.0),
                    "pnl": float(item.get("pnl") or 0.0),
                }
            )
        return holdings, round(cash_balance, 2)

    try:
        try:
            holdings, cash_balance = _query_kite(session["api_key"], session["access_token"])
        except urllib.error.HTTPError as http_exc:
            if http_exc.code in (401, 403) and not force_token_refresh:
                clear_daily_token_file()
                refreshed = resolve_kite_session(force_refresh=True)
                if refreshed["ready"]:
                    session = refreshed
                    holdings, cash_balance = _query_kite(session["api_key"], session["access_token"])
                else:
                    return {
                        "connected": False,
                        "status": f"Kite session expired ({refreshed['message']})",
                        "token_source": refreshed["source"],
                        "holdings": [],
                        "cash_balance": 0.0,
                    }
            else:
                raise
        return {
            "connected": True,
            "status": f"Zerodha Kite Live ({len(holdings)} holdings · {session['source']})",
            "token_source": session["source"],
            "holdings": holdings,
            "cash_balance": cash_balance,
        }
    except Exception as exc:
        return {
            "connected": False,
            "status": f"Kite API error ({exc}) — 0 holdings loaded",
            "token_source": session.get("source", "error"),
            "holdings": [],
            "cash_balance": 0.0,
        }
