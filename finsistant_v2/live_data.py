"""Live Market Data, Official NSE MCP & Zerodha Kite Connector for Finsistant Version 2.

Zero simulation policy:
- For Indian NSE equities & Nifty 50 benchmark, primary spot quotes (1-3 min delayed exchange
  feed) and 1-year daily OHLCV bars are fetched directly from Official NSE India MCP servers
  (`cm-market-mcp` and `nse-bhavcopy-redis-mcp`).
- Company-level fundamental ratios (P/E, ROE, OPM, Sales/Profit Growth, Market Cap, Debt/Equity)
  and non-NSE global equities (e.g., NVDA) are fetched from Yahoo Finance (and fallback if
  NSE MCP is unavailable). No random walks or synthetic fallback prices.
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

USE_NSE_MCP = os.environ.get("FINSISTANT_USE_NSE_MCP", "1").strip().lower() not in ("0", "false", "no")

DEFAULT_SYMBOL_CONFIG: dict[str, dict[str, str]] = {
    "RELIANCE": {"yahoo": "RELIANCE.NS", "nse": "RELIANCE", "exchange": "NSE", "setup_type": "Institutional Breakout"},
    "TCS": {"yahoo": "TCS.NS", "nse": "TCS", "exchange": "NSE", "setup_type": "Moat Accumulation"},
    "HDFCBANK": {"yahoo": "HDFCBANK.NS", "nse": "HDFCBANK", "exchange": "NSE", "setup_type": "High Quality Value"},
    "ICICIBANK": {"yahoo": "ICICIBANK.NS", "nse": "ICICIBANK", "exchange": "NSE", "setup_type": "Momentum Leader"},
    "INFY": {"yahoo": "INFY.NS", "nse": "INFY", "exchange": "NSE", "setup_type": "Channel Breakout"},
    "TATAMOTORS": {"yahoo": "TMCV.NS", "nse": "TMCV", "exchange": "NSE", "setup_type": "Deep Value Turnaround"},
    "NVDA": {"yahoo": "NVDA", "nse": "", "exchange": "NASDAQ", "setup_type": "AI Momentum Continuation"},
    "ZEEL": {"yahoo": "ZEEL.NS", "nse": "ZEEL", "exchange": "NSE", "setup_type": "Risk Breakdown Warning"},
    "ITC": {"yahoo": "ITC.NS", "nse": "ITC", "exchange": "NSE", "setup_type": "Dividend Compounder"},
    "SBIN": {"yahoo": "SBIN.NS", "nse": "SBIN", "exchange": "NSE", "setup_type": "PSU Banking Leader"},
}

_UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
_COOKIE_JAR = http.cookiejar.CookieJar()
_OPENER = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(_COOKIE_JAR))
_OPENER.addheaders = [("User-Agent", _UA)]
_CACHED_CRUMB: str | None = None
_CRUMB_TS: float = 0.0
_LIVE_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}
_CACHE_TTL_SEC = 60.0


def _get_yahoo_crumb(force_refresh: bool = False) -> str:
    global _CACHED_CRUMB, _CRUMB_TS
    if (not force_refresh) and _CACHED_CRUMB and (time.time() - _CRUMB_TS) < 1800:
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
    for attempt in range(4):
        try:
            info = ticker.info or {}
            if info and (info.get("marketCap") or info.get("trailingPE") or info.get("returnOnEquity")):
                break
        except Exception:
            info = {}
        if attempt < 3:
            time.sleep(1.5 * (attempt + 1))

    if not info:
        raise RuntimeError(f"Live Yahoo info/fundamentals unavailable for {yahoo_sym} after retries.")

    spot = float(info.get("regularMarketPrice") or info.get("currentPrice") or candles[-1]["close"])
    prev_bar = float(candles[-2]["close"]) if len(candles) >= 2 else spot
    change_pct = round(((spot - prev_bar) / max(1e-6, prev_bar)) * 100.0, 2)
    ret_1y = round(((spot - float(candles[0]["close"])) / max(1e-6, float(candles[0]["close"]))) * 100.0, 2)

    currency = str(info.get("currency") or ("INR" if yahoo_sym.endswith(".NS") else "USD"))
    exchange = str(info.get("exchange") or exchange_hint)
    if exchange == "NSI":
        exchange = "NSE"

    raw_mkt_ts = info.get("regularMarketTime")
    last_bar_date = candles[-1]["date"]
    if isinstance(raw_mkt_ts, (int, float)) and raw_mkt_ts > 0:
        market_timestamp = datetime.fromtimestamp(float(raw_mkt_ts), tz=timezone.utc).isoformat()
        market_ts_kind = "exchange_quote"
    else:
        close_suffix = "T10:00:00+00:00" if yahoo_sym.endswith((".NS", ".BO")) or yahoo_sym == "^NSEI" else "T21:00:00+00:00"
        market_timestamp = f"{last_bar_date}{close_suffix}"
        market_ts_kind = "end_of_day_bar"

    fund = _build_merged_fundamentals(
        info,
        yahoo_sym=yahoo_sym,
        currency=currency,
        spot=spot,
        nse_sym=sym if exchange == "NSE" or yahoo_sym.endswith(".NS") else None,
        yahoo_spot=round(spot, 2),
        strict=False,
    )

    return {
        "symbol": sym,
        "yahoo_symbol": yahoo_sym,
        "name": fund["name"],
        "exchange": exchange,
        "currency": currency,
        "sector": fund["sector"],
        "asset_type": "STOCK",
        "setup_type": DEFAULT_SYMBOL_CONFIG.get(sym, {}).get("setup_type", "Live Market Scan"),
        "price": round(spot, 2),
        "last_candle_date": last_bar_date,
        "last_candle_close": round(float(candles[-1]["close"]), 2),
        "change_pct": change_pct,
        "pe": fund["pe"],
        "roe": fund["roe"],
        "roce": fund["roce"],
        "opm": fund["opm"],
        "sales_growth": fund["sales_growth"],
        "profit_growth": fund["profit_growth"],
        "mcap_cr": fund["mcap_cr"],
        "beta": fund["beta"],
        "ret_1y": ret_1y,
        "debt_to_equity": fund["debt_to_equity"],
        "candles": candles,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "market_timestamp": market_timestamp,
        "market_timestamp_kind": market_ts_kind,
        "fundamentals_status": fund["fundamentals_status"],
        "fundamental_source_category": fund["fundamental_source_category"],
        "missing_fields": fund["missing_fields"],
        "proxy_fields": fund["proxy_fields"],
        "cache_hit": False,
        "cache_age_seconds": 0.0,
        "data_source": f"Yahoo Finance Live + {fund['fundamental_source_category']}",
    }


def resolve_yahoo_symbol(symbol: str) -> tuple[str, str]:
    sym = symbol.upper().strip()
    if sym in DEFAULT_SYMBOL_CONFIG:
        cfg = DEFAULT_SYMBOL_CONFIG[sym]
        return cfg["yahoo"], cfg["exchange"]
    if "." in sym:
        ex = "NSE" if sym.endswith(".NS") else ("BSE" if sym.endswith(".BO") else "GLOBAL")
        return sym, ex

    try:
        import yfinance as yf  # type: ignore

        for candidate, ex in ((f"{sym}.NS", "NSE"), (sym, "GLOBAL")):
            try:
                hist = yf.Ticker(candidate).history(period="5d", interval="1d", auto_adjust=False)
                if hist is not None and not hist.empty:
                    return candidate, ex
            except Exception:
                continue
    except ImportError:
        pass

    # Stdlib fallback: Try NSE first, then raw symbol (e.g., US tickers like AAPL, MSFT)
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


def _build_merged_fundamentals(
    info: dict[str, Any],
    *,
    yahoo_sym: str,
    currency: str,
    spot: float,
    nse_sym: str | None = None,
    yahoo_spot: float | None = None,
    strict: bool = False,
) -> dict[str, Any]:
    """Build fundamental ratios using Yahoo first and Screener.in when Yahoo lacks data. Never coerces missing ratios to 0.0."""
    proxy_fields: dict[str, str] = {}

    raw_pe = info.get("trailingPE") or info.get("forwardPE") if info else None
    pe: float | None = round(float(raw_pe), 2) if raw_pe is not None else None

    roe_raw = info.get("returnOnEquity") if info else None
    roa_raw = info.get("returnOnAssets") if info else None
    roe: float | None = round(float(roe_raw) * 100.0, 2) if roe_raw is not None else None
    roce: float | None = None
    if roe_raw is not None or roa_raw is not None:
        roce = round(((float(roe_raw or 0.0) * 0.7) + (float(roa_raw or 0.0) * 0.3)) * 100.0, 2)
        proxy_fields["roce"] = "Derived internally via 0.7*ROE + 0.3*ROA blend (ROCE not published natively by Yahoo)"

    opm_raw = info.get("operatingMargins") if info else None
    opm: float | None = round(float(opm_raw) * 100.0, 2) if opm_raw is not None else None

    rev_g_raw = info.get("revenueGrowth") if info else None
    sales_growth: float | None = round(float(rev_g_raw) * 100.0, 2) if rev_g_raw is not None else None

    earn_g_raw = (info.get("earningsGrowth") or info.get("earningsQuarterlyGrowth")) if info else None
    profit_growth: float | None = round(float(earn_g_raw) * 100.0, 2) if earn_g_raw is not None else None

    mcap_raw = float(info.get("marketCap") or 0.0) if info else 0.0
    if info and mcap_raw <= 0 and info.get("sharesOutstanding") and spot > 0:
        mcap_raw = float(info["sharesOutstanding"]) * spot
    mcap_cr: float | None = (
        (round(mcap_raw / 1e7, 2) if currency == "INR" else round(mcap_raw / 1e6, 2))
        if mcap_raw > 0
        else None
    )

    beta_raw = info.get("beta") if info else None
    beta: float | None = round(float(beta_raw), 2) if beta_raw is not None else None

    de_raw = info.get("debtToEquity") if info else None
    debt_to_equity: float | None = round(float(de_raw) / 100.0, 2) if de_raw is not None else None

    name = str((info or {}).get("longName") or (info or {}).get("shortName") or (nse_sym or yahoo_sym.replace(".NS", "")))
    sector_val = (info or {}).get("sector") or (info or {}).get("industry")
    sector = str(sector_val) if sector_val else ""

    used_yahoo = any(v is not None for v in (pe, roe, opm, sales_growth, profit_growth, mcap_cr))
    used_screener = False

    is_nse_stock = bool(nse_sym or yahoo_sym.endswith((".NS", ".BO")))
    needs_screener = is_nse_stock and any(
        v is None for v in (pe, roe, roce, opm, sales_growth, profit_growth, mcap_cr, debt_to_equity)
    )
    if needs_screener:
        scr_sym = nse_sym or yahoo_sym.replace(".NS", "").replace(".BO", "")
        try:
            from .screener_data import fetch_screener_fundamentals

            scr = fetch_screener_fundamentals(scr_sym)
            if pe is None and scr.get("pe") is not None:
                pe = scr["pe"]
                used_screener = True
            if roe is None and scr.get("roe") is not None:
                roe = scr["roe"]
                used_screener = True
            if scr.get("roce") is not None and (roce is None or "roce" in proxy_fields):
                roce = scr["roce"]
                proxy_fields.pop("roce", None)
                used_screener = True
            if opm is None and scr.get("opm") is not None:
                opm = scr["opm"]
                used_screener = True
            if sales_growth is None and scr.get("sales_growth") is not None:
                sales_growth = scr["sales_growth"]
                used_screener = True
            if profit_growth is None and scr.get("profit_growth") is not None:
                profit_growth = scr["profit_growth"]
                used_screener = True
            if mcap_cr is None and scr.get("mcap_cr") is not None:
                mcap_cr = scr["mcap_cr"]
                used_screener = True
            if debt_to_equity is None and scr.get("debt_to_equity") is not None:
                debt_to_equity = scr["debt_to_equity"]
                used_screener = True
            if (not sector) and scr.get("sector"):
                sector = str(scr["sector"])
            if (name == scr_sym) and scr.get("name"):
                name = str(scr["name"])
        except Exception as exc:
            if not used_yahoo and strict:
                raise RuntimeError(
                    f"Both Yahoo Finance and Screener.in failed for {scr_sym}: {exc}"
                ) from exc

    vals = {
        "pe": pe,
        "roe": roe,
        "roce": roce,
        "opm": opm,
        "sales_growth": sales_growth,
        "profit_growth": profit_growth,
        "mcap_cr": mcap_cr,
        "debt_to_equity": debt_to_equity,
    }
    missing_fields = [k for k, v in vals.items() if v is None]
    if beta is None:
        missing_fields.append("beta")

    has_any = any(v is not None for v in vals.values())
    if not has_any:
        if strict:
            raise RuntimeError(f"Fundamental data unavailable from both Yahoo and Screener.in for {yahoo_sym}.")
        fund_cat = "UNAVAILABLE"
        status = "unavailable"
    else:
        status = "live"
        if used_yahoo and used_screener:
            fund_cat = "YAHOO_LIVE_FEED+SCREENER_IN"
        elif used_screener:
            fund_cat = "SCREENER_IN"
        else:
            fund_cat = "YAHOO_LIVE_FEED"

    return {
        "fundamentals_status": status,
        "fundamental_source_category": fund_cat,
        "name": name,
        "sector": sector or "NSE Cash Equity",
        "pe": pe,
        "roe": roe,
        "roce": roce,
        "opm": opm,
        "sales_growth": sales_growth,
        "profit_growth": profit_growth,
        "mcap_cr": mcap_cr,
        "beta": beta,
        "debt_to_equity": debt_to_equity,
        "yahoo_spot": yahoo_spot,
        "missing_fields": missing_fields,
        "proxy_fields": proxy_fields,
    }


def _fetch_yahoo_fundamentals_only(
    yahoo_sym: str,
    currency: str,
    spot: float,
    *,
    strict: bool = False,
    nse_sym: str | None = None,
) -> dict[str, Any]:
    """Fetch company-level fundamental ratios (Yahoo primary, Screener.in when Yahoo is unavailable)."""
    info: dict[str, Any] = {}
    yahoo_spot: float | None = None

    max_attempts = 4
    for attempt in range(max_attempts):
        try:
            import yfinance as yf  # type: ignore

            t_info = yf.Ticker(yahoo_sym).info or {}
            if t_info and (t_info.get("marketCap") or t_info.get("trailingPE") or t_info.get("returnOnEquity")):
                info = t_info
                raw_sp = info.get("regularMarketPrice") or info.get("currentPrice")
                if raw_sp is not None:
                    yahoo_spot = round(float(raw_sp), 2)
                break
        except Exception:
            pass

        for host in ("query2.finance.yahoo.com", "query1.finance.yahoo.com"):
            try:
                crumb = _get_yahoo_crumb(force_refresh=(attempt > 0))
                qs_url = (
                    f"https://{host}/v10/finance/quoteSummary/{urllib.parse.quote(yahoo_sym)}"
                    f"?modules=price,summaryDetail,defaultKeyStatistics,financialData,assetProfile&crumb={urllib.parse.quote(crumb)}"
                )
                qs_json = _fetch_json(qs_url, retries=2)
                qs_res = ((qs_json.get("quoteSummary") or {}).get("result") or [{}])[0]
                if not qs_res:
                    continue
                price_mod = qs_res.get("price") or {}
                summary_mod = qs_res.get("summaryDetail") or {}
                stats_mod = qs_res.get("defaultKeyStatistics") or {}
                fin_mod = qs_res.get("financialData") or {}
                prof_mod = qs_res.get("assetProfile") or {}
                yahoo_spot_raw = _raw_num(price_mod.get("regularMarketPrice"))
                if yahoo_spot_raw is not None:
                    yahoo_spot = round(float(yahoo_spot_raw), 2)
                cand_info = {
                    "longName": price_mod.get("longName") or price_mod.get("shortName"),
                    "sector": prof_mod.get("sector") or prof_mod.get("industry"),
                    "trailingPE": _raw_num(summary_mod.get("trailingPE")),
                    "forwardPE": _raw_num(stats_mod.get("forwardPE")),
                    "returnOnEquity": _raw_num(fin_mod.get("returnOnEquity")),
                    "returnOnAssets": _raw_num(fin_mod.get("returnOnAssets")),
                    "operatingMargins": _raw_num(fin_mod.get("operatingMargins")),
                    "revenueGrowth": _raw_num(fin_mod.get("revenueGrowth")),
                    "earningsGrowth": _raw_num(fin_mod.get("earningsGrowth")),
                    "earningsQuarterlyGrowth": _raw_num(fin_mod.get("earningsQuarterlyGrowth")),
                    "marketCap": _raw_num(summary_mod.get("marketCap")) or _raw_num(price_mod.get("marketCap")),
                    "sharesOutstanding": _raw_num(stats_mod.get("sharesOutstanding")) or _raw_num(stats_mod.get("impliedSharesOutstanding")),
                    "beta": _raw_num(stats_mod.get("beta")) or _raw_num(summary_mod.get("beta")),
                    "debtToEquity": _raw_num(fin_mod.get("debtToEquity")),
                }
                if cand_info.get("marketCap") or cand_info.get("trailingPE") or cand_info.get("returnOnEquity"):
                    info = cand_info
                    break
            except Exception:
                continue
        if info:
            break
        if attempt < max_attempts - 1:
            time.sleep(min(10.0, 2.0 * (attempt + 1)))

    return _build_merged_fundamentals(
        info,
        yahoo_sym=yahoo_sym,
        currency=currency,
        spot=spot,
        nse_sym=nse_sym,
        yahoo_spot=yahoo_spot,
        strict=strict,
    )


def _fetch_via_nse_mcp(
    sym: str,
    nse_sym: str,
    yahoo_sym: str,
    *,
    strict_fundamentals: bool = False,
) -> dict[str, Any] | None:
    """Fetch live NSE quote & 1Y Bhavcopy OHLCV via Official NSE India MCP servers."""
    from .nse_mcp import fetch_nse_stock_history_1y, fetch_nse_stock_quote

    quote = fetch_nse_stock_quote(nse_sym)
    if not quote or float(quote.get("spot") or 0.0) <= 0:
        if strict_fundamentals:
            raise RuntimeError(f"Official NSE MCP quote unavailable for {nse_sym}.")
        return None
    candles, adjustments = fetch_nse_stock_history_1y(nse_sym)
    spot = round(float(quote["spot"]), 2)
    change_pct = round(float(quote.get("change_pct") or 0.0), 2)
    ret_1y = round(((spot - float(candles[0]["close"])) / max(1e-6, float(candles[0]["close"]))) * 100.0, 2)

    fund = _fetch_yahoo_fundamentals_only(yahoo_sym, "INR", spot, strict=strict_fundamentals, nse_sym=nse_sym)
    fund_live = fund["fundamentals_status"] == "live"
    fund_cat = fund.get("fundamental_source_category", "YAHOO_LIVE_FEED" if fund_live else "UNAVAILABLE")
    source_desc = (
        f"Official NSE MCP (cm-market + bhavcopy) + {fund_cat}"
        if fund_live
        else "Official NSE MCP (cm-market + bhavcopy)"
    )

    return {
        "symbol": sym,
        "yahoo_symbol": yahoo_sym,
        "nse_symbol": nse_sym,
        "name": fund["name"] if fund["name"] != nse_sym else sym,
        "exchange": "NSE",
        "currency": "INR",
        "sector": fund["sector"],
        "asset_type": "STOCK",
        "setup_type": DEFAULT_SYMBOL_CONFIG.get(sym, {}).get("setup_type", "Live NSE MCP Scan"),
        "price": spot,
        "last_candle_date": candles[-1]["date"],
        "last_candle_close": round(float(candles[-1]["close"]), 2),
        "change_pct": change_pct,
        "pe": fund["pe"],
        "roe": fund["roe"],
        "roce": fund["roce"],
        "opm": fund["opm"],
        "sales_growth": fund["sales_growth"],
        "profit_growth": fund["profit_growth"],
        "mcap_cr": fund["mcap_cr"],
        "beta": fund["beta"],
        "ret_1y": ret_1y,
        "debt_to_equity": fund["debt_to_equity"],
        "candles": candles,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "market_timestamp": quote.get("market_timestamp") or f"{candles[-1]['date']}T10:00:00+00:00",
        "market_timestamp_kind": "exchange_quote",
        "nse_raw_ist_timestamp": quote.get("raw_ist_timestamp"),
        "nse_corporate_adjustments": adjustments,
        "secondary_spot_price": fund.get("yahoo_spot"),
        "secondary_source": "Yahoo Finance Spot",
        "price_source_category": "NSE_OFFICIAL_MCP",
        "ohlcv_source_category": "NSE_OFFICIAL_MCP",
        "fundamental_source_category": fund_cat,
        "fundamentals_status": fund["fundamentals_status"],
        "missing_fields": fund["missing_fields"],
        "proxy_fields": fund["proxy_fields"],
        "cache_hit": False,
        "cache_age_seconds": 0.0,
        "data_source": source_desc,
    }


def fetch_live_market_snapshot(
    symbol: str,
    force_refresh: bool = False,
    *,
    strict_no_fallback: bool = False,
) -> dict[str, Any]:
    """Fetch real OHLCV bars and quotes (Official NSE MCP primary for NSE; fail upfront when strict_no_fallback=True)."""
    sym = symbol.upper().strip()
    now = time.time()
    if not force_refresh and sym in _LIVE_CACHE:
        ts, cached = _LIVE_CACHE[sym]
        if (now - ts) < _CACHE_TTL_SEC:
            hit = dict(cached)
            hit["cache_hit"] = True
            hit["cache_age_seconds"] = round(now - ts, 1)
            return hit

    cfg = DEFAULT_SYMBOL_CONFIG.get(sym)
    nse_candidate = None
    yahoo_candidate = None
    if cfg:
        if cfg.get("exchange") == "NSE" and cfg.get("nse"):
            nse_candidate = cfg["nse"]
            yahoo_candidate = cfg["yahoo"]
    elif "." not in sym or sym.endswith(".NS"):
        nse_candidate = sym[:-3] if sym.endswith(".NS") else sym
        yahoo_candidate = f"{nse_candidate}.NS"

    # 1. Official NSE MCP for NSE symbols
    if nse_candidate and yahoo_candidate:
        if strict_no_fallback and not USE_NSE_MCP:
            raise RuntimeError("FINSISTANT_USE_NSE_MCP is disabled while strict_no_fallback=True.")
        if USE_NSE_MCP:
            if strict_no_fallback:
                nse_snap = _fetch_via_nse_mcp(sym, nse_candidate, yahoo_candidate, strict_fundamentals=True)
                if not nse_snap:
                    raise RuntimeError(f"Official NSE MCP returned empty snapshot for {sym} ({nse_candidate}).")
                _LIVE_CACHE[sym] = (now, nse_snap)
                return nse_snap
            try:
                nse_snap = _fetch_via_nse_mcp(sym, nse_candidate, yahoo_candidate, strict_fundamentals=False)
                if nse_snap:
                    _LIVE_CACHE[sym] = (now, nse_snap)
                    return nse_snap
            except Exception:
                pass

    yahoo_sym, exchange_hint = resolve_yahoo_symbol(sym)

    # 2. Prefer yfinance for non-NSE / fallback feeds (handles Yahoo session/TLS cleanly)
    try:
        yf_snap = _fetch_via_yfinance(sym, yahoo_sym, exchange_hint)
        if yf_snap:
            yf_snap.setdefault("price_source_category", "YAHOO_LIVE_FEED")
            yf_snap.setdefault("ohlcv_source_category", "HISTORICAL_MARKET_DATA")
            yf_snap.setdefault("fundamental_source_category", "YAHOO_LIVE_FEED")
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
    prev_bar_close = float(candles[-2]["close"]) if len(candles) >= 2 else prev_close
    change_pct = round(((spot - prev_bar_close) / max(1e-6, prev_bar_close)) * 100.0, 2)
    ret_1y = round(((spot - float(candles[0]["close"])) / max(1e-6, float(candles[0]["close"]))) * 100.0, 2)

    raw_mkt_ts = meta.get("regularMarketTime")
    last_bar_date = candles[-1]["date"]
    if isinstance(raw_mkt_ts, (int, float)) and raw_mkt_ts > 0:
        market_timestamp = datetime.fromtimestamp(float(raw_mkt_ts), tz=timezone.utc).isoformat()
        market_ts_kind = "exchange_quote"
    else:
        close_suffix = "T10:00:00+00:00" if yahoo_sym.endswith((".NS", ".BO")) or yahoo_sym == "^NSEI" else "T21:00:00+00:00"
        market_timestamp = f"{last_bar_date}{close_suffix}"
        market_ts_kind = "end_of_day_bar"

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
    sector = prof_mod.get("sector") or prof_mod.get("industry")
    qs_info = {
        "longName": long_name,
        "sector": sector,
        "trailingPE": _raw_num(summary_mod.get("trailingPE")),
        "forwardPE": _raw_num(stats_mod.get("forwardPE")),
        "returnOnEquity": _raw_num(fin_mod.get("returnOnEquity")),
        "returnOnAssets": _raw_num(fin_mod.get("returnOnAssets")),
        "operatingMargins": _raw_num(fin_mod.get("operatingMargins")),
        "revenueGrowth": _raw_num(fin_mod.get("revenueGrowth")),
        "earningsGrowth": _raw_num(fin_mod.get("earningsGrowth")),
        "earningsQuarterlyGrowth": _raw_num(fin_mod.get("earningsQuarterlyGrowth")),
        "marketCap": _raw_num(summary_mod.get("marketCap")) or _raw_num(price_mod.get("marketCap")),
        "sharesOutstanding": _raw_num(stats_mod.get("sharesOutstanding")) or _raw_num(stats_mod.get("impliedSharesOutstanding")),
        "beta": _raw_num(stats_mod.get("beta")) or _raw_num(summary_mod.get("beta")),
        "debtToEquity": _raw_num(fin_mod.get("debtToEquity")),
    }
    fund = _build_merged_fundamentals(
        qs_info,
        yahoo_sym=yahoo_sym,
        currency=currency,
        spot=spot,
        nse_sym=sym if exchange == "NSE" or yahoo_sym.endswith(".NS") else None,
        yahoo_spot=round(spot, 2),
        strict=strict_no_fallback,
    )

    setup_type = DEFAULT_SYMBOL_CONFIG.get(sym, {}).get("setup_type", "Live Market Scan")

    snapshot = {
        "symbol": sym,
        "yahoo_symbol": yahoo_sym,
        "name": fund["name"],
        "exchange": exchange,
        "currency": currency,
        "sector": fund["sector"],
        "asset_type": "STOCK",
        "setup_type": setup_type,
        "price": round(spot, 2),
        "last_candle_date": last_bar_date,
        "last_candle_close": round(float(candles[-1]["close"]), 2),
        "change_pct": change_pct,
        "pe": fund["pe"],
        "roe": fund["roe"],
        "roce": fund["roce"],
        "opm": fund["opm"],
        "sales_growth": fund["sales_growth"],
        "profit_growth": fund["profit_growth"],
        "mcap_cr": fund["mcap_cr"],
        "beta": fund["beta"],
        "ret_1y": ret_1y,
        "debt_to_equity": fund["debt_to_equity"],
        "candles": candles,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "market_timestamp": market_timestamp,
        "market_timestamp_kind": market_ts_kind,
        "price_source_category": "YAHOO_LIVE_FEED",
        "ohlcv_source_category": "HISTORICAL_MARKET_DATA",
        "fundamental_source_category": fund["fundamental_source_category"],
        "fundamentals_status": fund["fundamentals_status"],
        "missing_fields": fund["missing_fields"],
        "proxy_fields": fund["proxy_fields"],
        "cache_hit": False,
        "cache_age_seconds": 0.0,
        "data_source": f"Yahoo Finance Live + {fund['fundamental_source_category']}",
    }
    _LIVE_CACHE[sym] = (now, snapshot)
    return snapshot


def fetch_benchmark_candles(yahoo_sym: str = "^NSEI", period: str = "1y", force_refresh: bool = False) -> dict[str, Any]:
    """Fetch real 1Y daily OHLCV bars for benchmark comparison (Official NSE MCP primary for Nifty 50)."""
    cache_key = f"__BENCH_{yahoo_sym}"
    now = time.time()
    if not force_refresh and cache_key in _LIVE_CACHE:
        ts, cached = _LIVE_CACHE[cache_key]
        if (now - ts) < 300.0:
            return dict(cached, cache_hit=True, cache_age_seconds=round(now - ts, 1))

    if USE_NSE_MCP and yahoo_sym in ("^NSEI", "NIFTY50", "Nifty 50"):
        try:
            from .nse_mcp import fetch_nse_index_1y

            nse_idx = fetch_nse_index_1y("Nifty 50")
            if nse_idx and len(nse_idx.get("candles") or []) >= 20:
                payload = {
                    "symbol": "NIFTY50",
                    "yahoo_symbol": "^NSEI",
                    "candles": nse_idx["candles"],
                    "live_quote": nse_idx.get("live_quote"),
                    "fetched_at": nse_idx["fetched_at"],
                    "market_timestamp": nse_idx["market_timestamp"],
                    "data_source": nse_idx["data_source"],
                    "source_category": "NSE_OFFICIAL_MCP",
                    "cache_hit": False,
                    "cache_age_seconds": 0.0,
                }
                _LIVE_CACHE[cache_key] = (now, payload)
                return payload
        except Exception:
            pass

    candles: list[dict[str, Any]] = []
    source = "Historical Market Data (Yahoo Finance)"
    try:
        import yfinance as yf  # type: ignore

        hist = yf.Ticker(yahoo_sym).history(period=period, interval="1d", auto_adjust=False)
        if hist is not None and not hist.empty:
            for idx, row in hist.iterrows():
                dt_str = idx.date().isoformat() if hasattr(idx, "date") else str(idx)[:10]
                o, h, l, c = row.get("Open"), row.get("High"), row.get("Low"), row.get("Close")
                if any(v is None or v != v for v in (o, h, l, c)):
                    continue
                candles.append({
                    "date": dt_str,
                    "open": round(float(o), 2),
                    "high": round(float(h), 2),
                    "low": round(float(l), 2),
                    "close": round(float(c), 2),
                    "volume": int(row.get("Volume") or 0),
                })
            source = "Historical Market Data (yfinance)"
    except Exception:
        candles = []

    if len(candles) < 20:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{urllib.parse.quote(yahoo_sym)}?range={period}&interval=1d"
        data = _fetch_json(url, retries=2)
        res = (data.get("chart", {}) or {}).get("result") or []
        if res:
            timestamps = res[0].get("timestamp") or []
            q0 = ((res[0].get("indicators") or {}).get("quote") or [{}])[0]
            opens, highs, lows, closes = q0.get("open") or [], q0.get("high") or [], q0.get("low") or [], q0.get("close") or []
            vols = q0.get("volume") or []
            candles = []
            for i, ts_val in enumerate(timestamps):
                o = opens[i] if i < len(opens) else None
                h = highs[i] if i < len(highs) else None
                l = lows[i] if i < len(lows) else None
                c = closes[i] if i < len(closes) else None
                if None in (o, h, l, c):
                    continue
                candles.append({
                    "date": datetime.fromtimestamp(int(ts_val), tz=timezone.utc).date().isoformat(),
                    "open": round(float(o), 2),
                    "high": round(float(h), 2),
                    "low": round(float(l), 2),
                    "close": round(float(c), 2),
                    "volume": int((vols[i] if i < len(vols) else 0) or 0),
                })

    if not candles:
        raise RuntimeError(f"Benchmark historical data unavailable for {yahoo_sym}.")

    payload = {
        "symbol": "NIFTY50" if yahoo_sym == "^NSEI" else yahoo_sym,
        "yahoo_symbol": yahoo_sym,
        "candles": candles,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "market_timestamp": f"{candles[-1]['date']}T10:00:00+00:00",
        "data_source": source,
        "cache_hit": False,
        "cache_age_seconds": 0.0,
    }
    _LIVE_CACHE[cache_key] = (now, payload)
    return payload


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
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "holdings": holdings,
            "cash_balance": cash_balance,
        }
    except Exception as exc:
        return {
            "connected": False,
            "status": f"Kite API error ({exc}) — 0 holdings loaded",
            "token_source": session.get("source", "error"),
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "holdings": [],
            "cash_balance": 0.0,
        }
