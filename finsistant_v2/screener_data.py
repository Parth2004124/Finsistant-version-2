"""Screener.in Fundamental Data Fetcher for Indian NSE Equities (Zero Fake Defaults).

Used exclusively for company-level fundamental ratios (Market Cap, Stock P/E, ROE, ROCE,
OPM %, Sales Growth, Profit Growth, Debt/Equity) when Yahoo Finance ratios are unavailable.
Never invents 0.0 defaults for missing fields — returns None for any absent metric.
"""

from __future__ import annotations

import html
import re
import time
from typing import Any
import urllib.error
import urllib.parse
import urllib.request

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

_NUM_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")
_TAG_RE = re.compile(r"<[^>]+>")


def _clean_text(raw: str) -> str:
    txt = _TAG_RE.sub(" ", html.unescape(raw))
    return re.sub(r"\s+", " ", txt).strip()


def _parse_number(raw: str | None) -> float | None:
    if not raw:
        return None
    s = _clean_text(raw).replace(",", "")
    m = _NUM_RE.search(s)
    if not m:
        return None
    try:
        return float(m.group(0))
    except ValueError:
        return None


def _http_get_html(url: str, *, max_attempts: int = 4, timeout: float = 15.0) -> tuple[int, str]:
    last_err: Exception | None = None
    for attempt in range(max_attempts):
        req = urllib.request.Request(url, headers=_HEADERS)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return int(resp.status), resp.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return 404, ""
            last_err = exc
            if exc.code in (429, 500, 502, 503, 504) and attempt < max_attempts - 1:
                time.sleep(min(15.0, 2.0 * (2 ** attempt)))
                continue
            raise
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_err = exc
            if attempt < max_attempts - 1:
                time.sleep(min(12.0, 2.0 * (attempt + 1)))
                continue
    raise last_err or RuntimeError(f"Failed to fetch Screener page: {url}")


def parse_screener_html(page_html: str, symbol: str) -> dict[str, Any]:
    """Parse Screener.in company HTML into fundamental ratios (None for absent fields)."""
    if not page_html or "top-ratios" not in page_html:
        raise ValueError(f"Screener.in page for {symbol} did not contain #top-ratios.")

    # Company display name
    h1_m = re.search(r"<h1[^>]*>(.*?)</h1>", page_html, re.IGNORECASE | re.DOTALL)
    name = _clean_text(h1_m.group(1)) if h1_m else symbol

    # Sector / Industry breadcrumbs (if present)
    sec_m = re.search(
        r'id="peers"[^>]*>.*?<p[^>]*class="sub"[^>]*>(.*?)</p>',
        page_html,
        re.IGNORECASE | re.DOTALL,
    )
    sector = None
    if sec_m:
        links = re.findall(r"<a[^>]*>(.*?)</a>", sec_m.group(1), re.IGNORECASE | re.DOTALL)
        if links:
            sector = _clean_text(links[-1])

    # Top ratios block (<ul id="top-ratios"> ... <span class="name">...</span> ... <span class="number">...</span>)
    top_m = re.search(r'<ul[^>]*id="top-ratios"[^>]*>(.*?)</ul>', page_html, re.IGNORECASE | re.DOTALL)
    ratios_raw: dict[str, float | None] = {}
    if top_m:
        for li in re.findall(r"<li[^>]*>(.*?)</li>", top_m.group(1), re.IGNORECASE | re.DOTALL):
            nm_m = re.search(r'class="[^"]*name[^"]*"[^>]*>(.*?)</span>', li, re.IGNORECASE | re.DOTALL)
            val_m = re.search(r'class="[^"]*nowrap[^"]*value[^"]*"[^>]*>(.*?)</span>', li, re.IGNORECASE | re.DOTALL)
            if not nm_m or not val_m:
                continue
            label = _clean_text(nm_m.group(1)).lower()
            num_m = re.search(r'class="[^"]*number[^"]*"[^>]*>(.*?)</span>', val_m.group(1), re.IGNORECASE | re.DOTALL)
            val = _parse_number(num_m.group(1) if num_m else val_m.group(1))
            ratios_raw[label] = val

    def _find_top(*needles: str) -> float | None:
        for k, v in ratios_raw.items():
            if any(n in k for n in needles) and v is not None:
                return v
        return None

    mcap_cr = _find_top("market cap")
    pe = _find_top("stock p/e", "p/e")
    roce = _find_top("roce")
    roe = _find_top("roe")
    debt_to_equity = _find_top("debt to equity")

    # Helper to extract latest value from a Screener financial table row
    def _latest_table_row_val(section_id: str, row_needles: tuple[str, ...]) -> float | None:
        sec_pat = re.search(
            rf'<section[^>]*id="{re.escape(section_id)}"[^>]*>(.*?)</section>',
            page_html,
            re.IGNORECASE | re.DOTALL,
        )
        if not sec_pat:
            return None
        for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", sec_pat.group(1), re.IGNORECASE | re.DOTALL):
            tds = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", tr, re.IGNORECASE | re.DOTALL)
            if len(tds) < 2:
                continue
            row_lbl = _clean_text(tds[0]).lower()
            if any(n in row_lbl for n in row_needles):
                for cell in reversed(tds[1:]):
                    v = _parse_number(cell)
                    if v is not None:
                        return v
        return None

    opm = _find_top("opm")
    if opm is None:
        opm = _latest_table_row_val("quarters", ("opm %", "financing margin %"))
    if opm is None:
        opm = _latest_table_row_val("profit-loss", ("opm %", "financing margin %"))

    # Comp. Sales Growth & Comp. Profit Growth tables inside #profit-loss ("TTM:" or last row)
    def _extract_growth(table_title: str) -> float | None:
        pl_m = re.search(r'<section[^>]*id="profit-loss"[^>]*>(.*?)</section>', page_html, re.IGNORECASE | re.DOTALL)
        if not pl_m:
            return None
        tbl_m = re.search(
            rf"{re.escape(table_title)}.*?<table[^>]*>(.*?)</table>",
            pl_m.group(1),
            re.IGNORECASE | re.DOTALL,
        )
        if not tbl_m:
            return None
        rows = re.findall(r"<tr[^>]*>(.*?)</tr>", tbl_m.group(1), re.IGNORECASE | re.DOTALL)
        # Prefer TTM, else 3 Years, else latest available row
        for preferred in ("ttm", "3 years", "5 years"):
            for tr in rows:
                tds = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", tr, re.IGNORECASE | re.DOTALL)
                if len(tds) >= 2 and preferred in _clean_text(tds[0]).lower():
                    v = _parse_number(tds[-1])
                    if v is not None:
                        return v
        for tr in reversed(rows):
            tds = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", tr, re.IGNORECASE | re.DOTALL)
            if len(tds) >= 2:
                v = _parse_number(tds[-1])
                if v is not None:
                    return v
        return None

    sales_growth = _find_top("sales growth")
    if sales_growth is None:
        sales_growth = _extract_growth("Compounded Sales Growth")

    profit_growth = _find_top("profit growth")
    if profit_growth is None:
        profit_growth = _extract_growth("Compounded Profit Growth")

    if debt_to_equity is None:
        borrowings = _latest_table_row_val("balance-sheet", ("borrowings",))
        reserves = _latest_table_row_val("balance-sheet", ("reserves",))
        eq_cap = _latest_table_row_val("balance-sheet", ("equity capital", "share capital"))
        if borrowings is not None and reserves is not None and eq_cap is not None:
            equity = reserves + eq_cap
            if equity > 0:
                debt_to_equity = round(max(0.0, borrowings) / equity, 2)

    return {
        "source": "SCREENER_IN",
        "name": name,
        "sector": sector,
        "mcap_cr": round(mcap_cr, 2) if mcap_cr is not None else None,
        "pe": round(pe, 2) if pe is not None else None,
        "roce": round(roce, 2) if roce is not None else None,
        "roe": round(roe, 2) if roe is not None else None,
        "opm": round(opm, 2) if opm is not None else None,
        "sales_growth": round(sales_growth, 2) if sales_growth is not None else None,
        "profit_growth": round(profit_growth, 2) if profit_growth is not None else None,
        "debt_to_equity": round(debt_to_equity, 2) if debt_to_equity is not None else None,
    }


def fetch_screener_fundamentals(nse_symbol: str) -> dict[str, Any]:
    """Fetch real fundamental ratios from Screener.in (consolidated first, standalone if no consolidated page)."""
    sym = re.sub(r"\.(NS|BO)$", "", str(nse_symbol).upper().strip())
    q_sym = urllib.parse.quote(sym)
    urls = [
        f"https://www.screener.in/company/{q_sym}/consolidated/",
        f"https://www.screener.in/company/{q_sym}/",
    ]
    last_err: Exception | None = None
    for url in urls:
        try:
            status, body = _http_get_html(url)
            if status == 404 or not body:
                continue
            parsed = parse_screener_html(body, sym)
            # Some symbols have an empty /consolidated/ shell with None ratios; try standalone URL
            if parsed.get("mcap_cr") is None and parsed.get("pe") is None and url.endswith("/consolidated/"):
                continue
            return parsed
        except Exception as exc:
            last_err = exc
            continue
    raise RuntimeError(f"Screener.in fundamentals unavailable for {sym}: {last_err or 'no data'}")
