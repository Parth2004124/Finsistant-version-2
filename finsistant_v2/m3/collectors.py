"""Authoritative Multi-Source Data Collectors for Moreshwar 3.0 NSE PIT Universe.

Primary Sources (Official NSE India Exchange Archives & APIs):
- Security Master: EQUITY_L.csv, delisted.csv, symbolchange.csv, namechange.csv
- Daily Full Bhavcopy + Delivery Data: sec_bhavdata_full_DDMMYYYY.csv
- Daily PR Archive (Indices Pd/Pr, Official Daily MCAPYYYYMMDD.csv, Bc CA notices): PRddmmyy.zip
- Corporate Actions API: corporates-corporateActions (Splits, Bonuses, Dividends, Rights, Restructuring)
- Index Snapshots (explicitly flagged static_as_of_date, never used as PIT history):
  ind_nifty50list.csv, ind_niftynext50list.csv, ind_niftymidcap150list.csv,
  ind_niftysmallcap250list.csv, ind_nifty500list.csv, ind_niftytotalmarket_list.csv

Secondary Verification Source:
- Yahoo Finance Chart API (v8/finance/chart) for OHLCV, splits, dividends, and delisting 404 probes.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import datetime, timedelta, timezone
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import re
import time
from typing import Any
import urllib.parse
import urllib3
import zipfile

from .schema import DEFAULT_M3_RAW_CACHE_DIR

_IST = timezone(timedelta(hours=5, minutes=30))
_MONTH_MAP = {
    "JAN": "01", "FEB": "02", "MAR": "03", "APR": "04", "MAY": "05", "JUN": "06",
    "JUL": "07", "AUG": "08", "SEP": "09", "OCT": "10", "NOV": "11", "DEC": "12",
}

EQUITY_SERIES_ELIGIBLE = {"EQ", "BE", "BZ", "SM", "ST", "SZ"}

NSE_STATIC_URLS = {
    "equity_l": "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv",
    "delisted": "https://nsearchives.nseindia.com/content/equities/delisted.csv",
    "symbolchange": "https://nsearchives.nseindia.com/content/equities/symbolchange.csv",
    "namechange": "https://nsearchives.nseindia.com/content/equities/namechange.csv",
    "etf_list": "https://nsearchives.nseindia.com/content/equities/eq_etfseclist.csv",
}

NSE_INDEX_SNAPSHOT_URLS = {
    "NIFTY_50": "https://nsearchives.nseindia.com/content/indices/ind_nifty50list.csv",
    "NIFTY_NEXT_50": "https://nsearchives.nseindia.com/content/indices/ind_niftynext50list.csv",
    "NIFTY_MIDCAP_150": "https://nsearchives.nseindia.com/content/indices/ind_niftymidcap150list.csv",
    "NIFTY_SMALLCAP_250": "https://nsearchives.nseindia.com/content/indices/ind_niftysmallcap250list.csv",
    "NIFTY_500": "https://nsearchives.nseindia.com/content/indices/ind_nifty500list.csv",
    "NIFTY_TOTAL_MARKET": "https://nsearchives.nseindia.com/content/indices/ind_niftytotalmarket_list.csv",
}

_HDRS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "*/*",
}


def build_http_pool(maxsize: int = 24) -> urllib3.PoolManager:
    proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy") or os.environ.get("HTTP_PROXY")
    ca_path = "/etc/ssl/certs/ca-certificates.crt"
    if proxy:
        kwargs: dict[str, Any] = {"num_pools": 8, "maxsize": maxsize}
        if Path(ca_path).exists():
            kwargs["ca_certs"] = ca_path
        return urllib3.ProxyManager(proxy, **kwargs)
    return urllib3.PoolManager(num_pools=8, maxsize=maxsize)


def parse_nse_date(raw: str | None) -> str | None:
    if not raw:
        return None
    s = str(raw).strip()
    if not s or s in ("-", "NA", "null", "None"):
        return None
    # Try ISO first
    if re.match(r"^\d{4}-\d{2}-\d{2}$", s):
        return s
    # DD-MMM-YYYY or DD MMM YYYY or DD-MMM-YY
    m = re.match(r"^(\d{1,2})[-/\s]([A-Za-z]{3})[-/\s](\d{2,4})$", s)
    if m:
        day, mon_s, yr = int(m.group(1)), m.group(2).upper(), m.group(3)
        mon = _MONTH_MAP.get(mon_s)
        if mon:
            if len(yr) == 2:
                y_i = int(yr)
                yr = f"{1900 + y_i if y_i >= 70 else 2000 + y_i}"
            return f"{int(yr):04d}-{mon}-{day:02d}"
    for fmt in ("%d-%m-%Y", "%d/%m/%Y"):
        try:
            return datetime.strptime(s, fmt).strftime("%Y-%m-%d")
        except ValueError:
            pass
    return None


def _http_get_bytes(pool: urllib3.PoolManager, url: str, timeout: float = 15.0, retries: int = 3) -> tuple[int, bytes]:
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            resp = pool.request("GET", url, headers=_HDRS, timeout=timeout)
            if resp.status == 200:
                return 200, resp.data
            if resp.status in (404, 410):
                return resp.status, b""
            if attempt < retries - 1:
                time.sleep(0.4 * (attempt + 1))
                continue
            return resp.status, resp.data
        except Exception as exc:
            last_exc = exc
            if attempt < retries - 1:
                time.sleep(0.5 * (attempt + 1))
    raise RuntimeError(f"HTTP GET failed for {url}: {last_exc}")


def _cached_text(
    pool: urllib3.PoolManager,
    url: str,
    cache_path: Path,
    force_refresh: bool = False,
) -> tuple[str, str]:
    if cache_path.exists() and not force_refresh:
        raw = cache_path.read_bytes()
        return raw.decode("utf-8", "replace"), hashlib.sha256(raw).hexdigest()
    status, data = _http_get_bytes(pool, url, timeout=20.0)
    if status != 200 or not data:
        raise RuntimeError(f"Failed fetching {url}: HTTP {status}")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_bytes(data)
    return data.decode("utf-8", "replace"), hashlib.sha256(data).hexdigest()


def collect_nse_master_files(
    pool: urllib3.PoolManager,
    raw_cache_dir: Path = DEFAULT_M3_RAW_CACHE_DIR,
    force_refresh: bool = False,
) -> dict[str, Any]:
    """Download/load EQUITY_L, delisted, symbolchange, namechange & static index lists."""
    masters_dir = raw_cache_dir / "masters"
    manifests: dict[str, Any] = {}
    parsed: dict[str, list[dict[str, str]]] = {}

    for key, url in NSE_STATIC_URLS.items():
        txt, sha = _cached_text(pool, url, masters_dir / f"{key}.csv", force_refresh=force_refresh)
        rows = []
        rdr = csv.DictReader(io.StringIO(txt))
        for row in rdr:
            clean = {str(k).strip(): str(v).strip() if v is not None else "" for k, v in row.items() if k is not None}
            rows.append(clean)
        parsed[key] = rows
        manifests[key] = {"url": url, "rows": len(rows), "sha256": sha}

    index_snapshots: dict[str, list[dict[str, str]]] = {}
    for idx_name, url in NSE_INDEX_SNAPSHOT_URLS.items():
        try:
            txt, sha = _cached_text(pool, url, masters_dir / f"idx_{idx_name}.csv", force_refresh=force_refresh)
            rows = []
            for row in csv.DictReader(io.StringIO(txt)):
                rows.append({str(k).strip(): str(v).strip() if v is not None else "" for k, v in row.items() if k is not None})
            index_snapshots[idx_name] = rows
            manifests[f"idx_{idx_name}"] = {"url": url, "rows": len(rows), "sha256": sha}
        except Exception as exc:
            index_snapshots[idx_name] = []
            manifests[f"idx_{idx_name}"] = {"url": url, "error": str(exc)}

    return {
        "masters": parsed,
        "index_snapshots": index_snapshots,
        "manifests": manifests,
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
    }


def parse_corporate_action_subject(subject: str, old_face_val_hint: float | None = None, new_face_val: str | None = None) -> dict[str, Any]:
    """Deterministically parse NSE corporate action strings without guessing."""
    s = " ".join(str(subject or "").split())
    sl = s.lower()

    # 1. Face Value Split / Sub-Division / Consolidation
    if "split" in sl or "sub-division" in sl or "subdivision" in sl or "consolidation" in sl:
        m = re.search(r"from\s+rs\.?\s*([0-9]+(?:\.[0-9]+)?)\s*/?-?.*?to\s+rs\.?\s*([0-9]+(?:\.[0-9]+)?)", sl)
        if m:
            f_old = float(m.group(1))
            f_new = float(m.group(2))
            if f_old > 0 and f_new > 0:
                return {
                    "action_type": "SPLIT",
                    "ratio_num": f_old,
                    "ratio_den": f_new,
                    "adj_factor": round(f_old / f_new, 8),
                    "cash_dividend_inr": 0.0,
                    "rights_premium_inr": None,
                }
        return {
            "action_type": "SPLIT",
            "ratio_num": None,
            "ratio_den": None,
            "adj_factor": 1.0,
            "cash_dividend_inr": 0.0,
            "rights_premium_inr": None,
        }

    # 2. Bonus (e.g. "Bonus 1:1", "Bonus 3:2" -> ratio A:B => factor (A+B)/B)
    if "bonus" in sl:
        m = re.search(r"bonus\s+(\d+(?:\.\d+)?)\s*:\s*(\d+(?:\.\d+)?)", sl)
        if m:
            a = float(m.group(1))
            b = float(m.group(2))
            if b > 0:
                return {
                    "action_type": "BONUS",
                    "ratio_num": a,
                    "ratio_den": b,
                    "adj_factor": round((a + b) / b, 8),
                    "cash_dividend_inr": 0.0,
                    "rights_premium_inr": None,
                }
        return {
            "action_type": "BONUS",
            "ratio_num": None,
            "ratio_den": None,
            "adj_factor": 1.0,
            "cash_dividend_inr": 0.0,
            "rights_premium_inr": None,
        }

    # 3. Rights Issue
    if "right" in sl:
        m = re.search(r"rights?\s+(\d+(?:\.\d+)?)\s*:\s*(\d+(?:\.\d+)?)", sl)
        num, den = (float(m.group(1)), float(m.group(2))) if m else (None, None)
        pm = re.search(r"(?:premium|@|price)\s*(?:rs\.?|inr)?\s*([0-9]+(?:\.[0-9]+)?)", sl)
        prem = float(pm.group(1)) if pm else None
        return {
            "action_type": "RIGHTS",
            "ratio_num": num,
            "ratio_den": den,
            "adj_factor": 1.0,  # Rights adjustment depends on cum-rights close; flagged honestly rather than guessed
            "cash_dividend_inr": 0.0,
            "rights_premium_inr": prem,
        }

    # 4. Demerger / Scheme of Arrangement / Amalgamation / Spin-off / Reduction
    if any(k in sl for k in ("demerg", "arrangement", "amalgam", "merger", "spin", "reduction")):
        return {
            "action_type": "DEMERGER_RESTRUCTURE",
            "ratio_num": None,
            "ratio_den": None,
            "adj_factor": 1.0,
            "cash_dividend_inr": 0.0,
            "rights_premium_inr": None,
        }

    # 5. Dividend (Cash / Special / Interim / Final)
    if "div" in sl:
        cash = 0.0
        dm = re.search(r"(?:rs\.?|re\.?)\s*([0-9]+(?:\.[0-9]+)?)", sl)
        if dm:
            cash = float(dm.group(1))
        return {
            "action_type": "DIVIDEND",
            "ratio_num": None,
            "ratio_den": None,
            "adj_factor": 1.0,
            "cash_dividend_inr": cash,
            "rights_premium_inr": None,
        }

    return {
        "action_type": "OTHER",
        "ratio_num": None,
        "ratio_den": None,
        "adj_factor": 1.0,
        "cash_dividend_inr": 0.0,
        "rights_premium_inr": None,
    }


def collect_nse_corporate_actions(
    pool: urllib3.PoolManager,
    start_year: int = 2021,
    end_date_iso: str = "2026-11-19",
    raw_cache_dir: Path = DEFAULT_M3_RAW_CACHE_DIR,
    force_refresh: bool = False,
) -> list[dict[str, Any]]:
    """Fetch complete NSE equities corporate actions calendar in annual slices."""
    ca_dir = raw_cache_dir / "corporate_actions"
    ca_dir.mkdir(parents=True, exist_ok=True)
    end_yr = int(end_date_iso[:4])
    all_rows: list[dict[str, Any]] = []

    for yr in range(start_year, end_yr + 1):
        fd = f"01-01-{yr}"
        if yr == end_yr:
            y_s, m_s, d_s = end_date_iso.split("-")
            td = f"{d_s}-{m_s}-{y_s}"
        else:
            td = f"31-12-{yr}"
        c_path = ca_dir / f"ca_{yr}.json"
        if c_path.exists() and not force_refresh:
            chunk = json.loads(c_path.read_text(encoding="utf-8"))
        else:
            u = f"https://www.nseindia.com/api/corporates-corporateActions?index=equities&from_date={fd}&to_date={td}"
            status, data = _http_get_bytes(pool, u, timeout=20.0)
            if status != 200:
                raise RuntimeError(f"Failed fetching NSE Corporate Actions {fd}..{td}: HTTP {status}")
            chunk = json.loads(data.decode("utf-8", "replace"))
            c_path.write_text(json.dumps(chunk), encoding="utf-8")
        if isinstance(chunk, list):
            all_rows.extend(chunk)
    return all_rows


def collect_trading_calendar_dates(
    pool: urllib3.PoolManager,
    start_date: str = "2021-01-01",
    end_date: str = "2026-11-19",
    raw_cache_dir: Path = DEFAULT_M3_RAW_CACHE_DIR,
    force_refresh: bool = False,
) -> list[str]:
    """Fetch authoritative NSE trading dates from ^NSEI chart metadata, filtered to [start_date, end_date]."""
    cal_path = raw_cache_dir / "nsei_calendar_raw.json"
    if cal_path.exists() and not force_refresh:
        payload = json.loads(cal_path.read_text(encoding="utf-8"))
    else:
        url = "https://query1.finance.yahoo.com/v8/finance/chart/%5ENSEI?range=10y&interval=1d"
        status, data = _http_get_bytes(pool, url, timeout=15.0)
        if status != 200:
            raise RuntimeError(f"Failed fetching ^NSEI calendar: HTTP {status}")
        payload = json.loads(data.decode("utf-8", "replace"))
        cal_path.parent.mkdir(parents=True, exist_ok=True)
        cal_path.write_text(json.dumps(payload), encoding="utf-8")

    res = (payload.get("chart") or {}).get("result") or []
    if not res:
        raise RuntimeError("Empty ^NSEI calendar payload")
    ts = res[0].get("timestamp") or []
    seen: set[str] = set()
    dates: list[str] = []
    for t in ts:
        # NSE trades 09:15-15:30 IST; UTC date equals IST date for cash-market timestamps
        d_iso = datetime.fromtimestamp(int(t), tz=_IST).strftime("%Y-%m-%d")
        if start_date <= d_iso <= end_date and d_iso not in seen:
            seen.add(d_iso)
            dates.append(d_iso)
    dates.sort()
    return dates


def _parse_pr_zip_essential(zip_bytes: bytes, d_iso: str) -> dict[str, Any]:
    """Extract only index bars (Pd/Pr), official MCAP (when present), and Bc CA from PR.zip."""
    out: dict[str, Any] = {"indices": [], "mcap": {}, "has_official_mcap": False}
    y, m, d = d_iso.split("-")
    tag6 = f"{d}{m}{y[2:]}"
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        names = {n.lower(): n for n in zf.namelist()}
        # Prefer Pd{ddmmyy}.csv (richer OHLC), fallback Pr{ddmmyy}.csv
        pd_key = f"pd{tag6}.csv"
        if pd_key in names:
            txt = zf.read(names[pd_key]).decode("utf-8", "replace")
            for row in csv.DictReader(io.StringIO(txt)):
                if str(row.get("IND_SEC") or "").strip().upper() == "Y" or str(row.get("MKT") or "").strip().upper() == "Y":
                    nm = str(row.get("SECURITY") or "").strip()
                    cl_s = str(row.get("CLOSE_PRICE") or "").strip()
                    if not nm or not cl_s:
                        continue
                    try:
                        cl = float(cl_s)
                    except ValueError:
                        continue
                    if cl <= 0:
                        continue
                    out["indices"].append({
                        "index_name": nm,
                        "prev_close": _safe_float(row.get("PREV_CL_PR")),
                        "open": _safe_float(row.get("OPEN_PRICE")),
                        "high": _safe_float(row.get("HIGH_PRICE")),
                        "low": _safe_float(row.get("LOW_PRICE")),
                        "close": cl,
                        "turnover_inr": _safe_float(row.get("NET_TRDVAL")),
                    })

        # Official MCAP{ddmmyyyy}.csv (published daily in PR.zip since 2024 UDiFF)
        mcap_key = next((orig for low, orig in names.items() if low.startswith("mcap") and low.endswith(".csv")), None)
        if mcap_key:
            out["has_official_mcap"] = True
            txt = zf.read(mcap_key).decode("utf-8", "replace")
            for row in csv.DictReader(io.StringIO(txt)):
                c = {str(k).strip(): str(v).strip() if v is not None else "" for k, v in row.items() if k is not None}
                sym = c.get("Symbol") or ""
                srs = c.get("Series") or ""
                if not sym:
                    continue
                mcap_v = _safe_float(c.get("Market Cap(Rs.)"))
                iss_v = _safe_int(c.get("Issue Size"))
                cat = c.get("Category") or ""
                out["mcap"][f"{sym}|{srs}"] = {
                    "mcap_inr": mcap_v,
                    "issue_size": iss_v,
                    "category": cat,
                }
    return out


def _safe_float(v: Any) -> float | None:
    if v is None:
        return None
    s = str(v).strip()
    if not s or s in ("-", "NA", "null"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _safe_int(v: Any) -> int | None:
    if v is None:
        return None
    s = str(v).strip()
    if not s or s in ("-", "NA", "null"):
        return None
    try:
        return int(float(s))
    except ValueError:
        return None


def collect_daily_nse_archives(
    pool: urllib3.PoolManager,
    dates: list[str],
    raw_cache_dir: Path = DEFAULT_M3_RAW_CACHE_DIR,
    max_workers: int = 16,
    fetch_pr_zip: bool = True,
) -> dict[str, Any]:
    """Concurrently fetch & cache sec_bhavdata_full_{DDMMYYYY}.csv (.gz) and PR{DDMMYY}.zip essentials."""
    bhav_dir = raw_cache_dir / "sec_bhav_gz"
    pr_dir = raw_cache_dir / "pr_json_gz"
    bhav_dir.mkdir(parents=True, exist_ok=True)
    pr_dir.mkdir(parents=True, exist_ok=True)

    def _one_day(d_iso: str) -> dict[str, Any]:
        y, m, d = d_iso.split("-")
        tag8 = f"{d}{m}{y}"
        tag6 = f"{d}{m}{y[2:]}"
        sb_url = f"https://nsearchives.nseindia.com/products/content/sec_bhavdata_full_{tag8}.csv"
        pr_url = f"https://nsearchives.nseindia.com/archives/equities/bhavcopy/pr/PR{tag6}.zip"
        sb_path = bhav_dir / f"{d_iso}.csv.gz"
        pr_path = pr_dir / f"{d_iso}.json.gz"

        sb_status = 200
        if not sb_path.exists():
            st, raw = _http_get_bytes(pool, sb_url, timeout=15.0)
            sb_status = st
            if st == 200 and raw:
                sb_path.write_bytes(gzip.compress(raw, compresslevel=5))

        pr_status = 200 if not fetch_pr_zip else 200
        if fetch_pr_zip and not pr_path.exists():
            st_p, raw_p = _http_get_bytes(pool, pr_url, timeout=18.0)
            pr_status = st_p
            if st_p == 200 and raw_p:
                try:
                    parsed_pr = _parse_pr_zip_essential(raw_p, d_iso)
                    pr_path.write_bytes(gzip.compress(json.dumps(parsed_pr).encode("utf-8"), compresslevel=5))
                except Exception:
                    pr_status = -1

        return {
            "date": d_iso,
            "sb_ok": sb_path.exists(),
            "sb_status": sb_status,
            "sb_url": sb_url,
            "pr_ok": pr_path.exists(),
            "pr_status": pr_status,
            "pr_url": pr_url,
        }

    results: dict[str, Any] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(_one_day, dt): dt for dt in dates}
        for fut in as_completed(futs):
            dt = futs[fut]
            try:
                results[dt] = fut.result()
            except Exception as exc:
                results[dt] = {"date": dt, "sb_ok": False, "sb_status": -1, "pr_ok": False, "error": str(exc)}
    return results


def fetch_yahoo_cross_check_sample(
    pool: urllib3.PoolManager,
    symbols: list[str],
    raw_cache_dir: Path = DEFAULT_M3_RAW_CACHE_DIR,
    max_workers: int = 8,
) -> dict[str, Any]:
    """Fetch Yahoo Finance Chart API (6Y + split/div events) for stratified verification sample."""
    yf_dir = raw_cache_dir / "yf_verify"
    yf_dir.mkdir(parents=True, exist_ok=True)

    def _one_sym(sym: str) -> tuple[str, dict[str, Any]]:
        c_file = yf_dir / f"{sym}.json.gz"
        if c_file.exists():
            return sym, json.loads(gzip.decompress(c_file.read_bytes()).decode("utf-8"))
        q_sym = urllib.parse.quote(f"{sym}.NS")
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{q_sym}?range=6y&interval=1d&events=div%2Csplit"
        status, data = _http_get_bytes(pool, url, timeout=12.0, retries=2)
        if status != 200 or not data:
            rec = {"status": "HTTP_ERROR", "http_status": status, "bars": {}, "splits": {}, "dividends": {}}
            c_file.write_bytes(gzip.compress(json.dumps(rec).encode("utf-8")))
            return sym, rec
        payload = json.loads(data.decode("utf-8", "replace"))
        res = (payload.get("chart") or {}).get("result") or []
        if not res:
            rec = {"status": "EMPTY", "http_status": status, "bars": {}, "splits": {}, "dividends": {}}
            c_file.write_bytes(gzip.compress(json.dumps(rec).encode("utf-8")))
            return sym, rec
        r0 = res[0]
        ts = r0.get("timestamp") or []
        q = ((r0.get("indicators") or {}).get("quote") or [{}])[0]
        adj = (((r0.get("indicators") or {}).get("adjclose") or [{}])[0]).get("adjclose") or []
        bars: dict[str, dict[str, Any]] = {}
        for i, t in enumerate(ts):
            d_iso = datetime.fromtimestamp(int(t), tz=_IST).strftime("%Y-%m-%d")
            o, h, l, c, v = (q.get("open") or [None])[i], (q.get("high") or [None])[i], (q.get("low") or [None])[i], (q.get("close") or [None])[i], (q.get("volume") or [None])[i]
            ac = adj[i] if i < len(adj) else None
            if None in (o, h, l, c):
                continue
            bars[d_iso] = {
                "open": float(o),
                "high": float(h),
                "low": float(l),
                "close": float(c),
                "adj_close": float(ac) if ac is not None else float(c),
                "volume": int(v or 0),
            }
        events = r0.get("events") or {}
        splits = {
            datetime.fromtimestamp(int(k), tz=_IST).strftime("%Y-%m-%d"): v
            for k, v in (events.get("splits") or {}).items()
        }
        divs = {
            datetime.fromtimestamp(int(k), tz=_IST).strftime("%Y-%m-%d"): v
            for k, v in (events.get("dividends") or {}).items()
        }
        rec = {"status": "OK", "http_status": 200, "bars": bars, "splits": splits, "dividends": divs}
        c_file.write_bytes(gzip.compress(json.dumps(rec).encode("utf-8")))
        return sym, rec

    out: dict[str, Any] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for sym, rec in ex.map(_one_sym, symbols):
            out[sym] = rec
    return out
