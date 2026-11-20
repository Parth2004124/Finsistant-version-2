"""Moreshwar 3.0 PIT NSE Research Universe Orchestrator & Reproducible Snapshot Builder.

Supports full multi-year builds (`2021-01-01` -> `2026-11-19`) and safe incremental
appends without mutating historical frozen version records.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import sqlite3
from typing import Any
import xml.etree.ElementTree as ET
import zipfile

from .collectors import (
    EQUITY_SERIES_ELIGIBLE,
    NSE_INDEX_SNAPSHOT_URLS,
    _safe_float,
    _safe_int,
    build_http_pool,
    collect_daily_nse_archives,
    collect_nse_corporate_actions,
    collect_nse_master_files,
    collect_trading_calendar_dates,
    fetch_yahoo_cross_check_sample,
    parse_corporate_action_subject,
    parse_nse_date,
)
from .liquidity_engine import (
    PITUniverseStore,
    compute_and_store_pit_liquidity,
)
from .quality_engine import (
    audit_nse_vs_yahoo_sample,
    run_post_ingestion_dq_audit,
    validate_raw_bar_fields,
)
from .schema import (
    DEFAULT_M3_DATA_DIR,
    DEFAULT_M3_DB_PATH,
    DEFAULT_M3_RAW_CACHE_DIR,
    DEFAULT_M3_REPORTS_DIR,
    connect_m3_db,
    create_m3_indexes,
    init_m3_schema,
)

# Stratified sample across Large/Mid/Small, Splits/Bonuses, Symbol Changes & Delisted Controls
YAHOO_VERIFICATION_SAMPLE = [
    # Large Cap Core
    "RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFY", "SBIN", "ITC", "LT", "BHARTIARTL", "AXISBANK",
    "SUNPHARMA", "MARUTI", "M&M", "TATASTEEL", "HINDUNILVR", "BAJFINANCE", "ASIANPAINT", "NTPC",
    # Split / Bonus / Demerger heavy movers
    "IRCTC", "IEX", "TRENT", "DIXON", "CDSL", "KPITTECH", "PERSISTENT", "COFORGE", "POLYCAB", "BEL",
    "HAL", "VBL", "SUZLON", "MAZAGON", "BSE", "TATAPOWER", "JIOFIN", "TMPV",
    # Symbol-changed securities
    "ZYDUSLIFE", "MOTHERSON", "ETERNAL", "UNITDSPR", "LTIM", "SWANENERGY",
    # Distressed / Delisted / Merged survivorship controls
    "YESBANK", "JETAIRWAYS", "RCOM", "VODAFONE", "IDEA", "ZEEL",
    "HDFC", "MINDTREE", "DHFL", "RELCAPITAL", "SINTEX", "INFRATEL", "CADILAHC", "MOTHERSUMI", "ZOMATO",
]


def _canonical_symbol_chains(
    sym_changes: list[dict[str, str]],
) -> tuple[dict[str, str], dict[str, list[tuple[str, str, str]]]]:
    """Resolve chained OLD_SYMBOL -> NEW_SYMBOL changes deterministically."""
    next_map: dict[str, tuple[str, str]] = {}
    for row in sym_changes:
        old_s = (row.get("OLD SYMBOL") or "").upper().strip()
        new_s = (row.get("NEW SYMBOL") or "").upper().strip()
        eff = parse_nse_date(row.get("EFFECTIVE DATE"))
        if old_s and new_s and eff and old_s != new_s:
            next_map[old_s] = (new_s, eff)

    def _terminal(sym: str) -> str:
        cur = sym
        seen = {cur}
        while cur in next_map:
            nxt, _ = next_map[cur]
            if nxt in seen:
                break
            seen.add(nxt)
            cur = nxt
        return cur

    alias_to_terminal: dict[str, str] = {}
    groups: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for old_s, (new_s, eff) in next_map.items():
        term = _terminal(new_s)
        alias_to_terminal[old_s] = term
        alias_to_terminal[new_s] = term
        groups[term].append((old_s, new_s, eff))
    return alias_to_terminal, groups


def build_m3_pit_universe(
    *,
    start_date: str = "2021-01-01",
    end_date: str = "2026-11-19",
    db_path: Path = DEFAULT_M3_DB_PATH,
    data_dir: Path = DEFAULT_M3_DATA_DIR,
    raw_cache_dir: Path = DEFAULT_M3_RAW_CACHE_DIR,
    reports_dir: Path = DEFAULT_M3_REPORTS_DIR,
    incremental: bool = False,
    force_refresh_masters: bool = False,
    fetch_pr_zip: bool = True,
    max_workers: int = 16,
    snapshot_tag: str | None = None,
) -> dict[str, Any]:
    """Build or incrementally update the Moreshwar 3.0 NSE PIT Research Universe."""
    pool = build_http_pool(maxsize=max(20, max_workers + 6))
    retrieved_at = datetime.now(timezone.utc).isoformat()
    version_id = f"m3_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"

    # 1. Collect authoritative NSE masters, index snapshots, and corporate actions
    master_bundle = collect_nse_master_files(pool, raw_cache_dir=raw_cache_dir, force_refresh=force_refresh_masters)
    masters = master_bundle["masters"]
    index_snapshots = master_bundle["index_snapshots"]

    ca_raw = collect_nse_corporate_actions(
        pool, start_year=int(start_date[:4]), end_date_iso=end_date, raw_cache_dir=raw_cache_dir, force_refresh=force_refresh_masters
    )

    # 2. Collect ^NSEI trading calendar & daily NSE Bhavcopy + PR.zip archives
    cal_dates = collect_trading_calendar_dates(
        pool, start_date=start_date, end_date=end_date, raw_cache_dir=raw_cache_dir, force_refresh=force_refresh_masters
    )
    archive_status = collect_daily_nse_archives(
        pool, cal_dates, raw_cache_dir=raw_cache_dir, max_workers=max_workers, fetch_pr_zip=fetch_pr_zip
    )

    # Secondary cross-provider Yahoo Finance sample
    yf_sample = fetch_yahoo_cross_check_sample(pool, YAHOO_VERIFICATION_SAMPLE, raw_cache_dir=raw_cache_dir, max_workers=8)

    # 3. Initialize / Connect SQLite DB (stage full builds on fast local /tmp to avoid 9p WAL FUSE bottlenecks)
    staging_db = Path(f"/tmp/m3_staging_{os.getpid()}.db") if not incremental else db_path
    if staging_db != db_path and staging_db.exists():
        staging_db.unlink()

    prior_versions: list[tuple[Any, ...]] = []
    if db_path.exists() and not incremental:
        try:
            old_conn = connect_m3_db(db_path)
            prior_versions = [tuple(r) for r in old_conn.execute("SELECT * FROM dataset_versions ORDER BY created_at ASC")]
            old_conn.close()
        except Exception:
            prior_versions = []

    conn = connect_m3_db(staging_db)
    init_m3_schema(conn, create_indexes=incremental)
    conn.execute("PRAGMA journal_mode = MEMORY;")
    conn.execute("PRAGMA synchronous = OFF;")
    conn.execute("PRAGMA temp_store = MEMORY;")
    conn.execute("PRAGMA cache_size = -256000;")

    if prior_versions and not incremental:
        conn.executemany(
            "INSERT OR IGNORE INTO dataset_versions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            prior_versions,
        )
        conn.commit()

    parent_version_id = None
    existing_dates: set[str] = set()
    prev_ver = conn.execute("SELECT version_id FROM dataset_versions ORDER BY created_at DESC LIMIT 1").fetchone()
    if prev_ver:
        parent_version_id = prev_ver["version_id"]
    if incremental:
        existing_dates = {r["trade_date"] for r in conn.execute("SELECT trade_date FROM trading_calendar")}

    # 4. Build Canonical Security Master (`sec_id` = ISIN when known, else NSE_SYM:<TERM>) & SCD2 Symbol History
    alias_to_term, sym_groups = _canonical_symbol_chains(masters["symbolchange"])
    equity_l_by_sym: dict[str, dict[str, str]] = {}
    sym_to_isin: dict[str, str] = {}
    for row in masters["equity_l"]:
        sym = row.get("SYMBOL", "").upper().strip()
        isin = row.get("ISIN NUMBER", "").upper().strip()
        if sym:
            equity_l_by_sym[sym] = row
            if isin:
                sym_to_isin[sym] = isin

    etf_by_sym: dict[str, dict[str, str]] = {}
    for row in masters.get("etf_list", []):
        sym = row.get("Symbol", "").upper().strip()
        isin = row.get("ISINNumber", "").upper().strip()
        if sym:
            etf_by_sym[sym] = row
            if isin and sym not in sym_to_isin:
                sym_to_isin[sym] = isin

    delisted_by_sym: dict[str, dict[str, str]] = {}
    for row in masters["delisted"]:
        sym = row.get("Symbol", "").upper().strip()
        if sym:
            delisted_by_sym[sym] = row

    sector_by_sym: dict[str, tuple[str, str]] = {}
    snap_date = retrieved_at[:10]
    snap_rows = []
    for idx_name, rows in index_snapshots.items():
        for r in rows:
            sym = (r.get("Symbol") or "").upper().strip()
            isin = (r.get("ISIN Code") or "").upper().strip()
            ind = (r.get("Industry") or "").strip()
            comp = (r.get("Company Name") or "").strip()
            srs = (r.get("Series") or "EQ").strip()
            if sym:
                if isin and sym not in sym_to_isin:
                    sym_to_isin[sym] = isin
                if ind and sym not in sector_by_sym:
                    sector_by_sym[sym] = (ind, idx_name)
                snap_rows.append((idx_name, sym, isin or None, comp, ind, srs, snap_date, 1, NSE_INDEX_SNAPSHOT_URLS[idx_name]))

    # Supplement ISIN map from Corporate Actions API (captures merged/delisted companies like HDFC/MINDTREE!)
    for ca in ca_raw:
        csym = str(ca.get("symbol") or "").upper().strip()
        cisin = str(ca.get("isin") or "").upper().strip()
        if csym and cisin and cisin.startswith("IN"):
            term = alias_to_term.get(csym, csym)
            if term not in sym_to_isin:
                sym_to_isin[term] = cisin
            if csym not in sym_to_isin:
                sym_to_isin[csym] = cisin

    def _resolve_sec_id(raw_sym: str) -> str:
        s = raw_sym.upper().strip()
        term = alias_to_term.get(s, s)
        if term in sym_to_isin:
            return sym_to_isin[term]
        if s in sym_to_isin:
            return sym_to_isin[s]
        return f"NSE_SYM:{term}"

    # 5. Parse & Deduplicate Corporate Actions first so `adj_factor` can be computed inline during streaming ingest
    ca_inserts: list[tuple[Any, ...]] = []
    ca_seen: set[tuple[str, str, str, str]] = set()
    ca_adj_by_sec: dict[str, list[tuple[str, float]]] = defaultdict(list)

    for ca in ca_raw:
        csym = str(ca.get("symbol") or "").upper().strip()
        srs = str(ca.get("series") or "").upper().strip()
        ex_iso = parse_nse_date(ca.get("exDate"))
        if not csym or not ex_iso:
            continue
        if srs and srs not in EQUITY_SERIES_ELIGIBLE:
            continue
        subj = str(ca.get("subject") or "").strip()
        parsed_ca = parse_corporate_action_subject(subj, new_face_val=str(ca.get("faceVal") or ""))
        sec_id = _resolve_sec_id(csym)
        key4 = (sec_id, ex_iso, parsed_ca["action_type"], subj[:80])
        if key4 in ca_seen:
            continue
        ca_seen.add(key4)

        ca_id = hashlib.sha1(f"{sec_id}|{ex_iso}|{parsed_ca['action_type']}|{subj}".encode()).hexdigest()[:20]
        rec_iso = parse_nse_date(ca.get("recDate"))
        isin_val = str(ca.get("isin") or sym_to_isin.get(csym, "")).upper().strip() or None
        ca_inserts.append((
            ca_id, sec_id, csym, isin_val, ex_iso, rec_iso, srs or None,
            parsed_ca["action_type"], parsed_ca["ratio_num"], parsed_ca["ratio_den"],
            parsed_ca["adj_factor"], parsed_ca["cash_dividend_inr"], parsed_ca["rights_premium_inr"],
            str(ca.get("faceVal") or "") or None, subj, "NSE_CA_API",
        ))
        if parsed_ca["action_type"] in ("SPLIT", "BONUS") and float(parsed_ca["adj_factor"]) > 1.0001:
            ca_adj_by_sec[sec_id].append((ex_iso, float(parsed_ca["adj_factor"])))

    # 6. Stream Daily Bhavcopy + PR.zip archives into SQLite
    bhav_dir = raw_cache_dir / "sec_bhav_gz"
    pr_dir = raw_cache_dir / "pr_json_gz"
    target_dates = [d for d in cal_dates if d not in existing_dates]

    obs_stats: dict[str, dict[str, Any]] = {}  # sec_id -> {first, last, count, syms, series_set}
    missing_dates: list[str] = []
    duplicate_records: list[tuple[str, str, str]] = []
    non_equity_rows_skipped = 0

    cal_inserts: list[tuple[Any, ...]] = []
    idx_inserts: list[tuple[Any, ...]] = []
    bars_batch: list[tuple[Any, ...]] = []

    def _flush_bars() -> None:
        if not bars_batch:
            return
        conn.executemany(
            """
            INSERT OR IGNORE INTO daily_bars (
                trade_date, symbol, series, sec_id, prev_close, open, high, low, close,
                last_price, vwap, volume, turnover_inr, no_of_trades, delivery_qty, delivery_pct,
                adj_factor, adj_open, adj_high, adj_low, adj_close,
                mcap_inr, issue_size, mcap_source, exchange_category, dq_flags, is_valid_bar
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            bars_batch,
        )
        bars_batch.clear()

    def _adj_factor_for(sec_id: str, dt: str) -> float:
        evs = ca_adj_by_sec.get(sec_id)
        if not evs:
            return 1.0
        fac = 1.0
        for ex_dt, f in evs:
            if dt < ex_dt:
                fac *= f
        return fac

    for d_iso in target_dates:
        sb_path = bhav_dir / f"{d_iso}.csv.gz"
        if not sb_path.exists():
            missing_dates.append(d_iso)
            continue

        pr_data: dict[str, Any] = {"indices": [], "mcap": {}, "has_official_mcap": False}
        pr_path = pr_dir / f"{d_iso}.json.gz"
        if pr_path.exists():
            try:
                pr_data = json.loads(gzip.decompress(pr_path.read_bytes()).decode("utf-8"))
            except Exception:
                pass

        n50 = next((ix for ix in pr_data["indices"] if ix["index_name"].lower() == "nifty 50"), None)
        n500 = next((ix for ix in pr_data["indices"] if ix["index_name"].lower() == "nifty 500"), None)
        mid150 = next((ix for ix in pr_data["indices"] if ix["index_name"].lower() == "nifty midcap 150"), None)
        sml250 = next((ix for ix in pr_data["indices"] if ix["index_name"].lower() == "nifty smallcap 250"), None)

        for ix in pr_data["indices"]:
            idx_inserts.append((
                d_iso, ix["index_name"], ix["prev_close"], ix["open"], ix["high"],
                ix["low"], ix["close"], ix["turnover_inr"], "NSE_PR_ARCHIVE",
            ))

        mcap_map = pr_data.get("mcap") or {}
        raw_bytes = gzip.decompress(sb_path.read_bytes())
        if raw_bytes.startswith(b"PK"):
            zf = zipfile.ZipFile(io.BytesIO(raw_bytes))
            ss = []
            if "xl/sharedStrings.xml" in zf.namelist():
                root_ss = ET.fromstring(zf.read("xl/sharedStrings.xml"))
                for si in root_ss:
                    ss.append("".join(t.text or "" for t in si.iter() if t.tag.endswith("}t")))
            sh_name = next((n for n in zf.namelist() if n.startswith("xl/worksheets/sheet") and n.endswith(".xml")), "xl/worksheets/sheet1.xml")
            sh_root = ET.fromstring(zf.read(sh_name))
            hdr_cols: list[str] = []
            parsed_rows: list[dict[str, str]] = []
            for xrow in sh_root.iter():
                if not xrow.tag.endswith("}row"):
                    continue
                vals = []
                for xcell in xrow:
                    ct = xcell.attrib.get("t", "")
                    vx = next((ch.text for ch in xcell if ch.tag.endswith("}v")), "")
                    vals.append(ss[int(vx)] if (ct == "s" and vx != "") else (vx or ""))
                if not hdr_cols:
                    hdr_cols = [str(v).strip() for v in vals]
                elif vals:
                    parsed_rows.append({hdr_cols[idx]: str(vals[idx]).strip() if idx < len(vals) else "" for idx in range(len(hdr_cols))})
            rdr_iter: Any = parsed_rows
        else:
            raw_csv_txt = raw_bytes.decode("utf-8", "replace")
            rdr_iter = csv.DictReader(io.StringIO(raw_csv_txt, newline=None))

        seen_day_keys: set[tuple[str, str]] = set()
        eq_syms_today = 0
        total_rows_today = 0

        for row in rdr_iter:
            total_rows_today += 1
            c = {str(k).strip(): str(v).strip() if v is not None else "" for k, v in row.items() if k is not None}
            sym = (c.get("SYMBOL") or "").upper()
            srs = (c.get("SERIES") or "").upper()
            if not sym or srs not in EQUITY_SERIES_ELIGIBLE:
                non_equity_rows_skipped += 1
                continue

            if (sym, srs) in seen_day_keys:
                duplicate_records.append((d_iso, sym, srs))
                continue
            seen_day_keys.add((sym, srs))
            eq_syms_today += 1

            bar_dt = parse_nse_date(c.get("DATE1"))
            op = _safe_float(c.get("OPEN_PRICE"))
            hi = _safe_float(c.get("HIGH_PRICE"))
            lo = _safe_float(c.get("LOW_PRICE"))
            cl = _safe_float(c.get("CLOSE_PRICE"))
            vol = _safe_int(c.get("TTL_TRD_QNTY"))
            is_valid, flags = validate_raw_bar_fields(
                expected_date=d_iso, bar_date=bar_dt, open_px=op, high_px=hi, low_px=lo, close_px=cl, volume=vol
            )

            prev_cl = _safe_float(c.get("PREV_CLOSE"))
            last_px = _safe_float(c.get("LAST_PRICE"))
            vwap = _safe_float(c.get("AVG_PRICE"))
            t_lacs = _safe_float(c.get("TURNOVER_LACS")) or 0.0
            t_inr = round(t_lacs * 100_000.0, 2)
            n_trd = _safe_int(c.get("NO_OF_TRADES")) or 0
            deliv_q = _safe_int(c.get("DELIV_QTY"))
            deliv_p = _safe_float(c.get("DELIV_PER"))

            sec_id = _resolve_sec_id(sym)
            fac = _adj_factor_for(sec_id, d_iso) if is_valid else 1.0
            o_val = op if op is not None else 0.0
            h_val = hi if hi is not None else 0.0
            l_val = lo if lo is not None else 0.0
            c_val = cl if cl is not None else 0.0

            mcap_entry = mcap_map.get(f"{sym}|{srs}") or mcap_map.get(f"{sym}|EQ") or {}
            mcap_inr = mcap_entry.get("mcap_inr")
            issue_size = mcap_entry.get("issue_size")
            ex_cat = mcap_entry.get("category") or None
            mcap_src = "NSE_OFFICIAL_MCAP" if mcap_inr is not None else "UNAVAILABLE_PRE_2024_UDIFF"

            bars_batch.append((
                d_iso, sym, srs, sec_id, prev_cl, o_val, h_val, l_val, c_val,
                last_px, vwap, int(vol or 0), t_inr, n_trd, deliv_q, deliv_p,
                round(fac, 8), round(o_val / fac, 4), round(h_val / fac, 4),
                round(l_val / fac, 4), round(c_val / fac, 4),
                mcap_inr, issue_size, mcap_src, ex_cat,
                ",".join(flags), 1 if is_valid else 0,
            ))

            st_ob = obs_stats.setdefault(sec_id, {"first": d_iso, "last": d_iso, "count": 0, "syms": set(), "series": Counter()})
            if d_iso < st_ob["first"]:
                st_ob["first"] = d_iso
            if d_iso > st_ob["last"]:
                st_ob["last"] = d_iso
            st_ob["count"] += 1
            st_ob["syms"].add(sym)
            st_ob["series"][srs] += 1

        if len(bars_batch) >= 35_000:
            _flush_bars()

        meta = archive_status.get(d_iso, {})
        cal_inserts.append((
            d_iso,
            n50["open"] if n50 else None,
            n50["high"] if n50 else None,
            n50["low"] if n50 else None,
            n50["close"] if n50 else None,
            n500["close"] if n500 else None,
            mid150["close"] if mid150 else None,
            sml250["close"] if sml250 else None,
            eq_syms_today, total_rows_today,
            1 if pr_data.get("has_official_mcap") else 0,
            meta.get("sb_url"), meta.get("pr_url"), retrieved_at,
        ))
    _flush_bars()

    # 7. Populate security_master, symbol_history, name_history, corporate_actions, calendar & index tables
    # If incremental, refresh full obs_stats from DB to avoid truncating older history counts
    if incremental:
        for r in conn.execute(
            "SELECT sec_id, MIN(trade_date) AS mn, MAX(trade_date) AS mx, COUNT(*) AS cnt FROM daily_bars GROUP BY sec_id"
        ):
            st_ob = obs_stats.setdefault(r["sec_id"], {"first": r["mn"], "last": r["mx"], "count": 0, "syms": set(), "series": Counter()})
            st_ob["first"] = r["mn"]
            st_ob["last"] = r["mx"]
            st_ob["count"] = int(r["cnt"])

    all_sec_ids: set[str] = set(obs_stats.keys())
    for sym in equity_l_by_sym:
        all_sec_ids.add(_resolve_sec_id(sym))
    for sym in delisted_by_sym:
        all_sec_ids.add(_resolve_sec_id(sym))
    for sym in etf_by_sym:
        all_sec_ids.add(_resolve_sec_id(sym))

    # Pre-index corporate actions company name fallback
    ca_comp_by_sec: dict[str, str] = {}
    for ca in ca_raw:
        csym = str(ca.get("symbol") or "").upper().strip()
        comp = str(ca.get("comp") or "").strip()
        if csym and comp:
            ca_comp_by_sec.setdefault(_resolve_sec_id(csym), comp)

    sec_inserts: list[tuple[Any, ...]] = []
    symhist_inserts: list[tuple[Any, ...]] = []
    latest_session = max(cal_dates) if cal_dates else end_date

    # Reverse lookup from sec_id -> EQUITY_L symbol
    sec_to_eq_sym = {_resolve_sec_id(s): s for s in equity_l_by_sym}
    sec_to_del_sym = {_resolve_sec_id(s): s for s in delisted_by_sym}
    sec_to_etf_sym = {_resolve_sec_id(s): s for s in etf_by_sym}

    for sec_id in sorted(all_sec_ids):
        eq_sym = sec_to_eq_sym.get(sec_id)
        del_sym = sec_to_del_sym.get(sec_id)
        etf_sym = sec_to_etf_sym.get(sec_id)
        ob = obs_stats.get(sec_id, {})
        eq_row = equity_l_by_sym.get(eq_sym or "", {})
        del_row = delisted_by_sym.get(del_sym or "", {})
        etf_row = etf_by_sym.get(etf_sym or "", {})

        if eq_sym:
            cur_sym = eq_sym
        elif etf_sym:
            cur_sym = etf_sym
        elif del_sym:
            cur_sym = del_sym
        elif sec_id.startswith("NSE_SYM:"):
            cur_sym = sec_id.split(":", 1)[1]
        else:
            syms_ob = sorted(ob.get("syms") or [])
            cur_sym = syms_ob[-1] if syms_ob else sec_id

        comp_name = (
            eq_row.get("NAME OF COMPANY")
            or etf_row.get("SecurityName")
            or del_row.get("Company")
            or ca_comp_by_sec.get(sec_id)
            or cur_sym
        )
        list_dt = parse_nse_date(eq_row.get("DATE OF LISTING") or etf_row.get("DateofListing"))
        delist_dt = parse_nse_date(del_row.get("Delisted Date"))
        delist_type = (del_row.get("Type of Delisting") or "").strip() or None
        isin_val = eq_row.get("ISIN NUMBER") or etf_row.get("ISINNumber") or (sec_id if not sec_id.startswith("NSE_SYM:") else None)
        last_seen = ob.get("last")
        first_seen = ob.get("first")
        srs_counter = ob.get("series") or Counter()
        dom_series = eq_row.get("SERIES") or (srs_counter.most_common(1)[0][0] if srs_counter else None)
        inst_type = "ETF" if (etf_sym and not eq_sym) else "EQUITY"

        # Honest lifecycle classification
        if delist_dt or del_sym:
            status = "DELISTED"
        elif inst_type == "ETF":
            status = "ACTIVE_ETF"
        elif eq_sym:
            # Listed in EQUITY_L, check if stopped trading months before latest_session
            if last_seen and last_seen < (cal_dates[-30] if len(cal_dates) >= 30 else latest_session):
                status = "SUSPENDED_OR_INACTIVE"
            else:
                status = "ACTIVE"
        else:
            status = "INACTIVE_OR_DELISTED_MERGED"

        sec_ind, _ = sector_by_sym.get(cur_sym, (None, None))
        if inst_type == "ETF" and not sec_ind:
            sec_ind = etf_row.get("Underlying Asset") or "ETF"
        sec_inserts.append((
            sec_id, isin_val, cur_sym, comp_name, dom_series, list_dt, delist_dt, delist_type,
            status, inst_type, _safe_float(eq_row.get("FACE VALUE") or etf_row.get("FaceValue")),
            _safe_float(eq_row.get("PAID UP VALUE")),
            _safe_int(eq_row.get("MARKET LOT") or etf_row.get("MarketLot")),
            sec_ind or ("ETF" if inst_type == "ETF" else "Unclassified (Historical/Non-Total-Market)"),
            sec_ind or ("ETF" if inst_type == "ETF" else "Unclassified"),
            "NSE_EQUITY_L" if eq_sym else ("NSE_ETF_LIST" if etf_sym else ("NSE_DELISTED" if del_sym else "NSE_BHAVCOPY_OBSERVED")),
            first_seen, last_seen, int(ob.get("count", 0)), retrieved_at,
        ))

        # Build disjoint SCD2 symbol history intervals
        term_sym = alias_to_term.get(cur_sym, cur_sym)
        chain = sorted(sym_groups.get(term_sym, []), key=lambda x: x[2])
        if chain:
            from datetime import date, timedelta
            cur_start = list_dt or "1990-01-01"
            for old_s, new_s, eff_dt in chain:
                try:
                    prev_day = (date.fromisoformat(eff_dt) - timedelta(days=1)).isoformat()
                except ValueError:
                    prev_day = eff_dt
                if prev_day < cur_start:
                    prev_day = cur_start
                symhist_inserts.append((sec_id, old_s, cur_start, prev_day, "SYMBOL_CHANGE", old_s, new_s, "NSE_SYMBOLCHANGE"))
                cur_start = eff_dt
            symhist_inserts.append((sec_id, term_sym, cur_start, delist_dt or "9999-12-31", "CURRENT_SYMBOL", chain[-1][0], term_sym, "NSE_SYMBOLCHANGE"))
        else:
            symhist_inserts.append((sec_id, cur_sym, list_dt or (first_seen or "1990-01-01"), delist_dt or "9999-12-31", "PRIMARY_LISTING", None, cur_sym, "NSE_SECURITY_MASTER"))

    conn.executemany(
        """
        INSERT OR REPLACE INTO trading_calendar (
            trade_date, nifty50_open, nifty50_high, nifty50_low, nifty50_close,
            nifty500_close, midcap150_close, smallcap250_close,
            eq_symbols_count, total_bhav_rows, has_official_mcap,
            bhav_source_url, pr_source_url, retrieved_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        cal_inserts,
    )
    conn.executemany(
        "INSERT OR REPLACE INTO daily_index_bars (trade_date, index_name, prev_close, open, high, low, close, turnover_inr, source_provider) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        idx_inserts,
    )
    conn.execute("DELETE FROM security_master")
    conn.executemany(
        """
        INSERT INTO security_master (
            sec_id, isin, current_symbol, company_name, series, listing_date, delisting_date,
            delisting_type, status, instrument_type, face_value, paid_up_value, market_lot, sector, industry,
            source_primary, first_seen_date, last_seen_date, total_sessions_observed, retrieved_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        sec_inserts,
    )
    conn.execute("DELETE FROM symbol_history")
    conn.executemany(
        "INSERT INTO symbol_history (sec_id, symbol, effective_from, effective_to, event_type, old_symbol, new_symbol, source_provider) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        symhist_inserts,
    )
    if not incremental:
        conn.executemany(
            "INSERT INTO name_history (symbol, old_name, new_name, effective_date, source_provider) VALUES (?, ?, ?, ?, ?)",
            [
                (
                    (r.get("SYMBOL") or "").upper().strip(),
                    r.get("OLD NAME OF THE COMPANY", ""),
                    r.get("NEW NAME OF THE COMPANY", ""),
                    parse_nse_date(r.get("EFFECTIVE DATE")) or "",
                    "NSE_NAMECHANGE",
                )
                for r in masters["namechange"] if r.get("SYMBOL")
            ],
        )
        conn.executemany(
            """
            INSERT OR REPLACE INTO corporate_actions (
                ca_id, sec_id, symbol, isin, ex_date, record_date, series, action_type,
                ratio_num, ratio_den, adj_factor, cash_dividend_inr, rights_premium_inr,
                face_value, subject, source_provider
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ca_inserts,
        )
        conn.executemany(
            "INSERT OR REPLACE INTO index_snapshots (index_name, symbol, isin, company_name, industry, series, snapshot_as_of_date, is_static_snapshot, source_url) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            snap_rows,
        )
    conn.commit()

    # 8. Build (sec_id, trade_date) index first for fast ordered scans, run DQ & Causal Liquidity, then build remaining indexes
    conn.execute("CREATE INDEX IF NOT EXISTS idx_bars_sec_date ON daily_bars(sec_id, trade_date)")
    conn.commit()
    dq_summary = run_post_ingestion_dq_audit(
        conn, version_id=version_id, missing_calendar_dates=missing_dates, duplicate_records=duplicate_records
    )
    discrepancies = audit_nse_vs_yahoo_sample(conn, yf_sample)
    liq_summary = compute_and_store_pit_liquidity(conn)
    create_m3_indexes(conn)

    row_counts = {
        "trading_calendar": conn.execute("SELECT COUNT(*) FROM trading_calendar").fetchone()[0],
        "security_master": conn.execute("SELECT COUNT(*) FROM security_master").fetchone()[0],
        "symbol_history": conn.execute("SELECT COUNT(*) FROM symbol_history").fetchone()[0],
        "name_history": conn.execute("SELECT COUNT(*) FROM name_history").fetchone()[0],
        "daily_bars": conn.execute("SELECT COUNT(*) FROM daily_bars").fetchone()[0],
        "daily_bars_valid": conn.execute("SELECT COUNT(*) FROM daily_bars WHERE is_valid_bar = 1").fetchone()[0],
        "corporate_actions": conn.execute("SELECT COUNT(*) FROM corporate_actions").fetchone()[0],
        "daily_liquidity": conn.execute("SELECT COUNT(*) FROM daily_liquidity").fetchone()[0],
        "daily_index_bars": conn.execute("SELECT COUNT(*) FROM daily_index_bars").fetchone()[0],
        "data_quality_events": conn.execute("SELECT COUNT(*) FROM data_quality_events").fetchone()[0],
        "non_equity_bhav_rows_filtered": non_equity_rows_skipped,
    }
    checksum_src = json.dumps(row_counts, sort_keys=True) + f"|{start_date}|{end_date}"
    checksum = hashlib.sha256(checksum_src.encode()).hexdigest()

    conn.execute(
        """
        INSERT INTO dataset_versions (
            version_id, snapshot_tag, created_at, date_start, date_end,
            parent_version_id, is_incremental, source_manifest_json, row_counts_json, checksum_sha256, notes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            version_id,
            snapshot_tag or f"NSE_PIT_{start_date}_TO_{end_date}",
            retrieved_at,
            start_date,
            end_date,
            parent_version_id,
            1 if incremental else 0,
            json.dumps(master_bundle["manifests"]),
            json.dumps(row_counts),
            checksum,
            "Authoritative NSE Bhavcopy + PR Archives + Security Master + ETF Master + CA API + Yahoo Finance verification.",
        ),
    )
    conn.commit()
    conn.close()
    if staging_db != db_path:
        for suf in ("", "-wal", "-shm"):
            p_old = Path(f"{db_path}{suf}")
            if p_old.exists():
                p_old.unlink()
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with staging_db.open("rb") as f_in, db_path.open("wb") as f_out:
            shutil.copyfileobj(f_in, f_out, length=4 * 1024 * 1024)
        staging_db.unlink(missing_ok=True)

    # 9. Export Versioned Portable Artifacts & Comprehensive Coverage / Documentation Reports
    artifacts = _export_m3_reports_and_snapshots(
        db_path=db_path,
        data_dir=data_dir,
        reports_dir=reports_dir,
        version_id=version_id,
        row_counts=row_counts,
        dq_summary=dq_summary,
        discrepancies=discrepancies,
        liq_summary=liq_summary,
        start_date=start_date,
        end_date=end_date,
        retrieved_at=retrieved_at,
        checksum=checksum,
    )
    return {
        "version_id": version_id,
        "row_counts": row_counts,
        "dq_summary": dq_summary,
        "liq_summary": liq_summary,
        "artifacts": artifacts,
    }


def _export_m3_reports_and_snapshots(
    *,
    db_path: Path,
    data_dir: Path,
    reports_dir: Path,
    version_id: str,
    row_counts: dict[str, int],
    dq_summary: dict[str, Any],
    discrepancies: list[dict[str, Any]],
    liq_summary: dict[str, int],
    start_date: str,
    end_date: str,
    retrieved_at: str,
    checksum: str,
) -> dict[str, str]:
    data_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    conn = connect_m3_db(db_path)
    store = PITUniverseStore(db_path)

    # Portable compressed snapshots of Security Master, Corporate Actions & Version Registry
    sec_rows = [dict(r) for r in conn.execute("SELECT * FROM security_master ORDER BY current_symbol")]
    (data_dir / "security_master.json.gz").write_bytes(gzip.compress(json.dumps(sec_rows).encode("utf-8")))

    ca_rows = [dict(r) for r in conn.execute("SELECT * FROM corporate_actions ORDER BY ex_date, symbol")]
    (data_dir / "corporate_actions.json.gz").write_bytes(gzip.compress(json.dumps(ca_rows).encode("utf-8")))

    ver_rows = [dict(r) for r in conn.execute("SELECT * FROM dataset_versions ORDER BY created_at DESC")]
    (data_dir / "dataset_version_registry.json").write_text(json.dumps(ver_rows, indent=2), encoding="utf-8")

    # Coverage Breakdown & Multi-Period Sample PIT Audit (early 2021 -> 2026)
    status_counts = {r["status"]: r["cnt"] for r in conn.execute("SELECT status, COUNT(*) AS cnt FROM security_master GROUP BY status")}
    ca_counts = {r["action_type"]: r["cnt"] for r in conn.execute("SELECT action_type, COUNT(*) AS cnt FROM corporate_actions GROUP BY action_type")}
    # Single pass over daily_bars using covering index idx_bars_date_valid or minimal columns
    series_counts: dict[str, int] = {}
    for r in conn.execute("SELECT series, COUNT(*) AS cnt FROM daily_bars GROUP BY series"):
        series_counts[r["series"]] = r["cnt"]
    mcap_from_row = conn.execute("SELECT MIN(trade_date) AS mcap_from FROM trading_calendar WHERE has_official_mcap = 1").fetchone()
    mcap_from_dt = mcap_from_row["mcap_from"] if mcap_from_row else None
    mcap_cnt_row = conn.execute(
        "SELECT COUNT(*) AS has_mcap FROM daily_bars WHERE trade_date >= ? AND mcap_inr IS NOT NULL",
        (mcap_from_dt or "2024-01-01",),
    ).fetchone()
    mcap_coverage = {"tot": row_counts["daily_bars"], "has_mcap": mcap_cnt_row["has_mcap"], "mcap_from": mcap_from_dt}
    # Sample delivery presence check on daily_bars using idx_bars_date_valid + exact count of valid EQ/BE/BZ delivery bars
    deliv_cnt_row = conn.execute("SELECT COUNT(*) AS has_deliv FROM daily_bars WHERE is_valid_bar = 1 AND delivery_qty IS NOT NULL").fetchone()
    deliv_coverage = {"tot": row_counts["daily_bars"], "has_deliv": deliv_cnt_row["has_deliv"]}

    all_dates = [r["trade_date"] for r in conn.execute("SELECT trade_date FROM trading_calendar ORDER BY trade_date")]
    probe_dates = []
    for target_dt in ("2021-03-15", "2022-03-15", "2023-03-15", "2024-03-15", "2025-03-17", "2026-11-18"):
        dt_match = next((d for d in reversed(all_dates) if d <= target_dt), None)
        if dt_match and dt_match not in probe_dates:
            probe_dates.append(dt_match)

    pit_audit_periods = []
    for dt in probe_dates:
        u_5cr = store.get_pit_universe(dt, min_adv_20d_inr=50_000_000.0, min_trading_freq_60d=0.80)
        u_50cr = store.get_pit_universe(dt, min_adv_20d_inr=500_000_000.0, min_trading_freq_60d=0.90)
        active_day = conn.execute("SELECT COUNT(*) FROM daily_bars WHERE trade_date = ? AND is_valid_bar = 1", (dt,)).fetchone()[0]
        del_hist = [x["symbol"] for x in u_5cr if x["status"] in ("DELISTED", "INACTIVE_OR_DELISTED_MERGED")]
        pit_audit_periods.append({
            "as_of_date": dt,
            "total_nse_equities_trading_that_day": active_day,
            "pit_eligible_broad_adv_5cr": len(u_5cr),
            "pit_eligible_core_adv_50cr": len(u_50cr),
            "subsequently_delisted_or_merged_present_pit": len(del_hist),
            "delisted_merged_examples": del_hist[:15],
            "top5_by_adv": [{"symbol": x["symbol"], "sec_id": x["sec_id"], "adv_20d_cr": round(x["adv_20d_inr"] / 1e7, 2)} for x in u_5cr[:5]],
        })

    # Specific Case Studies (Delisted/Merged Retention & Symbol Change Continuity & Split Adjustment)
    case_studies = {}
    for sym in ("HDFC", "MINDTREE", "CADILAHC", "ZYDUSLIFE", "MOTHERSUMI", "MOTHERSON", "IRCTC", "IEX", "JETAIRWAYS", "SUZLON", "YESBANK"):
        r = conn.execute(
            "SELECT sec_id, current_symbol, status, listing_date, delisting_date, first_seen_date, last_seen_date, total_sessions_observed FROM security_master WHERE current_symbol = ? OR sec_id IN (SELECT sec_id FROM symbol_history WHERE symbol = ?) LIMIT 1",
            (sym, sym),
        ).fetchone()
        bars_cnt = conn.execute("SELECT COUNT(*) AS n, MIN(trade_date) AS mn, MAX(trade_date) AS mx FROM daily_bars WHERE symbol = ?", (sym,)).fetchone()
        case_studies[sym] = {
            "master": dict(r) if r else None,
            "observed_as_symbol_bars": dict(bars_cnt) if bars_cnt else None,
        }

    known_limitations = [
        {
            "id": "LIM-01",
            "area": "Historical Index Membership",
            "limitation": "NSE provides current constituent snapshots (ind_nifty50list.csv, ind_nifty500list.csv, ind_niftytotalmarket_list.csv) as of retrieval date, not free point-in-time constituent change ledgers back to 2021.",
            "mitigation": "Static index CSVs are explicitly isolated in `index_snapshots (is_static_snapshot = 1)` and NEVER used for PIT eligibility. PIT selection instead uses actual daily NSE trading appearance, listing_date <= t, delisting_date > t, and strictly trailing 20D/60D ADV & trading frequency.",
        },
        {
            "id": "LIM-02",
            "area": "Official Daily Market Capitalisation Pre-July 2024",
            "limitation": "NSE began embedding `MCAP{ddmmyyyy}.csv` inside daily `PR.zip` with UDiFF migration in mid-2024. Sessions prior to that lack official NSE daily market-cap files inside `PR.zip`.",
            "mitigation": "Pre-UDiFF sessions store `mcap_inr = NULL` (`mcap_source = UNAVAILABLE_PRE_2024_UDIFF`) rather than fabricating backfilled market cap from today's share count.",
        },
        {
            "id": "LIM-03",
            "area": "Pre-2021 Delisted Companies",
            "limitation": "The dataset spans `2021-01-01` to `2026-11-19`. Companies delisted prior to 2021-01-01 (e.g., DHFL suspended/delisted earlier) remain in `security_master` (`delisted.csv`) but have 0 active trading sessions inside the 2021–2026 Bhavcopy window.",
            "mitigation": "Companies delisted/merged during 2021–2026 (e.g., HDFC, MINDTREE, INFRATEL, etc.) are fully retained with their real daily prices up to their exact final session.",
        },
        {
            "id": "LIM-04",
            "area": "Rights Issues & Demerger Price Adjustments",
            "limitation": "Splits and Bonuses compute exact deterministic adjustment factors (`old_fv/new_fv`, `(A+B)/B`). Rights issues (`RIGHTS`) and Demergers (`DEMERGER_RESTRUCTURE`, e.g., TATAMOTORS -> TMPV/TMCV or JIOFIN spin-off) depend on ex-date fair-value ratios not fixed strictly by ratio alone.",
            "mitigation": "Rights & Demergers are cataloged in `corporate_actions` (`adj_factor = 1.0`) and automatically exempt post-event bars from false `UNEXPLAINED_JUMP_SHOCK` alerts while remaining flagged for strategy-level corporate-event handling.",
        },
        {
            "id": "LIM-05",
            "area": "Bid-Ask Spread Proxy",
            "limitation": "EOD Exchange Bhavcopy does not contain intraday Level-1 best bid/ask quotes.",
            "mitigation": "Liquidity engine provides trailing 20D median `(High - Low) / Close * 100` (`hl_spread_proxy_pct`) alongside exact NSE `DELIV_QTY`, `DELIV_PER`, `NO_OF_TRADES`, and `TURNOVER_INR`.",
        },
        {
            "id": "LIM-06",
            "area": "Point-in-Time Fundamentals Not In Bhavcopy",
            "limitation": "Quarterly balance sheet / P&L / shareholding filings are separate XBRL corporate filings outside daily Bhavcopy archives.",
            "mitigation": "Moreshwar 3.0 Prompt 1 establishes the price, volume, delivery, market cap, corporate action, and PIT universe backbone; fundamental PIT ingestion should be added separately with filing-date lags if used.",
        },
    ]

    coverage_report = {
        "version_id": version_id,
        "generated_at": retrieved_at,
        "window": {"start_date": start_date, "end_date": end_date, "trading_sessions": row_counts["trading_calendar"]},
        "checksum_sha256": checksum,
        "row_counts": row_counts,
        "security_master_status_breakdown": status_counts,
        "daily_bars_series_breakdown": series_counts,
        "corporate_actions_breakdown": ca_counts,
        "delivery_data_coverage": {
            "total_bars": deliv_coverage["tot"],
            "bars_with_delivery_qty": deliv_coverage["has_deliv"],
            "pct": round(100.0 * (deliv_coverage["has_deliv"] or 0) / max(1, deliv_coverage["tot"]), 2),
        },
        "mcap_coverage": {
            "bars_with_official_nse_mcap": mcap_coverage["has_mcap"],
            "official_mcap_start_date": mcap_coverage["mcap_from"],
            "pct_of_all_bars": round(100.0 * (mcap_coverage["has_mcap"] or 0) / max(1, mcap_coverage["tot"]), 2),
        },
        "pit_universe_sample_audit": pit_audit_periods,
        "case_studies": case_studies,
        "cross_provider_verification_count": len(discrepancies),
        "known_limitations": known_limitations,
    }

    (reports_dir / "data_coverage_report.json").write_text(json.dumps(coverage_report, indent=2), encoding="utf-8")
    (reports_dir / "data_quality_report.json").write_text(json.dumps({
        **dq_summary,
        "source_discrepancies": discrepancies,
    }, indent=2), encoding="utf-8")
    (reports_dir / "sample_pit_audit.json").write_text(json.dumps({
        "periods": pit_audit_periods,
        "case_studies": case_studies,
    }, indent=2), encoding="utf-8")
    (reports_dir / "known_limitations.json").write_text(json.dumps(known_limitations, indent=2), encoding="utf-8")

    # Write Markdown Coverage Report & Dataset Documentation cleanly from observed stats
    _write_m3_markdown_docs(reports_dir=reports_dir, cov=coverage_report, dq=dq_summary, discrepancies=discrepancies)
    conn.close()

    return {
        "db_path": str(db_path),
        "version_registry": str(data_dir / "dataset_version_registry.json"),
        "security_master_gz": str(data_dir / "security_master.json.gz"),
        "corporate_actions_gz": str(data_dir / "corporate_actions.json.gz"),
        "coverage_json": str(reports_dir / "data_coverage_report.json"),
        "quality_json": str(reports_dir / "data_quality_report.json"),
        "coverage_md": str(reports_dir / "DATA_COVERAGE_REPORT.md"),
        "docs_md": str(reports_dir / "DATASET_DOCUMENTATION.md"),
    }


def _write_m3_markdown_docs(
    *,
    reports_dir: Path,
    cov: dict[str, Any],
    dq: dict[str, Any],
    discrepancies: list[dict[str, Any]],
) -> None:
    rc = cov["row_counts"]
    sm_st = cov["security_master_status_breakdown"]
    srs_bk = cov["daily_bars_series_breakdown"]
    ca_bk = cov["corporate_actions_breakdown"]
    dq_c = dq["summary_counts"]

    cov_lines = [
        "# MORESHWAR 3.0 — POINT-IN-TIME NSE UNIVERSE DATA COVERAGE REPORT",
        "",
        f"- **Dataset Version ID:** `{cov['version_id']}`",
        f"- **Window:** `{cov['window']['start_date']}` → `{cov['window']['end_date']}` (**{cov['window']['trading_sessions']}** NSE trading sessions)",
        f"- **Generated At (UTC):** `{cov['generated_at']}`",
        f"- **Checksum SHA-256:** `{cov['checksum_sha256']}`",
        "",
        "## 1. Universe Scale vs Moreshwar 2.1 / 2.2",
        "",
        "| Metric | Moreshwar 2.1 / 2.2 (Frozen) | Moreshwar 3.0 PIT Universe |",
        "| :--- | :---: | :---: |",
        f"| Total Securities in Security Master | `40` (`36` full-history INR) | **`{rc['security_master']}`** |",
        f"| Validated Daily Equity Bars | ~`43,000` | **`{rc['daily_bars_valid']:,}`** (`{rc['daily_bars']:,}` total) |",
        f"| Trading Sessions | `1,176` | **`{rc['trading_calendar']}`** |",
        f"| Corporate Actions Ledger | None (provider adj only) | **`{rc['corporate_actions']:,}`** events |",
        f"| Delivery Qty & % Coverage | Unavailable | **`{cov['delivery_data_coverage']['pct']}%`** (`{cov['delivery_data_coverage']['bars_with_delivery_qty']:,}` bars) |",
        f"| Official NSE Daily Market Cap | Unavailable | **`{cov['mcap_coverage']['bars_with_official_nse_mcap']:,}`** bars (from `{cov['mcap_coverage']['official_mcap_start_date']}`) |",
        "",
        "## 2. Security Master Status & Series Coverage",
        "",
        "| Lifecycle Status | Count | Series | Daily Bars |",
        "| :--- | ---: | :--- | ---: |",
    ]
    sm_items = sorted(sm_st.items(), key=lambda x: -x[1])
    srs_items = sorted(srs_bk.items(), key=lambda x: -x[1])
    for i in range(max(len(sm_items), len(srs_items))):
        k1, v1 = sm_items[i] if i < len(sm_items) else ("—", "")
        k2, v2 = srs_items[i] if i < len(srs_items) else ("—", "")
        cov_lines.append(f"| `{k1}` | {v1} | `{k2}` | {f'{v2:,}' if isinstance(v2, int) else v2} |")

    cov_lines += [
        "",
        "## 3. Corporate Actions Database Breakdown",
        "",
        "| Action Type | Count |",
        "| :--- | ---: |",
    ]
    for k, v in sorted(ca_bk.items(), key=lambda x: -x[1]):
        cov_lines.append(f"| `{k}` | {v:,} |")

    cov_lines += [
        "",
        "## 4. Point-in-Time Universe Sample Audit Across Regimes (2021–2026)",
        "",
        "| As-Of Date | Active NSE Equities | Broad Liquid PIT (ADV ≥ ₹5 Cr) | Core Liquid PIT (ADV ≥ ₹50 Cr) | Delisted/Merged Retained at Date | Examples Retained |",
        "| :---: | ---: | ---: | ---: | ---: | :--- |",
    ]
    for p in cov["pit_universe_sample_audit"]:
        ex = ", ".join(p["delisted_merged_examples"][:6]) or "—"
        cov_lines.append(
            f"| `{p['as_of_date']}` | {p['total_nse_equities_trading_that_day']:,} | **{p['pit_eligible_broad_adv_5cr']:,}** | **{p['pit_eligible_core_adv_50cr']:,}** | {p['subsequently_delisted_or_merged_present_pit']} | `{ex}` |"
        )

    cov_lines += [
        "",
        "## 5. Data-Quality Audit Summary",
        "",
        "| Check Code | Count | Policy |",
        "| :--- | ---: | :--- |",
    ]
    for k, v in sorted(dq_c.items(), key=lambda x: -x[1]):
        cov_lines.append(f"| `{k}` | {v:,} | Non-silent flag / exclusion logged |")

    cov_lines += [
        "",
        f"Cross-provider NSE vs Yahoo Finance sample verification checked **`{len(discrepancies)}`** active, split-adjusted, symbol-changed, and delisted tickers (details in `data_quality_report.json`).",
        "",
        "## 6. Known Limitations",
        "",
    ]
    for lim in cov["known_limitations"]:
        cov_lines.append(f"- **{lim['id']} ({lim['area']}):** {lim['limitation']} *Mitigation:* {lim['mitigation']}")

    (reports_dir / "DATA_COVERAGE_REPORT.md").write_text("\n".join(cov_lines) + "\n", encoding="utf-8")

    doc_lines = [
        "# MORESHWAR 3.0 — DATASET ARCHITECTURE & DOCUMENTATION",
        "",
        "## 1. Frozen Moreshwar 2.1 & 2.2 Protection",
        "- All Moreshwar 2.1 and 2.2 engines, `data/moreshwar_v2_*`, `data/expanded_history_5y_cache.json.gz`, and `reports/moreshwar_2_2/` are strictly frozen and unmodified.",
        "- Moreshwar 3.0 lives exclusively in `finsistant_v2/m3/`, `data/m3_universe/`, and `reports/moreshwar_3_0/`.",
        "",
        "## 2. Authoritative Data Sources",
        "1. **Security Master (`EQUITY_L.csv`, `delisted.csv`, `symbolchange.csv`, `namechange.csv`):** Official NSE India equity registers establishing listing dates, delisting dates/types, ISIN identifiers, face value, paid-up value, and SCD Type-2 symbol timelines.",
        "2. **Daily Exchange Market Data (`sec_bhavdata_full_DDMMYYYY.csv`):** Official NSE daily cash-market equity records with unadjusted O/H/L/C/PrevClose/VWAP, Traded Volume, Turnover (INR), No. of Trades, Delivery Volume (`DELIV_QTY`), and Delivery % (`DELIV_PER`).",
        "3. **Official Exchange PR Archives (`PRddmmyy.zip`):** Daily index OHLC (`Pd` / `Pr` CSVs) and official NSE daily Market Capitalisation & Issue Size (`MCAPddmmyyyy.csv`, available from UDiFF migration onward).",
        "4. **Corporate Actions (`corporates-corporateActions` API):** Official NSE Splits, Bonuses, Dividends, Rights Issues, and Demerger/Amalgamation notices keyed by `sec_id` + `ex_date`.",
        "5. **Secondary Verification (`Yahoo Finance Chart API`):** Cross-checks daily returns, splits, and delisted 404 probes on a stratified verification sample (`source_discrepancies`).",
        "",
        "## 3. Modular Schema (`finsistant_v2/m3/schema.py`)",
        "- `dataset_versions`: Immutable snapshot ledger with parent lineage & SHA-256 verification.",
        "- `trading_calendar`: Actual NSE market sessions with index OHLC and official MCAP availability flag.",
        "- `security_master`: Canonical `sec_id`-keyed master covering active, delisted, suspended, merged, and historical equities.",
        "- `symbol_history` & `name_history`: Slowly Changing Dimension Type-2 ticker intervals so historical lookups never confuse renamed companies.",
        "- `daily_bars`: Point-in-time unadjusted & split/bonus-adjusted OHLCV, delivery volume/%, turnover, official MCAP (`NULL` when unavailable before mid-2024), and `dq_flags`.",
        "- `corporate_actions`: Parsed Splits, Bonuses, Dividends, Rights, and Restructuring records.",
        "- `index_snapshots`: Static constituent lists explicitly tagged `is_static_snapshot = 1` (never substituted for PIT history).",
        "- `daily_liquidity`: Strictly causal (`<= trade_date`) rolling 20D/60D ADV, avg volume, turnover ratio, 60D trading frequency, 20D realized volatility, HL range spread proxy, and `pit_tier`.",
        "- `data_quality_events` & `source_discrepancies`: Full audit trail of every warning, excluded invalid bar, gap, or cross-provider variance.",
        "",
        "## 4. Usage & Incremental Updates",
        "```python",
        "from finsistant_v2.m3 import PITUniverseStore, build_m3_pit_universe",
        "",
        "# Query point-in-time liquid universe on any date (zero survivorship / zero look-ahead):",
        "store = PITUniverseStore()",
        "elig = store.get_pit_universe('2023-03-15', min_adv_20d_inr=50_000_000, min_trading_freq_60d=0.80)",
        "hist = store.get_security_history('ZYDUSLIFE', end_date='2023-03-15', adjusted=True)  # stitches CADILAHC -> ZYDUSLIFE!",
        "```",
        "",
        "## 5. Known Limitations",
    ]
    for lim in cov["known_limitations"]:
        doc_lines.append(f"- **{lim['id']} ({lim['area']}):** {lim['limitation']} {lim['mitigation']}")

    (reports_dir / "DATASET_DOCUMENTATION.md").write_text("\n".join(doc_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    summary = build_m3_pit_universe()
    print(json.dumps({
        "version_id": summary["version_id"],
        "row_counts": summary["row_counts"],
        "dq_summary": summary["dq_summary"]["summary_counts"],
        "liq_summary": summary["liq_summary"],
        "artifacts": summary["artifacts"],
    }, indent=2))
