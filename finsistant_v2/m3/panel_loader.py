"""High-Speed Vectorized Point-in-Time Equity Panel Loader for Moreshwar 3.0 Research.

Materializes normalized `[T x N]` NumPy matrices strictly from the validated
Prompt 1 SQLite database (`data/m3_universe/nse_pit_universe.db`) into a fast
local cache (`/tmp/m3_pit_panel_v1.npz`), guaranteeing zero survivorship bias
and zero look-ahead across Prompts 2, 3, and 4.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .schema import DEFAULT_M3_DB_PATH, connect_m3_db

DEFAULT_PANEL_CACHE_PATH = Path("/tmp/m3_pit_panel_v1.npz")


def build_or_load_m3_panel(
    *,
    db_path: Path = DEFAULT_M3_DB_PATH,
    cache_path: Path = DEFAULT_PANEL_CACHE_PATH,
    force_rebuild: bool = False,
    min_bars_per_sec: int = 20,
) -> dict[str, Any]:
    """Load or build [T x N] vectorized PIT matrices from NSE SQLite DB."""
    if cache_path.exists() and not force_rebuild:
        npz = np.load(cache_path, allow_pickle=True)
        return {k: npz[k] for k in npz.files}

    conn = connect_m3_db(db_path)
    cal_rows = conn.execute(
        """
        SELECT trade_date, nifty50_open, nifty50_high, nifty50_low, nifty50_close,
               nifty500_close, midcap150_close, smallcap250_close
        FROM trading_calendar
        ORDER BY trade_date ASC
        """
    ).fetchall()
    dates = np.array([r["trade_date"] for r in cal_rows], dtype=object)
    t_len = len(dates)
    d_idx = {str(d): i for i, d in enumerate(dates)}

    nifty50_open = np.array([r["nifty50_open"] if r["nifty50_open"] is not None else np.nan for r in cal_rows], dtype=np.float64)
    nifty50_high = np.array([r["nifty50_high"] if r["nifty50_high"] is not None else np.nan for r in cal_rows], dtype=np.float64)
    nifty50_low = np.array([r["nifty50_low"] if r["nifty50_low"] is not None else np.nan for r in cal_rows], dtype=np.float64)
    nifty50_close = np.array([r["nifty50_close"] if r["nifty50_close"] is not None else np.nan for r in cal_rows], dtype=np.float64)
    nifty500_close = np.array([r["nifty500_close"] if r["nifty500_close"] is not None else np.nan for r in cal_rows], dtype=np.float64)
    midcap150_close = np.array([r["midcap150_close"] if r["midcap150_close"] is not None else np.nan for r in cal_rows], dtype=np.float64)
    smallcap250_close = np.array([r["smallcap250_close"] if r["smallcap250_close"] is not None else np.nan for r in cal_rows], dtype=np.float64)

    # Backfill any missing Nifty 50 index bars (e.g. post-Oct 2025 PR format shift) from cached ^NSEI raw calendar
    raw_nsei_path = db_path.parent / "raw_cache" / "nsei_calendar_raw.json"
    if np.isnan(nifty50_close).any() and raw_nsei_path.exists():
        import datetime as _dt
        import json as _json
        _ist = _dt.timezone(_dt.timedelta(hours=5, minutes=30))
        _raw = _json.loads(raw_nsei_path.read_text(encoding="utf-8"))
        _res = (_raw.get("chart", {}).get("result") or [{}])[0]
        _ts = _res.get("timestamp") or []
        _q = ((_res.get("indicators", {}).get("quote")) or [{}])[0]
        for _i, _t in enumerate(_ts):
            _dt_s = _dt.datetime.fromtimestamp(_t, _ist).strftime("%Y-%m-%d")
            _ti = d_idx.get(_dt_s)
            if _ti is not None and np.isnan(nifty50_close[_ti]):
                _c = (_q.get("close") or [None])[_i]
                if _c is not None:
                    nifty50_close[_ti] = float(_c)
                    nifty50_open[_ti] = float((_q.get("open") or [None])[_i] or _c)
                    nifty50_high[_ti] = float((_q.get("high") or [None])[_i] or _c)
                    nifty50_low[_ti] = float((_q.get("low") or [None])[_i] or _c)
    # Causal forward-fill for any single special session (e.g., Muhurat session)
    for _ti in range(1, t_len):
        if np.isnan(nifty50_close[_ti]) and not np.isnan(nifty50_close[_ti - 1]):
            nifty50_close[_ti] = nifty50_close[_ti - 1]
            nifty50_open[_ti] = nifty50_close[_ti - 1]
            nifty50_high[_ti] = nifty50_close[_ti - 1]
            nifty50_low[_ti] = nifty50_close[_ti - 1]

    # Select all OPERATING EQUITIES (excluding ETFs) that traded >= min_bars_per_sec in EQ/BE/BZ
    sec_rows = conn.execute(
        """
        SELECT sm.sec_id, sm.current_symbol, sm.sector, sm.industry, sm.status,
               sm.listing_date, sm.delisting_date, COUNT(*) AS bar_cnt
        FROM security_master sm
        JOIN daily_bars db ON db.sec_id = sm.sec_id
        WHERE sm.instrument_type = 'EQUITY'
          AND db.is_valid_bar = 1
          AND db.series IN ('EQ', 'BE', 'BZ')
        GROUP BY sm.sec_id
        HAVING bar_cnt >= ?
        ORDER BY sm.sec_id ASC
        """,
        (min_bars_per_sec,),
    ).fetchall()

    n_len = len(sec_rows)
    sec_ids = np.array([r["sec_id"] for r in sec_rows], dtype=object)
    current_symbols = np.array([r["current_symbol"] for r in sec_rows], dtype=object)
    sectors = np.array([r["sector"] or "Unclassified" for r in sec_rows], dtype=object)
    industries = np.array([r["industry"] or "Unclassified" for r in sec_rows], dtype=object)
    statuses = np.array([r["status"] for r in sec_rows], dtype=object)
    listing_dates = np.array([r["listing_date"] or "" for r in sec_rows], dtype=object)
    delisting_dates = np.array([r["delisting_date"] or "" for r in sec_rows], dtype=object)
    s_idx = {str(s): j for j, s in enumerate(sec_ids)}

    shape = (t_len, n_len)
    open_px = np.full(shape, np.nan, dtype=np.float32)
    high_px = np.full(shape, np.nan, dtype=np.float32)
    low_px = np.full(shape, np.nan, dtype=np.float32)
    close_px = np.full(shape, np.nan, dtype=np.float32)
    adj_open = np.full(shape, np.nan, dtype=np.float32)
    adj_high = np.full(shape, np.nan, dtype=np.float32)
    adj_low = np.full(shape, np.nan, dtype=np.float32)
    adj_close = np.full(shape, np.nan, dtype=np.float32)
    volume = np.zeros(shape, dtype=np.float64)
    turnover_inr = np.zeros(shape, dtype=np.float64)
    delivery_pct = np.full(shape, np.nan, dtype=np.float32)
    mcap_inr = np.full(shape, np.nan, dtype=np.float64)
    adv_20d_inr = np.full(shape, np.nan, dtype=np.float64)
    adv_60d_inr = np.full(shape, np.nan, dtype=np.float64)
    trading_freq_60d = np.zeros(shape, dtype=np.float32)
    realized_vol_20d = np.full(shape, np.nan, dtype=np.float32)
    hl_spread_proxy = np.full(shape, np.nan, dtype=np.float32)
    is_pit_elig = np.zeros(shape, dtype=bool)
    is_valid_eq = np.zeros(shape, dtype=bool)
    div_inr_ex = np.zeros(shape, dtype=np.float32)

    # Stream bars + liquidity join sequentially
    # Pre-index corporate dividend cash amounts on ex_date
    for cr in conn.execute(
        "SELECT sec_id, ex_date, cash_dividend_inr FROM corporate_actions WHERE action_type = 'DIVIDEND' AND cash_dividend_inr > 0"
    ):
        ti = d_idx.get(cr["ex_date"])
        sj = s_idx.get(cr["sec_id"])
        if ti is not None and sj is not None:
            div_inr_ex[ti, sj] += float(cr["cash_dividend_inr"] or 0.0)

    q_bars = conn.execute(
        """
        SELECT b.trade_date, b.sec_id, b.series, b.open, b.high, b.low, b.close,
               b.adj_open, b.adj_high, b.adj_low, b.adj_close,
               b.volume, b.turnover_inr, b.delivery_pct, b.mcap_inr,
               l.adv_20d_inr, l.adv_60d_inr, l.trading_freq_60d,
               l.realized_vol_20d_ann, l.hl_spread_proxy_pct, l.is_pit_eligible
        FROM daily_bars b
        LEFT JOIN daily_liquidity l
          ON l.trade_date = b.trade_date AND l.symbol = b.symbol AND l.series = b.series
        WHERE b.is_valid_bar = 1 AND b.series IN ('EQ', 'BE', 'BZ')
        """
    )
    # Priority: EQ > BE > BZ if same sec_id somehow has multiple series on same day
    series_prio = {"EQ": 3, "BE": 2, "BZ": 1}
    cur_prio = np.zeros(shape, dtype=np.int8)
    for r in q_bars:
        ti = d_idx.get(r["trade_date"])
        sj = s_idx.get(r["sec_id"])
        if ti is None or sj is None:
            continue
        sp = series_prio.get(r["series"], 0)
        if sp < cur_prio[ti, sj]:
            continue
        cur_prio[ti, sj] = sp

        open_px[ti, sj] = float(r["open"])
        high_px[ti, sj] = float(r["high"])
        low_px[ti, sj] = float(r["low"])
        close_px[ti, sj] = float(r["close"])
        adj_open[ti, sj] = float(r["adj_open"])
        adj_high[ti, sj] = float(r["adj_high"])
        adj_low[ti, sj] = float(r["adj_low"])
        adj_close[ti, sj] = float(r["adj_close"])
        volume[ti, sj] = float(r["volume"] or 0.0)
        turnover_inr[ti, sj] = float(r["turnover_inr"] or 0.0)
        if r["delivery_pct"] is not None:
            delivery_pct[ti, sj] = float(r["delivery_pct"])
        if r["mcap_inr"] is not None:
            mcap_inr[ti, sj] = float(r["mcap_inr"])
        if r["adv_20d_inr"] is not None:
            adv_20d_inr[ti, sj] = float(r["adv_20d_inr"])
            adv_60d_inr[ti, sj] = float(r["adv_60d_inr"] or 0.0)
            trading_freq_60d[ti, sj] = float(r["trading_freq_60d"] or 0.0)
            realized_vol_20d[ti, sj] = float(r["realized_vol_20d_ann"] or 0.0)
            hl_spread_proxy[ti, sj] = float(r["hl_spread_proxy_pct"] or 0.0)
        is_valid_eq[ti, sj] = True

        # Strict PIT eligibility: series in ('EQ','BE'), ADV >= 5 Cr, Freq >= 0.80, Close >= 10, listing <= dt < delisting
        dt_s = r["trade_date"]
        l_dt = str(listing_dates[sj])
        d_dt = str(delisting_dates[sj])
        listed_ok = (not l_dt) or (l_dt <= dt_s)
        not_delisted = (not d_dt) or (d_dt > dt_s)
        elig_ok = (
            bool(r["is_pit_eligible"])
            and r["series"] in ("EQ", "BE")
            and float(r["close"]) >= 10.0
            and listed_ok
            and not_delisted
        )
        is_pit_elig[ti, sj] = elig_ok

    conn.close()

    payload = {
        "dates": dates,
        "sec_ids": sec_ids,
        "current_symbols": current_symbols,
        "sectors": sectors,
        "industries": industries,
        "statuses": statuses,
        "listing_dates": listing_dates,
        "delisting_dates": delisting_dates,
        "nifty50_open": nifty50_open,
        "nifty50_high": nifty50_high,
        "nifty50_low": nifty50_low,
        "nifty50_close": nifty50_close,
        "nifty500_close": nifty500_close,
        "midcap150_close": midcap150_close,
        "smallcap250_close": smallcap250_close,
        "open_px": open_px,
        "high_px": high_px,
        "low_px": low_px,
        "close_px": close_px,
        "adj_open": adj_open,
        "adj_high": adj_high,
        "adj_low": adj_low,
        "adj_close": adj_close,
        "volume": volume,
        "turnover_inr": turnover_inr,
        "delivery_pct": delivery_pct,
        "mcap_inr": mcap_inr,
        "adv_20d_inr": adv_20d_inr,
        "adv_60d_inr": adv_60d_inr,
        "trading_freq_60d": trading_freq_60d,
        "realized_vol_20d": realized_vol_20d,
        "hl_spread_proxy": hl_spread_proxy,
        "div_inr_ex": div_inr_ex,
        "is_pit_elig": is_pit_elig,
        "is_valid_eq": is_valid_eq,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **payload)
    return payload
