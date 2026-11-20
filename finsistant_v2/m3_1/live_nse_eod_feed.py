"""Moreshwar 3.1 — Strictly Causal Post-Close NSE Daily EOD Market Data Provider.

Fetches genuine completed post-close daily OHLCV and dividend actions for:
- `NIFTYBEES.NS` (Core Equity ETF + SMA200 regime series)
- `LIQUIDBEES.NS` (Liquid ETF)
- Frozen active NSE liquid equity universe (`data/m3_1_paper_trading/nse_prospective_universe.json`)
  plus any currently open tactical holdings.

Non-negotiable integrity rules:
1. ZERO broker credentials (uses only permitted public Yahoo Finance EOD data via `yfinance`).
2. Strictly causal (`index <= target_date` only — zero lookahead).
3. Zero fabrication: if `NIFTYBEES.NS` has NO completed bar for `target_date`, returns `None`
   so `PaperTradingScheduler` stays in `WAITING_FOR_DATA` cleanly.
4. Implements the EXACT frozen Family E (`STRAT_E_DEFENSIVE_YIELD_LOWVOL`) factor definitions
   from `finsistant_v2/m3/strategy_families.py` without changing a single threshold or weight.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from finsistant_v2.m3_1.prospective_paper_trading_engine import (
    DEFAULT_PAPER_DATA_DIR,
    DIAGNOSTIC_INDEX_DB_END_DATE,
    PaperOnlyExecutionGuard,
)

DEFAULT_PROSPECTIVE_UNIVERSE_PATH = DEFAULT_PAPER_DATA_DIR / "nse_prospective_universe.json"
IST_OFFSET = timezone(timedelta(hours=5, minutes=30))


def current_ist_date_str() -> str:
    """Return today's calendar date in Asia/Kolkata (IST, UTC+05:30)."""
    return datetime.now(IST_OFFSET).date().isoformat()


def _pct_rank_dict(values: dict[str, float], *, ascending: bool = True) -> dict[str, float]:
    """Cross-sectional percentile rank in [0.0, 1.0] matching `strategy_families._cross_sectional_pct_rank`."""
    clean = [(k, float(v)) for k, v in values.items() if v is not None and not np.isnan(float(v)) and np.isfinite(float(v))]
    m = len(clean)
    if m < 5:
        return {}
    clean.sort(key=lambda kv: kv[1] if ascending else -kv[1])
    if m == 1:
        return {clean[0][0]: 1.0}
    return {k: idx / float(m - 1) for idx, (k, _) in enumerate(clean)}


def _extract_ticker_frame(raw_df: Any, ticker: str, single_ticker: bool = False) -> Any | None:
    """Safely extract single-ticker OHLCV DataFrame from `yf.download` multi-index or single-index output."""
    if raw_df is None or getattr(raw_df, "empty", True):
        return None
    try:
        import pandas as pd

        if isinstance(raw_df.columns, pd.MultiIndex):
            lvl0 = raw_df.columns.get_level_values(0)
            lvl1 = raw_df.columns.get_level_values(1)
            if ticker in lvl0:
                df = raw_df[ticker].copy()
            elif ticker in lvl1:
                df = raw_df.xs(ticker, axis=1, level=1).copy()
            elif single_ticker:
                df = raw_df.copy()
                df.columns = raw_df.columns.get_level_values(0)
            else:
                return None
        else:
            df = raw_df.copy()
        df = df.dropna(subset=["Open", "High", "Low", "Close"], how="any")
        if df.empty:
            return None
        # Normalize DatetimeIndex to YYYY-MM-DD strings (strictly date-keyed, deduplicated)
        idx_dates = [str(ts.date()) if hasattr(ts, "date") else str(ts)[:10] for ts in df.index]
        df = df.copy()
        df["_date_str"] = idx_dates
        df = df.drop_duplicates(subset=["_date_str"], keep="last").set_index("_date_str").sort_index()
        return df
    except Exception:
        return None


def compute_family_e_candidates_from_bars(
    *,
    target_date: str,
    nb_returns: np.ndarray,
    equity_histories: dict[str, dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], dict[tuple[str, str], float]]:
    """Pure, deterministic, strictly causal Family E factor & bar builder from per-stock history <= `target_date`.

    Separated from network I/O so it is 100% unit-testable offline!
    """
    eq_bars: dict[str, dict[str, Any]] = {}
    metrics_for_rank: dict[str, dict[str, float]] = {}
    ret60_map: dict[str, np.ndarray] = {}

    for sec_id, h in equity_histories.items():
        dates = list(h["dates"])
        if not dates or dates[-1] != target_date:
            continue
        o_arr = np.asarray(h["open"], dtype=np.float64)
        h_arr = np.asarray(h["high"], dtype=np.float64)
        l_arr = np.asarray(h["low"], dtype=np.float64)
        c_arr = np.asarray(h["close"], dtype=np.float64)
        v_arr = np.asarray(h["volume"], dtype=np.float64)
        div_arr = np.asarray(h.get("dividends", np.zeros(len(dates))), dtype=np.float64)

        n = len(c_arr)
        if n < 2 or c_arr[-1] <= 0 or o_arr[-1] <= 0:
            continue

        cur_o, cur_h, cur_l, cur_c, cur_v = float(o_arr[-1]), float(h_arr[-1]), float(l_arr[-1]), float(c_arr[-1]), float(v_arr[-1])
        prev_c = float(c_arr[-2]) if c_arr[-2] > 0 else cur_o
        rets = np.diff(c_arr) / np.maximum(1e-6, c_arr[:-1])

        # Rolling 20D realized daily volatility & ADV & HL range
        vol_20d = float(np.std(rets[-20:], ddof=1)) if len(rets) >= 20 else (float(np.std(rets, ddof=1)) if len(rets) >= 5 else 0.018)
        vol_20d = max(0.005, vol_20d) if np.isfinite(vol_20d) else 0.018
        adv_20d = float(np.mean(c_arr[-20:] * np.maximum(0.0, v_arr[-20:]))) if n >= 20 else float(np.mean(c_arr * np.maximum(0.0, v_arr)))
        hl_pct_series = np.maximum(0.0, (h_arr - l_arr) / np.maximum(1e-6, c_arr)) * 100.0
        hl_range_pct = float(np.mean(hl_pct_series[-10:])) if n >= 10 else float(hl_pct_series[-1])

        # True Range -> ATR14 & ATR20
        tr = np.zeros(n, dtype=np.float64)
        tr[0] = max(0.0, h_arr[0] - l_arr[0])
        for k in range(1, n):
            tr[k] = max(h_arr[k] - l_arr[k], abs(h_arr[k] - c_arr[k - 1]), abs(l_arr[k] - c_arr[k - 1]))
        atr_14 = float(np.mean(tr[-14:])) if n >= 14 else float(np.mean(tr))
        atr_20 = float(np.mean(tr[-20:])) if n >= 20 else atr_14

        sma_50 = float(np.mean(c_arr[-50:])) if n >= 50 else None
        low_20_prev = float(np.min(l_arr[-21:-1])) if n >= 21 else None
        high_252 = float(np.max(c_arr[-252:])) if n >= 60 else float(np.max(c_arr))
        dist252 = cur_c / max(1e-6, high_252)

        # Trailing 12-month cash dividend yield (strictly <= target_date)
        dt_obj = date.fromisoformat(target_date)
        cutoff_1y = (dt_obj - timedelta(days=365)).isoformat()
        ttm_div_sum = float(sum(float(div_arr[k]) for k in range(n) if dates[k] >= cutoff_1y and div_arr[k] > 0))
        div_yield_ttm = ttm_div_sum / max(1e-6, cur_c)
        div_today = float(div_arr[-1]) if div_arr[-1] > 0 else 0.0

        # Amihud illiquidity (20D) & Idiosyncratic vol vs NIFTYBEES (20D)
        if len(rets) >= 20:
            dv20 = np.maximum(1.0, c_arr[-20:] * np.maximum(1.0, v_arr[-20:]))
            amihud_20d = float(np.mean(np.abs(rets[-20:]) / dv20))
            m_len = min(20, len(rets), len(nb_returns))
            if m_len >= 10:
                sr = rets[-m_len:]
                mr = nb_returns[-m_len:]
                mv = float(np.var(mr))
                beta = float(np.cov(sr, mr)[0, 1] / mv) if mv > 1e-10 else 1.0
                resid = sr - beta * mr
                idio_vol_20d = float(np.std(resid, ddof=1))
            else:
                idio_vol_20d = vol_20d
        else:
            amihud_20d = float("nan")
            idio_vol_20d = float("nan")

        if len(rets) >= 60:
            ret60_map[sec_id] = rets[-60:]

        eq_bars[sec_id] = {
            "sec_id": sec_id,
            "symbol": str(h["symbol"]),
            "sector": str(h.get("sector", "UNKNOWN_SECTOR")),
            "industry": str(h.get("industry", "UNKNOWN_INDUSTRY")),
            "is_unknown_sector": bool(h.get("is_unknown_sector", False)),
            "open": round(cur_o, 4),
            "high": round(cur_h, 4),
            "low": round(cur_l, 4),
            "close": round(cur_c, 4),
            "prev_close": round(prev_c, 4),
            "volume": round(cur_v, 2),
            "adv_20d_inr": round(adv_20d, 2),
            "vol_20d": round(vol_20d, 6),
            "hl_range_pct": round(hl_range_pct, 4),
            "atr_14": round(atr_14, 4),
            "sma_50": round(sma_50, 4) if sma_50 is not None else None,
            "low_20_prev": round(low_20_prev, 4) if low_20_prev is not None else None,
            "div_per_adj_share": round(div_today, 6),
            "div_yield_ttm": round(div_yield_ttm, 6),
            "valid": True,
        }

        if n >= 60 and sma_50 is not None and adv_20d >= 20_000_000.0 and np.isfinite(amihud_20d) and np.isfinite(idio_vol_20d):
            metrics_for_rank[sec_id] = {
                "close": cur_c,
                "sma_50": sma_50,
                "atr_20": atr_20,
                "dist252": dist252,
                "div_yield_ttm": div_yield_ttm,
                "neg_vol_20d": -vol_20d,
                "neg_idio_vol_20d": -idio_vol_20d,
                "neg_amihud_20d": -amihud_20d,
            }

    # Cross-sectional percentile ranks (exact Family E formula!)
    r_div = _pct_rank_dict({k: v["div_yield_ttm"] for k, v in metrics_for_rank.items()})
    r_lvol = _pct_rank_dict({k: v["neg_vol_20d"] for k, v in metrics_for_rank.items()})
    r_ivol = _pct_rank_dict({k: v["neg_idio_vol_20d"] for k, v in metrics_for_rank.items()})
    r_dist = _pct_rank_dict({k: v["dist252"] for k, v in metrics_for_rank.items()})
    r_amihud = _pct_rank_dict({k: v["neg_amihud_20d"] for k, v in metrics_for_rank.items()})

    candidates: list[dict[str, Any]] = []
    for sec_id, m in metrics_for_rank.items():
        if sec_id not in r_div or sec_id not in r_lvol or sec_id not in r_amihud or sec_id not in r_dist or sec_id not in r_ivol:
            continue
        # Exact frozen Family E eligibility gates (finsistant_v2/m3/strategy_families.py L363-370)
        if not (
            m["dist252"] >= 0.84
            and m["close"] >= m["sma_50"] * 0.98
            and m["div_yield_ttm"] > 0.004
            and r_lvol[sec_id] >= 0.50
            and r_amihud[sec_id] >= 0.45
        ):
            continue
        score = (
            0.40 * r_div[sec_id]
            + 0.25 * r_lvol[sec_id]
            + 0.20 * r_dist[sec_id]
            + 0.15 * r_ivol[sec_id]
        )
        atr20 = m["atr_20"] if m["atr_20"] > 0 else m["close"] * 0.020
        stop_ratio = min(0.14, max(0.08, (4.5 * atr20) / max(1.0, m["close"])))
        candidates.append({
            "sec_id": sec_id,
            "symbol": eq_bars[sec_id]["symbol"],
            "strategy_id": "STRAT_E_DEFENSIVE_YIELD_LOWVOL",
            "score": round(float(score), 6),
            "stop_ratio": round(float(stop_ratio), 6),
        })

    candidates.sort(key=lambda c: -float(c["score"]))
    top_cands = candidates[:40]

    # Causal 60D pairwise return correlations among top candidates & all tracked equities
    corr_map: dict[tuple[str, str], float] = {}
    cand_ids = [c["sec_id"] for c in top_cands if c["sec_id"] in ret60_map]
    for i in range(len(cand_ids)):
        for j in range(i + 1, len(cand_ids)):
            a, b = cand_ids[i], cand_ids[j]
            ra, rb = ret60_map[a], ret60_map[b]
            if float(np.std(ra)) > 1e-8 and float(np.std(rb)) > 1e-8:
                cval = float(np.corrcoef(ra, rb)[0, 1])
                if np.isfinite(cval):
                    corr_map[(a, b)] = round(cval, 4)

    return eq_bars, top_cands, corr_map


def fetch_completed_nse_eod_snapshot(
    target_date: str,
    *,
    universe_path: Path = DEFAULT_PROSPECTIVE_UNIVERSE_PATH,
    extra_open_symbols: list[dict[str, str]] | None = None,
    include_histories: bool = False,
) -> dict[str, Any] | None:
    """Fetch genuine post-close NSE EOD market data for `target_date` via Yahoo Finance (`yfinance`).

    Returns `None` if `NIFTYBEES.NS` does not yet have a completed daily bar for `target_date`
    (never fabricates prices!).
    """
    PaperOnlyExecutionGuard().assert_paper_only()
    import yfinance as yf

    dt = str(target_date)
    dt_obj = date.fromisoformat(dt)
    start_str = (dt_obj - timedelta(days=430)).isoformat()
    end_str = (dt_obj + timedelta(days=2)).isoformat()  # yfinance `end` is exclusive; we strictly slice <= dt!

    # 1. Fetch Core ETF (`NIFTYBEES.NS`) & Liquid ETF (`LIQUIDBEES.NS`) first to verify session availability!
    nb_raw = yf.download("NIFTYBEES.NS", start=start_str, end=end_str, auto_adjust=False, actions=True, progress=False, threads=False)
    nb_df = _extract_ticker_frame(nb_raw, "NIFTYBEES.NS", single_ticker=True)
    if nb_df is None:
        return None
    nb_df = nb_df[nb_df.index <= dt]
    if dt not in nb_df.index:
        # Session not completed / not yet published on NSE EOD feed -> DO NOT fabricate!
        return None

    nb_row = nb_df.loc[dt]
    nb_bar = {
        "open": round(float(nb_row["Open"]), 4),
        "high": round(float(nb_row["High"]), 4),
        "low": round(float(nb_row["Low"]), 4),
        "close": round(float(nb_row["Close"]), 4),
        "volume": round(float(nb_row.get("Volume", 0.0) or 0.0), 2),
        "valid": bool(float(nb_row["Open"]) > 0 and float(nb_row["Close"]) > 0),
    }
    if not nb_bar["valid"]:
        return None

    nb_closes = [float(x) for x in nb_df["Close"].tolist() if np.isfinite(float(x)) and float(x) > 0]
    nb_arr = np.asarray(nb_closes, dtype=np.float64)
    nb_rets = np.diff(nb_arr) / np.maximum(1e-6, nb_arr[:-1]) if len(nb_arr) >= 2 else np.array([], dtype=np.float64)

    lb_raw = yf.download("LIQUIDBEES.NS", start=(dt_obj - timedelta(days=30)).isoformat(), end=end_str, auto_adjust=False, actions=False, progress=False, threads=False)
    lb_df = _extract_ticker_frame(lb_raw, "LIQUIDBEES.NS", single_ticker=True)
    if lb_df is not None:
        lb_df = lb_df[lb_df.index <= dt]
    if lb_df is not None and dt in lb_df.index:
        lb_row = lb_df.loc[dt]
        lb_bar = {
            "open": round(float(lb_row["Open"]), 4),
            "high": round(float(lb_row["High"]), 4),
            "low": round(float(lb_row["Low"]), 4),
            "close": round(float(lb_row["Close"]), 4),
            "volume": round(float(lb_row.get("Volume", 0.0) or 0.0), 2),
            "valid": bool(float(lb_row["Open"]) > 0 and float(lb_row["Close"]) > 0),
        }
        if not lb_bar["valid"]:
            return None
    else:
        # Strict zero-fabrication rule: never carry forward prior session prices for target_date
        return None

    # 2. Load full frozen prospective PIT-eligible NSE equity universe + any open tactical holdings
    uni_items: list[dict[str, Any]] = []
    if universe_path.exists():
        uni_payload = json.loads(universe_path.read_text(encoding="utf-8"))
        uni_items = list(uni_payload.get("equities", []))
    seen_sids = {str(u["sec_id"]) for u in uni_items}
    for extra in extra_open_symbols or []:
        sid = str(extra.get("sec_id", ""))
        sym = str(extra.get("symbol", ""))
        if sid and sym and sid not in seen_sids:
            uni_items.append({
                "sec_id": sid,
                "symbol": sym,
                "yf_ticker": f"{sym}.NS",
                "sector": str(extra.get("sector", "UNKNOWN_SECTOR")),
                "industry": str(extra.get("industry", "UNKNOWN_INDUSTRY")),
                "is_unknown_sector": bool(extra.get("is_unknown_sector", False)),
            })
            seen_sids.add(sid)

    # 3. Batch-download full PIT-eligible equity universe histories (chunks of 100 tickers for Yahoo Finance stability)
    equity_histories: dict[str, dict[str, Any]] = {}
    chunk_size = 100
    for i in range(0, len(uni_items), chunk_size):
        chunk = uni_items[i : i + chunk_size]
        tickers = [str(u["yf_ticker"]) for u in chunk]
        try:
            raw_batch = yf.download(
                tickers,
                start=start_str,
                end=end_str,
                group_by="ticker",
                auto_adjust=False,
                actions=True,
                progress=False,
                threads=True,
            )
        except Exception:
            continue
        single = len(tickers) == 1
        for u in chunk:
            tkr = str(u["yf_ticker"])
            df_u = _extract_ticker_frame(raw_batch, tkr, single_ticker=single)
            if df_u is None:
                continue
            df_u = df_u[df_u.index <= dt]
            if dt not in df_u.index or len(df_u) < 2:
                continue
            divs = df_u["Dividends"].fillna(0.0).tolist() if "Dividends" in df_u.columns else [0.0] * len(df_u)
            equity_histories[str(u["sec_id"])] = {
                "symbol": str(u["symbol"]),
                "sector": str(u["sector"]),
                "industry": str(u["industry"]),
                "is_unknown_sector": bool(u.get("is_unknown_sector", False)),
                "dates": df_u.index.tolist(),
                "open": df_u["Open"].tolist(),
                "high": df_u["High"].tolist(),
                "low": df_u["Low"].tolist(),
                "close": df_u["Close"].tolist(),
                "volume": df_u["Volume"].fillna(0.0).tolist() if "Volume" in df_u.columns else [0.0] * len(df_u),
                "dividends": divs,
            }

    eq_bars, family_e_cands, corr_map = compute_family_e_candidates_from_bars(
        target_date=dt,
        nb_returns=nb_rets,
        equity_histories=equity_histories,
    )

    extra_payload: dict[str, Any] = {}
    if include_histories:
        # Moreshwar 3.2: expose the real per-stock OHLCV histories (<= trading_date) and NIFTYBEES
        # dated closes so the orchestrator can scan the FULL universe and mature its own outcomes.
        extra_payload = {
            "equity_histories": equity_histories,
            "niftybees_dates": [str(d) for d in nb_df.index.tolist()],
            "niftybees_closes_full": [float(x) for x in nb_df["Close"].tolist()],
        }

    return {
        **extra_payload,
        "trading_date": dt,
        "data_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "data_source": "yfinance_nse_post_close_eod",
        "niftybees_bar": nb_bar,
        "liquidbees_bar": lb_bar,
        "niftybees_close_history": nb_closes[-250:],
        "equity_bars": eq_bars,
        "family_e_candidates": family_e_cands,
        "pairwise_corr_map": corr_map,
        "diagnostic_index_available": dt <= DIAGNOSTIC_INDEX_DB_END_DATE,
        "data_quality_state": "PARTIAL_DIAGNOSTIC_ONLY" if dt > DIAGNOSTIC_INDEX_DB_END_DATE else "OK",
    }
