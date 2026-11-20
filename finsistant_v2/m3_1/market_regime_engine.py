"""Moreshwar 3.1 (Prompt 4/5) — Strictly Causal Market Regime Intelligence Engine.

Computes point-in-time market features across six dimensions using ONLY observations <= `ti`:
A. Market Trend (real 100%-coverage `NIFTYBEES` exchange bars: returns, MAs, slope, persistence, drawdown)
B. Point-in-Time Market Breadth (strictly `is_pit_elig[ti]` — NEVER surviving stocks!)
C. Volatility & Stress (realized 20D/60D vol, ATR%, expanding 252D vol percentile, gap frequency)
D. Liquidity & Stress Proxies (clearly labeled OHLCV/volume proxies — zero invented bid-ask/depth data)
E. Sector Leadership & Concentration (4-tier sector hierarchy: participation, dispersion, Defensive vs Cyclical)
F. Macro & External Index Audit (`India VIX` & `Nifty GS 10Yr` loaded strictly where available up to 2025-10-10
   as diagnostic-only series; RBI repo, CPI, FX excluded due to absent PIT publication timestamps)

Implements three classifiers compared head-to-head:
- Baseline: `SIMPLE_TREND` (`NIFTYBEES >= SMA200`)
- Model A: `RULE_BASED_6STATE` (transparent hierarchy + 3-day confirmation hysteresis)
- Model B: `STATISTICAL_KMEANS_4STATE` (standardized K-Means fitted STRICTLY on training slice only, frozen for outer folds)
"""

from __future__ import annotations

import math
from pathlib import Path
import sqlite3
from typing import Any

import numpy as np

from finsistant_v2.m3.schema import DEFAULT_M3_DB_PATH
from finsistant_v2.m3_1.sector_hierarchy import UNKNOWN_SECTOR_LABEL

REGIME_STATES_6 = (
    "BULL_BROAD",
    "BULL_NARROW",
    "SIDEWAYS_RANGE",
    "BEAR_DETERIORATING",
    "HIGH_VOL_STRESS",
    "RECOVERY_IMPROVING",
)

STAT_STATES_4 = (
    "STAT_BULL_BROAD",
    "STAT_BULL_NARROW_CHOP",
    "STAT_RECOVERY",
    "STAT_STRESS_BEAR",
)

DEFENSIVE_SECTORS = frozenset({
    "Fast Moving Consumer Goods",
    "Healthcare",
    "Information Technology",
    "Power",
    "Utilities",
})

CYCLICAL_SECTORS = frozenset({
    "Financial Services",
    "Automobile and Auto Components",
    "Metals & Mining",
    "Realty",
    "Capital Goods",
    "Construction",
    "Construction Materials",
    "Oil Gas & Consumable Fuels",
})


def _load_diagnostic_index_series(
    dates: list[str],
    db_path: Path = DEFAULT_M3_DB_PATH,
) -> dict[str, np.ndarray]:
    """Load real historical `India VIX`, `Nifty GS 10Yr`, and `NIFTY50 EQL Wgt` where available (no fabrication)."""
    t_len = len(dates)
    out = {
        "india_vix": np.full(t_len, np.nan, dtype=np.float64),
        "nifty_gs_10yr": np.full(t_len, np.nan, dtype=np.float64),
        "nifty50_ew": np.full(t_len, np.nan, dtype=np.float64),
    }
    if not db_path.exists():
        return out
    dt_to_idx = {str(d): i for i, d in enumerate(dates)}
    mapping = {
        "India VIX": "india_vix",
        "Nifty GS 10Yr": "nifty_gs_10yr",
        "NIFTY50 EQL Wgt": "nifty50_ew",
    }
    conn = sqlite3.connect(str(db_path))
    try:
        for idx_name, key in mapping.items():
            for dt, cl in conn.execute(
                "SELECT trade_date, close FROM daily_index_bars WHERE index_name = ? ORDER BY trade_date",
                (idx_name,),
            ):
                ti = dt_to_idx.get(str(dt))
                if ti is not None and cl is not None and float(cl) > 0:
                    out[key][ti] = float(cl)
    finally:
        conn.close()
    return out


def compute_causal_regime_features(
    *,
    panel: dict[str, Any],
    etf_bars: dict[str, dict[str, np.ndarray]],
    db_path: Path = DEFAULT_M3_DB_PATH,
) -> dict[str, Any]:
    """Compute strictly causal daily regime features `[T]` using only observations <= `ti`."""
    dates = [str(d) for d in panel["dates"]]
    t_len = len(dates)

    # 100%-coverage NIFTYBEES OHLCV bars (fallback to nifty50_close if synthetic test lacks ETF bars)
    nb = etf_bars.get("NIFTYBEES", {})
    nb_o = np.asarray(nb.get("open", panel["nifty50_close"]), dtype=np.float64).copy()
    nb_h = np.asarray(nb.get("high", panel["nifty50_close"]), dtype=np.float64).copy()
    nb_l = np.asarray(nb.get("low", panel["nifty50_close"]), dtype=np.float64).copy()
    nb_c = np.asarray(nb.get("close", panel["nifty50_close"]), dtype=np.float64).copy()
    nb_v = np.asarray(nb.get("volume", np.full(t_len, 1_000_000.0)), dtype=np.float64).copy()

    # Forward-fill any isolated NaN strictly causally (never backward-fill!)
    for arr in (nb_o, nb_h, nb_l, nb_c):
        for t in range(1, t_len):
            if np.isnan(arr[t]) or arr[t] <= 0:
                arr[t] = arr[t - 1]

    # --- A. Market Trend features (strictly <= t) ---
    nb_ret_1d = np.zeros(t_len, dtype=np.float64)
    nb_ret_20d = np.full(t_len, np.nan, dtype=np.float64)
    nb_ret_60d = np.full(t_len, np.nan, dtype=np.float64)
    nb_ret_120d = np.full(t_len, np.nan, dtype=np.float64)
    sma_50 = np.full(t_len, np.nan, dtype=np.float64)
    sma_200 = np.full(t_len, np.nan, dtype=np.float64)
    sma50_slope_20d = np.full(t_len, np.nan, dtype=np.float64)
    trend_consistency_60d = np.full(t_len, np.nan, dtype=np.float64)
    dist_252h_pct = np.full(t_len, np.nan, dtype=np.float64)
    dist_252l_pct = np.full(t_len, np.nan, dtype=np.float64)

    for t in range(1, t_len):
        if nb_c[t - 1] > 0 and nb_c[t] > 0:
            nb_ret_1d[t] = (nb_c[t] / nb_c[t - 1]) - 1.0
        if t >= 20 and nb_c[t - 20] > 0:
            nb_ret_20d[t] = (nb_c[t] / nb_c[t - 20] - 1.0) * 100.0
        if t >= 60 and nb_c[t - 60] > 0:
            nb_ret_60d[t] = (nb_c[t] / nb_c[t - 60] - 1.0) * 100.0
            # Rolling 60D return consistency: % of rolling 5-day sub-returns > 0 strictly within [t-60..t]
            w5 = nb_c[t - 55 : t + 1] / np.maximum(1e-6, nb_c[t - 60 : t - 4]) - 1.0
            trend_consistency_60d[t] = float(np.mean(w5 > 0.0)) * 100.0
        if t >= 120 and nb_c[t - 120] > 0:
            nb_ret_120d[t] = (nb_c[t] / nb_c[t - 120] - 1.0) * 100.0
        if t >= 49:
            sma_50[t] = float(np.mean(nb_c[t - 49 : t + 1]))
        if t >= 199:
            sma_200[t] = float(np.mean(nb_c[t - 199 : t + 1]))
        if t >= 69 and not np.isnan(sma_50[t]) and not np.isnan(sma_50[t - 20]) and sma_50[t - 20] > 0:
            sma50_slope_20d[t] = ((sma_50[t] / sma_50[t - 20]) - 1.0) * 100.0
        w_start = max(0, t - 251)
        h252 = float(np.max(nb_c[w_start : t + 1]))
        l252 = float(np.min(nb_c[w_start : t + 1]))
        if h252 > 0:
            dist_252h_pct[t] = ((nb_c[t] / h252) - 1.0) * 100.0
        if l252 > 0:
            dist_252l_pct[t] = ((nb_c[t] / l252) - 1.0) * 100.0

    px_vs_sma50_pct = np.where((~np.isnan(sma_50)) & (sma_50 > 0), ((nb_c / sma_50) - 1.0) * 100.0, np.nan)
    px_vs_sma200_pct = np.where((~np.isnan(sma_200)) & (sma_200 > 0), ((nb_c / sma_200) - 1.0) * 100.0, np.nan)

    # --- C. Volatility & D. ETF Liquidity/Stress Proxies (strictly <= t) ---
    vol_20d_ann = np.full(t_len, np.nan, dtype=np.float64)
    vol_60d_ann = np.full(t_len, np.nan, dtype=np.float64)
    vol_ratio_20_60 = np.full(t_len, np.nan, dtype=np.float64)
    vol_pctile_252d = np.full(t_len, np.nan, dtype=np.float64)
    atr_14_pct = np.full(t_len, np.nan, dtype=np.float64)
    etf_gap_freq_20d = np.full(t_len, np.nan, dtype=np.float64)
    etf_vol_ratio_5_20 = np.full(t_len, np.nan, dtype=np.float64)
    etf_hl_spread_proxy_10d = np.full(t_len, np.nan, dtype=np.float64)

    tr_nb = np.zeros(t_len, dtype=np.float64)
    gap_flag_nb = np.zeros(t_len, dtype=np.float64)
    hl_nb_pct = np.where(nb_c > 0, np.maximum(0.0, (nb_h - nb_l) / nb_c) * 100.0, 0.0)
    for t in range(1, t_len):
        tr_nb[t] = max(nb_h[t] - nb_l[t], abs(nb_h[t] - nb_c[t - 1]), abs(nb_l[t] - nb_c[t - 1]))
        if nb_c[t - 1] > 0:
            gap_flag_nb[t] = 1.0 if abs((nb_o[t] / nb_c[t - 1]) - 1.0) >= 0.008 else 0.0
        if t >= 14 and nb_c[t] > 0:
            atr_14_pct[t] = (float(np.mean(tr_nb[t - 13 : t + 1])) / nb_c[t]) * 100.0
        if t >= 20:
            vol_20d_ann[t] = float(np.std(nb_ret_1d[t - 19 : t + 1], ddof=1)) * math.sqrt(252.0) * 100.0
            etf_gap_freq_20d[t] = float(np.mean(gap_flag_nb[t - 19 : t + 1])) * 100.0
            v20 = float(np.mean(nb_v[t - 19 : t + 1]))
            v5 = float(np.mean(nb_v[t - 4 : t + 1]))
            etf_vol_ratio_5_20[t] = v5 / max(1.0, v20)
        if t >= 10:
            etf_hl_spread_proxy_10d[t] = float(np.mean(hl_nb_pct[t - 9 : t + 1]))
        if t >= 60:
            vol_60d_ann[t] = float(np.std(nb_ret_1d[t - 59 : t + 1], ddof=1)) * math.sqrt(252.0) * 100.0
            vol_ratio_20_60[t] = vol_20d_ann[t] / max(1.0, vol_60d_ann[t])
        if t >= 40 and not np.isnan(vol_20d_ann[t]):
            w_s = max(20, t - 251)
            hist_v = vol_20d_ann[w_s : t + 1]
            hist_v = hist_v[~np.isnan(hist_v)]
            if len(hist_v) >= 10:
                vol_pctile_252d[t] = float(np.mean(hist_v <= vol_20d_ann[t]))

    # --- B. PIT Market Breadth & E. 4-Tier Sector Leadership (strictly `is_pit_elig[t]`) ---
    close_p = panel["adj_close"].astype(np.float64)
    open_p = panel["adj_open"].astype(np.float64)
    high_p = panel["adj_high"].astype(np.float64)
    low_p = panel["adj_low"].astype(np.float64)
    vol_p = panel["volume"].astype(np.float64)
    elig = panel["is_pit_elig"].astype(bool)
    sectors = [str(s) for s in panel["sectors"]]
    n_secs = close_p.shape[1]

    # Causal rolling SMA50, SMA200, 20D return, 252D high/low per security
    cs_c = np.cumsum(np.where(np.isnan(close_p), 0.0, close_p), axis=0)
    cs_cnt = np.cumsum(np.where(np.isnan(close_p) | (close_p <= 0), 0.0, 1.0), axis=0)

    sma50_eq = np.full_like(close_p, np.nan)
    sma200_eq = np.full_like(close_p, np.nan)
    for w, dest in ((50, sma50_eq), (200, sma200_eq)):
        if t_len > w:
            s_val = cs_c[w:] - cs_c[:-w]
            n_val = cs_cnt[w:] - cs_cnt[:-w]
            dest[w:] = np.where(n_val >= int(w * 0.7), s_val / np.maximum(1.0, n_val), np.nan)

    # Pre-group classified sector indices (excluding UNKNOWN_SECTOR)
    sec_to_cols: dict[str, np.ndarray] = {}
    for j, s_nm in enumerate(sectors):
        if s_nm and s_nm != UNKNOWN_SECTOR_LABEL and "Unclassified" not in s_nm:
            sec_to_cols.setdefault(s_nm, []).append(j)
    sec_col_arrays = {k: np.array(v, dtype=int) for k, v in sec_to_cols.items() if len(v) >= 3}

    pct_ab50 = np.full(t_len, np.nan, dtype=np.float64)
    pct_ab200 = np.full(t_len, np.nan, dtype=np.float64)
    adv_pct_1d = np.full(t_len, np.nan, dtype=np.float64)
    adv_pct_10d = np.full(t_len, np.nan, dtype=np.float64)
    nh_nl_net_pct = np.full(t_len, np.nan, dtype=np.float64)
    breadth_delta_20d = np.full(t_len, np.nan, dtype=np.float64)
    ew_vs_nifty_20d = np.full(t_len, np.nan, dtype=np.float64)
    sector_participation_pct = np.full(t_len, np.nan, dtype=np.float64)
    sector_dispersion_20d = np.full(t_len, np.nan, dtype=np.float64)
    defensive_vs_cyclical_20d = np.full(t_len, np.nan, dtype=np.float64)
    mkt_median_hl_proxy = np.full(t_len, np.nan, dtype=np.float64)
    mkt_gap_freq_pct = np.full(t_len, np.nan, dtype=np.float64)
    mkt_vol_ratio_proxy = np.full(t_len, np.nan, dtype=np.float64)

    # Causal rolling 20D avg volume per stock
    v_clean = np.where(np.isnan(vol_p) | (vol_p < 0), 0.0, vol_p)
    cs_v = np.cumsum(v_clean, axis=0)
    avg_v20 = np.full_like(v_clean, np.nan)
    if t_len > 20:
        avg_v20[20:] = (cs_v[20:] - cs_v[:-20]) / 20.0

    # Causal rolling 252D max/min close per stock for NH/NL (strictly <= t)
    run_max_252 = np.full_like(close_p, np.nan)
    run_min_252 = np.full_like(close_p, np.nan)
    for t in range(t_len):
        e_mask = elig[t] & (~np.isnan(close_p[t])) & (close_p[t] > 0)
        if not np.any(e_mask):
            continue
        c_row = close_p[t]
        ok50 = e_mask & (~np.isnan(sma50_eq[t]))
        if np.any(ok50):
            pct_ab50[t] = float(np.mean(c_row[ok50] > sma50_eq[t, ok50])) * 100.0
        ok200 = e_mask & (~np.isnan(sma200_eq[t]))
        if np.any(ok200):
            pct_ab200[t] = float(np.mean(c_row[ok200] > sma200_eq[t, ok200])) * 100.0
        if t >= 1:
            ok1 = e_mask & (~np.isnan(close_p[t - 1])) & (close_p[t - 1] > 0)
            if np.any(ok1):
                adv_pct_1d[t] = float(np.mean(c_row[ok1] > close_p[t - 1, ok1])) * 100.0
                # Cross-sectional gap frequency proxy (|open/prev_close - 1| >= 2.0%)
                ok_g = ok1 & (~np.isnan(open_p[t])) & (open_p[t] > 0)
                if np.any(ok_g):
                    gaps = np.abs(open_p[t, ok_g] / close_p[t - 1, ok_g] - 1.0)
                    mkt_gap_freq_pct[t] = float(np.mean(gaps >= 0.02)) * 100.0
        if t >= 10:
            a10 = adv_pct_1d[t - 9 : t + 1]
            a10 = a10[~np.isnan(a10)]
            if len(a10) > 0:
                adv_pct_10d[t] = float(np.mean(a10))

        # HL range liquidity stress proxy on PIT eligible stocks
        ok_hl = e_mask & (~np.isnan(high_p[t])) & (~np.isnan(low_p[t])) & (high_p[t] >= low_p[t])
        if np.any(ok_hl):
            mkt_median_hl_proxy[t] = float(np.median(((high_p[t, ok_hl] - low_p[t, ok_hl]) / c_row[ok_hl]) * 100.0))

        # Volume ratio proxy on PIT eligible stocks
        if t >= 20:
            ok_v = e_mask & (avg_v20[t] > 0) & (v_clean[t] > 0)
            if np.any(ok_v):
                mkt_vol_ratio_proxy[t] = float(np.median(v_clean[t, ok_v] / avg_v20[t, ok_v]))

        # 252D New Highs vs New Lows (within 2% of rolling 252D high/low, strictly causal [w0..t])
        if t >= 60:
            w0 = max(0, t - 251)
            sub_c = close_p[w0 : t + 1, e_mask]
            with np.errstate(all="ignore"):
                hi_e = np.nanmax(sub_c, axis=0)
                lo_e = np.nanmin(sub_c, axis=0)
            cur_e = c_row[e_mask]
            nh = np.mean(cur_e >= hi_e * 0.98) * 100.0
            nl = np.mean(cur_e <= lo_e * 1.02) * 100.0
            nh_nl_net_pct[t] = float(nh - nl)

        # 20D breadth delta & Equal-Weight PIT Universe vs NIFTYBEES spread (Narrow vs Broad Rally!)
        if t >= 20:
            if not np.isnan(pct_ab50[t]) and not np.isnan(pct_ab50[t - 20]):
                breadth_delta_20d[t] = pct_ab50[t] - pct_ab50[t - 20]
            ok20 = e_mask & (~np.isnan(close_p[t - 20])) & (close_p[t - 20] > 0)
            if np.any(ok20) and not np.isnan(nb_ret_20d[t]):
                ew_r20 = float(np.median((c_row[ok20] / close_p[t - 20, ok20] - 1.0) * 100.0))
                ew_vs_nifty_20d[t] = ew_r20 - nb_ret_20d[t]

                # Sector participation, dispersion & Defensive vs Cyclical rotation
                sec_ab50_flags: list[float] = []
                sec_r20_vals: list[float] = []
                def_r20: list[float] = []
                cyc_r20: list[float] = []
                for s_nm, s_cols in sec_col_arrays.items():
                    s_ok = ok20[s_cols]
                    if int(np.sum(s_ok)) < 3:
                        continue
                    cols_u = s_cols[s_ok]
                    s_med_r20 = float(np.median((c_row[cols_u] / close_p[t - 20, cols_u] - 1.0) * 100.0))
                    sec_r20_vals.append(s_med_r20)
                    if s_nm in DEFENSIVE_SECTORS:
                        def_r20.append(s_med_r20)
                    elif s_nm in CYCLICAL_SECTORS:
                        cyc_r20.append(s_med_r20)
                    s_ok50 = ok50[s_cols]
                    if int(np.sum(s_ok50)) >= 3:
                        c50 = s_cols[s_ok50]
                        sec_ab50_flags.append(1.0 if float(np.mean(c_row[c50] > sma50_eq[t, c50])) >= 0.50 else 0.0)
                if sec_ab50_flags:
                    sector_participation_pct[t] = float(np.mean(sec_ab50_flags)) * 100.0
                if len(sec_r20_vals) >= 4:
                    s_sorted = sorted(sec_r20_vals)
                    sector_dispersion_20d[t] = float(np.mean(s_sorted[-3:]) - np.mean(s_sorted[:3]))
                if def_r20 and cyc_r20:
                    defensive_vs_cyclical_20d[t] = float(np.mean(def_r20) - np.mean(cyc_r20))

    diag_idx = _load_diagnostic_index_series(dates, db_path=db_path)

    return {
        "dates": dates,
        "nb_close": nb_c,
        "nb_ret_20d": nb_ret_20d,
        "nb_ret_60d": nb_ret_60d,
        "nb_ret_120d": nb_ret_120d,
        "px_vs_sma50_pct": px_vs_sma50_pct,
        "px_vs_sma200_pct": px_vs_sma200_pct,
        "sma50_slope_20d": sma50_slope_20d,
        "trend_consistency_60d": trend_consistency_60d,
        "dist_252h_pct": dist_252h_pct,
        "dist_252l_pct": dist_252l_pct,
        "pct_ab50": pct_ab50,
        "pct_ab200": pct_ab200,
        "adv_pct_10d": adv_pct_10d,
        "nh_nl_net_pct": nh_nl_net_pct,
        "breadth_delta_20d": breadth_delta_20d,
        "ew_vs_nifty_20d": ew_vs_nifty_20d,
        "vol_20d_ann": vol_20d_ann,
        "vol_60d_ann": vol_60d_ann,
        "vol_ratio_20_60": vol_ratio_20_60,
        "vol_pctile_252d": vol_pctile_252d,
        "atr_14_pct": atr_14_pct,
        "etf_gap_freq_20d": etf_gap_freq_20d,
        "etf_vol_ratio_5_20": etf_vol_ratio_5_20,
        "etf_hl_spread_proxy_10d": etf_hl_spread_proxy_10d,
        "mkt_median_hl_proxy": mkt_median_hl_proxy,
        "mkt_gap_freq_pct": mkt_gap_freq_pct,
        "mkt_vol_ratio_proxy": mkt_vol_ratio_proxy,
        "sector_participation_pct": sector_participation_pct,
        "sector_dispersion_20d": sector_dispersion_20d,
        "defensive_vs_cyclical_20d": defensive_vs_cyclical_20d,
        # Diagnostic-only partial-coverage external index series (ends 2025-10-10; NOT used in tradable model)
        "diag_india_vix": diag_idx["india_vix"],
        "diag_nifty_gs_10yr": diag_idx["nifty_gs_10yr"],
        "diag_nifty50_ew": diag_idx["nifty50_ew"],
    }


def classify_simple_trend_baseline(features: dict[str, Any], *, sma_window: int = 200) -> list[str]:
    """Simple 2-state benchmark regime classifier: `BULL_TREND` if NIFTYBEES >= SMA(sma_window) else `BEAR_TREND`."""
    if sma_window == 200:
        px200 = features["px_vs_sma200_pct"]
        return ["BEAR_TREND" if (not np.isnan(px200[t]) and px200[t] < 0.0) else "BULL_TREND" for t in range(len(features["dates"]))]
    nb_c = features["nb_close"]
    t_len = len(features["dates"])
    out: list[str] = []
    for t in range(t_len):
        if t >= sma_window - 1:
            sma_v = float(np.mean(nb_c[t - sma_window + 1 : t + 1]))
            out.append("BEAR_TREND" if nb_c[t] < sma_v else "BULL_TREND")
        else:
            out.append("BULL_TREND")
    return out


def classify_rule_based_6state(
    features: dict[str, Any],
    *,
    confirm_days: int = 3,
    vol_stress_pctile: float = 0.85,
    vol_stress_ann: float = 18.5,
    dd_stress_pct: float = -10.5,
) -> dict[str, Any]:
    """Model A: Transparent 6-state causal rule-based classifier with confirmation hysteresis."""
    dates = features["dates"]
    t_len = len(dates)

    raw_labels: list[str] = []
    confirmed_labels: list[str] = []
    confidences: list[float] = []
    reasons: list[str] = []
    avail_status: list[str] = []

    cur_state = "SIDEWAYS_RANGE"
    cand_state = "SIDEWAYS_RANGE"
    cand_streak = 0

    for t in range(t_len):
        px50 = float(features["px_vs_sma50_pct"][t]) if not np.isnan(features["px_vs_sma50_pct"][t]) else 0.0
        px200 = float(features["px_vs_sma200_pct"][t]) if not np.isnan(features["px_vs_sma200_pct"][t]) else 0.0
        sl50 = float(features["sma50_slope_20d"][t]) if not np.isnan(features["sma50_slope_20d"][t]) else 0.0
        r20 = float(features["nb_ret_20d"][t]) if not np.isnan(features["nb_ret_20d"][t]) else 0.0
        r60 = float(features["nb_ret_60d"][t]) if not np.isnan(features["nb_ret_60d"][t]) else 0.0
        dd252 = float(features["dist_252h_pct"][t]) if not np.isnan(features["dist_252h_pct"][t]) else 0.0
        ab50 = float(features["pct_ab50"][t]) if not np.isnan(features["pct_ab50"][t]) else 50.0
        b_del = float(features["breadth_delta_20d"][t]) if not np.isnan(features["breadth_delta_20d"][t]) else 0.0
        ew_sp = float(features["ew_vs_nifty_20d"][t]) if not np.isnan(features["ew_vs_nifty_20d"][t]) else 0.0
        v20 = float(features["vol_20d_ann"][t]) if not np.isnan(features["vol_20d_ann"][t]) else 12.0
        vpct = float(features["vol_pctile_252d"][t]) if not np.isnan(features["vol_pctile_252d"][t]) else 0.50
        sec_p = float(features["sector_participation_pct"][t]) if not np.isnan(features["sector_participation_pct"][t]) else 50.0

        has_vix = not np.isnan(features["diag_india_vix"][t])
        has_200 = not np.isnan(features["px_vs_sma200_pct"][t])
        if not has_200:
            status = "WARMUP_PARTIAL_SMA200"
        elif has_vix:
            status = "FULL_PIT_PLUS_DIAG_VIX_GS10Y"
        else:
            status = "FULL_PIT_ONLY_NO_INDEX_VIX_POST_2025_10"
        avail_status.append(status)

        # Strict transparent decision hierarchy (Model A)
        if v20 >= vol_stress_ann or (vpct >= vol_stress_pctile and dd252 <= -5.0) or dd252 <= dd_stress_pct:
            lbl = "HIGH_VOL_STRESS"
            margin = max((v20 - vol_stress_ann) / 10.0, (vpct - vol_stress_pctile) * 2.0, (-dd252 + dd_stress_pct) / 10.0)
            conf = min(0.98, max(0.60, 0.72 + margin * 0.25))
            rsn = f"vol20={v20:.1f}%,vpct={vpct:.2f},dd252={dd252:.1f}%"
        elif (px200 < 0.0 and (ab50 < 45.0 or r60 < -3.0)) or (px50 < -2.5 and ab50 < 35.0):
            lbl = "BEAR_DETERIORATING"
            conf = min(0.96, max(0.58, 0.68 + abs(min(0.0, px200)) / 20.0 + max(0.0, (45.0 - ab50) / 80.0)))
            rsn = f"px200={px200:.1f}%,ab50={ab50:.1f}%,r60={r60:.1f}%"
        elif (dd252 <= -3.5 or px200 < 1.5) and b_del >= 8.0 and r20 >= 1.2:
            lbl = "RECOVERY_IMPROVING"
            conf = min(0.94, max(0.55, 0.62 + (b_del - 8.0) / 40.0 + r20 / 20.0))
            rsn = f"dd252={dd252:.1f}%,b_delta20={b_del:+.1f}pp,r20={r20:+.1f}%"
        elif px50 >= 0.0 and sl50 >= 0.0 and ab50 >= 55.0 and sec_p >= 55.0:
            lbl = "BULL_BROAD"
            conf = min(0.97, max(0.60, 0.68 + (ab50 - 55.0) / 100.0 + min(0.15, px50 / 40.0)))
            rsn = f"px50={px50:+.1f}%,ab50={ab50:.1f}%,sec_part={sec_p:.0f}%"
        elif px50 >= 0.0 and (r20 > 0.5 or px200 >= 2.0) and (ab50 < 55.0 or ew_sp < -1.5):
            lbl = "BULL_NARROW"
            conf = min(0.92, max(0.52, 0.60 + max(0.0, (55.0 - ab50) / 100.0) + max(0.0, -ew_sp / 20.0)))
            rsn = f"px50={px50:+.1f}%,ab50={ab50:.1f}%,ew_vs_nb={ew_sp:+.1f}%"
        else:
            lbl = "SIDEWAYS_RANGE"
            conf = min(0.88, max(0.48, 0.58 + max(0.0, 1.0 - abs(r20) / 8.0) * 0.18))
            rsn = f"px50={px50:+.1f}%,r20={r20:+.1f}%,ab50={ab50:.1f}%"

        raw_labels.append(lbl)
        # Confirmation hysteresis (immediate transition allowed into HIGH_VOL_STRESS for capital protection!)
        if lbl == cur_state:
            cand_state = lbl
            cand_streak = 0
        elif lbl == "HIGH_VOL_STRESS":
            cur_state = lbl
            cand_state = lbl
            cand_streak = 0
        else:
            if lbl == cand_state:
                cand_streak += 1
            else:
                cand_state = lbl
                cand_streak = 1
            if cand_streak >= max(1, confirm_days):
                cur_state = cand_state
                cand_streak = 0

        confirmed_labels.append(cur_state)
        confidences.append(round(conf, 3))
        reasons.append(rsn if cur_state == lbl else f"HYSTERESIS_HOLD({cur_state};raw={lbl}:{rsn})")

    return {
        "model_name": "RULE_BASED_6STATE",
        "raw_labels": raw_labels,
        "labels": confirmed_labels,
        "confidences": confidences,
        "reasons": reasons,
        "data_availability": avail_status,
    }


def _build_stat_feature_matrix(features: dict[str, Any]) -> np.ndarray:
    """Assemble 6 clean, 100%-complete causal features for statistical clustering."""
    cols = [
        np.nan_to_num(features["nb_ret_60d"], nan=0.0),
        np.nan_to_num(features["px_vs_sma50_pct"], nan=0.0),
        np.nan_to_num(features["pct_ab50"], nan=50.0),
        np.nan_to_num(features["breadth_delta_20d"], nan=0.0),
        np.nan_to_num(features["vol_20d_ann"], nan=12.0),
        np.nan_to_num(features["ew_vs_nifty_20d"], nan=0.0),
    ]
    return np.column_stack(cols).astype(np.float64)


def fit_and_predict_statistical_regime_model(
    features: dict[str, Any],
    *,
    train_start: str,
    train_end: str,
    k: int = 4,
    confirm_days: int = 3,
    seed: int = 20261119,
) -> dict[str, Any]:
    """Model B: Strictly causal K-Means 4-state model fitted ONLY on `[train_start, train_end]` and frozen.

    Zero future observations enter feature standardization, cluster centroids, or semantic label mapping.
    """
    dates = features["dates"]
    x_all = _build_stat_feature_matrix(features)
    tr_idx = [i for i, d in enumerate(dates) if train_start <= d <= train_end]
    if len(tr_idx) < 40:
        tr_idx = list(range(min(len(dates), 252)))

    x_tr = x_all[tr_idx]
    mu = np.mean(x_tr, axis=0)
    sd = np.maximum(1e-6, np.std(x_tr, axis=0))
    z_tr = np.clip((x_tr - mu) / sd, -4.0, 4.0)
    z_all = np.clip((x_all - mu) / sd, -4.0, 4.0)

    # Deterministic K-Means++ initialization on training data only
    rng = np.random.default_rng(seed)
    n_tr = z_tr.shape[0]
    centroids = np.zeros((k, z_tr.shape[1]), dtype=np.float64)
    centroids[0] = z_tr[int(rng.integers(0, n_tr))]
    for c_i in range(1, k):
        d2 = np.min(np.sum((z_tr[:, None, :] - centroids[None, :c_i, :]) ** 2, axis=2), axis=1)
        probs = d2 / max(1e-12, float(np.sum(d2)))
        centroids[c_i] = z_tr[int(rng.choice(n_tr, p=probs))]

    for _ in range(40):
        dists = np.sum((z_tr[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
        assign = np.argmin(dists, axis=1)
        new_c = centroids.copy()
        for c_i in range(k):
            mask = assign == c_i
            if np.any(mask):
                new_c[c_i] = np.mean(z_tr[mask], axis=0)
        if np.max(np.abs(new_c - centroids)) < 1e-5:
            centroids = new_c
            break
        centroids = new_c

    # Deterministically map centroids to semantic labels using ONLY training centroid properties:
    # Feature indices: 0=ret_60d, 1=px_vs_sma50, 2=pct_ab50, 3=breadth_delta_20d, 4=vol_20d_ann, 5=ew_vs_nifty
    unmapped = set(range(k))
    c_to_label: dict[int, str] = {}
    # 1. Highest stress score (high vol, low trend/breadth) -> STAT_STRESS_BEAR
    stress_sc = {ci: float(centroids[ci, 4] - 0.7 * centroids[ci, 1] - 0.6 * centroids[ci, 2]) for ci in unmapped}
    c_stress = max(stress_sc, key=stress_sc.get)
    c_to_label[c_stress] = "STAT_STRESS_BEAR"
    unmapped.remove(c_stress)
    # 2. Highest broad bull score (high breadth + trend + ew spread) -> STAT_BULL_BROAD
    bull_sc = {ci: float(centroids[ci, 2] + 0.8 * centroids[ci, 0] + 0.4 * centroids[ci, 5]) for ci in unmapped}
    c_bull = max(bull_sc, key=bull_sc.get)
    c_to_label[c_bull] = "STAT_BULL_BROAD"
    unmapped.remove(c_bull)
    # 3. Highest breadth improvement among remaining -> STAT_RECOVERY
    rec_sc = {ci: float(centroids[ci, 3] + 0.3 * centroids[ci, 5]) for ci in unmapped}
    c_rec = max(rec_sc, key=rec_sc.get)
    c_to_label[c_rec] = "STAT_RECOVERY"
    unmapped.remove(c_rec)
    # 4. Remaining -> STAT_BULL_NARROW_CHOP
    for ci in unmapped:
        c_to_label[ci] = "STAT_BULL_NARROW_CHOP"

    # Frozen prediction across all dates using distance-based softmax confidence + confirmation hysteresis
    dists_all = np.sum((z_all[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
    labels: list[str] = []
    confidences: list[float] = []
    reasons: list[str] = []

    cur_st = "STAT_BULL_NARROW_CHOP"
    cand_st = cur_st
    cand_streak = 0
    for t in range(len(dates)):
        d_row = dists_all[t]
        c_best = int(np.argmin(d_row))
        lbl = c_to_label[c_best]
        # Softmax posterior confidence from negative squared Euclidean distances
        shifted = -0.5 * (d_row - float(np.min(d_row)))
        exps = np.exp(np.clip(shifted, -20.0, 0.0))
        conf = float(exps[c_best] / max(1e-12, float(np.sum(exps))))

        if lbl == cur_st:
            cand_st, cand_streak = lbl, 0
        elif lbl == "STAT_STRESS_BEAR":
            cur_st, cand_st, cand_streak = lbl, lbl, 0
        else:
            if lbl == cand_st:
                cand_streak += 1
            else:
                cand_st, cand_streak = lbl, 1
            if cand_streak >= max(1, confirm_days):
                cur_st, cand_streak = cand_st, 0
        labels.append(cur_st)
        confidences.append(round(conf, 3))
        reasons.append(f"KMEANS_FROZEN({train_start}..{train_end}):c={c_best},dist={math.sqrt(d_row[c_best]):.2f}")

    return {
        "model_name": "STATISTICAL_KMEANS_4STATE",
        "train_window": f"{train_start} -> {train_end}",
        "labels": labels,
        "confidences": confidences,
        "reasons": reasons,
        "centroids_z": {c_to_label[i]: [round(float(v), 3) for v in centroids[i]] for i in range(k)},
    }


def export_regime_history_rows(
    features: dict[str, Any],
    rule_out: dict[str, Any],
    stat_out: dict[str, Any],
    simple_trend: list[str],
    *,
    start_date: str = "2021-07-01",
) -> list[dict[str, Any]]:
    """Build daily audit rows for `REGIME_CLASSIFICATION_HISTORY.csv`."""
    rows: list[dict[str, Any]] = []
    dates = features["dates"]
    for t, dt in enumerate(dates):
        if dt < start_date:
            continue

        def _f(arr: np.ndarray, nd: int = 2) -> Any:
            v = arr[t]
            return "" if np.isnan(v) else round(float(v), nd)

        rows.append({
            "date": dt,
            "rule_regime_6state": rule_out["labels"][t],
            "rule_confidence": rule_out["confidences"][t],
            "rule_classification_reason": rule_out["reasons"][t],
            "stat_regime_4state": stat_out["labels"][t],
            "stat_confidence": stat_out["confidences"][t],
            "simple_trend_baseline": simple_trend[t],
            "data_availability_status": rule_out["data_availability"][t],
            "niftybees_ret_20d_pct": _f(features["nb_ret_20d"]),
            "niftybees_ret_60d_pct": _f(features["nb_ret_60d"]),
            "px_vs_sma50_pct": _f(features["px_vs_sma50_pct"]),
            "px_vs_sma200_pct": _f(features["px_vs_sma200_pct"]),
            "sma50_slope_20d_pct": _f(features["sma50_slope_20d"]),
            "dist_252h_pct": _f(features["dist_252h_pct"]),
            "pit_pct_above_sma50": _f(features["pct_ab50"]),
            "pit_pct_above_sma200": _f(features["pct_ab200"]),
            "pit_adv_pct_10d": _f(features["adv_pct_10d"]),
            "pit_nh_nl_net_pct": _f(features["nh_nl_net_pct"]),
            "breadth_delta_20d_pp": _f(features["breadth_delta_20d"]),
            "ew_vs_nifty_20d_pp": _f(features["ew_vs_nifty_20d"]),
            "vol_20d_ann_pct": _f(features["vol_20d_ann"]),
            "vol_pctile_252d": _f(features["vol_pctile_252d"], 3),
            "atr_14_pct": _f(features["atr_14_pct"]),
            "etf_hl_spread_proxy_10d": _f(features["etf_hl_spread_proxy_10d"]),
            "sector_participation_pct": _f(features["sector_participation_pct"]),
            "defensive_vs_cyclical_20d_pp": _f(features["defensive_vs_cyclical_20d"]),
            "diag_india_vix_partial": _f(features["diag_india_vix"]),
            "diag_nifty_gs_10yr_partial": _f(features["diag_nifty_gs_10yr"]),
        })
    return rows
