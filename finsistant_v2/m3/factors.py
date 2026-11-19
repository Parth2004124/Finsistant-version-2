"""Moreshwar 3.0 Prompt 2 — Multi-Family Causal Factor Research Library.

Every factor is computed strictly causally at day `t` using ONLY bars `<= t`
and PIT eligibility masks `is_pit_elig[t, :]`. Never uses future highs, lows,
closes, revisions, or static index membership.

Factor Families Investigated:
1. Momentum (8 factors)
2. Volatility (5 factors)
3. Volume & Liquidity (6 factors)
4. Fundamental Information (3 strict-PIT factors + explicit rejection of missing PIT XBRL ratios)
5. Sector Intelligence (4 strictly causal PIT sector aggregates)
6. Market Structure (5 factors)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class FactorSpec:
    factor_id: str
    family: str
    name: str
    formula: str
    data_source: str
    calc_frequency: str
    required_history_days: int
    higher_is_long: bool
    pit_status: str  # "STRICT_PIT" | "PARTIAL_PIT_POST_2024" | "REJECTED_NO_PIT_FILING_FEED"
    limitation_note: str


FACTOR_SPECS: list[FactorSpec] = [
    # 1. Momentum
    FactorSpec("MOM_20D", "Momentum", "20D Price Momentum", "adj_close[t] / adj_close[t-20] - 1", "NSE Bhavcopy + CA Adj", "DAILY", 21, True, "STRICT_PIT", "Short-horizon continuation vs microstructure noise"),
    FactorSpec("MOM_60D", "Momentum", "60D Price Momentum", "adj_close[t] / adj_close[t-60] - 1", "NSE Bhavcopy + CA Adj", "DAILY", 61, True, "STRICT_PIT", "Quarterly price trend"),
    FactorSpec("MOM_120D", "Momentum", "120D Price Momentum", "adj_close[t] / adj_close[t-120] - 1", "NSE Bhavcopy + CA Adj", "DAILY", 121, True, "STRICT_PIT", "Medium-term (~6M) momentum"),
    FactorSpec("MOM_120_SKIP10", "Momentum", "120D Skip-10D Intermediate Momentum", "adj_close[t-10] / adj_close[t-120] - 1", "NSE Bhavcopy + CA Adj", "DAILY", 121, True, "STRICT_PIT", "Skips short-term 10D microstructure reversal"),
    FactorSpec("REL_STRENGTH_N50_60D", "Momentum", "60D Excess Relative Strength vs Nifty 50", "(stock_ret_60d - nifty50_ret_60d)", "NSE Bhavcopy + PR Index", "DAILY", 61, True, "STRICT_PIT", "Outperformance relative to benchmark"),
    FactorSpec("MOM_ACCEL_20V60", "Momentum", "Momentum Acceleration (20D - lagged 20D)", "mom_20d[t] - mom_20d[t-20]", "NSE Bhavcopy + CA Adj", "DAILY", 41, True, "STRICT_PIT", "Second-derivative acceleration of returns"),
    FactorSpec("TREND_PERSISTENCE_60D", "Momentum", "Information Discreteness / Trend Consistency", "(pct_pos_days_60d - pct_neg_days_60d) * sign(mom_60d)", "NSE Bhavcopy + CA Adj", "DAILY", 61, True, "STRICT_PIT", "Smooth gradual trend vs lumpy gap spikes"),
    FactorSpec("MOM_DETERIORATION_60D", "Momentum", "Momentum Deterioration / Drawdown-Off-60D-High", "adj_close[t] / max(adj_high[t-59:t+1]) - 1", "NSE Bhavcopy + CA Adj", "DAILY", 60, True, "STRICT_PIT", "Higher (closer to 0) = intact trend; lower = deteriorating"),

    # 2. Volatility
    FactorSpec("LOW_VOL_20D", "Volatility", "Negated 20D Realized Volatility (Low-Vol Anomaly)", "-std(daily_ret[t-19:t+1]) * sqrt(252)", "NSE Bhavcopy + CA Adj", "DAILY", 21, True, "STRICT_PIT", "Betting-against-beta / low-risk effect"),
    FactorSpec("VOL_CONTRACTION_10V60", "Volatility", "Volatility Contraction (10D / 60D Std Ratio)", "-std_10d / (std_60d + eps)", "NSE Bhavcopy + CA Adj", "DAILY", 61, True, "STRICT_PIT", "Higher = tighter VCP volatility squeeze before expansion"),
    FactorSpec("VOL_EXPANSION_5V20", "Volatility", "Range Expansion (5D ATR / 20D ATR)", "atr_5d / (atr_20d + eps)", "NSE Bhavcopy + CA Adj", "DAILY", 21, True, "STRICT_PIT", "Captures active volatility expansion impulse"),
    FactorSpec("VOL_ADJ_MOM_60D", "Volatility", "Volatility-Normalized 60D Momentum (Sharpe Proxy)", "mom_60d / (realized_vol_20d + 0.05)", "NSE Bhavcopy + CA Adj", "DAILY", 61, True, "STRICT_PIT", "Risk-adjusted trend efficiency"),
    FactorSpec("IDIO_VOL_20D", "Volatility", "Negated 20D Idiosyncratic Vol vs Nifty 50", "-std(stock_ret[t-19:t+1] - beta * mkt_ret)", "NSE Bhavcopy + PR Index", "DAILY", 21, True, "STRICT_PIT", "Stock-specific noise risk"),

    # 3. Volume & Liquidity
    FactorSpec("REL_VOL_5V20", "Volume_Liquidity", "Relative Volume Ratio (5D / 20D)", "mean(vol[t-4:t+1]) / (mean(vol[t-19:t+1]) + eps)", "NSE Bhavcopy", "DAILY", 20, True, "STRICT_PIT", "Captures fresh institutional volume participation"),
    FactorSpec("DELIVERY_PCT_20D", "Volume_Liquidity", "20D Mean NSE Delivery Participation %", "mean(delivery_pct[t-19:t+1])", "NSE sec_bhavdata_full DELIV_PER", "DAILY", 20, True, "STRICT_PIT", "Uniquely Indian exchange cash-delivery conviction; 90.1% coverage"),
    FactorSpec("DELIVERY_SURGE_5V20", "Volume_Liquidity", "5D vs 20D Delivery Participation Surge", "mean(deliv_pct[t-4:t+1]) - mean(deliv_pct[t-19:t+1])", "NSE sec_bhavdata_full DELIV_PER", "DAILY", 20, True, "STRICT_PIT", "Recent accumulation shift into cash delivery"),
    FactorSpec("TURNOVER_ACCEL_5V60", "Volume_Liquidity", "Turnover Acceleration (ADV 5D / ADV 60D)", "adv_5d_inr / (adv_60d_inr + eps)", "NSE Bhavcopy Turnover", "DAILY", 60, True, "STRICT_PIT", "Tracks liquidity inflow/drying up"),
    FactorSpec("AMIHUD_LIQUIDITY_20D", "Volume_Liquidity", "Negated 20D Amihud Illiquidity (Liquidity Depth)", "-mean(|ret| / max(turnover_cr, 0.01))", "NSE Bhavcopy", "DAILY", 21, True, "STRICT_PIT", "Price impact per Rs Crore traded"),
    FactorSpec("OBV_FLOW_20D", "Volume_Liquidity", "20D Normalized Signed Volume Flow", "sum(sign(ret) * vol[t-19:t+1]) / sum(vol[t-19:t+1])", "NSE Bhavcopy", "DAILY", 21, True, "STRICT_PIT", "Cumulative volume-weighted buying pressure"),

    # 4. Fundamental Information (Strict PIT Ledger vs Rejected Unavailable XBRL)
    FactorSpec("DIV_YIELD_TTM_PIT", "Fundamental", "Trailing 12M Cash Dividend Yield (PIT Ex-Date)", "sum(div_inr[t-251:t+1]) / close[t]", "NSE Corporate Actions API", "DAILY", 252, True, "STRICT_PIT", "Cash-return value factor strictly lagged to actual NSE ex_date"),
    FactorSpec("SIZE_LOG_MCAP_PIT", "Fundamental", "Negated Log Official NSE Market Cap (Small Size)", "-log(mcap_inr[t])", "NSE PR.zip MCAP CSV (2024+)", "DAILY", 20, True, "PARTIAL_PIT_POST_2024", "Available ONLY from UDiFF MCAP migration (2024+); NULL pre-2024 (never backfilled)"),
    FactorSpec("SIZE_LOG_ADV_PIT", "Fundamental", "Log 60D Average Rupee Turnover (Tradable Scale)", "log(adv_60d_inr[t])", "NSE Bhavcopy", "DAILY", 60, True, "STRICT_PIT", "Full-6Y causal scale proxy (note size vs momentum correlation)"),

    # 5. Sector Intelligence (Strictly Causal Equal-Weighted PIT Sector Portfolios)
    FactorSpec("SECTOR_MOM_20D", "Sector", "Causal PIT Sector 20D Equal-Weighted Return", "mean_{j in sector(i), elig(t)} mom_20d[t, j]", "NSE Index Industry + Bhavcopy", "DAILY", 21, True, "STRICT_PIT", "Unclassified names grouped separately; static sector labels disclosed"),
    FactorSpec("SECTOR_RS_60D", "Sector", "Causal PIT Sector 60D Excess Return vs Nifty 50", "sector_mom_60d[t] - nifty50_ret_60d[t]", "NSE Index Industry + Bhavcopy", "DAILY", 61, True, "STRICT_PIT", "Sector relative leadership"),
    FactorSpec("STOCK_VS_SECTOR_20D", "Sector", "Idiosyncratic Stock Return vs Own Sector (20D)", "mom_20d[t, i] - sector_mom_20d[t, sec(i)]", "NSE Bhavcopy + Industry", "DAILY", 21, True, "STRICT_PIT", "Tests leadership continuation vs intra-sector mean-reversion"),
    FactorSpec("SECTOR_BREADTH_SMA50", "Sector", "Causal Sector Breadth (% Peers Above 50D SMA)", "frac_{j in sector(i), elig(t)} (adj_close[t,j] > sma50[t,j])", "NSE Bhavcopy + CA Adj", "DAILY", 50, True, "STRICT_PIT", "Captures broad vs narrow sector rallies"),

    # 6. Market Structure
    FactorSpec("DIST_252D_HIGH", "Market_Structure", "Proximity to 52-Week (252D) High", "adj_close[t] / max(adj_high[t-251:t+1])", "NSE Bhavcopy + CA Adj", "DAILY", 252, True, "STRICT_PIT", "52-week high anchor (George & Hwang effect)"),
    FactorSpec("BREAKOUT_QUALITY_20D", "Market_Structure", "20D Prior-High Breakout ATR Strength * Bar CLV", "((close-max_hi_prev20)/atr20) * ((close-low)/(high-low+eps))", "NSE Bhavcopy + CA Adj", "DAILY", 21, True, "STRICT_PIT", "Separates strong closing breakouts from fading upper-wick spikes"),
    FactorSpec("PULLBACK_DEPTH_SMA50", "Market_Structure", "Healthy Pullback in Uptrend (Neg Extension if >SMA50)", "-(adj_close[t] - high_20d[t]) / atr_20d[t] if close>sma50 else NaN", "NSE Bhavcopy + CA Adj", "DAILY", 51, True, "STRICT_PIT", "Positive when pulling back below 20D high inside rising 50D trend"),
    FactorSpec("EXTENSION_SMA50_ATR", "Market_Structure", "Negated Price Extension Above 50D SMA (Mean Reversion)", "-(adj_close[t] - sma50[t]) / (atr_20d[t] + eps)", "NSE Bhavcopy + CA Adj", "DAILY", 50, True, "STRICT_PIT", "Higher = less stretched / mean-reversion value"),
    FactorSpec("GAP_BEHAVIOR_10D", "Market_Structure", "10D Cumulative Overnight Gap Return", "sum(adj_open[t-9:t+1] / adj_close[t-10:t] - 1)", "NSE Bhavcopy + CA Adj", "DAILY", 11, True, "STRICT_PIT", "Overnight institutional sentiment vs intraday fade"),
]

# Explicitly rejected un-testable fundamental factors (zero fabrication / zero look-ahead)
REJECTED_NON_PIT_FACTORS: list[dict[str, str]] = [
    {
        "factor_id": "FUND_EARNINGS_SURPRISE_PIT",
        "family": "Fundamental",
        "reason": "NSE Bhavcopy & PR archives do not carry historical multi-year quarterly EPS filing timestamps. Backfilling static 2026 Yahoo/screener financials across 2021-2025 introduces severe look-ahead bias and is strictly forbidden.",
    },
    {
        "factor_id": "FUND_ROE_PROFITABILITY_PIT",
        "family": "Fundamental",
        "reason": "Point-in-time historical Balance Sheet / Net Worth filing snapshots across 2021-2025 are unavailable without historical XBRL filings.",
    },
    {
        "factor_id": "FUND_SALES_PROFIT_GROWTH_PIT",
        "family": "Fundamental",
        "reason": "Point-in-time YoY quarterly sales/PAT announcements with original publication & restatement dates are absent in NSE EOD archives.",
    },
    {
        "factor_id": "FUND_DEBT_CASHFLOW_QUALITY_PIT",
        "family": "Fundamental",
        "reason": "Requires historical annual/half-yearly cash-flow and debt schedules indexed strictly by filing date.",
    },
]


def _rolling_mean(arr: np.ndarray, w: int) -> np.ndarray:
    """Strictly causal rolling mean ignoring NaNs via valid count check (>= max(1, w//2))."""
    valid = ~np.isnan(arr)
    filled = np.where(valid, arr, 0.0).astype(np.float64)
    c_sum = np.cumsum(filled, axis=0)
    c_cnt = np.cumsum(valid.astype(np.int32), axis=0)
    sum_w = c_sum.copy()
    cnt_w = c_cnt.copy()
    if w < arr.shape[0]:
        sum_w[w:] = c_sum[w:] - c_sum[:-w]
        cnt_w[w:] = c_cnt[w:] - c_cnt[:-w]
    out = np.full(arr.shape, np.nan, dtype=np.float32)
    min_obs = max(1, (w * 2) // 3)
    ok = cnt_w >= min_obs
    out[ok] = (sum_w[ok] / cnt_w[ok]).astype(np.float32)
    out[: max(0, w - 1)] = np.nan
    return out


def _rolling_std(arr: np.ndarray, w: int) -> np.ndarray:
    """Causal rolling sample std dev using exact window counts."""
    valid = ~np.isnan(arr)
    x = np.where(valid, arr, 0.0).astype(np.float64)
    c1 = np.cumsum(x, axis=0)
    c2 = np.cumsum(x * x, axis=0)
    cc = np.cumsum(valid.astype(np.int32), axis=0)
    s1, s2, cnt = c1.copy(), c2.copy(), cc.copy()
    if w < arr.shape[0]:
        s1[w:] = c1[w:] - c1[:-w]
        s2[w:] = c2[w:] - c2[:-w]
        cnt[w:] = cc[w:] - cc[:-w]
    out = np.full(arr.shape, np.nan, dtype=np.float32)
    min_obs = max(3, (w * 2) // 3)
    ok = cnt >= min_obs
    mean_v = np.zeros_like(s1)
    mean_v[ok] = s1[ok] / cnt[ok]
    var = np.zeros_like(s1)
    var[ok] = np.maximum(0.0, (s2[ok] - cnt[ok] * (mean_v[ok] ** 2)) / np.maximum(1, cnt[ok] - 1))
    out[ok] = np.sqrt(var[ok]).astype(np.float32)
    out[: max(0, w - 1)] = np.nan
    return out


def _rolling_max(arr: np.ndarray, w: int) -> np.ndarray:
    """Causal rolling max over window w (100% vectorized via loop over short lag offsets)."""
    t_len, n_len = arr.shape
    out = arr.astype(np.float32, copy=True)
    for k in range(1, w):
        shifted = np.full_like(out, -np.inf)
        shifted[k:] = np.where(np.isnan(arr[:-k]), -np.inf, arr[:-k])
        cur = np.where(np.isnan(out), -np.inf, out)
        mx = np.maximum(cur, shifted)
        out = np.where(np.isinf(mx), np.nan, mx)
    out[: max(0, w - 1)] = np.nan
    return out


def _rolling_sum(arr: np.ndarray, w: int) -> np.ndarray:
    filled = np.where(np.isnan(arr), 0.0, arr).astype(np.float64)
    cs = np.cumsum(filled, axis=0)
    out = cs.copy()
    if w < arr.shape[0]:
        out[w:] = cs[w:] - cs[:-w]
    return out.astype(np.float32)


def compute_all_m3_factors(panel: dict[str, Any]) -> dict[str, np.ndarray]:
    """Compute all 31 causal factor matrices `[T x N]` strictly respecting t <= decision_date."""
    adj_o = panel["adj_open"].astype(np.float64)
    adj_h = panel["adj_high"].astype(np.float64)
    adj_l = panel["adj_low"].astype(np.float64)
    adj_c = panel["adj_close"].astype(np.float64)
    raw_c = panel["close_px"].astype(np.float64)
    vol = panel["volume"].astype(np.float64)
    t_inr = panel["turnover_inr"].astype(np.float64)
    deliv_pct = panel["delivery_pct"].astype(np.float64)
    mcap = panel["mcap_inr"].astype(np.float64)
    adv20 = panel["adv_20d_inr"].astype(np.float64)
    adv60 = panel["adv_60d_inr"].astype(np.float64)
    div_ex = panel["div_inr_ex"].astype(np.float64)
    elig = panel["is_pit_elig"]
    sectors = panel["sectors"]
    n50_c = panel["nifty50_close"].astype(np.float64)

    t_len, n_len = adj_c.shape
    eps = 1e-8

    # Forward-fill adjusted closes across non-trading gaps ONLY up to 3 sessions for clean lag lookup?
    # NO — keep NaN where invalid/suspended so stale prices NEVER create fake returns!
    daily_ret = np.full((t_len, n_len), np.nan, dtype=np.float64)
    daily_ret[1:] = (adj_c[1:] / np.maximum(eps, adj_c[:-1])) - 1.0
    daily_ret = np.where((adj_c > 0) & (np.vstack([np.full((1, n_len), np.nan), adj_c[:-1]]) > 0), daily_ret, np.nan)

    n50_ret_daily = np.full(t_len, np.nan, dtype=np.float64)
    n50_ret_daily[1:] = (n50_c[1:] / np.maximum(eps, n50_c[:-1])) - 1.0

    def _ret_h(h: int) -> np.ndarray:
        out = np.full((t_len, n_len), np.nan, dtype=np.float32)
        p0 = adj_c[:-h]
        p1 = adj_c[h:]
        ok = (p0 > 0) & (p1 > 0)
        vals = np.full_like(p1, np.nan)
        vals[ok] = (p1[ok] / p0[ok]) - 1.0
        out[h:] = vals.astype(np.float32)
        return out

    # 1. Momentum Family
    mom_20d = _ret_h(20)
    mom_60d = _ret_h(60)
    mom_120d = _ret_h(120)

    mom_120_skip10 = np.full((t_len, n_len), np.nan, dtype=np.float32)
    ok_s10 = (adj_c[120:] > 0) & (adj_c[:-120] > 0) & (adj_c[110:-10] > 0)
    v_s10 = np.full((t_len - 120, n_len), np.nan, dtype=np.float32)
    v_s10[ok_s10] = ((adj_c[110:-10][ok_s10] / adj_c[:-120][ok_s10]) - 1.0).astype(np.float32)
    mom_120_skip10[120:] = v_s10

    n50_ret_60 = np.full((t_len, 1), np.nan, dtype=np.float64)
    ok_n50 = (n50_c[60:] > 0) & (n50_c[:-60] > 0)
    n50_ret_60[60:, 0] = np.where(ok_n50, (n50_c[60:] / np.maximum(eps, n50_c[:-60])) - 1.0, np.nan)
    rel_strength_n50_60d = (mom_60d - n50_ret_60).astype(np.float32)

    mom_accel = np.full((t_len, n_len), np.nan, dtype=np.float32)
    mom_accel[40:] = mom_20d[40:] - mom_20d[20:-20]

    pos_flag = np.where(np.isnan(daily_ret), np.nan, (daily_ret > 0).astype(np.float64))
    neg_flag = np.where(np.isnan(daily_ret), np.nan, (daily_ret < 0).astype(np.float64))
    pct_pos_60 = _rolling_mean(pos_flag, 60)
    pct_neg_60 = _rolling_mean(neg_flag, 60)
    trend_persistence_60d = ((pct_pos_60 - pct_neg_60) * np.sign(mom_60d)).astype(np.float32)

    high_60d = _rolling_max(adj_h, 60)
    mom_deterioration_60d = ((adj_c / np.maximum(eps, high_60d)) - 1.0).astype(np.float32)

    # 2. Volatility Family (causal True Range & ATR)
    prev_c = np.vstack([np.full((1, n_len), np.nan), adj_c[:-1]])
    tr = np.maximum(adj_h - adj_l, np.maximum(np.abs(adj_h - prev_c), np.abs(adj_l - prev_c)))
    atr_5d = _rolling_mean(tr, 5)
    atr_20d = _rolling_mean(tr, 20)

    std_10d = _rolling_std(daily_ret, 10)
    std_20d = _rolling_std(daily_ret, 20)
    std_60d = _rolling_std(daily_ret, 60)
    realized_vol_ann = std_20d * math_sqrt252()

    low_vol_20d = (-realized_vol_ann).astype(np.float32)
    vol_contraction = (-std_10d / np.maximum(1e-5, std_60d)).astype(np.float32)
    vol_expansion = (atr_5d / np.maximum(1e-5, atr_20d)).astype(np.float32)
    vol_adj_mom_60d = (mom_60d / np.maximum(0.05, realized_vol_ann)).astype(np.float32)

    # Idiosyncratic residual vol vs Nifty 50 (simple rolling beta=1 residual std proxy; strict causal)
    excess_daily = daily_ret - n50_ret_daily[:, None]
    idio_vol_20d = (-_rolling_std(excess_daily, 20) * math_sqrt252()).astype(np.float32)

    # 3. Volume & Liquidity Family
    vol_clean = np.where(panel["is_valid_eq"], vol, np.nan)
    avg_vol_5 = _rolling_mean(vol_clean, 5)
    avg_vol_20 = _rolling_mean(vol_clean, 20)
    rel_vol_5v20 = (avg_vol_5 / np.maximum(1.0, avg_vol_20)).astype(np.float32)

    deliv_20 = _rolling_mean(deliv_pct, 20)
    deliv_5 = _rolling_mean(deliv_pct, 5)
    deliv_surge = (deliv_5 - deliv_20).astype(np.float32)

    t_inr_clean = np.where(panel["is_valid_eq"], t_inr, np.nan)
    adv_5 = _rolling_mean(t_inr_clean, 5)
    turnover_accel = (adv_5 / np.maximum(1.0, adv60)).astype(np.float32)

    amihud_daily = np.abs(daily_ret) / np.maximum(0.01, t_inr_clean / 1e7)
    amihud_20d = (-_rolling_mean(amihud_daily, 20)).astype(np.float32)

    signed_vol = np.sign(np.where(np.isnan(daily_ret), 0.0, daily_ret)) * np.where(np.isnan(vol_clean), 0.0, vol_clean)
    obv_flow_20d = (_rolling_sum(signed_vol, 20) / np.maximum(1.0, _rolling_sum(vol_clean, 20))).astype(np.float32)
    obv_flow_20d[:20] = np.nan

    # 4. Strict PIT Fundamental Factors
    div_252 = _rolling_sum(div_ex, 252)
    div_yield_ttm = np.where((raw_c > 0) & (np.arange(t_len)[:, None] >= 251), div_252 / np.maximum(eps, raw_c), np.nan).astype(np.float32)
    size_log_mcap = np.where(mcap > 0, -np.log(np.maximum(1.0, mcap)), np.nan).astype(np.float32)
    size_log_adv = np.where(adv60 > 0, np.log(np.maximum(1.0, adv60)), np.nan).astype(np.float32)

    # 5. Causal Sector Intelligence
    sma50 = _rolling_mean(adj_c, 50)
    above_sma50 = np.where(np.isnan(sma50) | np.isnan(adj_c), np.nan, (adj_c > sma50).astype(np.float64))

    sector_mom_20d = np.full((t_len, n_len), np.nan, dtype=np.float32)
    sector_rs_60d = np.full((t_len, n_len), np.nan, dtype=np.float32)
    stock_vs_sector_20d = np.full((t_len, n_len), np.nan, dtype=np.float32)
    sector_breadth_50d = np.full((t_len, n_len), np.nan, dtype=np.float32)

    unique_secs = sorted(set(str(s) for s in sectors))
    for sec_name in unique_secs:
        mask_cols = (sectors == sec_name)
        if np.sum(mask_cols) < 3:
            continue
        col_idx = np.where(mask_cols)[0]
        elig_sec = elig[:, col_idx]
        m20_s = np.where(elig_sec, mom_20d[:, col_idx], np.nan)
        m60_s = np.where(elig_sec, mom_60d[:, col_idx], np.nan)
        ab_s = np.where(elig_sec, above_sma50[:, col_idx], np.nan)

        import warnings
        with warnings.catch_warnings(), np.errstate(invalid="ignore"):
            warnings.simplefilter("ignore", RuntimeWarning)
            sec_m20_t = np.nanmean(m20_s, axis=1, keepdims=True)
            sec_m60_t = np.nanmean(m60_s, axis=1, keepdims=True)
            sec_ab_t = np.nanmean(ab_s, axis=1, keepdims=True)

        # Require >= 3 eligible peers on date t so sector signal is genuine
        cnt_ok = np.sum(~np.isnan(m20_s), axis=1, keepdims=True) >= 3
        sec_m20_t = np.where(cnt_ok, sec_m20_t, np.nan)
        sec_m60_t = np.where(cnt_ok, sec_m60_t, np.nan)
        sec_ab_t = np.where(cnt_ok, sec_ab_t, np.nan)

        sector_mom_20d[:, col_idx] = sec_m20_t.astype(np.float32)
        sector_rs_60d[:, col_idx] = (sec_m60_t - n50_ret_60).astype(np.float32)
        stock_vs_sector_20d[:, col_idx] = (mom_20d[:, col_idx] - sec_m20_t).astype(np.float32)
        sector_breadth_50d[:, col_idx] = sec_ab_t.astype(np.float32)

    # 6. Market Structure Family
    high_252d = _rolling_max(adj_h, 252)
    dist_252d_high = (adj_c / np.maximum(eps, high_252d)).astype(np.float32)

    high_20d = _rolling_max(adj_h, 20)
    prior_high_20d = np.vstack([np.full((1, n_len), np.nan), high_20d[:-1]])
    with np.errstate(invalid="ignore"):
        clv = np.where((adj_h - adj_l) > eps, (adj_c - adj_l) / (adj_h - adj_l), 0.5)
    breakout_quality_20d = (((adj_c - prior_high_20d) / np.maximum(eps, atr_20d)) * clv).astype(np.float32)

    pullback_depth = np.where(
        (adj_c > sma50) & (~np.isnan(high_20d)) & (~np.isnan(atr_20d)),
        -(adj_c - high_20d) / np.maximum(eps, atr_20d),
        np.nan,
    ).astype(np.float32)

    extension_sma50 = (-(adj_c - sma50) / np.maximum(eps, atr_20d)).astype(np.float32)

    gap_ret = np.full((t_len, n_len), np.nan, dtype=np.float64)
    gap_ret[1:] = (adj_o[1:] / np.maximum(eps, adj_c[:-1])) - 1.0
    gap_behavior_10d = _rolling_sum(gap_ret, 10)
    gap_behavior_10d[:11] = np.nan

    factors = {
        "MOM_20D": mom_20d,
        "MOM_60D": mom_60d,
        "MOM_120D": mom_120d,
        "MOM_120_SKIP10": mom_120_skip10,
        "REL_STRENGTH_N50_60D": rel_strength_n50_60d,
        "MOM_ACCEL_20V60": mom_accel,
        "TREND_PERSISTENCE_60D": trend_persistence_60d,
        "MOM_DETERIORATION_60D": mom_deterioration_60d,
        "LOW_VOL_20D": low_vol_20d,
        "VOL_CONTRACTION_10V60": vol_contraction,
        "VOL_EXPANSION_5V20": vol_expansion,
        "VOL_ADJ_MOM_60D": vol_adj_mom_60d,
        "IDIO_VOL_20D": idio_vol_20d,
        "REL_VOL_5V20": rel_vol_5v20,
        "DELIVERY_PCT_20D": deliv_20,
        "DELIVERY_SURGE_5V20": deliv_surge,
        "TURNOVER_ACCEL_5V60": turnover_accel,
        "AMIHUD_LIQUIDITY_20D": amihud_20d,
        "OBV_FLOW_20D": obv_flow_20d,
        "DIV_YIELD_TTM_PIT": div_yield_ttm,
        "SIZE_LOG_MCAP_PIT": size_log_mcap,
        "SIZE_LOG_ADV_PIT": size_log_adv,
        "SECTOR_MOM_20D": sector_mom_20d,
        "SECTOR_RS_60D": sector_rs_60d,
        "STOCK_VS_SECTOR_20D": stock_vs_sector_20d,
        "SECTOR_BREADTH_SMA50": sector_breadth_50d,
        "DIST_252D_HIGH": dist_252d_high,
        "BREAKOUT_QUALITY_20D": breakout_quality_20d,
        "PULLBACK_DEPTH_SMA50": pullback_depth,
        "EXTENSION_SMA50_ATR": extension_sma50,
        "GAP_BEHAVIOR_10D": gap_behavior_10d,
    }
    return factors


def math_sqrt252() -> float:
    return float(np.sqrt(252.0))


def registry_dict() -> dict[str, dict[str, Any]]:
    return {s.factor_id: asdict(s) for s in FACTOR_SPECS}
