"""Moreshwar 3.0 Prompt 2 — Multi-Factor Predictive, Orthogonality, FDR & OOS Validation Engine.

Evaluates all 31 causal factors across 5D, 10D, 20D, 30D, and 60D causal next-open
forward returns on the full NSE PIT equity panel, with Newey-West HAC standard errors,
Fama-MacBeth cross-sectional regressions, Benjamini-Hochberg FDR & Bonferroni
corrections, circular block bootstraps, Zerodha delivery cost adjustments, and strict
In-Sample (2021-07 -> 2025-03) vs Out-of-Sample (2025-04 -> 2026-11) separation.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .factors import (
    FACTOR_SPECS,
    REJECTED_NON_PIT_FACTORS,
    compute_all_m3_factors,
    registry_dict,
)
from .panel_loader import build_or_load_m3_panel
from .schema import DEFAULT_M3_REPORTS_DIR

HORIZONS = (5, 10, 20, 30, 60)
IS_CUTOFF_DATE = "2025-03-31"
MIN_DATE_START = "2021-07-01"  # Gives >=125 trading days warmup for 120D/60D factors
# Realistic Zerodha Equity Delivery round-trip friction:
# STT (0.1% buy + 0.1% sell = 20 bps) + Exchange/SEBI/Stamp (~2 bps) + GST + Slippage (7.5*2 = 15 bps) + ~₹15.93 DP on ~₹25k avg leg (~6.4 bps) ≈ 44 bps round-trip (~0.22% one-way)
ONE_WAY_DELIVERY_COST_RATE = 0.0022


def _norm_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _two_sided_p(t_val: float) -> float:
    if not math.isfinite(t_val):
        return 1.0
    return max(1e-12, min(1.0, 2.0 * (1.0 - _norm_cdf(abs(t_val)))))


def _newey_west_t(series: np.ndarray, lag: int) -> tuple[float, float]:
    """Newey-West HAC mean t-statistic & two-sided p-value accounting for overlap lag."""
    arr = series[~np.isnan(series)]
    n = len(arr)
    if n < 6:
        return 0.0, 1.0
    mu = float(np.mean(arr))
    dm = arr - mu
    gamma0 = float(np.dot(dm, dm)) / n
    var = gamma0
    max_lag = min(lag, n - 2)
    for l in range(1, max_lag + 1):
        w = 1.0 - l / (max_lag + 1.0)
        cov = float(np.dot(dm[l:], dm[:-l])) / n
        var += 2.0 * w * cov
    var = max(1e-14, var)
    se = math.sqrt(var / n)
    t_stat = mu / se if se > 1e-12 else 0.0
    return round(t_stat, 3), float(_two_sided_p(t_stat))


def _rankdata_avg_1d(x: np.ndarray) -> np.ndarray:
    """Fast vectorized 1D average ranking (no scipy dependency needed)."""
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(x) + 1, dtype=np.float64)
    # Handle ties fast if any exist
    sorted_x = x[order]
    if len(sorted_x) > 1 and np.any(sorted_x[1:] == sorted_x[:-1]):
        diffs = np.r_[True, sorted_x[1:] != sorted_x[:-1], True]
        starts = np.flatnonzero(diffs)
        for s, e in zip(starts[:-1], starts[1:]):
            if e - s > 1:
                avg_r = 0.5 * (s + 1 + e)
                ranks[order[s:e]] = avg_r
    return ranks


def _pearson_corr_1d(x: np.ndarray, y: np.ndarray) -> float:
    mx = x - np.mean(x)
    my = y - np.mean(y)
    denom = math.sqrt(float(np.dot(mx, mx) * np.dot(my, my)))
    return float(np.dot(mx, my) / denom) if denom > 1e-12 else 0.0


def _compute_forward_returns_next_open(panel: dict[str, Any], horizons: tuple[int, ...]) -> dict[int, np.ndarray]:
    """Causal signal on t Close executes at t+1 Open and exits at t+h Close (`adj_close[t+h]/adj_open[t+1] - 1`)."""
    adj_o = panel["adj_open"].astype(np.float64)
    adj_c = panel["adj_close"].astype(np.float64)
    t_len, n_len = adj_c.shape
    fwd: dict[int, np.ndarray] = {}
    for h in horizons:
        out = np.full((t_len, n_len), np.nan, dtype=np.float32)
        if t_len > h + 1:
            entry_o = adj_o[1 : t_len - h + 1]  # index t+1 (shape t_len - h, N)
            exit_c = adj_c[h:]                  # index t+h (wait: shift carefully!)
            # Alignment: for decision date index t in 0..t_len-h-1:
            # entry is t+1 -> adj_o[1 : t_len - h + 0]? Check indices:
            # max t such that t+h < t_len is t = t_len - 1 - h => slice [: t_len - h]
            e_o = adj_o[1 : t_len - h + 1]
            x_c = adj_c[h : t_len]
            # Wait: len(adj_o[1 : t_len - h + 1]) == t_len - h == len(adj_c[h : t_len]) ONLY when h >= 1! Wait:
            # For t=0: entry=1, exit=0+h => offset difference = h-1? Wait: slice [h:] has length t_len - h!
            # So for t in 0 .. t_len - h - 1: t+1 is 1 .. t_len - h (len t_len - h), t+h is h .. t_len - 1 (len t_len - h)!
            ok = (e_o > 0) & (x_c > 0)
            r_mat = np.full_like(e_o, np.nan)
            r_mat[ok] = (x_c[ok] / e_o[ok]) - 1.0
            # Clip extreme data anomalies solely for regression sanity? NO — raw values, only exclude invalid_bar==0 already done!
            out[: t_len - h] = r_mat.astype(np.float32)
        fwd[h] = out
    return fwd


def _evaluate_single_factor_horizon(
    *,
    fmat: np.ndarray,
    fwd_mat: np.ndarray,
    elig_mask: np.ndarray,
    dates: np.ndarray,
    rebal_indices: np.ndarray,
    is_bull_regime: np.ndarray,
    horizon: int,
    higher_is_long: bool,
) -> dict[str, Any]:
    """Compute IS/OOS IC, Rank IC, Quintiles, Spread, Monotonicity, Turnover, Fama-MacBeth & Block Bootstrap."""
    sig_mat = fmat if higher_is_long else -fmat
    ic_is, ric_is, fm_is, qrets_is, univ_is = [], [], [], [], []
    ic_oos, ric_oos, fm_oos, qrets_oos, univ_oos = [], [], [], [], []
    ric_bull, ric_bear = [], []

    prev_q5_set: set[int] | None = None
    prev_q1_set: set[int] | None = None
    turnovers_is: list[float] = []
    turnovers_oos: list[float] = []
    rank_persist_corrs: list[float] = []
    prev_rank_vec: np.ndarray | None = None

    for ti in rebal_indices:
        dt_s = str(dates[ti])
        if dt_s < MIN_DATE_START:
            continue
        valid = elig_mask[ti] & (~np.isnan(sig_mat[ti])) & (~np.isnan(fwd_mat[ti]))
        cnt = int(np.sum(valid))
        if cnt < 40:
            continue

        idx_valid = np.flatnonzero(valid)
        s_vec = sig_mat[ti, idx_valid].astype(np.float64)
        r_vec = fwd_mat[ti, idx_valid].astype(np.float64)
        # Winsorize forward returns ONLY in Pearson/FM regression at [1%, 99%] cross-sectionally? No, keep rank exact; winsorize raw r in FM only if extreme outlier!
        r_rank = _rankdata_avg_1d(r_vec)
        s_rank = _rankdata_avg_1d(s_vec)

        ic_val = _pearson_corr_1d(s_vec, r_vec)
        ric_val = _pearson_corr_1d(s_rank, r_rank)

        # Fama-MacBeth slope on cross-sectionally z-scored signal
        sd_s = float(np.std(s_vec))
        if sd_s > 1e-10:
            z_s = (s_vec - float(np.mean(s_vec))) / sd_s
            fm_slope = float(np.mean(z_s * (r_vec - float(np.mean(r_vec)))))
        else:
            fm_slope = 0.0

        # Quintile portfolios (Q1=lowest signal, Q5=highest signal)
        order = np.argsort(s_rank, kind="mergesort")
        q_edges = np.array_split(order, 5)
        q_means = [float(np.mean(r_vec[idxs])) for idxs in q_edges]
        u_mean = float(np.mean(r_vec))
        q5_cols = set(idx_valid[q_edges[4]].tolist())
        q1_cols = set(idx_valid[q_edges[0]].tolist())

        # Compute exact portfolio turnover at horizon step
        to_q5 = (1.0 - len(q5_cols & prev_q5_set) / max(1, len(q5_cols))) if prev_q5_set else 1.0
        to_q1 = (1.0 - len(q1_cols & prev_q1_set) / max(1, len(q1_cols))) if prev_q1_set else 1.0
        prev_q5_set, prev_q1_set = q5_cols, q1_cols
        ls_to = 0.5 * (to_q5 + to_q1)

        # Rank autocorrelation across full cross-section
        cur_full_rank = np.full(fmat.shape[1], np.nan, dtype=np.float64)
        cur_full_rank[idx_valid] = s_rank / max(1.0, cnt)
        if prev_rank_vec is not None:
            both = (~np.isnan(cur_full_rank)) & (~np.isnan(prev_rank_vec))
            if int(np.sum(both)) >= 30:
                rank_persist_corrs.append(_pearson_corr_1d(cur_full_rank[both], prev_rank_vec[both]))
        prev_rank_vec = cur_full_rank

        if bool(is_bull_regime[ti]):
            ric_bull.append(ric_val)
        else:
            ric_bear.append(ric_val)

        if dt_s <= IS_CUTOFF_DATE:
            ic_is.append(ic_val)
            ric_is.append(ric_val)
            fm_is.append(fm_slope)
            qrets_is.append(q_means)
            univ_is.append(u_mean)
            turnovers_is.append(ls_to)
        else:
            ic_oos.append(ic_val)
            ric_oos.append(ric_val)
            fm_oos.append(fm_slope)
            qrets_oos.append(q_means)
            univ_oos.append(u_mean)
            turnovers_oos.append(ls_to)

    def _summarize(ic_arr, ric_arr, fm_arr, q_arr, u_arr, to_arr) -> dict[str, Any]:
        if len(ric_arr) < 2:
            return {
                "n_periods": len(ric_arr),
                "mean_ic": 0.0,
                "mean_rank_ic": 0.0,
                "ic_ir": 0.0,
                "hit_rate_ic_pct": 0.0,
                "nw_tstat": 0.0,
                "p_value": 1.0,
                "fm_slope_bps": 0.0,
                "fm_nw_tstat": 0.0,
                "fm_p_value": 1.0,
                "quintile_mean_period_pct": [0.0, 0.0, 0.0, 0.0, 0.0],
                "monotonicity_rho": 0.0,
                "ls_gross_period_pct": 0.0,
                "ls_net_period_pct": 0.0,
                "ls_net_ann_pct": 0.0,
                "q5_excess_net_ann_pct": 0.0,
                "avg_turnover_frac": 0.0,
                "bh_q_value_155": 1.0,
                "bonferroni_p_155": 1.0,
            }
        a_ic = np.array(ic_arr, dtype=np.float64)
        a_ric = np.array(ric_arr, dtype=np.float64)
        a_fm = np.array(fm_arr, dtype=np.float64)
        q_mat = np.array(q_arr, dtype=np.float64)  # (P, 5)
        u_vec = np.array(u_arr, dtype=np.float64)
        to_mean = float(np.mean(to_arr)) if to_arr else 1.0

        nw_t, p_val = _newey_west_t(a_ric, lag=1)
        fm_t, fm_p = _newey_west_t(a_fm, lag=1)

        q_means = np.mean(q_mat, axis=0)
        q_ranks = _rankdata_avg_1d(q_means)
        mono = _pearson_corr_1d(np.arange(1.0, 6.0), q_ranks)

        # Spread Q5 - Q1 per period; Net spreads subtract round-trip Zerodha delivery cost * turnover
        ls_gross = q_mat[:, 4] - q_mat[:, 0]
        # Each side pays 2 * ONE_WAY_DELIVERY_COST_RATE per full round-trip turnover
        ls_cost_per_period = 2.0 * (2.0 * ONE_WAY_DELIVERY_COST_RATE) * to_mean
        q5_cost_per_period = (2.0 * ONE_WAY_DELIVERY_COST_RATE) * to_mean
        ls_net = ls_gross - ls_cost_per_period
        q5_excess_gross = q_mat[:, 4] - u_vec
        q5_excess_net = q5_excess_gross - q5_cost_per_period

        periods_per_yr = 252.0 / horizon
        ann_ls_net = float(np.mean(ls_net)) * periods_per_yr * 100.0
        ann_q5_net_excess = float(np.mean(q5_excess_net)) * periods_per_yr * 100.0
        ric_sd = float(np.std(a_ric, ddof=1)) if len(a_ric) > 1 else 1.0

        return {
            "n_periods": len(ric_arr),
            "mean_ic": round(float(np.mean(a_ic)), 4),
            "mean_rank_ic": round(float(np.mean(a_ric)), 4),
            "ic_ir": round(float(np.mean(a_ric)) / max(1e-6, ric_sd), 3),
            "hit_rate_ic_pct": round(float(np.mean(a_ric > 0)) * 100.0, 1),
            "nw_tstat": nw_t,
            "p_value": round(p_val, 8),
            "fm_slope_bps": round(float(np.mean(a_fm)) * 10_000.0, 2),
            "fm_nw_tstat": fm_t,
            "fm_p_value": round(fm_p, 8),
            "quintile_mean_period_pct": [round(float(x) * 100.0, 3) for x in q_means],
            "monotonicity_rho": round(mono, 3),
            "ls_gross_period_pct": round(float(np.mean(ls_gross)) * 100.0, 3),
            "ls_net_period_pct": round(float(np.mean(ls_net)) * 100.0, 3),
            "ls_net_ann_pct": round(ann_ls_net, 2),
            "q5_excess_net_ann_pct": round(ann_q5_net_excess, 2),
            "avg_turnover_frac": round(to_mean, 3),
        }

    is_res = _summarize(ic_is, ric_is, fm_is, qrets_is, univ_is, turnovers_is)
    oos_res = _summarize(ic_oos, ric_oos, fm_oos, qrets_oos, univ_oos, turnovers_oos)

    # Moving-block bootstrap on IS+OOS period Rank IC series (500 sims, block_len=3)
    all_ric = np.array(ric_is + ric_oos, dtype=np.float64)
    boot_p05, boot_p50, boot_prob_pos = 0.0, 0.0, 0.0
    if len(all_ric) >= 8:
        rng = np.random.default_rng(20261119)
        blk = min(3, max(1, len(all_ric) // 4))
        means_b = []
        n_r = len(all_ric)
        for _ in range(500):
            seq = []
            while len(seq) < n_r:
                st = int(rng.integers(0, n_r))
                for b in range(blk):
                    seq.append(all_ric[(st + b) % n_r])
                    if len(seq) >= n_r:
                        break
            means_b.append(float(np.mean(seq)))
        means_b.sort()
        boot_p05 = round(means_b[int(0.05 * len(means_b))], 4)
        boot_p50 = round(means_b[int(0.50 * len(means_b))], 4)
        boot_prob_pos = round(100.0 * sum(1 for x in means_b if x > 0) / len(means_b), 1)

    return {
        "in_sample": is_res,
        "out_of_sample": oos_res,
        "regime_stability": {
            "bull_mean_rank_ic": round(float(np.mean(ric_bull)), 4) if ric_bull else 0.0,
            "bull_periods": len(ric_bull),
            "non_bull_mean_rank_ic": round(float(np.mean(ric_bear)), 4) if ric_bear else 0.0,
            "non_bull_periods": len(ric_bear),
        },
        "rank_autocorr": round(float(np.mean(rank_persist_corrs)), 3) if rank_persist_corrs else 0.0,
        "bootstrap_rank_ic": {
            "p05": boot_p05,
            "p50": boot_p50,
            "prob_positive_pct": boot_prob_pos,
        },
    }


def _benjamini_hochberg(p_values: list[float]) -> list[float]:
    m = len(p_values)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: p_values[i])
    q_vals = [1.0] * m
    prev_q = 1.0
    for rank_rev in range(m - 1, -1, -1):
        idx = order[rank_rev]
        raw_q = p_values[idx] * m / (rank_rev + 1)
        prev_q = min(prev_q, raw_q, 1.0)
        q_vals[idx] = round(prev_q, 8)
    return q_vals


def run_moreshwar_v3_factor_research(
    *,
    panel: dict[str, Any] | None = None,
    reports_dir: Path = DEFAULT_M3_REPORTS_DIR / "factor_research",
) -> dict[str, Any]:
    """Run full Prompt 2 Multi-Factor Discovery, Independence, FDR & OOS Validation."""
    pnl = panel or build_or_load_m3_panel()
    dates = pnl["dates"]
    elig = pnl["is_pit_elig"]
    n50_c = pnl["nifty50_close"].astype(np.float64)

    # Causal Nifty 50 > 200D SMA regime flag (strictly <= t) for regime-stability diagnostics
    t_len = len(dates)
    n50_sma200 = np.full(t_len, np.nan, dtype=np.float64)
    for ti in range(199, t_len):
        win = n50_c[ti - 199 : ti + 1]
        if np.sum(~np.isnan(win)) >= 150:
            n50_sma200[ti] = float(np.nanmean(win))
    is_bull_regime = np.where(~np.isnan(n50_sma200), n50_c > n50_sma200, True)

    factor_mats = compute_all_m3_factors(pnl)
    fwd_mats = _compute_forward_returns_next_open(pnl, HORIZONS)

    # 1. Evaluate all 31 factors x 5 horizons (non-overlapping dates stepped by horizon)
    raw_results: dict[str, dict[int, dict[str, Any]]] = {}
    all_tests_index: list[tuple[str, int, float]] = []

    for spec in FACTOR_SPECS:
        fid = spec.factor_id
        fmat = factor_mats[fid]
        raw_results[fid] = {}
        for h in HORIZONS:
            step_idx = np.arange(0, t_len - h - 1, h)
            ev = _evaluate_single_factor_horizon(
                fmat=fmat,
                fwd_mat=fwd_mats[h],
                elig_mask=elig,
                dates=dates,
                rebal_indices=step_idx,
                is_bull_regime=is_bull_regime,
                horizon=h,
                higher_is_long=spec.higher_is_long,
            )
            raw_results[fid][h] = ev
            all_tests_index.append((fid, h, float(ev["in_sample"]["p_value"])))

    # 2. Benjamini-Hochberg FDR & Bonferroni across ALL 155 factor-horizon tests
    q_all = _benjamini_hochberg([p for _, _, p in all_tests_index])
    m_tests = len(all_tests_index)
    for idx_k, (fid, h, p_raw) in enumerate(all_tests_index):
        raw_results[fid][h]["in_sample"]["bh_q_value_155"] = q_all[idx_k]
        raw_results[fid][h]["in_sample"]["bonferroni_p_155"] = round(min(1.0, p_raw * m_tests), 8)

    # 3. Cross-Factor Rank Correlation Matrix (evaluated on 20D non-overlapping sample dates)
    corr_dates = np.arange(252, t_len - 21, 20)
    fids = [s.factor_id for s in FACTOR_SPECS]
    n_f = len(fids)
    corr_sum = np.zeros((n_f, n_f), dtype=np.float64)
    corr_cnt = np.zeros((n_f, n_f), dtype=np.int32)
    z_mats: dict[str, np.ndarray] = {}

    for ti in corr_dates:
        # Build normalized rank vector per factor on date ti
        rv: list[np.ndarray | None] = []
        base_ok = elig[ti]
        for fid in fids:
            col = factor_mats[fid][ti]
            ok = base_ok & (~np.isnan(col))
            if int(np.sum(ok)) < 40:
                rv.append(None)
            else:
                vec = np.full(col.shape, np.nan, dtype=np.float64)
                rks = _rankdata_avg_1d(col[ok].astype(np.float64))
                vec[ok] = (rks - np.mean(rks)) / max(1e-6, np.std(rks))
                rv.append(vec)
        for a in range(n_f):
            if rv[a] is None:
                continue
            for b in range(a, n_f):
                if rv[b] is None:
                    continue
                both = (~np.isnan(rv[a])) & (~np.isnan(rv[b]))
                if int(np.sum(both)) >= 40:
                    c_val = _pearson_corr_1d(rv[a][both], rv[b][both])
                    corr_sum[a, b] += c_val
                    corr_cnt[a, b] += 1
                    if a != b:
                        corr_sum[b, a] += c_val
                        corr_cnt[b, a] += 1

    corr_matrix: dict[str, dict[str, float]] = {}
    for a, fa in enumerate(fids):
        corr_matrix[fa] = {}
        for b, fb in enumerate(fids):
            corr_matrix[fa][fb] = round(float(corr_sum[a, b] / corr_cnt[a, b]), 3) if corr_cnt[a, b] > 0 else 0.0

    # 4. Select best horizon per factor & run Greedy Orthogonality Pruning + Multivariate Incremental Check
    spec_map = registry_dict()
    factor_scorecard: list[dict[str, Any]] = []
    for s in FACTOR_SPECS:
        fid = s.factor_id
        # Primary evaluation focuses on 20D & 60D swing horizons while inspecting 5..60D curve
        h_best = max((10, 20, 30, 60), key=lambda hh: raw_results[fid][hh]["in_sample"].get("ic_ir", 0.0))
        res_best = raw_results[fid][h_best]
        res_20 = raw_results[fid][20]
        is_m = res_best["in_sample"]
        oos_m = res_best["out_of_sample"]

        # Objective gate before redundancy check
        passes_stat = (
            is_m["mean_rank_ic"] > 0.012
            and is_m["bh_q_value_155"] <= 0.10
            and oos_m["mean_rank_ic"] > 0.0
            and is_m["monotonicity_rho"] >= 0.50
            and (is_m["q5_excess_net_ann_pct"] > 0.0 or is_m["ls_net_ann_pct"] > 0.0)
        )
        factor_scorecard.append({
            "factor_id": fid,
            "family": s.family,
            "name": s.name,
            "best_horizon_days": h_best,
            "is_mean_rank_ic": is_m["mean_rank_ic"],
            "is_ic_ir": is_m["ic_ir"],
            "is_nw_tstat": is_m["nw_tstat"],
            "is_bh_q_155": is_m["bh_q_value_155"],
            "is_bonferroni_p_155": is_m["bonferroni_p_155"],
            "is_monotonicity": is_m["monotonicity_rho"],
            "is_ls_net_ann_pct": is_m["ls_net_ann_pct"],
            "is_q5_excess_net_ann_pct": is_m["q5_excess_net_ann_pct"],
            "is_avg_turnover": is_m["avg_turnover_frac"],
            "oos_mean_rank_ic": oos_m["mean_rank_ic"],
            "oos_ic_ir": oos_m["ic_ir"],
            "oos_ls_net_ann_pct": oos_m["ls_net_ann_pct"],
            "oos_q5_excess_net_ann_pct": oos_m["q5_excess_net_ann_pct"],
            "h20_is_rank_ic": res_20["in_sample"]["mean_rank_ic"],
            "h20_oos_rank_ic": res_20["out_of_sample"]["mean_rank_ic"],
            "rank_autocorr": res_best["rank_autocorr"],
            "bootstrap_p05_ic": res_best["bootstrap_rank_ic"]["p05"],
            "bootstrap_prob_pos_pct": res_best["bootstrap_rank_ic"]["prob_positive_pct"],
            "bull_rank_ic": res_best["regime_stability"]["bull_mean_rank_ic"],
            "non_bull_rank_ic": res_best["regime_stability"]["non_bull_mean_rank_ic"],
            "passes_individual_gate": passes_stat,
            "pit_status": s.pit_status,
        })

    # Rank candidates by combined IS IC_IR + OOS consistency
    factor_scorecard.sort(
        key=lambda r: (
            1 if r["passes_individual_gate"] else 0,
            r["is_ic_ir"] + 0.5 * (1.0 if r["oos_mean_rank_ic"] > 0 else -1.0),
        ),
        reverse=True,
    )

    validated_factors: list[dict[str, Any]] = []
    rejected_factors: list[dict[str, Any]] = []

    for row in factor_scorecard:
        fid = row["factor_id"]
        if not row["passes_individual_gate"]:
            reasons = []
            if row["is_mean_rank_ic"] <= 0.012:
                reasons.append(f"weak IS Rank IC ({row['is_mean_rank_ic']:+.4f})")
            if row["is_bh_q_155"] > 0.10:
                reasons.append(f"failed BH FDR (q={row['is_bh_q_155']:.4f})")
            if row["oos_mean_rank_ic"] <= 0.0:
                reasons.append(f"OOS Rank IC reversed/zero ({row['oos_mean_rank_ic']:+.4f})")
            if row["is_monotonicity"] < 0.50:
                reasons.append(f"non-monotonic quintiles ({row['is_monotonicity']:+.2f})")
            if row["is_q5_excess_net_ann_pct"] <= 0.0 and row["is_ls_net_ann_pct"] <= 0.0:
                reasons.append("cost/turnover eroded net alpha")
            rejected_factors.append({**row, "status": "REJECTED", "rejection_reason": "; ".join(reasons)})
            continue

        # Check cross-sectional redundancy (|corr| >= 0.70 with an already selected stronger factor)
        dup_with = None
        for sel in validated_factors:
            c = abs(corr_matrix[fid].get(sel["factor_id"], 0.0))
            if c >= 0.70:
                dup_with = (sel["factor_id"], c)
                break
        if dup_with is not None:
            rejected_factors.append({
                **row,
                "status": "REJECTED_REDUNDANT",
                "rejection_reason": f"Redundant rank correlation |rho|={dup_with[1]:.3f} >= 0.70 with stronger {dup_with[0]}",
            })
        else:
            max_corr_sel = max((abs(corr_matrix[fid].get(s["factor_id"], 0.0)) for s in validated_factors), default=0.0)
            validated_factors.append({**row, "status": "VALIDATED", "max_corr_with_selected": round(max_corr_sel, 3)})

    # Append explicitly rejected un-testable PIT fundamental factors
    for r_non_pit in REJECTED_NON_PIT_FACTORS:
        rejected_factors.append({
            "factor_id": r_non_pit["factor_id"],
            "family": r_non_pit["family"],
            "status": "REJECTED_UNAVAILABLE_PIT",
            "rejection_reason": r_non_pit["reason"],
        })

    # Build versioned Factor Registry
    factor_registry = {}
    for row in factor_scorecard:
        fid = row["factor_id"]
        factor_registry[fid] = {
            **spec_map[fid],
            "horizon_results": {str(h): raw_results[fid][h] for h in HORIZONS},
            "summary": row,
            "top_correlated_peers": sorted(
                [(k, v) for k, v in corr_matrix[fid].items() if k != fid],
                key=lambda kv: -abs(kv[1]),
            )[:5],
        }

    report = {
        "research_version": "3.0.0-p2-factor-discovery",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_window": f"{dates[0]} -> {dates[-1]} ({t_len} sessions)",
        "split": {"in_sample": f"{MIN_DATE_START} -> {IS_CUTOFF_DATE}", "out_of_sample": f"2025-04-01 -> {dates[-1]}"},
        "cost_model": f"Zerodha Equity Delivery ({ONE_WAY_DELIVERY_COST_RATE*10000:.1f} bps one-way incl STT/DP/7.5bps slippage)",
        "total_factors_tested": len(FACTOR_SPECS),
        "total_horizon_hypotheses": m_tests,
        "validated_count": len(validated_factors),
        "rejected_count": len(rejected_factors),
        "validated_factors": validated_factors,
        "rejected_factors": rejected_factors,
        "full_scorecard": factor_scorecard,
        "correlation_matrix": corr_matrix,
        "factor_registry": factor_registry,
    }

    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "factor_research_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (reports_dir / "factor_correlation_matrix.json").write_text(json.dumps(corr_matrix, indent=2), encoding="utf-8")
    (reports_dir / "factor_registry.json").write_text(json.dumps(factor_registry, indent=2), encoding="utf-8")
    (reports_dir / "validated_factors.json").write_text(json.dumps({
        "validated_factors": validated_factors,
        "rejected_factors": rejected_factors,
    }, indent=2), encoding="utf-8")

    _write_factor_markdown_report(reports_dir=reports_dir, report=report)
    return report


def _write_factor_markdown_report(*, reports_dir: Path, report: dict[str, Any]) -> None:
    lines = [
        "# MORESHWAR 3.0 (PROMPT 2) — MULTI-FACTOR ALPHA DISCOVERY & VALIDATION REPORT",
        "",
        f"- **Dataset:** `{report['dataset_window']}` (NSE Active + Delisted/Merged PIT Equities, ETFs excluded)",
        f"- **Chronological Split:** IS `{report['split']['in_sample']}` | OOS `{report['split']['out_of_sample']}`",
        f"- **Cost Model:** {report['cost_model']}",
        f"- **Multiple Testing Scope:** `{report['total_factors_tested']}` factors × `5` horizons = `{report['total_horizon_hypotheses']}` hypotheses (Benjamini-Hochberg FDR + Bonferroni + Block Bootstrap)",
        f"- **Validated Independent Factors:** **`{report['validated_count']}`** | **Rejected / Redundant / Non-PIT:** **`{report['rejected_count']}`**",
        "",
        "## 1. Validated Independent Factors Entering Prompt 3",
        "",
        "| Rank | Factor ID | Family | Best H | IS Rank IC | IS IC_IR | NW t-stat | BH q (155) | OOS Rank IC | Mono | Net Q5 Excess/yr | Max |rho| |",
        "| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for i, f in enumerate(report["validated_factors"], 1):
        lines.append(
            f"| {i} | `{f['factor_id']}` | {f['family']} | {f['best_horizon_days']}D | "
            f"{f['is_mean_rank_ic']:+.4f} | {f['is_ic_ir']:+.3f} | {f['is_nw_tstat']:+.2f} | "
            f"{f['is_bh_q_155']:.4f} | **{f['oos_mean_rank_ic']:+.4f}** | {f['is_monotonicity']:+.2f} | "
            f"**{f['is_q5_excess_net_ann_pct']:+.2f}%** | {f.get('max_corr_with_selected', 0.0):.2f} |"
        )

    lines += [
        "",
        "## 2. Full 31-Factor Multi-Horizon Predictive & Cost-Adjusted Table",
        "",
        "| Factor ID | Family | H | IS Rank IC | OOS Rank IC | Net L-S/yr | Q5-Univ Net/yr | TO | Bull IC | Non-Bull IC | Boot P05 IC | Gate |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |",
    ]
    for f in report["full_scorecard"]:
        gate = "PASS" if f["passes_individual_gate"] else "FAIL"
        lines.append(
            f"| `{f['factor_id']}` | {f['family']} | {f['best_horizon_days']}D | "
            f"{f['is_mean_rank_ic']:+.4f} | {f['oos_mean_rank_ic']:+.4f} | "
            f"{f['is_ls_net_ann_pct']:+.2f}% | {f['is_q5_excess_net_ann_pct']:+.2f}% | "
            f"{f['is_avg_turnover']:.2f} | {f['bull_rank_ic']:+.4f} | {f['non_bull_rank_ic']:+.4f} | "
            f"{f['bootstrap_p05_ic']:+.4f} | {gate} |"
        )

    lines += [
        "",
        "## 3. Rejected & Redundant Factors (Explicitly Documented)",
        "",
        "| Factor ID | Family | Status | Exact Reason |",
        "| :--- | :--- | :--- | :--- |",
    ]
    for r in report["rejected_factors"]:
        lines.append(f"| `{r['factor_id']}` | {r.get('family', '—')} | `{r['status']}` | {r['rejection_reason']} |")

    (reports_dir / "FACTOR_DISCOVERY_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    rep = run_moreshwar_v3_factor_research()
    print(json.dumps({
        "validated_factors": rep["validated_factors"],
        "rejected_count": rep["rejected_count"],
        "split": rep["split"],
    }, indent=2))
