"""Moreshwar 3.0 Prompt 4 — Causal Market Regime Intelligence & Adaptive Allocation Engine.

Combines broad Indian market trend, cross-sectional PIT breadth (% > SMA50/SMA200,
Advance-Decline ratio), sector breadth, market drawdown, volatility state/transition,
and volume participation — evaluated strictly causally (`<= t`) on Prompt 1's PIT
NSE equity panel.

Compares 5 Regime Detection Models chronologically out-of-sample (`>= 2025-04-01`):
1. RULE_MULTI: Multi-factor rule-based classifier with hysteresis (9 rich states -> 5 core regimes)
2. VOL_STATE: Volatility level & vol-expansion state classifier
3. CUSUM_CP: Causal CUSUM structural change-point detector on benchmark/breadth drift
4. KMEANS_4: Unsupervised K-Means fitted ONLY on IS (`<= 2025-03-31`) standardized features
5. HMM_3: 3-state Gaussian Hidden Markov Model fitted ONLY on IS (`<= 2025-03-31`) with
   strictly causal forward-only state probability filtering `P(S_t | x_1:t)`!
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .factors import _rolling_max, _rolling_mean, _rolling_std
from .panel_loader import build_or_load_m3_panel
from .schema import DEFAULT_M3_REPORTS_DIR

IS_CUTOFF_DATE = "2025-03-31"
MIN_REGIME_EVAL_DATE = "2022-01-03"  # Needs ~252D SMA200 & 252D peak warmup


def compute_causal_market_features(panel: dict[str, Any]) -> dict[str, Any]:
    """Compute strictly causal daily Indian market & PIT cross-sectional breadth features."""
    dates = panel["dates"]
    t_len = len(dates)
    adj_c = panel["adj_close"].astype(np.float64)
    elig = panel["is_pit_elig"]
    sectors = panel["sectors"]
    n50 = panel["nifty50_close"].astype(np.float64)
    t_inr = panel["turnover_inr"].astype(np.float64)

    # Nifty 50 indicators (strictly trailing)
    n50_ret1 = np.zeros(t_len, dtype=np.float64)
    n50_ret1[1:] = np.where((n50[1:] > 0) & (n50[:-1] > 0), (n50[1:] / np.maximum(1e-6, n50[:-1])) - 1.0, 0.0)

    def _vec_sma(arr: np.ndarray, w: int) -> np.ndarray:
        out = np.full(len(arr), np.nan, dtype=np.float64)
        for ti in range(w - 1, len(arr)):
            win = arr[ti - w + 1 : ti + 1]
            v = win[~np.isnan(win)]
            if len(v) >= max(5, w // 2):
                out[ti] = float(np.mean(v))
        return out

    n50_sma20 = _vec_sma(n50, 20)
    n50_sma50 = _vec_sma(n50, 50)
    n50_sma200 = _vec_sma(n50, 200)

    n50_ret20 = np.full(t_len, np.nan, dtype=np.float64)
    n50_ret60 = np.full(t_len, np.nan, dtype=np.float64)
    n50_ret20[20:] = np.where((n50[20:] > 0) & (n50[:-20] > 0), (n50[20:] / np.maximum(1e-6, n50[:-20])) - 1.0, np.nan)
    n50_ret60[60:] = np.where((n50[60:] > 0) & (n50[:-60] > 0), (n50[60:] / np.maximum(1e-6, n50[:-60])) - 1.0, np.nan)

    n50_vol20 = np.full(t_len, np.nan, dtype=np.float64)
    n50_vol60 = np.full(t_len, np.nan, dtype=np.float64)
    for ti in range(19, t_len):
        n50_vol20[ti] = float(np.std(n50_ret1[ti - 19 : ti + 1], ddof=1)) * math.sqrt(252.0) * 100.0
    for ti in range(59, t_len):
        n50_vol60[ti] = float(np.std(n50_ret1[ti - 59 : ti + 1], ddof=1)) * math.sqrt(252.0) * 100.0

    # Trailing 252D Drawdown
    n50_dd252 = np.zeros(t_len, dtype=np.float64)
    for ti in range(t_len):
        win = n50[max(0, ti - 251) : ti + 1]
        v = win[~np.isnan(win)]
        if len(v) > 0 and not np.isnan(n50[ti]):
            pk = float(np.max(v))
            n50_dd252[ti] = ((n50[ti] / max(1e-6, pk)) - 1.0) * 100.0

    # Cross-sectional PIT breadth metrics (% > SMA50, % > SMA200, Advance-Decline 5D, Dispersion, Sector Breadth, Volume ratio)
    sma50 = _rolling_mean(adj_c, 50)
    sma200 = _rolling_mean(adj_c, 200)
    daily_ret = np.full(adj_c.shape, np.nan, dtype=np.float64)
    daily_ret[1:] = np.where((adj_c[1:] > 0) & (adj_c[:-1] > 0), (adj_c[1:] / np.maximum(1e-6, adj_c[:-1])) - 1.0, np.nan)

    with np.errstate(invalid="ignore"):
        ab50 = np.where(elig & (~np.isnan(sma50)) & (~np.isnan(adj_c)), (adj_c > sma50).astype(np.float64), np.nan)
        ab200 = np.where(elig & (~np.isnan(sma200)) & (~np.isnan(adj_c)), (adj_c > sma200).astype(np.float64), np.nan)
        adv_today = np.where(elig & (~np.isnan(daily_ret)), (daily_ret > 0).astype(np.float64), np.nan)
        pct_ab50 = np.nanmean(ab50, axis=1) * 100.0
        pct_ab200 = np.nanmean(ab200, axis=1) * 100.0
        ad_daily = np.nanmean(adv_today, axis=1) * 100.0
        xs_disp = np.nanstd(np.where(elig, daily_ret, np.nan), axis=1) * 100.0

    ad_5d = _vec_sma(ad_daily, 5)
    t_elig = np.where(elig, t_inr, np.nan)
    with np.errstate(invalid="ignore"):
        tot_t_daily = np.nansum(t_elig, axis=1)
    t_5d = _vec_sma(tot_t_daily, 5)
    t_20d = _vec_sma(tot_t_daily, 20)
    vol_part_ratio = np.where((t_20d > 0) & (~np.isnan(t_5d)), t_5d / np.maximum(1.0, t_20d), 1.0)

    # Sector breadth (% of eligible sectors with positive 20D mean return)
    ret20 = np.full(adj_c.shape, np.nan, dtype=np.float64)
    ret20[20:] = np.where((adj_c[20:] > 0) & (adj_c[:-20] > 0), (adj_c[20:] / np.maximum(1e-6, adj_c[:-20])) - 1.0, np.nan)
    unique_secs = [s for s in sorted(set(str(x) for x in sectors)) if int(np.sum(sectors == s)) >= 4 and "Unclassified" not in s]
    sec_pos_matrix = np.full((t_len, len(unique_secs)), np.nan, dtype=np.float64)
    for si, sname in enumerate(unique_secs):
        cols = np.where(sectors == sname)[0]
        sub = np.where(elig[:, cols], ret20[:, cols], np.nan)
        with np.errstate(invalid="ignore"):
            m = np.nanmean(sub, axis=1)
        cnt_s = np.sum(~np.isnan(sub), axis=1)
        sec_pos_matrix[:, si] = np.where(cnt_s >= 3, (m > 0).astype(np.float64), np.nan)
    with np.errstate(invalid="ignore"):
        sector_breadth_pos_pct = np.nanmean(sec_pos_matrix, axis=1) * 100.0

    n50_vs_sma50 = np.where((n50_sma50 > 0) & (~np.isnan(n50)), ((n50 / np.maximum(1e-6, n50_sma50)) - 1.0) * 100.0, 0.0)
    n50_vs_sma200 = np.where((n50_sma200 > 0) & (~np.isnan(n50)), ((n50 / np.maximum(1e-6, n50_sma200)) - 1.0) * 100.0, 0.0)
    vol_ratio = np.where((n50_vol60 > 0) & (~np.isnan(n50_vol20)), n50_vol20 / np.maximum(1e-6, n50_vol60), 1.0)

    # Feature Matrix X (T, 8) for clustering/HMM
    feat_names = [
        "n50_vs_sma50_pct",
        "n50_vs_sma200_pct",
        "n50_ret20_pct",
        "n50_dd252_pct",
        "n50_vol20_ann",
        "pct_above_sma50",
        "ad_5d_pct",
        "sector_breadth_pos_pct",
    ]
    X = np.column_stack([
        np.nan_to_num(n50_vs_sma50, nan=0.0),
        np.nan_to_num(n50_vs_sma200, nan=0.0),
        np.nan_to_num(n50_ret20 * 100.0, nan=0.0),
        np.nan_to_num(n50_dd252, nan=0.0),
        np.nan_to_num(n50_vol20, nan=15.0),
        np.nan_to_num(pct_ab50, nan=50.0),
        np.nan_to_num(ad_5d, nan=50.0),
        np.nan_to_num(sector_breadth_pos_pct, nan=50.0),
    ])

    return {
        "dates": dates,
        "feature_names": feat_names,
        "X": X,
        "n50_close": n50,
        "n50_ret1": n50_ret1,
        "n50_ret20_pct": np.nan_to_num(n50_ret20 * 100.0, nan=0.0),
        "n50_ret60_pct": np.nan_to_num(n50_ret60 * 100.0, nan=0.0),
        "n50_vs_sma50_pct": np.nan_to_num(n50_vs_sma50, nan=0.0),
        "n50_vs_sma200_pct": np.nan_to_num(n50_vs_sma200, nan=0.0),
        "n50_vol20_ann": np.nan_to_num(n50_vol20, nan=15.0),
        "n50_vol60_ann": np.nan_to_num(n50_vol60, nan=15.0),
        "vol_ratio_20v60": np.nan_to_num(vol_ratio, nan=1.0),
        "n50_dd252_pct": np.nan_to_num(n50_dd252, nan=0.0),
        "pct_above_sma50": np.nan_to_num(pct_ab50, nan=50.0),
        "pct_above_sma200": np.nan_to_num(pct_ab200, nan=50.0),
        "ad_5d_pct": np.nan_to_num(ad_5d, nan=50.0),
        "sector_breadth_pos_pct": np.nan_to_num(sector_breadth_pos_pct, nan=50.0),
        "vol_participation_ratio": np.nan_to_num(vol_part_ratio, nan=1.0),
        "xs_disp_pct": np.nan_to_num(xs_disp, nan=1.5),
    }


def classify_rule_based_9state(feats: dict[str, Any], *, confirm_days: int = 3) -> tuple[list[str], list[str]]:
    """Multi-input causal Rule Classifier returning (fine_9state, stable_5core_with_hysteresis)."""
    t_len = len(feats["dates"])
    fine: list[str] = []
    raw_core: list[str] = []
    for ti in range(t_len):
        vs50 = feats["n50_vs_sma50_pct"][ti]
        vs200 = feats["n50_vs_sma200_pct"][ti]
        dd = feats["n50_dd252_pct"][ti]
        vol20 = feats["n50_vol20_ann"][ti]
        vrat = feats["vol_ratio_20v60"][ti]
        ab50 = feats["pct_above_sma50"][ti]
        sec_br = feats["sector_breadth_pos_pct"][ti]
        r20 = feats["n50_ret20_pct"][ti]

        # Stress & Volatility States
        if dd <= -12.0 or vol20 >= 24.0:
            fine_s = "MARKET_STRESS"
            core_s = "STRESS_BEAR"
        elif vol20 >= 19.0:
            fine_s = "HIGH_VOLATILITY"
            core_s = "STRESS_BEAR"
        elif vrat >= 1.35 and vol20 >= 14.0:
            fine_s = "VOL_TRANSITION"
            core_s = "SIDEWAYS_CHOP"
        elif vs50 > 1.0 and vs200 > 2.0 and ab50 >= 58.0 and sec_br >= 55.0 and r20 > 1.0:
            fine_s = "STRONG_BULLISH"
            core_s = "STRONG_BULL"
        elif vs200 > 0.0 and (vs50 > -1.0 or ab50 >= 48.0):
            fine_s = "LOW_VOL_BULL" if vol20 < 10.5 else "WEAK_BULLISH"
            core_s = "WEAK_BULL"
        elif vs50 < -1.5 and vs200 < -1.5 and ab50 <= 35.0:
            fine_s = "STRONG_BEARISH"
            core_s = "STRESS_BEAR"
        elif vs200 < 0.0 and ab50 < 45.0:
            fine_s = "WEAK_BEARISH"
            core_s = "WEAK_BEAR"
        else:
            fine_s = "LOW_VOLATILITY" if vol20 < 10.0 else "SIDEWAYS"
            core_s = "SIDEWAYS_CHOP"
        fine.append(fine_s)
        raw_core.append(core_s)

    # Apply N-day hysteresis to prevent noisy daily whipsaws
    stable = [raw_core[0]]
    streak_lbl = raw_core[0]
    streak_cnt = 1
    for ti in range(1, t_len):
        cur = raw_core[ti]
        if cur == stable[-1]:
            streak_lbl = cur
            streak_cnt = 1
            stable.append(cur)
        else:
            if cur == streak_lbl:
                streak_cnt += 1
            else:
                streak_lbl = cur
                streak_cnt = 1
            # Stress triggers fast (2 days), bullish upgrades require full confirm_days
            need = 2 if cur == "STRESS_BEAR" else confirm_days
            if streak_cnt >= need:
                stable.append(cur)
            else:
                stable.append(stable[-1])
    return fine, stable


def fit_predict_causal_kmeans(feats: dict[str, Any], is_mask: np.ndarray, k: int = 4) -> list[str]:
    """Fit K-Means strictly on IS standardized features; assign every day strictly causally."""
    X = feats["X"]
    X_is = X[is_mask]
    mu = np.mean(X_is, axis=0)
    sd = np.maximum(1e-6, np.std(X_is, axis=0))
    Z_is = (X_is - mu) / sd
    Z_all = (X - mu) / sd

    # Deterministic K-Means++ style quantile init on Z_is
    rng = np.random.default_rng(42)
    idxs = np.linspace(0, len(Z_is) - 1, k, dtype=int)
    centers = Z_is[idxs].copy()
    for _ in range(40):
        dists = np.linalg.norm(Z_is[:, None, :] - centers[None, :, :], axis=2)
        labels = np.argmin(dists, axis=1)
        new_c = centers.copy()
        for ci in range(k):
            pts = Z_is[labels == ci]
            if len(pts) > 0:
                new_c[ci] = np.mean(pts, axis=0)
        if np.max(np.abs(new_c - centers)) < 1e-5:
            break
        centers = new_c

    # Order clusters semantically by raw IS mean n50_vs_sma200_pct - vol20 (bullish -> bear stress)
    raw_is_scores = []
    labels_is = np.argmin(np.linalg.norm(Z_is[:, None, :] - centers[None, :, :], axis=2), axis=1)
    for ci in range(k):
        pts_raw = X_is[labels_is == ci]
        sc = float(np.mean(pts_raw[:, 1] + 0.2 * pts_raw[:, 5] - 0.5 * pts_raw[:, 4])) if len(pts_raw) else 0.0
        raw_is_scores.append((sc, ci))
    raw_is_scores.sort(reverse=True)
    name_order = ["KM_BULL_EXPANSION", "KM_MILD_BULL", "KM_SIDEWAYS_MIX", "KM_BEAR_STRESS"]
    id_to_name = {ci: name_order[rank_i] for rank_i, (_, ci) in enumerate(raw_is_scores)}

    all_labels = np.argmin(np.linalg.norm(Z_all[:, None, :] - centers[None, :, :], axis=2), axis=1)
    return [id_to_name[int(l)] for l in all_labels]


def fit_predict_causal_hmm_3state(feats: dict[str, Any], is_mask: np.ndarray) -> tuple[list[str], np.ndarray]:
    """Fit 3-state diagonal Gaussian HMM on IS ONLY (`<= 2025-03-31`) and run strictly causal forward filtering `P(S_t | x_1:t)`."""
    # Use 4 clean stationary macro observables: [n50_ret20, n50_vol20, pct_ab50, n50_dd252]
    X4 = feats["X"][:, [2, 4, 5, 3]]
    X_is = X4[is_mask]
    mu_norm = np.mean(X_is, axis=0)
    sd_norm = np.maximum(1e-6, np.std(X_is, axis=0))
    Z_is = (X_is - mu_norm) / sd_norm
    Z_all = (X4 - mu_norm) / sd_norm

    k = 3
    # Quantile init across PC1 / composite health
    health = Z_is[:, 0] - Z_is[:, 1] + Z_is[:, 2] + Z_is[:, 3]
    q33, q66 = np.quantile(health, [0.33, 0.67])
    init_groups = [health >= q66, (health < q66) & (health >= q33), health < q33]
    means = np.array([np.mean(Z_is[g], axis=0) for g in init_groups], dtype=np.float64)
    vars_ = np.array([np.maximum(0.15, np.var(Z_is[g], axis=0)) for g in init_groups], dtype=np.float64)
    A = np.array([[0.94, 0.05, 0.01], [0.04, 0.92, 0.04], [0.01, 0.05, 0.94]], dtype=np.float64)
    pi = np.array([0.4, 0.4, 0.2], dtype=np.float64)

    def _gauss_lik(Z: np.ndarray) -> np.ndarray:
        # (T, k) log-likelihoods -> scaled likelihoods
        d = Z.shape[1]
        ll = np.zeros((len(Z), k), dtype=np.float64)
        for s in range(k):
            diff = Z - means[s]
            ll[:, s] = -0.5 * (np.sum(np.log(2.0 * math.pi * vars_[s])) + np.sum((diff ** 2) / vars_[s], axis=1))
        mx = np.max(ll, axis=1, keepdims=True)
        return np.maximum(1e-300, np.exp(ll - mx))

    # EM fitting strictly on Z_is (In-Sample only!)
    t_is = len(Z_is)
    for _ in range(25):
        B = _gauss_lik(Z_is)
        alpha = np.zeros((t_is, k), dtype=np.float64)
        alpha[0] = pi * B[0]
        alpha[0] /= np.sum(alpha[0])
        for t in range(1, t_is):
            alpha[t] = (alpha[t - 1] @ A) * B[t]
            s_a = np.sum(alpha[t])
            alpha[t] = alpha[t] / s_a if s_a > 0 else np.full(k, 1.0 / k)

        beta = np.ones((t_is, k), dtype=np.float64)
        for t in range(t_is - 2, -1, -1):
            beta[t] = A @ (B[t + 1] * beta[t + 1])
            s_b = np.sum(beta[t])
            beta[t] = beta[t] / s_b if s_b > 0 else np.full(k, 1.0 / k)

        gamma = alpha * beta
        gamma /= np.maximum(1e-12, np.sum(gamma, axis=1, keepdims=True))

        # Update parameters
        pi = gamma[0]
        for s in range(k):
            w = gamma[:, s : s + 1]
            ws = max(1e-6, float(np.sum(w)))
            means[s] = np.sum(w * Z_is, axis=0) / ws
            vars_[s] = np.maximum(0.10, np.sum(w * ((Z_is - means[s]) ** 2), axis=0) / ws)

        xi_sum = np.zeros((k, k), dtype=np.float64)
        for t in range(t_is - 1):
            mat = np.outer(alpha[t], B[t + 1] * beta[t + 1]) * A
            ms = np.sum(mat)
            if ms > 0:
                xi_sum += mat / ms
        row_s = np.maximum(1e-6, np.sum(xi_sum, axis=1, keepdims=True))
        A = np.maximum(1e-4, xi_sum / row_s)
        A /= np.sum(A, axis=1, keepdims=True)

    # Strict Causal Forward-Only Filtering across FULL Z_all (NO backward pass!)
    B_all = _gauss_lik(Z_all)
    t_all = len(Z_all)
    filt = np.zeros((t_all, k), dtype=np.float64)
    filt[0] = pi * B_all[0]
    filt[0] /= np.sum(filt[0])
    for t in range(1, t_all):
        filt[t] = (filt[t - 1] @ A) * B_all[t]
        s_f = np.sum(filt[t])
        filt[t] = filt[t] / s_f if s_f > 0 else np.full(k, 1.0 / k)

    # Order HMM states by IS composite score (healthiest -> worst)
    state_scores = [(float(means[s, 0] - means[s, 1] + means[s, 2] + means[s, 3]), s) for s in range(k)]
    state_scores.sort(reverse=True)
    hmm_names = ["HMM_BULL_EXPANSION", "HMM_CHOP_TRANSITION", "HMM_STRESS_BEAR"]
    s_map = {s: hmm_names[r_i] for r_i, (_, s) in enumerate(state_scores)}
    labels = [s_map[int(np.argmax(filt[t]))] for t in range(t_all)]
    return labels, filt


def detect_cusum_changepoints(feats: dict[str, Any], *, threshold_k: float = 0.35, h_limit: float = 4.0) -> list[str]:
    """Causal two-sided CUSUM detector on standardized daily Nifty 50 returns."""
    rets = feats["n50_ret1"]
    s_pos = 0.0
    s_neg = 0.0
    states: list[str] = []
    cur = "CP_BULL_TREND"
    for t in range(len(rets)):
        # Causal trailing 60D volatility scaling
        lo = max(0, t - 60)
        win = rets[lo:t] if t > 10 else np.array([0.01])
        sd = max(0.004, float(np.std(win)))
        z = (rets[t] / sd) if sd > 0 else 0.0
        s_pos = max(0.0, s_pos + z - threshold_k)
        s_neg = max(0.0, s_neg - z - threshold_k)
        if s_neg > h_limit:
            cur = "CP_DOWNTURN_BREAK"
            s_neg = 0.0
            s_pos = 0.0
        elif s_pos > h_limit:
            cur = "CP_BULL_TREND"
            s_pos = 0.0
            s_neg = 0.0
        states.append(cur)
    return states


def evaluate_regime_stability(labels: list[str], dates: np.ndarray, n50_ret1: np.ndarray, start_date: str) -> dict[str, Any]:
    """Measure transition frequency, false 1-day/2-day flips, duration, and forward Nifty behavior."""
    idx = [i for i, d in enumerate(dates) if str(d) >= start_date]
    if len(idx) < 20:
        return {}
    sub = [labels[i] for i in idx]
    transitions = sum(1 for k in range(1, len(sub)) if sub[k] != sub[k - 1])

    # Run lengths
    runs: list[tuple[str, int]] = []
    cur_l, cur_c = sub[0], 1
    for x in sub[1:]:
        if x == cur_l:
            cur_c += 1
        else:
            runs.append((cur_l, cur_c))
            cur_l, cur_c = x, 1
    runs.append((cur_l, cur_c))
    false_flips = sum(1 for _, c in runs[:-1] if c <= 2)
    years = max(0.1, len(sub) / 252.0)

    # Nifty 50 empirical behavior inside each regime (next-day return t+1 conditioned on regime at t!)
    by_reg: dict[str, list[float]] = {}
    for k in range(len(idx) - 1):
        lbl = labels[idx[k]]
        by_reg.setdefault(lbl, []).append(float(n50_ret1[idx[k + 1]]))
    reg_stats = {}
    for lbl, arr in sorted(by_reg.items()):
        a = np.array(arr, dtype=np.float64)
        ann_r = float(np.mean(a)) * 252.0 * 100.0
        ann_v = float(np.std(a, ddof=1)) * math.sqrt(252.0) * 100.0 if len(a) > 2 else 0.0
        reg_stats[lbl] = {
            "days": len(arr),
            "pct_of_time": round(100.0 * len(arr) / len(sub), 1),
            "nifty_next1d_ann_ret_pct": round(ann_r, 2),
            "nifty_next1d_ann_vol_pct": round(ann_v, 2),
        }

    return {
        "days": len(sub),
        "transitions": transitions,
        "transitions_per_year": round(transitions / years, 2),
        "avg_regime_duration_days": round(len(sub) / max(1, len(runs)), 1),
        "short_false_flips_le_2d": false_flips,
        "false_flip_rate_pct": round(100.0 * false_flips / max(1, transitions), 1) if transitions else 0.0,
        "regime_distribution": reg_stats,
    }


def run_regime_detection_comparison(
    *,
    panel: dict[str, Any] | None = None,
    reports_dir: Path = DEFAULT_M3_REPORTS_DIR / "regime_intelligence",
) -> dict[str, Any]:
    """Compare all 5 regime detection models strictly chronologically on IS & OOS."""
    pnl = panel or build_or_load_m3_panel()
    dates = pnl["dates"]
    feats = compute_causal_market_features(pnl)
    is_mask = np.array([(MIN_REGIME_EVAL_DATE <= str(d) <= IS_CUTOFF_DATE) for d in dates], dtype=bool)

    fine9, rule5 = classify_rule_based_9state(feats, confirm_days=3)
    kmeans4 = fit_predict_causal_kmeans(feats, is_mask=is_mask, k=4)
    hmm3, hmm_probs = fit_predict_causal_hmm_3state(feats, is_mask=is_mask)
    cusum_cp = detect_cusum_changepoints(feats)
    vol_only = [
        ("VOL_STRESS" if feats["n50_vol20_ann"][i] >= 20.0 else ("VOL_TRANSITION" if feats["vol_ratio_20v60"][i] >= 1.35 else ("VOL_LOW" if feats["n50_vol20_ann"][i] < 10.5 else "VOL_NORMAL")))
        for i in range(len(dates))
    ]

    models = {
        "RULE_5CORE_HYST": rule5,
        "RULE_9FINE": fine9,
        "KMEANS_4": kmeans4,
        "HMM_3_CAUSAL": hmm3,
        "CUSUM_CP": cusum_cp,
        "VOL_STATE": vol_only,
    }

    comparison = {}
    for mname, labels in models.items():
        comparison[mname] = {
            "in_sample": evaluate_regime_stability(labels, dates, feats["n50_ret1"], MIN_REGIME_EVAL_DATE),
            "full": evaluate_regime_stability(labels, dates, feats["n50_ret1"], MIN_REGIME_EVAL_DATE),
            "oos_only": evaluate_regime_stability(labels, dates, feats["n50_ret1"], "2025-04-01"),
        }
        # Restrict IS properly up to 2025-03-31
        is_end_i = max(i for i, d in enumerate(dates) if str(d) <= IS_CUTOFF_DATE)
        comparison[mname]["in_sample"] = evaluate_regime_stability(labels[: is_end_i + 1], dates[: is_end_i + 1], feats["n50_ret1"][: is_end_i + 1], MIN_REGIME_EVAL_DATE)

    # Save causal timeline & detection report cleanly
    timeline = [
        {
            "date": str(dates[i]),
            "n50_close": round(float(feats["n50_close"][i]), 2) if not np.isnan(feats["n50_close"][i]) else None,
            "pct_ab50": round(float(feats["pct_above_sma50"][i]), 1),
            "ad_5d": round(float(feats["ad_5d_pct"][i]), 1),
            "vol20": round(float(feats["n50_vol20_ann"][i]), 2),
            "dd252": round(float(feats["n50_dd252_pct"][i]), 2),
            "rule5": rule5[i],
            "fine9": fine9[i],
            "hmm3": hmm3[i],
            "kmeans4": kmeans4[i],
            "cusum": cusum_cp[i],
            "hmm_bull_prob": round(float(hmm_probs[i, 0]), 3),
        }
        for i in range(len(dates)) if str(dates[i]) >= MIN_REGIME_EVAL_DATE
    ]

    reports_dir.mkdir(parents=True, exist_ok=True)
    out = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "is_cutoff": IS_CUTOFF_DATE,
        "model_comparison": comparison,
        "timeline_count": len(timeline),
        "latest_regime_snapshot": timeline[-1] if timeline else {},
    }
    (reports_dir / "regime_models_comparison.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    (reports_dir / "regime_timeline.json").write_text(json.dumps(timeline, indent=2), encoding="utf-8")
    return {"comparison": out, "timeline": timeline, "models": models, "feats": feats, "hmm_probs": hmm_probs}


# Explicit Causal Strategy Allocation Policy by Core Regime (weights sum to deployed fraction; remainder in 6.5% risk-free cash)
REGIME_ALLOCATION_POLICY: dict[str, dict[str, float]] = {
    "STRONG_BULL": {
        "STRAT_A_CS_MOMENTUM": 0.30,
        "STRAT_B_VCP_BREAKOUT": 0.25,
        "STRAT_C_TREND_PULLBACK": 0.15,
        "STRAT_D_SECTOR_ROTATION": 0.20,
        "STRAT_E_DEFENSIVE_YIELD_LOWVOL": 0.10,
        "CASH_RESERVE": 0.00,
    },
    "WEAK_BULL": {
        "STRAT_A_CS_MOMENTUM": 0.25,
        "STRAT_B_VCP_BREAKOUT": 0.15,
        "STRAT_C_TREND_PULLBACK": 0.20,
        "STRAT_D_SECTOR_ROTATION": 0.20,
        "STRAT_E_DEFENSIVE_YIELD_LOWVOL": 0.15,
        "CASH_RESERVE": 0.05,
    },
    "SIDEWAYS_CHOP": {
        "STRAT_A_CS_MOMENTUM": 0.10,
        "STRAT_B_VCP_BREAKOUT": 0.05,
        "STRAT_C_TREND_PULLBACK": 0.25,
        "STRAT_D_SECTOR_ROTATION": 0.15,
        "STRAT_E_DEFENSIVE_YIELD_LOWVOL": 0.30,
        "CASH_RESERVE": 0.15,
    },
    "WEAK_BEAR": {
        "STRAT_A_CS_MOMENTUM": 0.05,
        "STRAT_B_VCP_BREAKOUT": 0.00,
        "STRAT_C_TREND_PULLBACK": 0.15,
        "STRAT_D_SECTOR_ROTATION": 0.10,
        "STRAT_E_DEFENSIVE_YIELD_LOWVOL": 0.35,
        "CASH_RESERVE": 0.35,
    },
    "STRESS_BEAR": {
        "STRAT_A_CS_MOMENTUM": 0.00,
        "STRAT_B_VCP_BREAKOUT": 0.00,
        "STRAT_C_TREND_PULLBACK": 0.10,
        "STRAT_D_SECTOR_ROTATION": 0.00,
        "STRAT_E_DEFENSIVE_YIELD_LOWVOL": 0.30,
        "CASH_RESERVE": 0.60,
    },
}


def _eval_return_stream(
    rets: np.ndarray,
    bench_rets: np.ndarray,
    dates_sub: list[str],
    initial_capital: float = 100_000.0,
    risk_free_annual: float = 0.065,
) -> dict[str, Any]:
    if len(rets) < 2:
        return {}
    navs = initial_capital * np.cumprod(1.0 + rets)
    tot_ret_pct = ((float(navs[-1]) / initial_capital) - 1.0) * 100.0
    years = max(0.05, len(rets) / 252.0)
    cagr_pct = (((float(navs[-1]) / initial_capital) ** (1.0 / years)) - 1.0) * 100.0
    pk = np.maximum.accumulate(np.concatenate([[initial_capital], navs]))
    dd = (np.concatenate([[initial_capital], navs]) - pk) / np.maximum(1.0, pk)
    max_dd_pct = abs(float(np.min(dd))) * 100.0

    rf_d = risk_free_annual / 252.0
    ex = rets - rf_d
    mu_e = float(np.mean(ex))
    sd_e = float(np.std(ex, ddof=1)) if len(ex) > 1 else 1.0
    sharpe = (mu_e / sd_e) * math.sqrt(252.0) if sd_e > 1e-10 else 0.0
    down = ex[ex < 0.0]
    ds_sd = math.sqrt(float(np.mean(down ** 2))) if len(down) > 0 else 1e-9
    sortino = (mu_e / ds_sd) * math.sqrt(252.0) if ds_sd > 1e-10 else 0.0
    ann_vol = float(np.std(rets, ddof=1)) * math.sqrt(252.0) * 100.0 if len(rets) > 1 else 0.0
    calmar = cagr_pct / max(0.5, max_dd_pct)
    b_cum = (float(np.prod(1.0 + np.where(np.isnan(bench_rets), 0.0, bench_rets))) - 1.0) * 100.0

    return {
        "window": f"{dates_sub[0]} -> {dates_sub[-1]} ({len(dates_sub)}d)",
        "total_return_pct": round(tot_ret_pct, 2),
        "cagr_pct": round(cagr_pct, 2),
        "max_drawdown_pct": round(max_dd_pct, 2),
        "sharpe_ratio": round(sharpe, 2),
        "sortino_ratio": round(sortino, 2),
        "calmar_ratio": round(calmar, 2),
        "annualised_vol_pct": round(ann_vol, 2),
        "benchmark_return_pct": round(b_cum, 2),
        "excess_vs_nifty_pp": round(tot_ret_pct - b_cum, 2),
    }


def run_adaptive_regime_allocation(
    *,
    panel: dict[str, Any] | None = None,
    strategy_bundle: dict[str, Any] | None = None,
    regime_bundle: dict[str, Any] | None = None,
    reports_dir: Path = DEFAULT_M3_REPORTS_DIR / "regime_intelligence",
    switching_cost_bps: float = 15.0,
) -> dict[str, Any]:
    """Evaluate causal regime-conditioned strategy attribution and adaptive meta-portfolio vs static baselines."""
    from .strategy_families import run_moreshwar_v3_strategy_families

    pnl = panel or build_or_load_m3_panel()
    reg_res = regime_bundle or run_regime_detection_comparison(panel=pnl, reports_dir=reports_dir)
    strat_res = strategy_bundle or run_moreshwar_v3_strategy_families(panel=pnl)

    dates = [str(d) for d in pnl["dates"]]
    d_to_idx = {d: i for i, d in enumerate(dates)}
    rule5 = reg_res["models"]["RULE_5CORE_HYST"]
    hmm3 = reg_res["models"]["HMM_3_CAUSAL"]
    feats = reg_res["feats"]
    n50_ret1 = feats["n50_ret1"]

    sids = list(strat_res["daily_ret_map"].keys())
    eval_dates = [d for d in strat_res["common_dates"] if d >= MIN_REGIME_EVAL_DATE]
    rf_daily = 0.065 / 252.0

    # 1. Regime-Conditioned Attribution per Strategy Family (using strictly causal regime at t-1 -> return at t!)
    regime_attribution: dict[str, dict[str, Any]] = {}
    for reg_name in REGIME_ALLOCATION_POLICY:
        reg_days = []
        for k in range(1, len(eval_dates)):
            dt = eval_dates[k]
            prev_dt = eval_dates[k - 1]
            ti_prev = d_to_idx[prev_dt]
            if rule5[ti_prev] == reg_name:
                reg_days.append(dt)
        strat_perf_in_reg = {}
        for sid in sids:
            r_arr = np.array([strat_res["daily_ret_map"][sid][d] for d in reg_days], dtype=np.float64) if reg_days else np.zeros(1)
            ann_ret = float(np.mean(r_arr)) * 252.0 * 100.0 if len(reg_days) else 0.0
            ann_vol = float(np.std(r_arr, ddof=1)) * math.sqrt(252.0) * 100.0 if len(reg_days) > 2 else 0.0
            sh = (ann_ret - 6.5) / max(1.0, ann_vol) if len(reg_days) > 2 else 0.0
            strat_perf_in_reg[sid] = {
                "days": len(reg_days),
                "ann_return_pct": round(ann_ret, 2),
                "ann_vol_pct": round(ann_vol, 2),
                "regime_sharpe": round(sh, 2),
            }
        regime_attribution[reg_name] = {
            "days": len(reg_days),
            "target_weights": REGIME_ALLOCATION_POLICY[reg_name],
            "strategy_behavior": strat_perf_in_reg,
        }

    # 2. Simulate Daily Meta-Portfolio Streams (strictly causal: weights decided at t-1 close applied to day t return)
    eq_w = {s: 1.0 / len(sids) for s in sids}
    prev_rule_w = {**eq_w, "CASH_RESERVE": 0.0}
    prev_hmm_w = {**eq_w, "CASH_RESERVE": 0.0}

    rets_equal: list[float] = []
    rets_rule_adaptive: list[float] = []
    rets_hmm_adaptive: list[float] = []
    rets_bench: list[float] = []
    rule_switch_events = 0
    total_switch_cost_pct = 0.0

    hmm_map = {
        "HMM_BULL_EXPANSION": REGIME_ALLOCATION_POLICY["STRONG_BULL"],
        "HMM_CHOP_TRANSITION": REGIME_ALLOCATION_POLICY["SIDEWAYS_CHOP"],
        "HMM_STRESS_BEAR": REGIME_ALLOCATION_POLICY["STRESS_BEAR"],
    }

    for k, dt in enumerate(eval_dates):
        ti = d_to_idx[dt]
        b_r = float(n50_ret1[ti])
        rets_bench.append(b_r)

        s_rets = {s: float(strat_res["daily_ret_map"][s][dt]) for s in sids}
        r_eq = sum(eq_w[s] * s_rets[s] for s in sids)
        rets_equal.append(r_eq)

        if k == 0:
            rets_rule_adaptive.append(r_eq)
            rets_hmm_adaptive.append(r_eq)
            continue

        prev_ti = d_to_idx[eval_dates[k - 1]]
        reg_lbl = rule5[prev_ti]
        target_w = REGIME_ALLOCATION_POLICY.get(reg_lbl, REGIME_ALLOCATION_POLICY["SIDEWAYS_CHOP"])

        # One-way weight turnover penalty on regime transition
        w_diff = 0.5 * sum(abs(target_w.get(s, 0.0) - prev_rule_w.get(s, 0.0)) for s in [*sids, "CASH_RESERVE"])
        sw_cost = w_diff * (switching_cost_bps / 10_000.0)
        if w_diff > 1e-4:
            rule_switch_events += 1
            total_switch_cost_pct += sw_cost * 100.0
        r_rule = sum(target_w[s] * s_rets[s] for s in sids) + target_w["CASH_RESERVE"] * rf_daily - sw_cost
        rets_rule_adaptive.append(r_rule)
        prev_rule_w = target_w

        # HMM Causal State Adaptive
        hmm_lbl = hmm3[prev_ti]
        hw = hmm_map.get(hmm_lbl, REGIME_ALLOCATION_POLICY["SIDEWAYS_CHOP"])
        hw_diff = 0.5 * sum(abs(hw.get(s, 0.0) - prev_hmm_w.get(s, 0.0)) for s in [*sids, "CASH_RESERVE"])
        h_cost = hw_diff * (switching_cost_bps / 10_000.0)
        r_hmm = sum(hw[s] * s_rets[s] for s in sids) + hw["CASH_RESERVE"] * rf_daily - h_cost
        rets_hmm_adaptive.append(r_hmm)
        prev_hmm_w = hw

    arr_eq = np.array(rets_equal, dtype=np.float64)
    arr_rule = np.array(rets_rule_adaptive, dtype=np.float64)
    arr_hmm = np.array(rets_hmm_adaptive, dtype=np.float64)
    arr_bench = np.array(rets_bench, dtype=np.float64)

    is_idx = [i for i, d in enumerate(eval_dates) if d <= IS_CUTOFF_DATE]
    oos_idx = [i for i, d in enumerate(eval_dates) if d > IS_CUTOFF_DATE]

    def _split_eval(arr: np.ndarray) -> dict[str, Any]:
        return {
            "full": _eval_return_stream(arr, arr_bench, eval_dates),
            "in_sample": _eval_return_stream(arr[is_idx], arr_bench[is_idx], [eval_dates[i] for i in is_idx]),
            "out_of_sample": _eval_return_stream(arr[oos_idx], arr_bench[oos_idx], [eval_dates[i] for i in oos_idx]),
        }

    portfolios = {
        "M3_RULE5_REGIME_ADAPTIVE": _split_eval(arr_rule),
        "M3_HMM3_CAUSAL_ADAPTIVE": _split_eval(arr_hmm),
        "M3_STATIC_EQUAL_WEIGHT_5FAM": _split_eval(arr_eq),
    }
    for sid in sids:
        s_arr = np.array([strat_res["daily_ret_map"][sid][d] for d in eval_dates], dtype=np.float64)
        portfolios[sid] = _split_eval(s_arr)

    adaptive_payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "eval_window": f"{eval_dates[0]} -> {eval_dates[-1]} ({len(eval_dates)}d)",
        "is_cutoff": IS_CUTOFF_DATE,
        "switching_cost_bps": switching_cost_bps,
        "rule5_switch_events": rule_switch_events,
        "total_switching_cost_drag_pct": round(total_switch_cost_pct, 3),
        "allocation_policy": REGIME_ALLOCATION_POLICY,
        "regime_conditioned_attribution": regime_attribution,
        "portfolio_comparison": portfolios,
    }
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "adaptive_allocation_report.json").write_text(json.dumps(adaptive_payload, indent=2), encoding="utf-8")
    _write_regime_intelligence_markdown(
        reports_dir / "REGIME_INTELLIGENCE_REPORT.md",
        reg_res["comparison"],
        adaptive_payload,
    )
    return {
        "regime_comparison": reg_res["comparison"],
        "adaptive_report": adaptive_payload,
    }


def _write_regime_intelligence_markdown(
    md_path: Path,
    reg_comp: dict[str, Any],
    adap: dict[str, Any],
) -> None:
    mc = reg_comp["model_comparison"]
    pcomp = adap["portfolio_comparison"]
    lines = [
        "# MORESHWAR 3.0 (PROMPT 4) — CAUSAL MARKET REGIME INTELLIGENCE & ADAPTIVE STRATEGY ALLOCATION REPORT",
        "",
        f"- **Evaluation Window:** `{adap['eval_window']}` (In-Sample `<= {adap['is_cutoff']}` | Out-of-Sample `>= 2025-04-01`)",
        f"- **Regime Models Compared:** `5` strictly causal architectures (`RULE_5CORE_HYST`, `HMM_3_CAUSAL`, `KMEANS_4`, `VOL_STATE`, `CUSUM_CP`, plus unsmoothed `RULE_9FINE` ablation)",
        f"- **Switching Friction:** `{adap['switching_cost_bps']} bps` one-way turnover cost charged on every regime allocation change (`{adap['rule5_switch_events']}` transitions, `{adap['total_switching_cost_drag_pct']:.2f}%` cumulative drag)",
        "",
        "## 1. Out-of-Sample (`2025-04-01 -> 2026-11-19`) & Full-Window Regime Detector Stability Comparison",
        "",
        "| Regime Model | Full Transitions/Yr | Full Avg Duration | Full False Flips (<=2d) | OOS Transitions/Yr | OOS Avg Duration | OOS False Flip Rate |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for mname, d in mc.items():
        f_m = d["full"]
        o_m = d["oos_only"]
        lines.append(
            f"| `{mname}` | {f_m['transitions_per_year']:.2f} | {f_m['avg_regime_duration_days']:.1f}d | "
            f"{f_m['short_false_flips_le_2d']} ({f_m['false_flip_rate_pct']:.1f}%) | "
            f"**{o_m['transitions_per_year']:.2f}** | **{o_m['avg_regime_duration_days']:.1f}d** | "
            f"**{o_m['false_flip_rate_pct']:.1f}%** |"
        )

    lines += [
        "",
        "## 2. Causal Core Regime Distribution & Next-Day Nifty 50 Behavior (`RULE_5CORE_HYST` Full Window)",
        "",
        "| Core Regime | Days | % of Time | Next-1D Nifty Ann. Return | Next-1D Nifty Ann. Vol | Target Equity Exposure | Target Cash Buffer |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    r5_dist = mc["RULE_5CORE_HYST"]["full"]["regime_distribution"]
    for r_lbl, r_st in r5_dist.items():
        pol = REGIME_ALLOCATION_POLICY.get(r_lbl, {})
        cash_w = pol.get("CASH_RESERVE", 0.0) * 100.0
        eq_w = 100.0 - cash_w
        lines.append(
            f"| `{r_lbl}` | {r_st['days']} | {r_st['pct_of_time']:.1f}% | {r_st['nifty_next1d_ann_ret_pct']:+.2f}% | "
            f"{r_st['nifty_next1d_ann_vol_pct']:.2f}% | **{eq_w:.0f}%** | **{cash_w:.0f}%** |"
        )

    lines += [
        "",
        "## 3. Adaptive Regime-Allocated Meta-Portfolio vs Static Equal-Weight & Single-Family Baselines",
        "",
        "| Portfolio / Strategy | Full Net Return | Full CAGR | Full Max DD | Full Sharpe | Full Calmar | OOS Net Return | OOS Max DD | OOS Sharpe | OOS Excess vs Nifty |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for pid, pd in pcomp.items():
        fm = pd["full"]
        om = pd["out_of_sample"]
        lines.append(
            f"| `{pid}` | **{fm['total_return_pct']:+.2f}%** | {fm['cagr_pct']:+.2f}% | **{fm['max_drawdown_pct']:.2f}%** | "
            f"**{fm['sharpe_ratio']:+.2f}** | {fm['calmar_ratio']:+.2f} | **{om['total_return_pct']:+.2f}%** | "
            f"**{om['max_drawdown_pct']:.2f}%** | **{om['sharpe_ratio']:+.2f}** | **{om['excess_vs_nifty_pp']:+.2f} pp** |"
        )

    lines += [
        "",
        "## 4. Regime-Conditioned Strategy Allocation Matrix",
        "",
        "| Core Regime | `A_CS_MOMENTUM` | `B_VCP_BREAKOUT` | `C_TREND_PULLBACK` | `D_SECTOR_ROTATION` | `E_DEFENSIVE_YIELD` | `CASH_RESERVE (6.5%)` |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for r_lbl, pol in REGIME_ALLOCATION_POLICY.items():
        lines.append(
            f"| `{r_lbl}` | {pol['STRAT_A_CS_MOMENTUM']*100:.0f}% | {pol['STRAT_B_VCP_BREAKOUT']*100:.0f}% | "
            f"{pol['STRAT_C_TREND_PULLBACK']*100:.0f}% | {pol['STRAT_D_SECTOR_ROTATION']*100:.0f}% | "
            f"{pol['STRAT_E_DEFENSIVE_YIELD_LOWVOL']*100:.0f}% | **{pol['CASH_RESERVE']*100:.0f}%** |"
        )

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

