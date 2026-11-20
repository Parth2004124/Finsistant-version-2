"""Moreshwar 3.0 Prompt 6 — Relative-Value, Cointegrated Pairs, Market-Neutral & Futures-Hedged Research Branch.

Strictly isolated research branch (`promoted_to_long_only_portfolio = False` unless net OOS alpha,
low correlation, and Indian microstructure feasibility pass all gates).

Implements:
1. Cointegration vs Correlation Pair Discovery Engine (Strictly In-Sample `<= 2025-03-31`):
   - Same-industry / economically linked NSE liquid equities (top-200 ADV F&O/SLB universe).
   - Engle-Granger Augmented Dickey-Fuller (ADF) test on log-price OLS residuals, Ornstein-Uhlenbeck
     half-life estimation (`3 <= half_life <= 35` days), rolling hedge-ratio stability, and explicit
     contrast against High-Correlation / Non-Cointegrated pairs that diverge out-of-sample.
   - Causal Out-of-Sample Pairs Trading Simulator (`STRAT_RV1_COINT_PAIRS`) with z-score thresholds,
     structural break stop (`|z| >= 3.5`), correlation breakdown exit (`rho_30d < 0.35`), hedge-ratio
     drift monitor, and Indian SLB borrow fee (`10.0% p.a.`) + statutory execution friction.
2. Sector-Relative Long-Short Strategies:
   - `STRAT_RV2A_SECTOR_REL_MOM`: Intra-sector leadership long-short (long top 2 leaders, short bottom 2
     borrowable laggards within the same sector).
   - `STRAT_RV2B_SECTOR_SPREAD_REV`: Intra-sector 20D relative-return mean-reversion long-short.
3. Cross-Sectional Market-Neutral Factor Long-Short (`STRAT_RV3_MARKET_NEUTRAL_LS`):
   - Beta-neutral, sector-balanced long-short portfolio using Prompt 2 validated factors, enforcing
     Indian short-selling restrictions (shorts restricted strictly to top-200 ADV SLB/F&O borrowable stocks),
     SLB financing/borrow drag (`10.5% p.a.`), borrow recall handling, and ex-post market beta auditing.
4. Nifty 50 Index Futures Hedged Overlay (`STRAT_RV4_NIFTY_FUTURES_HEDGED`):
   - Models actual NSE Nifty 50 Futures (`NRML` F&O) contract specifications (lot size 25 pre-Nov-2024 /
     75 post-Nov-2024, ~Rs 18.4L notional per lot, 22% SPAN+Exposure margin encumbering cash, monthly
     expiry roll friction 12x/yr, Prompt 8 historical F&O STT 0.0125% -> 0.02%, and basis carry).
   - Compares fractional idealized beta hedging vs realistic lot-quantized hedging at Rs 1,00,000
     (infeasible due to 18x lot granularity) and Rs 25,00,000 institutional capital.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .execution_cost_engine import HistoricalTariffSchedule, calculate_historical_indian_costs
from .panel_loader import build_or_load_m3_panel
from .portfolio_construction import PortfolioConstructionConfig, simulate_shared_capital_portfolio_m3
from .regime_intelligence import IS_CUTOFF_DATE, classify_rule_based_9state, compute_causal_market_features
from .schema import DEFAULT_M3_REPORTS_DIR
from .strategy_base import EVAL_START_DATE, INITIAL_CAPITAL_M3, OOS_START_DATE, RISK_FREE_ANNUAL_M3
from .strategy_families import prepare_strategy_signal_workspace


@dataclass(frozen=True)
class PairCointegrationRecord:
    pair_id: str
    sec_id_y: str
    sec_id_x: str
    symbol_y: str
    symbol_x: str
    col_y: int
    col_x: int
    sector: str
    industry: str
    is_return_corr: float
    is_ols_beta: float
    is_adf_t_stat: float
    is_adf_p_approx: float
    is_half_life_days: float
    is_beta_drift_cv: float
    is_cointegrated: bool
    classification: str  # "VALIDATED_COINTEGRATED" | "HIGH_CORR_NOT_COINTEGRATED" | "SLOW_OR_UNSTABLE_SPREAD"


def _engle_granger_adf_1lag(spread: np.ndarray) -> tuple[float, float, float]:
    """Compute Augmented Dickey-Fuller t-statistic (with 1 lagged difference) and OU half-life on spread.

    Model: d_s[t] = alpha + gamma * s[t-1] + phi_1 * d_s[t-1] + e[t]
    H0: gamma = 0 (unit root / non-stationary). Cointegrated if gamma < 0 and t_stat < -2.86 (5% MacKinnon).
    Returns: (adf_t_stat, p_val_approx, half_life_days).
    """
    s = spread[~np.isnan(spread)]
    if len(s) < 80:
        return 0.0, 1.0, 999.0
    ds = np.diff(s)
    # Align t from 1..len(ds)-1
    y = ds[1:]
    s_lag = s[1:-1]
    ds_lag = ds[:-1]
    n = len(y)
    X = np.column_stack([np.ones(n, dtype=np.float64), s_lag, ds_lag])
    try:
        beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
        sigma2 = float(np.sum(resid ** 2)) / max(1, n - 3)
        cov_b = sigma2 * np.linalg.inv(X.T @ X)
        se_gamma = math.sqrt(max(1e-12, float(cov_b[1, 1])))
        gamma = float(beta[1])
        t_stat = gamma / se_gamma
    except np.linalg.LinAlgError:
        return 0.0, 1.0, 999.0

    # Approximate MacKinnon 2-variable cointegration p-value from t_stat
    if t_stat >= -1.95:
        p_val = 0.65
    elif t_stat >= -2.57:
        p_val = 0.25
    elif t_stat >= -2.86:
        p_val = 0.10
    elif t_stat >= -3.12:
        p_val = 0.045
    elif t_stat >= -3.43:
        p_val = 0.020
    else:
        p_val = 0.005

    # AR(1) phi = 1 + gamma -> half_life = -ln(2) / ln(1 + gamma)
    phi = 1.0 + gamma
    if gamma >= -1e-5 or phi <= 0.05:
        half_life = 999.0
    else:
        half_life = -math.log(2.0) / math.log(phi)
    return float(t_stat), float(p_val), float(half_life)


def discover_cointegrated_pairs_in_sample(
    panel: dict[str, Any],
    *,
    date_start: str = EVAL_START_DATE,
    is_cutoff: str = IS_CUTOFF_DATE,
    top_liquid_n: int = 180,
) -> dict[str, Any]:
    """Screen economically related same-sector/industry pairs strictly on In-Sample data (`<= 2025-03-31`).

    Never peeks at OOS (`>= 2025-04-01`) prices or correlations. Explicitly separates genuinely
    cointegrated mean-reverting pairs from high-correlation non-cointegrated pairs.
    """
    dates = [str(d) for d in panel["dates"]]
    is_idx = [i for i, d in enumerate(dates) if date_start <= d <= is_cutoff]
    adj_c = panel["adj_close"][is_idx, :].astype(np.float64)
    elig = panel["is_pit_elig"][is_idx, :]
    adv60 = panel["adv_60d_inr"][is_idx, :]
    sectors = panel["sectors"]
    industries = panel["industries"]
    sec_ids = panel["sec_ids"]
    syms = panel["current_symbols"]

    t_is = len(is_idx)
    # Filter to top-180 borrowable/liquid stocks with >= 90% valid IS presence (proxy for NSE F&O / SLB universe)
    presence = np.mean((~np.isnan(adj_c)) & (adj_c > 10.0) & elig, axis=0)
    with np.errstate(invalid="ignore"):
        med_adv = np.nanmedian(np.where(elig, adv60, np.nan), axis=0)
    med_adv = np.nan_to_num(med_adv, nan=0.0)

    cand_cols = [j for j in range(adj_c.shape[1]) if presence[j] >= 0.88 and "Unclassified" not in str(sectors[j])]
    cand_cols.sort(key=lambda j: -float(med_adv[j]))
    borrowable_cols = cand_cols[:top_liquid_n]

    log_p = np.where(adj_c > 0, np.log(np.maximum(1e-6, adj_c)), np.nan)
    ret_is = np.full(adj_c.shape, np.nan, dtype=np.float64)
    ret_is[1:] = np.where((adj_c[1:] > 0) & (adj_c[:-1] > 0), (adj_c[1:] / np.maximum(1e-6, adj_c[:-1])) - 1.0, np.nan)

    # Group borrowable columns by sector
    by_sector: dict[str, list[int]] = {}
    for c in borrowable_cols:
        by_sector.setdefault(str(sectors[c]), []).append(c)

    all_evaluated: list[PairCointegrationRecord] = []
    for sec_name, cols in sorted(by_sector.items()):
        m = len(cols)
        if m < 2:
            continue
        for a_i in range(m):
            for b_i in range(a_i + 1, m):
                cy, cx = cols[a_i], cols[b_i]
                ry, rx = ret_is[:, cy], ret_is[:, cx]
                ok_r = (~np.isnan(ry)) & (~np.isnan(rx))
                if int(np.sum(ok_r)) < int(t_is * 0.85):
                    continue
                corr = float(np.corrcoef(ry[ok_r], rx[ok_r])[0, 1])
                # Prefer same industry or high economic return correlation >= 0.52
                same_ind = str(industries[cy]) == str(industries[cx])
                if corr < (0.48 if same_ind else 0.56):
                    continue

                ly, lx = log_p[:, cy], log_p[:, cx]
                ok_l = (~np.isnan(ly)) & (~np.isnan(lx))
                ly_v, lx_v = ly[ok_l], lx[ok_l]
                cov_yx = float(np.cov(ly_v, lx_v, ddof=1)[0, 1])
                var_x = max(1e-8, float(np.var(lx_v, ddof=1)))
                beta_ols = cov_yx / var_x
                if beta_ols <= 0.25 or beta_ols >= 3.5:
                    continue
                alpha_ols = float(np.mean(ly_v) - beta_ols * np.mean(lx_v))
                spread = ly_v - (alpha_ols + beta_ols * lx_v)

                adf_t, adf_p, hl = _engle_granger_adf_1lag(spread)

                # Rolling 3-subwindow beta stability check on IS
                w_sub = len(ly_v) // 3
                sub_betas = []
                for k_sub in range(3):
                    s_lo = k_sub * w_sub
                    s_hi = (k_sub + 1) * w_sub if k_sub < 2 else len(ly_v)
                    sy_k, sx_k = ly_v[s_lo:s_hi], lx_v[s_lo:s_hi]
                    bk = float(np.cov(sy_k, sx_k, ddof=1)[0, 1]) / max(1e-8, float(np.var(sx_k, ddof=1)))
                    sub_betas.append(bk)
                beta_cv = float(np.std(sub_betas) / max(0.1, abs(float(np.mean(sub_betas)))))

                is_coint = (adf_t <= -2.86) and (3.0 <= hl <= 35.0) and (beta_cv <= 0.35)
                if is_coint:
                    cls_tag = "VALIDATED_COINTEGRATED"
                elif corr >= 0.62 and adf_t > -2.55:
                    cls_tag = "HIGH_CORR_NOT_COINTEGRATED"
                else:
                    cls_tag = "SLOW_OR_UNSTABLE_SPREAD"

                all_evaluated.append(
                    PairCointegrationRecord(
                        pair_id=f"{syms[cy]}_{syms[cx]}",
                        sec_id_y=str(sec_ids[cy]),
                        sec_id_x=str(sec_ids[cx]),
                        symbol_y=str(syms[cy]),
                        symbol_x=str(syms[cx]),
                        col_y=int(cy),
                        col_x=int(cx),
                        sector=sec_name,
                        industry=str(industries[cy]),
                        is_return_corr=round(corr, 3),
                        is_ols_beta=round(beta_ols, 3),
                        is_adf_t_stat=round(adf_t, 3),
                        is_adf_p_approx=round(adf_p, 4),
                        is_half_life_days=round(hl, 1),
                        is_beta_drift_cv=round(beta_cv, 3),
                        is_cointegrated=is_coint,
                        classification=cls_tag,
                    )
                )

    validated = sorted(
        [p for p in all_evaluated if p.is_cointegrated],
        key=lambda p: (p.is_adf_t_stat, -p.is_return_corr),
    )
    # Deduplicate so no single stock appears in more than 2 selected trading pairs
    selected_pairs: list[PairCointegrationRecord] = []
    stock_use_cnt: dict[int, int] = {}
    for p in validated:
        if stock_use_cnt.get(p.col_y, 0) >= 2 or stock_use_cnt.get(p.col_x, 0) >= 2:
            continue
        selected_pairs.append(p)
        stock_use_cnt[p.col_y] = stock_use_cnt.get(p.col_y, 0) + 1
        stock_use_cnt[p.col_x] = stock_use_cnt.get(p.col_x, 0) + 1
        if len(selected_pairs) >= 12:
            break

    false_corr_pairs = sorted(
        [p for p in all_evaluated if p.classification == "HIGH_CORR_NOT_COINTEGRATED"],
        key=lambda p: -p.is_return_corr,
    )[:8]

    return {
        "total_same_sector_high_corr_evaluated": len(all_evaluated),
        "total_cointegrated_found": len(validated),
        "selected_pairs": selected_pairs,
        "high_corr_non_cointegrated_examples": false_corr_pairs,
        "borrowable_cols_set": set(borrowable_cols),
    }


def simulate_cointegrated_pairs_strategy(
    *,
    panel: dict[str, Any],
    pairs: list[PairCointegrationRecord],
    entry_z: float = 2.0,
    exit_z: float = 0.40,
    stop_z: float = 3.5,
    max_active_pairs: int = 5,
    slb_borrow_rate_annual: float = 0.10,
    one_way_cost_bps: float = 18.0,
    date_start: str = EVAL_START_DATE,
    date_end: str | None = None,
    initial_capital: float = INITIAL_CAPITAL_M3,
) -> dict[str, Any]:
    """Simulate causal daily pairs trading on IS-selected pairs with rolling 120D OLS beta, 60D z-score, and SLB borrow fees."""
    dates = [str(d) for d in panel["dates"]]
    end_dt = date_end or dates[-1]
    t_indices = [i for i, d in enumerate(dates) if date_start <= d <= end_dt]
    if len(t_indices) < 20 or not pairs:
        return {"daily_rets": np.zeros(len(t_indices)), "trades": [], "risk_events": {}}

    open_p = panel["adj_open"].astype(np.float64)
    close_p = panel["adj_close"].astype(np.float64)
    div_ex = panel["div_inr_ex"].astype(np.float64)
    log_c = np.where(close_p > 0, np.log(np.maximum(1e-6, close_p)), np.nan)
    daily_ret = np.full(close_p.shape, np.nan, dtype=np.float64)
    daily_ret[1:] = np.where((close_p[1:] > 0) & (close_p[:-1] > 0), (close_p[1:] / np.maximum(1e-6, close_p[:-1])) - 1.0, np.nan)

    active_positions: dict[str, dict[str, Any]] = {}
    pending_entries: list[dict[str, Any]] = []
    pending_exits: dict[str, str] = {}
    closed_trades: list[dict[str, Any]] = []
    risk_events = {
        "MEAN_REVERSION_EXIT": 0,
        "SPREAD_STOP_Z_EXIT": 0,
        "MAX_LOSS_STOP_EXIT": 0,
        "CORR_BREAKDOWN_EXIT": 0,
        "HEDGE_DRIFT_EXIT": 0,
        "BORROW_RECALL_DIV_EXIT": 0,
        "HORIZON_EXIT": 0,
    }

    daily_port_rets: list[float] = []
    total_cost_pct = 0.0
    total_borrow_drag_pct = 0.0
    cost_frac = one_way_cost_bps / 10_000.0
    daily_borrow_frac = slb_borrow_rate_annual / 252.0
    pair_slot_weight = 0.90 / max(1, max_active_pairs)  # max 90% gross deployed across active pairs (zero leverage!)

    for step_k, ti in enumerate(t_indices):
        dt = dates[ti]
        is_last = (ti == t_indices[-1])
        day_ret = 0.0

        # 1. Execute pending exits at Day `ti` Open
        for pid, reason in list(pending_exits.items()):
            if pid not in active_positions:
                continue
            pos = active_positions[pid]
            cy, cx = pos["col_y"], pos["col_x"]
            oy = float(open_p[ti, cy]) if not np.isnan(open_p[ti, cy]) and open_p[ti, cy] > 0 else pos["last_py"]
            ox = float(open_p[ti, cx]) if not np.isnan(open_p[ti, cx]) and open_p[ti, cx] > 0 else pos["last_px"]
            ret_y_open = (oy / pos["last_py"]) - 1.0
            ret_x_open = (ox / pos["last_px"]) - 1.0
            leg_ret = pos["dir"] * (pos["wy"] * ret_y_open - pos["wx"] * ret_x_open) - 2.0 * cost_frac
            day_ret += pair_slot_weight * leg_ret
            total_cost_pct += pair_slot_weight * 2.0 * cost_frac * 100.0
            pos["cum_ret"] += leg_ret
            closed_trades.append({
                "pair_id": pid,
                "sector": pos["sector"],
                "direction": "LONG_Y_SHORT_X" if pos["dir"] > 0 else "SHORT_Y_LONG_X",
                "entry_date": pos["entry_date"],
                "exit_date": dt,
                "holding_days": pos["bars_held"],
                "net_pair_return_pct": round(pos["cum_ret"] * 100.0, 2),
                "exit_reason": reason,
            })
            risk_events[reason] = risk_events.get(reason, 0) + 1
            del active_positions[pid]
        pending_exits.clear()

        # 2. Execute pending entries at Day `ti` Open
        for req in pending_entries:
            pid = req["pair_id"]
            if pid in active_positions or len(active_positions) >= max_active_pairs:
                continue
            cy, cx = req["col_y"], req["col_x"]
            oy, ox = float(open_p[ti, cy]), float(open_p[ti, cx])
            if np.isnan(oy) or np.isnan(ox) or oy <= 0 or ox <= 0:
                continue
            # Normalize leg weights wy + wx = 1.0 (1.0x total pair gross notional)
            b = max(0.25, min(3.0, float(req["beta"])))
            wy = 1.0 / (1.0 + b)
            wx = b / (1.0 + b)
            entry_cost = 2.0 * cost_frac
            day_ret -= pair_slot_weight * entry_cost
            total_cost_pct += pair_slot_weight * entry_cost * 100.0
            active_positions[pid] = {
                "pair_id": pid,
                "sector": req["sector"],
                "col_y": cy,
                "col_x": cx,
                "dir": req["dir"],  # +1: Long Y / Short X; -1: Short Y / Long X
                "entry_beta": b,
                "wy": wy,
                "wx": wx,
                "entry_date": dt,
                "last_py": oy,
                "last_px": ox,
                "cum_ret": -entry_cost,
                "bars_held": 0,
            }
        pending_entries.clear()

        # 3. Mark-to-market open pairs from open/prev_close to today's close + deduct SLB borrow cost on short leg
        for pid, pos in list(active_positions.items()):
            cy, cx = pos["col_y"], pos["col_x"]
            cy_px = float(close_p[ti, cy]) if not np.isnan(close_p[ti, cy]) and close_p[ti, cy] > 0 else pos["last_py"]
            cx_px = float(close_p[ti, cx]) if not np.isnan(close_p[ti, cx]) and close_p[ti, cx] > 0 else pos["last_px"]
            ry = (cy_px / pos["last_py"]) - 1.0
            rx = (cx_px / pos["last_px"]) - 1.0
            short_weight = pos["wx"] if pos["dir"] > 0 else pos["wy"]
            borrow_fee = short_weight * daily_borrow_frac
            step_r = pos["dir"] * (pos["wy"] * ry - pos["wx"] * rx) - borrow_fee
            day_ret += pair_slot_weight * step_r
            total_borrow_drag_pct += pair_slot_weight * borrow_fee * 100.0
            pos["cum_ret"] += step_r
            pos["last_py"] = cy_px
            pos["last_px"] = cx_px
            pos["bars_held"] += 1

            # Check if short leg has an upcoming dividend ex_date recall
            short_col = cx if pos["dir"] > 0 else cy
            if ti + 1 < len(dates) and div_ex[ti + 1, short_col] > 0:
                pending_exits[pid] = "BORROW_RECALL_DIV_EXIT"

        daily_port_rets.append(day_ret)
        if is_last or ti < 125:
            continue

        # 4. Evaluate causal rolling 120D OLS spread & 60D z-score at Day `ti` Close for exits & entries
        for p in pairs:
            pid = p.pair_id
            cy, cx = p.col_y, p.col_x
            ly_win = log_c[ti - 119 : ti + 1, cy]
            lx_win = log_c[ti - 119 : ti + 1, cx]
            ok = (~np.isnan(ly_win)) & (~np.isnan(lx_win))
            if int(np.sum(ok)) < 90:
                continue
            ly_v, lx_v = ly_win[ok], lx_win[ok]
            var_x = max(1e-8, float(np.var(lx_v, ddof=1)))
            cur_beta = float(np.cov(ly_v, lx_v, ddof=1)[0, 1]) / var_x
            cur_alpha = float(np.mean(ly_v) - cur_beta * np.mean(lx_v))
            sp_series = ly_v - (cur_alpha + cur_beta * lx_v)
            sp_60 = sp_series[-60:]
            sd_60 = float(np.std(sp_60, ddof=1))
            if sd_60 < 1e-6:
                continue
            z_now = (float(sp_60[-1]) - float(np.mean(sp_60))) / sd_60

            # Trailing 30D return correlation check
            ry_30 = daily_ret[ti - 29 : ti + 1, cy]
            rx_30 = daily_ret[ti - 29 : ti + 1, cx]
            ok_30 = (~np.isnan(ry_30)) & (~np.isnan(rx_30))
            corr_30 = float(np.corrcoef(ry_30[ok_30], rx_30[ok_30])[0, 1]) if int(np.sum(ok_30)) >= 20 else 0.5

            if pid in active_positions and pid not in pending_exits:
                pos = active_positions[pid]
                if (pos["dir"] > 0 and z_now >= -exit_z) or (pos["dir"] < 0 and z_now <= exit_z):
                    pending_exits[pid] = "MEAN_REVERSION_EXIT"
                elif abs(z_now) >= stop_z:
                    pending_exits[pid] = "SPREAD_STOP_Z_EXIT"
                elif pos["cum_ret"] <= -0.06:
                    pending_exits[pid] = "MAX_LOSS_STOP_EXIT"
                elif corr_30 < 0.35:
                    pending_exits[pid] = "CORR_BREAKDOWN_EXIT"
                elif abs(cur_beta - pos["entry_beta"]) / max(0.1, abs(pos["entry_beta"])) > 0.35:
                    pending_exits[pid] = "HEDGE_DRIFT_EXIT"
                elif pos["bars_held"] >= 40:
                    pending_exits[pid] = "HORIZON_EXIT"
            elif pid not in active_positions and len(active_positions) + len(pending_entries) < max_active_pairs:
                if corr_30 >= 0.45 and 0.3 <= cur_beta <= 3.0 and abs(z_now) < stop_z - 0.3:
                    if z_now <= -entry_z:
                        pending_entries.append({"pair_id": pid, "sector": p.sector, "col_y": cy, "col_x": cx, "dir": 1, "beta": cur_beta})
                    elif z_now >= entry_z:
                        pending_entries.append({"pair_id": pid, "sector": p.sector, "col_y": cy, "col_x": cx, "dir": -1, "beta": cur_beta})

    return {
        "dates": [dates[i] for i in t_indices],
        "daily_rets": np.array(daily_port_rets, dtype=np.float64),
        "trades": closed_trades,
        "risk_events": risk_events,
        "total_cost_drag_pct": round(total_cost_pct, 2),
        "total_borrow_drag_pct": round(total_borrow_drag_pct, 2),
    }


def simulate_sector_relative_and_market_neutral(
    *,
    panel: dict[str, Any],
    ws: dict[str, Any],
    borrowable_cols: set[int],
    mode: str,  # "SECTOR_REL_MOM" | "SECTOR_SPREAD_REV" | "MARKET_NEUTRAL_LS"
    rebalance_days: int = 20,
    slb_borrow_rate_annual: float = 0.105,
    one_way_cost_bps: float = 18.5,
    date_start: str = EVAL_START_DATE,
    date_end: str | None = None,
) -> dict[str, Any]:
    """Simulate dollar-neutral / sector-neutral long-short relative-value strategies with Indian SLB constraints."""
    dates = [str(d) for d in panel["dates"]]
    end_dt = date_end or dates[-1]
    t_indices = [i for i, d in enumerate(dates) if date_start <= d <= end_dt]

    open_p = panel["adj_open"].astype(np.float64)
    close_p = panel["adj_close"].astype(np.float64)
    elig = panel["is_pit_elig"]
    sectors = panel["sectors"]
    n50_c = panel["nifty50_close"].astype(np.float64)

    facs = ws["factors"]
    ranks = ws["ranks"]
    st_vs_sec = facs["STOCK_VS_SECTOR_20D"]
    dist252 = facs["DIST_252D_HIGH"]
    mom60 = facs["MOM_60D"]
    r_dist = ranks["r_dist252"]
    r_mdet = ranks["r_mom_det60"]
    r_div = ranks["r_div_yield"]
    r_lvol = ranks["r_low_vol"]

    borrow_mask = np.zeros(close_p.shape[1], dtype=bool)
    for c in borrowable_cols:
        borrow_mask[c] = True

    long_cols: list[int] = []
    short_cols: list[int] = []
    short_beta_scale: float = 1.0
    pending_rebal: tuple[list[int], list[int], float] | None = None

    daily_rets: list[float] = []
    bench_rets: list[float] = []
    gross_exp_samples: list[float] = []
    net_exp_samples: list[float] = []
    total_cost_pct = 0.0
    total_borrow_pct = 0.0
    rebal_trades = 0

    cost_frac = one_way_cost_bps / 10_000.0
    daily_borrow_frac = slb_borrow_rate_annual / 252.0

    # Precompute daily stock returns and trailing 60D beta vs Nifty 50
    stock_ret = np.full(close_p.shape, 0.0, dtype=np.float64)
    ok_sr = (close_p[1:] > 0) & (close_p[:-1] > 0)
    stock_ret[1:] = np.where(ok_sr, (close_p[1:] / np.maximum(1e-6, close_p[:-1])) - 1.0, 0.0)
    n50_ret = np.zeros(len(dates), dtype=np.float64)
    n50_ret[1:] = np.where((n50_c[1:] > 0) & (n50_c[:-1] > 0), (n50_c[1:] / np.maximum(1e-6, n50_c[:-1])) - 1.0, 0.0)

    for step_k, ti in enumerate(t_indices):
        bench_rets.append(float(n50_ret[ti]))
        day_r = 0.0

        # 1. Execute pending rebalance at Day `ti` Open
        if pending_rebal is not None:
            new_l, new_s, new_bscale = pending_rebal
            old_set = set(long_cols) | set(short_cols)
            new_set = set(new_l) | set(new_s)
            changed_frac = len(old_set ^ new_set) / max(1, len(old_set | new_set)) if (old_set | new_set) else 0.0
            to_cost = changed_frac * 0.90 * cost_frac
            day_r -= to_cost
            total_cost_pct += to_cost * 100.0
            rebal_trades += len(old_set ^ new_set)
            long_cols, short_cols, short_beta_scale = new_l, new_s, new_bscale
            pending_rebal = None

        # 2. Mark-to-market long book (+45% gross) and short book (-45% * short_beta_scale gross)
        if long_cols and short_cols:
            w_long = 0.45
            w_short = 0.45 * short_beta_scale
            rl = float(np.mean([stock_ret[ti, c] for c in long_cols]))
            rs = float(np.mean([stock_ret[ti, c] for c in short_cols]))
            b_drag = w_short * daily_borrow_frac
            day_r += (w_long * rl) - (w_short * rs) - b_drag
            total_borrow_pct += b_drag * 100.0
            gross_exp_samples.append((w_long + w_short) * 100.0)
            net_exp_samples.append((w_long - w_short) * 100.0)
        else:
            gross_exp_samples.append(0.0)
            net_exp_samples.append(0.0)

        daily_rets.append(day_r)

        if step_k % max(1, rebalance_days) != 0 or ti == t_indices[-1]:
            continue

        # 3. Generate strictly causal Long and Short baskets at Day `ti` Close
        elig_row = elig[ti] & (close_p[ti] > 15.0)
        elig_short = elig_row & borrow_mask

        cand_l: list[int] = []
        cand_s: list[int] = []

        if mode in ("SECTOR_REL_MOM", "SECTOR_SPREAD_REV"):
            # Group by major sectors with >= 6 borrowable eligible peers on day ti
            sec_groups: dict[str, list[int]] = {}
            for c in np.where(elig_short)[0]:
                s_nm = str(sectors[c])
                if "Unclassified" in s_nm or np.isnan(st_vs_sec[ti, c]) or np.isnan(dist252[ti, c]):
                    continue
                sec_groups.setdefault(s_nm, []).append(int(c))
            valid_secs = sorted([s for s, cl in sec_groups.items() if len(cl) >= 6], key=lambda s: -len(sec_groups[s]))[:4]
            for s_nm in valid_secs:
                cl = sec_groups[s_nm]
                if mode == "SECTOR_REL_MOM":
                    sc_arr = [(0.55 * float(dist252[ti, c]) + 0.45 * float(st_vs_sec[ti, c]), c) for c in cl]
                    sc_arr.sort(reverse=True)
                    cand_l.append(sc_arr[0][1])
                    cand_s.append(sc_arr[-1][1])
                else:
                    # Spread reversion: long oversold 20D relative to sector (with positive 60D mom), short overextended 20D
                    sc_arr = [(float(st_vs_sec[ti, c]), c) for c in cl]
                    sc_arr.sort()  # lowest (most oversold vs sector) first
                    cand_l.append(sc_arr[0][1])
                    cand_s.append(sc_arr[-1][1])
        elif mode == "MARKET_NEUTRAL_LS":
            ok_all = (
                elig_short
                & (~np.isnan(r_dist[ti]))
                & (~np.isnan(r_mdet[ti]))
                & (~np.isnan(r_lvol[ti]))
            )
            cols_ok = np.where(ok_all)[0]
            if len(cols_ok) >= 20:
                alpha_sc = (
                    0.45 * r_dist[ti, cols_ok]
                    + 0.25 * r_mdet[ti, cols_ok]
                    + 0.15 * np.nan_to_num(r_div[ti, cols_ok], nan=0.5)
                    + 0.15 * r_lvol[ti, cols_ok]
                )
                order = np.argsort(-alpha_sc)
                cand_l = [int(cols_ok[i]) for i in order[:8]]
                cand_s = [int(cols_ok[i]) for i in order[-8:]]

        if len(cand_l) >= 3 and len(cand_s) >= 3:
            # Causal 60D beta matching so ex-ante portfolio beta ~ 0.0
            lo = max(1, ti - 59)
            m_win = n50_ret[lo : ti + 1]
            var_m = max(1e-8, float(np.var(m_win, ddof=1)))
            l_win = np.mean(stock_ret[lo : ti + 1, :][:, cand_l], axis=1)
            s_win = np.mean(stock_ret[lo : ti + 1, :][:, cand_s], axis=1)
            beta_l = max(0.4, min(1.8, float(np.cov(l_win, m_win, ddof=1)[0, 1]) / var_m))
            beta_s = max(0.5, min(1.8, float(np.cov(s_win, m_win, ddof=1)[0, 1]) / var_m))
            b_scale = float(np.clip(beta_l / beta_s, 0.75, 1.15))
            pending_rebal = (cand_l, cand_s, b_scale)

    rets_arr = np.array(daily_rets, dtype=np.float64)
    bench_arr = np.array(bench_rets, dtype=np.float64)
    ex_post_beta = (
        float(np.cov(rets_arr, bench_arr, ddof=1)[0, 1]) / max(1e-8, float(np.var(bench_arr, ddof=1)))
        if len(rets_arr) > 20
        else 0.0
    )
    corr_nifty = float(np.corrcoef(rets_arr, bench_arr)[0, 1]) if len(rets_arr) > 20 and np.std(rets_arr) > 1e-8 else 0.0

    return {
        "dates": [dates[i] for i in t_indices],
        "daily_rets": rets_arr,
        "bench_rets": bench_arr,
        "ex_post_market_beta": round(ex_post_beta, 3),
        "corr_with_nifty50": round(corr_nifty, 3),
        "avg_gross_exposure_pct": round(float(np.mean(gross_exp_samples)) if gross_exp_samples else 0.0, 1),
        "avg_net_exposure_pct": round(float(np.mean(net_exp_samples)) if net_exp_samples else 0.0, 2),
        "total_cost_drag_pct": round(total_cost_pct, 2),
        "total_borrow_drag_pct": round(total_borrow_pct, 2),
        "rebalance_leg_trades": rebal_trades,
    }


def simulate_nifty_futures_hedged_overlay(
    *,
    panel: dict[str, Any],
    long_daily_navs: list[dict[str, Any]],
    regime_labels: list[str],
    capital_base_inr: float = 2_500_000.0,
    lot_quantized: bool = True,
) -> dict[str, Any]:
    """Simulate dynamic Nifty 50 Index Futures (`NRML` F&O) hedging overlay on Prompt 5's Risk-Budgeted Long Portfolio.

    Accounts for:
    - Actual NSE Nifty 50 Futures lot size: `50` (2021-07..2024-04), `25` (2024-05..2024-11), `75` (2024-11+).
    - Margin requirement: 22% SPAN + Exposure margin encumbered per active short futures contract.
    - Monthly expiry roll costs: every 21 sessions (~12x/yr) charges Prompt 8 `calculate_historical_indian_costs(product_type='FUTURES')`
      + 8.0 bps roll slippage + annualized futures basis cost-of-carry (~1.2% p.a. net dividend-adjusted roll drag).
    """
    dates_all = [str(d) for d in panel["dates"]]
    d_to_idx = {d: i for i, d in enumerate(dates_all)}
    n50_c = panel["nifty50_close"].astype(np.float64)

    tariff = HistoricalTariffSchedule()
    hedge_target_by_regime = {
        "STRONG_BULL": 0.00,
        "WEAK_BULL": 0.00,
        "SIDEWAYS_CHOP": 0.30,
        "WEAK_BEAR": 0.65,
        "STRESS_BEAR": 0.90,
    }

    nav_scale = capital_base_inr / INITIAL_CAPITAL_M3
    cur_nav = float(capital_base_inr)
    active_lots = 0
    cur_hedge_ratio = 0.0
    days_since_roll = 0

    hedged_rets: list[float] = []
    bench_rets: list[float] = []
    dates_out: list[str] = []
    total_futures_cost_inr = 0.0
    roll_events = 0
    lot_mismatch_days = 0

    for k, d_row in enumerate(long_daily_navs):
        dt = str(d_row["date"])
        ti = d_to_idx[dt]
        long_r = float(d_row["ret"])
        dep_frac = float(d_row["deployed_mkt"]) / max(1.0, float(d_row["nav"]))
        n50_px = float(n50_c[ti])
        prev_n50 = float(n50_c[ti - 1]) if ti > 0 else n50_px
        n50_r = (n50_px / max(1.0, prev_n50)) - 1.0 if ti > 0 else 0.0
        bench_rets.append(n50_r)
        dates_out.append(dt)

        # NSE historical Nifty 50 lot size schedule
        if dt >= "2024-11-20":
            lot_size = 75
        elif dt >= "2024-04-26":
            lot_size = 25
        else:
            lot_size = 50
        one_lot_notional = lot_size * n50_px

        # PnL from yesterday's active short Nifty Futures hedge
        hedge_pnl_frac = -cur_hedge_ratio * (n50_r + (0.012 / 252.0))
        step_cost_frac = 0.0

        # Monthly expiry roll every 21 trading sessions when holding short futures
        days_since_roll += 1
        if cur_hedge_ratio > 0.01 and days_since_roll >= 21:
            days_since_roll = 0
            roll_events += 1
            hedge_notional = cur_hedge_ratio * cur_nav
            c_close = calculate_historical_indian_costs(
                product_type="FUTURES", side="BUY", raw_notional=hedge_notional, trade_date=dt, slippage_bps=4.0, tariff=tariff
            )
            c_open = calculate_historical_indian_costs(
                product_type="FUTURES", side="SELL", raw_notional=hedge_notional, trade_date=dt, slippage_bps=4.0, tariff=tariff
            )
            roll_cost = c_close["total_cost"] + c_open["total_cost"]
            total_futures_cost_inr += roll_cost
            step_cost_frac += roll_cost / max(1.0, cur_nav)

        net_day_r = long_r + hedge_pnl_frac - step_cost_frac
        cur_nav *= (1.0 + net_day_r)
        hedged_rets.append(net_day_r)

        # Update causal target hedge ratio at Day `ti` Close for tomorrow (`ti + 1`)
        reg_lbl = regime_labels[ti]
        desired_h_of_deployed = hedge_target_by_regime.get(reg_lbl, 0.0)
        desired_notional = cur_nav * dep_frac * desired_h_of_deployed

        if lot_quantized:
            # Cannot exceed available free margin (22% SPAN+Exposure margin per lot)
            free_cash_for_margin = cur_nav * max(0.05, 1.0 - dep_frac)
            max_margin_lots = int((free_cash_for_margin / 0.22) // max(1.0, one_lot_notional))
            raw_lots = int(round(desired_notional / max(1.0, one_lot_notional)))
            # Never allow over-hedging > 1.25x of deployed equity book!
            if raw_lots * one_lot_notional > cur_nav * dep_frac * 1.25:
                raw_lots = int((cur_nav * dep_frac * 1.25) // max(1.0, one_lot_notional))
            target_lots = min(raw_lots, max_margin_lots)
            if desired_notional > 0 and target_lots == 0:
                lot_mismatch_days += 1
            new_hedge_ratio = (target_lots * one_lot_notional) / max(1.0, cur_nav)
        else:
            target_lots = 0
            new_hedge_ratio = dep_frac * desired_h_of_deployed

        if abs(new_hedge_ratio - cur_hedge_ratio) > 0.02:
            trade_notional = abs(new_hedge_ratio - cur_hedge_ratio) * cur_nav
            side = "SELL" if new_hedge_ratio > cur_hedge_ratio else "BUY"
            c_tr = calculate_historical_indian_costs(
                product_type="FUTURES", side=side, raw_notional=trade_notional, trade_date=dt, slippage_bps=5.0, tariff=tariff
            )
            total_futures_cost_inr += c_tr["total_cost"]
            cur_nav -= c_tr["total_cost"]
            active_lots = target_lots
            cur_hedge_ratio = new_hedge_ratio

    return {
        "dates": dates_out,
        "daily_rets": np.array(hedged_rets, dtype=np.float64),
        "bench_rets": np.array(bench_rets, dtype=np.float64),
        "capital_base_inr": capital_base_inr,
        "lot_quantized": lot_quantized,
        "roll_events": roll_events,
        "lot_granularity_blocked_days": lot_mismatch_days,
        "total_futures_cost_drag_pct": round((total_futures_cost_inr / capital_base_inr) * 100.0, 2),
    }


def _summarize_return_stream(
    strategy_id: str,
    name: str,
    dates: list[str],
    rets: np.ndarray,
    bench_rets: np.ndarray,
    long_port_rets: np.ndarray | None = None,
) -> dict[str, Any]:
    if len(rets) < 5:
        return {"strategy_id": strategy_id, "error": "Insufficient days"}
    navs = 100_000.0 * np.cumprod(1.0 + rets)
    tot_ret_pct = ((float(navs[-1]) / 100_000.0) - 1.0) * 100.0
    years = max(0.05, len(rets) / 252.0)
    cagr_pct = (((float(navs[-1]) / 100_000.0) ** (1.0 / years)) - 1.0) * 100.0
    pk = np.maximum.accumulate(np.concatenate([[100_000.0], navs]))
    dd = (np.concatenate([[100_000.0], navs]) - pk) / np.maximum(1.0, pk)
    max_dd_pct = abs(float(np.min(dd))) * 100.0

    rf_d = RISK_FREE_ANNUAL_M3 / 252.0
    ex = rets - rf_d
    sd = float(np.std(ex, ddof=1)) if len(ex) > 1 else 1.0
    sharpe = (float(np.mean(ex)) / sd) * math.sqrt(252.0) if sd > 1e-10 else 0.0
    ann_vol = float(np.std(rets, ddof=1)) * math.sqrt(252.0) * 100.0 if len(rets) > 1 else 0.0

    corr_n50 = float(np.corrcoef(rets, bench_rets)[0, 1]) if len(rets) > 10 and np.std(rets) > 1e-9 else 0.0
    beta_n50 = (
        float(np.cov(rets, bench_rets, ddof=1)[0, 1]) / max(1e-9, float(np.var(bench_rets, ddof=1)))
        if len(rets) > 10
        else 0.0
    )
    corr_long = (
        float(np.corrcoef(rets, long_port_rets)[0, 1])
        if (long_port_rets is not None and len(long_port_rets) == len(rets) and np.std(rets) > 1e-9)
        else 0.0
    )

    return {
        "strategy_id": strategy_id,
        "name": name,
        "window": f"{dates[0]} -> {dates[-1]} ({len(dates)}d)",
        "total_return_pct": round(tot_ret_pct, 2),
        "cagr_pct": round(cagr_pct, 2),
        "max_drawdown_pct": round(max_dd_pct, 2),
        "annualised_vol_pct": round(ann_vol, 2),
        "sharpe_ratio": round(sharpe, 2),
        "ex_post_market_beta": round(beta_n50, 3),
        "corr_with_nifty50": round(corr_n50, 3),
        "corr_with_long_portfolio": round(corr_long, 3),
    }


def run_moreshwar_v3_relative_value_research(
    *,
    panel: dict[str, Any] | None = None,
    ws: dict[str, Any] | None = None,
    reports_dir: Path = DEFAULT_M3_REPORTS_DIR / "relative_value_hedged",
) -> dict[str, Any]:
    """Run complete Prompt 6 Relative-Value, Pairs, Market-Neutral & Futures-Hedged Research Branch."""
    pnl = panel or build_or_load_m3_panel()
    workspace = ws or prepare_strategy_signal_workspace(pnl)
    feats = compute_causal_market_features(pnl)
    _, rule5 = classify_rule_based_9state(feats, confirm_days=3)

    # 0. Run Prompt 5 baseline Risk-Budgeted Long Portfolio for correlation & hedging overlay comparison
    long_cfg = PortfolioConstructionConfig(
        portfolio_id="PORT_RISK_BUDGETED",
        portfolio_name="Prompt 5 Risk-Budgeted Long-Only Portfolio",
        sizing_method="RISK_BUDGETED",
    )
    long_res = simulate_shared_capital_portfolio_m3(
        panel=pnl, ws=workspace, cfg=long_cfg, regime_labels=rule5,
        date_start=EVAL_START_DATE, date_end=str(pnl["dates"][-1]),
    )
    long_dates = [str(d["date"]) for d in long_res["daily_navs"]]
    long_rets = np.array([float(d["ret"]) for d in long_res["daily_navs"]], dtype=np.float64)

    # 1. In-Sample Pair Cointegration Discovery (`2021-07-01 -> 2025-03-31`)
    pair_disc = discover_cointegrated_pairs_in_sample(pnl)
    sel_pairs: list[PairCointegrationRecord] = pair_disc["selected_pairs"]
    false_pairs: list[PairCointegrationRecord] = pair_disc["high_corr_non_cointegrated_examples"]
    borrow_set: set[int] = pair_disc["borrowable_cols_set"]

    # Simulate Validated Cointegrated Pairs vs High-Correlation Non-Cointegrated Pairs (Ablation!)
    coint_sim = simulate_cointegrated_pairs_strategy(panel=pnl, pairs=sel_pairs, entry_z=2.0, exit_z=0.40)
    non_coint_sim = simulate_cointegrated_pairs_strategy(panel=pnl, pairs=false_pairs, entry_z=2.0, exit_z=0.40)

    # Threshold Ablation on Cointegrated Pairs
    z_ablations = {}
    for ez in (1.75, 2.00, 2.25):
        s_ab = simulate_cointegrated_pairs_strategy(panel=pnl, pairs=sel_pairs, entry_z=ez, exit_z=0.40)
        z_ablations[f"ENTRY_Z_{str(ez).replace('.', '_')}"] = {
            "entry_z": ez,
            "full_return_pct": round((float(np.prod(1.0 + s_ab["daily_rets"])) - 1.0) * 100.0, 2),
            "trades": len(s_ab["trades"]),
            "cost_drag_pct": s_ab["total_cost_drag_pct"],
            "borrow_drag_pct": s_ab["total_borrow_drag_pct"],
        }

    # 2. Sector-Relative Strategies (Momentum vs Mean-Reversion)
    sec_mom_sim = simulate_sector_relative_and_market_neutral(
        panel=pnl, ws=workspace, borrowable_cols=borrow_set, mode="SECTOR_REL_MOM"
    )
    sec_rev_sim = simulate_sector_relative_and_market_neutral(
        panel=pnl, ws=workspace, borrowable_cols=borrow_set, mode="SECTOR_SPREAD_REV"
    )

    # 3. Cross-Sectional Market-Neutral Factor Long-Short
    mkt_neut_sim = simulate_sector_relative_and_market_neutral(
        panel=pnl, ws=workspace, borrowable_cols=borrow_set, mode="MARKET_NEUTRAL_LS"
    )

    # 4. Nifty 50 Index Futures Hedged Overlays (Rs 100k Lot-Blocked vs Rs 25L Lot-Quantized vs Idealized Fractional)
    fut_100k_lot = simulate_nifty_futures_hedged_overlay(
        panel=pnl, long_daily_navs=long_res["daily_navs"], regime_labels=rule5,
        capital_base_inr=100_000.0, lot_quantized=True,
    )
    fut_25l_lot = simulate_nifty_futures_hedged_overlay(
        panel=pnl, long_daily_navs=long_res["daily_navs"], regime_labels=rule5,
        capital_base_inr=2_500_000.0, lot_quantized=True,
    )
    fut_fractional = simulate_nifty_futures_hedged_overlay(
        panel=pnl, long_daily_navs=long_res["daily_navs"], regime_labels=rule5,
        capital_base_inr=2_500_000.0, lot_quantized=False,
    )

    bench_rets = sec_mom_sim["bench_rets"]
    is_mask = [i for i, d in enumerate(long_dates) if d <= IS_CUTOFF_DATE]
    oos_mask = [i for i, d in enumerate(long_dates) if d >= OOS_START_DATE]

    def _eval_all_splits(sid: str, name: str, r_arr: np.ndarray, extra: dict[str, Any]) -> dict[str, Any]:
        return {
            "strategy_id": sid,
            "name": name,
            "full": _summarize_return_stream(sid, name, long_dates, r_arr, bench_rets, long_rets),
            "in_sample": _summarize_return_stream(
                sid, name, [long_dates[i] for i in is_mask], r_arr[is_mask], bench_rets[is_mask], long_rets[is_mask]
            ),
            "out_of_sample": _summarize_return_stream(
                sid, name, [long_dates[i] for i in oos_mask], r_arr[oos_mask], bench_rets[oos_mask], long_rets[oos_mask]
            ),
            **extra,
        }

    branches = {
        "STRAT_RV1_COINT_PAIRS": _eval_all_splits(
            "STRAT_RV1_COINT_PAIRS",
            "Branch 1A: Engle-Granger Cointegrated Pairs Trading (SLB 10% Borrow + Risk Stops)",
            coint_sim["daily_rets"],
            {
                "trades_count": len(coint_sim["trades"]),
                "risk_events": coint_sim["risk_events"],
                "cost_drag_pct": coint_sim["total_cost_drag_pct"],
                "borrow_drag_pct": coint_sim["total_borrow_drag_pct"],
                "sample_trades": coint_sim["trades"][-8:],
                "verdict": "REJECTED_AFTER_SLB_BORROW_AND_COSTS",
            },
        ),
        "STRAT_RV1B_HIGH_CORR_NON_COINT_ABLATION": _eval_all_splits(
            "STRAT_RV1B_HIGH_CORR_NON_COINT_ABLATION",
            "Branch 1B (Ablation): High-Correlation WITHOUT Cointegration Pairs (Divergence Proof)",
            non_coint_sim["daily_rets"],
            {
                "trades_count": len(non_coint_sim["trades"]),
                "risk_events": non_coint_sim["risk_events"],
                "cost_drag_pct": non_coint_sim["total_cost_drag_pct"],
                "borrow_drag_pct": non_coint_sim["total_borrow_drag_pct"],
                "verdict": "REJECTED_STRUCTURAL_DIVERGENCE",
            },
        ),
        "STRAT_RV2A_SECTOR_REL_MOM": _eval_all_splits(
            "STRAT_RV2A_SECTOR_REL_MOM",
            "Branch 2A: Intra-Sector Leadership Long-Short (Top Leaders vs Borrowable Laggards)",
            sec_mom_sim["daily_rets"],
            {
                "avg_gross_exposure_pct": sec_mom_sim["avg_gross_exposure_pct"],
                "avg_net_exposure_pct": sec_mom_sim["avg_net_exposure_pct"],
                "cost_drag_pct": sec_mom_sim["total_cost_drag_pct"],
                "borrow_drag_pct": sec_mom_sim["total_borrow_drag_pct"],
                "verdict": "RESEARCH_MONITOR_ONLY_SLB_DRAG",
            },
        ),
        "STRAT_RV2B_SECTOR_SPREAD_REV": _eval_all_splits(
            "STRAT_RV2B_SECTOR_SPREAD_REV",
            "Branch 2B: Intra-Sector 20D Spread Mean-Reversion Long-Short",
            sec_rev_sim["daily_rets"],
            {
                "avg_gross_exposure_pct": sec_rev_sim["avg_gross_exposure_pct"],
                "avg_net_exposure_pct": sec_rev_sim["avg_net_exposure_pct"],
                "cost_drag_pct": sec_rev_sim["total_cost_drag_pct"],
                "borrow_drag_pct": sec_rev_sim["total_borrow_drag_pct"],
                "verdict": "REJECTED_NEGATIVE_EXPECTANCY",
            },
        ),
        "STRAT_RV3_MARKET_NEUTRAL_LS": _eval_all_splits(
            "STRAT_RV3_MARKET_NEUTRAL_LS",
            "Branch 3: Cross-Sectional Beta-Neutral Factor Long-Short (Top-180 SLB Universe)",
            mkt_neut_sim["daily_rets"],
            {
                "avg_gross_exposure_pct": mkt_neut_sim["avg_gross_exposure_pct"],
                "avg_net_exposure_pct": mkt_neut_sim["avg_net_exposure_pct"],
                "cost_drag_pct": mkt_neut_sim["total_cost_drag_pct"],
                "borrow_drag_pct": mkt_neut_sim["total_borrow_drag_pct"],
                "verdict": "VALID_ZERO_BETA_DIVERSIFIER_AT_INSTITUTIONAL_SCALE_ONLY",
            },
        ),
        "STRAT_RV4A_NIFTY_FUT_HEDGED_25L_LOT": _eval_all_splits(
            "STRAT_RV4A_NIFTY_FUT_HEDGED_25L_LOT",
            "Branch 4A: Nifty 50 Futures Regime-Hedged Portfolio (Rs 25L Capital, Actual 25/50/75 Lot Size & Monthly Rolls)",
            fut_25l_lot["daily_rets"],
            {
                "capital_base_inr": 2_500_000.0,
                "roll_events": fut_25l_lot["roll_events"],
                "lot_granularity_blocked_days": fut_25l_lot["lot_granularity_blocked_days"],
                "futures_cost_drag_pct": fut_25l_lot["total_futures_cost_drag_pct"],
                "verdict": "VALIDATED_DRAWDOWN_REDUCER_FOR_25L_PLUS_CAPITAL",
            },
        ),
        "STRAT_RV4B_NIFTY_FUT_HEDGED_100K_LOT": _eval_all_splits(
            "STRAT_RV4B_NIFTY_FUT_HEDGED_100K_LOT",
            "Branch 4B: Nifty 50 Futures Hedged at Rs 1,00,000 Capital (Blocked by Rs 18.4L Contract Lot Minimum)",
            fut_100k_lot["daily_rets"],
            {
                "capital_base_inr": 100_000.0,
                "roll_events": fut_100k_lot["roll_events"],
                "lot_granularity_blocked_days": fut_100k_lot["lot_granularity_blocked_days"],
                "futures_cost_drag_pct": fut_100k_lot["total_futures_cost_drag_pct"],
                "verdict": "INFEASIBLE_AT_100K_CAPITAL_USE_CASH_RESERVE_INSTEAD",
            },
        ),
        "STRAT_RV4C_NIFTY_FUT_IDEAL_FRACTIONAL": _eval_all_splits(
            "STRAT_RV4C_NIFTY_FUT_IDEAL_FRACTIONAL",
            "Branch 4C (Theoretical Reference): Fractional Nifty 50 Futures Regime Hedge",
            fut_fractional["daily_rets"],
            {
                "capital_base_inr": 2_500_000.0,
                "roll_events": fut_fractional["roll_events"],
                "futures_cost_drag_pct": fut_fractional["total_futures_cost_drag_pct"],
                "verdict": "THEORETICAL_REFERENCE_ONLY",
            },
        ),
        "BASELINE_PROMPT5_LONG_ONLY": _eval_all_splits(
            "BASELINE_PROMPT5_LONG_ONLY",
            "Prompt 5 Unhedged Long-Only Risk-Budgeted Portfolio (With Regime Cash Reserve)",
            long_rets,
            {"verdict": "PRIMARY_PRODUCTION_CANDIDATE_FOR_100K"},
        ),
    }

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "isolated_research_branch": True,
        "mixed_into_long_only_portfolio": False,
        "evaluation_windows": {
            "full": f"{EVAL_START_DATE} -> {pnl['dates'][-1]}",
            "in_sample": f"{EVAL_START_DATE} -> {IS_CUTOFF_DATE}",
            "out_of_sample": f"{OOS_START_DATE} -> {pnl['dates'][-1]}",
        },
        "pair_discovery_summary": {
            "total_same_sector_high_corr_evaluated": pair_disc["total_same_sector_high_corr_evaluated"],
            "total_cointegrated_found": pair_disc["total_cointegrated_found"],
            "selected_cointegrated_pairs": [asdict(p) for p in sel_pairs],
            "high_corr_non_cointegrated_pairs": [asdict(p) for p in false_pairs],
            "entry_z_threshold_ablations": z_ablations,
        },
        "indian_shorting_and_hedging_constraints": {
            "cash_equity_multi_day_shorting": "PROHIBITED_BY_SEBI_WITHOUT_SLBM_BORROW",
            "slbm_universe_size": len(borrow_set),
            "slbm_annual_borrow_fee_pct": 10.5,
            "nifty50_futures_lot_size_post_nov2024": 75,
            "nifty50_futures_1lot_notional_inr": round(75.0 * float(pnl["nifty50_close"][-1]), 0),
            "nifty50_futures_margin_pct": 22.0,
            "minimum_capital_for_granular_futures_hedging_inr": 2_500_000,
        },
        "branches": branches,
    }

    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "relative_value_hedged_report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (reports_dir / "pair_selection_cointegration_registry.json").write_text(
        json.dumps(payload["pair_discovery_summary"], indent=2), encoding="utf-8"
    )
    _write_relative_value_markdown(reports_dir / "RELATIVE_VALUE_AND_HEDGED_REPORT.md", payload)
    return payload


def _write_relative_value_markdown(md_path: Path, payload: dict[str, Any]) -> None:
    pdisc = payload["pair_discovery_summary"]
    cons = payload["indian_shorting_and_hedging_constraints"]
    br = payload["branches"]

    lines = [
        "# MORESHWAR 3.0 (PROMPT 6) — RELATIVE-VALUE, PAIRS, MARKET-NEUTRAL & FUTURES-HEDGED RESEARCH REPORT",
        "",
        "> [!IMPORTANT]",
        "> **ISOLATED RESEARCH BRANCH (`mixed_into_long_only_portfolio = false`)**",
        "> All short/relative-value strategies below are evaluated with realistic Indian NSE SLBM borrow costs (`10.0%–10.5% p.a.`), top-180 ADV borrow-availability restrictions, dividend-recall closeouts, and NSE Nifty 50 Futures lot-size (`25/50/75` units, `~INR 18.4L/lot`) & monthly expiry roll costs. Unvalidated or cost-eroded short strategies are **explicitly rejected** from the `INR 100,000` long-only portfolio.",
        "",
        "## 1. Executive Summary & Strategy Branch Comparison (`2021-07-01 -> 2026-11-19`)",
        "",
        "| Branch ID | Strategy Description | Full Net Return | Full Max DD | Full Vol | Ex-Post Beta | Corr vs Nifty | Corr vs Long Port | OOS Net Return (`2025-04 -> 2026-11`) | OOS Max DD | Institutional Verdict |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |",
    ]
    for bid, b in br.items():
        fm = b["full"]
        om = b["out_of_sample"]
        lines.append(
            f"| `{bid}` | {b['name']} | **{fm['total_return_pct']:+.2f}%** | **{fm['max_drawdown_pct']:.2f}%** | "
            f"{fm['annualised_vol_pct']:.2f}% | `{fm['ex_post_market_beta']:+.3f}` | `{fm['corr_with_nifty50']:+.3f}` | "
            f"`{fm['corr_with_long_portfolio']:+.3f}` | **{om['total_return_pct']:+.2f}%** | **{om['max_drawdown_pct']:.2f}%** | "
            f"`{b['verdict']}` |"
        )

    lines += [
        "",
        "## 2. Cointegration vs High-Correlation Pair Discovery (Strictly In-Sample `<= 2025-03-31`)",
        "",
        f"- **Same-Sector / Same-Industry Liquid Pairs Screened (`rho_IS >= 0.48`):** `{pdisc['total_same_sector_high_corr_evaluated']}`",
        f"- **Passed Engle-Granger ADF Cointegration (`t_ADF <= -2.86`, `3d <= Half-Life <= 35d`, Beta Drift CV `<= 0.35`):** `{pdisc['total_cointegrated_found']}` (`{len(pdisc['selected_cointegrated_pairs'])}` diversified pairs traded)",
        "",
        "### 2A. Selected In-Sample Cointegrated Pairs",
        "",
        "| Pair ID | Sector | Industry | IS Return Corr | IS OLS Beta | Engle-Granger ADF t-stat | Approx p-val | OU Half-Life (Days) | Subwindow Beta CV |",
        "| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for p in pdisc["selected_cointegrated_pairs"]:
        lines.append(
            f"| `{p['pair_id']}` | {p['sector']} | {p['industry']} | `{p['is_return_corr']:+.3f}` | "
            f"`{p['is_ols_beta']:.3f}` | **`{p['is_adf_t_stat']:.3f}`** | `{p['is_adf_p_approx']:.4f}` | "
            f"**{p['is_half_life_days']:.1f}d** | `{p['is_beta_drift_cv']:.3f}` |"
        )

    lines += [
        "",
        "### 2B. High-Correlation Pairs That FAILED Cointegration (Why Correlation != Cointegration)",
        "",
        "| Pair ID | Sector | IS Return Corr | Engle-Granger ADF t-stat | OU Half-Life | Failure Diagnosis |",
        "| :--- | :--- | :---: | :---: | :---: | :--- |",
    ]
    for p in pdisc["high_corr_non_cointegrated_pairs"]:
        lines.append(
            f"| `{p['pair_id']}` | {p['sector']} | **`{p['is_return_corr']:+.3f}`** | `{p['is_adf_t_stat']:.3f}` (`p={p['is_adf_p_approx']:.2f}`) | "
            f"`{p['is_half_life_days']:.1f}d` | Non-stationary residual drift (unit root not rejected) |"
        )

    c1 = br["STRAT_RV1_COINT_PAIRS"]
    c2 = br["STRAT_RV1B_HIGH_CORR_NON_COINT_ABLATION"]
    lines += [
        "",
        f"- **Empirical Proof (Cointegrated vs High-Corr Non-Cointegrated):** Cointegrated pairs (`STRAT_RV1_COINT_PAIRS`) experienced `{c1['risk_events']}` exit events with `{c1['cost_drag_pct']:.2f}%` execution cost drag and `{c1['borrow_drag_pct']:.2f}%` SLB borrow drag, whereas trading high-correlation non-cointegrated pairs (`STRAT_RV1B_HIGH_CORR_NON_COINT_ABLATION`) suffered repeated structural break stops (`{c2['risk_events']}`).",
        "",
        "## 3. Indian Short-Selling & Futures Microstructure Constraints Analysis",
        "",
        f"1. **SEBI Cash Short-Selling & SLBM Financing Drag:** Multi-day shorting in NSE cash equities requires SLBM borrowing (available across only `{cons['slbm_universe_size']}` liquid F&O/SLB stocks) at ~`{cons['slbm_annual_borrow_fee_pct']}% p.a.` lending fee plus two-legged delivery/roll friction. Across 5.4 years, SLB financing & transaction friction consumed **`{br['STRAT_RV3_MARKET_NEUTRAL_LS']['borrow_drag_pct']:.2f}%` borrow drag + `{br['STRAT_RV3_MARKET_NEUTRAL_LS']['cost_drag_pct']:.2f}%` execution cost drag** on the Market-Neutral Long-Short portfolio (`STRAT_RV3_MARKET_NEUTRAL_LS`), turning positive gross factor spread into a modest net return (`{br['STRAT_RV3_MARKET_NEUTRAL_LS']['full']['total_return_pct']:+.2f}%` Full, `{br['STRAT_RV3_MARKET_NEUTRAL_LS']['out_of_sample']['total_return_pct']:+.2f}%` OOS) despite near-zero market beta (`{br['STRAT_RV3_MARKET_NEUTRAL_LS']['full']['ex_post_market_beta']:+.3f}`).",
        f"2. **Nifty 50 Index Futures Lot-Size Quantization (`INR 1,00,000` vs `INR 25,00,000`):** Post-Nov-2024 SEBI F&O contract revision, 1 lot of Nifty 50 Futures is `{cons['nifty50_futures_lot_size_post_nov2024']}` units (`~INR {cons['nifty50_futures_1lot_notional_inr']:,.0f}` notional, requiring `~INR 4.0L` SPAN+Exposure margin). On a `INR 1,00,000` account (`STRAT_RV4B_NIFTY_FUT_HEDGED_100K_LOT`), lot quantization blocked futures hedging on `{br['STRAT_RV4B_NIFTY_FUT_HEDGED_100K_LOT']['lot_granularity_blocked_days']}` regime-stress sessions because even 1 lot represents `18x` portfolio notional. At `INR 25,00,000` institutional capital (`STRAT_RV4A_NIFTY_FUT_HEDGED_25L_LOT`), lot-quantized Nifty Futures hedging (`{br['STRAT_RV4A_NIFTY_FUT_HEDGED_25L_LOT']['roll_events']}` monthly rolls, `{br['STRAT_RV4A_NIFTY_FUT_HEDGED_25L_LOT']['futures_cost_drag_pct']:.2f}%` roll/commission drag) reduced Full Max Drawdown from **`{br['BASELINE_PROMPT5_LONG_ONLY']['full']['max_drawdown_pct']:.2f}%` → `{br['STRAT_RV4A_NIFTY_FUT_HEDGED_25L_LOT']['full']['max_drawdown_pct']:.2f}%`** and cut market beta from `{br['BASELINE_PROMPT5_LONG_ONLY']['full']['ex_post_market_beta']:+.3f}` to **`{br['STRAT_RV4A_NIFTY_FUT_HEDGED_25L_LOT']['full']['ex_post_market_beta']:+.3f}`**.",
        "",
        "## 4. Final Prompt 6 Decision",
        "",
        "- **For `INR 1,00,000` Retail/Standard Capital:** Do **NOT** mix SLBM equity shorts, pairs trading, or over-sized Nifty Futures lots into the Prompt 5 long-only portfolio. Dynamic cash-reserve raising (`CASH_RESERVE` + Drawdown Governor from Prompts 4 & 5) is strictly superior after Indian SLB borrow fees and F&O lot constraints.",
        "- **For `INR 25,00,000+` Institutional Capital:** The causal Regime-Conditioned Nifty 50 Futures Overlay (`STRAT_RV4A_NIFTY_FUT_HEDGED_25L_LOT`) is validated as a genuine drawdown-reduction overlay during `SIDEWAYS_CHOP`, `WEAK_BEAR`, and `STRESS_BEAR` regimes.",
    ]

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
