"""Moreshwar 3.0 Prompt 3 — Five Independent Quantitative Strategy Families (A..E).

Executes five structurally distinct, strictly causal Indian equity strategy families
on the identical Prompt 1 Point-in-Time NSE Universe (`is_pit_elig`, ETFs excluded)
and Prompt 3 Fair-Comparison Portfolio Simulator (`simulate_independent_strategy_m3`):

- Family A (`STRAT_A_CS_MOMENTUM`): Cross-Sectional Factor Momentum & 52W-High Anchor
  (Validated Prompt 2 factors: `DIST_252D_HIGH`, `MOM_DETERIORATION_60D`, `VOL_ADJ_MOM_60D`, `AMIHUD_LIQUIDITY_20D`)
- Family B (`STRAT_B_VCP_BREAKOUT`): Volatility-Contraction & Delivery/Volume Breakout
  (Event-driven swing breakout on `BREAKOUT_QUALITY_20D`, `VOL_CONTRACTION_10V60`, `REL_VOL_5V20`, `DELIVERY_SURGE_5V20`)
- Family C (`STRAT_C_TREND_PULLBACK`): Primary-Uptrend ATR Pullback Mean-Reversion
  (Event-driven dip-buying in `DIST_252D_HIGH >= 0.82` uptrends via `PULLBACK_DEPTH_SMA50` & `DELIVERY_PCT_20D`)
- Family D (`STRAT_D_SECTOR_ROTATION`): Causal PIT Sector-First Leadership Rotation
  (Top-3 sectors by `SECTOR_BREADTH_SMA50` + `SECTOR_RS_60D`, selecting top relative-strength leaders per sector)
- Family E (`STRAT_E_DEFENSIVE_YIELD_LOWVOL`): Defensive PIT Dividend-Yield & Low-Volatility Compounder
  (Validated Prompt 2 factor `DIV_YIELD_TTM_PIT` + `LOW_VOL_20D` + `IDIO_VOL_20D` + `AMIHUD_LIQUIDITY_20D`)
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np

from .factors import _rolling_mean, compute_all_m3_factors
from .panel_loader import build_or_load_m3_panel
from .schema import DEFAULT_M3_REPORTS_DIR
from .strategy_base import (
    EVAL_START_DATE,
    INITIAL_CAPITAL_M3,
    IS_CUTOFF_DATE,
    OOS_START_DATE,
    StrategySimConfig,
    simulate_independent_strategy_m3,
)


def _cross_sectional_pct_rank(mat: np.ndarray, elig: np.ndarray) -> np.ndarray:
    """Compute strictly causal cross-sectional percentile rank in [0, 1] per date row across PIT-eligible stocks."""
    t_len, n_len = mat.shape
    out = np.full((t_len, n_len), np.nan, dtype=np.float32)
    for ti in range(t_len):
        row = mat[ti]
        ok = elig[ti] & (~np.isnan(row))
        idx = np.where(ok)[0]
        m = len(idx)
        if m < 5:
            continue
        vals = row[idx]
        order = np.argsort(vals, kind="mergesort")
        ranks = np.empty(m, dtype=np.float32)
        ranks[order] = np.linspace(0.0, 1.0, m, dtype=np.float32)
        out[ti, idx] = ranks
    return out


def prepare_strategy_signal_workspace(
    panel: dict[str, Any],
    factors: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    """Precompute causal ATR-20, SMA50, and cross-sectional PIT factor percentile matrices."""
    facs = factors or compute_all_m3_factors(panel)
    adj_h = panel["adj_high"].astype(np.float64)
    adj_l = panel["adj_low"].astype(np.float64)
    adj_c = panel["adj_close"].astype(np.float64)
    elig = panel["is_pit_elig"]
    t_len, n_len = adj_c.shape

    prev_c = np.vstack([np.full((1, n_len), np.nan), adj_c[:-1]])
    tr = np.maximum(adj_h - adj_l, np.maximum(np.abs(adj_h - prev_c), np.abs(adj_l - prev_c)))
    atr_20d = _rolling_mean(tr, 20)
    sma_50d = _rolling_mean(adj_c, 50)

    ranks = {
        "r_dist252": _cross_sectional_pct_rank(facs["DIST_252D_HIGH"], elig),
        "r_mom_det60": _cross_sectional_pct_rank(facs["MOM_DETERIORATION_60D"], elig),
        "r_vol_mom60": _cross_sectional_pct_rank(facs["VOL_ADJ_MOM_60D"], elig),
        "r_amihud20": _cross_sectional_pct_rank(facs["AMIHUD_LIQUIDITY_20D"], elig),
        "r_vcp": _cross_sectional_pct_rank(facs["VOL_CONTRACTION_10V60"], elig),
        "r_div_yield": _cross_sectional_pct_rank(facs["DIV_YIELD_TTM_PIT"], elig),
        "r_low_vol": _cross_sectional_pct_rank(facs["LOW_VOL_20D"], elig),
        "r_idio_vol": _cross_sectional_pct_rank(facs["IDIO_VOL_20D"], elig),
        "r_stock_sec": _cross_sectional_pct_rank(facs["STOCK_VS_SECTOR_20D"], elig),
    }
    return {
        "factors": facs,
        "ranks": ranks,
        "atr_20d": atr_20d,
        "sma_50d": sma_50d,
    }


def get_strategy_family_configs() -> list[StrategySimConfig]:
    """Return standardized fair-comparison configurations for Strategy Families A..E."""
    return [
        StrategySimConfig(
            strategy_id="STRAT_A_CS_MOMENTUM",
            strategy_name="Family A: Cross-Sectional 52W-High & Validated Factor Momentum",
            family="A_CS_MOMENTUM",
            max_positions=8,
            max_single_weight_pct=14.0,
            max_sector_positions=2,
            max_portfolio_exposure_pct=95.0,
            risk_per_trade_pct=1.5,
            rebalance_mode="PERIODIC",
            rebalance_interval_days=25,
            hold_max_days=90,
            initial_atr_stop_mult=4.0,
            trailing_atr_mult=4.8,
            breakeven_trigger_atr=3.0,
            target_atr_mult=0.0,
            cooldown_days=10,
        ),
        StrategySimConfig(
            strategy_id="STRAT_B_VCP_BREAKOUT",
            strategy_name="Family B: Volatility-Contraction & Volume/Delivery Breakout",
            family="B_VCP_BREAKOUT",
            max_positions=7,
            max_single_weight_pct=15.0,
            max_sector_positions=2,
            max_portfolio_exposure_pct=92.0,
            risk_per_trade_pct=1.5,
            rebalance_mode="DAILY_EVENT",
            rebalance_interval_days=5,
            hold_max_days=60,
            initial_atr_stop_mult=4.0,
            trailing_atr_mult=4.8,
            breakeven_trigger_atr=3.0,
            target_atr_mult=0.0,
            cooldown_days=10,
        ),
        StrategySimConfig(
            strategy_id="STRAT_C_TREND_PULLBACK",
            strategy_name="Family C: Primary-Uptrend ATR Pullback Mean-Reversion",
            family="C_TREND_PULLBACK",
            max_positions=7,
            max_single_weight_pct=15.0,
            max_sector_positions=2,
            max_portfolio_exposure_pct=90.0,
            risk_per_trade_pct=1.5,
            rebalance_mode="DAILY_EVENT",
            rebalance_interval_days=5,
            hold_max_days=45,
            initial_atr_stop_mult=3.8,
            trailing_atr_mult=4.5,
            breakeven_trigger_atr=2.8,
            target_atr_mult=5.5,
            cooldown_days=10,
        ),
        StrategySimConfig(
            strategy_id="STRAT_D_SECTOR_ROTATION",
            strategy_name="Family D: Causal PIT Sector-First Leadership Rotation",
            family="D_SECTOR_ROTATION",
            max_positions=8,
            max_single_weight_pct=14.0,
            max_sector_positions=3,
            max_portfolio_exposure_pct=95.0,
            risk_per_trade_pct=1.5,
            rebalance_mode="PERIODIC",
            rebalance_interval_days=20,
            hold_max_days=80,
            initial_atr_stop_mult=4.0,
            trailing_atr_mult=4.8,
            breakeven_trigger_atr=3.0,
            target_atr_mult=0.0,
            cooldown_days=10,
        ),
        StrategySimConfig(
            strategy_id="STRAT_E_DEFENSIVE_YIELD_LOWVOL",
            strategy_name="Family E: Defensive PIT Dividend-Yield & Low-Volatility Compounder",
            family="E_DEFENSIVE_YIELD_LOWVOL",
            max_positions=8,
            max_single_weight_pct=14.0,
            max_sector_positions=2,
            max_portfolio_exposure_pct=95.0,
            risk_per_trade_pct=1.5,
            rebalance_mode="PERIODIC",
            rebalance_interval_days=30,
            hold_max_days=120,
            initial_atr_stop_mult=4.5,
            trailing_atr_mult=5.2,
            breakeven_trigger_atr=3.2,
            target_atr_mult=0.0,
            cooldown_days=10,
        ),
    ]


def build_candidate_generator(
    cfg: StrategySimConfig,
    ws: dict[str, Any],
):
    """Build strictly causal candidate generator function `fn(ti, panel, open_pos)` for `cfg`."""
    facs = ws["factors"]
    ranks = ws["ranks"]
    atr_20d = ws["atr_20d"]
    sma_50d = ws["sma_50d"]

    dist252 = facs["DIST_252D_HIGH"]
    mom60 = facs["MOM_60D"]
    mom120_s10 = facs["MOM_120_SKIP10"]
    brk20 = facs["BREAKOUT_QUALITY_20D"]
    rel_vol = facs["REL_VOL_5V20"]
    deliv_surge = facs["DELIVERY_SURGE_5V20"]
    deliv_20 = facs["DELIVERY_PCT_20D"]
    pullback = facs["PULLBACK_DEPTH_SMA50"]
    sec_br50 = facs["SECTOR_BREADTH_SMA50"]
    sec_rs60 = facs["SECTOR_RS_60D"]
    sec_mom20 = facs["SECTOR_MOM_20D"]
    div_y = facs["DIV_YIELD_TTM_PIT"]

    r_dist = ranks["r_dist252"]
    r_mdet = ranks["r_mom_det60"]
    r_vmom = ranks["r_vol_mom60"]
    r_amihud = ranks["r_amihud20"]
    r_vcp = ranks["r_vcp"]
    r_div = ranks["r_div_yield"]
    r_lvol = ranks["r_low_vol"]
    r_ivol = ranks["r_idio_vol"]
    r_stsec = ranks["r_stock_sec"]

    def _cand_fn(ti: int, panel: dict[str, Any], open_pos: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
        elig_row = panel["is_pit_elig"][ti]
        close_row = panel["adj_close"][ti]
        sectors = panel["sectors"]
        cands: list[dict[str, Any]] = []

        if cfg.strategy_id == "STRAT_A_CS_MOMENTUM":
            ok = (
                elig_row
                & (close_row > sma_50d[ti])
                & (dist252[ti] >= 0.88)
                & (mom60[ti] > 0.04)
                & (r_amihud[ti] >= 0.45)
                & (~np.isnan(r_dist[ti]))
                & (~np.isnan(r_mdet[ti]))
                & (~np.isnan(r_vmom[ti]))
            )
            cols = np.where(ok)[0]
            if len(cols) == 0:
                return []
            scores = (
                0.50 * r_dist[ti, cols]
                + 0.25 * r_mdet[ti, cols]
                + 0.15 * r_vmom[ti, cols]
                + 0.10 * r_amihud[ti, cols]
            )
            order = np.argsort(-scores)
            for idx_o in order[:40]:
                c = int(cols[idx_o])
                px = float(close_row[c])
                atr = float(atr_20d[ti, c]) if not np.isnan(atr_20d[ti, c]) and atr_20d[ti, c] > 0 else px * 0.025
                sl_r = min(0.14, max(0.07, (cfg.initial_atr_stop_mult * atr) / max(1.0, px)))
                cands.append({"col": c, "score": float(scores[idx_o]), "atr": atr, "stop_ratio": sl_r, "target_ratio": None})
            return cands

        if cfg.strategy_id == "STRAT_B_VCP_BREAKOUT":
            vol_confirm = (rel_vol[ti] >= 1.10) | (np.nan_to_num(deliv_surge[ti], nan=0.0) >= 2.0)
            ok = (
                elig_row
                & (close_row > sma_50d[ti])
                & (dist252[ti] >= 0.90)
                & (brk20[ti] > -0.20)
                & (brk20[ti] <= 2.0)
                & vol_confirm
                & (r_vcp[ti] >= 0.45)
                & (r_amihud[ti] >= 0.40)
            )
            cols = np.where(ok)[0]
            if len(cols) == 0:
                return []
            scores = (
                0.40 * r_dist[ti, cols]
                + 0.30 * r_vcp[ti, cols]
                + 0.15 * np.clip((brk20[ti, cols] + 0.2) / 1.5, 0.0, 1.0)
                + 0.15 * r_amihud[ti, cols]
            )
            order = np.argsort(-scores)
            for idx_o in order[:35]:
                c = int(cols[idx_o])
                px = float(close_row[c])
                atr = float(atr_20d[ti, c]) if not np.isnan(atr_20d[ti, c]) and atr_20d[ti, c] > 0 else px * 0.025
                sl_r = min(0.13, max(0.065, (cfg.initial_atr_stop_mult * atr) / max(1.0, px)))
                cands.append({"col": c, "score": float(scores[idx_o]), "atr": atr, "stop_ratio": sl_r, "target_ratio": None})
            return cands

        if cfg.strategy_id == "STRAT_C_TREND_PULLBACK":
            deliv_ok = np.isnan(deliv_20[ti]) | (deliv_20[ti] >= 38.0)
            ok = (
                elig_row
                & (close_row > sma_50d[ti])
                & (dist252[ti] >= 0.85)
                & (np.nan_to_num(mom120_s10[ti], nan=0.0) > 0.06)
                & (pullback[ti] >= 0.6)
                & (pullback[ti] <= 2.8)
                & deliv_ok
                & (r_amihud[ti] >= 0.45)
            )
            cols = np.where(ok)[0]
            if len(cols) == 0:
                return []
            scores = (
                0.45 * r_dist[ti, cols]
                + 0.25 * r_lvol[ti, cols]
                + 0.15 * np.clip(pullback[ti, cols] / 2.8, 0.0, 1.0)
                + 0.15 * r_amihud[ti, cols]
            )
            order = np.argsort(-scores)
            for idx_o in order[:35]:
                c = int(cols[idx_o])
                px = float(close_row[c])
                atr = float(atr_20d[ti, c]) if not np.isnan(atr_20d[ti, c]) and atr_20d[ti, c] > 0 else px * 0.025
                sl_r = min(0.12, max(0.065, (cfg.initial_atr_stop_mult * atr) / max(1.0, px)))
                tp_r = min(0.22, max(0.10, (cfg.target_atr_mult * atr) / max(1.0, px)))
                cands.append({"col": c, "score": float(scores[idx_o]), "atr": atr, "stop_ratio": sl_r, "target_ratio": tp_r})
            return cands

        if cfg.strategy_id == "STRAT_D_SECTOR_ROTATION":
            ok_base = (
                elig_row
                & (close_row > sma_50d[ti])
                & (dist252[ti] >= 0.85)
                & (~np.isnan(sec_br50[ti]))
                & (~np.isnan(sec_rs60[ti]))
                & (sec_br50[ti] >= 0.52)
                & (r_amihud[ti] >= 0.40)
            )
            cols_all = np.where(ok_base)[0]
            if len(cols_all) == 0:
                return []
            sec_scores: dict[str, float] = {}
            for c in cols_all:
                s_nm = str(sectors[c])
                if "Unclassified" in s_nm or s_nm in sec_scores:
                    continue
                sec_scores[s_nm] = float(0.45 * sec_br50[ti, c] + 0.35 * sec_rs60[ti, c] + 0.20 * np.nan_to_num(sec_mom20[ti, c], nan=0.0))
            top_secs = [s for s, _ in sorted(sec_scores.items(), key=lambda kv: -kv[1])[:3]]
            if not top_secs:
                return []
            top_sec_set = set(top_secs)
            cols = np.array([c for c in cols_all if str(sectors[c]) in top_sec_set], dtype=int)
            if len(cols) == 0:
                return []
            scores = (
                0.45 * r_dist[ti, cols]
                + 0.25 * np.nan_to_num(r_stsec[ti, cols], nan=0.5)
                + 0.15 * r_mdet[ti, cols]
                + 0.15 * r_amihud[ti, cols]
            )
            order = np.argsort(-scores)
            for idx_o in order[:35]:
                c = int(cols[idx_o])
                px = float(close_row[c])
                atr = float(atr_20d[ti, c]) if not np.isnan(atr_20d[ti, c]) and atr_20d[ti, c] > 0 else px * 0.025
                sl_r = min(0.14, max(0.07, (cfg.initial_atr_stop_mult * atr) / max(1.0, px)))
                cands.append({"col": c, "score": float(scores[idx_o]), "atr": atr, "stop_ratio": sl_r, "target_ratio": None})
            return cands

        if cfg.strategy_id == "STRAT_E_DEFENSIVE_YIELD_LOWVOL":
            ok = (
                elig_row
                & (dist252[ti] >= 0.84)
                & (close_row >= sma_50d[ti] * 0.98)
                & (np.nan_to_num(div_y[ti], nan=0.0) > 0.004)
                & (r_lvol[ti] >= 0.50)
                & (r_amihud[ti] >= 0.45)
                & (~np.isnan(r_div[ti]))
            )
            cols = np.where(ok)[0]
            if len(cols) == 0:
                return []
            scores = (
                0.40 * r_div[ti, cols]
                + 0.25 * r_lvol[ti, cols]
                + 0.20 * r_dist[ti, cols]
                + 0.15 * r_ivol[ti, cols]
            )
            order = np.argsort(-scores)
            for idx_o in order[:40]:
                c = int(cols[idx_o])
                px = float(close_row[c])
                atr = float(atr_20d[ti, c]) if not np.isnan(atr_20d[ti, c]) and atr_20d[ti, c] > 0 else px * 0.020
                sl_r = min(0.14, max(0.08, (cfg.initial_atr_stop_mult * atr) / max(1.0, px)))
                cands.append({"col": c, "score": float(scores[idx_o]), "atr": atr, "stop_ratio": sl_r, "target_ratio": None})
            return cands

        return []

    return _cand_fn


def run_moreshwar_v3_strategy_families(
    *,
    panel: dict[str, Any] | None = None,
    ws: dict[str, Any] | None = None,
    reports_dir: Path = DEFAULT_M3_REPORTS_DIR / "strategies",
) -> dict[str, Any]:
    """Run all 5 independent Strategy Families (A..E) on Full, IS, and OOS windows and write Prompt 3 reports."""
    pnl = panel or build_or_load_m3_panel()
    workspace = ws or prepare_strategy_signal_workspace(pnl)
    cfgs = get_strategy_family_configs()

    family_results: dict[str, dict[str, Any]] = {}
    full_daily_ret_map: dict[str, dict[str, float]] = {}
    full_nav_series_map: dict[str, list[dict[str, Any]]] = {}

    for cfg in cfgs:
        cand_fn = build_candidate_generator(cfg, workspace)
        res_full = simulate_independent_strategy_m3(
            panel=pnl, cfg=cfg, candidate_fn=cand_fn,
            date_start=EVAL_START_DATE, date_end=str(pnl["dates"][-1]),
            initial_capital=INITIAL_CAPITAL_M3,
        )
        res_is = simulate_independent_strategy_m3(
            panel=pnl, cfg=cfg, candidate_fn=cand_fn,
            date_start=EVAL_START_DATE, date_end=IS_CUTOFF_DATE,
            initial_capital=INITIAL_CAPITAL_M3,
        )
        res_oos = simulate_independent_strategy_m3(
            panel=pnl, cfg=cfg, candidate_fn=cand_fn,
            date_start=OOS_START_DATE, date_end=str(pnl["dates"][-1]),
            initial_capital=INITIAL_CAPITAL_M3,
        )
        family_results[cfg.strategy_id] = {
            "config": res_full["config"],
            "full_metrics": res_full["metrics"],
            "is_metrics": res_is["metrics"],
            "oos_metrics": res_oos["metrics"],
            "sample_trades": res_full["trades"][-10:],
            "exit_reason_breakdown": _count_exit_reasons(res_full["trades"]),
        }
        full_daily_ret_map[cfg.strategy_id] = {d["date"]: float(d["ret"]) for d in res_full["daily_navs"]}
        full_nav_series_map[cfg.strategy_id] = res_full["daily_navs"]

    # Compute Pairwise Daily Return Correlation Matrix across Strategy Families A..E
    sids = [c.strategy_id for c in cfgs]
    common_dates = sorted(set.intersection(*(set(full_daily_ret_map[s].keys()) for s in sids)))
    ret_mat = np.array([[full_daily_ret_map[s][dt] for dt in common_dates] for s in sids], dtype=np.float64)
    corr_mat = np.corrcoef(ret_mat)
    corr_dict: dict[str, dict[str, float]] = {}
    for i, s1 in enumerate(sids):
        corr_dict[s1] = {s2: round(float(corr_mat[i, j]), 3) for j, s2 in enumerate(sids)}

    # Equal-weight 5-Family Composite Benchmark Comparison
    eq_rets = np.mean(ret_mat, axis=0)
    eq_navs = [INITIAL_CAPITAL_M3]
    for r in eq_rets[1:]:
        eq_navs.append(eq_navs[-1] * (1.0 + float(r)))
    eq_tot_ret = round(((eq_navs[-1] / INITIAL_CAPITAL_M3) - 1.0) * 100.0, 2)
    pk = np.maximum.accumulate(eq_navs)
    eq_max_dd = round(abs(float(np.min((np.array(eq_navs) - pk) / pk))) * 100.0, 2)

    reports_dir.mkdir(parents=True, exist_ok=True)
    report_payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "evaluation_windows": {
            "full": f"{EVAL_START_DATE} -> {pnl['dates'][-1]}",
            "in_sample": f"{EVAL_START_DATE} -> {IS_CUTOFF_DATE}",
            "out_of_sample": f"{OOS_START_DATE} -> {pnl['dates'][-1]}",
        },
        "initial_capital_inr": INITIAL_CAPITAL_M3,
        "equal_weight_5family_summary": {
            "full_total_return_pct": eq_tot_ret,
            "full_max_drawdown_pct": eq_max_dd,
        },
        "strategies": family_results,
        "daily_return_correlations": corr_dict,
    }
    (reports_dir / "strategy_families_report.json").write_text(json.dumps(report_payload, indent=2), encoding="utf-8")
    (reports_dir / "strategy_correlation_matrix.json").write_text(json.dumps(corr_dict, indent=2), encoding="utf-8")
    _write_strategy_families_markdown(reports_dir / "STRATEGY_FAMILIES_REPORT.md", report_payload)

    return {
        "report": report_payload,
        "daily_ret_map": full_daily_ret_map,
        "daily_nav_map": full_nav_series_map,
        "common_dates": common_dates,
    }


def _count_exit_reasons(trades: list[dict[str, Any]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for t in trades:
        r = str(t.get("exit_reason", "OTHER"))
        out[r] = out.get(r, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def _write_strategy_families_markdown(md_path: Path, payload: dict[str, Any]) -> None:
    strats = payload["strategies"]
    corr = payload["daily_return_correlations"]
    lines = [
        "# MORESHWAR 3.0 (PROMPT 3) — INDEPENDENT MULTI-STRATEGY FAMILIES REPORT",
        "",
        f"- **Evaluation Window (Full):** `{payload['evaluation_windows']['full']}`",
        f"- **In-Sample Window:** `{payload['evaluation_windows']['in_sample']}` | **Out-of-Sample Window:** `{payload['evaluation_windows']['out_of_sample']}`",
        f"- **Initial Capital:** `INR {payload['initial_capital_inr']:,.0f}` per strategy (Isolated Stateless Harness, Zerodha Delivery + 7.5 bps Slippage)",
        f"- **Equal-Weight 5-Family Baseline:** Full Return **`{payload['equal_weight_5family_summary']['full_total_return_pct']:+.2f}%`** | Max Drawdown **`{payload['equal_weight_5family_summary']['full_max_drawdown_pct']:.2f}%`**",
        "",
        "## 1. Full-Window Strategy Family Comparison (`2021-07-01 -> 2026-11-19`)",
        "",
        "| Strategy ID | Family | Net Return | CAGR | Max DD | Sharpe | Sortino | Net PF | Trades | Win % | Cost Drag | Util % | Reconciliation |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for sid, d in strats.items():
        m = d["full_metrics"]
        lines.append(
            f"| `{sid}` | `{d['config']['family']}` | **{m['total_return_pct']:+.2f}%** | {m['cagr_pct']:+.2f}% | "
            f"{m['max_drawdown_pct']:.2f}% | {m['sharpe_ratio']:+.2f} | {m['sortino_ratio']:+.2f} | "
            f"{m['net_profit_factor']:.2f} | {m['total_trades']} | {m['win_rate_pct']:.1f}% | "
            f"{m['cost_drag_pct']:.2f}% | {m['capital_utilisation_pct']:.1f}% | `{m['reconciliation_diff']:+.2f}` |"
        )

    lines += [
        "",
        "## 2. In-Sample (`2021-07 -> 2025-03`) vs Out-of-Sample (`2025-04 -> 2026-11`) Breakdown",
        "",
        "| Strategy ID | IS Net Return | IS CAGR | IS Max DD | IS Sharpe | IS PF | OOS Net Return | OOS CAGR | OOS Max DD | OOS Sharpe | OOS PF | OOS Excess vs Nifty |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for sid, d in strats.items():
        mi = d["is_metrics"]
        mo = d["oos_metrics"]
        lines.append(
            f"| `{sid}` | {mi['total_return_pct']:+.2f}% | {mi['cagr_pct']:+.2f}% | {mi['max_drawdown_pct']:.2f}% | "
            f"{mi['sharpe_ratio']:+.2f} | {mi['net_profit_factor']:.2f} | **{mo['total_return_pct']:+.2f}%** | "
            f"{mo['cagr_pct']:+.2f}% | {mo['max_drawdown_pct']:.2f}% | {mo['sharpe_ratio']:+.2f} | "
            f"{mo['net_profit_factor']:.2f} | **{mo['excess_vs_nifty_pp']:+.2f} pp** |"
        )

    lines += [
        "",
        "## 3. Daily Return Correlation Matrix Across Strategy Families",
        "",
        "| Strategy | " + " | ".join(f"`{s}`" for s in strats) + " |",
        "| :--- | " + " | ".join(":---:" for _ in strats) + " |",
    ]
    for s1 in strats:
        row_cells = " | ".join(f"{corr[s1][s2]:+.3f}" for s2 in strats)
        lines.append(f"| `{s1}` | {row_cells} |")

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
