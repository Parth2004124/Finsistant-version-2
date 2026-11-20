"""Moreshwar 3.1 (Prompt 4/5) — Family E Component Ablation & Controlled New Strategy Research.

Implements:
- Section 7: Isolated one-at-a-time Family E component investigation (dividend yield, low-vol,
  52W-high/SMA50 trend confirmation, sector RS tilt, sector cap, hybrid stop) on strict chronological
  inner vs outer windows with nearby parameter perturbation checks.
- Section 8: Controlled research of 4 genuinely distinct strictly-PIT strategy families PLUS formal
  evidence-based admission retest of suspended Family A, evaluated against all 6 admission gates
  (benchmark/diversification edge, chronological survival, non-outlier dependence, turnover/DD caps,
  perturbation stability, and PIT/accounting integrity).
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np

from finsistant_v2.m3.portfolio_construction import EVAL_START_DATE, INITIAL_CAPITAL_M3
from finsistant_v2.m3_1.adaptive_strategy_engine import (
    AdaptiveRegimePolicy,
    simulate_m3_1_adaptive_regime_portfolio,
)
from finsistant_v2.m3_1.execution_simulation_engine import ExecutionIntelligenceConfig
from finsistant_v2.m3_1.sector_hierarchy import UNKNOWN_SECTOR_LABEL
from finsistant_v2.m3_1.transaction_cost_intelligence import TransactionCostIntelligence


def _clone_exec_cfg(base: ExecutionIntelligenceConfig, **overrides: Any) -> ExecutionIntelligenceConfig:
    d = asdict(base)
    d.update(overrides)
    d["active_sleeves"] = tuple(d["active_sleeves"])
    d["sleeve_weights"] = tuple((str(k), float(v)) for k, v in d["sleeve_weights"])
    return ExecutionIntelligenceConfig(**d)


def _precompute_custom_candidates(
    *,
    ctx: dict[str, Any],
    strategy_id: str,
    variant_kind: str,
    param_shift: float = 0.0,
) -> dict[int, list[dict[str, Any]]]:
    """Build strictly causal PIT candidate lists `[ti -> list[dict]]` for a Family E variant or new strategy."""
    panel = ctx["panel"]
    ws = ctx["ws_m31"]
    facs = ws["factors"]
    ranks = ws["ranks"]
    atr_20d = ws["atr_20d"]
    sma_50d = ws["sma_50d"]

    elig = panel["is_pit_elig"]
    close_p = panel["adj_close"]
    sectors = panel["sectors"]
    t_len = close_p.shape[0]

    dist252 = facs["DIST_252D_HIGH"]
    div_y = facs["DIV_YIELD_TTM_PIT"]
    mom120 = facs["MOM_120_SKIP10"]
    sec_br50 = facs["SECTOR_BREADTH_SMA50"]
    sec_rs60 = facs["SECTOR_RS_60D"]

    r_div = ranks["r_div_yield"]
    r_lvol = ranks["r_low_vol"]
    r_dist = ranks["r_dist252"]
    r_ivol = ranks["r_idio_vol"]
    r_amihud = ranks["r_amihud20"]
    r_vmom = ranks["r_vol_mom60"]
    r_stsec = ranks["r_stock_sec"]

    out: dict[int, list[dict[str, Any]]] = {}
    for ti in range(t_len):
        elig_row = elig[ti]
        c_row = close_p[ti]
        if not np.any(elig_row):
            continue

        if variant_kind == "FAME_DROP_DIV_YIELD":
            ok = elig_row & (dist252[ti] >= 0.84) & (c_row >= sma_50d[ti] * 0.98) & (r_lvol[ti] >= 0.50) & (r_amihud[ti] >= 0.45) & (~np.isnan(r_lvol[ti])) & (~np.isnan(r_dist[ti]))
            cols = np.where(ok)[0]
            if len(cols) == 0:
                continue
            scores = 0.45 * r_lvol[ti, cols] + 0.35 * r_dist[ti, cols] + 0.20 * np.nan_to_num(r_ivol[ti, cols], nan=0.5)
        elif variant_kind == "FAME_DROP_LOW_VOL":
            ok = elig_row & (dist252[ti] >= 0.84) & (c_row >= sma_50d[ti] * 0.98) & (np.nan_to_num(div_y[ti], nan=0.0) > 0.004) & (r_amihud[ti] >= 0.45) & (~np.isnan(r_div[ti]))
            cols = np.where(ok)[0]
            if len(cols) == 0:
                continue
            scores = 0.55 * r_div[ti, cols] + 0.30 * np.nan_to_num(r_dist[ti, cols], nan=0.5) + 0.15 * np.nan_to_num(r_amihud[ti, cols], nan=0.5)
        elif variant_kind == "FAME_DROP_TREND_CONFIRM":
            ok = elig_row & (np.nan_to_num(div_y[ti], nan=0.0) > 0.004) & (r_lvol[ti] >= 0.50) & (r_amihud[ti] >= 0.45) & (~np.isnan(r_div[ti]))
            cols = np.where(ok)[0]
            if len(cols) == 0:
                continue
            scores = 0.50 * r_div[ti, cols] + 0.30 * r_lvol[ti, cols] + 0.20 * np.nan_to_num(r_ivol[ti, cols], nan=0.5)
        elif variant_kind == "FAME_DIV_FLOOR_1PCT":
            floor = max(0.004, 0.010 + param_shift)
            ok = elig_row & (dist252[ti] >= 0.84) & (c_row >= sma_50d[ti] * 0.98) & (np.nan_to_num(div_y[ti], nan=0.0) >= floor) & (r_lvol[ti] >= 0.50) & (r_amihud[ti] >= 0.45) & (~np.isnan(r_div[ti]))
            cols = np.where(ok)[0]
            if len(cols) == 0:
                continue
            scores = 0.40 * r_div[ti, cols] + 0.25 * r_lvol[ti, cols] + 0.20 * r_dist[ti, cols] + 0.15 * np.nan_to_num(r_ivol[ti, cols], nan=0.5)
        elif variant_kind == "FAME_ADD_SECTOR_RS":
            ok = elig_row & (dist252[ti] >= 0.84) & (c_row >= sma_50d[ti] * 0.98) & (np.nan_to_num(div_y[ti], nan=0.0) > 0.004) & (r_lvol[ti] >= 0.50) & (r_amihud[ti] >= 0.45) & (~np.isnan(r_div[ti]))
            cols = np.where(ok)[0]
            if len(cols) == 0:
                continue
            w_rs = max(0.05, 0.15 + param_shift)
            scores = (0.35 * r_div[ti, cols] + 0.22 * r_lvol[ti, cols] + (0.28 - w_rs) * r_dist[ti, cols] + 0.15 * np.nan_to_num(r_ivol[ti, cols], nan=0.5) + w_rs * np.nan_to_num(r_stsec[ti, cols], nan=0.5))
        elif variant_kind == "CAND_1_PURE_LOWVOL_STABILITY":
            lv_thr = 0.65 + param_shift
            ok = elig_row & (c_row >= sma_50d[ti] * 0.99) & (dist252[ti] >= 0.85) & (r_lvol[ti] >= lv_thr) & (r_ivol[ti] >= 0.55) & (r_amihud[ti] >= 0.50) & (~np.isnan(r_lvol[ti])) & (~np.isnan(r_ivol[ti]))
            cols = np.where(ok)[0]
            if len(cols) == 0:
                continue
            scores = 0.45 * r_lvol[ti, cols] + 0.30 * r_ivol[ti, cols] + 0.15 * np.nan_to_num(r_dist[ti, cols], nan=0.5) + 0.10 * r_amihud[ti, cols]
        elif variant_kind == "CAND_2_VOL_CONTROLLED_MED_MOM":
            vm_thr = 0.65 + param_shift
            ok = elig_row & (c_row > sma_50d[ti]) & (np.nan_to_num(mom120[ti], nan=-1.0) >= 0.08) & (r_vmom[ti] >= vm_thr) & (r_lvol[ti] >= 0.40) & (r_amihud[ti] >= 0.50) & (~np.isnan(r_vmom[ti]))
            cols = np.where(ok)[0]
            if len(cols) == 0:
                continue
            scores = 0.45 * r_vmom[ti, cols] + 0.25 * np.nan_to_num(r_dist[ti, cols], nan=0.5) + 0.20 * r_lvol[ti, cols] + 0.10 * r_amihud[ti, cols]
        elif variant_kind == "CAND_3_SECTOR_REL_STRENGTH":
            rs_thr = 0.65 + param_shift
            ok = elig_row & (c_row > sma_50d[ti]) & (np.nan_to_num(sec_br50[ti], nan=0.0) >= 0.55) & (np.nan_to_num(sec_rs60[ti], nan=-1.0) > 0.0) & (r_stsec[ti] >= rs_thr) & (r_amihud[ti] >= 0.50) & (~np.isnan(r_stsec[ti]))
            cols = [int(c) for c in np.where(ok)[0] if str(sectors[c]) != UNKNOWN_SECTOR_LABEL and "Unclassified" not in str(sectors[c])]
            if not cols:
                continue
            cols = np.array(cols, dtype=int)
            scores = 0.40 * r_stsec[ti, cols] + 0.30 * np.nan_to_num(r_dist[ti, cols], nan=0.5) + 0.15 * np.nan_to_num(r_lvol[ti, cols], nan=0.5) + 0.15 * r_amihud[ti, cols]
        elif variant_kind == "CAND_4_DEEP_DIV_VALUE_QUALITY":
            d_floor = max(0.008, 0.015 + param_shift)
            ok = elig_row & (np.nan_to_num(div_y[ti], nan=0.0) >= d_floor) & (dist252[ti] >= 0.78) & (r_lvol[ti] >= 0.55) & (r_amihud[ti] >= 0.45) & (~np.isnan(r_div[ti]))
            cols = np.where(ok)[0]
            if len(cols) == 0:
                continue
            scores = 0.50 * r_div[ti, cols] + 0.25 * r_lvol[ti, cols] + 0.15 * np.nan_to_num(r_ivol[ti, cols], nan=0.5) + 0.10 * np.nan_to_num(r_dist[ti, cols], nan=0.5)
        else:
            continue

        order = np.argsort(-scores)
        cands: list[dict[str, Any]] = []
        for idx_o in order[:40]:
            col = int(cols[idx_o])
            px = float(c_row[col])
            if np.isnan(px) or px <= 0:
                continue
            atr = float(atr_20d[ti, col]) if not np.isnan(atr_20d[ti, col]) and atr_20d[ti, col] > 0 else px * 0.020
            sl_r = min(0.14, max(0.08, (4.5 * atr) / max(1.0, px)))
            cands.append({"col": col, "score": float(scores[idx_o]), "atr": atr, "stop_ratio": sl_r, "target_ratio": None})
        if cands:
            out[ti] = cands
    return out


def _top_k_excluded_tactical_pnl(trades: list[dict[str, Any]], k: int = 3) -> float:
    pnls = sorted([float(t["total_pnl_with_div_inr"]) for t in trades], reverse=True)
    return round(sum(pnls[k:]) if len(pnls) > k else 0.0, 2)


def run_family_e_component_analysis(
    *,
    ctx: dict[str, Any],
    base_exec_cfg: ExecutionIntelligenceConfig,
    static_policy: AdaptiveRegimePolicy,
    rule_labels: list[str],
    cost_engine: TransactionCostIntelligence,
    inner_end: str,
    outer_start: str,
    end_dt: str,
) -> tuple[list[dict[str, Any]], int]:
    """Section 7: Controlled one-at-a-time Family E component investigation."""
    specs = [
        ("FAME_BASELINE_UNCHANGED", "Unchanged Prompt 3 Family E baseline (0.40 div + 0.25 lowvol + 0.20 dist252 + 0.15 idiovol)", "PIT price + split-adj DIV_YIELD_TTM_PIT", None, {}, 0.0),
        ("FAME_DROP_DIV_YIELD", "Ablation: Remove PIT dividend yield filter & weight to test dividend alpha contribution", "PIT price + low-vol only", "FAME_DROP_DIV_YIELD", {}, 0.0),
        ("FAME_DROP_LOW_VOL", "Ablation: Remove low-volatility filter & weight to test downside/Sharpe contribution", "PIT price + DIV_YIELD_TTM_PIT", "FAME_DROP_LOW_VOL", {}, 0.0),
        ("FAME_DROP_TREND_CONFIRM", "Ablation: Remove 52W-high >= 0.84 & SMA50 trend gate to test value-trap protection", "PIT DIV_YIELD + low-vol only", "FAME_DROP_TREND_CONFIRM", {}, 0.0),
        ("FAME_DIV_FLOOR_1PCT", "Modification: Raise PIT dividend yield floor from 0.4% to 1.0% (perturb 0.8%..1.2%)", "Split-adj DIV_YIELD_TTM_PIT >= 1.0%", "FAME_DIV_FLOOR_1PCT", {}, 0.002),
        ("FAME_ADD_SECTOR_RS", "Modification: Add 15% weight to STOCK_VS_SECTOR_20D relative strength (perturb 10%..20%)", "PIT STOCK_VS_SECTOR_20D + 4-tier sectors", "FAME_ADD_SECTOR_RS", {}, 0.05),
        ("FAME_TIGHTER_SECTOR_CAP_20PCT", "Modification: Tighten max sector exposure from 28% to 20% to reduce sector concentration", "4-tier sector hierarchy cap", None, {"max_sector_exposure_pct": 20.0}, 0.0),
        ("FAME_HYBRID_STOP_F", "Modification: Switch stop model from C_VOL_ADJUSTED_TRAILING to F_HYBRID_STOP_PROFIT_PROTECT", "Causal OHLCV + 20D low + SMA50", None, {"stop_model": "F_HYBRID_STOP_PROFIT_PROTECT"}, 0.0),
    ]

    rows: list[dict[str, Any]] = []
    runs = 0
    base_outer_sharpe = 0.0
    base_outer_ret = 0.0

    for vid, hyp, feat_src, custom_kind, cfg_ovr, p_shift in specs:
        vcfg = _clone_exec_cfg(base_exec_cfg, config_id=vid, **cfg_ovr)
        cmap = None
        if custom_kind is not None:
            cmap = {"STRAT_E_DEFENSIVE_YIELD_LOWVOL": _precompute_custom_candidates(ctx=ctx, strategy_id="STRAT_E_DEFENSIVE_YIELD_LOWVOL", variant_kind=custom_kind, param_shift=0.0)}

        r_in = simulate_m3_1_adaptive_regime_portfolio(ctx=ctx, exec_cfg=vcfg, policy=static_policy, regime_labels=rule_labels, candidate_map_override=cmap, cost_engine=cost_engine, date_start=EVAL_START_DATE, date_end=inner_end)
        r_out = simulate_m3_1_adaptive_regime_portfolio(ctx=ctx, exec_cfg=vcfg, policy=static_policy, regime_labels=rule_labels, candidate_map_override=cmap, cost_engine=cost_engine, date_start=outer_start, date_end=end_dt)
        r_full = simulate_m3_1_adaptive_regime_portfolio(ctx=ctx, exec_cfg=vcfg, policy=static_policy, regime_labels=rule_labels, candidate_map_override=cmap, cost_engine=cost_engine, date_start=EVAL_START_DATE, date_end=end_dt)
        runs += 3

        # Nearby perturbation check for modifications
        perturb_ret = r_full["metrics"]["total_return_pct"]
        if custom_kind is not None and p_shift > 0:
            cmap_p = {"STRAT_E_DEFENSIVE_YIELD_LOWVOL": _precompute_custom_candidates(ctx=ctx, strategy_id="STRAT_E_DEFENSIVE_YIELD_LOWVOL", variant_kind=custom_kind, param_shift=p_shift)}
            rp = simulate_m3_1_adaptive_regime_portfolio(ctx=ctx, exec_cfg=vcfg, policy=static_policy, regime_labels=rule_labels, candidate_map_override=cmap_p, cost_engine=cost_engine, date_start=EVAL_START_DATE, date_end=end_dt)
            runs += 1
            perturb_ret = rp["metrics"]["total_return_pct"]

        if vid == "FAME_BASELINE_UNCHANGED":
            base_outer_sharpe = float(r_out["metrics"]["sharpe_ratio"])
            base_outer_ret = float(r_out["metrics"]["total_return_pct"])
            decision = "REFERENCE_BASELINE"
        elif vid.startswith("FAME_DROP_"):
            decision = "DIAGNOSTIC_ABLATION_ONLY"
        else:
            # Strict retention rule: MUST improve outer Sharpe & outer return cleanly without breaking perturbation
            improves_outer = (float(r_out["metrics"]["sharpe_ratio"]) > base_outer_sharpe + 0.05) and (float(r_out["metrics"]["total_return_pct"]) > base_outer_ret + 1.0)
            stable_perturb = abs(perturb_ret - float(r_full["metrics"]["total_return_pct"])) <= 12.0
            decision = "RETAIN_CANDIDATE" if (improves_outer and stable_perturb) else "REJECT_RETAIN_BASELINE"

        rows.append({
            "variant_id": vid,
            "hypothesis": hyp,
            "pit_feature_source": feat_src,
            "inner_ret_pct": r_in["metrics"]["total_return_pct"],
            "inner_sharpe": r_in["metrics"]["sharpe_ratio"],
            "inner_max_dd_pct": r_in["metrics"]["max_drawdown_pct"],
            "outer_ret_pct": r_out["metrics"]["total_return_pct"],
            "outer_sharpe": r_out["metrics"]["sharpe_ratio"],
            "outer_max_dd_pct": r_out["metrics"]["max_drawdown_pct"],
            "full_ret_pct": r_full["metrics"]["total_return_pct"],
            "full_cagr_pct": r_full["metrics"]["cagr_pct"],
            "full_sharpe": r_full["metrics"]["sharpe_ratio"],
            "full_max_dd_pct": r_full["metrics"]["max_drawdown_pct"],
            "full_pf": r_full["metrics"]["profit_factor"],
            "full_trades": r_full["metrics"]["num_trades"],
            "full_turnover_x": r_full["metrics"]["total_turnover_x"],
            "full_cost_inr": r_full["metrics"]["total_cost_inr"],
            "perturb_full_ret_pct": perturb_ret,
            "decision": decision,
        })
    return rows, runs


def run_new_strategy_research(
    *,
    ctx: dict[str, Any],
    base_exec_cfg: ExecutionIntelligenceConfig,
    static_policy: AdaptiveRegimePolicy,
    rule_labels: list[str],
    cost_engine: TransactionCostIntelligence,
    fame_full_sim: dict[str, Any],
    nifty_full_ret: float,
    inner_end: str,
    outer_start: str,
    end_dt: str,
) -> tuple[list[dict[str, Any]], int]:
    """Section 8: Controlled research of 4 genuinely distinct PIT strategies + formal Family A retest."""
    candidates = [
        (
            "CAND_1_PURE_LOWVOL_STABILITY",
            "Defensive Low-Volatility & Idiosyncratic Stability (No Dividend Filter)",
            "Low-beta/low-idio-vol compounders deliver superior risk-adjusted compounding across cycles",
            "Works in sideways/late-cycle/mild bear; lags speculative high-beta rallies",
            "Daily OHLCV + PIT eligibility (no earnings/ROE DB data available)",
            "CAND_1_PURE_LOWVOL_STABILITY",
            0.05,
        ),
        (
            "CAND_2_VOL_CONTROLLED_MED_MOM",
            "Medium-Term Volatility-Controlled Momentum (120D-skip-10 + LowVol Gate)",
            "Intermediate momentum with volatility control captures persistent trends while avoiding high-beta whipsaws",
            "Works in sustained directional trends; fails during sharp V-reversals & chop",
            "Daily OHLCV + PIT eligibility",
            "CAND_2_VOL_CONTROLLED_MED_MOM",
            0.05,
        ),
        (
            "CAND_3_SECTOR_REL_STRENGTH",
            "4-Tier PIT Sector-Relative Strength Leader Rotation",
            "Capital rotates persistently into leading NSE macro-sectors with broad internal participation",
            "Works during clear sector leadership cycles; fails during rapid factor rotation",
            "Daily OHLCV + 4-tier PIT sector hierarchy (31.6% UNKNOWN excluded)",
            "CAND_3_SECTOR_REL_STRENGTH",
            0.05,
        ),
        (
            "CAND_4_DEEP_DIV_VALUE_QUALITY",
            "High-Yield PIT Dividend Quality & Low-Drawdown Value (>=1.5% TTM Yield)",
            "Cash-returning deep value equities cushion drawdowns via real cash dividend carry",
            "Works in value/rate-normalization cycles; lags growth-heavy narrow rallies",
            "Daily OHLCV + split-adjusted PIT corporate actions dividends",
            "CAND_4_DEEP_DIV_VALUE_QUALITY",
            0.003,
        ),
        (
            "CAND_5_FAMILY_A_ADMISSION_RETEST",
            "Formal Re-Evaluation of Suspended Family A (CS Momentum) Under Prompt 3 Execution",
            "Test objectively whether Prompt 3 execution intelligence repairs Family A's negative alpha",
            "Works in strong broad momentum; historically failed post-2024",
            "Existing M3.0 STRAT_A_CS_MOMENTUM PIT signals",
            None,
            0.0,
        ),
    ]

    fame_rets = np.array([float(d["ret"]) for d in fame_full_sim["daily_navs"][1:]], dtype=np.float64)
    fame_full_sh = float(fame_full_sim["metrics"]["sharpe_ratio"])
    rows: list[dict[str, Any]] = []
    runs = 0

    for cid, cname, rationale, reg_fit, data_req, vkind, p_shift in candidates:
        if vkind is not None:
            cand_map_solo = {cid: _precompute_custom_candidates(ctx=ctx, strategy_id=cid, variant_kind=vkind, param_shift=0.0)}
            cand_map_pert = {cid: _precompute_custom_candidates(ctx=ctx, strategy_id=cid, variant_kind=vkind, param_shift=p_shift)}
            sid_use = cid
        else:
            cand_map_solo = None
            cand_map_pert = None
            sid_use = "STRAT_A_CS_MOMENTUM"

        cfg_solo = _clone_exec_cfg(
            base_exec_cfg,
            config_id=cid,
            config_name=cname,
            active_sleeves=(sid_use,),
            sleeve_weights=((sid_use, 1.0),),
        )
        r_in = simulate_m3_1_adaptive_regime_portfolio(ctx=ctx, exec_cfg=cfg_solo, policy=static_policy, regime_labels=rule_labels, candidate_map_override=cand_map_solo, cost_engine=cost_engine, date_start=EVAL_START_DATE, date_end=inner_end)
        r_out = simulate_m3_1_adaptive_regime_portfolio(ctx=ctx, exec_cfg=cfg_solo, policy=static_policy, regime_labels=rule_labels, candidate_map_override=cand_map_solo, cost_engine=cost_engine, date_start=outer_start, date_end=end_dt)
        r_full = simulate_m3_1_adaptive_regime_portfolio(ctx=ctx, exec_cfg=cfg_solo, policy=static_policy, regime_labels=rule_labels, candidate_map_override=cand_map_solo, cost_engine=cost_engine, date_start=EVAL_START_DATE, date_end=end_dt)
        runs += 3

        # 50/50 Tactical Blend with Family E inside 60/30/10 to test genuine diversification benefit!
        blend_cmap = {"STRAT_E_DEFENSIVE_YIELD_LOWVOL": ctx["precomputed_cands"]["STRAT_E_DEFENSIVE_YIELD_LOWVOL"]}
        if vkind is not None and cand_map_solo is not None:
            blend_cmap[cid] = cand_map_solo[cid]
        else:
            blend_cmap["STRAT_A_CS_MOMENTUM"] = ctx["precomputed_cands"]["STRAT_A_CS_MOMENTUM"]
        cfg_blend = _clone_exec_cfg(
            base_exec_cfg,
            config_id=f"BLEND_FAME_{cid}",
            active_sleeves=("STRAT_E_DEFENSIVE_YIELD_LOWVOL", sid_use),
            sleeve_weights=(("STRAT_E_DEFENSIVE_YIELD_LOWVOL", 0.50), (sid_use, 0.50)),
        )
        r_blend = simulate_m3_1_adaptive_regime_portfolio(ctx=ctx, exec_cfg=cfg_blend, policy=static_policy, regime_labels=rule_labels, candidate_map_override=blend_cmap, cost_engine=cost_engine, date_start=EVAL_START_DATE, date_end=end_dt)
        runs += 1

        # Perturbation stability check
        pert_ret = r_full["metrics"]["total_return_pct"]
        if cand_map_pert is not None:
            r_p = simulate_m3_1_adaptive_regime_portfolio(ctx=ctx, exec_cfg=cfg_solo, policy=static_policy, regime_labels=rule_labels, candidate_map_override=cand_map_pert, cost_engine=cost_engine, date_start=EVAL_START_DATE, date_end=end_dt)
            runs += 1
            pert_ret = r_p["metrics"]["total_return_pct"]

        # Daily return correlation with Family E portfolio (measuring tactical diversification)
        c_rets = np.array([float(d["ret"]) for d in r_full["daily_navs"][1:]], dtype=np.float64)
        corr_fame = float(np.corrcoef(fame_rets[:len(c_rets)], c_rets)[0, 1]) if len(c_rets) > 20 else 1.0
        ex_top3_pnl = _top_k_excluded_tactical_pnl(r_full["trades"], k=3)

        # Evaluate ALL 6 Section 8 Admission Gates strictly!
        mf = r_full["metrics"]
        mo = r_out["metrics"]
        mb = r_blend["metrics"]
        g1_edge = (float(mf["total_return_pct"]) > nifty_full_ret + 2.0) or (float(mb["sharpe_ratio"]) > fame_full_sh + 0.03 and corr_fame < 0.90)
        g2_chrono = (float(r_in["metrics"]["sharpe_ratio"]) >= 0.60) and (float(mo["sharpe_ratio"]) >= 0.05) and (float(mo["total_return_pct"]) > 0.0)
        g3_non_outlier = ex_top3_pnl > 0.0
        g4_turn_dd = (float(mf["total_turnover_x"]) <= 30.0) and (float(mf["max_drawdown_pct"]) <= 16.0)
        g5_perturb = abs(float(pert_ret) - float(mf["total_return_pct"])) <= 12.0
        g6_pit_acct = abs(float(mf["reconciliation_diff_inr"])) <= 0.50

        passed_all = g1_edge and g2_chrono and g3_non_outlier and g4_turn_dd and g5_perturb and g6_pit_acct
        # Also verify blend actually beats standalone Family E Sharpe (otherwise don't dilute Family E!)
        improves_blend = float(mb["sharpe_ratio"]) > fame_full_sh and float(mb["max_drawdown_pct"]) <= float(fame_full_sim["metrics"]["max_drawdown_pct"]) + 0.5
        admitted = passed_all and improves_blend

        fail_reasons: list[str] = []
        if not g1_edge:
            fail_reasons.append("NO_BENCHMARK_OR_DIV_EDGE")
        if not g2_chrono:
            fail_reasons.append("FAILED_CHRONO_OUTER")
        if not g3_non_outlier:
            fail_reasons.append("OUTLIER_DEPENDENT_EX_TOP3_NEG")
        if not g4_turn_dd:
            fail_reasons.append("EXCESS_TURNOVER_OR_DD")
        if not g5_perturb:
            fail_reasons.append("UNSTABLE_PERTURBATION")
        if not improves_blend:
            fail_reasons.append("DILUTES_FAMILY_E_BLEND")

        rows.append({
            "candidate_id": cid,
            "candidate_name": cname,
            "economic_rationale": rationale,
            "regime_profile": reg_fit,
            "data_requirements": data_req,
            "inner_ret_pct": r_in["metrics"]["total_return_pct"],
            "inner_sharpe": r_in["metrics"]["sharpe_ratio"],
            "outer_ret_pct": mo["total_return_pct"],
            "outer_sharpe": mo["sharpe_ratio"],
            "full_ret_pct": mf["total_return_pct"],
            "full_cagr_pct": mf["cagr_pct"],
            "full_max_dd_pct": mf["max_drawdown_pct"],
            "full_sharpe": mf["sharpe_ratio"],
            "full_pf": mf["profit_factor"],
            "full_trades": mf["num_trades"],
            "full_turnover_x": mf["total_turnover_x"],
            "full_cost_inr": mf["total_cost_inr"],
            "ex_top3_tactical_pnl_inr": ex_top3_pnl,
            "corr_with_family_e": round(corr_fame, 3),
            "blend_50_50_full_ret_pct": mb["total_return_pct"],
            "blend_50_50_sharpe": mb["sharpe_ratio"],
            "blend_50_50_max_dd_pct": mb["max_drawdown_pct"],
            "perturb_full_ret_pct": pert_ret,
            "admission_decision": "ADMITTED" if admitted else "REJECTED",
            "rejection_reasons": "NONE" if admitted else ";".join(fail_reasons),
        })
    return rows, runs
