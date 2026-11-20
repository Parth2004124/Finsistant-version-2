"""Moreshwar 3.1 (Prompt 2/5) — Intelligent Three-Pillar Capital Allocation & Portfolio Engine.

Transforms Moreshwar from a collection of standalone stock-selection strategies into a
capital-efficient, cost-aware, zero-leverage Indian equity portfolio engine built on three
independently managed capital pillars:

1. PASSIVE CORE (`NIFTYBEES` real NSE exchange-traded accumulating ETF):
   - Continuous strategic Indian equity market exposure with threshold-based rebalancing
     (`core_rebal_band_pct`) and zero dividend double-counting.
2. TACTICAL STRATEGY SLEEVES (reusing validated Moreshwar 3.0 Family E & Family A generators
   on split-adjusted `DIV_YIELD_TTM_PIT` and 4-tier sector classification):
   - Event-driven replenishment after exits (eliminating the 20-trading-day slot starvation),
     cost-aware minimum lot sizing & rotation hurdles (crushing flat `INR 15.93` CDSL DP fee drag),
     and verified ex-dividend cash crediting + ex-dividend stop-loss protection.
3. LIQUIDITY & RISK RESERVE (`LIQUIDBEES` real NSE overnight liquid ETF):
   - Operational dry powder, defensive capital in Bear/Stress regimes, and automatic sweep
     of surplus cash so capital is never unintentionally left at 0% yield.

Preserves Moreshwar 3.0 (`finsistant_v2/m3/`) completely frozen and untouched.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import numpy as np

from finsistant_v2.m3.execution_cost_engine import (
    HistoricalTariffSchedule,
    calculate_historical_indian_costs,
)
from finsistant_v2.m3.portfolio_construction import (
    PortfolioConstructionConfig,
    _causal_pairwise_corr_60d,
    compute_euler_risk_contributions,
    simulate_shared_capital_portfolio_m3,
)
from finsistant_v2.m3.regime_intelligence import (
    classify_rule_based_9state,
    compute_causal_market_features,
)
from finsistant_v2.m3.strategy_base import (
    EVAL_START_DATE,
    INITIAL_CAPITAL_M3,
    OOS_START_DATE,
    RISK_FREE_ANNUAL_M3,
)
from finsistant_v2.m3.strategy_families import (
    build_candidate_generator,
    get_strategy_family_configs,
)

from .dividend_accounting import (
    LIQUID_BEES_DAILY_YIELD,
    apply_ex_dividend_accounting_and_stop_adjustment,
    load_real_nse_etf_bars,
    prepare_m3_1_signal_workspace,
)
from .sector_hierarchy import UNKNOWN_SECTOR_LABEL, build_4tier_sector_classification

TRAIN_END_DATE = "2023-12-31"
VAL_START_DATE = "2024-01-01"
VAL_END_DATE = "2025-03-31"
# Untouched OOS remains identical to Moreshwar 3.0: 2025-04-01 -> 2026-11-19


@dataclass(frozen=True)
class CapitalAllocationConfig:
    """Configurable three-pillar Moreshwar 3.1 capital allocation & risk architecture."""

    config_id: str
    config_name: str
    # Base strategic weights (must sum strictly to 100.0)
    core_target_pct: float = 55.0
    tactical_target_pct: float = 35.0
    liquid_target_pct: float = 10.0
    # Allocation policy mode:
    # "FIXED" | "VOLATILITY_TARGETED" | "DRAWDOWN_AWARE" | "REGIME_AWARE" | "RISK_BUDGETED" | "CONFIDENCE_WEIGHTED"
    allocation_mode: str = "REGIME_AWARE"
    # Active tactical sleeves included (reusing Moreshwar 3.0 validated strategy IDs)
    active_sleeves: tuple[str, ...] = (
        "STRAT_E_DEFENSIVE_YIELD_LOWVOL",
        "STRAT_A_CS_MOMENTUM",
    )
    sleeve_weights: tuple[tuple[str, float], ...] = (
        ("STRAT_E_DEFENSIVE_YIELD_LOWVOL", 0.60),
        ("STRAT_A_CS_MOMENTUM", 0.40),
    )
    # Event-driven deployment & anti-starvation controls
    event_driven_replenishment: bool = True
    periodic_review_interval_days: int = 20
    replenish_min_gap_days: int = 2
    sweep_unused_tactical_to_core: bool = True
    # Position sizing & cost-aware hurdles
    max_tactical_positions: int = 6
    max_single_stock_pct: float = 15.0
    min_tactical_lot_inr: float = 6_500.0
    min_rotation_score_upgrade: float = 0.16

    min_expected_net_alpha_bps: float = 120.0  # minimum expected alpha above round-trip costs
    core_rebal_band_pct: float = 5.5  # threshold band to avoid micro-rebalance churn
    # 4-tier sector / industry / unknown concentration limits
    max_sector_exposure_pct: float = 28.0
    max_sector_positions: int = 2
    max_industry_exposure_pct: float = 20.0
    max_unknown_sector_exposure_pct: float = 22.0
    max_pairwise_corr: float = 0.72
    # Portfolio risk management
    target_portfolio_vol_pct: float = 13.5
    max_daily_vol_participation: float = 0.01
    settlement_lag_days: int = 1  # NSE T+1 settlement
    slippage_bps: float = 7.5
    # Tactical exit discipline (Prompt 1 fix: remove tight +0.65% BE ratchet on M3.1)
    hold_max_days: int = 120
    trailing_atr_mult: float = 5.0
    disable_tight_breakeven_stop: bool = True
    cooldown_days: int = 10
    enable_dividend_accounting: bool = True
    use_4tier_sector_hierarchy: bool = True


def precompute_m3_1_research_context(
    panel: dict[str, Any],
    *,
    db_path: Any | None = None,
) -> dict[str, Any]:
    """Precompute 4-tier sectors, split-adjusted signal workspace, real NSE ETFs, regimes, and causal candidates once."""
    sec_bundle = (
        build_4tier_sector_classification(panel, db_path=db_path)
        if db_path is not None
        else build_4tier_sector_classification(panel)
    )
    ws_m31 = prepare_m3_1_signal_workspace(panel, resolved_sectors=sec_bundle["sectors"])
    etf_bars = (
        load_real_nse_etf_bars(panel["dates"], db_path=db_path, nifty50_close=panel["nifty50_close"])
        if db_path is not None
        else load_real_nse_etf_bars(panel["dates"], nifty50_close=panel["nifty50_close"])
    )

    feats = compute_causal_market_features(panel)
    fine9, rule5 = classify_rule_based_9state(feats, confirm_days=3)

    # Precompute daily return matrix for causal 60D correlation & Euler risk contributions
    adj_c64 = panel["adj_close"].astype(np.float64)
    daily_ret = np.full(adj_c64.shape, np.nan, dtype=np.float64)
    daily_ret[1:] = np.where(
        (adj_c64[1:] > 0) & (adj_c64[:-1] > 0),
        (adj_c64[1:] / np.maximum(1e-6, adj_c64[:-1])) - 1.0,
        np.nan,
    )

    # Precompute causal candidates per strategy & date across all valid dates (so 40+ portfolio runs execute fast!)
    work_panel = dict(panel)
    work_panel["sectors"] = sec_bundle["sectors"]
    work_panel["industries"] = sec_bundle["industries"]
    work_panel["is_unknown_sector"] = sec_bundle["is_unknown"]

    strat_cfgs = {c.strategy_id: c for c in get_strategy_family_configs()}
    cand_fns = {sid: build_candidate_generator(sc, ws_m31) for sid, sc in strat_cfgs.items()}

    dates = [str(d) for d in panel["dates"]]
    t_len = len(dates)
    start_pre = next((i for i, d in enumerate(dates) if d >= EVAL_START_DATE), 0)
    precomputed_cands: dict[str, dict[int, list[dict[str, Any]]]] = {sid: {} for sid in strat_cfgs}
    for ti in range(max(0, start_pre - 5), t_len):
        for sid, fn in cand_fns.items():
            # Strictly causal: uses only panel/ws at index <= ti; open_pos is unused by candidate_fn
            c_list = fn(ti, work_panel, {})
            if c_list:
                precomputed_cands[sid][ti] = c_list[:25]

    return {
        "panel": work_panel,
        "raw_panel": panel,
        "ws_m31": ws_m31,
        "sector_bundle": sec_bundle,
        "etf_bars": etf_bars,
        "regime_features": feats,
        "regime_fine9": fine9,
        "regime_rule5": rule5,
        "daily_ret": daily_ret,
        "strat_cfgs": strat_cfgs,
        "precomputed_cands": precomputed_cands,
    }


def _compute_dynamic_pillar_targets(
    *,
    cfg: CapitalAllocationConfig,
    cur_regime: str,
    cur_dd_pct: float,
    port_vol_20d_ann: float,
    core_vol_20d_ann: float,
    tact_vol_20d_ann: float,
    avg_signal_score: float,
    signal_count: int,
) -> tuple[float, float, float]:
    """Compute strictly causal target weights `(core_pct, tactical_pct, liquid_pct)` summing to 100.0%."""
    base_c = float(cfg.core_target_pct)
    base_t = float(cfg.tactical_target_pct)
    base_l = float(cfg.liquid_target_pct)
    tot_base = max(1e-6, base_c + base_t + base_l)
    base_c, base_t, base_l = (base_c / tot_base) * 100.0, (base_t / tot_base) * 100.0, (base_l / tot_base) * 100.0

    mode = cfg.allocation_mode.upper().strip()

    if mode == "FIXED":
        return round(base_c, 2), round(base_t, 2), round(base_l, 2)

    if mode == "VOLATILITY_TARGETED":
        # Never levers > 100% equity; scales equity down to LIQUIDBEES when portfolio vol > target_portfolio_vol_pct
        eq_base = base_c + base_t
        if port_vol_20d_ann > cfg.target_portfolio_vol_pct and port_vol_20d_ann > 1.0:
            scale = max(0.55, min(1.0, cfg.target_portfolio_vol_pct / port_vol_20d_ann))
        else:
            scale = 1.0
        eq_new = eq_base * scale
        ratio_c = base_c / max(1e-6, eq_base)
        c_new = eq_new * ratio_c
        t_new = eq_new * (1.0 - ratio_c)
        l_new = 100.0 - c_new - t_new
        return round(c_new, 2), round(t_new, 2), round(l_new, 2)

    if mode == "DRAWDOWN_AWARE":
        if cur_dd_pct >= 15.0:
            eq_scale = 0.60
        elif cur_dd_pct >= 11.0:
            eq_scale = 0.78
        elif cur_dd_pct >= 7.5:
            eq_scale = 0.90
        else:
            eq_scale = 1.0
        eq_base = base_c + base_t
        eq_new = eq_base * eq_scale
        ratio_c = base_c / max(1e-6, eq_base)
        c_new = eq_new * ratio_c
        t_new = eq_new * (1.0 - ratio_c)
        l_new = 100.0 - c_new - t_new
        return round(c_new, 2), round(t_new, 2), round(l_new, 2)

    if mode == "REGIME_AWARE":
        # Causal 5-state regime modulation anchored around base (core, tactical, liquid) targets
        # Preserves continuous bull equity participation while adding liquid buffer in Stress Bear
        regime_mults = {
            "STRONG_BULL": (1.08, 1.05, 0.45),
            "WEAK_BULL": (1.02, 1.02, 0.85),
            "SIDEWAYS_CHOP": (0.95, 0.92, 1.35),
            "WEAK_BEAR": (0.78, 0.65, 2.10),
            "STRESS_BEAR": (0.60, 0.45, 2.80),
        }
        mc, mt, ml = regime_mults.get(cur_regime, (1.0, 1.0, 1.0))
        # Ensure liquid floor scales sensibly even when base_l is small
        raw_c = base_c * mc
        raw_t = base_t * mt
        raw_l = max(base_l * ml, 20.0 if cur_regime == "WEAK_BEAR" else (30.0 if cur_regime == "STRESS_BEAR" else base_l * ml))
        # Apply gentle drawdown protection on top
        if cur_dd_pct >= 14.0:
            raw_t *= 0.65
            raw_l += 10.0
        s = max(1e-6, raw_c + raw_t + raw_l)
        return round((raw_c / s) * 100.0, 2), round((raw_t / s) * 100.0, 2), round((raw_l / s) * 100.0, 2)

    if mode == "RISK_BUDGETED":
        # Inverse-volatility risk parity split between Core (`NIFTYBEES`) and Tactical sleeves
        eq_base = base_c + base_t
        v_c = max(0.08, core_vol_20d_ann / 100.0)
        v_t = max(0.08, tact_vol_20d_ann / 100.0)
        inv_c = (base_c / max(1.0, eq_base)) / v_c
        inv_t = (base_t / max(1.0, eq_base)) / v_t
        # Gentle vol ceiling if combined vol exceeds target
        vol_scale = min(1.0, cfg.target_portfolio_vol_pct / max(1.0, port_vol_20d_ann)) if port_vol_20d_ann > cfg.target_portfolio_vol_pct else 1.0
        eq_new = eq_base * max(0.65, vol_scale)
        c_new = eq_new * (inv_c / max(1e-6, inv_c + inv_t))
        t_new = eq_new * (inv_t / max(1e-6, inv_c + inv_t))
        l_new = 100.0 - c_new - t_new
        return round(c_new, 2), round(t_new, 2), round(l_new, 2)

    if mode == "CONFIDENCE_WEIGHTED":
        # Scale tactical weight by causal signal availability & validated FDR factor confidence;
        # redirect surplus tactical weight into Passive Core (Bull/Sideways) or Liquid Reserve (Bear)
        if signal_count == 0:
            conf_mult = 0.0
        else:
            # avg_signal_score is in [0, 1]; scale smoothly from 0.5x to 1.15x
            conf_mult = float(np.clip(0.55 + 0.70 * (avg_signal_score - 0.35), 0.35, 1.15))
        t_new = min(80.0, base_t * conf_mult)
        delta_t = base_t - t_new
        if cur_regime in ("STRONG_BULL", "WEAK_BULL", "SIDEWAYS_CHOP"):
            c_new = max(0.0, base_c + delta_t)
            l_new = max(0.0, 100.0 - c_new - t_new)
        else:
            c_new = max(0.0, base_c * 0.80)
            l_new = max(0.0, 100.0 - c_new - t_new)
        s = max(1e-6, c_new + t_new + l_new)
        return round((c_new / s) * 100.0, 2), round((t_new / s) * 100.0, 2), round((l_new / s) * 100.0, 2)

    return round(base_c, 2), round(base_t, 2), round(base_l, 2)


def simulate_m3_1_capital_allocation(
    *,
    ctx: dict[str, Any],
    cfg: CapitalAllocationConfig,
    date_start: str = EVAL_START_DATE,
    date_end: str | None = None,
    initial_capital: float = INITIAL_CAPITAL_M3,
) -> dict[str, Any]:
    """Simulate the three-pillar Moreshwar 3.1 Intelligent Capital Allocation Engine strictly causally."""
    panel = ctx["panel"] if cfg.use_4tier_sector_hierarchy else ctx["raw_panel"]
    ws = ctx["ws_m31"]
    etf_bars = ctx["etf_bars"]
    regimes = ctx["regime_rule5"]
    daily_ret = ctx["daily_ret"]
    pre_cands = ctx["precomputed_cands"]
    div_per_adj = ws["div_per_adj_share"]
    sma_50d = ws["sma_50d"]

    dates = [str(d) for d in panel["dates"]]
    end_dt = date_end or dates[-1]
    t_indices = [i for i, d in enumerate(dates) if date_start <= d <= end_dt]
    if len(t_indices) < 20:
        return {"config": asdict(cfg), "metrics": {}, "daily_navs": [], "trades": []}

    open_p = panel["adj_open"]
    high_p = panel["adj_high"]
    low_p = panel["adj_low"]
    close_p = panel["adj_close"]
    vol_p = panel["volume"]
    sectors = panel["sectors"]
    industries = panel["industries"]
    is_unknown_arr = panel.get("is_unknown_sector", np.zeros(len(sectors), dtype=bool))
    sec_ids = panel["sec_ids"]
    cur_syms = panel["current_symbols"]
    n50_c = panel["nifty50_close"]

    nb_open = etf_bars["NIFTYBEES"]["open"]
    nb_close = etf_bars["NIFTYBEES"]["close"]
    lb_open = etf_bars["LIQUIDBEES"]["open"]
    lb_close = etf_bars["LIQUIDBEES"]["close"]

    tariff = HistoricalTariffSchedule()
    sleeve_w_map = dict(cfg.sleeve_weights)

    # Four-ledger state
    settled_cash = float(initial_capital)
    unsettled_queue: list[tuple[int, float]] = []  # (settle_ti, amount_inr)
    core_qty = 0
    core_cost_basis = 0.0
    liquid_units = 0.0  # Liquid BeES units (NSE allows fractional dividend reinvestment units!)
    liquid_cost_basis = 0.0
    open_pos: dict[int, dict[str, Any]] = {}

    pending_tact_sells: set[int] = set()
    pending_tact_buys: list[dict[str, Any]] = []
    pending_core_target_inr: float | None = None
    pending_liquid_sweep: bool = False
    cooldown_until: dict[int, int] = {}
    dp_pairs: set[tuple[str, str]] = set()

    closed_trades: list[dict[str, Any]] = []
    dividend_events_log: list[dict[str, Any]] = []
    daily_navs: list[dict[str, Any]] = []
    replenishment_events: list[dict[str, Any]] = []
    sector_exposure_samples: list[dict[str, float]] = []
    euler_rc_samples: list[dict[str, Any]] = []

    rejection_diagnostics = {
        "SECTOR_CAP_BLOCKED": 0,
        "INDUSTRY_CAP_BLOCKED": 0,
        "UNKNOWN_SECTOR_CAP_BLOCKED": 0,
        "CORRELATION_CAP_BLOCKED": 0,
        "MAX_POSITIONS_BLOCKED": 0,
        "COST_ALPHA_HURDLE_BLOCKED": 0,
        "ROTATION_UPGRADE_HURDLE_SKIPPED": 0,
        "MIN_LOT_SIZE_BLOCKED": 0,
        "LIQUIDITY_PARTIAL_FILLS": 0,
        "EX_DIV_STOP_PROTECTIONS": 0,
    }

    total_cost_tactical = 0.0
    total_cost_core = 0.0
    total_cost_liquid = 0.0
    total_dp_charges = 0.0
    total_stt = 0.0
    total_slippage = 0.0
    turnover_tactical = 0.0
    turnover_core = 0.0
    turnover_liquid = 0.0
    cum_dividends_inr = 0.0
    cum_liquid_yield_inr = 0.0
    realized_core_pnl_inr = 0.0
    realized_liquid_pnl_inr = 0.0


    port_ret_history: list[float] = []
    core_ret_history: list[float] = []
    tact_ret_history: list[float] = []
    n50_ret_history: list[float] = []

    last_replenish_step = -999
    vacated_slot_flag = False
    t_last = t_indices[-1]
    peak_nav = float(initial_capital)
    prev_nav = float(initial_capital)
    prev_core_px = float(nb_close[t_indices[0]])
    prev_tact_mkt = 0.0

    # Target weights initialized on day 0
    cur_tgt_c, cur_tgt_t, cur_tgt_l = cfg.core_target_pct, cfg.tactical_target_pct, cfg.liquid_target_pct

    for step_k, ti in enumerate(t_indices):
        dt = dates[ti]
        prev_ti = ti - 1
        is_last = (ti == t_last)

        # 0A. Release T+1 settled cash
        still_unsettled: list[tuple[int, float]] = []
        for s_ti, amt in unsettled_queue:
            if ti >= s_ti:
                settled_cash += amt
            else:
                still_unsettled.append((s_ti, amt))
        unsettled_queue = still_unsettled

        # 0B. Accrue overnight Liquid BeES daily fractional unit yield (net of TER) on held Liquid BeES
        lb_px_o = float(lb_open[ti]) if not np.isnan(lb_open[ti]) and lb_open[ti] > 0 else 1000.0
        lb_px_c = float(lb_close[ti]) if not np.isnan(lb_close[ti]) and lb_close[ti] > 0 else 1000.0
        if step_k > 0 and liquid_units > 1e-6:
            liq_yield_today = liquid_units * lb_px_c * LIQUID_BEES_DAILY_YIELD
            liquid_units += liq_yield_today / max(1.0, lb_px_c)
            cum_liquid_yield_inr += liq_yield_today

        # 0C. Apply Ex-Dividend Cash Crediting & Stop-Loss Protection on Open Tactical Positions BEFORE checking stops!
        if cfg.enable_dividend_accounting and open_pos:
            for col, pos in open_pos.items():
                d_adj = float(div_per_adj[ti, col])
                if d_adj > 1e-8 and ti > int(pos["entry_ti"]):
                    div_ev = apply_ex_dividend_accounting_and_stop_adjustment(
                        pos=pos, ti=ti, dt=dt, div_per_adj_share=d_adj
                    )
                    if div_ev["credited_cash_inr"] > 0:
                        settled_cash += div_ev["credited_cash_inr"]
                        cum_dividends_inr += div_ev["credited_cash_inr"]
                        dividend_events_log.append(div_ev)
                        if div_ev["adjusted_sl"]:
                            rejection_diagnostics["EX_DIV_STOP_PROTECTIONS"] += 1

        # 1. Execute Pending Tactical Rebalance Sells at Day `ti` Open
        for col in list(pending_tact_sells):
            if col not in open_pos or not panel["is_valid_eq"][ti, col]:
                continue
            pos = open_pos[col]
            raw_open = float(open_p[ti, col])
            bar_vol = float(vol_p[ti, col])
            if np.isnan(raw_open) or raw_open <= 0 or bar_vol <= 0:
                continue
            raw_sell = pos["qty"] * raw_open
            sym = str(cur_syms[col])
            sc = calculate_historical_indian_costs(
                product_type="DELIVERY",
                side="SELL",
                raw_notional=raw_sell,
                trade_date=dt,
                slippage_bps=cfg.slippage_bps,
                dp_charged_today=((dt, sym) in dp_pairs),
                tariff=tariff,
            )
            if sc["dp_charge"] > 0:
                dp_pairs.add((dt, sym))
            net_proc = sc["exec_notional"] - sc["regulatory_total"]
            if cfg.settlement_lag_days <= 0:
                settled_cash += net_proc
            else:
                unsettled_queue.append((ti + cfg.settlement_lag_days, net_proc))
            c_leg = sc["regulatory_total"] + sc["slippage_cost"]
            total_cost_tactical += c_leg
            total_dp_charges += sc["dp_charge"]
            total_stt += sc["stt"]
            total_slippage += sc["slippage_cost"]
            turnover_tactical += sc["exec_notional"]
            pnl_price = net_proc - pos["cost_basis"]
            pnl_total = pnl_price + pos["dividends_received_inr"]
            closed_trades.append({
                "sec_id": str(sec_ids[col]),
                "symbol": sym,
                "sector": str(sectors[col]),
                "industry": str(industries[col]),
                "strategy_id": pos["strategy_id"],
                "entry_date": pos["entry_date"],
                "exit_date": dt,
                "holding_days": pos["bars_held"],
                "buy_notional_inr": round(pos["buy_exec_notional"], 2),
                "sell_notional_inr": round(sc["exec_notional"], 2),
                "trade_cost_inr": round(pos["buy_total_cost"] + c_leg, 2),
                "dividends_received_inr": round(pos["dividends_received_inr"], 2),
                "pnl_amount": round(pnl_price, 2),
                "total_pnl_with_div_inr": round(pnl_total, 2),
                "net_return_pct": round((pnl_total / max(1e-6, pos["cost_basis"])) * 100.0, 2),
                "exit_reason": "REBALANCE_ROTATE",
            })
            del open_pos[col]
        pending_tact_sells.clear()

        # 2. If Cash Needed for Pending Tactical Buys or Core Rebalance, Liquidate Required LIQUIDBEES First (T+0 instant liquidity reserve)
        nb_px_o = float(nb_open[ti]) if not np.isnan(nb_open[ti]) and nb_open[ti] > 0 else prev_core_px
        nb_px_c = float(nb_close[ti]) if not np.isnan(nb_close[ti]) and nb_close[ti] > 0 else nb_px_o

        def _redeem_liquid_bees(needed_inr: float) -> None:
            nonlocal liquid_units, liquid_cost_basis, realized_liquid_pnl_inr, settled_cash, total_cost_liquid, total_dp_charges, total_stt, total_slippage, turnover_liquid
            if needed_inr <= 500.0 or liquid_units * lb_px_o < 500.0:
                return
            sell_amt = min(needed_inr * 1.005, liquid_units * lb_px_o)
            units_sell = min(liquid_units, sell_amt / max(1.0, lb_px_o))
            # Liquid ETFs on NSE have 0% STT (debt ETF) and ~1 bps bid-ask spread, 0 brokerage; CDSL DP INR 15.93 once/day
            dp_used = (dt, "LIQUIDBEES") in dp_pairs
            dp_fee = 0.0 if dp_used else 15.93
            if dp_fee > 0:
                dp_pairs.add((dt, "LIQUIDBEES"))
            slip = sell_amt * 0.0001
            exch = sell_amt * 0.0000297 * 1.18
            tot_c = dp_fee + slip + exch
            net_cash = max(0.0, sell_amt - tot_c)
            avg_lcb = liquid_cost_basis / max(1e-9, liquid_units)
            sold_cb = avg_lcb * units_sell
            realized_liquid_pnl_inr += net_cash - sold_cb
            liquid_cost_basis = max(0.0, liquid_cost_basis - sold_cb)
            liquid_units = max(0.0, liquid_units - units_sell)
            settled_cash += net_cash
            total_cost_liquid += tot_c
            total_dp_charges += dp_fee
            total_slippage += slip
            turnover_liquid += sell_amt

        def _buy_liquid_bees(surplus_inr: float) -> None:
            nonlocal liquid_units, liquid_cost_basis, settled_cash, total_cost_liquid, total_slippage, turnover_liquid
            if surplus_inr < 2000.0:
                return
            # Liquid BeES buy: ~1 bps spread + stamp duty 0.015% + exchange
            cost_rate = 0.0001 + 0.00015 + 0.000035
            buy_amt = surplus_inr / (1.0 + cost_rate)
            cost_inr = surplus_inr - buy_amt
            settled_cash -= surplus_inr
            liquid_units += buy_amt / max(1.0, lb_px_o)
            liquid_cost_basis += surplus_inr
            total_cost_liquid += cost_inr
            total_slippage += buy_amt * 0.0001
            turnover_liquid += buy_amt

        def _trade_niftybees_to_target(target_core_inr: float, *, force_trim: bool = False) -> None:
            nonlocal core_qty, core_cost_basis, settled_cash, total_cost_core, total_dp_charges, total_stt, total_slippage, turnover_core, realized_core_pnl_inr
            cur_core_val = core_qty * nb_px_o
            delta_inr = target_core_inr - cur_core_val
            band_inr = 2000.0 if (force_trim and delta_inr < 0) else max(2500.0, prev_nav * (cfg.core_rebal_band_pct / 100.0))
            if abs(delta_inr) < band_inr:
                return
            if delta_inr < 0 and core_qty > 0:
                # Trim NIFTYBEES
                sell_q = min(core_qty, int(abs(delta_inr) // max(1.0, nb_px_o)))
                if sell_q < 1 or (sell_q * nb_px_o) < 2000.0:
                    return
                raw_s = sell_q * nb_px_o
                # Equity ETF delivery: STT 0.001% on sell only + exchange + SEBI + GST + CDSL DP once/day + 2 bps ETF spread
                sc = calculate_historical_indian_costs(
                    product_type="DELIVERY",
                    side="SELL",
                    raw_notional=raw_s,
                    trade_date=dt,
                    slippage_bps=2.0,
                    dp_charged_today=((dt, "NIFTYBEES") in dp_pairs),
                    tariff=tariff,
                )
                # Override STT to statutory equity ETF sell rate (0.001% instead of equity 0.10%)
                etf_stt = round(sc["exec_notional"] * 0.00001, 4)
                reg_tot = round(sc["regulatory_total"] - sc["stt"] + etf_stt, 4)
                if sc["dp_charge"] > 0:
                    dp_pairs.add((dt, "NIFTYBEES"))
                net_p = sc["exec_notional"] - reg_tot
                avg_cb = core_cost_basis / max(1, core_qty)
                realized_core_pnl_inr += net_p - (avg_cb * sell_q)
                core_cost_basis = max(0.0, core_cost_basis - avg_cb * sell_q)
                core_qty -= sell_q
                if cfg.settlement_lag_days <= 0:
                    settled_cash += net_p
                else:
                    unsettled_queue.append((ti + cfg.settlement_lag_days, net_p))
                total_cost_core += reg_tot + sc["slippage_cost"]
                total_dp_charges += sc["dp_charge"]
                total_stt += etf_stt
                total_slippage += sc["slippage_cost"]
                turnover_core += sc["exec_notional"]
            elif delta_inr > 0:
                # Buy NIFTYBEES (redeem LIQUIDBEES if needed)
                if settled_cash < delta_inr:
                    _redeem_liquid_bees(delta_inr - settled_cash)
                avail_c = max(0.0, settled_cash - 500.0)
                spend_cap = min(delta_inr, avail_c)
                buy_q = int((spend_cap * 0.997) // max(1.0, nb_px_o * 1.0002))
                if buy_q < 1 or (buy_q * nb_px_o) < 2000.0:
                    return
                raw_b = buy_q * nb_px_o
                bc = calculate_historical_indian_costs(
                    product_type="DELIVERY",
                    side="BUY",
                    raw_notional=raw_b,
                    trade_date=dt,
                    slippage_bps=2.0,
                    tariff=tariff,
                )
                # Statutory equity ETF buy STT is 0.00%
                reg_tot = round(bc["regulatory_total"] - bc["stt"], 4)
                tot_b = bc["exec_notional"] + reg_tot
                if tot_b > settled_cash:
                    return
                settled_cash -= tot_b
                core_qty += buy_q
                core_cost_basis += tot_b
                total_cost_core += reg_tot + bc["slippage_cost"]
                total_slippage += bc["slippage_cost"]
                turnover_core += bc["exec_notional"]

        # Execute Core trim BEFORE tactical buys if tactical needs cash
        if pending_core_target_inr is not None and pending_core_target_inr < core_qty * nb_px_o:
            _trade_niftybees_to_target(pending_core_target_inr, force_trim=bool(pending_tact_buys))


        # 3. Execute Pending Tactical Buys at Day `ti` Open
        remaining_buys: list[dict[str, Any]] = []
        for ord_req in pending_tact_buys:
            col = int(ord_req["col"])
            exp_ti = int(ord_req.get("expires_ti", ti))
            if col in open_pos or ti > exp_ti:
                continue
            if not panel["is_valid_eq"][ti, col] or not panel["is_pit_elig"][ti, col]:
                continue
            raw_open = float(open_p[ti, col])
            bar_vol = float(vol_p[ti, col])
            if np.isnan(raw_open) or raw_open <= 0 or bar_vol <= 0:
                continue
            exec_px = raw_open * (1.0 + cfg.slippage_bps / 10_000.0)
            req_qty = int(ord_req["req_qty"])
            need_inr = req_qty * exec_px * 1.0025
            if settled_cash < need_inr:
                _redeem_liquid_bees(need_inr - settled_cash)
            max_liq_qty = max(1, int(bar_vol * cfg.max_daily_vol_participation))
            if req_qty > max_liq_qty:
                rejection_diagnostics["LIQUIDITY_PARTIAL_FILLS"] += 1
            max_cash_qty = int((settled_cash * 0.992) // max(1.0, exec_px))
            fill_qty = min(req_qty, max_liq_qty, max_cash_qty)
            if fill_qty < 1 or (fill_qty * exec_px) < cfg.min_tactical_lot_inr * 0.70:
                if ti < exp_ti and max_cash_qty < req_qty:
                    remaining_buys.append(ord_req)
                continue
            raw_buy = fill_qty * raw_open
            bc = calculate_historical_indian_costs(
                product_type="DELIVERY",
                side="BUY",
                raw_notional=raw_buy,
                trade_date=dt,
                slippage_bps=cfg.slippage_bps,
                tariff=tariff,
            )
            tot_buy = bc["exec_notional"] + bc["regulatory_total"]
            if tot_buy > settled_cash:
                if ti < exp_ti:
                    remaining_buys.append(ord_req)
                continue
            settled_cash -= tot_buy
            c_leg = bc["regulatory_total"] + bc["slippage_cost"]
            total_cost_tactical += c_leg
            total_stt += bc["stt"]
            total_slippage += bc["slippage_cost"]
            turnover_tactical += bc["exec_notional"]
            sl_r = float(ord_req.get("stop_ratio") or 0.09)
            tp_r = ord_req.get("target_ratio")
            open_pos[col] = {
                "col": col,
                "symbol": str(cur_syms[col]),
                "strategy_id": ord_req["strategy_id"],
                "entry_date": dt,
                "entry_ti": ti,
                "entry_px": exec_px,
                "entry_score": float(ord_req.get("score", 0.5)),
                "last_px": exec_px,
                "max_px": exec_px,
                "qty": fill_qty,
                "buy_exec_notional": bc["exec_notional"],
                "buy_total_cost": c_leg,
                "cost_basis": tot_buy,
                "dividends_received_inr": 0.0,
                "sl_px": exec_px * (1.0 - sl_r) if sl_r > 0 else 0.0,
                "tp_px": exec_px * (1.0 + float(tp_r)) if tp_r else 0.0,
                "atr_entry": float(ord_req.get("atr") or (exec_px * 0.024)),
                "hold_max_days": int(ord_req.get("hold_max_days", cfg.hold_max_days)),
                "trailing_atr_mult": float(ord_req.get("trailing_atr_mult", cfg.trailing_atr_mult)),
                "be_locked": False,
                "bars_held": 0,
            }
        pending_tact_buys = remaining_buys

        # 4. Execute Core Buys & Surplus Cash Sweep into LIQUIDBEES (never leave idle cash stranded at 0%!)
        if pending_core_target_inr is not None and pending_core_target_inr > core_qty * nb_px_o:
            _trade_niftybees_to_target(pending_core_target_inr)
        pending_core_target_inr = None

        if pending_liquid_sweep and settled_cash > 3_000.0 and not pending_tact_buys:
            # Keep tiny operational cash buffer (INR 1,500) and sweep rest into LIQUIDBEES
            _buy_liquid_bees(settled_cash - 1_500.0)
        pending_liquid_sweep = False

        # 5. Intraday / EOD Stop-Loss, Target, Horizon, and Final-Bar Exit Check on Open Tactical Positions
        exits_today = 0
        for col in list(open_pos.keys()):
            if not panel["is_valid_eq"][ti, col]:
                if is_last:
                    pos = open_pos[col]
                    raw_sell = pos["qty"] * pos["last_px"]
                    sym = str(cur_syms[col])
                    sc = calculate_historical_indian_costs(
                        product_type="DELIVERY", side="SELL", raw_notional=raw_sell, trade_date=dt,
                        slippage_bps=cfg.slippage_bps, dp_charged_today=((dt, sym) in dp_pairs), tariff=tariff,
                    )
                    if sc["dp_charge"] > 0:
                        dp_pairs.add((dt, sym))
                    net_proc = sc["exec_notional"] - sc["regulatory_total"]
                    settled_cash += net_proc
                    c_leg = sc["regulatory_total"] + sc["slippage_cost"]
                    total_cost_tactical += c_leg
                    total_dp_charges += sc["dp_charge"]
                    total_stt += sc["stt"]
                    total_slippage += sc["slippage_cost"]
                    turnover_tactical += sc["exec_notional"]
                    pnl_price = net_proc - pos["cost_basis"]
                    pnl_total = pnl_price + pos["dividends_received_inr"]
                    closed_trades.append({
                        "sec_id": str(sec_ids[col]), "symbol": sym, "sector": str(sectors[col]),
                        "industry": str(industries[col]), "strategy_id": pos["strategy_id"],
                        "entry_date": pos["entry_date"], "exit_date": dt, "holding_days": pos["bars_held"],
                        "buy_notional_inr": round(pos["buy_exec_notional"], 2),
                        "sell_notional_inr": round(sc["exec_notional"], 2),
                        "trade_cost_inr": round(pos["buy_total_cost"] + c_leg, 2),
                        "dividends_received_inr": round(pos["dividends_received_inr"], 2),
                        "pnl_amount": round(pnl_price, 2), "total_pnl_with_div_inr": round(pnl_total, 2),
                        "net_return_pct": round((pnl_total / max(1e-6, pos["cost_basis"])) * 100.0, 2),
                        "exit_reason": "FINAL_BAR_CLOSE",
                    })
                    del open_pos[col]
                continue

            pos = open_pos[col]
            o_v, h_v, l_v, c_v, v_v = (
                float(open_p[ti, col]), float(high_p[ti, col]), float(low_p[ti, col]), float(close_p[ti, col]), float(vol_p[ti, col])
            )
            if np.isnan(c_v) or c_v <= 0:
                continue
            pos["bars_held"] += 1
            pos["last_px"] = c_v
            if v_v <= 0 and not is_last:
                continue

            exit_px: float | None = None
            reason = ""
            p_hold_max = int(pos.get("hold_max_days", cfg.hold_max_days))
            if pos["sl_px"] > 0 and o_v <= pos["sl_px"]:
                exit_px = o_v
                reason = "STOP_LOSS_BE_GAP" if pos["be_locked"] else "STOP_LOSS_GAP"
            elif pos["sl_px"] > 0 and l_v <= pos["sl_px"]:
                exit_px = pos["sl_px"]
                reason = "STOP_LOSS_BE" if pos["be_locked"] else "STOP_LOSS"
            elif pos["tp_px"] > 0 and o_v >= pos["tp_px"]:
                exit_px = o_v
                reason = "TARGET_GAP"
            elif pos["tp_px"] > 0 and h_v >= pos["tp_px"]:
                exit_px = pos["tp_px"]
                reason = "TARGET"
            elif p_hold_max > 0 and pos["bars_held"] >= p_hold_max:
                exit_px = c_v
                reason = "HORIZON_EXIT"
            elif is_last:
                exit_px = c_v
                reason = "FINAL_BAR_CLOSE"

            if exit_px is not None:
                raw_sell = pos["qty"] * exit_px
                sym = str(cur_syms[col])
                sc = calculate_historical_indian_costs(
                    product_type="DELIVERY", side="SELL", raw_notional=raw_sell, trade_date=dt,
                    slippage_bps=cfg.slippage_bps, dp_charged_today=((dt, sym) in dp_pairs), tariff=tariff,
                )
                if sc["dp_charge"] > 0:
                    dp_pairs.add((dt, sym))
                net_proc = sc["exec_notional"] - sc["regulatory_total"]
                if is_last or cfg.settlement_lag_days <= 0:
                    settled_cash += net_proc
                else:
                    unsettled_queue.append((ti + cfg.settlement_lag_days, net_proc))
                c_leg = sc["regulatory_total"] + sc["slippage_cost"]
                total_cost_tactical += c_leg
                total_dp_charges += sc["dp_charge"]
                total_stt += sc["stt"]
                total_slippage += sc["slippage_cost"]
                turnover_tactical += sc["exec_notional"]
                pnl_price = net_proc - pos["cost_basis"]
                pnl_total = pnl_price + pos["dividends_received_inr"]
                closed_trades.append({
                    "sec_id": str(sec_ids[col]), "symbol": sym, "sector": str(sectors[col]),
                    "industry": str(industries[col]), "strategy_id": pos["strategy_id"],
                    "entry_date": pos["entry_date"], "exit_date": dt, "holding_days": pos["bars_held"],
                    "buy_notional_inr": round(pos["buy_exec_notional"], 2),
                    "sell_notional_inr": round(sc["exec_notional"], 2),
                    "trade_cost_inr": round(pos["buy_total_cost"] + c_leg, 2),
                    "dividends_received_inr": round(pos["dividends_received_inr"], 2),
                    "pnl_amount": round(pnl_price, 2), "total_pnl_with_div_inr": round(pnl_total, 2),
                    "net_return_pct": round((pnl_total / max(1e-6, pos["cost_basis"])) * 100.0, 2),
                    "exit_reason": reason,
                })
                if "STOP_LOSS" in reason and cfg.cooldown_days > 0:
                    cooldown_until[col] = ti + cfg.cooldown_days
                del open_pos[col]
                exits_today += 1
                vacated_slot_flag = True
            else:
                pos["max_px"] = max(pos["max_px"], h_v)
                if not cfg.disable_tight_breakeven_stop and not pos["be_locked"]:
                    if c_v >= pos["entry_px"] + 3.0 * pos["atr_entry"]:
                        be_px = pos["entry_px"] * 1.0065
                        if be_px > pos["sl_px"]:
                            pos["sl_px"] = be_px
                            pos["be_locked"] = True
                p_tr_atr = float(pos.get("trailing_atr_mult", cfg.trailing_atr_mult))
                if p_tr_atr > 0:
                    tr_px = pos["max_px"] - p_tr_atr * pos["atr_entry"]
                    if tr_px > pos["sl_px"]:
                        pos["sl_px"] = tr_px

        # 6. End-of-Day Four-Ledger Mark-to-Market Valuation
        unsettled_amt = sum(a for _, a in unsettled_queue)
        if is_last and unsettled_amt > 0:
            settled_cash += unsettled_amt
            unsettled_queue.clear()
            unsettled_amt = 0.0

        core_mkt = core_qty * nb_px_c
        liquid_mkt = liquid_units * lb_px_c
        tact_mkt = sum(p["last_px"] * p["qty"] for p in open_pos.values())
        nav = settled_cash + unsettled_amt + core_mkt + liquid_mkt + tact_mkt
        peak_nav = max(peak_nav, nav)
        cur_dd_pct = ((peak_nav - nav) / max(1.0, peak_nav)) * 100.0

        d_ret = (nav / max(1.0, prev_nav)) - 1.0 if step_k > 0 else 0.0
        core_d_ret = (nb_px_c / max(1.0, prev_core_px)) - 1.0 if step_k > 0 else 0.0
        n50_d_ret = (
            float(n50_c[ti] / n50_c[prev_ti] - 1.0)
            if ti > 0 and not np.isnan(n50_c[ti]) and not np.isnan(n50_c[prev_ti]) and n50_c[prev_ti] > 0
            else 0.0
        )
        if open_pos and prev_ti >= 0:
            t_num = 0.0
            t_den = 0.0
            for p_col, p_obj in open_pos.items():
                c_now = float(close_p[ti, p_col])
                c_prev = float(close_p[prev_ti, p_col])
                if not np.isnan(c_now) and not np.isnan(c_prev) and c_prev > 0:
                    t_num += p_obj["qty"] * (c_now - c_prev)
                    t_den += p_obj["qty"] * c_prev
            tact_d_ret = (t_num / t_den) if t_den > 1000.0 else core_d_ret
        else:
            tact_d_ret = core_d_ret
        port_ret_history.append(d_ret)
        core_ret_history.append(core_d_ret)
        tact_ret_history.append(float(np.clip(tact_d_ret, -0.15, 0.15)))
        n50_ret_history.append(n50_d_ret)

        core_pct_now = (core_mkt / max(1.0, nav)) * 100.0
        tact_pct_now = (tact_mkt / max(1.0, nav)) * 100.0
        liquid_pct_now = (liquid_mkt / max(1.0, nav)) * 100.0
        idle_uninvested_pct = ((settled_cash + unsettled_amt) / max(1.0, nav)) * 100.0
        invested_productive_pct = core_pct_now + tact_pct_now + liquid_pct_now

        daily_navs.append({
            "date": dt,
            "nav": round(nav, 2),
            "core_mkt_inr": round(core_mkt, 2),
            "tactical_mkt_inr": round(tact_mkt, 2),
            "liquid_mkt_inr": round(liquid_mkt, 2),
            "uninvested_cash_inr": round(settled_cash + unsettled_amt, 2),
            "core_pct": round(core_pct_now, 2),
            "tactical_pct": round(tact_pct_now, 2),
            "liquid_pct": round(liquid_pct_now, 2),
            "uninvested_cash_pct": round(idle_uninvested_pct, 2),
            "productive_capital_pct": round(invested_productive_pct, 2),
            "equity_exposure_pct": round(core_pct_now + tact_pct_now, 2),
            "open_tactical_positions": len(open_pos),
            "regime": regimes[ti],
            "drawdown_pct": round(cur_dd_pct, 2),
            "ret": d_ret,
            "core_target_pct": cur_tgt_c,
            "tactical_target_pct": cur_tgt_t,
            "liquid_target_pct": cur_tgt_l,
        })
        prev_nav = nav
        prev_core_px = nb_px_c
        prev_tact_mkt = tact_mkt

        if is_last:
            continue

        # 7. Intelligent Event-Driven + Periodic Capital Deployment Evaluation at Day `ti` Close
        is_periodic_day = (step_k == 0) or (step_k % max(1, cfg.periodic_review_interval_days) == 0)
        is_event_replenish = (
            cfg.event_driven_replenishment
            and vacated_slot_flag
            and (step_k - last_replenish_step >= cfg.replenish_min_gap_days)
            and cfg.tactical_target_pct > 0.0
        )
        # Also trigger sweep if uninvested cash exceeds 4% of NAV
        needs_cash_sweep = (settled_cash > max(3_500.0, nav * 0.04))

        if not (is_periodic_day or is_event_replenish or needs_cash_sweep):
            continue

        # Gather causal signals across active sleeves
        merged_by_col: dict[int, dict[str, Any]] = {}
        keep_cols: set[int] = set()
        for sid in cfg.active_sleeves:
            s_w = float(sleeve_w_map.get(sid, 1.0 / max(1, len(cfg.active_sleeves))))
            c_list = pre_cands.get(sid, {}).get(ti, [])
            for c_keep in c_list[:20]:
                keep_cols.add(int(c_keep["col"]))
            for rank_idx, c in enumerate(c_list[:12]):
                col = int(c["col"])
                prio = s_w / (1.0 + 0.25 * rank_idx) + 0.15 * float(c["score"])
                if col not in merged_by_col or prio > merged_by_col[col]["composite_score"]:
                    sc_ref = ctx["strat_cfgs"][sid]
                    merged_by_col[col] = {
                        **c,
                        "strategy_id": sid,
                        "hold_max_days": max(cfg.hold_max_days, sc_ref.hold_max_days),
                        "trailing_atr_mult": cfg.trailing_atr_mult,
                        "composite_score": prio,
                    }

        ranked_cands = sorted(merged_by_col.values(), key=lambda x: -float(x["composite_score"]))
        avg_sig_score = float(np.mean([float(c["score"]) for c in ranked_cands[:6]])) if ranked_cands else 0.0

        # Causal trailing 20D annualized volatilities
        p_vol20 = float(np.std(port_ret_history[-20:], ddof=1)) * math.sqrt(252.0) * 100.0 if len(port_ret_history) >= 15 else 11.0
        c_vol20 = float(np.std(core_ret_history[-20:], ddof=1)) * math.sqrt(252.0) * 100.0 if len(core_ret_history) >= 15 else 13.0
        t_vol20 = float(np.std(tact_ret_history[-20:], ddof=1)) * math.sqrt(252.0) * 100.0 if len(tact_ret_history) >= 15 else 15.0

        # Compute dynamic pillar targets on periodic review or day 0 (and on event replenish)
        if is_periodic_day or is_event_replenish:
            cur_tgt_c, cur_tgt_t, cur_tgt_l = _compute_dynamic_pillar_targets(
                cfg=cfg,
                cur_regime=regimes[ti],
                cur_dd_pct=cur_dd_pct,
                port_vol_20d_ann=p_vol20,
                core_vol_20d_ann=c_vol20,
                tact_vol_20d_ann=t_vol20,
                avg_signal_score=avg_sig_score,
                signal_count=len(ranked_cands),
            )

        # Periodic cost-aware rotation check: NEVER rotate an intact open position into a marginally better signal
        if is_periodic_day and open_pos and cfg.tactical_target_pct > 0.0:
            sma50_row = sma_50d[ti]
            top_new_score = float(ranked_cands[0]["score"]) if ranked_cands else 0.0
            for col, pos in open_pos.items():
                trend_intact = (not np.isnan(sma50_row[col])) and (float(close_p[ti, col]) >= float(sma50_row[col]) * 0.96)
                if trend_intact or col in keep_cols:
                    # Check if a vastly superior replacement exists (hysteresis hurdle)
                    if (not trend_intact) and (top_new_score - float(pos.get("entry_score", 0.5)) < cfg.min_rotation_score_upgrade):
                        rejection_diagnostics["ROTATION_UPGRADE_HURDLE_SKIPPED"] += 1
                    continue
                # Only rotate if trend broke below 0.96*SMA50 AND dropped out of sleeve buffer
                if pos["bars_held"] >= 15:
                    pending_tact_sells.add(col)

        # Tactical Capital Deployment (both on periodic review and event-driven replenishment!)
        tact_budget_inr = nav * (cur_tgt_t / 100.0)
        retained_cols = [c for c in open_pos if c not in pending_tact_sells]
        retained_tact_inr = sum(open_pos[c]["last_px"] * open_pos[c]["qty"] for c in retained_cols)
        rem_tact_budget = max(0.0, tact_budget_inr - retained_tact_inr)

        # Track sector, industry, and unknown exposure across retained tactical positions
        sec_counts: dict[str, int] = {}
        sec_exp_inr: dict[str, float] = {}
        ind_exp_inr: dict[str, float] = {}
        unknown_exp_inr = 0.0
        for col in retained_cols:
            v_pos = open_pos[col]["last_px"] * open_pos[col]["qty"]
            s_nm = str(sectors[col])
            i_nm = str(industries[col])
            if bool(is_unknown_arr[col]) or s_nm == UNKNOWN_SECTOR_LABEL:
                unknown_exp_inr += v_pos
            else:
                sec_counts[s_nm] = sec_counts.get(s_nm, 0) + 1
                sec_exp_inr[s_nm] = sec_exp_inr.get(s_nm, 0.0) + v_pos
                ind_exp_inr[i_nm] = ind_exp_inr.get(i_nm, 0.0) + v_pos

        active_cols = list(retained_cols)
        new_orders_queued = 0
        if (is_periodic_day or is_event_replenish) and rem_tact_budget >= cfg.min_tactical_lot_inr and cfg.tactical_target_pct > 0.0:
            pending_tact_buys.clear()
            max_sec_inr = nav * (cfg.max_sector_exposure_pct / 100.0)
            max_ind_inr = nav * (cfg.max_industry_exposure_pct / 100.0)
            max_unk_inr = nav * (cfg.max_unknown_sector_exposure_pct / 100.0)
            max_single_inr = nav * (cfg.max_single_stock_pct / 100.0)
            eff_slots = max(1, min(cfg.max_tactical_positions, int(tact_budget_inr // (cfg.min_tactical_lot_inr * 1.35))))
            target_slot_inr = min(max_single_inr, max(cfg.min_tactical_lot_inr * 1.30, tact_budget_inr / eff_slots))

            for c in ranked_cands:
                if len(active_cols) >= cfg.max_tactical_positions or rem_tact_budget < cfg.min_tactical_lot_inr:
                    if len(active_cols) >= cfg.max_tactical_positions:
                        rejection_diagnostics["MAX_POSITIONS_BLOCKED"] += 1
                    break
                col = int(c["col"])
                if col in open_pos or ti < cooldown_until.get(col, -1):
                    continue
                s_nm = str(sectors[col])
                i_nm = str(industries[col])
                is_unk = bool(is_unknown_arr[col]) or s_nm == UNKNOWN_SECTOR_LABEL

                # 4-Tier Sector, Industry & Unknown Concentration Checks
                if is_unk:
                    if unknown_exp_inr + cfg.min_tactical_lot_inr > max_unk_inr:
                        rejection_diagnostics["UNKNOWN_SECTOR_CAP_BLOCKED"] += 1
                        continue
                else:
                    if sec_counts.get(s_nm, 0) >= cfg.max_sector_positions or sec_exp_inr.get(s_nm, 0.0) + cfg.min_tactical_lot_inr > max_sec_inr:
                        rejection_diagnostics["SECTOR_CAP_BLOCKED"] += 1
                        continue
                    if ind_exp_inr.get(i_nm, 0.0) + cfg.min_tactical_lot_inr > max_ind_inr:
                        rejection_diagnostics["INDUSTRY_CAP_BLOCKED"] += 1
                        continue

                # Pairwise 60D correlation cap
                corrs = [_causal_pairwise_corr_60d(daily_ret, ti, col, ac) for ac in active_cols]
                if corrs and max(corrs) > cfg.max_pairwise_corr:
                    rejection_diagnostics["CORRELATION_CAP_BLOCKED"] += 1
                    continue

                spot = float(close_p[ti, col])
                if np.isnan(spot) or spot <= 0:
                    continue

                alloc_amt = min(target_slot_inr, rem_tact_budget, max_single_inr)
                if not is_unk:
                    alloc_amt = min(alloc_amt, max(0.0, max_sec_inr - sec_exp_inr.get(s_nm, 0.0)), max(0.0, max_ind_inr - ind_exp_inr.get(i_nm, 0.0)))
                else:
                    alloc_amt = min(alloc_amt, max(0.0, max_unk_inr - unknown_exp_inr))

                qty = int(alloc_amt // spot)
                lot_val = qty * spot
                if qty < 1 or lot_val < cfg.min_tactical_lot_inr:
                    rejection_diagnostics["MIN_LOT_SIZE_BLOCKED"] += 1
                    continue

                # Transaction-cost-aware expected net alpha hurdle:
                # Round-trip delivery friction in bps = 20 bps STT + 1.5 bps stamp + 15 bps slip + (15.93 / lot_val)*10000
                rt_cost_bps = 37.5 + (15.93 / max(1.0, lot_val)) * 10_000.0
                # Expected gross upside proxy from ATR stop scale & validated rank confidence
                sl_r = float(c.get("stop_ratio") or 0.09)
                est_gross_edge_bps = float(c["score"]) * 420.0 + sl_r * 1200.0
                if (est_gross_edge_bps - rt_cost_bps) < cfg.min_expected_net_alpha_bps:
                    rejection_diagnostics["COST_ALPHA_HURDLE_BLOCKED"] += 1
                    continue

                pending_tact_buys.append({**c, "req_qty": qty, "expires_ti": ti + cfg.settlement_lag_days + 2})
                active_cols.append(col)
                rem_tact_budget -= lot_val
                if is_unk:
                    unknown_exp_inr += lot_val
                else:
                    sec_counts[s_nm] = sec_counts.get(s_nm, 0) + 1
                    sec_exp_inr[s_nm] = sec_exp_inr.get(s_nm, 0.0) + lot_val
                    ind_exp_inr[i_nm] = ind_exp_inr.get(i_nm, 0.0) + lot_val
                new_orders_queued += 1

            if is_event_replenish:
                last_replenish_step = step_k
                vacated_slot_flag = False
                replenishment_events.append({
                    "date": dt,
                    "trigger": "EVENT_DRIVEN_POST_EXIT",
                    "regime": regimes[ti],
                    "open_positions_before": len(retained_cols),
                    "new_orders_queued": new_orders_queued,
                    "remaining_tactical_budget_inr": round(rem_tact_budget, 2),
                })
            elif is_periodic_day:
                last_replenish_step = step_k
                vacated_slot_flag = False

        # Compute Passive Core target INR: reserve cash for queued pending_tact_buys so Core never starves Tactical!
        pending_buys_notional_inr = sum(
            float(o["req_qty"]) * float(close_p[ti, int(o["col"])]) * 1.005
            for o in pending_tact_buys
            if not np.isnan(close_p[ti, int(o["col"])])
        )
        undeployed_tact_surplus = max(0.0, rem_tact_budget)
        if cfg.sweep_unused_tactical_to_core and regimes[ti] in ("STRONG_BULL", "WEAK_BULL", "SIDEWAYS_CHOP"):
            eff_core_target_inr = nav * (cur_tgt_c / 100.0) + undeployed_tact_surplus
        else:
            eff_core_target_inr = nav * (cur_tgt_c / 100.0)
        eff_core_target_inr = min(
            eff_core_target_inr,
            nav * (1.0 - max(0.0, cur_tgt_l / 100.0)) - retained_tact_inr - pending_buys_notional_inr,
        )
        eff_core_target_inr = max(0.0, eff_core_target_inr)

        if is_periodic_day or is_event_replenish or needs_cash_sweep:
            pending_core_target_inr = eff_core_target_inr
            pending_liquid_sweep = True

        # Snapshot sector exposures & Euler risk contributions periodically for Risk Report
        if is_periodic_day and active_cols:
            sec_w_snap = {k: round((v / max(1.0, nav)) * 100.0, 2) for k, v in sec_exp_inr.items()}
            if unknown_exp_inr > 0:
                sec_w_snap[UNKNOWN_SECTOR_LABEL] = round((unknown_exp_inr / max(1.0, nav)) * 100.0, 2)
            sec_w_snap["PASSIVE_CORE_NIFTYBEES"] = round(core_pct_now, 2)
            sec_w_snap["LIQUID_RESERVE_LIQUIDBEES"] = round(liquid_pct_now, 2)
            sector_exposure_samples.append(sec_w_snap)

            if len(retained_cols) >= 2:
                w_rc = np.array([(open_pos[c]["last_px"] * open_pos[c]["qty"]) / max(1.0, nav) for c in retained_cols], dtype=np.float64)
                rc = compute_euler_risk_contributions(
                    daily_ret, ti, retained_cols, w_rc, sectors, [open_pos[c]["strategy_id"] for c in retained_cols]
                )
                euler_rc_samples.append({"date": dt, **rc})

    # Compute comprehensive portfolio metrics + 3-tier risk decomposition
    nav_arr = np.array([float(d["nav"]) for d in daily_navs], dtype=np.float64)
    ret_arr = np.array([float(d["ret"]) for d in daily_navs], dtype=np.float64)
    n50_arr = np.array(n50_ret_history, dtype=np.float64)
    n_days = len(nav_arr)
    years = max(0.05, (n_days - 1) / 252.0)

    final_nav = float(nav_arr[-1])
    tot_ret_pct = ((final_nav / initial_capital) - 1.0) * 100.0
    cagr_pct = (((final_nav / initial_capital) ** (1.0 / years)) - 1.0) * 100.0
    ann_vol = float(np.std(ret_arr[1:], ddof=1)) * math.sqrt(252.0) * 100.0 if n_days > 2 else 0.0
    pk = np.maximum.accumulate(nav_arr)
    mdd_pct = abs(float(np.min((nav_arr - pk) / np.maximum(1.0, pk)))) * 100.0

    rf_d = RISK_FREE_ANNUAL_M3 / 252.0
    ex = ret_arr[1:] - rf_d
    sd_e = float(np.std(ex, ddof=1)) if len(ex) > 1 else 0.0
    sharpe = (float(np.mean(ex)) / sd_e) * math.sqrt(252.0) if sd_e > 1e-6 else 0.0
    down = ex[ex < 0.0]
    ds_sd = math.sqrt(float(np.mean(down ** 2))) if len(down) > 0 else 0.0
    sortino = (float(np.mean(ex)) / ds_sd) * math.sqrt(252.0) if (sd_e > 1e-6 and ds_sd > 1e-6) else 0.0
    calmar = (cagr_pct / mdd_pct) if mdd_pct > 1e-4 else (99.0 if cagr_pct > 0 else 0.0)

    # Market-wide systemic risk decomposition (Beta & R^2 vs Nifty 50)
    if len(ret_arr) > 20 and float(np.std(n50_arr[1:])) > 1e-8:
        cov_pn = float(np.cov(ret_arr[1:], n50_arr[1:], ddof=1)[0, 1])
        var_n = float(np.var(n50_arr[1:], ddof=1))
        port_beta = cov_pn / max(1e-10, var_n)
        corr_pn = float(np.corrcoef(ret_arr[1:], n50_arr[1:])[0, 1])
        systemic_var_share_pct = (corr_pn ** 2) * 100.0
    else:
        port_beta, corr_pn, systemic_var_share_pct = 0.0, 0.0, 0.0

    # Exact four-ledger pillar attribution (reconciles to INR 0.00!)
    core_unrealized_pnl = (core_qty * float(nb_close[t_last])) - core_cost_basis
    core_total_pnl = realized_core_pnl_inr + core_unrealized_pnl
    lb_last_px = float(lb_close[t_last]) if not np.isnan(lb_close[t_last]) and lb_close[t_last] > 0 else 1000.0
    liquid_unrealized_pnl = (liquid_units * lb_last_px) - liquid_cost_basis
    liquid_total_pnl = realized_liquid_pnl_inr + liquid_unrealized_pnl
    tact_price_pnl = sum(float(t["pnl_amount"]) for t in closed_trades)
    sum_all_pillars_gain = core_total_pnl + tact_price_pnl + cum_dividends_inr + liquid_total_pnl
    nav_gain = final_nav - initial_capital


    total_all_costs = total_cost_tactical + total_cost_core + total_cost_liquid
    total_all_turnover = turnover_tactical + turnover_core + turnover_liquid

    # Trade stats
    n_tr = len(closed_trades)
    if n_tr > 0:
        pnls = [float(t["total_pnl_with_div_inr"]) for t in closed_trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        pf = (sum(wins) / max(1e-6, abs(sum(losses)))) if losses else (99.0 if wins else 0.0)
        win_rate = 100.0 * len(wins) / n_tr
        avg_hold = sum(int(t["holding_days"]) for t in closed_trades) / n_tr
    else:
        pf, win_rate, avg_hold = 0.0, 0.0, 0.0

    # Individual sleeve attribution
    sleeve_attr: dict[str, dict[str, Any]] = {}
    for t in closed_trades:
        s = str(t["strategy_id"])
        rec = sleeve_attr.setdefault(s, {"trades": 0, "wins": 0, "total_pnl_with_div_inr": 0.0, "dividends_inr": 0.0, "costs_inr": 0.0})
        rec["trades"] += 1
        if float(t["total_pnl_with_div_inr"]) > 0:
            rec["wins"] += 1
        rec["total_pnl_with_div_inr"] = round(rec["total_pnl_with_div_inr"] + float(t["total_pnl_with_div_inr"]), 2)
        rec["dividends_inr"] = round(rec["dividends_inr"] + float(t["dividends_received_inr"]), 2)
        rec["costs_inr"] = round(rec["costs_inr"] + float(t["trade_cost_inr"]), 2)

    # Nifty BeES benchmark return over exact same dates
    nb_ret_window_pct = ((float(nb_close[t_last]) / max(1.0, float(nb_close[t_indices[0]]))) - 1.0) * 100.0
    cvar_95 = float(np.mean(ret_arr[1:][ret_arr[1:] <= np.percentile(ret_arr[1:], 5.0)])) * 100.0 if n_days > 20 else 0.0

    metrics = {
        "config_id": cfg.config_id,
        "config_name": cfg.config_name,
        "allocation_mode": cfg.allocation_mode,
        "base_weights": f"{cfg.core_target_pct:.0f}/{cfg.tactical_target_pct:.0f}/{cfg.liquid_target_pct:.0f}",
        "window": f"{dates[t_indices[0]]} -> {dates[t_last]} ({n_days}d)",
        "final_nav_inr": round(final_nav, 2),
        "total_return_pct": round(tot_ret_pct, 2),
        "cagr_pct": round(cagr_pct, 2),
        "annualized_vol_pct": round(ann_vol, 2),
        "max_drawdown_pct": round(mdd_pct, 2),
        "sharpe_vs_65_rf": round(sharpe, 2),
        "sortino_vs_65_rf": round(sortino, 2),
        "calmar_ratio": round(calmar, 2),
        "cvar_95_daily_pct": round(cvar_95, 2),
        "niftybees_return_pct": round(nb_ret_window_pct, 2),
        "excess_vs_niftybees_pp": round(tot_ret_pct - nb_ret_window_pct, 2),
        "avg_core_pct": round(float(np.mean([d["core_pct"] for d in daily_navs])), 1),
        "avg_tactical_pct": round(float(np.mean([d["tactical_pct"] for d in daily_navs])), 1),
        "avg_liquid_pct": round(float(np.mean([d["liquid_pct"] for d in daily_navs])), 1),
        "avg_uninvested_cash_pct": round(float(np.mean([d["uninvested_cash_pct"] for d in daily_navs])), 1),
        "avg_productive_capital_pct": round(float(np.mean([d["productive_capital_pct"] for d in daily_navs])), 1),
        "avg_equity_exposure_pct": round(float(np.mean([d["equity_exposure_pct"] for d in daily_navs])), 1),
        "avg_open_tactical_positions": round(float(np.mean([d["open_tactical_positions"] for d in daily_navs])), 2),
        "total_tactical_trades": n_tr,
        "tactical_win_rate_pct": round(win_rate, 1),
        "tactical_profit_factor": round(min(99.0, pf), 2),
        "avg_holding_days": round(avg_hold, 1),
        "total_turnover_x": round(total_all_turnover / initial_capital, 2),
        "tactical_turnover_x": round(turnover_tactical / initial_capital, 2),
        "core_turnover_x": round(turnover_core / initial_capital, 2),
        "total_cost_inr": round(total_all_costs, 2),
        "tactical_cost_inr": round(total_cost_tactical, 2),
        "core_cost_inr": round(total_cost_core, 2),
        "liquid_cost_inr": round(total_cost_liquid, 2),
        "total_dp_charges_inr": round(total_dp_charges, 2),
        "total_stt_inr": round(total_stt, 2),
        "total_slippage_inr": round(total_slippage, 2),
        "core_pnl_contribution_inr": round(core_total_pnl, 2),
        "tactical_price_pnl_inr": round(tact_price_pnl, 2),
        "dividends_credited_inr": round(cum_dividends_inr, 2),
        "liquid_yield_earned_inr": round(cum_liquid_yield_inr, 2),
        "portfolio_beta_vs_nifty50": round(port_beta, 3),
        "systemic_variance_share_pct": round(systemic_var_share_pct, 1),
        "event_replenishment_count": len(replenishment_events),
        "reconciliation_check_diff_inr": round(nav_gain - sum_all_pillars_gain, 2),
    }

    return {
        "config": asdict(cfg),
        "metrics": metrics,
        "daily_navs": daily_navs,
        "trades": closed_trades,
        "dividend_events": dividend_events_log,
        "replenishment_events": replenishment_events,
        "rejection_diagnostics": rejection_diagnostics,
        "sleeve_attribution": sleeve_attr,
        "sector_exposure_samples": sector_exposure_samples,
        "euler_rc_samples": euler_rc_samples,
    }
