"""Moreshwar 3.1 (Prompt 3/5) — Modular Execution-Aware Portfolio Simulation Engine.

Integrates:
- `TransactionCostIntelligence` (single authoritative cost engine — zero duplication)
- `estimate_trade_economic_value` (pre-trade economic value & hurdle approval)
- `evaluate_position_replacement` (10-factor hysteresis replacement engine)
- `evaluate_position_exit` (6 protective stop-loss & profit-protection models A..F)
- `simulate_execution_fill` (6 execution timing models with circuit/liquidity/gap fidelity)
- `compute_intelligent_position_size` (6 position-sizing models A..F)

Preserves exact four-ledger reconciliation (`reconciliation_check_diff_inr == 0.00`),
strict causality (zero future leakage), and real NSE `NIFTYBEES` / `LIQUIDBEES` bars.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import numpy as np

from finsistant_v2.m3.portfolio_construction import EVAL_START_DATE, INITIAL_CAPITAL_M3, RISK_FREE_ANNUAL_M3
from finsistant_v2.m3_1.capital_allocation_engine import (
    CapitalAllocationConfig,
    _causal_pairwise_corr_60d,
    _compute_dynamic_pillar_targets,
)
from finsistant_v2.m3_1.dividend_accounting import (
    LIQUID_BEES_DAILY_YIELD,
    apply_ex_dividend_accounting_and_stop_adjustment,
)
from finsistant_v2.m3_1.sector_hierarchy import UNKNOWN_SECTOR_LABEL
from finsistant_v2.m3_1.trade_quality_models import (
    compute_intelligent_position_size,
    estimate_trade_economic_value,
    evaluate_position_exit,
    evaluate_position_replacement,
    simulate_execution_fill,
)
from finsistant_v2.m3_1.transaction_cost_intelligence import (
    IndianTariffRegimeSchedule,
    TransactionCostIntelligence,
)


@dataclass(frozen=True)
class ExecutionIntelligenceConfig:
    """Complete deterministic configuration for Prompt 3/5 execution & trade-quality experiments."""

    config_id: str
    config_name: str
    # Portfolio pillar architecture (inherited cleanly from Prompt 2)
    core_target_pct: float = 70.0
    tactical_target_pct: float = 20.0
    liquid_target_pct: float = 10.0
    allocation_mode: str = "RISK_BUDGETED"
    active_sleeves: tuple[str, ...] = ("STRAT_E_DEFENSIVE_YIELD_LOWVOL", "STRAT_A_CS_MOMENTUM")
    sleeve_weights: tuple[tuple[str, float], ...] = (
        ("STRAT_E_DEFENSIVE_YIELD_LOWVOL", 0.60),
        ("STRAT_A_CS_MOMENTUM", 0.40),
    )

    # Prompt 3/5 Modular Intelligence Controls
    hurdle_mode: str = "FIXED_120"                   # FIXED_60|FIXED_90|FIXED_120|FIXED_150|FIXED_200|DYNAMIC_VOL
    replacement_policy: str = "PROMPT2_BASELINE"     # PROMPT2_BASELINE|INTELLIGENT_HYSTERESIS|WINNER_SHIELD_LONG_HOLD|OPPORTUNITY_UPGRADE_ONLY|NO_PROACTIVE_ROTATION
    min_replacement_edge_bps: float = 180.0
    min_hold_days_before_rotate: int = 20
    protect_winner_pct: float = 4.0
    stop_model: str = "B_ATR_DYNAMIC"                # A_EXISTING_M3_0_BE|B_ATR_DYNAMIC|C_VOL_ADJUSTED_TRAILING|D_STRUCTURE_BASED|E_TIME_DECAY|F_HYBRID_STOP_PROFIT_PROTECT
    execution_mode: str = "NEXT_SESSION_OPEN"        # NEXT_SESSION_OPEN|NEXT_SESSION_VWAP_APPROX|LIMIT_ORDER_SIM|LIQUIDITY_AWARE|SPREAD_AWARE|VOLATILITY_ADJUSTED
    sizing_mode: str = "A_FIXED"                     # A_FIXED|B_VOLATILITY_ADJUSTED|C_RISK_BUDGETED|D_CONFIDENCE_WEIGHTED|E_EXPECTED_ALPHA_WEIGHTED|F_HYBRID_RISK_ALPHA

    # Turnover & operational controls
    event_driven_replenishment: bool = True
    periodic_review_interval_days: int = 20
    replenish_min_gap_days: int = 2
    sweep_unused_tactical_to_core: bool = True
    core_rebal_band_pct: float = 5.5
    liquid_sweep_min_cash_inr: float = 3_000.0
    min_core_trade_inr: float = 2_000.0
    max_tactical_positions: int = 6
    max_single_stock_pct: float = 15.0
    min_tactical_lot_inr: float = 6_500.0
    max_sector_exposure_pct: float = 28.0
    max_sector_positions: int = 2
    max_industry_exposure_pct: float = 20.0
    max_unknown_sector_exposure_pct: float = 22.0
    max_pairwise_corr: float = 0.72
    target_portfolio_vol_pct: float = 13.5
    max_daily_vol_participation: float = 0.01
    settlement_lag_days: int = 1
    slippage_override_bps: float | None = None       # None -> dynamic spread + sqrt impact; float -> fixed override
    hold_max_days: int = 120
    cooldown_days: int = 10
    enable_dividend_accounting: bool = True


def enrich_m3_1_execution_context(ctx: dict[str, Any]) -> dict[str, Any]:
    """Precompute strictly causal rolling execution features (vol_20d, low_20d, hl_range_pct, adv_20d) once."""
    if "exec_features" in ctx:
        return ctx
    panel = ctx["panel"]
    close_p = panel["adj_close"]
    high_p = panel["adj_high"]
    low_p = panel["adj_low"]
    vol_p = panel["volume"]
    daily_ret = ctx["daily_ret"]
    t_len, n_len = close_p.shape

    # Causal rolling 20D daily volatility (ddof=1)
    vol_20d = np.full((t_len, n_len), 0.018, dtype=np.float64)
    # Causal rolling 20D low (using strictly [t-20 : t], excluding current bar t for stop floor)
    low_20d_prev = np.full((t_len, n_len), np.nan, dtype=np.float64)
    # Causal 20D ADV in INR
    adv_20d_inr = np.full((t_len, n_len), 100_000_000.0, dtype=np.float64)
    # Daily HL range percentage
    hl_range_pct = np.full((t_len, n_len), 2.0, dtype=np.float64)

    valid_hl = (close_p > 0) & (~np.isnan(high_p)) & (~np.isnan(low_p)) & (~np.isnan(close_p)) & (high_p >= low_p)
    hl_range_pct[valid_hl] = np.clip(((high_p[valid_hl] - low_p[valid_hl]) / close_p[valid_hl]) * 100.0, 0.2, 20.0)

    dv = np.where((~np.isnan(close_p)) & (~np.isnan(vol_p)) & (close_p > 0) & (vol_p > 0), close_p * vol_p, 0.0)
    cs_dv = np.cumsum(dv, axis=0)
    for t in range(20, t_len):
        adv_20d_inr[t] = (cs_dv[t] - cs_dv[t - 20]) / 20.0
    adv_20d_inr = np.maximum(100_000.0, adv_20d_inr)

    # Vectorized rolling 20D volatility via cumsum of r and r^2
    r_clean = np.where(np.isnan(daily_ret), 0.0, np.clip(daily_ret, -0.25, 0.25))
    r_cnt = np.where(np.isnan(daily_ret), 0.0, 1.0)
    cs_r = np.cumsum(r_clean, axis=0)
    cs_r2 = np.cumsum(r_clean * r_clean, axis=0)
    cs_n = np.cumsum(r_cnt, axis=0)
    for t in range(20, t_len):
        n_w = np.maximum(2.0, cs_n[t] - cs_n[t - 20])
        s_r = cs_r[t] - cs_r[t - 20]
        s_r2 = cs_r2[t] - cs_r2[t - 20]
        var_w = np.maximum(1e-6, (s_r2 - (s_r * s_r) / n_w) / np.maximum(1.0, n_w - 1.0))
        vol_20d[t] = np.clip(np.sqrt(var_w), 0.006, 0.08)

    # Fast O(20) causal rolling 20D prior low: min over bars [t-20 .. t-1]
    low_clean = np.where(np.isnan(low_p) | (low_p <= 0), np.inf, low_p)
    roll_min = low_clean[:-20].copy()
    for offset in range(1, 20):
        np.minimum(roll_min, low_clean[offset : t_len - 20 + offset], out=roll_min)
    roll_min = np.where(np.isinf(roll_min), np.nan, roll_min)
    low_20d_prev[20:] = roll_min

    # Causal 14D ATR matrix (from ws_m31 if present, else compute cleanly)
    atr_14 = ctx["ws_m31"].get("atr_14")
    if atr_14 is None:
        tr = np.zeros_like(close_p)
        tr[1:] = np.maximum(
            high_p[1:] - low_p[1:],
            np.maximum(np.abs(high_p[1:] - close_p[:-1]), np.abs(low_p[1:] - close_p[:-1])),
        )
        tr = np.where(np.isnan(tr), 0.0, tr)
        cs_tr = np.cumsum(tr, axis=0)
        atr_14 = np.full_like(close_p, np.nan)
        atr_14[14:] = (cs_tr[14:] - cs_tr[:-14]) / 14.0

    ctx["exec_features"] = {
        "vol_20d": vol_20d,
        "low_20d_prev": low_20d_prev,
        "adv_20d_inr": adv_20d_inr,
        "hl_range_pct": hl_range_pct,
        "atr_14": atr_14,
    }
    return ctx


def _to_alloc_cfg(ecfg: ExecutionIntelligenceConfig) -> CapitalAllocationConfig:
    return CapitalAllocationConfig(
        config_id=ecfg.config_id,
        config_name=ecfg.config_name,
        core_target_pct=ecfg.core_target_pct,
        tactical_target_pct=ecfg.tactical_target_pct,
        liquid_target_pct=ecfg.liquid_target_pct,
        allocation_mode=ecfg.allocation_mode,
        active_sleeves=ecfg.active_sleeves,
        sleeve_weights=ecfg.sleeve_weights,
        target_portfolio_vol_pct=ecfg.target_portfolio_vol_pct,
    )


def simulate_m3_1_execution_intelligence(
    *,
    ctx: dict[str, Any],
    cfg: ExecutionIntelligenceConfig,
    cost_engine: TransactionCostIntelligence | None = None,
    date_start: str = EVAL_START_DATE,
    date_end: str | None = None,
    initial_capital: float = INITIAL_CAPITAL_M3,
) -> dict[str, Any]:
    """Run strictly causal Moreshwar 3.1 execution-intelligence simulation with exact 4-ledger accounting."""
    enrich_m3_1_execution_context(ctx)
    ce = cost_engine or TransactionCostIntelligence(IndianTariffRegimeSchedule())

    panel = ctx["panel"]
    ws = ctx["ws_m31"]
    ef = ctx["exec_features"]
    etf_bars = ctx["etf_bars"]
    regimes = ctx["regime_rule5"]
    daily_ret = ctx["daily_ret"]
    pre_cands = ctx["precomputed_cands"]
    div_per_adj = ws["div_per_adj_share"]
    div_yield_ttm = ws["factors"]["DIV_YIELD_TTM_PIT"]
    sma_50d = ws["sma_50d"]

    vol_20d_mat = ef["vol_20d"]
    low_20d_prev_mat = ef["low_20d_prev"]
    adv_20d_mat = ef["adv_20d_inr"]
    hl_range_mat = ef["hl_range_pct"]
    atr_14_mat = ef["atr_14"]

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

    alloc_cfg_view = _to_alloc_cfg(cfg)
    sleeve_w_map = dict(cfg.sleeve_weights)

    # Four-ledger state
    settled_cash = float(initial_capital)
    unsettled_queue: list[tuple[int, float]] = []
    core_qty = 0
    core_cost_basis = 0.0
    liquid_units = 0.0
    liquid_cost_basis = 0.0
    open_pos: dict[int, dict[str, Any]] = {}

    pending_tact_sells: set[int] = set()
    pending_tact_buys: list[dict[str, Any]] = []
    pending_core_target_inr: float | None = None
    pending_liquid_sweep: bool = False
    cooldown_until: dict[int, int] = {}
    dp_pairs: set[tuple[str, str]] = set()

    closed_trades: list[dict[str, Any]] = []
    daily_navs: list[dict[str, Any]] = []
    ev_approved_samples: list[float] = []
    ev_rejected_samples: list[float] = []
    ev_audit_rows: list[dict[str, Any]] = []

    diagnostics = {
        "COST_ALPHA_HURDLE_BLOCKED": 0,
        "COST_ALPHA_HURDLE_APPROVED": 0,
        "REPLACEMENT_EXECUTED": 0,
        "REPLACEMENT_BLOCKED_HYSTERESIS": 0,
        "WINNER_SHIELD_RETENTIONS": 0,
        "EXEC_FILLED_FULL": 0,
        "EXEC_FILLED_PARTIAL": 0,
        "EXEC_UNFILLED_LIMIT_OR_GAP": 0,
        "EXEC_FAILED_CIRCUIT_OR_LIQ": 0,
        "EX_DIV_STOP_PROTECTIONS": 0,
        "SECTOR_OR_CORR_BLOCKED": 0,
        "MIN_LOT_BLOCKED": 0,
    }

    total_cost_tactical = 0.0
    total_cost_core = 0.0
    total_cost_liquid = 0.0
    total_dp_charges = 0.0
    total_stt = 0.0
    total_stamp = 0.0
    total_exch_sebi_gst = 0.0
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
    cur_tgt_c, cur_tgt_t, cur_tgt_l = cfg.core_target_pct, cfg.tactical_target_pct, cfg.liquid_target_pct

    def _record_cost_components(c_dict: dict[str, Any], pillar: str) -> None:
        nonlocal total_cost_tactical, total_cost_core, total_cost_liquid
        nonlocal total_dp_charges, total_stt, total_stamp, total_exch_sebi_gst, total_slippage
        c_tot = float(c_dict["total_cost"])
        if pillar == "TACTICAL":
            total_cost_tactical += c_tot
        elif pillar == "CORE":
            total_cost_core += c_tot
        else:
            total_cost_liquid += c_tot
        total_dp_charges += float(c_dict["dp_charge"])
        total_stt += float(c_dict["stt"])
        total_stamp += float(c_dict["stamp_duty"])
        total_exch_sebi_gst += float(c_dict["exchange_txn"]) + float(c_dict["sebi_fee"]) + float(c_dict["gst"])
        total_slippage += float(c_dict["slippage_cost"])

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

        # 0B. Accrue overnight Liquid BeES fractional unit yield
        lb_px_o = float(lb_open[ti]) if not np.isnan(lb_open[ti]) and lb_open[ti] > 0 else 1000.0
        lb_px_c = float(lb_close[ti]) if not np.isnan(lb_close[ti]) and lb_close[ti] > 0 else 1000.0
        if step_k > 0 and liquid_units > 1e-6:
            liq_yield_today = liquid_units * lb_px_c * LIQUID_BEES_DAILY_YIELD
            liquid_units += liq_yield_today / max(1.0, lb_px_c)
            cum_liquid_yield_inr += liq_yield_today

        # 0C. Apply Ex-Dividend Cash Crediting & Stop-Loss Protection BEFORE checking stops
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
                        if div_ev["adjusted_sl"]:
                            diagnostics["EX_DIV_STOP_PROTECTIONS"] += 1

        # 1. Execute Pending Tactical Replacement Sells at Day `ti`
        for col in list(pending_tact_sells):
            if col not in open_pos or not panel["is_valid_eq"][ti, col]:
                continue
            pos = open_pos[col]
            prev_c_v = float(close_p[prev_ti, col]) if prev_ti >= 0 and not np.isnan(close_p[prev_ti, col]) else float(pos["last_px"])
            fill = simulate_execution_fill(
                execution_mode=cfg.execution_mode,
                side="SELL",
                req_qty=int(pos["qty"]),
                prev_close=prev_c_v,
                bar_open=float(open_p[ti, col]),
                bar_high=float(high_p[ti, col]),
                bar_low=float(low_p[ti, col]),
                bar_close=float(close_p[ti, col]),
                bar_vol=float(vol_p[ti, col]),
                adv_20d_inr=float(adv_20d_mat[ti, col]),
                atr_14=float(pos["atr_entry"]),
                hl_range_pct=float(hl_range_mat[ti, col]),
                max_bar_vol_part=max(0.05, cfg.max_daily_vol_participation),
            )
            if fill["fill_qty"] <= 0:
                continue
            raw_sell = int(pos["qty"]) * float(fill["raw_fill_px"])
            sym = str(cur_syms[col])
            sc = ce.calculate_leg_cost(
                instrument_class="DELIVERY",
                side="SELL",
                raw_notional=raw_sell,
                trade_date=dt,
                slippage_bps=cfg.slippage_override_bps,
                adv_20d_inr=float(adv_20d_mat[ti, col]),
                daily_vol_frac=float(vol_20d_mat[ti, col]),
                hl_range_pct=float(hl_range_mat[ti, col]),
                execution_mode=cfg.execution_mode,
                dp_already_charged_today=((dt, sym) in dp_pairs),
            )
            if sc["dp_charge"] > 0:
                dp_pairs.add((dt, sym))
            net_proc = float(sc["exec_notional"]) - float(sc["regulatory_total"])
            if cfg.settlement_lag_days <= 0:
                settled_cash += net_proc
            else:
                unsettled_queue.append((ti + cfg.settlement_lag_days, net_proc))
            _record_cost_components(sc, "TACTICAL")
            turnover_tactical += float(sc["exec_notional"])
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
                "sell_notional_inr": round(float(sc["exec_notional"]), 2),
                "trade_cost_inr": round(pos["buy_total_cost"] + float(sc["total_cost"]), 2),
                "dividends_received_inr": round(pos["dividends_received_inr"], 2),
                "pnl_amount": round(pnl_price, 2),
                "total_pnl_with_div_inr": round(pnl_total, 2),
                "net_return_pct": round((pnl_total / max(1e-6, pos["cost_basis"])) * 100.0, 2),
                "exit_reason": "INTELLIGENT_REPLACE",
            })
            diagnostics["REPLACEMENT_EXECUTED"] += 1
            del open_pos[col]
        pending_tact_sells.clear()

        nb_px_o = float(nb_open[ti]) if not np.isnan(nb_open[ti]) and nb_open[ti] > 0 else prev_core_px
        nb_px_c = float(nb_close[ti]) if not np.isnan(nb_close[ti]) and nb_close[ti] > 0 else nb_px_o

        def _redeem_liquid_bees(needed_inr: float) -> None:
            nonlocal liquid_units, liquid_cost_basis, realized_liquid_pnl_inr, settled_cash, turnover_liquid
            if needed_inr <= 500.0 or liquid_units * lb_px_o < 500.0:
                return
            sell_amt = min(needed_inr * 1.005, liquid_units * lb_px_o)
            units_sell = min(liquid_units, sell_amt / max(1.0, lb_px_o))
            raw_sell_val = units_sell * lb_px_o
            lc = ce.calculate_leg_cost(
                instrument_class="LIQUID_ETF",
                side="SELL",
                raw_notional=raw_sell_val,
                trade_date=dt,
                dp_already_charged_today=((dt, "LIQUIDBEES") in dp_pairs),
            )
            if lc["dp_charge"] > 0:
                dp_pairs.add((dt, "LIQUIDBEES"))
            net_cash = float(lc["exec_notional"]) - float(lc["regulatory_total"])
            if net_cash <= 0:
                return
            avg_u_cost = liquid_cost_basis / max(1e-8, liquid_units)
            sold_basis = avg_u_cost * units_sell
            realized_liquid_pnl_inr += net_cash - sold_basis
            liquid_cost_basis = max(0.0, liquid_cost_basis - sold_basis)
            liquid_units = max(0.0, liquid_units - units_sell)
            settled_cash += net_cash
            _record_cost_components(lc, "LIQUID")
            turnover_liquid += float(lc["exec_notional"])

        def _trade_niftybees_to_target(target_core_inr: float, *, force_trim: bool = False) -> None:
            nonlocal core_qty, core_cost_basis, realized_core_pnl_inr, settled_cash, turnover_core
            cur_core_val = core_qty * nb_px_o
            cur_total_est = settled_cash + sum(a for _, a in unsettled_queue) + cur_core_val + (liquid_units * lb_px_o) + sum(p["last_px"] * p["qty"] for p in open_pos.values())
            band_inr = cur_total_est * (cfg.core_rebal_band_pct / 100.0)
            diff_inr = target_core_inr - cur_core_val
            if (not force_trim) and abs(diff_inr) < max(cfg.min_core_trade_inr, band_inr):
                return
            if diff_inr > cfg.min_core_trade_inr:
                if settled_cash < diff_inr:
                    _redeem_liquid_bees(diff_inr - settled_cash)
                avail_for_core = min(diff_inr, settled_cash * 0.996)
                buy_qty = int(avail_for_core // max(1.0, nb_px_o * 1.001))
                if buy_qty < 1 or (buy_qty * nb_px_o) < cfg.min_core_trade_inr:
                    return
                cc = ce.calculate_leg_cost(
                    instrument_class="EQUITY_ETF",
                    side="BUY",
                    raw_notional=buy_qty * nb_px_o,
                    trade_date=dt,
                    adv_20d_inr=1_500_000_000.0,
                    daily_vol_frac=0.008,
                )
                tot_out = float(cc["exec_notional"]) + float(cc["regulatory_total"])
                if tot_out > settled_cash:
                    buy_qty = int((settled_cash * 0.995) // max(1.0, nb_px_o * 1.001))
                    if buy_qty < 1:
                        return
                    cc = ce.calculate_leg_cost(
                        instrument_class="EQUITY_ETF",
                        side="BUY",
                        raw_notional=buy_qty * nb_px_o,
                        trade_date=dt,
                        adv_20d_inr=1_500_000_000.0,
                        daily_vol_frac=0.008,
                    )
                    tot_out = float(cc["exec_notional"]) + float(cc["regulatory_total"])
                if tot_out <= settled_cash:
                    settled_cash -= tot_out
                    core_qty += buy_qty
                    core_cost_basis += tot_out
                    _record_cost_components(cc, "CORE")
                    turnover_core += float(cc["exec_notional"])
            elif diff_inr < -cfg.min_core_trade_inr and core_qty > 0:
                sell_qty = min(core_qty, int(abs(diff_inr) // max(1.0, nb_px_o)))
                if sell_qty < 1:
                    return
                cc = ce.calculate_leg_cost(
                    instrument_class="EQUITY_ETF",
                    side="SELL",
                    raw_notional=sell_qty * nb_px_o,
                    trade_date=dt,
                    adv_20d_inr=1_500_000_000.0,
                    daily_vol_frac=0.008,
                    dp_already_charged_today=((dt, "NIFTYBEES") in dp_pairs),
                )
                if cc["dp_charge"] > 0:
                    dp_pairs.add((dt, "NIFTYBEES"))
                net_in = float(cc["exec_notional"]) - float(cc["regulatory_total"])
                avg_basis = core_cost_basis / max(1, core_qty)
                sold_basis = avg_basis * sell_qty
                realized_core_pnl_inr += net_in - sold_basis
                core_cost_basis = max(0.0, core_cost_basis - sold_basis)
                core_qty -= sell_qty
                if cfg.settlement_lag_days <= 0:
                    settled_cash += net_in
                else:
                    unsettled_queue.append((ti + cfg.settlement_lag_days, net_in))
                _record_cost_components(cc, "CORE")
                turnover_core += float(cc["exec_notional"])

        # 2. If tactical buys are queued, trim Core if above target first, then redeem Liquid BeES
        if pending_tact_buys and pending_core_target_inr is not None and (core_qty * nb_px_o) > pending_core_target_inr + cfg.min_core_trade_inr:
            _trade_niftybees_to_target(pending_core_target_inr, force_trim=True)

        if pending_tact_buys:
            est_buy_cash = sum(float(o["req_qty"]) * float(open_p[ti, int(o["col"])]) for o in pending_tact_buys if not np.isnan(open_p[ti, int(o["col"])]))
            if est_buy_cash > settled_cash:
                _redeem_liquid_bees(est_buy_cash - settled_cash)

        # 3. Execute Pending Tactical Buys via Execution Timing Engine
        next_pending_buys: list[dict[str, Any]] = []
        for order in pending_tact_buys:
            col = int(order["col"])
            if col in open_pos or len(open_pos) >= cfg.max_tactical_positions or not panel["is_valid_eq"][ti, col]:
                continue
            prev_c_v = float(close_p[prev_ti, col]) if prev_ti >= 0 and not np.isnan(close_p[prev_ti, col]) else float(open_p[ti, col])
            atr_v = float(atr_14_mat[ti, col]) if not np.isnan(atr_14_mat[ti, col]) and atr_14_mat[ti, col] > 0 else float(open_p[ti, col]) * 0.025
            fill = simulate_execution_fill(
                execution_mode=cfg.execution_mode,
                side="BUY",
                req_qty=int(order["req_qty"]),
                prev_close=prev_c_v,
                bar_open=float(open_p[ti, col]),
                bar_high=float(high_p[ti, col]),
                bar_low=float(low_p[ti, col]),
                bar_close=float(close_p[ti, col]),
                bar_vol=float(vol_p[ti, col]),
                adv_20d_inr=float(adv_20d_mat[ti, col]),
                atr_14=atr_v,
                hl_range_pct=float(hl_range_mat[ti, col]),
                max_bar_vol_part=cfg.max_daily_vol_participation,
                min_viable_notional_inr=cfg.min_tactical_lot_inr * 0.70,
            )
            if fill["status"] == "UNFILLED":
                diagnostics["EXEC_UNFILLED_LIMIT_OR_GAP"] += 1
                if ti < int(order.get("expires_ti", ti)):
                    next_pending_buys.append(order)
                continue
            if fill["status"] == "FAILED" or fill["fill_qty"] < 1:
                diagnostics["EXEC_FAILED_CIRCUIT_OR_LIQ"] += 1
                continue

            raw_fill_px = float(fill["raw_fill_px"])
            fill_qty = int(fill["fill_qty"])
            max_aff = int((settled_cash * 0.996) // max(1e-6, raw_fill_px * 1.003))
            fill_qty = min(fill_qty, max_aff)
            if fill_qty < 1 or (fill_qty * raw_fill_px) < cfg.min_tactical_lot_inr * 0.70:
                if ti < int(order.get("expires_ti", ti)):
                    next_pending_buys.append(order)
                continue

            if fill["status"] == "PARTIAL" or fill_qty < int(order["req_qty"]):
                diagnostics["EXEC_FILLED_PARTIAL"] += 1
            else:
                diagnostics["EXEC_FILLED_FULL"] += 1

            raw_buy = fill_qty * raw_fill_px
            bc = ce.calculate_leg_cost(
                instrument_class="DELIVERY",
                side="BUY",
                raw_notional=raw_buy,
                trade_date=dt,
                slippage_bps=cfg.slippage_override_bps,
                adv_20d_inr=float(adv_20d_mat[ti, col]),
                daily_vol_frac=float(vol_20d_mat[ti, col]),
                hl_range_pct=float(hl_range_mat[ti, col]),
                execution_mode=cfg.execution_mode,
            )
            tot_out = float(bc["exec_notional"]) + float(bc["regulatory_total"])
            if tot_out > settled_cash:
                continue
            settled_cash -= tot_out
            _record_cost_components(bc, "TACTICAL")
            turnover_tactical += float(bc["exec_notional"])

            sl_r = float(order.get("stop_ratio") or 0.09)
            eff_entry = float(bc["exec_notional"]) / fill_qty
            open_pos[col] = {
                "qty": fill_qty,
                "symbol": str(cur_syms[col]),
                "entry_date": dt,
                "entry_ti": ti,
                "entry_px": eff_entry,
                "last_px": eff_entry,
                "cost_basis": tot_out,
                "buy_exec_notional": float(bc["exec_notional"]),
                "buy_total_cost": float(bc["total_cost"]),
                "strategy_id": str(order["strategy_id"]),
                "entry_score": float(order["score"]),
                "expected_net_alpha_bps": float(order.get("expected_net_alpha_bps") or 0.0),
                "bars_held": 0,
                "hold_max_days": int(order.get("hold_max_days") or cfg.hold_max_days),
                "sl_px": eff_entry * (1.0 - sl_r),
                "atr_entry": max(eff_entry * 0.015, eff_entry * sl_r / 2.6),
                "max_px": eff_entry,
                "be_locked": False,
                "dividends_received_inr": 0.0,
            }
        pending_tact_buys = next_pending_buys

        # 4. Execute Passive Core Rebalance & Liquid BeES Sweep
        if pending_core_target_inr is not None and not is_last:
            _trade_niftybees_to_target(pending_core_target_inr)
            pending_core_target_inr = None

        if pending_liquid_sweep and not is_last:
            # Reserve cash for any still-pending tactical buy orders so we don't needlessly round-trip Liquid BeES
            reserved_for_pending = sum(
                float(o["req_qty"]) * float(close_p[ti, int(o["col"])]) * 1.005
                for o in pending_tact_buys
                if not np.isnan(close_p[ti, int(o["col"])])
            )
            excess_cash = settled_cash - reserved_for_pending - cfg.liquid_sweep_min_cash_inr
            if excess_cash >= max(2_500.0, cfg.liquid_sweep_min_cash_inr * 0.75):
                buy_liq_amt = excess_cash * 0.997
                lc = ce.calculate_leg_cost(
                    instrument_class="LIQUID_ETF",
                    side="BUY",
                    raw_notional=buy_liq_amt,
                    trade_date=dt,
                )
                tot_out = float(lc["exec_notional"]) + float(lc["regulatory_total"])
                if tot_out <= settled_cash:
                    settled_cash -= tot_out
                    liquid_units += buy_liq_amt / max(1.0, lb_px_o)
                    liquid_cost_basis += tot_out
                    _record_cost_components(lc, "LIQUID")
                    turnover_liquid += float(lc["exec_notional"])
            pending_liquid_sweep = False

        # 5. Intraday / EOD Protective Stop-Loss & Exit Intelligence on Open Tactical Positions
        for col in list(open_pos.keys()):
            pos = open_pos[col]
            pos["bars_held"] += 1
            if not panel["is_valid_eq"][ti, col]:
                if is_last:
                    raw_val = pos["qty"] * pos["last_px"]
                    settled_cash += raw_val
                    pnl_price = raw_val - pos["cost_basis"]
                    pnl_total = pnl_price + pos["dividends_received_inr"]
                    closed_trades.append({
                        "sec_id": str(sec_ids[col]),
                        "symbol": str(cur_syms[col]),
                        "sector": str(sectors[col]),
                        "industry": str(industries[col]),
                        "strategy_id": pos["strategy_id"],
                        "entry_date": pos["entry_date"],
                        "exit_date": dt,
                        "holding_days": pos["bars_held"],
                        "buy_notional_inr": round(pos["buy_exec_notional"], 2),
                        "sell_notional_inr": round(raw_val, 2),
                        "trade_cost_inr": round(pos["buy_total_cost"], 2),
                        "dividends_received_inr": round(pos["dividends_received_inr"], 2),
                        "pnl_amount": round(pnl_price, 2),
                        "total_pnl_with_div_inr": round(pnl_total, 2),
                        "net_return_pct": round((pnl_total / max(1e-6, pos["cost_basis"])) * 100.0, 2),
                        "exit_reason": "END_OF_WINDOW_LAST_PX",
                    })
                    del open_pos[col]
                continue
            o_v = float(open_p[ti, col])
            h_v = float(high_p[ti, col])
            l_v = float(low_p[ti, col])
            c_v = float(close_p[ti, col])
            pos["last_px"] = c_v

            s50 = float(sma_50d[ti, col]) if not np.isnan(sma_50d[ti, col]) else None
            l20 = float(low_20d_prev_mat[ti, col]) if not np.isnan(low_20d_prev_mat[ti, col]) else None
            exit_reason, exit_px = evaluate_position_exit(
                pos=pos,
                stop_model=cfg.stop_model,
                o_v=o_v,
                h_v=h_v,
                l_v=l_v,
                c_v=c_v,
                sma_50_v=s50,
                low_20_prev=l20,
                vol_20d_daily=float(vol_20d_mat[ti, col]),
                hold_max_days=int(pos["hold_max_days"]),
                is_last_bar=is_last,
            )

            if exit_reason is not None and exit_px is not None:
                raw_sell = pos["qty"] * exit_px
                sym = str(cur_syms[col])
                sc = ce.calculate_leg_cost(
                    instrument_class="DELIVERY",
                    side="SELL",
                    raw_notional=raw_sell,
                    trade_date=dt,
                    slippage_bps=cfg.slippage_override_bps,
                    adv_20d_inr=float(adv_20d_mat[ti, col]),
                    daily_vol_frac=float(vol_20d_mat[ti, col]),
                    hl_range_pct=float(hl_range_mat[ti, col]),
                    execution_mode="NEXT_SESSION_OPEN",
                    dp_already_charged_today=((dt, sym) in dp_pairs),
                )
                if sc["dp_charge"] > 0:
                    dp_pairs.add((dt, sym))
                net_proc = float(sc["exec_notional"]) - float(sc["regulatory_total"])
                if is_last or cfg.settlement_lag_days <= 0:
                    settled_cash += net_proc
                else:
                    unsettled_queue.append((ti + cfg.settlement_lag_days, net_proc))
                _record_cost_components(sc, "TACTICAL")
                turnover_tactical += float(sc["exec_notional"])
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
                    "sell_notional_inr": round(float(sc["exec_notional"]), 2),
                    "trade_cost_inr": round(pos["buy_total_cost"] + float(sc["total_cost"]), 2),
                    "dividends_received_inr": round(pos["dividends_received_inr"], 2),
                    "pnl_amount": round(pnl_price, 2),
                    "total_pnl_with_div_inr": round(pnl_total, 2),
                    "net_return_pct": round((pnl_total / max(1e-6, pos["cost_basis"])) * 100.0, 2),
                    "exit_reason": exit_reason,
                })
                if "STOP_LOSS" in exit_reason:
                    cooldown_until[col] = ti + cfg.cooldown_days
                del open_pos[col]
                vacated_slot_flag = True

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
        # Cashflow-neutral time-weighted tactical daily return
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
        cum_all_costs = total_cost_tactical + total_cost_core + total_cost_liquid

        daily_navs.append({
            "date": dt,
            "nav": round(nav, 2),
            "gross_nav": round(nav + cum_all_costs, 2),
            "cum_cost_inr": round(cum_all_costs, 2),
            "core_pct": round(core_pct_now, 2),
            "tactical_pct": round(tact_pct_now, 2),
            "liquid_pct": round(liquid_pct_now, 2),
            "uninvested_cash_pct": round(idle_uninvested_pct, 2),
            "productive_capital_pct": round(core_pct_now + tact_pct_now + liquid_pct_now, 2),
            "equity_exposure_pct": round(core_pct_now + tact_pct_now, 2),
            "open_tactical_positions": len(open_pos),
            "regime": regimes[ti],
            "drawdown_pct": round(cur_dd_pct, 2),
            "ret": d_ret,
        })
        prev_nav = nav
        prev_core_px = nb_px_c

        if is_last:
            continue

        # 7. EOD Signal, Replacement & Capital Deployment Decision
        is_periodic_day = (step_k == 0) or (step_k % max(1, cfg.periodic_review_interval_days) == 0)
        is_event_replenish = (
            cfg.event_driven_replenishment
            and vacated_slot_flag
            and (step_k - last_replenish_step >= cfg.replenish_min_gap_days)
            and cfg.tactical_target_pct > 0.0
        )
        needs_cash_sweep = (settled_cash > max(cfg.liquid_sweep_min_cash_inr + 1_500.0, nav * 0.045))
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
                        "composite_score": prio,
                    }

        ranked_cands = sorted(merged_by_col.values(), key=lambda x: -float(x["composite_score"]))
        avg_sig_score = float(np.mean([float(c["score"]) for c in ranked_cands[:6]])) if ranked_cands else 0.0

        p_vol20 = float(np.std(port_ret_history[-20:], ddof=1)) * math.sqrt(252.0) * 100.0 if len(port_ret_history) >= 15 else 11.0
        c_vol20 = float(np.std(core_ret_history[-20:], ddof=1)) * math.sqrt(252.0) * 100.0 if len(core_ret_history) >= 15 else 13.0
        t_vol20 = float(np.std(tact_ret_history[-20:], ddof=1)) * math.sqrt(252.0) * 100.0 if len(tact_ret_history) >= 15 else 15.0

        if is_periodic_day or is_event_replenish:
            cur_tgt_c, cur_tgt_t, cur_tgt_l = _compute_dynamic_pillar_targets(
                cfg=alloc_cfg_view,
                cur_regime=regimes[ti],
                cur_dd_pct=cur_dd_pct,
                port_vol_20d_ann=p_vol20,
                core_vol_20d_ann=c_vol20,
                tact_vol_20d_ann=t_vol20,
                avg_signal_score=avg_sig_score,
                signal_count=len(ranked_cands),
            )

        # Intelligent Trade Replacement Evaluation on Periodic Reviews
        if is_periodic_day and open_pos and cfg.tactical_target_pct > 0.0:
            sma50_row = sma_50d[ti]
            # Evaluate top non-held candidate's economic value as replacement benchmark
            best_new_c = next((c for c in ranked_cands if int(c["col"]) not in open_pos), None)
            best_new_ev: dict[str, Any] | None = None
            if best_new_c is not None:
                bcol = int(best_new_c["col"])
                b_dy = float(div_yield_ttm[ti, bcol]) if not np.isnan(div_yield_ttm[ti, bcol]) else 0.0
                best_new_ev = estimate_trade_economic_value(
                    candidate=best_new_c,
                    lot_notional_inr=max(cfg.min_tactical_lot_inr, nav * (cur_tgt_t / 100.0) / 3.0),
                    trade_date=dt,
                    daily_vol_frac=float(vol_20d_mat[ti, bcol]),
                    hl_range_pct=float(hl_range_mat[ti, bcol]),
                    adv_20d_inr=float(adv_20d_mat[ti, bcol]),
                    div_yield_ttm_frac=b_dy,
                    regime=regimes[ti],
                    cost_engine=ce,
                    execution_mode=cfg.execution_mode,
                    hurdle_mode=cfg.hurdle_mode,
                    slippage_override_bps=cfg.slippage_override_bps,
                )

            # Current sector exposure snapshot
            sec_val_now: dict[str, float] = {}
            for oc, op in open_pos.items():
                sn = str(sectors[oc])
                sec_val_now[sn] = sec_val_now.get(sn, 0.0) + op["last_px"] * op["qty"]

            for col, pos in open_pos.items():
                s50_v = float(sma50_row[col]) if not np.isnan(sma50_row[col]) else None
                sn = str(sectors[col])
                sec_pct_c = (sec_val_now.get(sn, 0.0) / max(1.0, nav)) * 100.0
                rep = evaluate_position_replacement(
                    pos=pos,
                    cur_close_px=float(close_p[ti, col]),
                    sma_50_px=s50_v,
                    in_keep_buffer=(col in keep_cols),
                    best_candidate=best_new_c,
                    best_cand_ev=best_new_ev,
                    nav_inr=nav,
                    trade_date=dt,
                    regime=regimes[ti],
                    sector_exposure_pct=sec_pct_c,
                    cost_engine=ce,
                    replacement_policy=cfg.replacement_policy,
                    min_improvement_bps=cfg.min_replacement_edge_bps,
                    min_hold_days_before_rotate=cfg.min_hold_days_before_rotate,
                    protect_winner_pct=cfg.protect_winner_pct,
                )
                if rep["should_replace"]:
                    pending_tact_sells.add(col)
                else:
                    if rep["reason"] == "RETAIN_WINNER_SHIELD_INTACT":
                        diagnostics["WINNER_SHIELD_RETENTIONS"] += 1
                    elif "HYSTERESIS" in rep["reason"] or "INSUFFICIENT_UPGRADE" in rep["reason"]:
                        diagnostics["REPLACEMENT_BLOCKED_HYSTERESIS"] += 1

        # Tactical Capital Deployment with Economic Value Approval & Intelligent Sizing
        tact_budget_inr = nav * (cur_tgt_t / 100.0)
        retained_cols = [c for c in open_pos if c not in pending_tact_sells]
        retained_tact_inr = sum(open_pos[c]["last_px"] * open_pos[c]["qty"] for c in retained_cols)
        rem_tact_budget = max(0.0, tact_budget_inr - retained_tact_inr)

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
        if (is_periodic_day or is_event_replenish) and rem_tact_budget >= cfg.min_tactical_lot_inr and cfg.tactical_target_pct > 0.0:
            pending_tact_buys.clear()
            max_sec_inr = nav * (cfg.max_sector_exposure_pct / 100.0)
            max_ind_inr = nav * (cfg.max_industry_exposure_pct / 100.0)
            max_unk_inr = nav * (cfg.max_unknown_sector_exposure_pct / 100.0)
            max_single_inr = nav * (cfg.max_single_stock_pct / 100.0)
            eff_slots = max(1, min(cfg.max_tactical_positions, int(tact_budget_inr // (cfg.min_tactical_lot_inr * 1.35))))
            base_slot_inr = min(max_single_inr, max(cfg.min_tactical_lot_inr * 1.30, tact_budget_inr / eff_slots))

            for c in ranked_cands:
                if len(active_cols) >= cfg.max_tactical_positions or rem_tact_budget < cfg.min_tactical_lot_inr:
                    break
                col = int(c["col"])
                if col in open_pos or ti < cooldown_until.get(col, -1):
                    continue
                s_nm = str(sectors[col])
                i_nm = str(industries[col])
                is_unk = bool(is_unknown_arr[col]) or s_nm == UNKNOWN_SECTOR_LABEL

                if is_unk:
                    if unknown_exp_inr + cfg.min_tactical_lot_inr > max_unk_inr:
                        diagnostics["SECTOR_OR_CORR_BLOCKED"] += 1
                        continue
                    sec_room = max(0.0, max_unk_inr - unknown_exp_inr)
                    ind_room = sec_room
                else:
                    if sec_counts.get(s_nm, 0) >= cfg.max_sector_positions or sec_exp_inr.get(s_nm, 0.0) + cfg.min_tactical_lot_inr > max_sec_inr:
                        diagnostics["SECTOR_OR_CORR_BLOCKED"] += 1
                        continue
                    if ind_exp_inr.get(i_nm, 0.0) + cfg.min_tactical_lot_inr > max_ind_inr:
                        diagnostics["SECTOR_OR_CORR_BLOCKED"] += 1
                        continue
                    sec_room = max(0.0, max_sec_inr - sec_exp_inr.get(s_nm, 0.0))
                    ind_room = max(0.0, max_ind_inr - ind_exp_inr.get(i_nm, 0.0))

                corrs = [_causal_pairwise_corr_60d(daily_ret, ti, col, ac) for ac in active_cols]
                max_c = max(corrs) if corrs else 0.0
                if max_c > cfg.max_pairwise_corr:
                    diagnostics["SECTOR_OR_CORR_BLOCKED"] += 1
                    continue

                spot = float(close_p[ti, col])
                if np.isnan(spot) or spot <= 0:
                    continue

                dy_c = float(div_yield_ttm[ti, col]) if not np.isnan(div_yield_ttm[ti, col]) else 0.0
                # Preliminary lot notional for exact cost-aware EV calculation
                prelim_lot = min(base_slot_inr, rem_tact_budget, max_single_inr, sec_room, ind_room)
                if prelim_lot < cfg.min_tactical_lot_inr:
                    diagnostics["MIN_LOT_BLOCKED"] += 1
                    continue

                ev = estimate_trade_economic_value(
                    candidate=c,
                    lot_notional_inr=prelim_lot,
                    trade_date=dt,
                    daily_vol_frac=float(vol_20d_mat[ti, col]),
                    hl_range_pct=float(hl_range_mat[ti, col]),
                    adv_20d_inr=float(adv_20d_mat[ti, col]),
                    div_yield_ttm_frac=dy_c,
                    regime=regimes[ti],
                    cost_engine=ce,
                    execution_mode=cfg.execution_mode,
                    hurdle_mode=cfg.hurdle_mode,
                    slippage_override_bps=cfg.slippage_override_bps,
                )
                if len(ev_audit_rows) < 60:
                    ev_audit_rows.append({"date": dt, "symbol": str(cur_syms[col]), "strategy_id": c["strategy_id"], **ev})

                if not ev["approved"]:
                    diagnostics["COST_ALPHA_HURDLE_BLOCKED"] += 1
                    ev_rejected_samples.append(float(ev["expected_net_alpha_bps"]))
                    continue

                diagnostics["COST_ALPHA_HURDLE_APPROVED"] += 1
                ev_approved_samples.append(float(ev["expected_net_alpha_bps"]))

                # Position-Sizing Intelligence
                alloc_amt = compute_intelligent_position_size(
                    sizing_mode=cfg.sizing_mode,
                    base_slot_inr=base_slot_inr,
                    rem_tact_budget_inr=rem_tact_budget,
                    nav_inr=nav,
                    max_single_stock_pct=cfg.max_single_stock_pct,
                    sector_room_inr=sec_room,
                    industry_room_inr=ind_room,
                    adv_20d_inr=float(adv_20d_mat[ti, col]),
                    daily_vol_frac=float(vol_20d_mat[ti, col]),
                    stop_ratio=float(c.get("stop_ratio") or 0.09),
                    signal_score=float(c["score"]),
                    expected_net_alpha_bps=float(ev["expected_net_alpha_bps"]),
                    hurdle_bps=float(ev["hurdle_bps"]),
                    max_corr_with_open=max_c,
                    min_lot_inr=cfg.min_tactical_lot_inr,
                )
                qty = int(alloc_amt // spot)
                lot_val = qty * spot
                if qty < 1 or lot_val < cfg.min_tactical_lot_inr:
                    diagnostics["MIN_LOT_BLOCKED"] += 1
                    continue

                pending_tact_buys.append({
                    **c,
                    "req_qty": qty,
                    "expected_net_alpha_bps": float(ev["expected_net_alpha_bps"]),
                    "expires_ti": ti + cfg.settlement_lag_days + 2,
                })
                active_cols.append(col)
                rem_tact_budget -= lot_val
                if is_unk:
                    unknown_exp_inr += lot_val
                else:
                    sec_counts[s_nm] = sec_counts.get(s_nm, 0) + 1
                    sec_exp_inr[s_nm] = sec_exp_inr.get(s_nm, 0.0) + lot_val
                    ind_exp_inr[i_nm] = ind_exp_inr.get(i_nm, 0.0) + lot_val

            last_replenish_step = step_k
            vacated_slot_flag = False

        # Passive Core Target INR (strictly reserving cash for queued pending_tact_buys)
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

    # -------------------------------------------------------------------------
    # COMPUTE ALL SECTION 10 PERFORMANCE METRICS + EXACT FOUR-LEDGER RECONCILIATION
    # -------------------------------------------------------------------------
    nav_arr = np.array([float(d["nav"]) for d in daily_navs], dtype=np.float64)
    ret_arr = np.array([float(d["ret"]) for d in daily_navs], dtype=np.float64)
    n_days = len(nav_arr)
    years = max(0.05, (n_days - 1) / 252.0)

    final_nav = float(nav_arr[-1])
    total_all_costs = total_cost_tactical + total_cost_core + total_cost_liquid
    total_all_turnover = turnover_tactical + turnover_core + turnover_liquid
    tot_ret_pct = ((final_nav / initial_capital) - 1.0) * 100.0
    gross_ret_pct = (((final_nav + total_all_costs) / initial_capital) - 1.0) * 100.0
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

    # Exact four-ledger reconciliation (reconciles to INR 0.00!)
    core_unrealized_pnl = (core_qty * float(nb_close[t_last])) - core_cost_basis
    core_total_pnl = realized_core_pnl_inr + core_unrealized_pnl
    lb_last_px = float(lb_close[t_last]) if not np.isnan(lb_close[t_last]) and lb_close[t_last] > 0 else 1000.0
    liquid_unrealized_pnl = (liquid_units * lb_last_px) - liquid_cost_basis
    liquid_total_pnl = realized_liquid_pnl_inr + liquid_unrealized_pnl
    tact_price_pnl = sum(float(t["pnl_amount"]) for t in closed_trades)
    sum_all_pillars_gain = core_total_pnl + tact_price_pnl + cum_dividends_inr + liquid_total_pnl
    nav_gain = final_nav - initial_capital

    gross_profit_inr = nav_gain + total_all_costs
    cost_pct_gross_profit = (total_all_costs / gross_profit_inr) * 100.0 if gross_profit_inr > 100.0 else 999.0

    # Complete Trade Quality Statistics
    n_tr = len(closed_trades)
    exit_breakdown: dict[str, int] = {}
    if n_tr > 0:
        pnls = [float(t["total_pnl_with_div_inr"]) for t in closed_trades]
        rets = [float(t["net_return_pct"]) for t in closed_trades]
        wins_inr = [p for p in pnls if p > 0]
        losses_inr = [p for p in pnls if p <= 0]
        wins_pct = [r for r in rets if r > 0]
        losses_pct = [r for r in rets if r <= 0]
        pf = (sum(wins_inr) / max(1e-6, abs(sum(losses_inr)))) if losses_inr else (99.0 if wins_inr else 0.0)
        win_rate = 100.0 * len(wins_inr) / n_tr
        avg_win_inr = float(np.mean(wins_inr)) if wins_inr else 0.0
        avg_loss_inr = float(np.mean(losses_inr)) if losses_inr else 0.0
        avg_win_pct = float(np.mean(wins_pct)) if wins_pct else 0.0
        avg_loss_pct = float(np.mean(losses_pct)) if losses_pct else 0.0
        exp_trade_val_inr = float(np.mean(pnls))
        exp_trade_val_bps = float(np.mean(rets)) * 100.0
        avg_hold = float(np.mean([int(t["holding_days"]) for t in closed_trades]))
        avg_txn_cost = float(np.mean([float(t["trade_cost_inr"]) for t in closed_trades]))
        for t in closed_trades:
            er = str(t["exit_reason"])
            exit_breakdown[er] = exit_breakdown.get(er, 0) + 1
    else:
        pf = win_rate = avg_win_inr = avg_loss_inr = avg_win_pct = avg_loss_pct = 0.0
        exp_trade_val_inr = exp_trade_val_bps = avg_hold = avg_txn_cost = 0.0

    # Strategy family attribution (Family E vs Family A)
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

    nb_ret_window_pct = ((float(nb_close[t_last]) / max(1.0, float(nb_close[t_indices[0]]))) - 1.0) * 100.0

    metrics = {
        "config_id": cfg.config_id,
        "config_name": cfg.config_name,
        "window": f"{dates[t_indices[0]]} -> {dates[t_last]} ({n_days}d)",
        "allocation_mode": cfg.allocation_mode,
        "base_weights": f"{cfg.core_target_pct:.0f}/{cfg.tactical_target_pct:.0f}/{cfg.liquid_target_pct:.0f}",
        "hurdle_mode": cfg.hurdle_mode,
        "replacement_policy": cfg.replacement_policy,
        "stop_model": cfg.stop_model,
        "execution_mode": cfg.execution_mode,
        "is_execution_approximation": (cfg.execution_mode == "NEXT_SESSION_VWAP_APPROX"),
        "sizing_mode": cfg.sizing_mode,
        "final_nav_inr": round(final_nav, 2),
        "gross_return_pct": round(gross_ret_pct, 2),
        "total_return_pct": round(tot_ret_pct, 2),
        "cagr_pct": round(cagr_pct, 2),
        "annualized_vol_pct": round(ann_vol, 2),
        "max_drawdown_pct": round(mdd_pct, 2),
        "sharpe_vs_65_rf": round(sharpe, 2),
        "sortino_vs_65_rf": round(sortino, 2),
        "calmar_ratio": round(calmar, 2),
        "profit_factor": round(min(99.0, pf), 2),
        "win_rate_pct": round(win_rate, 1),
        "avg_win_inr": round(avg_win_inr, 2),
        "avg_win_pct": round(avg_win_pct, 2),
        "avg_loss_inr": round(avg_loss_inr, 2),
        "avg_loss_pct": round(avg_loss_pct, 2),
        "expected_trade_value_inr": round(exp_trade_val_inr, 2),
        "expected_trade_value_bps": round(exp_trade_val_bps, 1),
        "avg_holding_days": round(avg_hold, 1),
        "total_turnover_x": round(total_all_turnover / initial_capital, 2),
        "tactical_turnover_x": round(turnover_tactical / initial_capital, 2),
        "core_turnover_x": round(turnover_core / initial_capital, 2),
        "liquid_turnover_x": round(turnover_liquid / initial_capital, 2),
        "total_tactical_trades": n_tr,
        "avg_transaction_cost_inr": round(avg_txn_cost, 2),
        "total_cost_inr": round(total_all_costs, 2),
        "tactical_cost_inr": round(total_cost_tactical, 2),
        "core_cost_inr": round(total_cost_core, 2),
        "liquid_cost_inr": round(total_cost_liquid, 2),
        "total_dp_charges_inr": round(total_dp_charges, 2),
        "total_stt_inr": round(total_stt, 2),
        "total_stamp_inr": round(total_stamp, 2),
        "total_exch_sebi_gst_inr": round(total_exch_sebi_gst, 2),
        "total_slippage_inr": round(total_slippage, 2),
        "cost_pct_of_gross_profits": round(cost_pct_gross_profit, 2),
        "avg_core_pct": round(float(np.mean([d["core_pct"] for d in daily_navs])), 1),
        "avg_tactical_pct": round(float(np.mean([d["tactical_pct"] for d in daily_navs])), 1),
        "avg_liquid_pct": round(float(np.mean([d["liquid_pct"] for d in daily_navs])), 1),
        "avg_productive_capital_pct": round(float(np.mean([d["productive_capital_pct"] for d in daily_navs])), 1),
        "avg_uninvested_cash_pct": round(float(np.mean([d["uninvested_cash_pct"] for d in daily_navs])), 1),
        "niftybees_return_pct": round(nb_ret_window_pct, 2),
        "excess_vs_niftybees_pp": round(tot_ret_pct - nb_ret_window_pct, 2),
        "trade_replacement_count": diagnostics["REPLACEMENT_EXECUTED"],
        "trade_replacement_freq_per_year": round(diagnostics["REPLACEMENT_EXECUTED"] / years, 2),
        "winner_shield_retentions": diagnostics["WINNER_SHIELD_RETENTIONS"],
        "cost_alpha_hurdle_blocked": diagnostics["COST_ALPHA_HURDLE_BLOCKED"],
        "avg_approved_ev_bps": round(float(np.mean(ev_approved_samples)), 1) if ev_approved_samples else 0.0,
        "avg_rejected_ev_bps": round(float(np.mean(ev_rejected_samples)), 1) if ev_rejected_samples else 0.0,
        "core_pnl_contribution_inr": round(core_total_pnl, 2),
        "tactical_price_pnl_inr": round(tact_price_pnl, 2),
        "dividends_credited_inr": round(cum_dividends_inr, 2),
        "liquid_yield_earned_inr": round(cum_liquid_yield_inr, 2),
        "reconciliation_check_diff_inr": round(nav_gain - sum_all_pillars_gain, 2),
    }

    return {
        "config": asdict(cfg),
        "metrics": metrics,
        "diagnostics": diagnostics,
        "exit_breakdown": exit_breakdown,
        "sleeve_attribution": sleeve_attr,
        "ev_audit_sample": ev_audit_rows,
        "daily_navs": daily_navs,
        "trades": closed_trades,
    }
