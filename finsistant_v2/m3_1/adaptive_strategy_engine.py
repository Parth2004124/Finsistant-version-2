"""Moreshwar 3.1 (Prompt 4/5) — Regime-Aware Adaptive Capital Allocation & Conditional Edge Engine.

Preserves Prompt 3's centralized `TransactionCostIntelligence`, dividend/stop protection, and
exact `0.00 INR` four-ledger accounting while adding:
- Exact per-trade MAE (Maximum Adverse Excursion), MFE (Maximum Favourable Excursion), and entry-regime attribution.
- Regime-aware target allocation schedules (`Core + Tactical + Liquid == 100%`) with hysteresis,
  regime-conditional trade eligibility & hurdle scaling, and optional stress-state tactical halts.
- Full 15-metric conditional edge analysis for Family E across all detected market regimes.
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
from finsistant_v2.m3_1.execution_simulation_engine import (
    ExecutionIntelligenceConfig,
    enrich_m3_1_execution_context,
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
class AdaptiveRegimePolicy:
    """Declarative regime-to-allocation & trade-gating schedule (zero future lookahead)."""

    policy_id: str
    policy_name: str
    regime_source: str = "NONE"  # "NONE" | "RULE_6STATE" | "STAT_4STATE" | "SIMPLE_TREND"
    # Mapping: regime_label -> (core_pct, tactical_pct, liquid_pct, allow_new_entries, hurdle_adder_bps, size_mult)
    regime_rules: tuple[tuple[str, float, float, float, bool, float, float], ...] = ()
    # Minimum consecutive days a regime target must persist before rebalancing Core ETF (hysteresis)
    allocation_confirm_days: int = 3


def get_standard_adaptive_policies() -> dict[str, AdaptiveRegimePolicy]:
    """Return the transparent, low-parameter Section 6 adaptive allocation policies."""
    return {
        "A_STATIC_PROMPT3_PRIMARY": AdaptiveRegimePolicy(
            policy_id="A_STATIC_PROMPT3_PRIMARY",
            policy_name="A. Static Prompt 3 Primary (60/30/10, Family E Only)",
            regime_source="NONE",
        ),
        "B_STATIC_PROMPT3_CHALLENGER": AdaptiveRegimePolicy(
            policy_id="B_STATIC_PROMPT3_CHALLENGER",
            policy_name="B. Static Prompt 3 Challenger (RISK_BUDGETED 65/25/10, Family E Only)",
            regime_source="NONE",
        ),
        "C_RULE_ADAPTIVE_FULL": AdaptiveRegimePolicy(
            policy_id="C_RULE_ADAPTIVE_FULL",
            policy_name="C. Rule-Based 6-State Adaptive Allocation (Core + Tactical + Liquid)",
            regime_source="RULE_6STATE",
            regime_rules=(
                ("BULL_BROAD",         60.0, 32.0,  8.0, True,   0.0, 1.05),
                ("RECOVERY_IMPROVING", 60.0, 30.0, 10.0, True,   0.0, 1.00),
                ("BULL_NARROW",        68.0, 20.0, 12.0, True,  40.0, 0.85),
                ("SIDEWAYS_RANGE",     58.0, 27.0, 15.0, True,  20.0, 0.95),
                ("BEAR_DETERIORATING", 45.0, 15.0, 40.0, False, 80.0, 0.70),
                ("HIGH_VOL_STRESS",    40.0, 10.0, 50.0, False, 100.0, 0.65),
            ),
            allocation_confirm_days=3,
        ),
        "D_STAT_REGIME_ADAPTIVE": AdaptiveRegimePolicy(
            policy_id="D_STAT_REGIME_ADAPTIVE",
            policy_name="D. Statistical 4-State K-Means Regime Adaptive Allocation (Train-Frozen)",
            regime_source="STAT_4STATE",
            regime_rules=(
                ("STAT_BULL_BROAD",       60.0, 32.0,  8.0, True,   0.0, 1.05),
                ("STAT_RECOVERY",         60.0, 30.0, 10.0, True,   0.0, 1.00),
                ("STAT_BULL_NARROW_CHOP", 62.0, 24.0, 14.0, True,  30.0, 0.90),
                ("STAT_STRESS_BEAR",      42.0, 13.0, 45.0, False, 90.0, 0.70),
            ),
            allocation_confirm_days=3,
        ),
        "E_ADAPTIVE_TACTICAL_STATIC_CORE": AdaptiveRegimePolicy(
            policy_id="E_ADAPTIVE_TACTICAL_STATIC_CORE",
            policy_name="E. Adaptive Tactical (0-30%) + Static 60% Core ETF + Dynamic Liquid (10-40%)",
            regime_source="RULE_6STATE",
            regime_rules=(
                ("BULL_BROAD",         60.0, 30.0, 10.0, True,   0.0, 1.05),
                ("RECOVERY_IMPROVING", 60.0, 30.0, 10.0, True,   0.0, 1.00),
                ("BULL_NARROW",        60.0, 18.0, 22.0, True,  40.0, 0.85),
                ("SIDEWAYS_RANGE",     60.0, 25.0, 15.0, True,  20.0, 0.95),
                ("BEAR_DETERIORATING", 60.0, 10.0, 30.0, False, 80.0, 0.70),
                ("HIGH_VOL_STRESS",    60.0,  0.0, 40.0, False, 120.0, 0.50),
            ),
            allocation_confirm_days=3,
        ),
        "F_DEFENSIVE_STRESS_HALT": AdaptiveRegimePolicy(
            policy_id="F_DEFENSIVE_STRESS_HALT",
            policy_name="F. Defensive Stress Halt (Static 60/30/10 Normally; Halt New Entries in Stress/Bear)",
            regime_source="RULE_6STATE",
            regime_rules=(
                ("BULL_BROAD",         60.0, 30.0, 10.0, True,   0.0, 1.00),
                ("RECOVERY_IMPROVING", 60.0, 30.0, 10.0, True,   0.0, 1.00),
                ("BULL_NARROW",        60.0, 30.0, 10.0, True,   0.0, 1.00),
                ("SIDEWAYS_RANGE",     60.0, 30.0, 10.0, True,   0.0, 1.00),
                ("BEAR_DETERIORATING", 55.0,  0.0, 45.0, False, 100.0, 0.50),
                ("HIGH_VOL_STRESS",    50.0,  0.0, 50.0, False, 120.0, 0.50),
            ),
            allocation_confirm_days=2,
        ),
        "G_SIMPLE_TREND_BASELINE": AdaptiveRegimePolicy(
            policy_id="G_SIMPLE_TREND_BASELINE",
            policy_name="G. Simple Trend Baseline (NIFTYBEES >= SMA200: 60/30/10 else 45/10/45)",
            regime_source="SIMPLE_TREND",
            regime_rules=(
                ("BULL_TREND", 60.0, 30.0, 10.0, True,   0.0, 1.00),
                ("BEAR_TREND", 45.0, 10.0, 45.0, False, 80.0, 0.70),
            ),
            allocation_confirm_days=3,
        ),
    }


def simulate_m3_1_adaptive_regime_portfolio(
    *,
    ctx: dict[str, Any],
    exec_cfg: ExecutionIntelligenceConfig,
    policy: AdaptiveRegimePolicy,
    regime_labels: list[str] | None = None,
    candidate_map_override: dict[str, dict[int, list[dict[str, Any]]]] | None = None,
    cost_engine: TransactionCostIntelligence | None = None,
    date_start: str = EVAL_START_DATE,
    date_end: str | None = None,
    initial_capital: float = INITIAL_CAPITAL_M3,
) -> dict[str, Any]:
    """Simulate Prompt 4 adaptive regime portfolio with exact Prompt 3 costs, MAE/MFE, and 0.00 INR reconciliation."""
    enrich_m3_1_execution_context(ctx)
    ce = cost_engine or TransactionCostIntelligence(IndianTariffRegimeSchedule())

    panel = ctx["panel"]
    ws = ctx["ws_m31"]
    ef = ctx["exec_features"]
    etf_bars = ctx["etf_bars"]
    daily_ret = ctx["daily_ret"]
    pre_cands = candidate_map_override if candidate_map_override is not None else ctx["precomputed_cands"]
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
    if len(t_indices) < 15:
        return {"config": asdict(exec_cfg), "policy": asdict(policy), "metrics": {}, "daily_navs": [], "trades": []}

    reg_seq = regime_labels if regime_labels is not None else ctx["regime_rule5"]
    policy_map = {
        r[0]: (float(r[1]), float(r[2]), float(r[3]), bool(r[4]), float(r[5]), float(r[6]))
        for r in policy.regime_rules
    }

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

    nb_open = etf_bars["NIFTYBEES"]["open"]
    nb_close = etf_bars["NIFTYBEES"]["close"]
    lb_open = etf_bars["LIQUIDBEES"]["open"]
    lb_close = etf_bars["LIQUIDBEES"]["close"]

    alloc_cfg_view = CapitalAllocationConfig(
        config_id=exec_cfg.config_id,
        config_name=exec_cfg.config_name,
        core_target_pct=exec_cfg.core_target_pct,
        tactical_target_pct=exec_cfg.tactical_target_pct,
        liquid_target_pct=exec_cfg.liquid_target_pct,
        allocation_mode=exec_cfg.allocation_mode,
        active_sleeves=exec_cfg.active_sleeves,
        sleeve_weights=exec_cfg.sleeve_weights,
        target_portfolio_vol_pct=exec_cfg.target_portfolio_vol_pct,
    )
    sleeve_w_map = dict(exec_cfg.sleeve_weights)

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
    regime_blocked_entries = 0

    total_cost_tactical = 0.0
    total_cost_core = 0.0
    total_cost_liquid = 0.0
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

    last_replenish_step = -999
    vacated_slot_flag = False
    t_last = t_indices[-1]
    peak_nav = float(initial_capital)
    prev_nav = float(initial_capital)
    prev_core_px = float(nb_close[t_indices[0]])
    cur_tgt_c, cur_tgt_t, cur_tgt_l = exec_cfg.core_target_pct, exec_cfg.tactical_target_pct, exec_cfg.liquid_target_pct
    cur_allow_entries = True
    cur_hurdle_adder = 0.0
    cur_size_mult = 1.0
    prev_reg_for_hyst = str(reg_seq[t_indices[0]])
    reg_persist_count = policy.allocation_confirm_days

    def _record_cost(c_dict: dict[str, Any], pillar: str) -> None:
        nonlocal total_cost_tactical, total_cost_core, total_cost_liquid
        ct = float(c_dict["total_cost"])
        if pillar == "TACTICAL":
            total_cost_tactical += ct
        elif pillar == "CORE":
            total_cost_core += ct
        else:
            total_cost_liquid += ct

    for step_k, ti in enumerate(t_indices):
        dt = dates[ti]
        prev_ti = ti - 1
        is_last = (ti == t_last)
        cur_reg = str(reg_seq[ti])

        if cur_reg == prev_reg_for_hyst:
            reg_persist_count += 1
        else:
            prev_reg_for_hyst = cur_reg
            reg_persist_count = 1

        # 0A. Release T+1 settled cash
        still_unsettled: list[tuple[int, float]] = []
        for s_ti, amt in unsettled_queue:
            if ti >= s_ti:
                settled_cash += amt
            else:
                still_unsettled.append((s_ti, amt))
        unsettled_queue = still_unsettled

        # 0B. Accrue overnight Liquid BeES daily fractional unit yield
        lb_px_o = float(lb_open[ti]) if not np.isnan(lb_open[ti]) and lb_open[ti] > 0 else 1000.0
        lb_px_c = float(lb_close[ti]) if not np.isnan(lb_close[ti]) and lb_close[ti] > 0 else 1000.0
        if step_k > 0 and liquid_units > 1e-6:
            liq_yield_today = liquid_units * lb_px_c * LIQUID_BEES_DAILY_YIELD
            liquid_units += liq_yield_today / max(1.0, lb_px_c)
            cum_liquid_yield_inr += liq_yield_today

        # 0C. Apply Ex-Dividend Cash Crediting & Stop-Loss Protection
        if exec_cfg.enable_dividend_accounting and open_pos:
            for col, pos in open_pos.items():
                d_adj = float(div_per_adj[ti, col])
                if d_adj > 1e-8 and ti > int(pos["entry_ti"]):
                    div_ev = apply_ex_dividend_accounting_and_stop_adjustment(pos=pos, ti=ti, dt=dt, div_per_adj_share=d_adj)
                    if div_ev["credited_cash_inr"] > 0:
                        settled_cash += div_ev["credited_cash_inr"]
                        cum_dividends_inr += div_ev["credited_cash_inr"]

        # 1. Execute Pending Tactical Sells
        for col in list(pending_tact_sells):
            if col not in open_pos or not panel["is_valid_eq"][ti, col]:
                continue
            pos = open_pos[col]
            prev_c_v = float(close_p[prev_ti, col]) if prev_ti >= 0 and not np.isnan(close_p[prev_ti, col]) else float(pos["last_px"])
            fill = simulate_execution_fill(
                execution_mode=exec_cfg.execution_mode,
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
                max_bar_vol_part=max(0.05, exec_cfg.max_daily_vol_participation),
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
                slippage_bps=exec_cfg.slippage_override_bps,
                adv_20d_inr=float(adv_20d_mat[ti, col]),
                daily_vol_frac=float(vol_20d_mat[ti, col]),
                hl_range_pct=float(hl_range_mat[ti, col]),
                execution_mode=exec_cfg.execution_mode,
                dp_already_charged_today=((dt, sym) in dp_pairs),
            )
            if sc["dp_charge"] > 0:
                dp_pairs.add((dt, sym))
            net_proc = float(sc["exec_notional"]) - float(sc["regulatory_total"])
            if exec_cfg.settlement_lag_days <= 0:
                settled_cash += net_proc
            else:
                unsettled_queue.append((ti + exec_cfg.settlement_lag_days, net_proc))
            _record_cost(sc, "TACTICAL")
            turnover_tactical += float(sc["exec_notional"])
            pnl_price = net_proc - pos["cost_basis"]
            pnl_total = pnl_price + pos["dividends_received_inr"]
            closed_trades.append({
                "sec_id": str(sec_ids[col]),
                "symbol": sym,
                "sector": str(sectors[col]),
                "strategy_id": pos["strategy_id"],
                "entry_date": pos["entry_date"],
                "exit_date": dt,
                "entry_regime": pos["entry_regime"],
                "exit_regime": cur_reg,
                "holding_days": pos["bars_held"],
                "buy_notional_inr": round(pos["buy_exec_notional"], 2),
                "sell_notional_inr": round(float(sc["exec_notional"]), 2),
                "trade_cost_inr": round(pos["buy_total_cost"] + float(sc["total_cost"]), 2),
                "dividends_received_inr": round(pos["dividends_received_inr"], 2),
                "pnl_amount": round(pnl_price, 2),
                "total_pnl_with_div_inr": round(pnl_total, 2),
                "gross_pnl_with_div_inr": round(pnl_total + pos["buy_total_cost"] + float(sc["total_cost"]), 2),
                "net_return_pct": round((pnl_total / max(1e-6, pos["cost_basis"])) * 100.0, 2),
                "mae_pct": round(pos["mae_pct"], 2),
                "mfe_pct": round(pos["mfe_pct"], 2),
                "exit_reason": "INTELLIGENT_REPLACE",
            })
            del open_pos[col]
        pending_tact_sells.clear()

        nb_px_o = float(nb_open[ti]) if not np.isnan(nb_open[ti]) and nb_open[ti] > 0 else prev_core_px
        nb_px_c = float(nb_close[ti]) if not np.isnan(nb_close[ti]) and nb_close[ti] > 0 else nb_px_o

        def _redeem_liquid(needed_inr: float) -> None:
            nonlocal liquid_units, liquid_cost_basis, realized_liquid_pnl_inr, settled_cash, turnover_liquid
            if needed_inr <= 500.0 or liquid_units * lb_px_o < 500.0:
                return
            sell_amt = min(needed_inr * 1.005, liquid_units * lb_px_o)
            units_sell = min(liquid_units, sell_amt / max(1.0, lb_px_o))
            lc = ce.calculate_leg_cost(
                instrument_class="LIQUID_ETF",
                side="SELL",
                raw_notional=units_sell * lb_px_o,
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
            _record_cost(lc, "LIQUID")
            turnover_liquid += float(lc["exec_notional"])

        def _trade_core(target_core_inr: float, *, force_trim: bool = False) -> None:
            nonlocal core_qty, core_cost_basis, realized_core_pnl_inr, settled_cash, turnover_core
            cur_core_val = core_qty * nb_px_o
            cur_total_est = settled_cash + sum(a for _, a in unsettled_queue) + cur_core_val + (liquid_units * lb_px_o) + sum(p["last_px"] * p["qty"] for p in open_pos.values())
            band_inr = cur_total_est * (exec_cfg.core_rebal_band_pct / 100.0)
            diff_inr = target_core_inr - cur_core_val
            if (not force_trim) and abs(diff_inr) < max(exec_cfg.min_core_trade_inr, band_inr):
                return
            if diff_inr > exec_cfg.min_core_trade_inr:
                if settled_cash < diff_inr:
                    _redeem_liquid(diff_inr - settled_cash)
                avail_core = min(diff_inr, settled_cash * 0.996)
                buy_qty = int(avail_core // max(1.0, nb_px_o * 1.001))
                if buy_qty < 1 or (buy_qty * nb_px_o) < exec_cfg.min_core_trade_inr:
                    return
                cc = ce.calculate_leg_cost(
                    instrument_class="EQUITY_ETF", side="BUY", raw_notional=buy_qty * nb_px_o,
                    trade_date=dt, adv_20d_inr=1_500_000_000.0, daily_vol_frac=0.008,
                )
                tot_out = float(cc["exec_notional"]) + float(cc["regulatory_total"])
                if tot_out > settled_cash:
                    buy_qty = int((settled_cash * 0.995) // max(1.0, nb_px_o * 1.001))
                    if buy_qty < 1:
                        return
                    cc = ce.calculate_leg_cost(
                        instrument_class="EQUITY_ETF", side="BUY", raw_notional=buy_qty * nb_px_o,
                        trade_date=dt, adv_20d_inr=1_500_000_000.0, daily_vol_frac=0.008,
                    )
                    tot_out = float(cc["exec_notional"]) + float(cc["regulatory_total"])
                if tot_out <= settled_cash:
                    settled_cash -= tot_out
                    core_qty += buy_qty
                    core_cost_basis += tot_out
                    _record_cost(cc, "CORE")
                    turnover_core += float(cc["exec_notional"])
            elif diff_inr < -exec_cfg.min_core_trade_inr and core_qty > 0:
                sell_qty = min(core_qty, int(abs(diff_inr) // max(1.0, nb_px_o)))
                if sell_qty < 1:
                    return
                cc = ce.calculate_leg_cost(
                    instrument_class="EQUITY_ETF", side="SELL", raw_notional=sell_qty * nb_px_o,
                    trade_date=dt, adv_20d_inr=1_500_000_000.0, daily_vol_frac=0.008,
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
                if exec_cfg.settlement_lag_days <= 0:
                    settled_cash += net_in
                else:
                    unsettled_queue.append((ti + exec_cfg.settlement_lag_days, net_in))
                _record_cost(cc, "CORE")
                turnover_core += float(cc["exec_notional"])

        # 2. Fund queued Tactical Buys
        if pending_tact_buys and pending_core_target_inr is not None and (core_qty * nb_px_o) > pending_core_target_inr + exec_cfg.min_core_trade_inr:
            _trade_core(pending_core_target_inr, force_trim=True)

        if pending_tact_buys:
            est_buy = sum(float(o["req_qty"]) * float(open_p[ti, int(o["col"])]) for o in pending_tact_buys if not np.isnan(open_p[ti, int(o["col"])]))
            if est_buy > settled_cash:
                _redeem_liquid(est_buy - settled_cash)

        # 3. Execute Pending Tactical Buys
        next_pending: list[dict[str, Any]] = []
        for order in pending_tact_buys:
            col = int(order["col"])
            if col in open_pos or len(open_pos) >= exec_cfg.max_tactical_positions or not panel["is_valid_eq"][ti, col]:
                continue
            prev_c_v = float(close_p[prev_ti, col]) if prev_ti >= 0 and not np.isnan(close_p[prev_ti, col]) else float(open_p[ti, col])
            atr_v = float(atr_14_mat[ti, col]) if not np.isnan(atr_14_mat[ti, col]) and atr_14_mat[ti, col] > 0 else float(open_p[ti, col]) * 0.025
            fill = simulate_execution_fill(
                execution_mode=exec_cfg.execution_mode,
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
                max_bar_vol_part=exec_cfg.max_daily_vol_participation,
                min_viable_notional_inr=exec_cfg.min_tactical_lot_inr * 0.70,
            )
            if fill["status"] == "UNFILLED":
                if ti < int(order.get("expires_ti", ti)):
                    next_pending.append(order)
                continue
            if fill["status"] == "FAILED" or fill["fill_qty"] < 1:
                continue

            raw_fill_px = float(fill["raw_fill_px"])
            fill_qty = min(int(fill["fill_qty"]), int((settled_cash * 0.996) // max(1e-6, raw_fill_px * 1.003)))
            if fill_qty < 1 or (fill_qty * raw_fill_px) < exec_cfg.min_tactical_lot_inr * 0.70:
                if ti < int(order.get("expires_ti", ti)):
                    next_pending.append(order)
                continue

            bc = ce.calculate_leg_cost(
                instrument_class="DELIVERY",
                side="BUY",
                raw_notional=fill_qty * raw_fill_px,
                trade_date=dt,
                slippage_bps=exec_cfg.slippage_override_bps,
                adv_20d_inr=float(adv_20d_mat[ti, col]),
                daily_vol_frac=float(vol_20d_mat[ti, col]),
                hl_range_pct=float(hl_range_mat[ti, col]),
                execution_mode=exec_cfg.execution_mode,
            )
            tot_out = float(bc["exec_notional"]) + float(bc["regulatory_total"])
            if tot_out > settled_cash:
                continue
            settled_cash -= tot_out
            _record_cost(bc, "TACTICAL")
            turnover_tactical += float(bc["exec_notional"])

            sl_r = float(order.get("stop_ratio") or 0.09)
            eff_entry = float(bc["exec_notional"]) / fill_qty
            open_pos[col] = {
                "qty": fill_qty,
                "symbol": str(cur_syms[col]),
                "entry_date": dt,
                "entry_ti": ti,
                "entry_regime": cur_reg,
                "entry_px": eff_entry,
                "last_px": eff_entry,
                "cost_basis": tot_out,
                "buy_exec_notional": float(bc["exec_notional"]),
                "buy_total_cost": float(bc["total_cost"]),
                "strategy_id": str(order["strategy_id"]),
                "entry_score": float(order["score"]),
                "bars_held": 0,
                "hold_max_days": int(order.get("hold_max_days") or exec_cfg.hold_max_days),
                "sl_px": eff_entry * (1.0 - sl_r),
                "atr_entry": max(eff_entry * 0.015, eff_entry * sl_r / 2.6),
                "max_px": eff_entry,
                "mae_pct": 0.0,
                "mfe_pct": 0.0,
                "be_locked": False,
                "dividends_received_inr": 0.0,
            }
        pending_tact_buys = next_pending

        # 4. Core Rebalance & Liquid Sweep
        if pending_core_target_inr is not None and not is_last:
            _trade_core(pending_core_target_inr)
            pending_core_target_inr = None

        if pending_liquid_sweep and not is_last:
            reserved_pending = sum(
                float(o["req_qty"]) * float(close_p[ti, int(o["col"])]) * 1.005
                for o in pending_tact_buys
                if not np.isnan(close_p[ti, int(o["col"])])
            )
            excess_cash = settled_cash - reserved_pending - exec_cfg.liquid_sweep_min_cash_inr
            if excess_cash >= max(2_500.0, exec_cfg.liquid_sweep_min_cash_inr * 0.75):
                buy_liq_amt = excess_cash * 0.997
                lc = ce.calculate_leg_cost(instrument_class="LIQUID_ETF", side="BUY", raw_notional=buy_liq_amt, trade_date=dt)
                tot_out = float(lc["exec_notional"]) + float(lc["regulatory_total"])
                if tot_out <= settled_cash:
                    settled_cash -= tot_out
                    liquid_units += buy_liq_amt / max(1.0, lb_px_o)
                    liquid_cost_basis += tot_out
                    _record_cost(lc, "LIQUID")
                    turnover_liquid += float(lc["exec_notional"])
            pending_liquid_sweep = False

        # 5. Stop-Loss, MAE/MFE Update & Exit Evaluation
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
                        "strategy_id": pos["strategy_id"],
                        "entry_date": pos["entry_date"],
                        "exit_date": dt,
                        "entry_regime": pos["entry_regime"],
                        "exit_regime": cur_reg,
                        "holding_days": pos["bars_held"],
                        "buy_notional_inr": round(pos["buy_exec_notional"], 2),
                        "sell_notional_inr": round(raw_val, 2),
                        "trade_cost_inr": round(pos["buy_total_cost"], 2),
                        "dividends_received_inr": round(pos["dividends_received_inr"], 2),
                        "pnl_amount": round(pnl_price, 2),
                        "total_pnl_with_div_inr": round(pnl_total, 2),
                        "gross_pnl_with_div_inr": round(pnl_total + pos["buy_total_cost"], 2),
                        "net_return_pct": round((pnl_total / max(1e-6, pos["cost_basis"])) * 100.0, 2),
                        "mae_pct": round(pos["mae_pct"], 2),
                        "mfe_pct": round(pos["mfe_pct"], 2),
                        "exit_reason": "END_OF_WINDOW_LAST_PX",
                    })
                    del open_pos[col]
                continue

            o_v, h_v, l_v, c_v = float(open_p[ti, col]), float(high_p[ti, col]), float(low_p[ti, col]), float(close_p[ti, col])
            pos["last_px"] = c_v
            # Track exact intraday MAE & MFE relative to entry_px
            ex_lo = ((l_v / max(1e-6, pos["entry_px"])) - 1.0) * 100.0
            ex_hi = ((h_v / max(1e-6, pos["entry_px"])) - 1.0) * 100.0
            pos["mae_pct"] = min(float(pos["mae_pct"]), ex_lo)
            pos["mfe_pct"] = max(float(pos["mfe_pct"]), ex_hi)

            s50 = float(sma_50d[ti, col]) if not np.isnan(sma_50d[ti, col]) else None
            l20 = float(low_20d_prev_mat[ti, col]) if not np.isnan(low_20d_prev_mat[ti, col]) else None
            exit_reason, exit_px = evaluate_position_exit(
                pos=pos,
                stop_model=exec_cfg.stop_model,
                o_v=o_v, h_v=h_v, l_v=l_v, c_v=c_v,
                sma_50_v=s50, low_20_prev=l20,
                vol_20d_daily=float(vol_20d_mat[ti, col]),
                hold_max_days=int(pos["hold_max_days"]),
                is_last_bar=is_last,
            )

            if exit_reason is not None and exit_px is not None:
                sym = str(cur_syms[col])
                sc = ce.calculate_leg_cost(
                    instrument_class="DELIVERY",
                    side="SELL",
                    raw_notional=pos["qty"] * exit_px,
                    trade_date=dt,
                    slippage_bps=exec_cfg.slippage_override_bps,
                    adv_20d_inr=float(adv_20d_mat[ti, col]),
                    daily_vol_frac=float(vol_20d_mat[ti, col]),
                    hl_range_pct=float(hl_range_mat[ti, col]),
                    execution_mode="NEXT_SESSION_OPEN",
                    dp_already_charged_today=((dt, sym) in dp_pairs),
                )
                if sc["dp_charge"] > 0:
                    dp_pairs.add((dt, sym))
                net_proc = float(sc["exec_notional"]) - float(sc["regulatory_total"])
                if is_last or exec_cfg.settlement_lag_days <= 0:
                    settled_cash += net_proc
                else:
                    unsettled_queue.append((ti + exec_cfg.settlement_lag_days, net_proc))
                _record_cost(sc, "TACTICAL")
                turnover_tactical += float(sc["exec_notional"])
                pnl_price = net_proc - pos["cost_basis"]
                pnl_total = pnl_price + pos["dividends_received_inr"]
                closed_trades.append({
                    "sec_id": str(sec_ids[col]),
                    "symbol": sym,
                    "sector": str(sectors[col]),
                    "strategy_id": pos["strategy_id"],
                    "entry_date": pos["entry_date"],
                    "exit_date": dt,
                    "entry_regime": pos["entry_regime"],
                    "exit_regime": cur_reg,
                    "holding_days": pos["bars_held"],
                    "buy_notional_inr": round(pos["buy_exec_notional"], 2),
                    "sell_notional_inr": round(float(sc["exec_notional"]), 2),
                    "trade_cost_inr": round(pos["buy_total_cost"] + float(sc["total_cost"]), 2),
                    "dividends_received_inr": round(pos["dividends_received_inr"], 2),
                    "pnl_amount": round(pnl_price, 2),
                    "total_pnl_with_div_inr": round(pnl_total, 2),
                    "gross_pnl_with_div_inr": round(pnl_total + pos["buy_total_cost"] + float(sc["total_cost"]), 2),
                    "net_return_pct": round((pnl_total / max(1e-6, pos["cost_basis"])) * 100.0, 2),
                    "mae_pct": round(pos["mae_pct"], 2),
                    "mfe_pct": round(pos["mfe_pct"], 2),
                    "exit_reason": exit_reason,
                })
                if "STOP_LOSS" in exit_reason:
                    cooldown_until[col] = ti + exec_cfg.cooldown_days
                del open_pos[col]
                vacated_slot_flag = True

        # 6. End-of-Day Four-Ledger Mark-to-Market
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
        if open_pos and prev_ti >= 0:
            t_num = 0.0
            t_den = 0.0
            for p_col, p_obj in open_pos.items():
                c_now, c_prev = float(close_p[ti, p_col]), float(close_p[prev_ti, p_col])
                if not np.isnan(c_now) and not np.isnan(c_prev) and c_prev > 0:
                    t_num += p_obj["qty"] * (c_now - c_prev)
                    t_den += p_obj["qty"] * c_prev
            tact_d_ret = (t_num / t_den) if t_den > 1000.0 else core_d_ret
        else:
            tact_d_ret = core_d_ret

        port_ret_history.append(d_ret)
        core_ret_history.append(core_d_ret)
        tact_ret_history.append(float(np.clip(tact_d_ret, -0.15, 0.15)))

        core_pct_now = (core_mkt / max(1.0, nav)) * 100.0
        tact_pct_now = (tact_mkt / max(1.0, nav)) * 100.0
        liquid_pct_now = (liquid_mkt / max(1.0, nav)) * 100.0
        idle_pct = ((settled_cash + unsettled_amt) / max(1.0, nav)) * 100.0

        daily_navs.append({
            "date": dt,
            "nav": round(nav, 2),
            "core_pct": round(core_pct_now, 2),
            "tactical_pct": round(tact_pct_now, 2),
            "liquid_pct": round(liquid_pct_now, 2),
            "uninvested_cash_pct": round(idle_pct, 2),
            "productive_capital_pct": round(core_pct_now + tact_pct_now + liquid_pct_now, 2),
            "open_tactical_positions": len(open_pos),
            "regime": cur_reg,
            "drawdown_pct": round(cur_dd_pct, 2),
            "ret": d_ret,
            "nb_ret": core_d_ret,
        })
        prev_nav = nav
        prev_core_px = nb_px_c

        if is_last:
            continue

        # 7. EOD Regime-Aware Allocation, Replacement & Signal Deployment
        is_periodic = (step_k == 0) or (step_k % max(1, exec_cfg.periodic_review_interval_days) == 0)
        is_event = (
            exec_cfg.event_driven_replenishment
            and vacated_slot_flag
            and (step_k - last_replenish_step >= exec_cfg.replenish_min_gap_days)
            and exec_cfg.tactical_target_pct > 0.0
        )
        is_confirmed_regime_shift = (
            policy.regime_source != "NONE"
            and reg_persist_count == max(1, policy.allocation_confirm_days)
            and cur_reg in policy_map
        )
        needs_sweep = (settled_cash > max(exec_cfg.liquid_sweep_min_cash_inr + 1_500.0, nav * 0.045))
        if not (is_periodic or is_event or is_confirmed_regime_shift or needs_sweep):
            continue

        # Gather causal candidate pool
        merged_by_col: dict[int, dict[str, Any]] = {}
        keep_cols: set[int] = set()
        for sid in exec_cfg.active_sleeves:
            s_w = float(sleeve_w_map.get(sid, 1.0 / max(1, len(exec_cfg.active_sleeves))))
            c_list = pre_cands.get(sid, {}).get(ti, [])
            for c_keep in c_list[:20]:
                keep_cols.add(int(c_keep["col"]))
            for rank_idx, c in enumerate(c_list[:12]):
                col = int(c["col"])
                prio = s_w / (1.0 + 0.25 * rank_idx) + 0.15 * float(c["score"])
                if col not in merged_by_col or prio > merged_by_col[col]["composite_score"]:
                    sc_ref = ctx["strat_cfgs"].get(sid)
                    hm_d = max(exec_cfg.hold_max_days, sc_ref.hold_max_days) if sc_ref is not None else exec_cfg.hold_max_days
                    merged_by_col[col] = {**c, "strategy_id": sid, "hold_max_days": hm_d, "composite_score": prio}

        ranked_cands = sorted(merged_by_col.values(), key=lambda x: -float(x["composite_score"]))
        avg_sig_score = float(np.mean([float(c["score"]) for c in ranked_cands[:6]])) if ranked_cands else 0.0

        # Update pillar targets cleanly (either via Policy regime schedule OR Prompt 3 dynamic allocator)
        if policy.regime_source != "NONE" and cur_reg in policy_map and (is_periodic or is_confirmed_regime_shift or step_k == 0):
            rc, rt, rl, allow_e, h_add, s_mult = policy_map[cur_reg]
            # Verify strict 100% reconciliation
            tot_w = max(1e-6, rc + rt + rl)
            cur_tgt_c, cur_tgt_t, cur_tgt_l = (rc / tot_w) * 100.0, (rt / tot_w) * 100.0, (rl / tot_w) * 100.0
            cur_allow_entries = allow_e
            cur_hurdle_adder = h_add
            cur_size_mult = s_mult
        elif policy.regime_source == "NONE" and (is_periodic or is_event):
            p_vol20 = float(np.std(port_ret_history[-20:], ddof=1)) * math.sqrt(252.0) * 100.0 if len(port_ret_history) >= 15 else 11.0
            c_vol20 = float(np.std(core_ret_history[-20:], ddof=1)) * math.sqrt(252.0) * 100.0 if len(core_ret_history) >= 15 else 13.0
            t_vol20 = float(np.std(tact_ret_history[-20:], ddof=1)) * math.sqrt(252.0) * 100.0 if len(tact_ret_history) >= 15 else 15.0
            cur_tgt_c, cur_tgt_t, cur_tgt_l = _compute_dynamic_pillar_targets(
                cfg=alloc_cfg_view,
                cur_regime=str(ctx["regime_rule5"][ti]),
                cur_dd_pct=cur_dd_pct,
                port_vol_20d_ann=p_vol20,
                core_vol_20d_ann=c_vol20,
                tact_vol_20d_ann=t_vol20,
                avg_signal_score=avg_sig_score,
                signal_count=len(ranked_cands),
            )
            cur_allow_entries, cur_hurdle_adder, cur_size_mult = True, 0.0, 1.0

        # Position Replacement Check
        if is_periodic and open_pos and cur_tgt_t > 0.0:
            sma50_row = sma_50d[ti]
            best_new_c = next((c for c in ranked_cands if int(c["col"]) not in open_pos), None)
            best_new_ev: dict[str, Any] | None = None
            if best_new_c is not None:
                bcol = int(best_new_c["col"])
                b_dy = float(div_yield_ttm[ti, bcol]) if not np.isnan(div_yield_ttm[ti, bcol]) else 0.0
                best_new_ev = estimate_trade_economic_value(
                    candidate=best_new_c,
                    lot_notional_inr=max(exec_cfg.min_tactical_lot_inr, nav * (cur_tgt_t / 100.0) / 3.0),
                    trade_date=dt,
                    daily_vol_frac=float(vol_20d_mat[ti, bcol]),
                    hl_range_pct=float(hl_range_mat[ti, bcol]),
                    adv_20d_inr=float(adv_20d_mat[ti, bcol]),
                    div_yield_ttm_frac=b_dy,
                    regime=str(ctx["regime_rule5"][ti]),
                    cost_engine=ce,
                    execution_mode=exec_cfg.execution_mode,
                    hurdle_mode=exec_cfg.hurdle_mode,
                    slippage_override_bps=exec_cfg.slippage_override_bps,
                )
            sec_val_now: dict[str, float] = {}
            for oc, op in open_pos.items():
                sn = str(sectors[oc])
                sec_val_now[sn] = sec_val_now.get(sn, 0.0) + op["last_px"] * op["qty"]
            for col, pos in open_pos.items():
                s50_v = float(sma50_row[col]) if not np.isnan(sma50_row[col]) else None
                rep = evaluate_position_replacement(
                    pos=pos,
                    cur_close_px=float(close_p[ti, col]),
                    sma_50_px=s50_v,
                    in_keep_buffer=(col in keep_cols),
                    best_candidate=best_new_c,
                    best_cand_ev=best_new_ev,
                    nav_inr=nav,
                    trade_date=dt,
                    regime=str(ctx["regime_rule5"][ti]),
                    sector_exposure_pct=(sec_val_now.get(str(sectors[col]), 0.0) / max(1.0, nav)) * 100.0,
                    cost_engine=ce,
                    replacement_policy=exec_cfg.replacement_policy,
                    min_improvement_bps=exec_cfg.min_replacement_edge_bps,
                    min_hold_days_before_rotate=exec_cfg.min_hold_days_before_rotate,
                    protect_winner_pct=exec_cfg.protect_winner_pct,
                )
                if rep["should_replace"]:
                    pending_tact_sells.add(col)

        # Tactical Deployment (Respecting Regime Entry Gating & Hurdle Adders!)
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
            s_nm, i_nm = str(sectors[col]), str(industries[col])
            if bool(is_unknown_arr[col]) or s_nm == UNKNOWN_SECTOR_LABEL:
                unknown_exp_inr += v_pos
            else:
                sec_counts[s_nm] = sec_counts.get(s_nm, 0) + 1
                sec_exp_inr[s_nm] = sec_exp_inr.get(s_nm, 0.0) + v_pos
                ind_exp_inr[i_nm] = ind_exp_inr.get(i_nm, 0.0) + v_pos

        active_cols = list(retained_cols)
        if (is_periodic or is_event) and cur_tgt_t > 0.0 and rem_tact_budget >= exec_cfg.min_tactical_lot_inr:
            pending_tact_buys.clear()
            if not cur_allow_entries:
                regime_blocked_entries += min(3, len(ranked_cands))
            else:
                max_sec_inr = nav * (exec_cfg.max_sector_exposure_pct / 100.0)
                max_ind_inr = nav * (exec_cfg.max_industry_exposure_pct / 100.0)
                max_unk_inr = nav * (exec_cfg.max_unknown_sector_exposure_pct / 100.0)
                max_single_inr = nav * (exec_cfg.max_single_stock_pct / 100.0)
                eff_slots = max(1, min(exec_cfg.max_tactical_positions, int(tact_budget_inr // (exec_cfg.min_tactical_lot_inr * 1.35))))
                base_slot_inr = min(max_single_inr, max(exec_cfg.min_tactical_lot_inr * 1.30, (tact_budget_inr / eff_slots) * cur_size_mult))

                for c in ranked_cands:
                    if len(active_cols) >= exec_cfg.max_tactical_positions or rem_tact_budget < exec_cfg.min_tactical_lot_inr:
                        break
                    col = int(c["col"])
                    if col in open_pos or ti < cooldown_until.get(col, -1):
                        continue
                    s_nm, i_nm = str(sectors[col]), str(industries[col])
                    is_unk = bool(is_unknown_arr[col]) or s_nm == UNKNOWN_SECTOR_LABEL
                    if is_unk:
                        if unknown_exp_inr + exec_cfg.min_tactical_lot_inr > max_unk_inr:
                            continue
                        sec_room = ind_room = max(0.0, max_unk_inr - unknown_exp_inr)
                    else:
                        if sec_counts.get(s_nm, 0) >= exec_cfg.max_sector_positions or sec_exp_inr.get(s_nm, 0.0) + exec_cfg.min_tactical_lot_inr > max_sec_inr:
                            continue
                        if ind_exp_inr.get(i_nm, 0.0) + exec_cfg.min_tactical_lot_inr > max_ind_inr:
                            continue
                        sec_room = max(0.0, max_sec_inr - sec_exp_inr.get(s_nm, 0.0))
                        ind_room = max(0.0, max_ind_inr - ind_exp_inr.get(i_nm, 0.0))

                    corrs = [_causal_pairwise_corr_60d(daily_ret, ti, col, ac) for ac in active_cols]
                    max_c = max(corrs) if corrs else 0.0
                    if max_c > exec_cfg.max_pairwise_corr:
                        continue
                    spot = float(close_p[ti, col])
                    if np.isnan(spot) or spot <= 0:
                        continue
                    prelim_lot = min(base_slot_inr, rem_tact_budget, max_single_inr, sec_room, ind_room)
                    if prelim_lot < exec_cfg.min_tactical_lot_inr:
                        continue
                    dy_c = float(div_yield_ttm[ti, col]) if not np.isnan(div_yield_ttm[ti, col]) else 0.0
                    ev = estimate_trade_economic_value(
                        candidate=c,
                        lot_notional_inr=prelim_lot,
                        trade_date=dt,
                        daily_vol_frac=float(vol_20d_mat[ti, col]),
                        hl_range_pct=float(hl_range_mat[ti, col]),
                        adv_20d_inr=float(adv_20d_mat[ti, col]),
                        div_yield_ttm_frac=dy_c,
                        regime=str(ctx["regime_rule5"][ti]),
                        cost_engine=ce,
                        execution_mode=exec_cfg.execution_mode,
                        hurdle_mode=exec_cfg.hurdle_mode,
                        slippage_override_bps=exec_cfg.slippage_override_bps,
                    )
                    if (not ev["approved"]) or (float(ev["expected_net_alpha_bps"]) < float(ev["hurdle_bps"]) + cur_hurdle_adder):
                        continue

                    alloc_amt = compute_intelligent_position_size(
                        sizing_mode=exec_cfg.sizing_mode,
                        base_slot_inr=base_slot_inr,
                        rem_tact_budget_inr=rem_tact_budget,
                        nav_inr=nav,
                        max_single_stock_pct=exec_cfg.max_single_stock_pct,
                        sector_room_inr=sec_room,
                        industry_room_inr=ind_room,
                        adv_20d_inr=float(adv_20d_mat[ti, col]),
                        daily_vol_frac=float(vol_20d_mat[ti, col]),
                        stop_ratio=float(c.get("stop_ratio") or 0.09),
                        signal_score=float(c["score"]),
                        expected_net_alpha_bps=float(ev["expected_net_alpha_bps"]),
                        hurdle_bps=float(ev["hurdle_bps"]),
                        max_corr_with_open=max_c,
                        min_lot_inr=exec_cfg.min_tactical_lot_inr,
                    )
                    qty = int(alloc_amt // spot)
                    lot_val = qty * spot
                    if qty < 1 or lot_val < exec_cfg.min_tactical_lot_inr:
                        continue
                    pending_tact_buys.append({**c, "req_qty": qty, "expires_ti": ti + exec_cfg.settlement_lag_days + 2})
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

        # Passive Core Target INR
        pending_buys_inr = sum(
            float(o["req_qty"]) * float(close_p[ti, int(o["col"])]) * 1.005
            for o in pending_tact_buys
            if not np.isnan(close_p[ti, int(o["col"])])
        )
        # In Stress/Bear defensive halt, route unused tactical budget into LIQUIDBEES (NOT Core equities!)
        if policy.regime_source != "NONE" and (not cur_allow_entries):
            eff_core_target_inr = nav * (cur_tgt_c / 100.0)
        elif exec_cfg.sweep_unused_tactical_to_core and str(ctx["regime_rule5"][ti]) in ("STRONG_BULL", "WEAK_BULL", "SIDEWAYS_CHOP"):
            eff_core_target_inr = nav * (cur_tgt_c / 100.0) + max(0.0, rem_tact_budget)
        else:
            eff_core_target_inr = nav * (cur_tgt_c / 100.0)
        eff_core_target_inr = max(
            0.0,
            min(eff_core_target_inr, nav * (1.0 - max(0.0, cur_tgt_l / 100.0)) - retained_tact_inr - pending_buys_inr),
        )
        pending_core_target_inr = eff_core_target_inr
        pending_liquid_sweep = True

    # Metrics & Exact 4-Ledger Reconciliation
    nav_arr = np.array([float(d["nav"]) for d in daily_navs], dtype=np.float64)
    ret_arr = np.array([float(d["ret"]) for d in daily_navs], dtype=np.float64)
    n_days = len(nav_arr)
    years = max(0.05, (n_days - 1) / 252.0)

    final_nav = float(nav_arr[-1])
    total_all_costs = total_cost_tactical + total_cost_core + total_cost_liquid
    total_all_turnover = turnover_tactical + turnover_core + turnover_liquid
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

    core_total_pnl = realized_core_pnl_inr + ((core_qty * float(nb_close[t_last])) - core_cost_basis)
    lb_last_px = float(lb_close[t_last]) if not np.isnan(lb_close[t_last]) and lb_close[t_last] > 0 else 1000.0
    liquid_total_pnl = realized_liquid_pnl_inr + ((liquid_units * lb_last_px) - liquid_cost_basis)
    tact_price_pnl = sum(float(t["pnl_amount"]) for t in closed_trades)
    recon_diff = (final_nav - initial_capital) - (core_total_pnl + tact_price_pnl + cum_dividends_inr + liquid_total_pnl)

    n_tr = len(closed_trades)
    if n_tr > 0:
        pnls = [float(t["total_pnl_with_div_inr"]) for t in closed_trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        pf = (sum(wins) / max(1e-6, abs(sum(losses)))) if losses else (99.0 if wins else 0.0)
        win_rate = 100.0 * len(wins) / n_tr
        avg_hold = float(np.mean([int(t["holding_days"]) for t in closed_trades]))
    else:
        pf = win_rate = avg_hold = 0.0

    nb_ret_win = ((float(nb_close[t_last]) / max(1.0, float(nb_close[t_indices[0]]))) - 1.0) * 100.0

    metrics = {
        "policy_id": policy.policy_id,
        "policy_name": policy.policy_name,
        "config_id": exec_cfg.config_id,
        "window": f"{dates[t_indices[0]]} -> {dates[t_last]} ({n_days}d)",
        "final_nav_inr": round(final_nav, 2),
        "total_return_pct": round(tot_ret_pct, 2),
        "cagr_pct": round(cagr_pct, 2),
        "annualized_vol_pct": round(ann_vol, 2),
        "max_drawdown_pct": round(mdd_pct, 2),
        "sharpe_ratio": round(sharpe, 2),
        "sortino_ratio": round(sortino, 2),
        "calmar_ratio": round(calmar, 2),
        "profit_factor": round(min(99.0, pf), 2),
        "win_rate_pct": round(win_rate, 1),
        "avg_holding_days": round(avg_hold, 1),
        "num_trades": n_tr,
        "total_turnover_x": round(total_all_turnover / initial_capital, 2),
        "tactical_turnover_x": round(turnover_tactical / initial_capital, 2),
        "total_cost_inr": round(total_all_costs, 2),
        "avg_core_pct": round(float(np.mean([d["core_pct"] for d in daily_navs])), 1),
        "avg_tactical_pct": round(float(np.mean([d["tactical_pct"] for d in daily_navs])), 1),
        "avg_liquid_pct": round(float(np.mean([d["liquid_pct"] for d in daily_navs])), 1),
        "productive_capital_pct": round(float(np.mean([d["productive_capital_pct"] for d in daily_navs])), 1),
        "excess_vs_niftybees_pp": round(tot_ret_pct - nb_ret_win, 2),
        "regime_blocked_entries": regime_blocked_entries,
        "reconciliation_diff_inr": round(recon_diff, 2),
    }

    return {
        "config": asdict(exec_cfg),
        "policy": asdict(policy),
        "metrics": metrics,
        "daily_navs": daily_navs,
        "trades": closed_trades,
    }


def analyze_family_e_by_regime(
    *,
    sim: dict[str, Any],
    regime_states: tuple[str, ...],
) -> dict[str, Any]:
    """Compute complete Section 5 conditional-edge diagnostics for Family E across every regime."""
    daily = sim["daily_navs"]
    trades = sim["trades"]

    rows: list[dict[str, Any]] = []
    for st in regime_states:
        d_st = [d for d in daily[1:] if str(d["regime"]) == st]
        t_st = [t for t in trades if str(t.get("entry_regime")) == st]

        n_days = len(d_st)
        n_tr = len(t_st)
        if d_st:
            p_rets = np.array([float(d["ret"]) for d in d_st], dtype=np.float64)
            nb_rets = np.array([float(d["nb_ret"]) for d in d_st], dtype=np.float64)
            cum_port = (float(np.prod(1.0 + p_rets)) - 1.0) * 100.0
            cum_nb = (float(np.prod(1.0 + nb_rets)) - 1.0) * 100.0
            eq_st = np.cumprod(1.0 + p_rets)
            pk_st = np.maximum.accumulate(eq_st)
            dd_st = abs(float(np.min((eq_st - pk_st) / np.maximum(1e-6, pk_st)))) * 100.0
        else:
            cum_port = cum_nb = dd_st = 0.0

        if n_tr > 0:
            net_pnls = [float(t["total_pnl_with_div_inr"]) for t in t_st]
            gross_pnls = [float(t["gross_pnl_with_div_inr"]) for t in t_st]
            costs = [float(t["trade_cost_inr"]) for t in t_st]
            wins = [p for p in net_pnls if p > 0]
            losses = [p for p in net_pnls if p <= 0]
            wr = 100.0 * len(wins) / n_tr
            avg_w = float(np.mean(wins)) if wins else 0.0
            avg_l = float(np.mean(losses)) if losses else 0.0
            pf = (sum(wins) / max(1e-6, abs(sum(losses)))) if losses else (99.0 if wins else 0.0)
            avg_h = float(np.mean([int(t["holding_days"]) for t in t_st]))
            avg_mae = float(np.mean([float(t["mae_pct"]) for t in t_st]))
            avg_mfe = float(np.mean([float(t["mfe_pct"]) for t in t_st]))
            turn_inr = sum(float(t["buy_notional_inr"]) + float(t["sell_notional_inr"]) for t in t_st)
        else:
            net_pnls = gross_pnls = costs = []
            wr = avg_w = avg_l = pf = avg_h = avg_mae = avg_mfe = turn_inr = 0.0

        rows.append({
            "regime": st,
            "eligible_trading_days": n_days,
            "num_trades_entered": n_tr,
            "win_rate_pct": round(wr, 1),
            "avg_win_inr": round(avg_w, 2),
            "avg_loss_inr": round(avg_l, 2),
            "profit_factor": round(min(99.0, pf), 2),
            "gross_profit_inr": round(sum(gross_pnls), 2),
            "net_profit_inr": round(sum(net_pnls), 2),
            "avg_holding_days": round(avg_h, 1),
            "avg_mae_pct": round(avg_mae, 2),
            "avg_mfe_pct": round(avg_mfe, 2),
            "regime_drawdown_pct": round(dd_st, 2),
            "regime_turnover_x": round(turn_inr / INITIAL_CAPITAL_M3, 2),
            "regime_txn_costs_inr": round(sum(costs), 2),
            "portfolio_cum_ret_pct": round(cum_port, 2),
            "niftybees_cum_ret_pct": round(cum_nb, 2),
            "benchmark_rel_ret_pp": round(cum_port - cum_nb, 2),
        })

    # Security & sector concentration analysis
    by_sym: dict[str, float] = {}
    by_sec: dict[str, float] = {}
    total_tact_pnl = sum(float(t["total_pnl_with_div_inr"]) for t in trades)
    for t in trades:
        sym = str(t["symbol"])
        sec = str(t["sector"])
        p = float(t["total_pnl_with_div_inr"])
        by_sym[sym] = by_sym.get(sym, 0.0) + p
        by_sec[sec] = by_sec.get(sec, 0.0) + p

    top_syms = sorted(by_sym.items(), key=lambda kv: -kv[1])[:5]
    top_secs = sorted(by_sec.items(), key=lambda kv: -kv[1])[:5]
    return {
        "regime_rows": rows,
        "total_tactical_net_pnl_inr": round(total_tact_pnl, 2),
        "top_5_securities": [{"symbol": k, "net_pnl_inr": round(v, 2)} for k, v in top_syms],
        "top_5_sectors": [{"sector": k, "net_pnl_inr": round(v, 2)} for k, v in top_secs],
    }
