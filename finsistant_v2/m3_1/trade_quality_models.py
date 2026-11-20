"""Moreshwar 3.1 (Prompt 3/5) — Trade Economic Value, Replacement, Stop-Loss, Execution Timing & Sizing Models.

Every decision function in this module is strictly causal (uses only data available at or before
signal/execution bar `ti`), deterministic, and integrated with `TransactionCostIntelligence`.
"""

from __future__ import annotations

import math
from typing import Any

from finsistant_v2.m3_1.transaction_cost_intelligence import TransactionCostIntelligence


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


# ---------------------------------------------------------------------------
# 1. TRADE ECONOMIC VALUE MODEL (Section 3)
# ---------------------------------------------------------------------------
def estimate_trade_economic_value(
    *,
    candidate: dict[str, Any],
    lot_notional_inr: float,
    trade_date: str,
    daily_vol_frac: float,
    hl_range_pct: float,
    adv_20d_inr: float,
    div_yield_ttm_frac: float,
    regime: str,
    cost_engine: TransactionCostIntelligence,
    execution_mode: str = "NEXT_SESSION_OPEN",
    hurdle_mode: str = "FIXED_120",  # "FIXED_60"|"FIXED_90"|"FIXED_120"|"FIXED_150"|"FIXED_200"|"DYNAMIC_VOL"
    existing_position_expected_bps: float = 0.0,
    slippage_override_bps: float | None = None,
) -> dict[str, Any]:
    """Estimate complete pre-trade economic value strictly causally before order approval."""
    strat_id = str(candidate.get("strategy_id") or "")
    score = float(candidate.get("score") or 0.5)
    stop_ratio = max(0.04, min(0.25, float(candidate.get("stop_ratio") or 0.09)))
    d_vol = max(0.006, min(0.08, float(daily_vol_frac)))

    # Realistic expected holding period & empirical cross-sectional alpha scale by strategy family
    if "STRAT_E" in strat_id:
        exp_hold_days = 48.0
        base_edge_mult = 200.0
    elif "STRAT_A" in strat_id:
        exp_hold_days = 28.0
        base_edge_mult = 155.0
    else:
        exp_hold_days = 35.0
        base_edge_mult = 175.0

    # Causal regime multiplier (never lookahead; modulates expected drift in bear/chop vs bull)
    reg_mult_map = {
        "STRONG_BULL": 1.15,
        "WEAK_BULL": 1.05,
        "SIDEWAYS_CHOP": 0.88,
        "WEAK_BEAR": 0.70,
        "STRESS_BEAR": 0.55,
    }
    r_mult = reg_mult_map.get(str(regime), 1.0)

    # Expected gross return over expected holding horizon (bps):
    # Signal score alpha component + causal TTM dividend yield carry over holding horizon
    # + bounded volatility-scaled drift (capped tightly so high volatility alone never fakes alpha!)
    div_carry_bps = min(140.0, max(0.0, float(div_yield_ttm_frac)) * (exp_hold_days / 252.0) * 10_000.0)
    signal_edge_bps = max(0.0, score) * base_edge_mult * r_mult
    vol_drift_bps = min(70.0, stop_ratio * 450.0) * r_mult
    exp_gross_return_bps = signal_edge_bps + div_carry_bps + vol_drift_bps

    # Expected volatility & downside over holding horizon
    horizon_vol_bps = d_vol * math.sqrt(exp_hold_days) * 10_000.0
    exp_downside_bps = stop_ratio * 10_000.0

    # Exact entry + exit + slippage + DP cost via centralized cost engine
    rt_cost = cost_engine.estimate_round_trip_cost(
        instrument_class="DELIVERY",
        notional_inr=lot_notional_inr,
        trade_date=trade_date,
        adv_20d_inr=adv_20d_inr,
        daily_vol_frac=d_vol,
        hl_range_pct=hl_range_pct,
        execution_mode=execution_mode,
        slippage_bps=slippage_override_bps,
    )
    rt_cost_bps = float(rt_cost["round_trip_cost_bps"])

    # Expected net alpha over holding period
    exp_net_alpha_bps = exp_gross_return_bps - rt_cost_bps
    exp_risk_adj_ret = exp_net_alpha_bps / max(50.0, horizon_vol_bps)
    # Probability of outperforming existing position / core benchmark
    z_out = (exp_net_alpha_bps - float(existing_position_expected_bps)) / max(100.0, horizon_vol_bps)
    prob_outperform = _norm_cdf(z_out)

    # Determine required minimum expected net alpha hurdle (bps)
    hm = hurdle_mode.upper().strip()
    if hm == "FIXED_60":
        hurdle_bps = 60.0
    elif hm == "FIXED_90":
        hurdle_bps = 90.0
    elif hm == "FIXED_120":
        hurdle_bps = 120.0
    elif hm == "FIXED_150":
        hurdle_bps = 150.0
    elif hm == "FIXED_200":
        hurdle_bps = 200.0
    elif hm == "DYNAMIC_VOL":
        # Scales smoothly with stock daily volatility & actual round-trip friction
        daily_vol_bps = d_vol * 10_000.0
        hurdle_bps = max(95.0, min(260.0, 0.55 * rt_cost_bps + 0.65 * daily_vol_bps))
    else:
        hurdle_bps = 120.0

    approved = (exp_net_alpha_bps >= hurdle_bps) and (prob_outperform >= 0.50)

    return {
        "approved": bool(approved),
        "hurdle_mode": hm,
        "hurdle_bps": round(hurdle_bps, 2),
        "expected_gross_return_bps": round(exp_gross_return_bps, 2),
        "expected_holding_days": round(exp_hold_days, 1),
        "expected_horizon_vol_bps": round(horizon_vol_bps, 2),
        "expected_downside_bps": round(exp_downside_bps, 2),
        "expected_entry_cost_inr": rt_cost["entry_cost_inr"],
        "expected_exit_cost_inr": rt_cost["exit_cost_inr"],
        "expected_slippage_bps": round(rt_cost["entry_slippage_bps"] + rt_cost["exit_slippage_bps"], 2),
        "expected_rt_cost_bps": round(rt_cost_bps, 2),
        "expected_net_alpha_bps": round(exp_net_alpha_bps, 2),
        "expected_risk_adj_return": round(exp_risk_adj_ret, 3),
        "prob_outperform_existing": round(prob_outperform, 4),
    }


# ---------------------------------------------------------------------------
# 2. INTELLIGENT TRADE REPLACEMENT ENGINE (Section 4)
# ---------------------------------------------------------------------------
def evaluate_position_replacement(
    *,
    pos: dict[str, Any],
    cur_close_px: float,
    sma_50_px: float | None,
    in_keep_buffer: bool,
    best_candidate: dict[str, Any] | None,
    best_cand_ev: dict[str, Any] | None,
    nav_inr: float,
    trade_date: str,
    regime: str,
    sector_exposure_pct: float,
    cost_engine: TransactionCostIntelligence,
    replacement_policy: str = "INTELLIGENT_HYSTERESIS",
    min_improvement_bps: float = 180.0,
    min_hold_days_before_rotate: int = 20,
    protect_winner_pct: float = 4.0,
) -> dict[str, Any]:
    """Evaluate all 10 replacement dimensions to prevent low-value position churn."""
    bars_held = int(pos.get("bars_held") or 0)
    entry_px = max(1e-6, float(pos.get("entry_px") or cur_close_px))
    unrealized_pct = ((cur_close_px / entry_px) - 1.0) * 100.0
    pos_val_inr = float(pos.get("qty") or 0) * cur_close_px
    pos_weight_pct = (pos_val_inr / max(1.0, nav_inr)) * 100.0

    trend_intact = (
        (sma_50_px is not None)
        and (not math.isnan(sma_50_px))
        and (cur_close_px >= sma_50_px * 0.96)
    )
    pol = replacement_policy.upper().strip()

    # Exact cost to close existing position + open replacement
    close_leg = cost_engine.calculate_leg_cost(
        instrument_class="DELIVERY",
        side="SELL",
        raw_notional=pos_val_inr,
        trade_date=trade_date,
    )
    open_leg = cost_engine.calculate_leg_cost(
        instrument_class="DELIVERY",
        side="BUY",
        raw_notional=pos_val_inr,
        trade_date=trade_date,
    )
    switch_cost_inr = float(close_leg["total_cost"]) + float(open_leg["total_cost"])
    switch_cost_bps = (switch_cost_inr / max(1.0, pos_val_inr)) * 10_000.0

    # Baseline Prompt 2 legacy rotation rule
    if pol == "PROMPT2_BASELINE":
        if trend_intact or in_keep_buffer:
            return {
                "should_replace": False,
                "reason": "RETAIN_TREND_OR_BUFFER_INTACT",
                "switching_cost_bps": round(switch_cost_bps, 2),
                "net_incremental_edge_bps": 0.0,
            }
        if bars_held >= 15:
            return {
                "should_replace": True,
                "reason": "PROMPT2_TREND_AND_BUFFER_LOST",
                "switching_cost_bps": round(switch_cost_bps, 2),
                "net_incremental_edge_bps": 0.0,
            }
        return {
            "should_replace": False,
            "reason": "RETAIN_MIN_HOLD",
            "switching_cost_bps": round(switch_cost_bps, 2),
            "net_incremental_edge_bps": 0.0,
        }

    # Mode: STRICT_NO_ROTATION (never proactively rotate; let stop/time-exit govern)
    if pol == "NO_PROACTIVE_ROTATION":
        return {
            "should_replace": False,
            "reason": "RETAIN_NO_PROACTIVE_ROTATION",
            "switching_cost_bps": round(switch_cost_bps, 2),
            "net_incremental_edge_bps": 0.0,
        }

    # Estimate current position's remaining expected return (bps)
    hold_max = float(pos.get("hold_max_days") or 120.0)
    rem_hold_frac = max(0.15, min(1.0, (hold_max - bars_held) / max(20.0, hold_max)))
    cur_quality_score = float(pos.get("entry_score") or 0.55)
    if trend_intact:
        cur_quality_score += 0.12
    if in_keep_buffer:
        cur_quality_score += 0.15
    if unrealized_pct >= protect_winner_pct:
        cur_quality_score += 0.12
    cur_expected_rem_bps = max(-100.0, cur_quality_score * 320.0 * rem_hold_frac)

    cand_net_alpha_bps = float(best_cand_ev["expected_net_alpha_bps"]) if (best_cand_ev and best_cand_ev.get("approved")) else 0.0
    net_incremental_bps = cand_net_alpha_bps - cur_expected_rem_bps - switch_cost_bps

    # Regime & concentration hurdle adjustment: raise bar in chop/bear or when over-concentrated
    reg_adder = 60.0 if regime in ("SIDEWAYS_CHOP", "WEAK_BEAR", "STRESS_BEAR") else 0.0
    conc_relief = -30.0 if (pos_weight_pct > 13.0 or sector_exposure_pct > 25.0) else 0.0
    req_edge_bps = max(80.0, float(min_improvement_bps) + reg_adder + conc_relief)

    # Winner retention shield: NEVER proactively rotate an intact compounding winner!
    if pol in ("INTELLIGENT_HYSTERESIS", "WINNER_SHIELD_LONG_HOLD"):
        if unrealized_pct >= protect_winner_pct and trend_intact:
            return {
                "should_replace": False,
                "reason": "RETAIN_WINNER_SHIELD_INTACT",
                "switching_cost_bps": round(switch_cost_bps, 2),
                "net_incremental_edge_bps": round(net_incremental_bps, 2),
            }
        if bars_held < min_hold_days_before_rotate:
            return {
                "should_replace": False,
                "reason": "RETAIN_MIN_HOLDING_PERIOD",
                "switching_cost_bps": round(switch_cost_bps, 2),
                "net_incremental_edge_bps": round(net_incremental_bps, 2),
            }

    # Stagnant loser / broken trend exit into superior candidate (or cash/core)
    if (not trend_intact) and (not in_keep_buffer):
        # If trend broke AND unrealized loss > 2% or stagnant after min_hold_days, allow exit
        if pol == "WINNER_SHIELD_LONG_HOLD":
            # Longer winner retention: only exit if deeply below SMA50 (<0.93) OR strong upgrade exists
            deep_break = (sma_50_px is not None) and (not math.isnan(sma_50_px)) and (cur_close_px < sma_50_px * 0.93)
            if deep_break or (best_candidate is not None and net_incremental_bps >= req_edge_bps):
                return {
                    "should_replace": True,
                    "reason": "REPLACE_DEEP_TREND_BREAK_OR_STRONG_UPGRADE",
                    "switching_cost_bps": round(switch_cost_bps, 2),
                    "net_incremental_edge_bps": round(net_incremental_bps, 2),
                }
            return {
                "should_replace": False,
                "reason": "RETAIN_LONG_HOLD_HYSTERESIS",
                "switching_cost_bps": round(switch_cost_bps, 2),
                "net_incremental_edge_bps": round(net_incremental_bps, 2),
            }

        # Standard INTELLIGENT_HYSTERESIS: require either negative thesis break OR net edge > req_edge_bps
        if unrealized_pct < 0.0 or (best_candidate is not None and net_incremental_bps >= req_edge_bps):
            return {
                "should_replace": True,
                "reason": "REPLACE_BROKEN_THESIS_COST_JUSTIFIED",
                "switching_cost_bps": round(switch_cost_bps, 2),
                "net_incremental_edge_bps": round(net_incremental_bps, 2),
            }
        return {
            "should_replace": False,
            "reason": "RETAIN_INSUFFICIENT_UPGRADE_EDGE",
            "switching_cost_bps": round(switch_cost_bps, 2),
            "net_incremental_edge_bps": round(net_incremental_bps, 2),
        }

    # If trend/buffer is intact, only rotate under aggressive upgrade mode if incremental net edge clears 1.5x hurdle
    if pol == "OPPORTUNITY_UPGRADE_ONLY" and best_candidate is not None and net_incremental_bps >= (req_edge_bps * 1.5) and unrealized_pct < 2.0:
        return {
            "should_replace": True,
            "reason": "REPLACE_MAJOR_ALPHA_UPGRADE",
            "switching_cost_bps": round(switch_cost_bps, 2),
            "net_incremental_edge_bps": round(net_incremental_bps, 2),
        }

    return {
        "should_replace": False,
        "reason": "RETAIN_HIGH_QUALITY_INCUMBENT",
        "switching_cost_bps": round(switch_cost_bps, 2),
        "net_incremental_edge_bps": round(net_incremental_bps, 2),
    }


# ---------------------------------------------------------------------------
# 3. STOP-LOSS & EXIT INTELLIGENCE (Section 5 — 6 Exit Models A..F)
# ---------------------------------------------------------------------------
def evaluate_position_exit(
    *,
    pos: dict[str, Any],
    stop_model: str,  # "A_EXISTING_M3_0_BE"|"B_ATR_DYNAMIC"|"C_VOL_ADJUSTED_TRAILING"|"D_STRUCTURE_BASED"|"E_TIME_DECAY"|"F_HYBRID_STOP_PROFIT_PROTECT"
    o_v: float,
    h_v: float,
    l_v: float,
    c_v: float,
    sma_50_v: float | None,
    low_20_prev: float | None,
    vol_20d_daily: float,
    hold_max_days: int,
    is_last_bar: bool,
) -> tuple[str | None, float | None]:
    """Evaluate protective stop/trailing/time exit honestly on OHLC bar without impossible fills.

    Always enforces a protective catastrophe stop (never removes stops to inflate backtests).
    """
    sm = stop_model.upper().strip()
    sl_px = float(pos["sl_px"])
    entry_px = float(pos["entry_px"])
    atr_e = max(1e-4, float(pos["atr_entry"]))
    bars_held = int(pos["bars_held"])
    unreal_pct = ((c_v / max(1e-6, entry_px)) - 1.0) * 100.0

    # 1. Check hard protective stop trigger FIRST (honest gap-through-open fill!)
    if o_v <= sl_px:
        reason = "STOP_LOSS_BE" if pos.get("be_locked") else "STOP_LOSS_GAP"
        return reason, o_v
    if l_v <= sl_px:
        reason = "STOP_LOSS_BE" if pos.get("be_locked") else "STOP_LOSS_INTRADAY"
        return reason, sl_px

    # 2. Check end-of-window or max holding period exit at close
    if is_last_bar:
        return "END_OF_WINDOW", c_v
    if bars_held >= hold_max_days:
        return "TIME_EXIT_MAX_HOLD", c_v

    # 3. Model-specific EOD checks & stop ratcheting (applied for NEXT session's stop!)
    pos["max_px"] = max(float(pos["max_px"]), h_v)

    if sm == "A_EXISTING_M3_0_BE":
        # Legacy M3.0 behavior: tight +0.65% BE lock at +3*ATR + 3.5x ATR trailing stop
        if (not pos.get("be_locked")) and c_v >= entry_px + 3.0 * atr_e:
            be_px = entry_px * 1.0065
            if be_px > pos["sl_px"]:
                pos["sl_px"] = be_px
                pos["be_locked"] = True
        tr_px = pos["max_px"] - 3.5 * atr_e
        if tr_px > pos["sl_px"]:
            pos["sl_px"] = tr_px

    elif sm == "B_ATR_DYNAMIC":
        # Wide dynamic ATR trailing stop: 5.0x ATR initially, tightening to 3.8x ATR after +18% gain
        # Zero tight breakeven lock so normal pullbacks never shake out winners
        tr_mult = 3.8 if unreal_pct >= 18.0 else 5.0
        tr_px = pos["max_px"] - tr_mult * atr_e
        if tr_px > pos["sl_px"]:
            pos["sl_px"] = tr_px

    elif sm == "C_VOL_ADJUSTED_TRAILING":
        # Realized 20D volatility-scaled trailing stop (adapts to low-vol vs high-vol stocks)
        # Trailing band = clip(2.6 * sigma_20d * sqrt(20), 9%, 22%)
        d_vol = max(0.008, min(0.06, float(vol_20d_daily)))
        trail_frac = max(0.09, min(0.22, 2.6 * d_vol * math.sqrt(20.0)))
        if unreal_pct >= 20.0:
            trail_frac *= 0.80  # Protect large gains moderately
        tr_px = pos["max_px"] * (1.0 - trail_frac)
        if tr_px > pos["sl_px"]:
            pos["sl_px"] = tr_px

    elif sm == "D_STRUCTURE_BASED":
        # Market-structure stop: keep hard catastrophe stop, ratchet to min(20D swing low, 0.96*SMA50)
        # ONLY once position is profitable (>+5%) and held >= 10d so intact structure is respected
        if bars_held >= 10 and unreal_pct >= 5.0:
            struct_cands: list[float] = []
            if low_20_prev is not None and not math.isnan(low_20_prev) and low_20_prev > 0:
                struct_cands.append(low_20_prev * 0.985)
            if sma_50_v is not None and not math.isnan(sma_50_v) and sma_50_v > 0:
                struct_cands.append(sma_50_v * 0.95)
            if struct_cands:
                # Choose conservative lower structural floor so single wick doesn't stop out
                s_floor = min(struct_cands)
                if s_floor > pos["sl_px"] and s_floor < c_v * 0.94:
                    pos["sl_px"] = s_floor

    elif sm == "E_TIME_DECAY":
        # Progressive time-decay exit for dead-money / stagnant positions; winners ride wide 5.0x ATR
        if bars_held >= 40 and unreal_pct < 1.5:
            return "TIME_DECAY_STAGNANT", c_v
        if bars_held >= 25 and unreal_pct < -3.0:
            # Tighten stop to 2.2x ATR on persistent laggards
            tight_px = c_v - 2.2 * atr_e
            if tight_px > pos["sl_px"]:
                pos["sl_px"] = tight_px
        else:
            tr_px = pos["max_px"] - 5.0 * atr_e
            if tr_px > pos["sl_px"]:
                pos["sl_px"] = tr_px

    elif sm == "F_HYBRID_STOP_PROFIT_PROTECT":
        # Best-of-breed institutional hybrid:
        # 1. Hard initial protective ATR stop remains active from Day 1 (already checked above).
        # 2. Dead-money time decay: if held >= 45d and still <= +1.0% with broken 50D SMA -> exit cleanly.
        if bars_held >= 45 and unreal_pct <= 1.0:
            if sma_50_v is not None and not math.isnan(sma_50_v) and c_v < sma_50_v * 0.98:
                return "TIME_DECAY_HYBRID", c_v
        # 3. Structural + volatility profit protection (never tight BE lock!):
        #    - Gain < 10%: wide 5.2x ATR trail (retain intact thesis)
        #    - Gain 10%..25%: ratchet profit lock using max(4.2x ATR, 20D structural low)
        #    - Gain >= 25%: protect profit with 3.2x ATR trail
        if unreal_pct >= 25.0:
            tr_px = pos["max_px"] - 3.2 * atr_e
        elif unreal_pct >= 10.0:
            tr_atr = pos["max_px"] - 4.2 * atr_e
            tr_struct = (low_20_prev * 0.985) if (low_20_prev is not None and not math.isnan(low_20_prev)) else 0.0
            tr_px = max(tr_atr, min(tr_struct, c_v * 0.93)) if tr_struct > 0 else tr_atr
        else:
            tr_px = pos["max_px"] - 5.2 * atr_e
        if tr_px > pos["sl_px"]:
            pos["sl_px"] = tr_px

    return None, None


# ---------------------------------------------------------------------------
# 4. EXECUTION TIMING ENGINE (Section 6 — 6 Timing Models)
# ---------------------------------------------------------------------------
def simulate_execution_fill(
    *,
    execution_mode: str,  # "NEXT_SESSION_OPEN"|"NEXT_SESSION_VWAP_APPROX"|"LIMIT_ORDER_SIM"|"LIQUIDITY_AWARE"|"SPREAD_AWARE"|"VOLATILITY_ADJUSTED"
    side: str,            # "BUY" | "SELL"
    req_qty: int,
    prev_close: float,
    bar_open: float,
    bar_high: float,
    bar_low: float,
    bar_close: float,
    bar_vol: float,
    adv_20d_inr: float,
    atr_14: float,
    hl_range_pct: float,
    max_bar_vol_part: float = 0.01,
    max_adv_part: float = 0.02,
    min_viable_notional_inr: float = 4_500.0,
) -> dict[str, Any]:
    """Simulate realistic daily OHLCV fill with circuit limits, liquidity caps, gap checks & honest VWAP labeling."""
    mode = execution_mode.upper().strip()
    sd = side.upper().strip()
    is_approx = (mode == "NEXT_SESSION_VWAP_APPROX")

    if req_qty < 1 or bar_vol <= 0 or min(bar_open, bar_high, bar_low, bar_close) <= 0 or bar_high < bar_low:
        return {"status": "FAILED", "reason": "INVALID_OR_ZERO_VOLUME_BAR", "fill_qty": 0, "raw_fill_px": 0.0, "is_approximation": is_approx}

    # Circuit-limit / locked-bar protection (realistic NSE 5%/10%/20% circuit behavior)
    gap_frac = abs((bar_open / max(1e-6, prev_close)) - 1.0) if prev_close > 0 else 0.0
    if (bar_high == bar_low and gap_frac >= 0.048) or gap_frac >= 0.195:
        return {"status": "FAILED", "reason": "CIRCUIT_LIMIT_LOCKED", "fill_qty": 0, "raw_fill_px": 0.0, "is_approximation": is_approx}

    # Liquidity & participation caps
    eff_vol_part = 0.005 if mode == "LIQUIDITY_AWARE" else max_bar_vol_part
    eff_adv_part = 0.010 if mode == "LIQUIDITY_AWARE" else max_adv_part
    max_qty_vol = int(bar_vol * eff_vol_part)
    max_qty_adv = int((adv_20d_inr * eff_adv_part) // max(1e-6, bar_open)) if adv_20d_inr > 0 else req_qty
    cap_qty = min(req_qty, max_qty_vol, max_qty_adv)
    if cap_qty < 1:
        return {"status": "FAILED", "reason": "LIQUIDITY_VOLUME_INSUFFICIENT", "fill_qty": 0, "raw_fill_px": 0.0, "is_approximation": is_approx}
    if sd == "BUY" and (cap_qty * bar_open) < min_viable_notional_inr:
        return {"status": "FAILED", "reason": "PARTIAL_FILL_BELOW_MIN_NOTIONAL", "fill_qty": 0, "raw_fill_px": 0.0, "is_approximation": is_approx}

    partial = cap_qty < req_qty

    if mode == "NEXT_SESSION_OPEN":
        raw_px = bar_open
    elif mode == "NEXT_SESSION_VWAP_APPROX":
        # EXPLICIT APPROXIMATION: Daily OHLCV only — NO intraday tick/minute data exists!
        raw_px = (bar_open + bar_high + bar_low + bar_close) / 4.0
    elif mode == "LIMIT_ORDER_SIM":
        if sd == "BUY":
            # Passive limit slightly below prior close (-0.25%)
            limit_px = prev_close * 0.9975 if prev_close > 0 else bar_open
            if bar_open <= limit_px:
                raw_px = bar_open
            elif bar_low <= limit_px:
                raw_px = limit_px
            else:
                return {"status": "UNFILLED", "reason": "BUY_LIMIT_NOT_REACHED", "fill_qty": 0, "raw_fill_px": 0.0, "is_approximation": False}
        else:
            # Sells execute at open so protective/rebalance exits aren't stranded
            raw_px = bar_open
    elif mode == "SPREAD_AWARE":
        if sd == "BUY" and hl_range_pct > 7.5:
            return {"status": "FAILED", "reason": "SPREAD_RANGE_TOO_WIDE", "fill_qty": 0, "raw_fill_px": 0.0, "is_approximation": False}
        raw_px = bar_open
    elif mode == "VOLATILITY_ADJUSTED":
        if sd == "BUY" and prev_close > 0 and atr_14 > 0:
            # Refuse to chase extreme opening gap-ups (> 1.15 * ATR above prior close)
            if (bar_open - prev_close) > 1.15 * atr_14:
                return {"status": "UNFILLED", "reason": "ADVERSE_GAP_UP_SKIPPED", "fill_qty": 0, "raw_fill_px": 0.0, "is_approximation": False}
        raw_px = bar_open
    elif mode == "LIQUIDITY_AWARE":
        raw_px = bar_open
    else:
        raw_px = bar_open

    return {
        "status": "PARTIAL" if partial else "FILLED",
        "reason": "PARTIAL_LIQUIDITY_CAP" if partial else "FILLED_OK",
        "fill_qty": int(cap_qty),
        "raw_fill_px": float(raw_px),
        "is_approximation": is_approx,
    }


# ---------------------------------------------------------------------------
# 5. POSITION-SIZING INTELLIGENCE (Section 7 — 6 Sizing Models A..F)
# ---------------------------------------------------------------------------
def compute_intelligent_position_size(
    *,
    sizing_mode: str,  # "A_FIXED"|"B_VOLATILITY_ADJUSTED"|"C_RISK_BUDGETED"|"D_CONFIDENCE_WEIGHTED"|"E_EXPECTED_ALPHA_WEIGHTED"|"F_HYBRID_RISK_ALPHA"
    base_slot_inr: float,
    rem_tact_budget_inr: float,
    nav_inr: float,
    max_single_stock_pct: float,
    sector_room_inr: float,
    industry_room_inr: float,
    adv_20d_inr: float,
    daily_vol_frac: float,
    stop_ratio: float,
    signal_score: float,
    expected_net_alpha_bps: float,
    hurdle_bps: float,
    max_corr_with_open: float,
    min_lot_inr: float,
) -> float:
    """Compute target rupee allocation for a candidate respecting all risk & concentration caps."""
    sm = sizing_mode.upper().strip()
    d_vol = max(0.008, min(0.06, float(daily_vol_frac)))
    sl_r = max(0.05, min(0.22, float(stop_ratio)))
    ann_vol = d_vol * math.sqrt(252.0)

    if sm == "A_FIXED":
        mult = 1.0
    elif sm == "B_VOLATILITY_ADJUSTED":
        # Target 24% annualized single-stock volatility benchmark -> calm stocks get larger lots, choppy smaller
        mult = max(0.65, min(1.35, 0.24 / max(0.10, ann_vol)))
    elif sm == "C_RISK_BUDGETED":
        # Equal risk contribution per trade (~1.1% NAV risk at stop distance)
        target_risk_inr = nav_inr * 0.011
        desired_notional = target_risk_inr / sl_r
        mult = max(0.65, min(1.35, desired_notional / max(1.0, base_slot_inr)))
    elif sm == "D_CONFIDENCE_WEIGHTED":
        # Scale by validated signal score (typically 0.45..0.95)
        mult = max(0.70, min(1.30, 0.70 + 0.75 * max(0.0, float(signal_score) - 0.35)))
    elif sm == "E_EXPECTED_ALPHA_WEIGHTED":
        # Scale proportionally to expected net alpha surplus above hurdle
        alpha_ratio = float(expected_net_alpha_bps) / max(60.0, float(hurdle_bps))
        mult = max(0.70, min(1.35, 0.72 + 0.35 * max(0.0, alpha_ratio - 1.0)))
    elif sm == "F_HYBRID_RISK_ALPHA":
        # Combine inverse-volatility risk normalization + expected net alpha edge + correlation haircut
        vol_m = max(0.70, min(1.25, 0.24 / max(0.10, ann_vol)))
        alpha_ratio = float(expected_net_alpha_bps) / max(60.0, float(hurdle_bps))
        alpha_m = max(0.75, min(1.25, 0.78 + 0.28 * max(0.0, alpha_ratio - 1.0)))
        corr_haircut = 0.85 if max_corr_with_open > 0.50 else 1.0
        mult = max(0.65, min(1.35, vol_m * alpha_m * corr_haircut))
    else:
        mult = 1.0

    raw_alloc = base_slot_inr * mult
    max_single_inr = nav_inr * (max_single_stock_pct / 100.0)
    max_adv_inr = adv_20d_inr * 0.015 if adv_20d_inr > 0 else max_single_inr
    capped = min(raw_alloc, rem_tact_budget_inr, max_single_inr, sector_room_inr, industry_room_inr, max_adv_inr)

    # Ensure if rem_tact_budget can fit min_lot_inr, we don't artificially shrink a valid trade just under min_lot
    if capped < min_lot_inr and min(rem_tact_budget_inr, max_single_inr, sector_room_inr, industry_room_inr, max_adv_inr) >= min_lot_inr * 1.05:
        capped = min_lot_inr * 1.05
    return max(0.0, float(capped))
