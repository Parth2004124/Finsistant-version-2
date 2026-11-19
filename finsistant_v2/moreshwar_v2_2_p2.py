"""Moreshwar 2.2 Prompt 2 — Causal Entry Archetypes, Exit Intelligence & Profit Retention.

Strictly preserves Moreshwar 1.0, 2.0, 2.1, and Prompt 1 modules untouched.
Implements:
1. Three causal entry archetypes (`BREAKOUT`, `PULLBACK`, `CONTINUATION`) + combinations
   using strictly historical OHLCV bars (`<= signal_date`).
2. Six competing exit & profit-retention engines (`P0..P5`):
   - `P0_EXISTING`: static 2.0x ATR SL, static horizon TP, 45D cliff
   - `P1_BREAKEVEN_LOCK`: moves stop to cost-covered breakeven (+65 bps) after ATR profit trigger
   - `P2_ATR_TRAILING`: removes TP ceiling, trails k * ATR(14) purely from prior closes
   - `P3_PARTIAL_BOOKING`: books 50% at 1.5x Risk, locks BE, trails remaining 50%
   - `P4_ADAPTIVE_TRAILING`: adapts trailing ATR tightness dynamically by causal 6-regime & momentum
   - `P5_TIME_MOMENTUM`: cuts dead/stagnant trades early while extending profitable trends
3. Exact profit-retention diagnostics (MFE, MAE, % MFE retained, profitable-to-loss reversals,
   premature exits) across regimes, entry types, volatility buckets, stocks, and holding durations.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from .moreshwar_v2_2 import (
    DeliveryCostConfig,
    compute_repaired_delivery_costs,
)


@dataclass(frozen=True)
class MoreshwarV22P2Config:
    """Explainable configuration for Prompt 2 Entry + Exit + Portfolio Interaction experiments."""

    model_version: str = "2.2.0-p2"
    entry_mode: str = "M21_BASE"  # "M21_BASE" | "BREAKOUT" | "PULLBACK" | "CONTINUATION" | "BREAK_PULL" | "ALL_THREE"
    exit_mode: str = "P0_EXISTING"  # "P0_EXISTING" | "P1_BREAKEVEN_LOCK" | "P2_ATR_TRAILING" | "P3_PARTIAL_BOOKING" | "P4_ADAPTIVE_TRAILING" | "P5_TIME_MOMENTUM"

    # Base horizon & risk sizing
    horizon_days: int = 45
    max_extended_horizon_days: int = 75
    atr_period: int = 14
    initial_atr_mult: float = 2.0
    min_stop_pct: float = 2.5
    max_stop_pct: float = 10.0
    target_atr_horizon_scale: float = 1.05
    max_target_pct: float = 22.0
    min_rr_ratio: float = 1.5

    # Exit tuning parameters (small, interpretable grid only)
    breakeven_trigger_atr: float = 1.5
    breakeven_fee_buffer_pct: float = 0.65  # Covers NSE buy+sell STT+DP+slippage (~0.55%)
    trailing_atr_mult: float = 2.5
    trailing_activate_atr: float = 1.0
    partial_book_rr: float = 1.5
    stagnant_check_days: int = 15
    stagnant_min_atr_gain: float = 0.75

    # Selection & PIT eligibility
    use_ranked_selection: bool = True
    require_pit_filters: bool = True
    require_bull_regime_entry: bool = False
    min_avg_daily_turnover_inr: float = 50_000_000.0  # ₹5 Cr/day institutional floor
    fallback_avg_turnover_inr: float = 2_500_000.0
    max_atr_pct: float = 3.5
    max_60d_drawdown_pct: float = 25.0

    # Portfolio & re-entry interaction controls (Section 6)
    max_open_positions: int = 6
    max_sector_positions: int = 2
    max_pairwise_corr: float = 0.78
    risk_per_trade_pct: float = 1.0
    max_portfolio_stop_risk_pct: float = 6.0
    max_single_stock_pct: float = 18.0
    max_portfolio_exposure_pct: float = 85.0
    max_daily_volume_participation: float = 0.01
    reentry_cooldown_days: int = 0  # Cooldown bars after STOP_LOSS on same symbol
    min_viable_notional_inr: float = 1500.0


def evaluate_entry_archetypes_p2(
    *,
    symbol: str,
    feats: dict[str, Any],
    candles_window: list[dict[str, Any]],
    regime6: dict[str, Any],
    cfg: MoreshwarV22P2Config,
    tariff: DeliveryCostConfig | None = None,
) -> dict[str, Any]:
    """Evaluate causal Entry Archetypes (`BREAKOUT`, `PULLBACK`, `CONTINUATION`, `M21_BASE`).

    All quantities use strictly bars `<= idx` inside `candles_window`.
    """
    tc = tariff or DeliveryCostConfig()
    spot = float(feats["close"])
    sma20 = float(feats["sma20"])
    sma50 = float(feats["sma50"])
    rsi14 = float(feats["rsi14"])
    pct20 = float(feats["pct_above_sma20"])
    rs60 = float(feats["rs_vs_nifty_60d"])
    mom_cons = float(feats["mom_consistency"])
    atr_pct = float(feats["atr_pct"])
    dd60 = float(feats["dd60_pct"])
    zv20 = int(feats["zero_vol_20"])
    avg_turn = float(feats["avg_turnover20"])
    r_name = str(regime6.get("regime", "SIDEWAYS"))

    # 1. Market regime eligibility at entry
    if cfg.require_bull_regime_entry:
        if r_name not in ("STRONG_BULLISH", "WEAK_BULLISH"):
            return {"matched": False, "archetype": None, "reason": f"REGIME_GATE ({r_name})"}
    else:
        if r_name in ("STRONG_BEARISH", "WEAK_BEARISH", "HIGH_VOLATILITY"):
            return {"matched": False, "archetype": None, "reason": f"REGIME_GATE ({r_name})"}

    # 2. Point-in-time stock quality & liquidity eligibility
    min_liq = cfg.min_avg_daily_turnover_inr if cfg.require_pit_filters else cfg.fallback_avg_turnover_inr
    if feats["avg_vol20"] <= 0 or avg_turn < min_liq:
        return {"matched": False, "archetype": None, "reason": "LIQUIDITY_CAP"}
    if cfg.require_pit_filters:
        if zv20 > 0 or atr_pct > cfg.max_atr_pct or dd60 > cfg.max_60d_drawdown_pct:
            return {"matched": False, "archetype": None, "reason": "PIT_RISK_FILTER"}

    # Derived causal bar context
    prev_bar = candles_window[-2] if len(candles_window) >= 2 else candles_window[-1]
    cur_bar = candles_window[-1]
    prev_close = float(prev_bar.get("close") or spot)
    cur_open = float(cur_bar.get("open") or spot)
    cur_low = float(cur_bar.get("low") or spot)
    recent_lows3 = [float(b.get("low") or b.get("close") or spot) for b in candles_window[-3:]]
    touched_sma20_support = min(recent_lows3) <= (sma20 * 1.015)

    # Prior 20D closes to assess SMA50 upward slope
    sma50_prev10 = sma50
    if len(candles_window) >= 60:
        sma50_prev10 = sum(float(b["close"]) for b in candles_window[-60:-10]) / 50.0
    sma50_rising = sma50 >= sma50_prev10

    # Archetype A: Breakout (genuine 20D high breakout in established uptrend, capped extension)
    is_breakout = (
        bool(feats["is_20d_breakout"])
        and spot > sma50
        and sma20 > sma50
        and rs60 > 0.0
        and 0.0 <= pct20 <= 6.5
        and 52.0 <= rsi14 <= 68.0
    )

    # Archetype B: Pullback (established rising 50D uptrend, controlled pullback near SMA20 + bounce close)
    is_pullback = (
        spot > sma50
        and sma20 > sma50
        and sma50_rising
        and touched_sma20_support
        and -2.0 <= pct20 <= 2.5
        and 40.0 <= rsi14 <= 58.0
        and spot > prev_close
        and spot >= cur_open * 0.995
        and dd60 <= 18.0
    )

    # Archetype C: Momentum Continuation (persistent leadership pulling pause <= 4% above SMA20)
    is_continuation = (
        spot > sma20
        and sma20 > sma50
        and rs60 >= 2.0
        and mom_cons >= 0.52
        and 0.0 <= pct20 <= 4.0
        and 48.0 <= rsi14 <= 64.0
        and feats["tech_sig"] == "BULLISH"
    )

    is_m21_base = (spot >= sma20) and (42.0 <= rsi14 <= 70.0) and (feats["tech_sig"] == "BULLISH")

    mode = cfg.entry_mode
    matched_arch: str | None = None
    if mode == "M21_BASE" and is_m21_base:
        matched_arch = "M21_BASE"
    elif mode == "BREAKOUT" and is_breakout:
        matched_arch = "BREAKOUT"
    elif mode == "PULLBACK" and is_pullback:
        matched_arch = "PULLBACK"
    elif mode == "CONTINUATION" and is_continuation:
        matched_arch = "CONTINUATION"
    elif mode == "BREAK_PULL":
        matched_arch = "PULLBACK" if is_pullback else ("BREAKOUT" if is_breakout else None)
    elif mode == "PULL_CONT":
        matched_arch = "PULLBACK" if is_pullback else ("CONTINUATION" if is_continuation else None)
    elif mode == "ALL_THREE":
        if is_pullback:
            matched_arch = "PULLBACK"
        elif is_continuation:
            matched_arch = "CONTINUATION"
        elif is_breakout:
            matched_arch = "BREAKOUT"

    if not matched_arch:
        return {"matched": False, "archetype": None, "reason": "SIGNAL_NOT_TRIGGERED"}

    # Entry quality score (rewards high trend leadership + pullback/low-extension + stability)
    t_norm = max(0.0, min(1.0, (feats["trend_strength"] + 5.0) / 35.0))
    rs_norm = max(0.0, min(1.0, (rs60 + 10.0) / 40.0))
    cons_norm = max(0.0, min(1.0, (mom_cons - 0.35) / 0.45))
    ext_bonus = 1.0 if -1.0 <= pct20 <= 3.0 else max(0.0, 1.0 - (pct20 - 3.0) / 5.0)
    stab = max(0.0, min(1.0, (4.0 - atr_pct) / 2.5))
    arch_bonus = 0.08 if matched_arch == "PULLBACK" else (0.04 if matched_arch == "CONTINUATION" else 0.0)
    rank_score = round((0.30 * t_norm + 0.25 * rs_norm + 0.15 * cons_norm + 0.18 * ext_bonus + 0.12 * stab + arch_bonus) * 100.0, 3)

    # Stop distance & target geometry
    atr = float(feats["atr14"])
    lows = [float(b.get("low") or b.get("close") or spot) for b in candles_window[-20:]]
    highs = [float(b.get("high") or b.get("close") or spot) for b in candles_window[-20:]]
    raw_stop_dist = atr * cfg.initial_atr_mult
    sup_dist = spot - max(0.0, min(lows) - atr * 0.25)
    stop_dist = sup_dist if (0.85 * raw_stop_dist <= sup_dist <= 1.25 * raw_stop_dist) else raw_stop_dist
    stop_dist = max(spot * (cfg.min_stop_pct / 100.0), min(spot * (cfg.max_stop_pct / 100.0), stop_dist))

    expected_move = atr * math.sqrt(max(1, cfg.horizon_days)) * cfg.target_atr_horizon_scale
    breakout_cap = max(max(highs) * 1.03, spot + expected_move * 0.90)
    raw_tp_dist = min(expected_move, breakout_cap - spot, spot * (cfg.max_target_pct / 100.0))
    tp_dist = max(raw_tp_dist, stop_dist * cfg.min_rr_ratio) if raw_tp_dist >= stop_dist * 1.2 else raw_tp_dist

    # Trailing exit engines deliberately remove fixed TP ceiling so winners run cleanly
    use_fixed_tp = cfg.exit_mode == "P0_EXISTING"
    if use_fixed_tp and (raw_tp_dist / max(0.01, stop_dist)) < cfg.min_rr_ratio:
        return {"matched": False, "archetype": None, "reason": "UNFAVORABLE_RR"}

    sl_px = round(spot - stop_dist, 2)
    tp_px = round(spot + max(spot * 0.015, tp_dist), 2) if use_fixed_tp else 0.0
    slip_bps = tc.illiquid_slippage_bps if avg_turn < tc.illiquid_turnover_threshold_inr else tc.base_slippage_bps

    return {
        "matched": True,
        "symbol": symbol,
        "archetype": matched_arch,
        "rank_score": rank_score,
        "trend_strength": feats["trend_strength"],
        "spot": spot,
        "atr_entry": atr,
        "atr_pct": atr_pct,
        "sl_ratio": max(0.01, (spot - sl_px) / max(1e-6, spot)),
        "tp_ratio": max(0.015, (tp_px - spot) / max(1e-6, spot)) if use_fixed_tp else 0.0,
        "use_fixed_tp": use_fixed_tp,
        "slippage_bps": slip_bps,
        "market_regime": r_name,
        "pct_above_sma20": pct20,
    }


def check_bar_exit_action_p2(
    *,
    pos: dict[str, Any],
    bar: dict[str, Any],
    bar_feats: dict[str, Any] | None,
    regime6_name: str,
    is_final_bar: bool,
    cfg: MoreshwarV22P2Config,
) -> dict[str, Any]:
    """Chronological intra-bar exit evaluation + post-close stop ratcheting (zero look-ahead).

    Rules:
    1. Stop / target checks inside day `t` MUST strictly use `pos['sl']` and `pos['tp']`
       set BEFORE day `t` opened!
    2. Gap-through-stop fills at `open` (`gap_through_stop=True`), never at idealized `sl`.
    3. Partial booking checks `partial_target` against `bar['high']` (fills at `max(open, partial_target)`).
    4. Only AFTER surviving day `t` without exiting may stops trail upward using close `t`
       for bar `t+1`.
    """
    hi = float(bar.get("high") or pos["last_price"])
    lo = float(bar.get("low") or pos["last_price"])
    cl = float(bar.get("close") or pos["last_price"])
    op = float(bar.get("open") or cl)
    vol = int(bar.get("volume") or 0)

    entry_px = pos["entry_price"]
    sl_px = pos["sl"]
    tp_px = pos.get("tp", 0.0)
    atr_entry = pos["atr_entry"]

    # Track excursion using intraday extremes
    pos["last_price"] = cl
    pos["bars_held"] += 1
    pos["max_price_seen"] = max(pos["max_price_seen"], hi)
    pos["mfe_pct"] = max(pos["mfe_pct"], ((hi - entry_px) / entry_px) * 100.0)
    pos["mae_pct"] = min(pos["mae_pct"], ((lo - entry_px) / entry_px) * 100.0)

    partial_fill_px: float | None = None
    partial_qty = 0

    # Conservative intra-bar priority: Stop Loss checked BEFORE Targets / Partial Profit
    exit_px: float | None = None
    exit_reason: str | None = None
    is_gap_stop = False

    if lo <= sl_px:
        if op < sl_px:
            exit_px = op
            is_gap_stop = True
        else:
            exit_px = sl_px
        exit_reason = "STOP_LOSS_TRAIL" if pos.get("trailing_active") else ("STOP_LOSS_BE" if pos.get("breakeven_locked") else "STOP_LOSS")
    elif tp_px > 0.0 and hi >= tp_px:
        exit_px = max(op, tp_px) if op > tp_px else tp_px
        exit_reason = "TARGET_HIT"
    else:
        # Check Partial Profit Booking milestone (only if stop did NOT hit today)
        if (
            cfg.exit_mode == "P3_PARTIAL_BOOKING"
            and not pos.get("partial_booked")
            and pos["qty"] >= 2
            and hi >= pos["partial_target"]
        ):
            partial_qty = pos["qty"] // 2
            partial_fill_px = max(op, pos["partial_target"]) if op > pos["partial_target"] else pos["partial_target"]

        # Final backtest day liquidation
        if is_final_bar:
            exit_px = cl
            exit_reason = "FINAL_MARK_EXIT"
        elif cfg.exit_mode == "P5_TIME_MOMENTUM":
            unrealized_atr = (cl - entry_px) / max(1e-6, atr_entry)
            sma20_now = float(bar_feats["sma20"]) if bar_feats else entry_px
            # Exit dead/stagnant capital early if momentum failed to materialize after check window
            if pos["bars_held"] >= cfg.stagnant_check_days and unrealized_atr < cfg.stagnant_min_atr_gain:
                exit_px = cl
                exit_reason = "MOMENTUM_STALL_EXIT"
            elif pos["bars_held"] >= cfg.horizon_days:
                # Only extend profitable trending trades; exit stalled/unprofitable ones at base horizon
                if not (cl > entry_px * (1.0 + cfg.breakeven_fee_buffer_pct / 100.0) and cl >= sma20_now):
                    exit_px = cl
                    exit_reason = "HORIZON_EXIT"
            if exit_px is None and pos["bars_held"] >= cfg.max_extended_horizon_days:
                exit_px = cl
                exit_reason = "MAX_HORIZON_EXIT"
        elif pos["bars_held"] >= cfg.horizon_days:
            exit_px = cl
            exit_reason = "HORIZON_EXIT"

    # Respect halted / zero-volume bars cleanly
    if exit_px is not None and vol <= 0 and not is_final_bar:
        return {
            "action": "DEFER_ZERO_VOL",
            "partial_qty": 0,
            "partial_px": None,
            "exit_px": None,
            "exit_reason": None,
            "is_gap_stop": False,
        }

    if exit_px is not None:
        return {
            "action": "FULL_EXIT",
            "partial_qty": partial_qty,
            "partial_px": partial_fill_px,
            "exit_px": exit_px,
            "exit_reason": exit_reason,
            "is_gap_stop": is_gap_stop,
        }

    # --- Post-Close Trailing / Breakeven Ratchet for Bar `t+1` ---
    if partial_qty > 0 and partial_fill_px is not None:
        pos["partial_booked"] = True
        be_px = round(entry_px * (1.0 + cfg.breakeven_fee_buffer_pct / 100.0), 4)
        if be_px > pos["sl"]:
            pos["sl"] = be_px
            pos["breakeven_locked"] = True

    atr_now = float(bar_feats["atr14"]) if bar_feats else atr_entry
    unrealized_atr = (cl - entry_px) / max(1e-6, atr_entry)
    peak_close = max(pos.get("max_close_seen", entry_px), cl)
    pos["max_close_seen"] = peak_close
    be_px = round(entry_px * (1.0 + cfg.breakeven_fee_buffer_pct / 100.0), 4)

    if cfg.exit_mode == "P1_BREAKEVEN_LOCK":
        if not pos.get("breakeven_locked") and peak_close >= entry_px + cfg.breakeven_trigger_atr * atr_entry:
            if be_px > pos["sl"]:
                pos["sl"] = be_px
                pos["breakeven_locked"] = True

    elif cfg.exit_mode in ("P2_ATR_TRAILING", "P3_PARTIAL_BOOKING"):
        if peak_close >= entry_px + cfg.trailing_activate_atr * atr_entry:
            trail_cand = round(peak_close - cfg.trailing_atr_mult * atr_now, 4)
            if trail_cand > pos["sl"]:
                pos["sl"] = trail_cand
                pos["trailing_active"] = True

    elif cfg.exit_mode == "P4_ADAPTIVE_TRAILING":
        # Adaptive trailing: wider in healthy strong bull, tighter on momentum weakness or regime shift
        sma20_now = float(bar_feats["sma20"]) if bar_feats else entry_px
        rsi_now = float(bar_feats["rsi14"]) if bar_feats else 55.0
        weak_mom = (cl < sma20_now) or (rsi_now < 48.0)

        if regime6_name in ("STRONG_BEARISH", "WEAK_BEARISH", "HIGH_VOLATILITY", "SIDEWAYS") and unrealized_atr >= 0.5:
            # Regime rolled adverse while green -> tighten sharply to protect capital
            mult = 1.4
        elif weak_mom and unrealized_atr >= 1.0:
            mult = 1.75
        elif regime6_name == "STRONG_BULLISH" and not weak_mom:
            mult = 2.8
        else:
            mult = cfg.trailing_atr_mult

        if peak_close >= entry_px + 1.0 * atr_entry:
            trail_cand = round(peak_close - mult * atr_now, 4)
            if unrealized_atr >= 1.25 and be_px > trail_cand:
                trail_cand = be_px
                pos["breakeven_locked"] = True
            if trail_cand > pos["sl"]:
                pos["sl"] = trail_cand
                pos["trailing_active"] = True

    elif cfg.exit_mode == "P5_TIME_MOMENTUM":
        # Protect profits once extended past 1.5x ATR via cost-covered BE + 2.5x ATR trail
        if peak_close >= entry_px + 1.5 * atr_entry:
            trail_cand = round(max(be_px, peak_close - 2.5 * atr_now), 4)
            if trail_cand > pos["sl"]:
                pos["sl"] = trail_cand
                pos["trailing_active"] = True

    return {
        "action": "PARTIAL_BOOK" if partial_qty > 0 else "HOLD",
        "partial_qty": partial_qty,
        "partial_px": partial_fill_px,
        "exit_px": None,
        "exit_reason": None,
        "is_gap_stop": False,
    }


def compute_profit_retention_metrics(closed_trades: list[dict[str, Any]]) -> dict[str, Any]:
    """Prompt 2 Section 4: Comprehensive Profit Retention & MFE/MAE Diagnostic Metrics."""
    n = len(closed_trades)
    if n == 0:
        return {"total_trades": 0}

    avg_mfe = sum(t["mfe_pct"] for t in closed_trades) / n
    avg_mae = sum(t["mae_pct"] for t in closed_trades) / n
    avg_ret = sum(t["net_return_pct"] for t in closed_trades) / n

    # Max unrealized dollar gain per trade = MFE% * entry_notional
    max_unreal_amts = [(t["mfe_pct"] / 100.0) * t["raw_buy_notional"] for t in closed_trades]
    tot_max_unreal = sum(max(0.0, u) for u in max_unreal_amts)
    tot_realized = sum(t["pnl_amount"] for t in closed_trades)
    winners_only_unreal = sum(max(0.0, (t["mfe_pct"] / 100.0) * t["raw_buy_notional"]) for t in closed_trades if t["pnl_amount"] > 0)
    winners_realized = sum(t["pnl_amount"] for t in closed_trades if t["pnl_amount"] > 0)

    # Trade-level MFE capture ratios for trades achieving meaningful positive excursion (>= +1.0%)
    pos_mfe_trades = [t for t in closed_trades if t["mfe_pct"] >= 1.0]
    capture_ratios = [t["net_return_pct"] / max(0.01, t["mfe_pct"]) for t in pos_mfe_trades]
    mean_capture_pct = (sum(capture_ratios) / len(capture_ratios)) * 100.0 if capture_ratios else 0.0

    # Portfolio MFE retention: Winners realized / Winners peak unrealized AND Overall net / Total peak unrealized
    win_retention_pct = (winners_realized / max(1e-6, winners_only_unreal)) * 100.0 if winners_only_unreal > 0 else 0.0
    net_retention_pct = (tot_realized / max(1e-6, tot_max_unreal)) * 100.0 if tot_max_unreal > 0 else 0.0

    rev_from_15 = sum(1 for t in closed_trades if t["mfe_pct"] >= 1.5 and t["pnl_amount"] <= 0)
    rev_from_25 = sum(1 for t in closed_trades if t["mfe_pct"] >= 2.5 and t["pnl_amount"] <= 0)
    premature_exits = sum(
        1 for t in closed_trades
        if "HORIZON" in str(t.get("exit_reason", "")) and t["pnl_amount"] > 0 and (t["mfe_pct"] - t["net_return_pct"]) >= 2.0
    )

    def _bucket(key_fn) -> dict[str, dict[str, Any]]:
        grp: dict[str, list[dict[str, Any]]] = {}
        for tr in closed_trades:
            k = str(key_fn(tr))
            grp.setdefault(k, []).append(tr)
        out = {}
        for k, rows in sorted(grp.items()):
            rn = len(rows)
            rw = sum(1 for r in rows if r["pnl_amount"] > 0)
            rmfe = sum(r["mfe_pct"] for r in rows) / rn
            rret = sum(r["net_return_pct"] for r in rows) / rn
            rpnl = sum(r["pnl_amount"] for r in rows)
            rmax_u = sum(max(0.0, (r["mfe_pct"] / 100.0) * r["raw_buy_notional"]) for r in rows)
            out[k] = {
                "trades": rn,
                "win_rate_pct": round((rw / rn) * 100.0, 1),
                "avg_mfe_pct": round(rmfe, 2),
                "avg_mae_pct": round(sum(r["mae_pct"] for r in rows) / rn, 2),
                "avg_net_ret_pct": round(rret, 2),
                "net_pnl": round(rpnl, 2),
                "net_mfe_retention_pct": round((rpnl / max(1e-6, rmax_u)) * 100.0, 1) if rmax_u > 0 else 0.0,
                "rev_15_loss": sum(1 for r in rows if r["mfe_pct"] >= 1.5 and r["pnl_amount"] <= 0),
            }
        return out

    return {
        "total_trades": n,
        "avg_mfe_pct": round(avg_mfe, 2),
        "avg_mae_pct": round(avg_mae, 2),
        "avg_realized_ret_pct": round(avg_ret, 2),
        "avg_max_unrealized_pnl": round(sum(max_unreal_amts) / n, 2),
        "total_max_unrealized_pnl": round(tot_max_unreal, 2),
        "total_realized_pnl": round(tot_realized, 2),
        "winner_mfe_retention_pct": round(win_retention_pct, 2),
        "net_portfolio_mfe_retention_pct": round(net_retention_pct, 2),
        "mean_pos_mfe_capture_pct": round(mean_capture_pct, 2),
        "profitable_1_5pct_closed_loss": rev_from_15,
        "profitable_2_5pct_closed_loss": rev_from_25,
        "premature_horizon_exits": premature_exits,
        "avg_holding_days": round(sum(t["holding_days"] for t in closed_trades) / n, 1),
        "by_regime": _bucket(lambda t: t.get("signal_regime6", "UNKNOWN")),
        "by_entry_type": _bucket(lambda t: t.get("archetype", "UNKNOWN")),
        "by_exit_reason": _bucket(lambda t: t.get("exit_reason", "UNKNOWN")),
        "by_holding_bucket": _bucket(
            lambda t: "1-5d" if t["holding_days"] <= 5 else ("6-15d" if t["holding_days"] <= 15 else ("16-30d" if t["holding_days"] <= 30 else "31d+"))
        ),
        "by_volatility_bucket": _bucket(
            lambda t: "<2.0% ATR" if t.get("atr_pct", 2.5) < 2.0 else ("2.0-3.0% ATR" if t.get("atr_pct", 2.5) <= 3.0 else ">3.0% ATR")
        ),
    }
