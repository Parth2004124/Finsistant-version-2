"""Moreshwar Mathematical Model 2.1 — Research, Portfolio-Risk & 6-Regime Engine.

Strict Version Isolation:
- Moreshwar 1.0 (`scoring_engine.calculate_moreshwar_levels`) is untouched.
- Moreshwar 2.0 (`moreshwar_v2.evaluate_moreshwar_v2`) is untouched.
- Moreshwar 2.1 lives strictly here and in `backtest_engine_v2_1.py`, never auto-promoted.

Core Upgrades in 2.1:
1. Granular 6-state causal market regime classifier:
   STRONG_BULLISH, WEAK_BULLISH, SIDEWAYS, WEAK_BEARISH, STRONG_BEARISH, HIGH_VOLATILITY.
2. Portfolio-aware risk gating:
   - Max simultaneous open positions & max sector concentration.
   - 60D rolling pairwise correlation throttle against already-open holdings.
   - Aggregate portfolio stop-loss risk cap (equity at risk if all stops trigger simultaneously).
   - Volume participation limit (`<= 1%` of 20D avg daily volume) to flag/cap partial fills.
3. Exact itemized Indian Equity Delivery cost model (brokerage, STT, exchange txn,
   SEBI fee, stamp duty, GST, DP sell fee, liquidity-tiered slippage).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from .moreshwar_v2 import (
    MoreshwarV2Config,
    compute_atr,
    compute_confidence_adjustment,
)


REGIMES_6 = (
    "STRONG_BULLISH",
    "WEAK_BULLISH",
    "SIDEWAYS",
    "WEAK_BEARISH",
    "STRONG_BEARISH",
    "HIGH_VOLATILITY",
)


@dataclass(frozen=True)
class MoreshwarV21Config:
    """Explicitly versioned, reproducible parameters for Moreshwar 2.1 research."""

    model_version: str = "Moreshwar 2.1"
    atr_period: int = 14
    atr_multiplier: float = 2.0
    min_stop_pct: float = 2.5
    max_stop_pct: float = 10.0
    high_vol_atr_pct: float = 4.5
    high_vol_action: str = "throttle"
    horizon_days: int = 14
    target_atr_horizon_scale: float = 1.05
    min_rr_ratio: float = 1.5
    max_target_pct: float = 22.0
    risk_per_trade_pct: float = 1.0  # Conservative 1.0% NAV risk per trade
    max_single_stock_pct: float = 18.0
    max_portfolio_exposure_pct: float = 80.0
    max_portfolio_stop_risk_pct: float = 5.0  # Max concurrent open SL risk as % of equity
    max_open_positions: int = 6
    max_sector_positions: int = 2
    max_pairwise_corr: float = 0.78
    min_confidence: float = 0.45
    confidence_sizing_floor: float = 0.45
    max_analytical_confidence: float = 0.85
    min_avg_daily_turnover: float = 2_500_000.0  # ₹25L/day minimum liquidity
    max_daily_volume_participation: float = 0.01  # Max 1% of 20D avg share volume per order
    regime_policy: str = "defensive_6state"  # "defensive_6state" | "permissive" | "bull_only_6state"
    # Indian Equity Delivery Execution Cost Assumptions (strictly equity delivery)
    execution_mode: str = "INDIAN_EQUITY_DELIVERY"
    base_slippage_bps: float = 7.5
    illiquid_slippage_bps: float = 20.0
    illiquid_turnover_threshold: float = 25_000_000.0  # ₹2.5Cr/day


def classify_market_regime_6(
    benchmark_candles: list[dict[str, Any]] | None,
    decision_date: str | None = None,
    annualized_vol_high_pct: float = 18.5,
) -> dict[str, Any]:
    """Strictly causal 6-state market regime classification on Nifty 50 (`bars <= decision_date`)."""
    if not benchmark_candles:
        return {
            "regime": "SIDEWAYS",
            "available": False,
            "as_of_date": decision_date,
            "close": None,
            "sma_20": None,
            "sma_50": None,
            "sma_20_slope_pct": None,
            "ret_20d_pct": None,
            "ann_vol_20d_pct": None,
            "atr_14_pct": None,
            "reason": "Benchmark candles unavailable; defaulting neutrally to SIDEWAYS.",
        }

    causal_bars = (
        [c for c in benchmark_candles if str(c.get("date", "")) <= str(decision_date)]
        if decision_date
        else list(benchmark_candles)
    )
    if len(causal_bars) < 30:
        return {
            "regime": "SIDEWAYS",
            "available": False,
            "as_of_date": causal_bars[-1]["date"] if causal_bars else decision_date,
            "close": float(causal_bars[-1]["close"]) if causal_bars else None,
            "sma_20": None,
            "sma_50": None,
            "sma_20_slope_pct": None,
            "ret_20d_pct": None,
            "ann_vol_20d_pct": None,
            "atr_14_pct": None,
            "reason": "Insufficient benchmark warmup (<30 bars).",
        }

    closes = [float(c["close"]) for c in causal_bars if float(c.get("close") or 0.0) > 0]
    spot = closes[-1]
    sma_20 = sum(closes[-20:]) / 20.0
    sma_50 = sum(closes[-50:]) / min(50, len(closes))
    sma_20_prev = sum(closes[-25:-5]) / 20.0 if len(closes) >= 25 else sma_20
    sma_20_slope_pct = ((sma_20 - sma_20_prev) / max(1e-6, sma_20_prev)) * 100.0
    ret_20d_pct = ((spot - closes[-21]) / max(1e-6, closes[-21])) * 100.0 if len(closes) >= 21 else 0.0
    ret_5d_pct = ((spot - closes[-6]) / max(1e-6, closes[-6])) * 100.0 if len(closes) >= 6 else 0.0

    rets: list[float] = []
    for i in range(max(1, len(closes) - 20), len(closes)):
        rets.append((closes[i] - closes[i - 1]) / max(1e-6, closes[i - 1]))
    mean_r = sum(rets) / max(1, len(rets))
    var_r = sum((r - mean_r) ** 2 for r in rets) / max(1, len(rets) - 1) if len(rets) > 1 else 0.0
    ann_vol_pct = math.sqrt(var_r) * math.sqrt(252.0) * 100.0

    atr_bench = compute_atr(causal_bars[-25:], period=14)
    atr_bench_pct = (atr_bench / max(1e-6, spot)) * 100.0

    if ann_vol_pct >= annualized_vol_high_pct or atr_bench_pct >= 1.65 or ret_5d_pct <= -3.5:
        regime = "HIGH_VOLATILITY"
        reason = (
            f"Elevated Nifty stress: 20D ann vol={ann_vol_pct:.1f}%, "
            f"ATR(14)={atr_bench_pct:.2f}%, 5D ret={ret_5d_pct:+.1f}%."
        )
    elif spot >= sma_20 and sma_20 >= sma_50 and sma_20_slope_pct >= 0.4 and ret_20d_pct >= 2.0:
        regime = "STRONG_BULLISH"
        reason = f"Aligned uptrend above rising 20D & 50D SMAs (20D ret={ret_20d_pct:+.1f}%, slope={sma_20_slope_pct:+.2f}%)."
    elif spot >= sma_50 and (spot >= sma_20 or sma_20_slope_pct > 0.0):
        regime = "WEAK_BULLISH"
        reason = f"Above 50D SMA with mild/moderating momentum (20D ret={ret_20d_pct:+.1f}%)."
    elif spot < sma_20 and sma_20 < sma_50 and sma_20_slope_pct <= -0.4 and ret_20d_pct <= -2.0:
        regime = "STRONG_BEARISH"
        reason = f"Confirmed downtrend below falling 20D & 50D SMAs (20D ret={ret_20d_pct:+.1f}%)."
    elif spot < sma_50 and (spot < sma_20 or sma_20_slope_pct < 0.0):
        regime = "WEAK_BEARISH"
        reason = f"Below 50D SMA with weak/negative bias (20D ret={ret_20d_pct:+.1f}%)."
    else:
        regime = "SIDEWAYS"
        reason = f"Range-bound consolidation near SMAs (20D ret={ret_20d_pct:+.1f}%, slope={sma_20_slope_pct:+.2f}%)."

    return {
        "regime": regime,
        "available": True,
        "as_of_date": causal_bars[-1]["date"],
        "close": round(spot, 2),
        "sma_20": round(sma_20, 2),
        "sma_50": round(sma_50, 2),
        "sma_20_slope_pct": round(sma_20_slope_pct, 2),
        "ret_20d_pct": round(ret_20d_pct, 2),
        "ret_5d_pct": round(ret_5d_pct, 2),
        "ann_vol_20d_pct": round(ann_vol_pct, 2),
        "atr_14_pct": round(atr_bench_pct, 2),
        "reason": reason,
    }


def compute_indian_delivery_costs(
    *,
    side: str,  # "BUY" | "SELL"
    notional: float,
    currency: str = "INR",
    slippage_bps: float = 7.5,
) -> dict[str, float]:
    """Explicit, unmixed Equity Delivery transaction cost model.

    For INR (NSE Cash Delivery):
    - Brokerage: min(₹20, 0.03% of notional) per executed order leg
    - STT: 0.10% on both BUY and SELL equity delivery turnover
    - NSE Exchange Txn Charge: 0.00297%
    - SEBI Turnover Charge: 0.0001%
    - Stamp Duty: 0.015% on BUY only (Indian Stamp Act, eq delivery)
    - GST: 18% on (Brokerage + Exchange Txn + SEBI fee)
    - CDSL DP Charge: ₹15.93 flat per SELL delivery exit (incl. GST)
    - Execution Slippage / Half-Spread: `slippage_bps` bps on notional

    For USD (NVDA non-NSE control):
    - Zero brokerage + SEC/TAF regulatory micro-fee (0.0028% sell) + slippage_bps.
    """
    amt = max(0.0, float(notional))
    slip_cost = amt * (slippage_bps / 10000.0)
    if amt <= 0.0:
        return {
            "brokerage": 0.0,
            "stt": 0.0,
            "exchange_txn": 0.0,
            "sebi_fee": 0.0,
            "stamp_duty": 0.0,
            "gst": 0.0,
            "dp_charge": 0.0,
            "slippage_cost": 0.0,
            "regulatory_total": 0.0,
            "total_cost": 0.0,
        }

    side_u = side.upper().strip()
    if currency != "INR":
        reg = amt * 0.000028 if side_u == "SELL" else 0.0
        tot = reg + slip_cost
        return {
            "brokerage": 0.0,
            "stt": 0.0,
            "exchange_txn": 0.0,
            "sebi_fee": 0.0,
            "stamp_duty": 0.0,
            "gst": 0.0,
            "dp_charge": 0.0,
            "slippage_cost": round(slip_cost, 4),
            "regulatory_total": round(reg, 4),
            "total_cost": round(tot, 4),
        }

    brokerage = min(20.0, amt * 0.0003)
    stt = amt * 0.0010  # 0.10% on delivery buy & sell
    exchange_txn = amt * 0.0000297
    sebi_fee = amt * 0.000001
    stamp_duty = amt * 0.00015 if side_u == "BUY" else 0.0
    gst = 0.18 * (brokerage + exchange_txn + sebi_fee)
    dp_charge = 15.93 if side_u == "SELL" else 0.0

    regulatory_total = brokerage + stt + exchange_txn + sebi_fee + stamp_duty + gst + dp_charge
    total_cost = regulatory_total + slip_cost
    return {
        "brokerage": round(brokerage, 4),
        "stt": round(stt, 4),
        "exchange_txn": round(exchange_txn, 4),
        "sebi_fee": round(sebi_fee, 4),
        "stamp_duty": round(stamp_duty, 4),
        "gst": round(gst, 4),
        "dp_charge": round(dp_charge, 4),
        "slippage_cost": round(slip_cost, 4),
        "regulatory_total": round(regulatory_total, 4),
        "total_cost": round(total_cost, 4),
    }


def compute_pairwise_return_correlation(
    closes_a: list[float],
    closes_b: list[float],
    window: int = 60,
) -> float:
    """Compute strictly causal trailing Pearson correlation of daily returns."""
    n = min(len(closes_a), len(closes_b), window + 1)
    if n < 20:
        return 0.0
    ra = [(closes_a[-n + i] - closes_a[-n + i - 1]) / max(1e-6, closes_a[-n + i - 1]) for i in range(1, n)]
    rb = [(closes_b[-n + i] - closes_b[-n + i - 1]) / max(1e-6, closes_b[-n + i - 1]) for i in range(1, n)]
    ma = sum(ra) / len(ra)
    mb = sum(rb) / len(rb)
    cov = sum((a - ma) * (b - mb) for a, b in zip(ra, rb))
    va = sum((a - ma) ** 2 for a in ra)
    vb = sum((b - mb) ** 2 for b in rb)
    denom = math.sqrt(va * vb)
    if denom <= 1e-12:
        return 0.0
    return round(max(-1.0, min(1.0, cov / denom)), 3)


def evaluate_moreshwar_v2_1(
    *,
    symbol: str,
    sector: str,
    candles: list[dict[str, Any]],
    spot_price: float,
    fundamental_score: float | None = None,
    porters_score: float | None = None,
    missing_fields: list[str] | None = None,
    fundamental_mode: str = "historical_unavailable",
    tech_signal: str = "NEUTRAL",
    regression_curve: str = "Upward Curve",
    sma_20: float | None = None,
    rsi_14: float | None = None,
    benchmark_candles: list[dict[str, Any]] | None = None,
    decision_date: str | None = None,
    capital: float = 100_000.0,
    current_deployed_capital: float = 0.0,
    open_positions_context: list[dict[str, Any]] | None = None,
    currency: str = "INR",
    cfg: MoreshwarV21Config | None = None,
    enable_regime_filter: bool = True,
    enable_portfolio_constraints: bool = True,
) -> dict[str, Any]:
    """Evaluate a candidate signal under Moreshwar 2.1 (never alters M1.0 or M2.0)."""
    c = cfg or MoreshwarV21Config()
    spot = float(spot_price)
    if spot <= 0 or len(candles) < 20:
        return {
            "model_version": c.model_version,
            "accepted": False,
            "symbol": symbol,
            "sector": sector,
            "rejection_category": "INSUFFICIENT_HISTORY",
            "rejection_reasons": ["Insufficient OHLCV history (<20 bars) or invalid price."],
            "market_regime": "SIDEWAYS",
            "confidence": 0.0,
            "entry_price": round(spot, 2),
            "stop_loss": 0.0,
            "target": 0.0,
            "rr_ratio": 0.0,
            "position_qty": 0,
        }

    closes = [float(bar.get("close") or spot) for bar in candles]
    lows = [float(bar.get("low") or bar.get("close") or spot) for bar in candles]
    highs = [float(bar.get("high") or bar.get("close") or spot) for bar in candles]
    vols = [float(bar.get("volume") or 0.0) for bar in candles[-20:]]

    if sma_20 is None:
        sma_20 = sum(closes[-20:]) / min(20, len(closes))
    if rsi_14 is None:
        rs = closes[-15:]
        gains = [max(0.0, rs[j] - rs[j - 1]) for j in range(1, len(rs))]
        losses = [max(0.0, rs[j - 1] - rs[j]) for j in range(1, len(rs))]
        ag = sum(gains) / max(1, len(gains))
        al = sum(losses) / max(1, len(losses))
        rsi_14 = 50.0 if (ag == 0 and al == 0) else (85.0 if al == 0 else 100.0 - 100.0 / (1.0 + ag / al))

    atr = compute_atr(candles, period=c.atr_period)
    atr_pct = round((atr / max(1e-6, spot)) * 100.0, 2)

    regime6 = classify_market_regime_6(benchmark_candles, decision_date=decision_date)
    r_name = regime6["regime"]

    # Map 6-state regime conservatively to 4-state confidence helper
    r_mapped = (
        "BULLISH" if r_name in ("STRONG_BULLISH", "WEAK_BULLISH")
        else ("BEARISH" if r_name in ("STRONG_BEARISH", "WEAK_BEARISH")
        else ("HIGH_VOLATILITY" if r_name == "HIGH_VOLATILITY" else "NEUTRAL"))
    )
    v2_proxy_cfg = MoreshwarV2Config(
        atr_period=c.atr_period,
        atr_multiplier=c.atr_multiplier,
        min_stop_pct=c.min_stop_pct,
        max_stop_pct=c.max_stop_pct,
        min_confidence=c.min_confidence,
        max_analytical_confidence=c.max_analytical_confidence,
    )
    conf_info = compute_confidence_adjustment(
        fundamental_score=fundamental_score,
        porters_score=porters_score,
        missing_fields=missing_fields,
        fundamental_mode=fundamental_mode,
        price_above_sma20=(spot >= sma_20),
        tech_signal=tech_signal,
        regression_curve=regression_curve,
        rsi_14=rsi_14,
        regime=r_mapped,
        cfg=v2_proxy_cfg,
    )
    conf = conf_info["confidence"]

    # ATR-Normalized Stop-Loss & Realistic Horizon Target
    entry = round(spot, 2)
    dyn_mult = c.atr_multiplier
    if r_name in ("HIGH_VOLATILITY", "WEAK_BEARISH"):
        dyn_mult = max(1.35, round(c.atr_multiplier * 0.85, 2))
    raw_stop_dist = atr * dyn_mult
    recent_support = min(lows[-20:])
    recent_resistance = max(highs[-20:])
    sup_dist = entry - max(0.0, recent_support - atr * 0.25)
    stop_dist = sup_dist if (0.85 * raw_stop_dist <= sup_dist <= 1.25 * raw_stop_dist) else raw_stop_dist
    stop_dist = max(entry * (c.min_stop_pct / 100.0), min(entry * (c.max_stop_pct / 100.0), stop_dist))
    stop_loss = round(entry - stop_dist, 2)

    expected_move = atr * math.sqrt(max(1, c.horizon_days)) * c.target_atr_horizon_scale
    breakout_cap = max(recent_resistance * 1.03, entry + expected_move * 0.90)
    raw_target_dist = min(expected_move, breakout_cap - entry, entry * (c.max_target_pct / 100.0))
    natural_rr = round(raw_target_dist / max(0.01, stop_dist), 2)
    target_dist = max(raw_target_dist, stop_dist * c.min_rr_ratio) if raw_target_dist >= stop_dist * 1.2 else raw_target_dist
    target = round(entry + max(entry * 0.015, target_dist), 2)

    risk_per_share = max(0.01, entry - stop_loss)
    rr_ratio = round((target - entry) / risk_per_share, 2)

    # Liquidity & Slippage tier
    avg_vol = (sum(vols) / len(vols)) if vols else 0.0
    avg_turnover = avg_vol * spot
    min_liq = c.min_avg_daily_turnover if currency == "INR" else (c.min_avg_daily_turnover / 80.0)
    liq_thresh_slip = c.illiquid_turnover_threshold if currency == "INR" else (c.illiquid_turnover_threshold / 80.0)
    slippage_bps = c.illiquid_slippage_bps if (avg_vol > 0 and avg_turnover < liq_thresh_slip) else c.base_slippage_bps
    liquidity_ok = (avg_vol > 0 and avg_turnover >= min_liq)

    # Regime-modulated Risk Budget & Portfolio-Level Risk Guardrails
    regime_risk_mult = (
        1.0 if r_name == "STRONG_BULLISH"
        else (0.80 if r_name == "WEAK_BULLISH"
        else (0.60 if r_name == "SIDEWAYS" else 0.45))
    )
    vol_throttled = atr_pct > c.high_vol_atr_pct
    conf_scaler = max(c.confidence_sizing_floor, min(1.0, conf / max(0.1, c.max_analytical_confidence)))
    if vol_throttled:
        conf_scaler *= max(0.4, c.high_vol_atr_pct / max(0.1, atr_pct))

    # Portfolio Context Check (open slots, sector concentration, correlation, portfolio SL risk)
    open_list = open_positions_context or []
    open_count = len(open_list)
    sector_count = sum(1 for p in open_list if p.get("sector") == sector)
    existing_sl_risk = sum(max(0.0, (p.get("entry_price", 0.0) - p.get("sl", 0.0)) * p.get("qty", 0)) for p in open_list)
    max_corr_seen = 0.0
    for p in open_list:
        p_closes = p.get("recent_closes") or []
        if p_closes:
            corr_val = compute_pairwise_return_correlation(closes, p_closes, window=60)
            max_corr_seen = max(max_corr_seen, corr_val)

    corr_scaler = 0.65 if (enable_portfolio_constraints and max_corr_seen >= 0.65) else 1.0
    effective_risk_amt = capital * (c.risk_per_trade_pct / 100.0) * conf_scaler * regime_risk_mult * corr_scaler

    # Cap by remaining aggregate portfolio SL risk budget
    max_sl_budget_amt = capital * (c.max_portfolio_stop_risk_pct / 100.0)
    rem_sl_budget = max(0.0, max_sl_budget_amt - existing_sl_risk)
    effective_risk_amt = min(effective_risk_amt, rem_sl_budget) if enable_portfolio_constraints else effective_risk_amt

    rem_cap = max(0.0, capital * (c.max_portfolio_exposure_pct / 100.0) - current_deployed_capital)
    allowed_notional = min(capital * (c.max_single_stock_pct / 100.0), rem_cap)
    risk_qty = int(effective_risk_amt // risk_per_share)
    cap_qty = int(allowed_notional // max(1e-6, entry))
    vol_cap_qty = int((avg_vol * c.max_daily_volume_participation)) if avg_vol > 0 else cap_qty
    partial_fill_capped = (min(risk_qty, cap_qty) > vol_cap_qty > 0)
    qty = max(0, min(risk_qty, cap_qty, vol_cap_qty if avg_vol > 0 else cap_qty))

    rejection_reasons: list[str] = []
    rejection_cat: str | None = None

    def _add_rej(cat: str, msg: str) -> None:
        nonlocal rejection_cat
        if rejection_cat is None:
            rejection_cat = cat
        rejection_reasons.append(msg)

    if enable_regime_filter:
        if c.regime_policy == "defensive_6state" and r_name in ("STRONG_BEARISH", "WEAK_BEARISH", "HIGH_VOLATILITY"):
            _add_rej("REGIME_GATE", f"6-State Regime gate ({c.regime_policy}): blocked in {r_name}.")
        elif c.regime_policy == "bull_only_6state" and r_name not in ("STRONG_BULLISH", "WEAK_BULLISH"):
            _add_rej("REGIME_GATE", f"6-State Regime gate ({c.regime_policy}): requires bullish trend (current={r_name}).")

    if enable_portfolio_constraints:
        if open_count >= c.max_open_positions:
            _add_rej("PORTFOLIO_CAP_OR_CORR", f"Max simultaneous positions reached ({open_count}/{c.max_open_positions}).")
        if sector_count >= c.max_sector_positions:
            _add_rej("PORTFOLIO_CAP_OR_CORR", f"Max sector concentration reached ({sector}: {sector_count}/{c.max_sector_positions}).")
        if max_corr_seen > c.max_pairwise_corr:
            _add_rej("PORTFOLIO_CAP_OR_CORR", f"Excess pairwise correlation ({max_corr_seen:.2f} > {c.max_pairwise_corr:.2f}) with open holding.")
        if rem_sl_budget < risk_per_share:
            _add_rej("PORTFOLIO_CAP_OR_CORR", f"Aggregate portfolio SL risk cap ({c.max_portfolio_stop_risk_pct:.1f}% NAV) exhausted.")

    if not liquidity_ok:
        _add_rej("LIQUIDITY_CAP", f"Insufficient 20D avg turnover ({avg_turnover:,.0f} < {min_liq:,.0f}) or zero volume.")

    if conf < c.min_confidence:
        _add_rej("CONFIDENCE_BELOW_MIN", f"Adjusted confidence {conf:.2f} below {c.min_confidence:.2f} threshold.")

    if natural_rr < c.min_rr_ratio or rr_ratio < c.min_rr_ratio:
        _add_rej("UNFAVORABLE_RR", f"ATR R:R {natural_rr:.2f} < {c.min_rr_ratio:.2f} minimum.")

    min_viable_notional = 1500.0 if currency == "INR" else 25.0
    if qty < 1 or (qty * entry) < min_viable_notional:
        if "PORTFOLIO_CAP_OR_CORR" != rejection_cat and "LIQUIDITY_CAP" != rejection_cat:
            _add_rej("ZERO_SHARES", f"Order notional below minimum viable delivery size ({min_viable_notional:,.0f} {currency}).")
        qty = 0

    accepted = len(rejection_reasons) == 0
    return {
        "model_version": c.model_version,
        "accepted": accepted,
        "symbol": symbol,
        "sector": sector,
        "market_regime": r_name,
        "regime_details": regime6,
        "confidence": conf,
        "confidence_breakdown": conf_info,
        "atr_value": round(atr, 2),
        "atr_pct": atr_pct,
        "atr_multiplier": dyn_mult,
        "entry_price": entry,
        "stop_loss": stop_loss,
        "target": target,
        "stop_distance_pct": round((stop_dist / entry) * 100.0, 2),
        "target_distance_pct": round(((target - entry) / entry) * 100.0, 2),
        "rr_ratio": rr_ratio,
        "natural_rr": natural_rr,
        "position_qty": qty,
        "risk_budget_amount": round(effective_risk_amt, 2),
        "max_capital_exposure": round(qty * entry, 2),
        "exposure_pct": round(((qty * entry) / max(1.0, capital)) * 100.0, 2),
        "slippage_bps": slippage_bps,
        "partial_fill_capped": partial_fill_capped,
        "max_pairwise_corr": max_corr_seen,
        "avg_turnover": round(avg_turnover, 2),
        "rejection_category": rejection_cat,
        "rejection_reasons": rejection_reasons,
        "blocked_by_regime": rejection_cat == "REGIME_GATE",
    }
