"""Moreshwar Mathematical Model 2.0 — Confidence-Adjusted, Volatility-Normalised Risk Engine.

Research & Shadow Module (explicitly separated from Moreshwar 1.0 production baseline
in `scoring_engine.calculate_moreshwar_levels()` so A/B comparisons remain reproducible).

Core Philosophy Preserved:
- A high-quality company (even 99-100 fundamental score) is NOT a low-risk trade.
- Analytical confidence is capped strictly below 1.00 and penalized for missing fields,
  point-in-time fundamental unavailability, technical contradictions, and hostile market regimes.
- Stop-loss distance is driven by ATR(14) price volatility and structure — NEVER compressed
  by `downside_pts = 100 - X` or distorted by arbitrary price scaling floors.
- Position sizing is governed by strict portfolio risk budgeting (`Capital * Risk% / RiskPerShare`)
  and hard exposure/liquidity caps, with confidence only throttling within the risk budget.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any


@dataclass(frozen=True)
class MoreshwarV2Config:
    """Configurable parameters for Moreshwar 2.0 risk & regime experiments."""

    atr_period: int = 14
    atr_multiplier: float = 2.0
    min_stop_pct: float = 2.5
    max_stop_pct: float = 10.0
    high_vol_atr_pct: float = 4.5  # Reject or throttle extremes above 4.5% daily ATR
    high_vol_action: str = "throttle"  # "throttle" or "reject"
    horizon_days: int = 14
    target_atr_horizon_scale: float = 1.05
    min_rr_ratio: float = 1.5
    max_target_pct: float = 22.0
    risk_per_trade_pct: float = 1.25  # Portfolio risk budget per trade (% of capital)
    max_single_stock_pct: float = 20.0  # Hard max notional allocation per symbol
    max_portfolio_exposure_pct: float = 85.0
    min_confidence: float = 0.45
    min_avg_daily_turnover: float = 2_000_000.0  # ₹20L (or $ equivalent scaled lower for USD)
    regime_policy: str = "defensive_bear_block"  # "permissive" | "defensive_bear_block" | "strict_bull_only"
    confidence_sizing_floor: float = 0.45
    max_analytical_confidence: float = 0.85  # Preserve anti-certainty rule: 100/100 != 100% certainty


def compute_atr(candles: list[dict[str, Any]], period: int = 14) -> float:
    """Compute strictly causal Average True Range (Wilder-style EMA or SMA warmup) on OHLC bars."""
    if len(candles) < 2:
        return 0.0
    trs: list[float] = []
    for i in range(1, len(candles)):
        cur = candles[i]
        prev = candles[i - 1]
        hi = float(cur.get("high") or cur.get("close") or 0.0)
        lo = float(cur.get("low") or cur.get("close") or 0.0)
        prev_c = float(prev.get("close") or 0.0)
        if prev_c <= 0 or hi <= 0 or lo <= 0:
            continue
        tr = max(hi - lo, abs(hi - prev_c), abs(lo - prev_c))
        trs.append(tr)
    if not trs:
        return 0.0
    if len(trs) <= period:
        return round(sum(trs) / len(trs), 4)
    # Wilder smoothing
    atr = sum(trs[:period]) / period
    for tr in trs[period:]:
        atr = ((atr * (period - 1)) + tr) / period
    return round(atr, 4)


def classify_market_regime(
    benchmark_candles: list[dict[str, Any]] | None,
    decision_date: str | None = None,
    annualized_vol_high_pct: float = 18.0,
) -> dict[str, Any]:
    """Classify Nifty 50 regime strictly using bars on or before `decision_date` (no look-ahead).

    Returns one of: BULLISH, NEUTRAL, BEARISH, HIGH_VOLATILITY (plus component metrics).
    """
    if not benchmark_candles:
        return {
            "regime": "NEUTRAL",
            "available": False,
            "as_of_date": decision_date,
            "close": None,
            "sma_20": None,
            "sma_50": None,
            "ann_vol_20d_pct": None,
            "atr_14_pct": None,
            "reason": "Benchmark candles unavailable; defaulting safely to NEUTRAL regime.",
        }

    if decision_date:
        causal_bars = [c for c in benchmark_candles if str(c.get("date", "")) <= str(decision_date)]
    else:
        causal_bars = list(benchmark_candles)

    if len(causal_bars) < 25:
        return {
            "regime": "NEUTRAL",
            "available": False,
            "as_of_date": causal_bars[-1]["date"] if causal_bars else decision_date,
            "close": float(causal_bars[-1]["close"]) if causal_bars else None,
            "sma_20": None,
            "sma_50": None,
            "ann_vol_20d_pct": None,
            "atr_14_pct": None,
            "reason": "Insufficient benchmark history (<25 bars) before decision date.",
        }

    closes = [float(c["close"]) for c in causal_bars if float(c.get("close") or 0.0) > 0]
    spot = closes[-1]
    sma_20 = sum(closes[-20:]) / min(20, len(closes))
    sma_50 = sum(closes[-50:]) / min(50, len(closes))
    sma_20_prev = sum(closes[-25:-5]) / 20.0 if len(closes) >= 25 else sma_20

    # Realized 20D annualized log/pct return volatility + ATR(14)%
    rets: list[float] = []
    tail = closes[-21:]
    for i in range(1, len(tail)):
        rets.append((tail[i] - tail[i - 1]) / max(1e-6, tail[i - 1]))
    mean_r = sum(rets) / max(1, len(rets))
    var_r = sum((r - mean_r) ** 2 for r in rets) / max(1, len(rets) - 1) if len(rets) > 1 else 0.0
    ann_vol_pct = round(math.sqrt(max(0.0, var_r)) * math.sqrt(252.0) * 100.0, 2)

    bench_atr = compute_atr(causal_bars[-40:], period=14)
    bench_atr_pct = round((bench_atr / max(1e-6, spot)) * 100.0, 2)

    trend_short_up = spot >= sma_20
    trend_med_up = spot >= sma_50 and sma_20 >= sma_50
    slope_up = sma_20 >= sma_20_prev
    five_day_drop = ((spot - closes[-6]) / max(1e-6, closes[-6])) * 100.0 if len(closes) >= 6 else 0.0

    # Regime precedence: High Volatility spike first, then structural Bearish, Bullish, Neutral
    if ann_vol_pct >= annualized_vol_high_pct or bench_atr_pct >= 1.65 or five_day_drop <= -3.5:
        regime = "HIGH_VOLATILITY"
        reason = (
            f"Elevated market turbulence (20D ann vol {ann_vol_pct:.1f}%, "
            f"Nifty ATR {bench_atr_pct:.2f}%, 5D chg {five_day_drop:+.1f}%)."
        )
    elif spot < sma_50 and (spot < sma_20 or not slope_up):
        regime = "BEARISH"
        reason = f"Nifty ({spot:,.1f}) below 50D SMA ({sma_50:,.1f}) & weak short-term momentum vs 20D SMA ({sma_20:,.1f})."
    elif trend_short_up and trend_med_up and slope_up:
        regime = "BULLISH"
        reason = f"Nifty ({spot:,.1f}) aligned above rising 20D SMA ({sma_20:,.1f}) & 50D SMA ({sma_50:,.1f})."
    else:
        regime = "NEUTRAL"
        reason = f"Mixed Nifty structure around 20D SMA ({sma_20:,.1f}) / 50D SMA ({sma_50:,.1f})."

    return {
        "regime": regime,
        "available": True,
        "as_of_date": causal_bars[-1]["date"],
        "close": round(spot, 2),
        "sma_20": round(sma_20, 2),
        "sma_50": round(sma_50, 2),
        "ann_vol_20d_pct": ann_vol_pct,
        "atr_14_pct": bench_atr_pct,
        "reason": reason,
    }


def compute_confidence_adjustment(
    *,
    fundamental_score: float | None,
    porters_score: float | None,
    missing_fields: list[str] | None,
    fundamental_mode: str,  # "point_in_time_live" | "historical_unavailable"
    price_above_sma20: bool,
    tech_signal: str,
    regression_curve: str,
    rsi_14: float,
    regime: str,
    cfg: MoreshwarV2Config,
) -> dict[str, Any]:
    """Compute Confidence Adjustment cleanly separated from business quality.

    Key design rule: a 100/100 fundamental score NEVER grants 1.00 confidence.
    It undergoes asymptotic capping (`max_analytical_confidence`) and explicit penalties
    for missing/unavailable fundamentals, contradictory technicals, overextension, and regime.
    """
    penalties: list[str] = []

    if fundamental_mode == "historical_unavailable" or fundamental_score is None:
        # Point-in-time fundamentals unknown: neutral quality base (0.66) minus explicit unavailability haircut (-0.08) -> 0.58
        # Preserves conservative sizing while allowing regime policies to differentiate NEUTRAL/HIGH_VOLATILITY vs BEARISH.
        base_quality_conf = 0.66
        missing_penalty = 0.08
        penalties.append("Point-in-time historical fundamentals unavailable (-0.08 neutral haircut)")
    else:
        f_val = max(0.0, min(100.0, float(fundamental_score)))
        p_val = max(0.0, min(100.0, float(porters_score if porters_score is not None else fundamental_score)))
        avg_q = 0.5 * (f_val + p_val)
        # Concave diminishing returns: 60 -> 0.64, 80 -> 0.74, 95 -> 0.80, 100 -> 0.82 (never 1.0!)
        base_quality_conf = min(cfg.max_analytical_confidence, 0.32 + 0.50 * math.sqrt(avg_q / 100.0))
        if avg_q >= 90:
            penalties.append("Anti-certainty cap: high fundamental score capped below certainty")

        miss_count = len(missing_fields or [])
        missing_penalty = min(0.18, miss_count * 0.04)
        if miss_count > 0:
            penalties.append(f"{miss_count} missing fundamental fields (-{missing_penalty:.2f})")

    contradiction_penalty = 0.0
    if not price_above_sma20:
        contradiction_penalty += 0.12
        penalties.append("Price below 20D SMA (-0.12)")
    if tech_signal == "BEARISH":
        contradiction_penalty += 0.14
        penalties.append("Bearish technical momentum (-0.14)")
    elif tech_signal == "NEUTRAL":
        contradiction_penalty += 0.04
        penalties.append("Neutral technical momentum (-0.04)")
    if regression_curve == "Downward Curve":
        contradiction_penalty += 0.09
        penalties.append("Downward polynomial curve (-0.09)")
    if rsi_14 >= 74.0:
        contradiction_penalty += 0.08
        penalties.append(f"Overbought RSI {rsi_14:.1f} (-0.08)")

    regime_penalty = 0.0
    if regime == "NEUTRAL":
        regime_penalty = 0.04
        penalties.append("Neutral market regime (-0.04)")
    elif regime == "HIGH_VOLATILITY":
        regime_penalty = 0.08
        penalties.append("High-volatility regime (-0.08)")
    elif regime == "BEARISH":
        regime_penalty = 0.14
        penalties.append("Bearish market regime (-0.14)")

    adj_conf = base_quality_conf - missing_penalty - contradiction_penalty - regime_penalty
    adj_conf = round(max(0.12, min(cfg.max_analytical_confidence, adj_conf)), 3)

    return {
        "confidence": adj_conf,
        "base_quality_conf": round(base_quality_conf, 3),
        "missing_penalty": round(missing_penalty, 3),
        "contradiction_penalty": round(contradiction_penalty, 3),
        "regime_penalty": round(regime_penalty, 3),
        "penalties": penalties,
    }


def evaluate_moreshwar_v2(
    *,
    symbol: str,
    candles: list[dict[str, Any]],
    spot_price: float,
    fundamental_score: float | None = None,
    porters_score: float | None = None,
    missing_fields: list[str] | None = None,
    fundamental_mode: str = "point_in_time_live",
    tech_signal: str = "NEUTRAL",
    regression_curve: str = "Upward Curve",
    sma_20: float | None = None,
    rsi_14: float | None = None,
    benchmark_candles: list[dict[str, Any]] | None = None,
    decision_date: str | None = None,
    capital: float = 100_000.0,
    current_deployed_capital: float = 0.0,
    currency: str = "INR",
    cfg: MoreshwarV2Config | None = None,
    # Phase 5 switches for independent isolation testing:
    enable_atr_stops: bool = True,
    enable_vol_sizing: bool = True,
    enable_regime_filter: bool = True,
    enable_confidence_and_targets: bool = True,
) -> dict[str, Any]:
    """Evaluate a trade under Moreshwar 2.0 (or any controlled Phase 5 intermediate variant).

    Exposes all 12 Phase 8 explainability fields cleanly without touching Moreshwar 1.0.
    """
    c = cfg or MoreshwarV2Config()
    spot = float(spot_price)
    if spot <= 0 or len(candles) < 15:
        return {
            "accepted": False,
            "symbol": symbol,
            "fundamental_score": fundamental_score,
            "fundamental_mode": fundamental_mode,
            "confidence": 0.0,
            "atr_value": 0.0,
            "atr_pct": 0.0,
            "atr_multiplier": c.atr_multiplier,
            "market_regime": "UNKNOWN",
            "entry_price": round(spot, 2),
            "stop_loss": 0.0,
            "target": 0.0,
            "rr_ratio": 0.0,
            "position_qty": 0,
            "max_capital_exposure": 0.0,
            "exposure_pct": 0.0,
            "rejection_reasons": ["Insufficient OHLCV history (<15 bars) or invalid price"],
            "accept_reasons": [],
            "reason_summary": "REJECTED: Insufficient historical bars for ATR calculation.",
        }

    # Causal indicator context from candles
    closes = [float(bar.get("close") or spot) for bar in candles]
    lows = [float(bar.get("low") or bar.get("close") or spot) for bar in candles]
    highs = [float(bar.get("high") or bar.get("close") or spot) for bar in candles]
    vols = [float(bar.get("volume") or 0.0) for bar in candles[-20:]]

    if sma_20 is None:
        sma_20 = sum(closes[-20:]) / min(20, len(closes))
    if rsi_14 is None:
        rsi_slice = closes[-15:]
        gains = [max(0.0, rsi_slice[j] - rsi_slice[j - 1]) for j in range(1, len(rsi_slice))]
        losses = [max(0.0, rsi_slice[j - 1] - rsi_slice[j]) for j in range(1, len(rsi_slice))]
        ag = sum(gains) / max(1, len(gains))
        al = sum(losses) / max(1, len(losses))
        rsi_14 = 50.0 if (ag == 0 and al == 0) else (85.0 if al == 0 else 100.0 - 100.0 / (1.0 + ag / al))

    atr = compute_atr(candles, period=c.atr_period)
    atr_pct = round((atr / max(1e-6, spot)) * 100.0, 2)

    regime_info = classify_market_regime(benchmark_candles, decision_date=decision_date)
    regime = regime_info["regime"]

    # 1. Confidence adjustment (A)
    conf_info = compute_confidence_adjustment(
        fundamental_score=fundamental_score,
        porters_score=porters_score,
        missing_fields=missing_fields,
        fundamental_mode=fundamental_mode,
        price_above_sma20=(spot >= sma_20),
        tech_signal=tech_signal,
        regression_curve=regression_curve,
        rsi_14=rsi_14,
        regime=regime,
        cfg=c,
    )
    conf = conf_info["confidence"]

    # 2. Entry, Stop-Loss & Target (B & D)
    entry = round(spot, 2)
    recent_support = min(lows[-20:]) if len(lows) >= 20 else min(lows)
    recent_resistance = max(highs[-20:]) if len(highs) >= 20 else max(highs)

    if enable_atr_stops:
        # Dynamic multiplier gently tightens in HIGH_VOLATILITY or relaxes slightly in calm BULLISH trend
        dyn_mult = c.atr_multiplier
        if regime == "HIGH_VOLATILITY":
            dyn_mult = max(1.35, round(c.atr_multiplier * 0.85, 2))
        raw_stop_dist = atr * dyn_mult
        # Structural blend: if nearby support sits reasonably within [0.85x, 1.2x] ATR stop, honor support buffer
        sup_stop = max(0.0, recent_support - atr * 0.25)
        sup_dist = entry - sup_stop
        if 0.85 * raw_stop_dist <= sup_dist <= 1.25 * raw_stop_dist:
            stop_dist = sup_dist
        else:
            stop_dist = raw_stop_dist

        min_dist = entry * (c.min_stop_pct / 100.0)
        max_dist = entry * (c.max_stop_pct / 100.0)
        stop_dist = max(min_dist, min(max_dist, stop_dist))
        stop_loss = round(entry - stop_dist, 2)
    else:
        # Fallback placeholder if ATR stops disabled in ablation (callers pass 1.0 stops)
        dyn_mult = 0.0
        stop_dist = max(entry * 0.03, entry * 0.05)
        stop_loss = round(entry - stop_dist, 2)

    if enable_confidence_and_targets:
        # Realistic expected movement over trade horizon using sqrt(T) diffusion scaled cleanly
        expected_horizon_move = atr * math.sqrt(max(1, c.horizon_days)) * c.target_atr_horizon_scale
        # Compare against structural headroom (resistance + modest breakout extension)
        breakout_cap = max(recent_resistance * 1.03, entry + expected_horizon_move * 0.90)
        raw_target_dist = min(expected_horizon_move, breakout_cap - entry)
        # Enforce statistical realism: cap upside within c.max_target_pct and require min R:R separately
        raw_target_dist = min(raw_target_dist, entry * (c.max_target_pct / 100.0))
        target_dist = max(raw_target_dist, stop_dist * c.min_rr_ratio) if raw_target_dist >= stop_dist * 1.2 else raw_target_dist
        target = round(entry + max(entry * 0.015, target_dist), 2)
    else:
        # Simple 1:2 R:R on stop distance when only testing stops/sizing/regime
        target_dist = stop_dist * 2.0
        target = round(entry + target_dist, 2)

    risk_per_share = max(0.01, entry - stop_loss)
    reward_per_share = max(0.0, target - entry)
    rr_ratio = round(reward_per_share / risk_per_share, 2)

    # 3. Position Sizing & Hard Guardrails (C)
    avg_vol = (sum(vols) / len(vols)) if vols else 0.0
    avg_turnover = avg_vol * spot
    min_liq_threshold = c.min_avg_daily_turnover if currency == "INR" else (c.min_avg_daily_turnover / 80.0)
    liquidity_ok = (avg_turnover >= min_liq_threshold) or (avg_vol == 0.0)  # vol=0 only if feed omits vol

    risk_budget_amount = capital * (c.risk_per_trade_pct / 100.0)
    # Confidence throttles size within [floor..1.0] without EVER exceeding risk_budget
    if enable_confidence_and_targets:
        conf_scaler = max(c.confidence_sizing_floor, min(1.0, conf / max(0.1, c.max_analytical_confidence)))
    else:
        conf_scaler = 1.0

    # Extra volatility throttling if ATR% is elevated
    vol_throttled = atr_pct > c.high_vol_atr_pct
    if vol_throttled and c.high_vol_action == "throttle":
        conf_scaler *= max(0.4, c.high_vol_atr_pct / max(0.1, atr_pct))

    effective_risk_amount = risk_budget_amount * conf_scaler
    remaining_portfolio_cap = max(0.0, capital * (c.max_portfolio_exposure_pct / 100.0) - current_deployed_capital)
    max_stock_cap = capital * (c.max_single_stock_pct / 100.0)
    allowed_notional = min(max_stock_cap, remaining_portfolio_cap)

    if enable_vol_sizing:
        risk_qty = int(effective_risk_amount // risk_per_share)
        cap_qty = int(allowed_notional // max(1e-6, entry))
        qty = max(0, min(risk_qty, cap_qty))
    else:
        qty = max(1, int((capital * 0.25) // max(1e-6, entry)))

    max_exposure = round(qty * entry, 2)
    exposure_pct = round((max_exposure / max(1.0, capital)) * 100.0, 2)

    # 4. Acceptance & Rejection Gates (Phase 2, Phase 3, Phase 8)
    rejection_reasons: list[str] = []
    accept_reasons: list[str] = []

    if enable_regime_filter:
        if c.regime_policy == "strict_bull_only" and regime != "BULLISH":
            rejection_reasons.append(f"Regime gate ({c.regime_policy}): market is {regime}, requires BULLISH.")
        elif c.regime_policy == "bear_and_highvol_block" and regime in ("BEARISH", "HIGH_VOLATILITY"):
            rejection_reasons.append(f"Regime gate ({c.regime_policy}): Nifty regime is {regime}.")
        elif c.regime_policy == "defensive_bear_block" and regime == "BEARISH":
            rejection_reasons.append(f"Regime gate ({c.regime_policy}): Nifty regime is BEARISH (below 50D SMA).")

    if vol_throttled and c.high_vol_action == "reject":
        rejection_reasons.append(f"Excessive volatility: ATR(14)={atr_pct:.2f}% exceeds {c.high_vol_atr_pct:.2f}% limit.")

    if enable_confidence_and_targets:
        if conf < c.min_confidence:
            rejection_reasons.append(f"Adjusted confidence {conf:.2f} below {c.min_confidence:.2f} minimum.")
        # Check whether ATR-realistic move satisfies R:R before forcing floor
        natural_rr = round((min(atr * math.sqrt(max(1, c.horizon_days)) * c.target_atr_horizon_scale, entry * (c.max_target_pct / 100.0))) / risk_per_share, 2)
        if natural_rr < c.min_rr_ratio or rr_ratio < c.min_rr_ratio:
            rejection_reasons.append(f"Unfavorable risk-to-reward: realistic ATR R:R {natural_rr:.2f} < {c.min_rr_ratio:.2f}.")

    if not liquidity_ok:
        rejection_reasons.append(f"Insufficient 20D avg turnover ({avg_turnover:,.0f} < {min_liq_threshold:,.0f}).")

    if enable_vol_sizing and qty < 1:
        rejection_reasons.append("Risk budget or exposure limit yields 0 integer shares.")

    accepted = len(rejection_reasons) == 0
    if accepted:
        accept_reasons.append(
            f"ATR(14) stop {stop_loss:,.2f} (-{(stop_dist/entry)*100:.1f}%, {dyn_mult}x ATR), "
            f"Target {target:,.2f} (+{((target-entry)/entry)*100:.1f}%, R:R 1:{rr_ratio}), "
            f"Regime={regime}, Conf={conf:.2f}, Size={qty} sh ({exposure_pct:.1f}% cap)."
        )

    return {
        "model_version": "Moreshwar 2.0",
        "accepted": accepted,
        "symbol": symbol,
        "fundamental_score": fundamental_score,
        "porters_score": porters_score,
        "fundamental_mode": fundamental_mode,
        "confidence": conf,
        "confidence_breakdown": conf_info,
        "atr_value": round(atr, 2),
        "atr_pct": atr_pct,
        "atr_multiplier": dyn_mult,
        "market_regime": regime,
        "regime_details": regime_info,
        "entry_price": entry,
        "stop_loss": stop_loss,
        "target": target,
        "stop_distance_pct": round((stop_dist / entry) * 100.0, 2),
        "target_distance_pct": round(((target - entry) / entry) * 100.0, 2),
        "rr_ratio": rr_ratio,
        "position_qty": qty,
        "max_capital_exposure": max_exposure,
        "exposure_pct": exposure_pct,
        "risk_budget_amount": round(effective_risk_amount, 2),
        "vol_throttled": vol_throttled,
        "rejection_reasons": rejection_reasons,
        "accept_reasons": accept_reasons,
        "blocked_by_regime": any("Regime gate" in r for r in rejection_reasons),
        "reason_summary": ("REJECTED: " + " | ".join(rejection_reasons)) if not accepted else accept_reasons[0],
    }
