"""Moreshwar 2.2 — Repaired Delivery Economics, Causal Eligibility & Ranked Selection Engine.

Design Rules (Prompt 1 of 3):
- Strictly preserves Moreshwar 1.0 (`scoring_engine.py`), Moreshwar 2.0 (`moreshwar_v2.py`),
  and frozen Moreshwar 2.1 (`moreshwar_v2_1.py`, `backtest_engine_v2_1.py`) untouched.
- Fixes NSE Equity Delivery cost accounting:
  * Zerodha historical equity delivery brokerage = ₹0.00 (with full-service 0.03%/₹20
    sensitivity toggle).
  * Per-scrip-per-day CDSL DP fee (₹15.93 including 18% GST) applied strictly once per
    settlement day per symbol sold.
  * Exact slippage attribution without double-multiplying `(1 + slip)` inside cost totals.
- Separates NSE (`INR`) equity accounting completely from US (`USD`) equity (`NVDA`).
- Eliminates alphabetical slot saturation via deterministic cross-sectional candidate ranking.
- Eliminates `volume_confirmation` (0.0006 IC) and avoids redundant collinear copies while
  incorporating orthogonal structural trend strength, anti-extension penalties, and
  point-in-time stability/liquidity filters (no ticker blacklisting, zero hindsight).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

from .moreshwar_v2 import compute_atr
from .moreshwar_v2_1 import (
    classify_market_regime_6,
    compute_pairwise_return_correlation,
)


@dataclass(frozen=True)
class DeliveryCostConfig:
    """Exact historical tariff & execution slippage configuration."""

    brokerage_mode: str = "zerodha_delivery"  # "zerodha_delivery" (₹0) | "legacy_0_03pct"
    stt_rate: float = 0.0010  # 0.10% BUY & SELL on equity delivery
    exchange_txn_rate: float = 0.0000297  # NSE 0.00297% (Oct 2024+ uniform; pre-Oct 2024 ~0.00325%)
    sebi_fee_rate: float = 0.000001  # ₹10 per crore = 0.0001%
    stamp_duty_buy_rate: float = 0.00015  # 0.015% BUY only (effective Jul 2020)
    gst_rate: float = 0.18  # 18% on (brokerage + exchange_txn + sebi_fee)
    dp_charge_per_scrip_day: float = 15.93  # ₹13.50 + 18% GST once per scrip per sell day
    base_slippage_bps: float = 7.5
    illiquid_slippage_bps: float = 20.0
    illiquid_turnover_threshold_inr: float = 25_000_000.0  # ₹2.5 Cr daily turnover threshold


@dataclass(frozen=True)
class MoreshwarV22Config:
    """Versioned controlled configuration for Moreshwar 2.2 Prompt 1 experiments."""

    model_version: str = "2.2.0-p1"
    horizon_days: int = 30
    atr_period: int = 14
    atr_multiplier: float = 2.0
    min_stop_pct: float = 2.5
    max_stop_pct: float = 10.0
    target_atr_horizon_scale: float = 1.05
    max_target_pct: float = 22.0
    min_rr_ratio: float = 1.5

    # Portfolio risk & capacity (matched to M2.1 baseline defaults unless ablated)
    max_open_positions: int = 6
    max_sector_positions: int = 2
    max_pairwise_corr: float = 0.78
    risk_per_trade_pct: float = 1.0
    max_portfolio_stop_risk_pct: float = 5.0
    max_single_stock_pct: float = 18.0
    max_portfolio_exposure_pct: float = 80.0
    max_daily_volume_participation: float = 0.01
    min_viable_notional_inr: float = 1500.0
    min_viable_notional_usd: float = 25.0

    # Selection & PIT risk-filter toggles (ablation flags)
    use_ranked_selection: bool = True
    require_structural_uptrend: bool = False  # close > sma50 and sma20 > sma50
    require_anti_extension: bool = False      # pct_above_sma20 <= max_extension & RSI band
    require_pit_risk_filter: bool = False     # ADV >= min_adv, ATR% <= max_atr, DD60 <= max_dd60, zero halt bars
    require_regime_gate: bool = False         # True -> block SIDEWAYS/BEARISH/HIGH_VOL
    regime_mode: str = "bull_only_no_sideways"  # "defensive_6state" | "bull_only_no_sideways" | "strong_bull_only"
    require_cost_hurdle: bool = True          # min target distance >= cost_hurdle_mult * est roundtrip cost
    cost_hurdle_mult: float = 4.0

    # Thresholds (structural, non-overfitted)
    min_avg_daily_turnover_inr: float = 50_000_000.0  # ₹5 Cr/day institutional tradability (when PIT filter enabled)
    baseline_avg_daily_turnover_inr: float = 2_500_000.0  # ₹25L exact M2.1 baseline turnover floor
    max_atr_pct: float = 3.5
    max_60d_drawdown_pct: float = 25.0
    max_extension_sma20_pct: float = 5.0
    breakout_extension_cap_pct: float = 6.5
    rsi_min: float = 50.0
    rsi_max: float = 66.0


def compute_repaired_delivery_costs(
    *,
    side: str,  # "BUY" | "SELL"
    raw_notional: float,
    currency: str = "INR",
    slippage_bps: float = 7.5,
    dp_charged_today: bool = False,
    tariff: DeliveryCostConfig | None = None,
) -> dict[str, float]:
    """Reconciled transaction cost calculator for NSE (INR) and US (USD).

    Reconciliation convention:
    - Execution price moves adversely by `slip_frac = slippage_bps / 10000.0`:
      * BUY:  exec_px = raw_px * (1 + slip_frac) -> exec_notional = raw_notional * (1 + slip_frac)
      * SELL: exec_px = raw_px * (1 - slip_frac) -> exec_notional = raw_notional * (1 - slip_frac)
    - Actual execution slippage cost = `abs(exec_notional - raw_notional) = raw_notional * slip_frac`
      (avoids the second-order `exec_notional * slip_frac` reporting bias in M2.1).
    - Statutory fees & taxes are assessed on the executed turnover `exec_notional`.
    - For INR Zerodha Equity Delivery (`zerodha_delivery`): brokerage = ₹0.00; DP charge ₹15.93
      is assessed ONLY once per scrip per sell day (`not dp_charged_today`).
    """
    cfg = tariff or DeliveryCostConfig()
    raw_amt = max(0.0, float(raw_notional))
    side_u = side.upper().strip()
    slip_frac = max(0.0, float(slippage_bps)) / 10_000.0

    if raw_amt <= 0.0:
        return {
            "raw_notional": 0.0,
            "exec_notional": 0.0,
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

    if side_u == "BUY":
        exec_amt = raw_amt * (1.0 + slip_frac)
    else:
        exec_amt = raw_amt * max(0.0, 1.0 - slip_frac)
    slip_cost = abs(exec_amt - raw_amt)

    if currency != "INR":
        # Independent US equity execution fees (SEC Section 31 ~0.00278% sell + tiny TAF)
        reg = exec_amt * 0.000028 if side_u == "SELL" else 0.0
        tot = reg + slip_cost
        return {
            "raw_notional": round(raw_amt, 4),
            "exec_notional": round(exec_amt, 4),
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

    brokerage = min(20.0, exec_amt * 0.0003) if cfg.brokerage_mode == "legacy_0_03pct" else 0.0
    stt = exec_amt * cfg.stt_rate
    exchange_txn = exec_amt * cfg.exchange_txn_rate
    sebi_fee = exec_amt * cfg.sebi_fee_rate
    stamp_duty = exec_amt * cfg.stamp_duty_buy_rate if side_u == "BUY" else 0.0
    gst = cfg.gst_rate * (brokerage + exchange_txn + sebi_fee)
    dp_charge = cfg.dp_charge_per_scrip_day if (side_u == "SELL" and not dp_charged_today) else 0.0

    regulatory_total = brokerage + stt + exchange_txn + sebi_fee + stamp_duty + gst + dp_charge
    total_cost = regulatory_total + slip_cost
    return {
        "raw_notional": round(raw_amt, 4),
        "exec_notional": round(exec_amt, 4),
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


def compute_causal_factor_snapshot(
    candles: list[dict[str, Any]],
    benchmark_candles: list[dict[str, Any]] | None,
    idx: int,
    bench_ret60_pct: float = 0.0,
) -> dict[str, Any] | None:
    """Compute strictly causal selection, anti-extension & PIT risk features at bar `idx`."""
    if idx < 60 or idx >= len(candles):
        return None
    bar = candles[idx]
    close = float(bar.get("close") or 0.0)
    if close <= 0.0:
        return None

    closes_w = [float(c.get("close") or 0.0) for c in candles[idx - 60 : idx + 1]]
    if any(x <= 0.0 for x in closes_w):
        return None

    sma20 = sum(closes_w[-20:]) / 20.0
    sma50 = sum(closes_w[-50:]) / 50.0
    pct_above_sma20 = ((close / max(1e-6, sma20)) - 1.0) * 100.0
    pct_above_sma50 = ((close / max(1e-6, sma50)) - 1.0) * 100.0
    sma20_vs_sma50 = ((sma20 / max(1e-6, sma50)) - 1.0) * 100.0

    # RSI(14)
    gains = [max(0.0, closes_w[-15 + j] - closes_w[-16 + j]) for j in range(1, 15)]
    losses = [max(0.0, closes_w[-16 + j] - closes_w[-15 + j]) for j in range(1, 15)]
    ag = sum(gains) / 14.0
    al = sum(losses) / 14.0
    rsi14 = 50.0 if (ag == 0 and al == 0) else (85.0 if al == 0 else max(5.0, min(95.0, 100.0 - 100.0 / (1.0 + ag / al))))

    # EMA-12/26 alignment proxy over trailing 61 bars
    k12, k26, k9 = 2.0 / 13.0, 2.0 / 27.0, 2.0 / 10.0
    e12, e26, sig_l = closes_w[0], closes_w[0], 0.0
    for c_val in closes_w[1:]:
        e12 = c_val * k12 + e12 * (1.0 - k12)
        e26 = c_val * k26 + e26 * (1.0 - k26)
        m = e12 - e26
        sig_l = m * k9 + sig_l * (1.0 - k9)
    macd = e12 - e26

    # Orthogonal selection factors (verified against collinearity audit):
    # 1) Trend strength (longest-horizon causal edge)
    trend_strength = pct_above_sma50 + sma20_vs_sma50
    # 2) 60D Relative Strength vs Nifty
    ret60_pct = ((close / max(1e-6, closes_w[0])) - 1.0) * 100.0
    rs_vs_nifty_60d = ret60_pct - bench_ret60_pct
    # 3) Momentum consistency (fraction of positive days over 20D)
    pos_days_20 = sum(1 for j in range(-20, 0) if closes_w[j] > closes_w[j - 1]) / 20.0
    # 4) ATR(14)% & Trailing 60D peak-to-close drawdown & zero-vol count
    atr14 = compute_atr(candles[max(0, idx - 25) : idx + 1], period=14)
    atr_pct = (atr14 / max(1e-6, close)) * 100.0
    peak60 = max(closes_w)
    dd60_pct = max(0.0, ((peak60 - close) / max(1e-6, peak60)) * 100.0)

    vols20 = [int(c.get("volume") or 0) for c in candles[idx - 19 : idx + 1]]
    zero_vol_20 = sum(1 for v in vols20 if v <= 0)
    avg_vol20 = sum(vols20) / 20.0
    avg_turnover20 = avg_vol20 * close

    # Prior 20D high breakout check (strictly prior bars excluding bar idx)
    prev_highs20 = [float(c.get("high") or c.get("close") or 0.0) for c in candles[idx - 20 : idx]]
    is_20d_breakout = close > max(prev_highs20) if prev_highs20 else False

    # Base M2.1-compatible technical confidence score
    slope_up = (close >= sum(closes_w[-15:]) / 15.0) and (sma20 >= sum(closes_w[-25:-5]) / 20.0)
    t_score = 0.50
    t_score += 0.14 if close >= sma20 else -0.12
    t_score += 0.12 if e12 >= e26 else -0.10
    t_score += 0.08 if macd >= sig_l else -0.08
    t_score += 0.10 if slope_up else -0.10
    if 46.0 <= rsi14 <= 70.0:
        t_score += 0.07
    tech_conf = round(max(0.18, min(0.96, t_score)), 3)
    tech_sig = "BULLISH" if tech_conf >= 0.58 else ("BEARISH" if tech_conf <= 0.44 else "NEUTRAL")

    return {
        "idx": idx,
        "date": str(bar.get("date", "")),
        "close": close,
        "sma20": round(sma20, 4),
        "sma50": round(sma50, 4),
        "rsi14": round(rsi14, 2),
        "atr14": round(atr14, 4),
        "atr_pct": round(atr_pct, 2),
        "pct_above_sma20": round(pct_above_sma20, 2),
        "pct_above_sma50": round(pct_above_sma50, 2),
        "sma20_vs_sma50": round(sma20_vs_sma50, 2),
        "trend_strength": round(trend_strength, 2),
        "rs_vs_nifty_60d": round(rs_vs_nifty_60d, 2),
        "mom_consistency": round(pos_days_20, 3),
        "dd60_pct": round(dd60_pct, 2),
        "zero_vol_20": zero_vol_20,
        "avg_vol20": avg_vol20,
        "avg_turnover20": avg_turnover20,
        "is_20d_breakout": is_20d_breakout,
        "tech_conf": tech_conf,
        "tech_sig": tech_sig,
    }


def score_candidate_v2_2(
    *,
    symbol: str,
    sector: str,
    feats: dict[str, Any],
    candles_window: list[dict[str, Any]],
    regime6: dict[str, Any],
    currency: str = "INR",
    capital: float = 100_000.0,
    cfg: MoreshwarV22Config | None = None,
    tariff: DeliveryCostConfig | None = None,
) -> dict[str, Any]:
    """Evaluate causal eligibility, stop/target geometry & cross-sectional rank score.

    Does NOT allocate portfolio slots yet; returns candidate score so ALL eligible
    candidates on date `t` can be ranked deterministically without alphabetical bias.
    """
    c = cfg or MoreshwarV22Config()
    tc = tariff or DeliveryCostConfig()
    spot = float(feats["close"])
    r_name = str(regime6.get("regime", "SIDEWAYS"))

    reasons: list[str] = []
    cat: str | None = None

    def _rej(c_tag: str, msg: str) -> None:
        nonlocal cat
        if cat is None:
            cat = c_tag
        reasons.append(msg)

    # 1. Base entry trigger (causal technical alignment)
    base_entry_ok = (spot >= feats["sma20"]) and (42.0 <= feats["rsi14"] <= 70.0) and (feats["tech_sig"] == "BULLISH")
    if not base_entry_ok:
        _rej("SIGNAL_NOT_TRIGGERED", "Base SMA20/RSI/Tech alignment inactive.")

    # 2. Optional Regime Gate (Area C)
    if c.require_regime_gate:
        if c.regime_mode == "defensive_6state" and r_name in ("STRONG_BEARISH", "WEAK_BEARISH", "HIGH_VOLATILITY"):
            _rej("REGIME_GATE", f"Blocked in {r_name}.")
        elif c.regime_mode == "bull_only_no_sideways" and r_name not in ("STRONG_BULLISH", "WEAK_BULLISH"):
            _rej("REGIME_GATE", f"Requires STRONG/WEAK BULLISH (current={r_name}).")
        elif c.regime_mode == "strong_bull_only" and r_name != "STRONG_BULLISH":
            _rej("REGIME_GATE", f"Requires STRONG_BULLISH (current={r_name}).")
    else:
        # Match M2.1 baseline 6-state defensive guard when comparing selection steps
        if r_name in ("STRONG_BEARISH", "WEAK_BEARISH", "HIGH_VOLATILITY"):
            _rej("REGIME_GATE", f"Blocked in defensive regime {r_name}.")

    # 3. Structural Trend Filter (Area A / B)
    if c.require_structural_uptrend:
        if not (spot > feats["sma50"] and feats["sma20"] > feats["sma50"]):
            _rej("WEAK_TREND_STRUCTURE", f"Lacks 50D uptrend (close>{feats['sma50']:.1f}, sma20>{feats['sma50']:.1f}).")

    # 4. Anti-Extension Filter (Area A / B: prevent chasing >5% above SMA20 unless tight breakout)
    if c.require_anti_extension:
        ext_cap = c.breakout_extension_cap_pct if feats["is_20d_breakout"] else c.max_extension_sma20_pct
        if feats["pct_above_sma20"] > ext_cap:
            _rej("OVER_EXTENDED", f"Extended +{feats['pct_above_sma20']:.1f}% above SMA20 (cap {ext_cap:.1f}%).")
        if feats["rsi14"] < c.rsi_min or feats["rsi14"] > c.rsi_max:
            _rej("RSI_OUT_OF_BAND", f"RSI {feats['rsi14']:.1f} outside [{c.rsi_min:.0f}, {c.rsi_max:.0f}].")

    # 5. Point-in-Time Stock Risk & Liquidity Hygiene (blocks ZEEL/distressed/illiquid whipsaws causally)
    min_liq = c.min_avg_daily_turnover_inr if c.require_pit_risk_filter else c.baseline_avg_daily_turnover_inr
    if currency != "INR":
        min_liq = min_liq / 80.0
    if feats["avg_vol20"] <= 0 or feats["avg_turnover20"] < min_liq:
        _rej("LIQUIDITY_CAP", f"20D avg turnover {feats['avg_turnover20']:,.0f} < {min_liq:,.0f}.")
    if c.require_pit_risk_filter:
        if feats["zero_vol_20"] > 0:
            _rej("PIT_HALT_RISK", f"{feats['zero_vol_20']} zero-volume bars in trailing 20D.")
        if feats["atr_pct"] > c.max_atr_pct:
            _rej("PIT_HIGH_ATR_RISK", f"ATR {feats['atr_pct']:.2f}% > {c.max_atr_pct:.2f}% stability cap.")
        if feats["dd60_pct"] > c.max_60d_drawdown_pct:
            _rej("PIT_BROKEN_DRAWDOWN", f"Trailing 60D drawdown {feats['dd60_pct']:.1f}% > {c.max_60d_drawdown_pct:.1f}%.")

    # 6. ATR Stop-Loss & Horizon Target Geometry
    lows = [float(b.get("low") or b.get("close") or spot) for b in candles_window]
    highs = [float(b.get("high") or b.get("close") or spot) for b in candles_window]
    atr = float(feats["atr14"])
    raw_stop_dist = atr * c.atr_multiplier
    recent_support = min(lows[-20:]) if lows else spot * 0.95
    recent_resistance = max(highs[-20:]) if highs else spot * 1.05
    sup_dist = spot - max(0.0, recent_support - atr * 0.25)
    stop_dist = sup_dist if (0.85 * raw_stop_dist <= sup_dist <= 1.25 * raw_stop_dist) else raw_stop_dist
    stop_dist = max(spot * (c.min_stop_pct / 100.0), min(spot * (c.max_stop_pct / 100.0), stop_dist))
    sl_px = round(spot - stop_dist, 2)

    expected_move = atr * math.sqrt(max(1, c.horizon_days)) * c.target_atr_horizon_scale
    breakout_cap = max(recent_resistance * 1.03, spot + expected_move * 0.90)
    raw_tp_dist = min(expected_move, breakout_cap - spot, spot * (c.max_target_pct / 100.0))
    natural_rr = round(raw_tp_dist / max(0.01, stop_dist), 2)
    tp_dist = max(raw_tp_dist, stop_dist * c.min_rr_ratio) if raw_tp_dist >= stop_dist * 1.2 else raw_tp_dist
    tp_px = round(spot + max(spot * 0.015, tp_dist), 2)

    risk_per_share = max(0.01, spot - sl_px)
    rr_ratio = round((tp_px - spot) / risk_per_share, 2)
    if natural_rr < c.min_rr_ratio or rr_ratio < c.min_rr_ratio:
        _rej("UNFAVORABLE_RR", f"Natural R:R {natural_rr:.2f} < {c.min_rr_ratio:.2f}.")

    liq_thresh_slip = tc.illiquid_turnover_threshold_inr if currency == "INR" else (tc.illiquid_turnover_threshold_inr / 80.0)
    slip_bps = tc.illiquid_slippage_bps if (feats["avg_turnover20"] < liq_thresh_slip) else tc.base_slippage_bps

    # 7. Cost Hurdle Check (Area F: ensure expected target move clears round-trip cost by multiple)
    tp_pct = ((tp_px - spot) / max(1e-6, spot)) * 100.0
    # Conservative benchmark ticket ~15k INR for percentage fee + DP estimation
    ref_notional = min(capital * (c.max_single_stock_pct / 100.0), max(15_000.0, capital / max(1, c.max_open_positions)))
    est_buy = compute_repaired_delivery_costs(side="BUY", raw_notional=ref_notional, currency=currency, slippage_bps=slip_bps, tariff=tc)
    est_sell = compute_repaired_delivery_costs(side="SELL", raw_notional=ref_notional, currency=currency, slippage_bps=slip_bps, tariff=tc)
    est_rt_cost_pct = ((est_buy["total_cost"] + est_sell["total_cost"]) / max(1.0, ref_notional)) * 100.0
    if c.require_cost_hurdle and tp_pct < est_rt_cost_pct * c.cost_hurdle_mult:
        _rej("COST_HURDLE_FAIL", f"Expected TP +{tp_pct:.2f}% < {c.cost_hurdle_mult:.1f}x cost ({est_rt_cost_pct:.2f}%).")

    # 8. Deterministic Cross-Sectional Composite Rank Score (orthogonal non-volume factors)
    # * Trend Strength (~35%): clipped to [-20, +35] -> normalized [0, 1]
    # * RS 60D vs Nifty (~25%): clipped to [-20, +35] -> normalized [0, 1]
    # * Momentum Consistency (~20%): pos_days_20 (0.35..0.80 -> [0, 1])
    # * Anti-Extension & Stability (~20%): peak score near +0.5%..+3.0% above SMA20 & calm ATR% (<2.5%)
    t_norm = max(0.0, min(1.0, (feats["trend_strength"] + 5.0) / 35.0))
    rs_norm = max(0.0, min(1.0, (feats["rs_vs_nifty_60d"] + 10.0) / 40.0))
    cons_norm = max(0.0, min(1.0, (feats["mom_consistency"] - 0.35) / 0.45))

    ext = feats["pct_above_sma20"]
    ext_score = 1.0 if 0.0 <= ext <= 3.5 else (max(0.0, 1.0 - (ext - 3.5) / 5.0) if ext > 3.5 else 0.3)
    stab_score = max(0.0, min(1.0, (4.0 - feats["atr_pct"]) / 2.5))
    qual_norm = 0.6 * ext_score + 0.4 * stab_score

    rank_score = round((0.35 * t_norm + 0.25 * rs_norm + 0.20 * cons_norm + 0.20 * qual_norm) * 100.0, 3)

    return {
        "symbol": symbol,
        "sector": sector,
        "eligible": len(reasons) == 0,
        "rejection_category": cat,
        "rejection_reasons": reasons,
        "rank_score": rank_score,
        "trend_strength": feats["trend_strength"],
        "rs_vs_nifty_60d": feats["rs_vs_nifty_60d"],
        "pct_above_sma20": feats["pct_above_sma20"],
        "atr_pct": feats["atr_pct"],
        "spot": spot,
        "sl_px": sl_px,
        "tp_px": tp_px,
        "sl_ratio": max(0.01, (spot - sl_px) / max(1e-6, spot)),
        "tp_ratio": max(0.015, (tp_px - spot) / max(1e-6, spot)),
        "rr_ratio": rr_ratio,
        "slippage_bps": slip_bps,
        "est_rt_cost_pct": round(est_rt_cost_pct, 3),
        "market_regime": r_name,
    }


def allocate_ranked_slots_v2_2(
    *,
    candidates: list[dict[str, Any]],
    candles_by_symbol: dict[str, list[dict[str, Any]]],
    open_positions: dict[str, dict[str, Any]],
    cash: float,
    portfolio_nav: float,
    currency: str = "INR",
    cfg: MoreshwarV22Config | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Allocate open portfolio slots across pre-evaluated candidates.

    When `cfg.use_ranked_selection` is True, candidates are sorted strictly by
    `(-rank_score, -trend_strength, symbol)` so portfolio capacity flows to the
    highest-quality opportunities regardless of ticker alphabetical order.
    """
    c = cfg or MoreshwarV22Config()
    eligible = [cand for cand in candidates if cand["eligible"] and cand["symbol"] not in open_positions]
    if c.use_ranked_selection:
        eligible.sort(key=lambda x: (-x["rank_score"], -x["trend_strength"], x["symbol"]))

    # Track tentative portfolio state during batch allocation on day t
    tentative_open = [
        {
            "symbol": s,
            "sector": p["sector"],
            "entry_price": p["entry_price"],
            "sl": p["sl"],
            "qty": p["qty"],
            "recent_closes": p.get("recent_closes", []),
            "notional": p["last_price"] * p["qty"],
        }
        for s, p in open_positions.items()
    ]
    deployed_amt = sum(p["notional"] for p in tentative_open)
    orders: list[dict[str, Any]] = []
    reject_counts: dict[str, int] = {}

    def _bump(cat: str) -> None:
        reject_counts[cat] = reject_counts.get(cat, 0) + 1

    for cand in eligible:
        sym = cand["symbol"]
        sec = cand["sector"]
        spot = cand["spot"]
        sl_px = cand["sl_px"]
        r_name = cand["market_regime"]

        if len(tentative_open) >= c.max_open_positions:
            _bump("PORTFOLIO_CAP_OR_CORR")
            continue
        if sum(1 for p in tentative_open if p["sector"] == sec) >= c.max_sector_positions:
            _bump("PORTFOLIO_CAP_OR_CORR")
            continue

        c_closes = [float(b["close"]) for b in (candles_by_symbol.get(sym) or [])[-61:]]
        max_corr = 0.0
        for op in tentative_open:
            if op["recent_closes"] and c_closes:
                max_corr = max(max_corr, compute_pairwise_return_correlation(c_closes, op["recent_closes"], window=60))
        if max_corr > c.max_pairwise_corr:
            _bump("PORTFOLIO_CAP_OR_CORR")
            continue

        existing_sl_risk = sum(max(0.0, (p["entry_price"] - p["sl"]) * p["qty"]) for p in tentative_open)
        rem_sl_budget = max(0.0, portfolio_nav * (c.max_portfolio_stop_risk_pct / 100.0) - existing_sl_risk)
        risk_per_share = max(0.01, spot - sl_px)
        if rem_sl_budget < risk_per_share:
            _bump("PORTFOLIO_CAP_OR_CORR")
            continue

        regime_mult = 1.0 if r_name == "STRONG_BULLISH" else (0.80 if r_name == "WEAK_BULLISH" else 0.60)
        corr_mult = 0.65 if max_corr >= 0.65 else 1.0
        vol_mult = max(0.4, c.max_atr_pct / max(0.1, cand["atr_pct"])) if cand["atr_pct"] > c.max_atr_pct else 1.0
        eff_risk_amt = min(
            portfolio_nav * (c.risk_per_trade_pct / 100.0) * regime_mult * corr_mult * vol_mult,
            rem_sl_budget,
        )

        rem_cap = max(0.0, portfolio_nav * (c.max_portfolio_exposure_pct / 100.0) - deployed_amt)
        allowed_notional = min(portfolio_nav * (c.max_single_stock_pct / 100.0), rem_cap, cash * 0.96)
        risk_qty = int(eff_risk_amt // risk_per_share)
        cap_qty = int(allowed_notional // max(1e-6, spot))
        qty = max(0, min(risk_qty, cap_qty))

        min_viable = c.min_viable_notional_inr if currency == "INR" else c.min_viable_notional_usd
        if qty < 1 or (qty * spot) < min_viable:
            _bump("ZERO_SHARES")
            continue

        notional = qty * spot
        deployed_amt += notional
        cash -= notional
        tentative_open.append({
            "symbol": sym,
            "sector": sec,
            "entry_price": spot,
            "sl": sl_px,
            "qty": qty,
            "recent_closes": c_closes,
            "notional": notional,
        })
        orders.append({**cand, "req_qty": qty})

    return orders, reject_counts
