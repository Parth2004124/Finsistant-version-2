"""Adversarial Bear Prosecutor & Survival Engine (Zero Mercy, Zero Fabrication).

Purpose
-------
Before a single rupee is allocated to ANY candidate (e.g., RELIANCE or any NSE equity),
this engine actively tries to PROVE that buying right now is the WORST possible decision
across 10 adversarial stress vectors built strictly on real OHLCV, Nifty 50 benchmark,
verified fundamentals, and execution telemetry.

Rules:
- Zero fabricated metrics or default scores: missing required evidence becomes an explicit
  prosecution penalty / veto rather than being filled with optimism.
- Fatal vetoes CANNOT be rebutted — even a single fatal defect kills the BUY immediately.
- Non-fatal negative points deduct directly from the bull conviction defense score (`survival_score`)
  and can only be partially offset by hard quantitative counter-evidence.
- Learned per-rupee attack multipliers from previous losing trades harden future prosecution
  so every lost rupee permanently sharpens the system's skepticism.
"""

from __future__ import annotations

import math
from typing import Any

DEFAULT_MIN_SURVIVAL_SCORE = 58.0
ATTACK_MULTIPLIER_MIN = 0.75
ATTACK_MULTIPLIER_MAX = 3.50


def _clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(val)))


def _safe_float(val: Any) -> float | None:
    if val is None or isinstance(val, bool):
        return None
    try:
        f = float(val)
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def _compute_realized_candle_risk(candles: list[dict[str, Any]]) -> dict[str, float | None]:
    """Compute realized 36D/60D tail drawdown, worst 1D drop, ATR%, and annualized vol strictly from real bars."""
    if not candles or len(candles) < 5:
        return {
            "max_dd_pct": None,
            "worst_1d_drop_pct": None,
            "atr14_pct": None,
            "hl_range_pct": None,
            "ann_vol_pct": None,
            "stock_window_ret_pct": None,
        }

    closes = [_safe_float(c.get("close")) for c in candles]
    highs = [_safe_float(c.get("high")) for c in candles]
    lows = [_safe_float(c.get("low")) for c in candles]

    valid_closes = [x for x in closes if x is not None and x > 0]
    if len(valid_closes) < 5:
        return {
            "max_dd_pct": None,
            "worst_1d_drop_pct": None,
            "atr14_pct": None,
            "hl_range_pct": None,
            "ann_vol_pct": None,
            "stock_window_ret_pct": None,
        }

    # Trailing window peak-to-trough drawdown
    peak = valid_closes[0]
    max_dd = 0.0
    for c in valid_closes:
        if c > peak:
            peak = c
        dd = ((peak - c) / peak) * 100.0 if peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd

    # Daily log/simple returns & worst 1-day drop
    daily_rets: list[float] = []
    worst_drop = 0.0
    for i in range(1, len(valid_closes)):
        r = ((valid_closes[i] / valid_closes[i - 1]) - 1.0) * 100.0
        daily_rets.append(r)
        if r < worst_drop:
            worst_drop = r

    ann_vol = None
    if len(daily_rets) >= 5:
        mean_r = sum(daily_rets) / len(daily_rets)
        var_r = sum((r - mean_r) ** 2 for r in daily_rets) / max(1, len(daily_rets) - 1)
        ann_vol = math.sqrt(max(0.0, var_r)) * math.sqrt(252.0)

    # 14-day ATR % of spot close
    trs: list[float] = []
    for i in range(1, len(candles)):
        h = highs[i]
        l = lows[i]
        pc = closes[i - 1]
        c = closes[i]
        if h is not None and l is not None and pc is not None and c is not None and c > 0:
            tr = max(h - l, abs(h - pc), abs(l - pc))
            trs.append(tr / c * 100.0)
    atr_tail = trs[-14:]
    atr14_pct = (sum(atr_tail) / len(atr_tail)) if atr_tail else None

    last_h = highs[-1]
    last_l = lows[-1]
    last_c = closes[-1]
    hl_range_pct = (
        max(0.0, ((last_h - last_l) / last_c) * 100.0)
        if (last_h is not None and last_l is not None and last_c is not None and last_c > 0)
        else None
    )

    win_ret = ((valid_closes[-1] / valid_closes[0]) - 1.0) * 100.0 if valid_closes[0] > 0 else None
    return {
        "max_dd_pct": round(max_dd, 2),
        "worst_1d_drop_pct": round(abs(min(0.0, worst_drop)), 2),
        "atr14_pct": round(atr14_pct, 2) if atr14_pct is not None else None,
        "hl_range_pct": round(hl_range_pct, 2) if hl_range_pct is not None else None,
        "ann_vol_pct": round(ann_vol, 2) if ann_vol is not None else None,
        "stock_window_ret_pct": round(win_ret, 2) if win_ret is not None else None,
    }


def _compute_benchmark_rel_strength(
    candles: list[dict[str, Any]], benchmark_candles: list[dict[str, Any]] | None
) -> float | None:
    if not candles or not benchmark_candles or len(candles) < 5 or len(benchmark_candles) < 5:
        return None
    stock_by_dt = {str(c.get("date")): _safe_float(c.get("close")) for c in candles if c.get("date")}
    bench_by_dt = {str(c.get("date")): _safe_float(c.get("close")) for c in benchmark_candles if c.get("date")}
    common = sorted(d for d in stock_by_dt if d in bench_by_dt and stock_by_dt[d] and bench_by_dt[d])
    if len(common) >= 5:
        s0, s1 = stock_by_dt[common[0]], stock_by_dt[common[-1]]
        b0, b1 = bench_by_dt[common[0]], bench_by_dt[common[-1]]
        if s0 and s1 and b0 and b1 and s0 > 0 and b0 > 0:
            return round(((s1 / s0 - 1.0) - (b1 / b0 - 1.0)) * 100.0, 2)
    return None


def prosecute_trade_candidate(
    *,
    symbol: str,
    snap: dict[str, Any],
    tech_conf: float,
    fund_conf: float,
    rr_ratio: float,
    stop_loss: float,
    target: float,
    price: float,
    adv_20d_inr: float | None = None,
    hl_range_pct: float | None = None,
    price_discrepancy_pct: float | None = None,
    benchmark_candles: list[dict[str, Any]] | None = None,
    attack_multipliers: dict[str, float] | None = None,
    min_survival_score: float = DEFAULT_MIN_SURVIVAL_SCORE,
    est_round_trip_cost_bps: float = 42.0,
) -> dict[str, Any]:
    """Run all 10 adversarial bear prosecution vectors against a BUY candidate.

    Every negative point must either be quantitatively outlasted or the BUY is KILLED.
    """
    mult_map = attack_multipliers or {}
    f_score = snap.get("fundamental_score") or {}
    p_score = snap.get("porters_score") or {}
    ratios = f_score.get("ratios") or {}
    industry = str(f_score.get("industry") or "").upper()
    is_financial = industry == "BANKING"

    pe = _safe_float(ratios.get("pe"))
    roe = _safe_float(ratios.get("roe"))
    roce = _safe_float(ratios.get("roce"))
    opm = _safe_float(ratios.get("opm"))
    sales_g = _safe_float(ratios.get("sales_growth"))
    profit_g = _safe_float(ratios.get("profit_growth"))
    dte = _safe_float(ratios.get("debt_to_equity"))
    mcap_cr = _safe_float(ratios.get("mcap_cr"))
    ret_1y = _safe_float(snap.get("ret_1y") if snap.get("ret_1y") is not None else ratios.get("ret_1y"))

    rsi_14 = _safe_float(snap.get("rsi_14"))
    sma_20 = _safe_float(snap.get("sma_20"))
    macd = _safe_float(snap.get("macd"))
    macd_sig = _safe_float(snap.get("macd_signal"))
    bb_upper = _safe_float(snap.get("bb_upper"))
    reg_curve = str(snap.get("regression_curve") or "")
    proj_30d = _safe_float(snap.get("projected_30d_target"))

    candles = list(snap.get("candles") or [])
    c_risk = _compute_realized_candle_risk(candles)
    rel_nifty_pct = _compute_benchmark_rel_strength(candles, benchmark_candles)
    eff_hl = hl_range_pct if hl_range_pct is not None else c_risk["hl_range_pct"]

    f_total = float(f_score.get("total") or 0.0)
    p_total = float(p_score.get("total") or 0.0)
    missing_fields = list(snap.get("missing_fields") or [])

    attacks: list[dict[str, Any]] = []

    def _add_attack(
        attack_id: str,
        title: str,
        severity: str,
        base_damage: float,
        evidence: str,
        prosecution_case: str,
        rebuttal_evidence: str | None = None,
        mitigation_pts: float = 0.0,
    ) -> None:
        mult = _clamp(float(mult_map.get(attack_id, 1.0)), ATTACK_MULTIPLIER_MIN, ATTACK_MULTIPLIER_MAX)
        scaled = base_damage * mult
        # FATAL_VETO can NEVER be mitigated or offset
        eff_mitigation = 0.0 if severity == "FATAL_VETO" else min(scaled * 0.65, max(0.0, mitigation_pts))
        net_damage = round(max(0.0, scaled - eff_mitigation), 2)
        status = (
            "FATAL_VETO"
            if severity == "FATAL_VETO"
            else ("MITIGATED_SURVIVED" if eff_mitigation >= scaled * 0.50 else "UNREFUTED")
        )
        attacks.append(
            {
                "attack_id": attack_id,
                "title": title,
                "severity": severity,
                "status": status,
                "base_damage": round(base_damage, 2),
                "learned_multiplier": round(mult, 3),
                "mitigation_pts": round(eff_mitigation, 2),
                "net_damage": net_damage,
                "evidence": evidence,
                "prosecution_case": prosecution_case,
                "rebuttal": rebuttal_evidence if eff_mitigation > 0 else "No quantitative rebuttal cleared this objection.",
            }
        )

    # 1. Valuation Overextension / PEG Trap
    if pe is not None and pe <= 0:
        can_rebut_loss_pe = mcap_cr is not None and mcap_cr > 75_000 and sales_g is not None and sales_g > 12 and f_total >= 72
        _add_attack(
            "VALUATION_EXTREME_TRAP",
            "Negative / Loss-Making TTM Earnings",
            "FATAL_VETO" if not can_rebut_loss_pe else "SEVERE",
            28.0,
            f"P/E={pe:.1f}x, MCap=₹{mcap_cr or 0:,.0f}Cr, SalesG={sales_g if sales_g is not None else 'NA'}%",
            f"{symbol} has non-positive trailing earnings — paying equity capital for losses carries terminal dilution risk.",
            rebuttal_evidence="Mega-cap turnaround (>₹75k Cr) with >12% sales acceleration partially offsets loss multiple." if can_rebut_loss_pe else None,
            mitigation_pts=6.0 if can_rebut_loss_pe else 0.0,
        )
    elif pe is not None and pe > 50.0 and (profit_g is None or profit_g < 22.0 or (roe is not None and roe < 18.0)):
        _add_attack(
            "VALUATION_EXTREME_TRAP",
            "Bubble Multiple / PEG Distortion",
            "SEVERE" if pe < 75.0 else "FATAL_VETO",
            22.0 if pe < 75.0 else 32.0,
            f"P/E={pe:.1f}x vs ProfitGrowth={profit_g if profit_g is not None else 'NA'}%, ROE={roe if roe is not None else 'NA'}%",
            f"{symbol} trades at an unforgiving {pe:.1f}x earnings without elite (>22%) profit compounding to justify it.",
        )
    elif pe is not None and pe > 28.0 and profit_g is not None and profit_g > 0 and (pe / profit_g) > 2.4:
        reb = None
        mit = 0.0
        if roe is not None and roce is not None and roe >= 22.0 and roce >= 22.0 and (dte is None or dte <= 0.35):
            reb = f"Elite capital efficiency (ROE {roe:.1f}%, ROCE {roce:.1f}%, D/E {dte or 0:.2f}) cushions elevated PEG {pe/profit_g:.2f}."
            mit = 5.0
        _add_attack(
            "VALUATION_EXTREME_TRAP",
            "Expensive PEG Valuation Stretch",
            "MODERATE",
            11.0,
            f"P/E={pe:.1f}x, ProfitG={profit_g:.1f}% (PEG={pe/profit_g:.2f})",
            f"Paying {pe:.1f}x for {profit_g:.1f}% profit growth leaves thin margin of safety against any earnings miss.",
            rebuttal_evidence=reb,
            mitigation_pts=mit,
        )

    # 2. Capital Efficiency & Debt-Inflated ROE Trap
    if not is_financial and roce is not None and roce < 11.0:
        _add_attack(
            "CAPITAL_DESTRUCTION_ROCE",
            "Sub-WACC Capital Efficiency Trap",
            "FATAL_VETO" if roce < 7.0 else "SEVERE",
            26.0 if roce < 7.0 else 16.0,
            f"ROCE={roce:.1f}% (hurdle >=11%), ROE={roe if roe is not None else 'NA'}%",
            f"{symbol} earns only {roce:.1f}% ROCE — barely clearing Indian risk-free/debt cost; growth destroys or dilutes real rupee value.",
        )
    elif not is_financial and roe is not None and roce is not None and (roe - roce) > 10.0 and (dte or 0.0) > 1.0:
        _add_attack(
            "CAPITAL_DESTRUCTION_ROCE",
            "Leverage-Inflated ROE Illusion",
            "SEVERE",
            14.0,
            f"ROE={roe:.1f}% vs ROCE={roce:.1f}% (gap {roe - roce:.1f}pp), D/E={dte:.2f}x",
            f"Headline ROE ({roe:.1f}%) is artificially leveraged over operating ROCE ({roce:.1f}%) via debt.",
        )

    # 3. Balance-Sheet Leverage & Margin Fragility
    if not is_financial and dte is not None and dte > 1.25:
        sev = "FATAL_VETO" if dte > 2.2 else "SEVERE"
        _add_attack(
            "LEVERAGE_SOLVENCY_BOMB",
            "Excessive Balance-Sheet Leverage",
            sev,
            30.0 if sev == "FATAL_VETO" else 17.0,
            f"Debt/Equity={dte:.2f}x, OPM={opm if opm is not None else 'NA'}%",
            f"Debt/Equity of {dte:.2f}x exposes {symbol} to severe refinancing, credit-downgrade, and cyclical margin compression.",
        )
    if not is_financial and opm is not None and opm < 8.0:
        _add_attack(
            "LEVERAGE_SOLVENCY_BOMB",
            "Razor-Thin Operating Moat",
            "SEVERE" if opm < 4.5 else "MODERATE",
            15.0 if opm < 4.5 else 9.0,
            f"OPM={opm:.1f}%",
            f"Operating margin of {opm:.1f}% offers zero buffer against raw-material inflation or pricing wars.",
        )

    # 4. Growth Deceleration / Contracting Earnings
    if sales_g is not None and profit_g is not None and (sales_g < 0.0 or profit_g < 0.0):
        both_bad = sales_g < -5.0 and profit_g < -8.0
        _add_attack(
            "GROWTH_DECELERATION",
            "Contracting Revenue / Earnings Trajectory",
            "FATAL_VETO" if both_bad else "SEVERE",
            28.0 if both_bad else 16.0,
            f"SalesGrowth={sales_g:.1f}%, ProfitGrowth={profit_g:.1f}%",
            f"Buying a contracting business (Sales {sales_g:+.1f}%, Profit {profit_g:+.1f}%) risks catching a falling earnings knife.",
        )

    # 5. Technical Blow-Off / Bull-Trap Momentum Divergence
    sma_stretch_pct = ((price / sma_20) - 1.0) * 100.0 if (sma_20 is not None and sma_20 > 0) else 0.0
    if (rsi_14 is not None and rsi_14 > 74.0) or sma_stretch_pct > 9.0 or (bb_upper is not None and price > bb_upper * 1.005):
        reb = None
        mit = 0.0
        if macd is not None and macd_sig is not None and macd > macd_sig and f_total >= 78 and p_total >= 75:
            reb = "Elite Fundamental (78+) & Porter (75+) confirmation with intact MACD trend slightly eases mean-reversion risk."
            mit = 4.0
        _add_attack(
            "TECHNICAL_BLOWOFF_TRAP",
            "Overbought Chasing / Mean-Reversion Blow-Off",
            "SEVERE" if (rsi_14 or 0.0) > 78.0 or sma_stretch_pct > 12.0 else "MODERATE",
            16.0 if (rsi_14 or 0.0) > 78.0 else 11.0,
            f"RSI(14)={rsi_14 if rsi_14 is not None else 'NA'}, StretchVsSMA20={sma_stretch_pct:+.1f}%, Price={price:.2f} vs BB_U={bb_upper if bb_upper is not None else 'NA'}",
            f"Chasing {symbol} stretched {sma_stretch_pct:+.1f}% above its 20D average (RSI {rsi_14 or 0:.1f}) invites immediate pullback onto our stop-loss.",
            rebuttal_evidence=reb,
            mitigation_pts=mit,
        )
    if (macd is not None and macd_sig is not None and macd < macd_sig) or ("Downward" in reg_curve):
        _add_attack(
            "TECHNICAL_BLOWOFF_TRAP",
            "Bearish Momentum Divergence / Regression Roll-Over",
            "SEVERE" if ("Downward" in reg_curve and (macd or 0) < (macd_sig or 0)) else "MODERATE",
            15.0 if ("Downward" in reg_curve) else 9.0,
            f"MACD={macd if macd is not None else 'NA'} vs Sig={macd_sig if macd_sig is not None else 'NA'}, Curve={reg_curve or 'NA'}",
            f"Short-term momentum is rolling over ({reg_curve}, MACD below signal) — trend timing contradicts an aggressive BUY.",
        )

    # 6. Historical Tail Crash & Tight Stop Whipsaw Trap
    stop_dist_pct = ((price - stop_loss) / price) * 100.0 if (price > 0 and stop_loss > 0 and price > stop_loss) else 0.0
    if c_risk["max_dd_pct"] is not None and (c_risk["max_dd_pct"] > 15.0 or (c_risk["worst_1d_drop_pct"] or 0.0) > 5.5):
        _add_attack(
            "TAIL_CRASH_DRAWDOWN_RISK",
            "Recent Tail-Crash & Gap-Down Vulnerability",
            "SEVERE" if c_risk["max_dd_pct"] > 20.0 else "MODERATE",
            15.0 if c_risk["max_dd_pct"] > 20.0 else 10.0,
            f"RecentMaxDD={c_risk['max_dd_pct']:.1f}%, Worst1DDrop=-{c_risk['worst_1d_drop_pct']:.1f}%, AnnVol={c_risk['ann_vol_pct'] or 'NA'}%",
            f"{symbol} recently suffered a -{c_risk['max_dd_pct']:.1f}% drawdown and -{c_risk['worst_1d_drop_pct']:.1f}% single-day shock.",
        )
    if c_risk["atr14_pct"] is not None and stop_dist_pct > 0 and stop_dist_pct < (1.15 * c_risk["atr14_pct"]):
        _add_attack(
            "TAIL_CRASH_DRAWDOWN_RISK",
            "Stop-Loss Inside Daily ATR Noise (Whipsaw Guarantee)",
            "SEVERE",
            14.0,
            f"StopDist={stop_dist_pct:.2f}% < 1.15 * ATR14({c_risk['atr14_pct']:.2f}%)={1.15 * c_risk['atr14_pct']:.2f}%",
            f"Stop-loss at ₹{stop_loss:,.2f} ({stop_dist_pct:.2f}% away) sits inside routine 14D ATR noise ({c_risk['atr14_pct']:.2f}%) — high risk of random stop-out.",
        )

    # 7. Net Expectancy, 20%+ Compounding Hurdle & Execution Cost Drag Attack
    upside_pct = ((target / price) - 1.0) * 100.0 if (price > 0 and target > price) else 0.0
    cost_pct = est_round_trip_cost_bps / 100.0
    net_upside_pct = upside_pct - cost_pct
    net_rr = (net_upside_pct / stop_dist_pct) if stop_dist_pct > 0 else 0.0
    if rr_ratio < 1.45 or net_upside_pct <= 2.0 or net_rr < 1.30:
        is_fatal = rr_ratio < 1.30 or net_upside_pct <= 2.0
        _add_attack(
            "RISK_REWARD_EXPECTANCY_FAIL",
            "Insufficient Post-Cost Risk/Reward for 20%+ Survival Target",
            "FATAL_VETO" if is_fatal else "SEVERE",
            32.0 if is_fatal else 19.0,
            f"Raw R:R=1:{rr_ratio:.2f}, TargetUpside={upside_pct:.2f}%, Friction={cost_pct:.2f}%, Net R:R=1:{net_rr:.2f}",
            f"After STT, brokerage, DP, and slippage ({est_round_trip_cost_bps:.0f} bps), net upside ({net_upside_pct:.2f}%) cannot compound portfolio past the 20% survival hurdle.",
        )
    elif rr_ratio < 1.85 or net_upside_pct < 8.5:
        _add_attack(
            "RISK_REWARD_EXPECTANCY_FAIL",
            "Sub-20% Compounding Upside Cushion",
            "SEVERE" if net_upside_pct < 6.0 else "MODERATE",
            15.0 if net_upside_pct < 6.0 else 10.0,
            f"Raw R:R=1:{rr_ratio:.2f}, NetUpside={net_upside_pct:.2f}% (high-stakes hurdle >=8.5% upside, >=1.85 R:R)",
            f"Shallow {net_upside_pct:.2f}% net upside / 1:{rr_ratio:.2f} R:R drags capital below the mandatory 20% survival pace.",
            rebuttal_evidence=f"Tech ({tech_conf*100:.0f}%) & Fund ({fund_conf*100:.0f}%) alignment partially supports breakeven velocity." if (tech_conf >= 0.78 and fund_conf >= 0.80) else None,
            mitigation_pts=3.5 if (tech_conf >= 0.78 and fund_conf >= 0.80) else 0.0,
        )

    # 8. Liquidity Impact & Intraday Spread Instability
    if adv_20d_inr is not None and adv_20d_inr > 0 and adv_20d_inr < 20_000_000.0:
        _add_attack(
            "LIQUIDITY_GAP_TRAP",
            "Thin Institutional Liquidity / Impact Slippage",
            "SEVERE" if adv_20d_inr < 10_000_000.0 else "MODERATE",
            16.0 if adv_20d_inr < 10_000_000.0 else 9.0,
            f"20D ADV=₹{adv_20d_inr/1e7:.2f}Cr (minimum institutional comfort ₹2.0Cr+)",
            f"20D average turnover of ₹{adv_20d_inr/1e7:.2f}Cr increases entry/exit market-impact slippage during stress exits.",
        )
    if eff_hl is not None and eff_hl > 6.5:
        _add_attack(
            "LIQUIDITY_GAP_TRAP",
            "Extreme Intraday Spread / Instability",
            "MODERATE",
            9.0,
            f"H-L Range={eff_hl:.2f}%",
            f"Wide {eff_hl:.2f}% daily high-low swing signals unstable order-book depth or event volatility.",
        )

    # 9. Opportunity Cost vs Nifty 50 & 1Y Structural Laggard Check
    if (rel_nifty_pct is not None and rel_nifty_pct < -5.0) or (ret_1y is not None and ret_1y < -8.0):
        _add_attack(
            "RELATIVE_LAGGARD_OPPORTUNITY_COST",
            "Structural Laggard vs Benchmark / 1Y Underperformance",
            "SEVERE" if ((rel_nifty_pct or 0.0) < -10.0 and (ret_1y or 0.0) < -12.0) else "MODERATE",
            15.0 if ((rel_nifty_pct or 0.0) < -10.0) else 9.0,
            f"RelNifty={rel_nifty_pct if rel_nifty_pct is not None else 'NA'}%, Ret1Y={ret_1y if ret_1y is not None else 'NA'}%",
            f"Capital locked in a laggard (1Y {ret_1y if ret_1y is not None else 'NA'}%, vs Nifty {rel_nifty_pct if rel_nifty_pct is not None else 'NA'}%) carries severe opportunity cost.",
        )

    # 10. Data Completeness & Price Integrity Zero-Mercy Check
    if len(missing_fields) >= 2 or (price_discrepancy_pct is not None and price_discrepancy_pct > 2.5):
        sev = "FATAL_VETO" if (len(missing_fields) >= 4 or (price_discrepancy_pct or 0.0) > 4.0) else "SEVERE"
        _add_attack(
            "DATA_OPACITY_PROVENANCE",
            "Missing Verified Telemetry / Price Feed Divergence",
            sev,
            30.0 if sev == "FATAL_VETO" else 14.0,
            f"missing_fields={missing_fields}, discrepancy={price_discrepancy_pct if price_discrepancy_pct is not None else 0.0:.2f}%",
            f"Unverified fields ({', '.join(missing_fields) or 'none'}) or {price_discrepancy_pct or 0:.2f}% feed gap violates zero-assumption capital safety.",
        )

    # Quantitative Bull Defense vs Brutal Bear Prosecution Damage
    raw_defense = 100.0 * (0.45 * float(tech_conf) + 0.55 * float(fund_conf)) + min(12.0, max(0.0, (float(rr_ratio) - 1.5) * 8.0))
    if proj_30d is not None and price > 0 and proj_30d > price * 1.04:
        raw_defense += min(4.0, ((proj_30d / price - 1.0) * 100.0) * 0.35)
    total_damage = round(sum(a["net_damage"] for a in attacks), 2)
    fatal_attacks = [a for a in attacks if a["severity"] == "FATAL_VETO"]
    unrefuted_severe = [a for a in attacks if a["severity"] == "SEVERE" and a["status"] != "MITIGATED_SURVIVED"]

    survival_score = round(_clamp(raw_defense - total_damage, 0.0, 100.0), 2)
    survival_margin = round(survival_score - float(min_survival_score), 2)

    outlasted = (
        len(fatal_attacks) == 0
        and len(unrefuted_severe) <= 1
        and survival_score >= float(min_survival_score)
    )
    verdict = "OUTLASTED" if outlasted else "KILLED_BY_ADVERSARY"

    if not outlasted:
        size_multiplier = 0.0
    else:
        # Even surviving decisions get down-sized proportionally to remaining unrefuted risk
        span = max(8.0, 82.0 - float(min_survival_score))
        frac = _clamp((survival_score - float(min_survival_score)) / span, 0.0, 1.0)
        size_multiplier = round(0.50 + 0.50 * frac, 3)

    if fatal_attacks:
        kill_reason = f"FATAL VETO [{fatal_attacks[0]['attack_id']}]: {fatal_attacks[0]['prosecution_case']}"
    elif len(unrefuted_severe) > 1:
        ids = ", ".join(a["attack_id"] for a in unrefuted_severe)
        kill_reason = f"{len(unrefuted_severe)} unrefuted SEVERE red flags ({ids}) overwhelmed bull thesis."
    elif not outlasted:
        kill_reason = (
            f"Survival score {survival_score:.1f} < required {float(min_survival_score):.1f} "
            f"after {total_damage:.1f} Bear Damage pts across {len(attacks)} negative objections."
        )
    else:
        kill_reason = (
            f"OUTLASTED {len(attacks)} bear objection(s) (-{total_damage:.1f} dmg pts): "
            f"Survival {survival_score:.1f} >= {float(min_survival_score):.1f} (Size x{size_multiplier:.2f})."
        )

    return {
        "symbol": symbol,
        "verdict": verdict,
        "outlasted": outlasted,
        "survival_score": survival_score,
        "min_survival_score": round(float(min_survival_score), 2),
        "survival_margin": survival_margin,
        "raw_defense_score": round(raw_defense, 2),
        "total_damage_pts": total_damage,
        "size_multiplier": size_multiplier,
        "attack_count": len(attacks),
        "fatal_count": len(fatal_attacks),
        "unrefuted_severe_count": len(unrefuted_severe),
        "attack_ids": sorted({a["attack_id"] for a in attacks}),
        "summary": kill_reason,
        "realized_risk": c_risk,
        "rel_nifty_pct": rel_nifty_pct,
        "attacks": attacks,
    }
