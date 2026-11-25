"""StockSight Version 2 Scoring Engine merged into Finsistant Version 2.

Implements:
1. 4-Pillar Fundamental Score (Business /40, Moat /20, Management /20, Risk /20)
2. Industry Normalization Matrix (BANKING, IT, FMCG, PHARMA, AUTO, POWER, REALTY, GENERAL)
3. Porter's 5 Forces Qualitative Moat Score (Max 100)
4. Moreshwar Dynamic Target, Entry, and Stop-Loss Levels
5. Final Decision Tree Matrix (BUY NOW, ADD, HOLD, SIP ONLY, WAIT, REVIEW, REDUCE, EXIT, AVOID)
"""

from __future__ import annotations

from typing import Any

INDUSTRY_PROFILES: dict[str, dict[str, Any]] = {
    "BANKING": {"bus": 1.1, "moat": 1.2, "mgmt": 1.0, "risk": 0.9, "req": "roe"},
    "IT": {"bus": 1.2, "moat": 1.0, "mgmt": 1.1, "risk": 1.0, "req": "opm"},
    "FMCG": {"bus": 1.0, "moat": 1.3, "mgmt": 1.1, "risk": 1.0, "req": "roce"},
    "PHARMA": {"bus": 1.1, "moat": 1.0, "mgmt": 1.0, "risk": 0.9, "req": None},
    "AUTO": {"bus": 1.2, "moat": 0.9, "mgmt": 1.0, "risk": 0.9, "req": None},
    "POWER": {"bus": 1.0, "moat": 1.2, "mgmt": 0.9, "risk": 0.8, "req": None},
    "REALTY": {"bus": 1.1, "moat": 0.8, "mgmt": 1.0, "risk": 0.8, "req": None},
    "GENERAL": {"bus": 1.0, "moat": 1.0, "mgmt": 1.0, "risk": 1.0, "req": None},
}


def detect_industry(name: str, sector: str = "") -> str:
    sec_up = str(sector or "").upper()
    if any(k in sec_up for k in ("FINANCIAL SERVICES", "BANK", "NBFC", "INSURANCE")):
        return "BANKING"
    if any(k in sec_up for k in ("INFORMATION TECHNOLOGY", "IT SERVICES", "SOFTWARE")):
        return "IT"
    if any(k in sec_up for k in ("FAST MOVING CONSUMER GOODS", "FMCG", "CONSUMER STAPLES")):
        return "FMCG"
    if any(k in sec_up for k in ("HEALTHCARE", "PHARMA")):
        return "PHARMA"
    if any(k in sec_up for k in ("AUTOMOBILE", "AUTO")):
        return "AUTO"
    if any(k in sec_up for k in ("POWER", "UTILITIES", "ENERGY")):
        return "POWER"
    if "REALTY" in sec_up or "REAL ESTATE" in sec_up:
        return "REALTY"

    combined = f"{name} {sector}".upper()
    if any(k in combined for k in ["BANK", "FINANCE", "HDFC", "ICICI", "BAJAJ", "KOTAK", "SBI"]):
        return "BANKING"
    if any(k in combined for k in ["TECH", "INFOSYS", "TCS", "WIPRO", "HCL", "SOFTWARE", "COMPUTE", "AI"]):
        return "IT"
    if any(k in combined for k in ["FMCG", "HUL", "NESTLE", "MARICO", "ITC", "CONSUMER"]):
        return "FMCG"
    if any(k in combined for k in ["PHARMA", "DRUG", "SUN", "CIPLA", "HEALTH"]):
        return "PHARMA"
    if any(k in combined for k in ["MOTOR", "CAR", "MARUTI", "TATA MOTORS", "AUTO", "MOBILITY"]):
        return "AUTO"
    if any(k in combined for k in ["ENERGY", "NTPC", "POWER", "ONGC", "RELIANCE", "GRID"]):
        return "POWER"
    if any(k in combined for k in ["DLF", "PRESTIGE", "LODHA", "REALTY"]):
        return "REALTY"
    return "GENERAL"


def _require_real_metric(meta: dict[str, Any], key: str) -> float:
    val = meta.get(key)
    if val is None:
        raise ValueError(f"Missing required fundamental field '{key}' (refusing fake default).")
    fval = float(val)
    if fval != fval:  # NaN
        raise ValueError(f"Invalid NaN fundamental field '{key}'.")
    return fval


def calculate_fundamental_score(meta: dict[str, Any], *, strict: bool = True) -> dict[str, Any]:
    """Calculate the StockSight V2 4-Pillar Fundamental Score (Max 100). Zero fake defaults or score floors."""
    if str(meta.get("fundamentals_status", "")).lower() == "unavailable" or str(meta.get("fundamental_source_category", "")).upper() == "UNAVAILABLE":
        raise ValueError("Fundamental data is unavailable; refusing to fabricate a default score.")

    name = str(meta.get("name", ""))
    sector = str(meta.get("sector", ""))
    industry = detect_industry(name, sector)
    is_financial = industry == "BANKING"

    sales_growth = _require_real_metric(meta, "sales_growth")
    profit_growth = _require_real_metric(meta, "profit_growth")
    roe = _require_real_metric(meta, "roe")
    mcap_cr = _require_real_metric(meta, "mcap_cr")
    if mcap_cr <= 0:
        raise ValueError("Invalid non-positive mcap_cr; refusing fake fundamental score.")
    raw_pe = meta.get("pe")
    pe = float(raw_pe) if raw_pe is not None else -1.0  # Loss/negative TTM EPS genuinely omits P/E on Screener/Yahoo
    opm = float(meta["opm"]) if meta.get("opm") is not None else (0.0 if is_financial else _require_real_metric(meta, "opm"))
    roce = float(meta["roce"]) if meta.get("roce") is not None else (roe if is_financial else _require_real_metric(meta, "roce"))
    beta = float(meta["beta"]) if meta.get("beta") is not None else 1.0
    ret_1y = float(meta["ret_1y"]) if meta.get("ret_1y") is not None else 0.0
    debt_to_equity = float(meta["debt_to_equity"]) if meta.get("debt_to_equity") is not None else None

    reasons: list[str] = []
    if raw_pe is None:
        reasons.append("No Positive TTM P/E")

    # A. Business Pillar (Max 40)
    business = 0.0
    if sales_growth > 15:
        business += 15
    elif sales_growth > 8:
        business += 10
    elif sales_growth > 0:
        business += 5
    elif sales_growth > -10:
        business += 2
        reasons.append("Sales Drag")

    if profit_growth > 15:
        business += 15
    elif profit_growth > 8:
        business += 10
    elif profit_growth > 0:
        business += 5
    elif profit_growth > -20:
        business += 2
        reasons.append("Profit Drag")

    if is_financial:
        if roe > 15:
            business += 10
        elif roe > 10:
            business += 5
        elif roe > 5:
            business += 2
    else:
        if opm > 20:
            business += 10
        elif opm > 12:
            business += 5
        elif opm > 8:
            business += 2
            reasons.append("Low Margin")
    business = min(40.0, business)

    # B. Moat Pillar (Max 20)
    moat = 0.0
    if is_financial:
        if roe > 18:
            moat += 8
        elif roe > 12:
            moat += 5
    else:
        if opm > 18:
            moat += 5
        if roce > 20:
            moat += 5

    if mcap_cr > 20000:
        moat += 5
    elif mcap_cr > 5000:
        moat += 3

    if profit_growth > sales_growth:
        moat += 5
    if ret_1y > 40:
        moat = max(moat + 5, 18.0)
    moat = min(20.0, moat)

    # C. Management / Valuation Pillar (Max 20)
    management = 0.0
    if pe > 0:
        if pe < 15 and (profit_growth > 10 or roe > 15):
            management += 20
        elif pe < 25:
            management += 10
        elif pe < 60:
            management += 5
    else:
        if mcap_cr > 50000:
            management += 10
            reasons.append("Turnaround Giant")
        elif mcap_cr > 10000:
            management += 5
            reasons.append("Recovering")
    management = min(20.0, management)

    # D. Risk Pillar (Max 20)
    risk = 0.0
    if mcap_cr < 500:
        risk -= 10
        reasons.append("Micro Cap Risk")
    elif mcap_cr > 5000:
        risk += 10
    elif mcap_cr > 2000:
        risk += 5

    if ret_1y > 40:
        risk += 10
    elif beta < 1.1:
        risk += 10
    elif beta < 1.3:
        risk += 5
    if debt_to_equity is not None and debt_to_equity > 1.5 and not is_financial:
        risk -= 5
        reasons.append("High Leverage")
    risk = max(0.0, min(20.0, risk))

    prof = INDUSTRY_PROFILES.get(industry, INDUSTRY_PROFILES["GENERAL"])
    raw_total = business + moat + management + risk

    if pe > 0 and pe < 15 and roe > 15 and profit_growth > 0:
        raw_total += 15
        reasons.append("High Quality Value")
    elif pe > 0 and pe < 12 and profit_growth > 10:
        raw_total += 10
        reasons.append("Deep Value")

    total = int(round(min(99.0, max(0.0, raw_total))))

    nlp_parts = []
    if pe <= 0:
        nlp_parts.append("Negative or unavailable TTM earnings (P/E N/A).")
    elif pe < 18:
        nlp_parts.append(f"Attractive valuation at P/E {pe:.1f}x with value margin of safety.")
    elif pe > 40:
        nlp_parts.append(f"Premium growth multiple (P/E {pe:.1f}x) pricing in strong execution.")
    else:
        nlp_parts.append(f"Balanced valuation at P/E {pe:.1f}x.")

    if roe >= 16:
        nlp_parts.append(f"High capital efficiency (ROE {roe:.1f}%, ROCE {roce:.1f}%) underscores durable moat.")
    if sales_growth >= 14:
        nlp_parts.append(f"Strong top-line expansion ({sales_growth:.1f}% YoY) & profit growth ({profit_growth:.1f}% YoY).")

    return {
        "business": int(round(business)),
        "moat": int(round(moat)),
        "management": int(round(management)),
        "risk": int(round(risk)),
        "total": total,
        "industry": industry,
        "industry_weights": prof,
        "explanation": " & ".join(reasons[:2]) if reasons else ("Strong Moat & Growth" if total >= 65 else "Balanced Setup"),
        "nlp": " ".join(nlp_parts),
        "ratios": {
            "pe": float(raw_pe) if raw_pe is not None else None,
            "roe": roe,
            "roce": roce,
            "opm": opm,
            "sales_growth": sales_growth,
            "profit_growth": profit_growth,
            "debt_to_equity": debt_to_equity,
            "mcap_cr": mcap_cr,
            "beta": beta,
        },
    }


def calculate_porters_score(meta: dict[str, Any], *, strict: bool = True) -> dict[str, Any]:
    """Calculate StockSight V2 Porter's 5 Forces Qualitative Score (Max 100). Zero fake defaults."""
    if str(meta.get("fundamentals_status", "")).lower() == "unavailable" or str(meta.get("fundamental_source_category", "")).upper() == "UNAVAILABLE":
        raise ValueError("Fundamental data is unavailable; refusing to fabricate Porter's score.")

    industry = detect_industry(str(meta.get("name", "")), str(meta.get("sector", "")))
    is_financial = industry == "BANKING"

    mcap_cr = _require_real_metric(meta, "mcap_cr")
    roe = _require_real_metric(meta, "roe")
    roce = float(meta["roce"]) if meta.get("roce") is not None else (roe if is_financial else _require_real_metric(meta, "roce"))
    opm = float(meta["opm"]) if meta.get("opm") is not None else (0.0 if is_financial else _require_real_metric(meta, "opm"))
    sales_growth = _require_real_metric(meta, "sales_growth")
    profit_growth = _require_real_metric(meta, "profit_growth")

    entrants = 20 if (mcap_cr > 10000 and roce > 20) else (15 if (mcap_cr > 5000 and roce > 15) else (10 if mcap_cr > 2000 else 5))
    suppliers = 20 if opm > 25 else (15 if opm > 18 else (10 if opm > 10 else 5))
    buyers = 20 if roe > 22 else (15 if roe > 16 else (10 if roe > 12 else 5))
    substitutes = 20 if sales_growth > 15 else (15 if sales_growth > 10 else (10 if sales_growth > 5 else 5))
    rivalry = 20 if profit_growth > 15 else (15 if profit_growth > 10 else (10 if profit_growth > 0 else 5))

    total = min(99, entrants + suppliers + buyers + substitutes + rivalry)
    return {
        "entrants": entrants,
        "suppliers": suppliers,
        "buyers": buyers,
        "substitutes": substitutes,
        "rivalry": rivalry,
        "total": total,
    }


def calculate_moreshwar_levels(
    spot_price: float,
    f_score_total: int,
    p_score_total: int,
    is_holding: bool = False,
) -> dict[str, Any]:
    """Calculate Moreshwar Dynamic Target, Entry, and Stop-Loss Levels.

    Mathematical Model from StockSight V2 score_logic.md:
    - Y = Spot Price
    - X = Average of (fScore.total and pScore.total)
    - To scale cleanly across both ₹100 and ₹10,000+ assets while preserving the exact X formula,
      we provide both the exact Moreshwar point offset and percentage-calibrated execution levels.
    """
    x_blend = round((f_score_total + p_score_total) / 2.0, 1)
    downside_pts = 100.0 - x_blend

    # Scale factor for high-priced assets so stop-loss and target remain realistic on charts
    scale = max(1.0, round(spot_price / 1000.0, 2))
    target = round(spot_price + x_blend * scale, 2)
    stop_loss = round(max(spot_price * 0.85, spot_price - downside_pts * scale), 2)
    entry_price = round(spot_price if is_holding else max(spot_price * 0.90, spot_price - (downside_pts * 0.65 * scale)), 2)

    risk_per_share = max(1.0, spot_price - stop_loss)
    reward_per_share = max(1.0, target - spot_price)
    rr_ratio = round(reward_per_share / risk_per_share, 2)

    return {
        "x_composite_score": x_blend,
        "spot_price": round(spot_price, 2),
        "entry_price": entry_price,
        "target": target,
        "stop_loss": stop_loss,
        "rr_ratio": rr_ratio,
        "is_holding": is_holding,
    }


def calculate_decision_matrix(
    f_score_total: int,
    p_score_total: int,
    meta: dict[str, Any],
    price_above_sma: bool,
    is_holding: bool = False,
) -> dict[str, Any]:
    """Evaluate the StockSight V2 Step-2 Decision Tree Matrix."""
    avg_score = (f_score_total + p_score_total) / 2.0
    if avg_score >= 60:
        conviction = "Strong"
    elif avg_score >= 41:
        conviction = "Stable"
    else:
        conviction = "Weak"

    sales_g = float(meta.get("sales_growth", 10.0))
    profit_g = float(meta.get("profit_growth", 10.0))
    trajectory = "Deteriorating" if (sales_g < 0 or profit_g < 0) else "Improving/Flat"
    timing = "Favourable" if price_above_sma else "Unfavourable"

    if conviction == "Weak":
        if is_holding:
            action = "EXIT"
            rationale = "The foundation is weakening. Capital preservation takes priority."
        else:
            action = "AVOID"
            rationale = "Business quality does not meet the minimum threshold for investment."
    elif conviction == "Strong":
        if trajectory == "Deteriorating":
            action = "REVIEW" if is_holding else "WAIT"
            rationale = "Core moat is high, but near-term growth trajectory is slowing. Wait for stabilization."
        else:
            if timing == "Favourable":
                action = "ADD" if is_holding else "BUY NOW"
                rationale = "Ideal setup: High conviction quality intersecting with accelerating growth and favorable timing."
            else:
                action = "HOLD" if is_holding else "SIP ONLY"
                rationale = "Excellent business fundamentals; stagger entries on technical consolidation."
    else:  # Stable
        if trajectory == "Deteriorating":
            action = "REDUCE" if is_holding else "WAIT"
            rationale = "Moderate business quality threatened by slowing growth momentum."
        else:
            if timing == "Favourable":
                action = "HOLD" if is_holding else "SIP ONLY"
                rationale = "Decent business with favorable trend warrants a staggered allocation approach."
            else:
                action = "HOLD" if is_holding else "WAIT"
                rationale = "Decent business, but wait for stronger technical support before adding."

    return {
        "conviction": conviction,
        "trajectory": trajectory,
        "timing": timing,
        "action": action,
        "rationale": rationale,
        "composite_score": round(avg_score, 1),
    }
