"""Quantitative Regression & Technical Engine for Finsistant Version 2.

Combines:
- Finsistant Degree-2 Polynomial Regression + 30-Day Forward Projection + StdDev Channels
- Multi-Agent Technical Indicators (RSI-14, MACD, 20D SMA, EMA-12/26, Bollinger Bands)
- Rich Indian (NSE) & Global Equity Universe
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import math
import random
from typing import Any

from .scoring_engine import (
    calculate_decision_matrix,
    calculate_fundamental_score,
    calculate_moreshwar_levels,
    calculate_porters_score,
)

ASSET_UNIVERSE: dict[str, dict[str, Any]] = {
    "RELIANCE": {
        "name": "Reliance Industries Ltd",
        "exchange": "NSE",
        "sector": "Energy, Telecom & Retail Conglomerate",
        "asset_type": "STOCK",
        "base_price": 2985.00,
        "volatility": 0.016,
        "drift": 0.0021,
        "pe": 26.4,
        "roe": 16.8,
        "roce": 18.4,
        "opm": 22.5,
        "sales_growth": 16.2,
        "profit_growth": 18.5,
        "mcap_cr": 2015000.0,
        "beta": 0.98,
        "ret_1y": 24.5,
        "debt_to_equity": 0.34,
        "setup_type": "Institutional Breakout",
        "default_qty": 15,
        "default_avg": 2790.00,
    },
    "TCS": {
        "name": "Tata Consultancy Services Ltd",
        "exchange": "NSE",
        "sector": "IT Services & Enterprise AI Consulting",
        "asset_type": "STOCK",
        "base_price": 4120.00,
        "volatility": 0.014,
        "drift": 0.0018,
        "pe": 29.8,
        "roe": 48.5,
        "roce": 58.2,
        "opm": 26.4,
        "sales_growth": 12.4,
        "profit_growth": 15.8,
        "mcap_cr": 1490000.0,
        "beta": 0.82,
        "ret_1y": 19.2,
        "debt_to_equity": 0.04,
        "setup_type": "Moat Accumulation",
        "default_qty": 10,
        "default_avg": 3890.00,
    },
    "HDFCBANK": {
        "name": "HDFC Bank Ltd",
        "exchange": "NSE",
        "sector": "Private Sector Banking & Financial Services",
        "asset_type": "STOCK",
        "base_price": 1745.00,
        "volatility": 0.015,
        "drift": 0.0019,
        "pe": 18.6,
        "roe": 17.4,
        "roce": 16.2,
        "opm": 28.0,
        "sales_growth": 18.5,
        "profit_growth": 19.4,
        "mcap_cr": 1325000.0,
        "beta": 0.94,
        "ret_1y": 16.8,
        "debt_to_equity": 0.45,
        "setup_type": "High Quality Value",
        "default_qty": 25,
        "default_avg": 1610.00,
    },
    "ICICIBANK": {
        "name": "ICICI Bank Ltd",
        "exchange": "NSE",
        "sector": "Private Sector Banking & Credit",
        "asset_type": "STOCK",
        "base_price": 1288.00,
        "volatility": 0.015,
        "drift": 0.0024,
        "pe": 19.2,
        "roe": 18.9,
        "roce": 17.5,
        "opm": 31.0,
        "sales_growth": 19.8,
        "profit_growth": 22.4,
        "mcap_cr": 908000.0,
        "beta": 0.96,
        "ret_1y": 29.4,
        "debt_to_equity": 0.40,
        "setup_type": "Momentum Leader",
        "default_qty": 30,
        "default_avg": 1165.00,
    },
    "INFY": {
        "name": "Infosys Ltd",
        "exchange": "NSE",
        "sector": "Digital Cloud & Enterprise AI Tech",
        "asset_type": "STOCK",
        "base_price": 1865.00,
        "volatility": 0.016,
        "drift": 0.0017,
        "pe": 27.1,
        "roe": 31.8,
        "roce": 39.4,
        "opm": 23.8,
        "sales_growth": 11.2,
        "profit_growth": 14.6,
        "mcap_cr": 774000.0,
        "beta": 0.89,
        "ret_1y": 21.0,
        "debt_to_equity": 0.06,
        "setup_type": "Channel Breakout",
        "default_qty": 0,
        "default_avg": 0.0,
    },
    "TATAMOTORS": {
        "name": "Tata Motors Ltd",
        "exchange": "NSE",
        "sector": "Automotive, EV & Commercial Mobility",
        "asset_type": "STOCK",
        "base_price": 945.00,
        "volatility": 0.022,
        "drift": 0.0022,
        "pe": 11.8,
        "roe": 24.6,
        "roce": 21.8,
        "opm": 15.6,
        "sales_growth": 17.4,
        "profit_growth": 26.0,
        "mcap_cr": 348000.0,
        "beta": 1.18,
        "ret_1y": 42.5,
        "debt_to_equity": 0.52,
        "setup_type": "Deep Value Turnaround",
        "default_qty": 0,
        "default_avg": 0.0,
    },
    "NVDA": {
        "name": "NVIDIA Corporation",
        "exchange": "NASDAQ",
        "sector": "AI Compute & Accelerated Semiconductors",
        "asset_type": "STOCK",
        "base_price": 142.80,
        "volatility": 0.024,
        "drift": 0.0029,
        "pe": 44.2,
        "roe": 68.0,
        "roce": 62.0,
        "opm": 58.4,
        "sales_growth": 78.0,
        "profit_growth": 89.0,
        "mcap_cr": 2850000.0,
        "beta": 1.24,
        "ret_1y": 84.0,
        "debt_to_equity": 0.19,
        "setup_type": "AI Momentum Continuation",
        "default_qty": 0,
        "default_avg": 0.0,
    },
    "ZEEL": {
        "name": "Zee Entertainment Enterprises",
        "exchange": "NSE",
        "sector": "Media & Broadcasting",
        "asset_type": "STOCK",
        "base_price": 134.50,
        "volatility": 0.032,
        "drift": -0.0022,
        "pe": 48.0,
        "roe": 4.2,
        "roce": 5.8,
        "opm": 8.4,
        "sales_growth": -6.5,
        "profit_growth": -18.0,
        "mcap_cr": 12900.0,
        "beta": 1.44,
        "ret_1y": -32.0,
        "debt_to_equity": 0.85,
        "setup_type": "Risk Breakdown Warning",
        "default_qty": 40,
        "default_avg": 158.00,
    },
}


def _fit_degree2_polynomial(y_vals: list[float], future_steps: int = 15) -> tuple[list[float], list[float], float, float]:
    """Fit a degree-2 polynomial regression y = a*x^2 + b*x + c using pure numerical linear algebra."""
    n = len(y_vals)
    if n < 3:
        return y_vals, y_vals, 0.0, 0.0

    xs = [float(i) for i in range(n)]
    s0 = float(n)
    s1 = sum(xs)
    s2 = sum(x * x for x in xs)
    s3 = sum(x**3 for x in xs)
    s4 = sum(x**4 for x in xs)

    sy = sum(y_vals)
    sxy = sum(x * y for x, y in zip(xs, y_vals))
    sx2y = sum((x * x) * y for x, y in zip(xs, y_vals))

    # Solve 3x3 normal equations via Gaussian elimination
    mat = [
        [s0, s1, s2, sy],
        [s1, s2, s3, sxy],
        [s2, s3, s4, sx2y],
    ]
    for i in range(3):
        pivot = mat[i][i] if abs(mat[i][i]) > 1e-12 else 1e-12
        for j in range(i, 4):
            mat[i][j] /= pivot
        for k in range(3):
            if k != i:
                factor = mat[k][i]
                for j in range(i, 4):
                    mat[k][j] -= factor * mat[i][j]

    c, b, a = mat[0][3], mat[1][3], mat[2][3]
    fitted = [round(a * (x * x) + b * x + c, 2) for x in xs]
    residuals = [y - yf for y, yf in zip(y_vals, fitted)]
    std_dev = math.sqrt(sum(r * r for r in residuals) / max(1, n))

    future = [
        round(a * ((n + step) ** 2) + b * (n + step) + c, 2)
        for step in range(1, future_steps + 1)
    ]
    return fitted, future, round(std_dev, 2), a


def generate_ohlcv_and_regression(symbol: str, periods: int = 45) -> dict[str, Any]:
    """Generate OHLCV candles, technical indicators, and Finsistant degree-2 regression channel."""
    symbol = symbol.upper().strip()
    meta = ASSET_UNIVERSE.get(symbol, ASSET_UNIVERSE["RELIANCE"])
    seed = int(hashlib.sha256(symbol.encode("utf-8")).hexdigest()[:8], 16)
    rng = random.Random(seed)

    base = float(meta["base_price"])
    vol = float(meta["volatility"])
    drift = float(meta["drift"])

    today = datetime.now(timezone.utc).date()
    candles: list[dict[str, Any]] = []
    cur_close = base * (0.91 if drift >= 0 else 1.09)

    for i in range(periods):
        wave = 0.005 * math.sin((i / max(1, periods - 1)) * 3.0 * math.pi)
        impulse = drift * 1.4 if i >= periods - 10 else drift
        shock = rng.gauss(impulse, vol) + wave
        open_p = cur_close
        cur_close = open_p * math.exp(shock)
        spread = abs(rng.gauss(0.010, 0.004))
        high_p = max(open_p, cur_close) * (1.0 + spread)
        low_p = min(open_p, cur_close) * (1.0 - spread)
        dt = (today - timedelta(days=periods - 1 - i)).isoformat()
        candles.append(
            {
                "date": dt,
                "open": round(open_p, 2),
                "high": round(high_p, 2),
                "low": round(low_p, 2),
                "close": round(cur_close, 2),
                "volume": rng.randint(2_500_000, 19_000_000),
            }
        )

    closes = [c["close"] for c in candles]
    fitted, future_pred, std_dev, curve_a = _fit_degree2_polynomial(closes, future_steps=12)

    ema12 = closes[0]
    ema26 = closes[0]
    macd_sig = 0.0
    k12, k26, k9 = 2.0 / 13.0, 2.0 / 27.0, 2.0 / 10.0

    for idx, row in enumerate(candles):
        win = closes[max(0, idx - 19) : idx + 1]
        sma20 = sum(win) / len(win)
        var = sum((x - sma20) ** 2 for x in win) / len(win)
        std20 = math.sqrt(var)

        c = row["close"]
        ema12 = c * k12 + ema12 * (1.0 - k12)
        ema26 = c * k26 + ema26 * (1.0 - k26)
        macd = ema12 - ema26
        macd_sig = macd * k9 + macd_sig * (1.0 - k9)

        rsi_slice = closes[max(0, idx - 14) : idx + 1]
        gains = [max(0.0, rsi_slice[j] - rsi_slice[j - 1]) for j in range(1, len(rsi_slice))]
        losses = [max(0.0, rsi_slice[j - 1] - rsi_slice[j]) for j in range(1, len(rsi_slice))]
        avg_g = (sum(gains) / len(gains)) if gains else 0.01
        avg_l = (sum(losses) / len(losses)) if losses else 0.01
        rsi14 = max(18.0, min(86.0, 100.0 - (100.0 / (1.0 + avg_g / max(avg_l, 1e-9)))))

        row["sma_20"] = round(sma20, 2)
        row["ema_12"] = round(ema12, 2)
        row["ema_26"] = round(ema26, 2)
        row["macd"] = round(macd, 3)
        row["macd_signal"] = round(macd_sig, 3)
        row["rsi_14"] = round(rsi14, 2)
        row["bb_upper"] = round(sma20 + 2.0 * std20, 2)
        row["bb_lower"] = round(sma20 - 2.0 * std20, 2)
        row["reg_value"] = fitted[idx]
        row["reg_upper"] = round(fitted[idx] + std_dev, 2)
        row["reg_lower"] = round(fitted[idx] - std_dev, 2)

    return {
        "candles": candles[-36:],
        "projected_30d_target": future_pred[-1] if future_pred else closes[-1],
        "regression_std_dev": std_dev,
        "regression_curve": "Upward Curve" if curve_a >= -1e-4 and future_pred[-1] >= closes[-1] else "Downward Curve",
        "future_projection": future_pred,
    }


def get_asset_full_analysis(symbol: str, is_holding: bool = False) -> dict[str, Any]:
    """Compute unified StockSight V2 + Finsistant + Multi-Agent snapshot for an asset."""
    symbol = symbol.upper().strip()
    meta = ASSET_UNIVERSE.get(symbol)
    if not meta:
        # Dynamic synthesis for custom symbols added by user
        meta = {
            "name": f"{symbol} Corp Ltd",
            "exchange": "NSE",
            "sector": "Diversified Industrials & Technology",
            "asset_type": "STOCK",
            "base_price": 1250.0,
            "volatility": 0.018,
            "drift": 0.0018,
            "pe": 22.5,
            "roe": 19.2,
            "roce": 21.0,
            "opm": 20.4,
            "sales_growth": 15.4,
            "profit_growth": 17.8,
            "mcap_cr": 42000.0,
            "beta": 1.02,
            "ret_1y": 22.0,
            "debt_to_equity": 0.22,
            "setup_type": "Breakout Setup",
            "default_qty": 0,
            "default_avg": 0.0,
        }
        ASSET_UNIVERSE[symbol] = meta

    quant = generate_ohlcv_and_regression(symbol)
    candles = quant["candles"]
    latest = candles[-1]
    prev = candles[-2]
    spot = float(latest["close"])
    change_pct = round(((spot - float(prev["close"])) / float(prev["close"])) * 100.0, 2)

    f_score = calculate_fundamental_score(meta)
    p_score = calculate_porters_score(meta)
    moreshwar = calculate_moreshwar_levels(
        spot_price=spot,
        f_score_total=f_score["total"],
        p_score_total=p_score["total"],
        is_holding=is_holding,
    )
    decision = calculate_decision_matrix(
        f_score_total=f_score["total"],
        p_score_total=p_score["total"],
        meta=meta,
        price_above_sma=spot >= float(latest["sma_20"]),
        is_holding=is_holding,
    )

    return {
        "symbol": symbol,
        "name": meta["name"],
        "exchange": meta.get("exchange", "NSE"),
        "sector": meta["sector"],
        "asset_type": meta.get("asset_type", "STOCK"),
        "setup_type": meta.get("setup_type", "Breakout"),
        "price": spot,
        "change_pct": change_pct,
        "rsi_14": float(latest["rsi_14"]),
        "macd": float(latest["macd"]),
        "macd_signal": float(latest["macd_signal"]),
        "sma_20": float(latest["sma_20"]),
        "ema_12": float(latest["ema_12"]),
        "ema_26": float(latest["ema_26"]),
        "bb_upper": float(latest["bb_upper"]),
        "bb_lower": float(latest["bb_lower"]),
        "projected_30d_target": quant["projected_30d_target"],
        "regression_std_dev": quant["regression_std_dev"],
        "regression_curve": quant["regression_curve"],
        "future_projection": quant["future_projection"],
        "fundamental_score": f_score,
        "porters_score": p_score,
        "moreshwar_levels": moreshwar,
        "decision_matrix": decision,
        "candles": candles,
    }
