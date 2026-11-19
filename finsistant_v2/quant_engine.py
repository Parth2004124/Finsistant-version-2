"""Quantitative Regression & Technical Engine for Finsistant Version 2.

Zero simulation policy:
- All OHLCV candles, spot prices, 1D changes, and fundamental ratios come from
  live market data (finsistant_v2.live_data).
- Degree-2 Polynomial Regression (y = ax^2 + bx + c), RSI-14, MACD, 20D SMA,
  EMA-12/26, and Bollinger Bands are computed strictly on real OHLCV history.
"""

from __future__ import annotations

import math
from typing import Any

from .live_data import DEFAULT_SYMBOL_CONFIG, fetch_live_market_snapshot
from .scoring_engine import (
    calculate_decision_matrix,
    calculate_fundamental_score,
    calculate_moreshwar_levels,
    calculate_porters_score,
)

# Watchlist symbol registry (no synthetic prices or fake default holdings)
ASSET_UNIVERSE: dict[str, dict[str, str]] = dict(DEFAULT_SYMBOL_CONFIG)


def _fit_degree2_polynomial(y_vals: list[float], future_steps: int = 12) -> tuple[list[float], list[float], float, float]:
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


def compute_ohlcv_and_regression_from_candles(raw_candles: list[dict[str, Any]], periods: int = 45) -> dict[str, Any]:
    """Compute technical indicators and Degree-2 Polynomial Regression strictly on real OHLCV bars."""
    candles = [dict(c) for c in raw_candles[-periods:]]
    if len(candles) < 10:
        raise RuntimeError("Insufficient real OHLCV candles to compute indicators and regression.")

    closes = [float(c["close"]) for c in candles]
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

        c = float(row["close"])
        ema12 = c * k12 + ema12 * (1.0 - k12)
        ema26 = c * k26 + ema26 * (1.0 - k26)
        macd = ema12 - ema26
        macd_sig = macd * k9 + macd_sig * (1.0 - k9)

        rsi_slice = closes[max(0, idx - 14) : idx + 1]
        gains = [max(0.0, rsi_slice[j] - rsi_slice[j - 1]) for j in range(1, len(rsi_slice))]
        losses = [max(0.0, rsi_slice[j - 1] - rsi_slice[j]) for j in range(1, len(rsi_slice))]
        avg_g = (sum(gains) / len(gains)) if gains else 0.0
        avg_l = (sum(losses) / len(losses)) if losses else 0.0
        if avg_l == 0 and avg_g == 0:
            rsi14 = 50.0
        elif avg_l == 0:
            rsi14 = 85.0
        else:
            rsi14 = max(5.0, min(95.0, 100.0 - (100.0 / (1.0 + avg_g / avg_l))))

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


def generate_ohlcv_and_regression(symbol: str, periods: int = 45, force_refresh: bool = False) -> dict[str, Any]:
    """Fetch live OHLCV candles for symbol and compute technical indicators + degree-2 regression."""
    live = fetch_live_market_snapshot(symbol, force_refresh=force_refresh)
    return compute_ohlcv_and_regression_from_candles(live["candles"], periods=periods)


def get_asset_full_analysis(symbol: str, is_holding: bool = False, force_refresh: bool = False) -> dict[str, Any]:
    """Compute unified StockSight V2 + Finsistant + Multi-Agent snapshot strictly from live data."""
    sym = symbol.upper().strip()
    live = fetch_live_market_snapshot(sym, force_refresh=force_refresh)
    quant = compute_ohlcv_and_regression_from_candles(live["candles"], periods=45)
    candles = quant["candles"]
    latest = candles[-1]

    spot = float(live["price"])
    change_pct = float(live["change_pct"])

    f_score = calculate_fundamental_score(live)
    p_score = calculate_porters_score(live)
    moreshwar = calculate_moreshwar_levels(
        spot_price=spot,
        f_score_total=f_score["total"],
        p_score_total=p_score["total"],
        is_holding=is_holding,
    )
    decision = calculate_decision_matrix(
        f_score_total=f_score["total"],
        p_score_total=p_score["total"],
        meta=live,
        price_above_sma=spot >= float(latest["sma_20"]),
        is_holding=is_holding,
    )

    return {
        "symbol": sym,
        "yahoo_symbol": live["yahoo_symbol"],
        "name": live["name"],
        "exchange": live["exchange"],
        "currency": live["currency"],
        "sector": live["sector"],
        "asset_type": live["asset_type"],
        "setup_type": live["setup_type"],
        "price": spot,
        "last_candle_date": live.get("last_candle_date", candles[-1]["date"]),
        "last_candle_close": live.get("last_candle_close", float(latest["close"])),
        "history_bars_count": len(live.get("candles") or []),
        "change_pct": change_pct,
        "fetched_at": live["fetched_at"],
        "market_timestamp": live.get("market_timestamp") or f"{candles[-1]['date']}T10:00:00+00:00",
        "market_timestamp_kind": live.get("market_timestamp_kind", "end_of_day_bar"),
        "missing_fields": list(live.get("missing_fields") or []),
        "proxy_fields": dict(live.get("proxy_fields") or {}),
        "cache_hit": bool(live.get("cache_hit", False)),
        "cache_age_seconds": float(live.get("cache_age_seconds", 0.0)),
        "price_source_category": live.get("price_source_category", "YAHOO_LIVE_FEED"),
        "ohlcv_source_category": live.get("ohlcv_source_category", "HISTORICAL_MARKET_DATA"),
        "fundamental_source_category": live.get("fundamental_source_category", "YAHOO_LIVE_FEED"),
        "secondary_spot_price": live.get("secondary_spot_price"),
        "secondary_source": live.get("secondary_source"),
        "nse_raw_ist_timestamp": live.get("nse_raw_ist_timestamp"),
        "nse_corporate_adjustments": list(live.get("nse_corporate_adjustments") or []),
        "data_source": live["data_source"],
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
