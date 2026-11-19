"""Unit, Causal-Integrity, and Regression Tests for Moreshwar Mathematical Model 2.0."""

from __future__ import annotations

import unittest

from finsistant_v2.backtest_engine_v2 import run_moreshwar_v2_research_suite
from finsistant_v2.moreshwar_v2 import (
    MoreshwarV2Config,
    classify_market_regime,
    compute_atr,
    compute_confidence_adjustment,
    evaluate_moreshwar_v2,
)
from finsistant_v2.scoring_engine import calculate_moreshwar_levels


def _make_trend_candles(
    n: int = 60,
    start_px: float = 1000.0,
    daily_step: float = 2.0,
    hl_spread_pct: float = 0.018,
    vol: float = 500_000.0,
    start_day_offset: int = 1,
) -> list[dict]:
    out = []
    px = start_px
    for i in range(n):
        month = 1 + ((i + start_day_offset) // 28)
        day = 1 + ((i + start_day_offset) % 28)
        dt = f"2026-{month:02d}-{day:02d}"
        close = max(10.0, px)
        hi = round(close * (1.0 + hl_spread_pct * 0.55), 2)
        lo = round(close * (1.0 - hl_spread_pct * 0.45), 2)
        op = round(close * 0.998, 2)
        out.append({
            "date": dt,
            "open": op,
            "high": hi,
            "low": lo,
            "close": round(close, 2),
            "volume": vol,
        })
        px += daily_step
    return out


class TestMoreshwarV2(unittest.TestCase):
    def test_moreshwar_v1_preserved_as_baseline(self) -> None:
        m1_80 = calculate_moreshwar_levels(1000.0, 80, 80, is_holding=False)
        self.assertEqual(m1_80["entry_price"], 987.0)
        self.assertEqual(m1_80["target"], 1080.0)
        self.assertEqual(m1_80["stop_loss"], 980.0)
        # Documented Moreshwar 1.0 compression behavior on near-perfect scores
        m1_99 = calculate_moreshwar_levels(1000.0, 99, 99, is_holding=False)
        self.assertEqual(m1_99["stop_loss"], 999.0)
        # Documented scale=1.0 distortion on lower-priced stocks (Rs 100 -> -15% floor SL, +80% TP)
        m1_low = calculate_moreshwar_levels(100.0, 80, 80, is_holding=False)
        self.assertEqual(m1_low["stop_loss"], 85.0)
        self.assertEqual(m1_low["target"], 180.0)

    def test_confidence_never_reaches_certainty_and_penalizes_defects(self) -> None:
        cfg = MoreshwarV2Config()
        c_perf = compute_confidence_adjustment(
            fundamental_score=100,
            porters_score=100,
            missing_fields=[],
            fundamental_mode="point_in_time_live",
            price_above_sma20=True,
            tech_signal="BULLISH",
            regression_curve="Upward Curve",
            rsi_14=58.0,
            regime="BULLISH",
            cfg=cfg,
        )
        self.assertLessEqual(c_perf["confidence"], cfg.max_analytical_confidence)
        self.assertLess(c_perf["confidence"], 1.0)

        c_missing = compute_confidence_adjustment(
            fundamental_score=100,
            porters_score=100,
            missing_fields=["pe", "roe", "opm"],
            fundamental_mode="point_in_time_live",
            price_above_sma20=True,
            tech_signal="BULLISH",
            regression_curve="Upward Curve",
            rsi_14=58.0,
            regime="BULLISH",
            cfg=cfg,
        )
        self.assertLess(c_missing["confidence"], c_perf["confidence"])

        c_unavail = compute_confidence_adjustment(
            fundamental_score=None,
            porters_score=None,
            missing_fields=[],
            fundamental_mode="historical_unavailable",
            price_above_sma20=True,
            tech_signal="BULLISH",
            regression_curve="Upward Curve",
            rsi_14=58.0,
            regime="BULLISH",
            cfg=cfg,
        )
        self.assertLess(c_unavail["confidence"], c_perf["confidence"])
        self.assertEqual(c_unavail["confidence"], 0.58)

        c_bear = compute_confidence_adjustment(
            fundamental_score=85,
            porters_score=85,
            missing_fields=[],
            fundamental_mode="point_in_time_live",
            price_above_sma20=False,
            tech_signal="BEARISH",
            regression_curve="Downward Curve",
            rsi_14=78.0,
            regime="BEARISH",
            cfg=cfg,
        )
        self.assertLess(c_bear["confidence"], 0.45)

    def test_atr_stops_not_compressed_by_high_fundamental_score(self) -> None:
        stock_candles = _make_trend_candles(n=45, start_px=1000.0, daily_step=3.0, hl_spread_pct=0.024)
        bench_candles = _make_trend_candles(n=60, start_px=24000.0, daily_step=35.0, hl_spread_pct=0.009)
        spot = stock_candles[-1]["close"]

        res_low_f = evaluate_moreshwar_v2(
            symbol="TEST",
            candles=stock_candles,
            spot_price=spot,
            fundamental_score=55,
            porters_score=55,
            tech_signal="BULLISH",
            benchmark_candles=bench_candles,
            capital=200_000.0,
        )
        res_high_f = evaluate_moreshwar_v2(
            symbol="TEST",
            candles=stock_candles,
            spot_price=spot,
            fundamental_score=99,
            porters_score=99,
            tech_signal="BULLISH",
            benchmark_candles=bench_candles,
            capital=200_000.0,
        )
        self.assertEqual(res_low_f["stop_loss"], res_high_f["stop_loss"])
        self.assertGreater(res_high_f["stop_distance_pct"], 2.0)

        # Higher volatility asset must produce a wider ATR and wider stop distance
        volatile_candles = _make_trend_candles(n=45, start_px=1000.0, daily_step=3.0, hl_spread_pct=0.05)
        res_vol = evaluate_moreshwar_v2(
            symbol="VOLTEST",
            candles=volatile_candles,
            spot_price=volatile_candles[-1]["close"],
            fundamental_score=90,
            porters_score=90,
            tech_signal="BULLISH",
            benchmark_candles=bench_candles,
            capital=200_000.0,
        )
        self.assertGreater(res_vol["atr_value"], res_high_f["atr_value"])
        self.assertGreaterEqual(res_vol["stop_distance_pct"], res_high_f["stop_distance_pct"])

    def test_position_sizing_risk_budget_caps_and_liquidity(self) -> None:
        stock_candles = _make_trend_candles(n=45, start_px=500.0, daily_step=1.5, hl_spread_pct=0.02, vol=100_000.0)
        bench_candles = _make_trend_candles(n=60, start_px=24000.0, daily_step=35.0, hl_spread_pct=0.008)
        spot = stock_candles[-1]["close"]
        cfg = MoreshwarV2Config(risk_per_trade_pct=1.25, max_single_stock_pct=20.0)
        cap = 100_000.0

        res = evaluate_moreshwar_v2(
            symbol="SIZE",
            candles=stock_candles,
            spot_price=spot,
            fundamental_score=82,
            porters_score=80,
            tech_signal="BULLISH",
            benchmark_candles=bench_candles,
            capital=cap,
            cfg=cfg,
        )
        self.assertTrue(res["accepted"])
        self.assertIsInstance(res["position_qty"], int)
        risk_per_share = res["entry_price"] - res["stop_loss"]
        self.assertLessEqual(res["position_qty"] * risk_per_share, cap * (cfg.risk_per_trade_pct / 100.0) + 1e-6)
        self.assertLessEqual(res["exposure_pct"], cfg.max_single_stock_pct + 0.01)

        # Illiquid stock rejection
        illiquid = _make_trend_candles(n=45, start_px=500.0, daily_step=1.5, hl_spread_pct=0.02, vol=100.0)
        res_illq = evaluate_moreshwar_v2(
            symbol="ILLQ",
            candles=illiquid,
            spot_price=illiquid[-1]["close"],
            fundamental_score=90,
            porters_score=90,
            tech_signal="BULLISH",
            benchmark_candles=bench_candles,
            capital=cap,
            cfg=cfg,
        )
        self.assertFalse(res_illq["accepted"])
        self.assertTrue(any("turnover" in r.lower() for r in res_illq["rejection_reasons"]))

    def test_market_regime_causality_no_lookahead(self) -> None:
        bull = _make_trend_candles(n=60, start_px=24000.0, daily_step=30.0, hl_spread_pct=0.008)
        cutoff_dt = bull[39]["date"]
        r_before = classify_market_regime(bull, decision_date=cutoff_dt)

        # Corrupt all future benchmark bars violently
        mutated = [dict(c) for c in bull]
        for j in range(40, len(mutated)):
            mutated[j]["close"] = 1000.0
            mutated[j]["low"] = 500.0
            mutated[j]["high"] = 30000.0
        r_after = classify_market_regime(mutated, decision_date=cutoff_dt)

        self.assertEqual(r_before, r_after)
        self.assertEqual(r_before["regime"], "BULLISH")

        bear = _make_trend_candles(n=60, start_px=26000.0, daily_step=-45.0, hl_spread_pct=0.01)
        r_bear = classify_market_regime(bear)
        self.assertEqual(r_bear["regime"], "BEARISH")

        shock = _make_trend_candles(n=60, start_px=25000.0, daily_step=0.0, hl_spread_pct=0.045)
        r_shock = classify_market_regime(shock)
        self.assertEqual(r_shock["regime"], "HIGH_VOLATILITY")

    def test_12_explainability_fields_and_short_history_graceful(self) -> None:
        short = _make_trend_candles(n=8, start_px=1000.0)
        bad = evaluate_moreshwar_v2(symbol="SHORT", candles=short, spot_price=1000.0)
        self.assertFalse(bad["accepted"])
        self.assertEqual(bad["position_qty"], 0)
        self.assertGreater(len(bad["rejection_reasons"]), 0)

        full = _make_trend_candles(n=45, start_px=1000.0, daily_step=2.5, hl_spread_pct=0.02)
        bench = _make_trend_candles(n=60, start_px=24000.0, daily_step=30.0, hl_spread_pct=0.008)
        ok = evaluate_moreshwar_v2(
            symbol="FULL",
            candles=full,
            spot_price=full[-1]["close"],
            fundamental_score=78,
            porters_score=76,
            tech_signal="BULLISH",
            benchmark_candles=bench,
            capital=250_000.0,
        )
        required_keys = [
            "fundamental_score",
            "confidence",
            "atr_value",
            "atr_multiplier",
            "market_regime",
            "entry_price",
            "stop_loss",
            "target",
            "rr_ratio",
            "position_qty",
            "max_capital_exposure",
            "rejection_reasons",
            "accept_reasons",
        ]
        for k in required_keys:
            self.assertIn(k, ok)

    def test_research_suite_shadow_flag_and_edge_safety(self) -> None:
        candles = _make_trend_candles(n=100, start_px=1000.0, daily_step=2.0, hl_spread_pct=0.02)
        bench = _make_trend_candles(n=100, start_px=24000.0, daily_step=25.0, hl_spread_pct=0.009)
        suite = run_moreshwar_v2_research_suite(
            universe_history={
                "TEST1": {
                    "symbol": "TEST1",
                    "currency": "INR",
                    "sector": "IT",
                    "candles": candles,
                    "fundamental_score": {"total": 75, "ratios": {"sales_growth": 14, "profit_growth": 16}},
                    "porters_score": {"total": 72},
                    "missing_fields": [],
                }
            },
            benchmark_data={"candles": bench},
            run_sensitivity=False,
        )
        self.assertFalse(suite["promoted_to_production"])
        self.assertGreaterEqual(len(suite["models_comparison"]), 8)

    def test_server_shadow_non_promotion_regression(self) -> None:
        import json
        from pathlib import Path
        import server

        m1 = calculate_moreshwar_levels(1500.0, 80, 75, is_holding=False)
        self.assertEqual(m1["entry_price"], round(1500.0 - ((100.0 - 77.5) * 0.65 * 1.5), 2))
        scan_path = Path(server.BASE_DIR) / "live_scan.json"
        if scan_path.exists():
            payload = json.loads(scan_path.read_text(encoding="utf-8"))
            v2_res = payload.get("moreshwar_v2_research")
            if v2_res:
                self.assertFalse(v2_res.get("promoted_to_production", True))


if __name__ == "__main__":
    unittest.main()

