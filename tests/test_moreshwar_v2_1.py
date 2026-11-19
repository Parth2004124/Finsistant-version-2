"""Unit, Causal-Integrity, Cost-Realism, Statistical, and Regression Tests for Moreshwar 2.1."""

from __future__ import annotations

import json
from pathlib import Path
import unittest

from finsistant_v2.backtest_engine_v2_1 import (
    _compute_statistical_significance,
    precompute_causal_features,
    run_monte_carlo_suite,
    simulate_portfolio_v2_1,
)
from finsistant_v2.moreshwar_v2 import evaluate_moreshwar_v2
from finsistant_v2.moreshwar_v2_1 import (
    MoreshwarV21Config,
    classify_market_regime_6,
    compute_indian_delivery_costs,
    evaluate_moreshwar_v2_1,
)
from finsistant_v2.scoring_engine import calculate_moreshwar_levels

ROOT_DIR = Path(__file__).resolve().parent.parent


def _make_candles(
    n: int = 80,
    start_px: float = 1000.0,
    step: float = 2.0,
    hl_spread: float = 0.015,
    vol: float = 800_000.0,
) -> list[dict]:
    out = []
    px = start_px
    for i in range(n):
        yr = 2023 + (i // 280)
        rem = i % 280
        mo = 1 + (rem // 28)
        dy = 1 + (rem % 28)
        dt = f"{yr:04d}-{mo:02d}-{dy:02d}"
        c = max(5.0, px)
        out.append({
            "date": dt,
            "open": round(c * 0.999, 2),
            "high": round(c * (1.0 + hl_spread * 0.55), 2),
            "low": round(c * (1.0 - hl_spread * 0.45), 2),
            "close": round(c, 2),
            "volume": vol,
        })
        px += step
    return out


class TestMoreshwarV21(unittest.TestCase):
    def test_m1_and_m2_baseline_behavior_intact(self) -> None:
        m1 = calculate_moreshwar_levels(1000.0, 80, 80, is_holding=False)
        self.assertEqual((m1["entry_price"], m1["target"], m1["stop_loss"]), (987.0, 1080.0, 980.0))

        stock = _make_candles(60, start_px=500.0, step=1.5)
        bench = _make_candles(70, start_px=24000.0, step=35.0, hl_spread=0.008)
        v2 = evaluate_moreshwar_v2(
            symbol="KEEPV2",
            candles=stock,
            spot_price=stock[-1]["close"],
            fundamental_score=80,
            porters_score=78,
            tech_signal="BULLISH",
            benchmark_candles=bench,
            capital=200_000.0,
        )
        self.assertEqual(v2["model_version"], "Moreshwar 2.0")
        self.assertTrue(v2["accepted"])

    def test_6state_regime_classifier_and_strict_causality(self) -> None:
        bench = _make_candles(90, start_px=22000.0, step=30.0, hl_spread=0.008)
        cutoff = bench[65]["date"]
        r_before = classify_market_regime_6(bench, decision_date=cutoff)
        mutated = [dict(b) for b in bench]
        for j in range(66, len(mutated)):
            mutated[j]["close"] = 500.0
            mutated[j]["high"] = 40000.0
            mutated[j]["low"] = 100.0
        r_after = classify_market_regime_6(mutated, decision_date=cutoff)
        self.assertEqual(r_before, r_after)
        self.assertIn(r_before["regime"], ("STRONG_BULLISH", "WEAK_BULLISH"))

        bear = _make_candles(90, start_px=25000.0, step=-40.0, hl_spread=0.009)
        self.assertIn(classify_market_regime_6(bear)["regime"], ("STRONG_BEARISH", "WEAK_BEARISH"))

        hv = _make_candles(90, start_px=24000.0, step=0.0, hl_spread=0.05)
        self.assertEqual(classify_market_regime_6(hv)["regime"], "HIGH_VOLATILITY")

    def test_precompute_causal_features_zero_lookahead(self) -> None:
        bars = _make_candles(90, start_px=400.0, step=1.2)
        bench = _make_candles(90, start_px=22000.0, step=25.0)
        sym_hist = {"TEST": {"candles": bars, "sector": "IT"}}
        full_feats = precompute_causal_features(sym_hist, bench)

        cutoff_i = 68
        cutoff_dt = bars[cutoff_i]["date"]
        trunc_feats = precompute_causal_features({"TEST": {"candles": bars[: cutoff_i + 1], "sector": "IT"}}, bench[: cutoff_i + 1])

        self.assertEqual(full_feats["features"]["TEST"][cutoff_dt], trunc_feats["features"]["TEST"][cutoff_dt])

    def test_indian_equity_delivery_itemized_costs(self) -> None:
        buy_c = compute_indian_delivery_costs(side="BUY", notional=50_000.0, currency="INR", slippage_bps=7.5)
        sell_c = compute_indian_delivery_costs(side="SELL", notional=50_000.0, currency="INR", slippage_bps=7.5)
        self.assertEqual(buy_c["dp_charge"], 0.0)
        self.assertEqual(sell_c["dp_charge"], 15.93)
        self.assertGreater(buy_c["stamp_duty"], 0.0)
        self.assertEqual(sell_c["stamp_duty"], 0.0)
        self.assertAlmostEqual(buy_c["stt"], 50.0, places=2)
        self.assertAlmostEqual(sell_c["stt"], 50.0, places=2)
        self.assertAlmostEqual(buy_c["slippage_cost"], 37.5, places=2)

        usd_c = compute_indian_delivery_costs(side="BUY", notional=1000.0, currency="USD", slippage_bps=5.0)
        self.assertEqual(usd_c["stt"], 0.0)
        self.assertEqual(usd_c["dp_charge"], 0.0)
        self.assertAlmostEqual(usd_c["slippage_cost"], 0.5, places=2)

    def test_portfolio_constraints_sector_corr_and_dust_order_rejection(self) -> None:
        stock = _make_candles(65, start_px=600.0, step=1.8, vol=500_000.0)
        bench = _make_candles(75, start_px=23000.0, step=35.0, hl_spread=0.008)
        spot = stock[-1]["close"]
        cfg = MoreshwarV21Config(max_open_positions=6, max_sector_positions=2, max_pairwise_corr=0.78)

        # Max sector saturation (2 IT positions already open)
        open_it = [
            {"symbol": "TCS", "sector": "IT", "entry_price": 3000.0, "sl": 2850.0, "qty": 2, "recent_closes": [b["close"] for b in stock]},
            {"symbol": "INFY", "sector": "IT", "entry_price": 1500.0, "sl": 1420.0, "qty": 3, "recent_closes": [b["close"] for b in stock]},
        ]
        res_sec = evaluate_moreshwar_v2_1(
            symbol="WIPRO", sector="IT", candles=stock, spot_price=spot,
            tech_signal="BULLISH", benchmark_candles=bench, capital=200_000.0,
            open_positions_context=open_it, cfg=cfg,
        )
        self.assertFalse(res_sec["accepted"])
        self.assertEqual(res_sec["rejection_category"], "PORTFOLIO_CAP_OR_CORR")

        # Dust order rejection on tiny capital where fixed DP fee would exceed notional
        res_dust = evaluate_moreshwar_v2_1(
            symbol="DUST", sector="Metals", candles=stock, spot_price=spot,
            tech_signal="BULLISH", benchmark_candles=bench, capital=500.0,
            open_positions_context=[], cfg=cfg,
        )
        self.assertFalse(res_dust["accepted"])
        self.assertEqual(res_dust["position_qty"], 0)

    def test_execution_realism_gap_through_stop_and_zero_volume(self) -> None:
        bench = _make_candles(65, start_px=23000.0, step=35.0, hl_spread=0.008)
        dates = [b["date"] for b in bench]
        # Construct asset in uptrend that triggers entry signal then gaps violently down below SL
        candles = []
        for idx, dt in enumerate(dates):
            if idx < 45:
                px = 100.0 + idx * 1.5
                candles.append({"date": dt, "open": px - 0.5, "high": px + 1.0, "low": px - 1.0, "close": px, "volume": 500_000})
            elif idx == 45:
                # Zero-volume halted day right after signal -> cannot fill
                candles.append({"date": dt, "open": 168.0, "high": 168.0, "low": 168.0, "close": 168.0, "volume": 0})
            elif idx == 48:
                # Violent gap down opening well below stop loss (~158)
                candles.append({"date": dt, "open": 120.0, "high": 125.0, "low": 118.0, "close": 121.0, "volume": 900_000})
            else:
                px = 170.0 + (idx - 45) * 0.4
                candles.append({"date": dt, "open": px, "high": px + 1.2, "low": px - 0.8, "close": px, "volume": 500_000})

        hist = {"GAPT": {"candles": candles, "sector": "Auto", "cap_tier": "Mid Cap", "currency": "INR", "missing_fields": []}}
        precomp = precompute_causal_features(hist, bench)
        sim = simulate_portfolio_v2_1(
            hist,
            bench,
            precomp,
            model_name="M21_TEST",
            model_family="M2_1",
            cfg_v21=MoreshwarV21Config(),
        )
        ex = sim["metrics"]["execution_realism"]
        self.assertGreaterEqual(ex["unfillable_zero_vol"] + ex["gap_through_stops"], 0)
        for t in sim["trades"]:
            if t["gap_through_stop"]:
                # Gap stop fill must be around open (~120 * slip), never ideal intraday SL
                self.assertLess(t["exit_price"], 130.0)

    def test_monte_carlo_and_statistical_significance_bounds(self) -> None:
        trades = [{"pnl_amount": 400.0 if i % 2 == 0 else -300.0, "net_return_pct": 2.0 if i % 2 == 0 else -1.5} for i in range(60)]
        daily_rets = [0.002 if i % 3 != 0 else -0.0025 for i in range(250)]
        mc = run_monte_carlo_suite(trades, daily_rets, n_sims=1000, block_len=10)

        tb = mc["trade_bootstrap"]
        bb = mc["daily_block_bootstrap"]
        self.assertGreaterEqual(tb["simulations"], 1000)
        self.assertGreaterEqual(bb["simulations"], 1000)
        self.assertLessEqual(tb["p05_return_pct"], tb["median_return_pct"])
        self.assertLessEqual(tb["median_return_pct"], tb["p95_return_pct"])
        self.assertLessEqual(bb["p05_return_pct"], bb["median_return_pct"])
        self.assertLessEqual(bb["median_return_pct"], bb["p95_return_pct"])
        self.assertTrue(0.0 <= tb["prob_ruin_25pct_dd"] <= 100.0)
        self.assertTrue(0.0 <= bb["prob_ruin_50pct_dd"] <= 100.0)

        sig = _compute_statistical_significance(trades)
        low_ci, hi_ci = sig["win_rate_ci95"]
        self.assertTrue(0.0 <= low_ci <= sig["win_rate_pct"] <= hi_ci <= 100.0)
        self.assertTrue(0.0 <= sig["p_value_vs_zero"] <= 1.0)

    def test_persisted_experiment_report_integrity_and_non_promotion(self) -> None:
        rep_path = ROOT_DIR / "data" / "moreshwar_v2_1_experiment_report.json"
        self.assertTrue(rep_path.exists(), "Moreshwar 2.1 report must exist in data/")
        report = json.loads(rep_path.read_text(encoding="utf-8"))

        self.assertIs(report["promoted_to_production"], False)
        self.assertGreaterEqual(report["dataset_summary"]["trading_days"], 1000)
        self.assertGreaterEqual(report["dataset_summary"]["instruments_count"], 35)
        self.assertEqual(report["monte_carlo"]["moreshwar_2_1"]["trade_bootstrap"]["simulations"], 1200)
        self.assertEqual(report["monte_carlo"]["moreshwar_2_1"]["daily_block_bootstrap"]["simulations"], 1200)
        self.assertEqual(report["monte_carlo"]["moreshwar_1_0"]["trade_bootstrap"]["simulations"], 1200)
        self.assertGreaterEqual(len(report["walk_forward_validation"]["folds"]), 4)
        self.assertIn("regimes_6", report["models"]["Moreshwar 2.1"])
        self.assertEqual(len(report["models"]["Moreshwar 2.1"]["regimes_6"]), 6)
        self.assertLessEqual(report["models"]["Moreshwar 2.1"]["portfolio_risk"]["max_simultaneous_positions"], 6)
        self.assertIn("selectivity_counterfactuals", report["strategy_investigation"])
        self.assertIn("classification", report["final_decision"])


if __name__ == "__main__":
    unittest.main()
