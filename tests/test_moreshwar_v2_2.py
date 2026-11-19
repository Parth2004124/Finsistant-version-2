"""Unit & Quantitative Integrity Tests for Moreshwar 2.2 (Prompt 1: Economics & Ranked Selection)."""

from __future__ import annotations

from datetime import date, timedelta
import unittest

from finsistant_v2.backtest_engine_v2_2 import (
    precompute_v22_universe,
    simulate_portfolio_v2_2,
)
from finsistant_v2.moreshwar_v2_1 import MoreshwarV21Config, evaluate_moreshwar_v2_1
from finsistant_v2.moreshwar_v2_2 import (
    DeliveryCostConfig,
    MoreshwarV22Config,
    allocate_ranked_slots_v2_2,
    compute_causal_factor_snapshot,
    compute_repaired_delivery_costs,
    score_candidate_v2_2,
)


def _dates(n: int, start: date = date(2024, 1, 1)) -> list[str]:
    out = []
    d = start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d.isoformat())
        d += timedelta(days=1)
    return out


class TestMoreshwarV22(unittest.TestCase):
    def test_zerodha_delivery_zero_brokerage_dp_per_scrip_day_and_exact_reconciliation(self) -> None:
        buy = compute_repaired_delivery_costs(
            side="BUY",
            raw_notional=20_000.0,
            currency="INR",
            slippage_bps=7.5,
            dp_charged_today=False,
        )
        # Exec notional = 20000 * 1.00075 = 20015.0; actual slippage = 15.0 (not 20015 * 0.00075)
        self.assertEqual(buy["brokerage"], 0.0)
        self.assertAlmostEqual(buy["exec_notional"], 20_015.0, places=4)
        self.assertAlmostEqual(buy["slippage_cost"], 15.0, places=4)
        self.assertAlmostEqual(buy["stt"], round(20_015.0 * 0.001, 4), places=4)
        self.assertAlmostEqual(buy["stamp_duty"], round(20_015.0 * 0.00015, 4), places=4)
        self.assertEqual(buy["dp_charge"], 0.0)

        # First sell of scrip on date D incurs ₹15.93 DP fee; second sell of same scrip on same day = ₹0 DP
        sell1 = compute_repaired_delivery_costs(
            side="SELL", raw_notional=22_000.0, currency="INR", slippage_bps=7.5, dp_charged_today=False
        )
        sell2 = compute_repaired_delivery_costs(
            side="SELL", raw_notional=22_000.0, currency="INR", slippage_bps=7.5, dp_charged_today=True
        )
        self.assertEqual(sell1["brokerage"], 0.0)
        self.assertAlmostEqual(sell1["dp_charge"], 15.93, places=4)
        self.assertEqual(sell2["dp_charge"], 0.0)
        self.assertAlmostEqual(sell1["total_cost"] - sell2["total_cost"], 15.93, places=4)

        # Verify exact accounting identity: gross_unslipped - total_cost == net_pnl
        gross_unslipped = 22_000.0 - 20_000.0
        net_proceeds = sell1["exec_notional"] - sell1["regulatory_total"]
        net_outlay = buy["exec_notional"] + buy["regulatory_total"]
        net_pnl = net_proceeds - net_outlay
        self.assertAlmostEqual(gross_unslipped - (buy["total_cost"] + sell1["total_cost"]), net_pnl, places=3)

    def test_causal_factor_snapshot_zero_lookahead(self) -> None:
        dts = _dates(90)
        candles = []
        px = 500.0
        for i in range(90):
            px = round(px + (1.4 if i % 2 == 0 else -0.6), 2)
            candles.append({"date": dts[i], "open": px - 0.5, "high": px + 1.5, "low": px - 1.5, "close": px, "volume": 300_000})
        bench = [{"date": dts[i], "close": 22000.0 + i * 25.0} for i in range(90)]

        snap_clean = compute_causal_factor_snapshot(candles, bench, 70, bench_ret60_pct=4.0)
        candles_corrupted = list(candles)
        for j in range(71, 90):
            candles_corrupted[j] = {"date": dts[j], "open": 50.0, "high": 52.0, "low": 40.0, "close": 45.0, "volume": 99_999_999}
        snap_corrupt = compute_causal_factor_snapshot(candles_corrupted, bench, 70, bench_ret60_pct=4.0)
        self.assertEqual(snap_clean, snap_corrupt)

    def test_anti_extension_and_pit_distressed_filters(self) -> None:
        dts = _dates(75)
        # Alternating drift keeping RSI ~60 inside [42,70], followed by extended +5.8% jump above SMA20
        candles = []
        px = 100.0
        for i in range(74):
            px = round(px + (0.9 if i % 2 == 0 else -0.8), 2)
            candles.append({"date": dts[i], "open": px - 0.3, "high": px + 1.5, "low": px - 1.5, "close": px, "volume": 1_000_000})
        ext_px = round(px * 1.058, 2)
        candles.append({"date": dts[74], "open": px, "high": ext_px + 1.5, "low": px - 1.0, "close": ext_px, "volume": 2_000_000})
        bench = [{"date": dts[i], "close": 22000.0 + i * 30.0} for i in range(75)]
        feats = compute_causal_factor_snapshot(candles, bench, 74, bench_ret60_pct=5.0)
        self.assertIsNotNone(feats)
        self.assertLessEqual(feats["rsi14"], 70.0)
        self.assertGreater(feats["pct_above_sma20"], 5.0)
        cfg = MoreshwarV22Config(
            horizon_days=30,
            require_structural_uptrend=True,
            require_anti_extension=True,
            require_pit_risk_filter=True,
        )
        res = score_candidate_v2_2(
            symbol="CHASE",
            sector="Media",
            feats=feats,
            candles_window=candles[-61:],
            regime6={"regime": "STRONG_BULLISH"},
            cfg=cfg,
        )
        self.assertFalse(res["eligible"])
        self.assertIn(res["rejection_category"], ("OVER_EXTENDED", "RSI_OUT_OF_BAND", "PIT_HIGH_ATR_RISK"))

    def test_ranked_allocation_eliminates_alphabetical_bias(self) -> None:
        dts = _dates(100)
        # Ticker 'AAA_WEAK' vs 'ZZZ_STRONG': sorted order MUST prefer ZZZ_STRONG cleanly even if dict reversed
        def _mk_series(step_pos: float, step_neg: float):
            arr = []
            px = 200.0
            for i in range(100):
                px = round(px + (step_pos if i % 2 == 0 else -step_neg), 2)
                arr.append({"date": dts[i], "open": px - 0.5, "high": px + 1.5, "low": px - 1.5, "close": px, "volume": 600_000})
            return arr

        c_a = _mk_series(1.8, 1.4)  # Weak drift
        c_z = _mk_series(3.0, 1.5)  # Strong trend leadership
        bench = [{"date": dts[i], "open": 22000.0 + i * 25.0, "high": 22040.0 + i * 25.0, "low": 21960.0 + i * 25.0, "close": 22010.0 + i * 25.0, "volume": 0} for i in range(100)]

        uni_az = {
            "AAA_WEAK": {"symbol": "AAA_WEAK", "currency": "INR", "sector": "IT Services", "candles": c_a},
            "ZZZ_STRONG": {"symbol": "ZZZ_STRONG", "currency": "INR", "sector": "Capital Goods", "candles": c_z},
        }
        uni_za = {
            "ZZZ_STRONG": uni_az["ZZZ_STRONG"],
            "AAA_WEAK": uni_az["AAA_WEAK"],
        }
        cfg = MoreshwarV22Config(horizon_days=30, max_open_positions=1, use_ranked_selection=True)
        tariff = DeliveryCostConfig()
        pre1 = precompute_v22_universe(uni_az, bench, currency_filter="INR", min_date=dts[0])
        pre2 = precompute_v22_universe(uni_za, bench, currency_filter="INR", min_date=dts[0])
        r1 = simulate_portfolio_v2_2(pre1, experiment_id="T1", label="AZ", cfg=cfg, tariff=tariff)
        r2 = simulate_portfolio_v2_2(pre2, experiment_id="T2", label="ZA", cfg=cfg, tariff=tariff)
        # Permutation invariant: reversing dictionary order produces identical results!
        self.assertEqual(r1["metrics"]["total_return_pct"], r2["metrics"]["total_return_pct"])
        self.assertEqual(r1["metrics"]["total_trades"], r2["metrics"]["total_trades"])

        # Explicit slot allocator picks higher rank_score even if listed second
        cands = [
            {"symbol": "AAA_LOW", "sector": "Banking", "eligible": True, "rank_score": 45.0, "trend_strength": 5.0, "spot": 200.0, "sl_px": 190.0, "tp_px": 225.0, "sl_ratio": 0.05, "tp_ratio": 0.125, "atr_pct": 2.0, "slippage_bps": 7.5, "market_regime": "STRONG_BULLISH"},
            {"symbol": "ZZZ_TOP", "sector": "Pharma", "eligible": True, "rank_score": 88.0, "trend_strength": 18.0, "spot": 200.0, "sl_px": 190.0, "tp_px": 225.0, "sl_ratio": 0.05, "tp_ratio": 0.125, "atr_pct": 2.0, "slippage_bps": 7.5, "market_regime": "STRONG_BULLISH"},
        ]
        orders, _ = allocate_ranked_slots_v2_2(
            candidates=cands,
            candles_by_symbol={"AAA_LOW": c_a, "ZZZ_TOP": c_z},
            open_positions={},
            cash=100_000.0,
            portfolio_nav=100_000.0,
            cfg=cfg,
        )
        self.assertEqual(len(orders), 1)
        self.assertEqual(orders[0]["symbol"], "ZZZ_TOP")

    def test_currency_separation_excludes_usd_from_inr_universe(self) -> None:
        dts = _dates(75)
        c = [{"date": dts[i], "open": 200 + i, "high": 202 + i, "low": 198 + i, "close": 200 + i, "volume": 500_000} for i in range(75)]
        bench = [{"date": dts[i], "close": 22000.0 + i * 20} for i in range(75)]
        uni = {
            "INFY": {"symbol": "INFY", "currency": "INR", "sector": "IT", "candles": c},
            "NVDA": {"symbol": "NVDA", "currency": "USD", "sector": "US Semi", "candles": c},
        }
        pre_inr = precompute_v22_universe(uni, bench, currency_filter="INR", min_date=dts[0])
        pre_usd = precompute_v22_universe(uni, bench, currency_filter="USD", min_date=dts[0])
        self.assertEqual(pre_inr["symbols"], ["INFY"])
        self.assertEqual(pre_usd["symbols"], ["NVDA"])

    def test_m21_frozen_intact(self) -> None:
        dts = _dates(60)
        c = [{"date": dts[i], "open": 300.0, "high": 303.0, "low": 297.0, "close": 300.0 + i * 0.5, "volume": 500_000} for i in range(60)]
        r = evaluate_moreshwar_v2_1(
            symbol="TCS", sector="IT Services", candles=c, spot_price=c[-1]["close"], cfg=MoreshwarV21Config()
        )
        self.assertEqual(r["model_version"], "Moreshwar 2.1")


if __name__ == "__main__":
    unittest.main()
