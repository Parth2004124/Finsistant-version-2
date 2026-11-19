"""Unit & Integration Tests for Moreshwar 2.2 Prompt 2 (Entries, Exits & Profit Retention)."""

from __future__ import annotations

from datetime import date, timedelta
import unittest

from finsistant_v2.backtest_engine_v2_2 import precompute_v22_universe
from finsistant_v2.backtest_engine_v2_2_p2 import simulate_portfolio_v2_2_p2
from finsistant_v2.moreshwar_v2_2 import DeliveryCostConfig, compute_causal_factor_snapshot
from finsistant_v2.moreshwar_v2_2_p2 import (
    MoreshwarV22P2Config,
    check_bar_exit_action_p2,
    compute_profit_retention_metrics,
    evaluate_entry_archetypes_p2,
)


def _dates(n: int, start: date = date(2024, 1, 1)) -> list[str]:
    out = []
    d = start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d.isoformat())
        d += timedelta(days=1)
    return out


class TestMoreshwarV22Prompt2(unittest.TestCase):
    def test_trailing_stop_has_zero_intraday_lookahead_and_gaps_fill_at_open(self) -> None:
        # Bar opens at 205, shoots high=220, then plunges low=194 in SAME bar.
        # Prior SL was 196.0 -> MUST stop out at 196.0 (NOT trail up using 220 and stop at 210!).
        pos = {
            "entry_price": 200.0,
            "last_price": 205.0,
            "max_price_seen": 206.0,
            "max_close_seen": 205.0,
            "qty": 20,
            "atr_entry": 4.0,
            "sl": 196.0,
            "tp": 0.0,
            "partial_target": 212.0,
            "partial_booked": False,
            "breakeven_locked": False,
            "trailing_active": False,
            "mfe_pct": 0.0,
            "mae_pct": 0.0,
            "bars_held": 3,
        }
        bar = {"open": 205.0, "high": 220.0, "low": 194.0, "close": 195.0, "volume": 500_000}
        act = check_bar_exit_action_p2(
            pos=pos,
            bar=bar,
            bar_feats={"sma20": 202.0, "atr14": 4.0, "rsi14": 50.0},
            regime6_name="STRONG_BULLISH",
            is_final_bar=False,
            cfg=MoreshwarV22P2Config(exit_mode="P2_ATR_TRAILING", trailing_atr_mult=2.5),
        )
        self.assertEqual(act["action"], "FULL_EXIT")
        self.assertEqual(act["exit_px"], 196.0)
        self.assertFalse(act["is_gap_stop"])

        # Gap-through-stop check: open=185 < sl=196 -> must fill at 185 (not 196!)
        pos2 = dict(pos)
        gap_bar = {"open": 185.0, "high": 192.0, "low": 182.0, "close": 188.0, "volume": 500_000}
        act2 = check_bar_exit_action_p2(
            pos=pos2,
            bar=gap_bar,
            bar_feats={"sma20": 202.0, "atr14": 4.0, "rsi14": 45.0},
            regime6_name="STRONG_BULLISH",
            is_final_bar=False,
            cfg=MoreshwarV22P2Config(exit_mode="P2_ATR_TRAILING", trailing_atr_mult=2.5),
        )
        self.assertEqual(act2["action"], "FULL_EXIT")
        self.assertEqual(act2["exit_px"], 185.0)
        self.assertTrue(act2["is_gap_stop"])

    def test_breakeven_lock_includes_cost_buffer_post_close(self) -> None:
        pos = {
            "entry_price": 200.0,
            "last_price": 205.0,
            "max_price_seen": 205.0,
            "max_close_seen": 205.0,
            "qty": 20,
            "atr_entry": 4.0,
            "sl": 192.0,
            "tp": 0.0,
            "partial_target": 212.0,
            "partial_booked": False,
            "breakeven_locked": False,
            "trailing_active": False,
            "mfe_pct": 0.0,
            "mae_pct": 0.0,
            "bars_held": 4,
        }
        # Close rises to 207.0 (>= 200 + 1.5*4 = 206) -> after close, stop ratchets to 200 * 1.0065 = 201.30
        bar = {"open": 205.0, "high": 208.0, "low": 204.0, "close": 207.0, "volume": 400_000}
        act = check_bar_exit_action_p2(
            pos=pos,
            bar=bar,
            bar_feats={"sma20": 201.0, "atr14": 4.0, "rsi14": 60.0},
            regime6_name="STRONG_BULLISH",
            is_final_bar=False,
            cfg=MoreshwarV22P2Config(exit_mode="P1_BREAKEVEN_LOCK", breakeven_trigger_atr=1.5),
        )
        self.assertEqual(act["action"], "HOLD")
        self.assertTrue(pos["breakeven_locked"])
        self.assertAlmostEqual(pos["sl"], 201.3, places=2)

    def test_pullback_and_breakout_archetype_discrimination(self) -> None:
        dts = _dates(75)
        candles = []
        px = 200.0
        for i in range(72):
            px = round(px + (1.8 if i % 2 == 0 else -0.8), 2)
            candles.append({"date": dts[i], "open": px - 0.5, "high": px + 1.5, "low": px - 1.5, "close": px, "volume": 500_000})
        # Controlled 2-bar dip to near SMA20 then bounce bar
        sma20_approx = sum(c["close"] for c in candles[-20:]) / 20.0
        candles.append({"date": dts[72], "open": sma20_approx + 1.5, "high": sma20_approx + 2.0, "low": sma20_approx - 0.5, "close": sma20_approx, "volume": 500_000})
        candles.append({"date": dts[73], "open": sma20_approx - 0.5, "high": sma20_approx + 1.0, "low": sma20_approx - 1.0, "close": sma20_approx + 0.5, "volume": 500_000})
        candles.append({"date": dts[74], "open": sma20_approx + 0.5, "high": sma20_approx + 3.0, "low": sma20_approx, "close": sma20_approx + 2.5, "volume": 600_000})

        bench = [{"date": dts[i], "close": 22000.0 + i * 20.0} for i in range(75)]
        feats = compute_causal_factor_snapshot(candles, bench, 74, bench_ret60_pct=3.0)
        self.assertIsNotNone(feats)

        ev_pb = evaluate_entry_archetypes_p2(
            symbol="PB_TEST",
            feats=feats,
            candles_window=candles[-61:],
            regime6={"regime": "STRONG_BULLISH"},
            cfg=MoreshwarV22P2Config(entry_mode="PULLBACK", require_pit_filters=False),
        )
        self.assertTrue(ev_pb["matched"])
        self.assertEqual(ev_pb["archetype"], "PULLBACK")

    def test_partial_booking_full_portfolio_integration_and_reconciliation(self) -> None:
        # Synthetic uptrend hitting 1.5R partial booking then trailing stop/horizon exit
        dts = _dates(110)
        candles = []
        px = 200.0
        for i in range(70):
            px = round(px + (1.6 if i % 2 == 0 else -0.7), 2)
            candles.append({"date": dts[i], "open": px - 0.4, "high": px + 1.5, "low": px - 1.5, "close": px, "volume": 600_000})
        # Strong rally phase triggering 1.5R partial booking cleanly, then steady fade hitting trailing stop
        for i in range(70, 95):
            px = round(px + 2.2, 2)
            candles.append({"date": dts[i], "open": px - 1.0, "high": px + 2.5, "low": px - 1.5, "close": px, "volume": 650_000})
        for i in range(95, 110):
            px = round(px - 2.8, 2)
            candles.append({"date": dts[i], "open": px + 0.5, "high": px + 1.0, "low": px - 2.0, "close": px, "volume": 650_000})

        bench = [{"date": dts[i], "open": 22000.0 + i * 25.0, "high": 22040.0 + i * 25.0, "low": 21960.0 + i * 25.0, "close": 22010.0 + i * 25.0, "volume": 0} for i in range(110)]
        uni = {"SYNTH_P3": {"symbol": "SYNTH_P3", "currency": "INR", "sector": "IT Services", "candles": candles}}
        pre = precompute_v22_universe(uni, bench, currency_filter="INR", min_date=dts[0])

        res = simulate_portfolio_v2_2_p2(
            pre,
            experiment_id="INT_P3",
            label="Partial Integration",
            cfg=MoreshwarV22P2Config(entry_mode="M21_BASE", exit_mode="P3_PARTIAL_BOOKING", require_pit_filters=False, horizon_days=45),
            tariff=DeliveryCostConfig(),
        )
        m = res["metrics"]
        self.assertGreaterEqual(m["total_trades"], 1)
        self.assertGreaterEqual(m["execution_realism"]["partial_exits_booked"], 1)
        # Critical accounting identity: sum(trade pnls) MUST match NAV change within rounding tolerance!
        self.assertLessEqual(abs(m["reconciliation_diff"]), 1.0)

    def test_profit_retention_metrics_computation(self) -> None:
        trades = [
            {"symbol": "A", "archetype": "PULLBACK", "signal_regime6": "STRONG_BULLISH", "atr_pct": 1.8, "holding_days": 18, "raw_buy_notional": 10000.0, "pnl_amount": 800.0, "net_return_pct": 8.0, "mfe_pct": 10.0, "mae_pct": -1.2, "exit_reason": "STOP_LOSS_TRAIL"},
            {"symbol": "B", "archetype": "BREAKOUT", "signal_regime6": "WEAK_BULLISH", "atr_pct": 2.6, "holding_days": 10, "raw_buy_notional": 10000.0, "pnl_amount": -400.0, "net_return_pct": -4.0, "mfe_pct": 2.0, "mae_pct": -4.2, "exit_reason": "STOP_LOSS"},
        ]
        ret = compute_profit_retention_metrics(trades)
        self.assertEqual(ret["total_trades"], 2)
        self.assertEqual(ret["profitable_1_5pct_closed_loss"], 1)
        self.assertAlmostEqual(ret["winner_mfe_retention_pct"], 80.0, places=2)


if __name__ == "__main__":
    unittest.main()
