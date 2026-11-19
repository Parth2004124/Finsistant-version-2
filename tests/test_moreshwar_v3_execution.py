"""Automated Validation Tests for Moreshwar 3.0 Prompt 8 (Execution Intelligence & Realistic Cost Modelling)."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from finsistant_v2.m3.execution_cost_engine import (
    HistoricalTariffSchedule,
    ZerodhaBrokerAdapter,
    calculate_historical_indian_costs,
    compute_market_impact_and_slippage_bps,
    simulate_order_execution_bar,
)
from finsistant_v2.m3.execution_report import (
    FORBIDDEN_CREDENTIAL_MARKERS,
    generate_execution_and_security_report,
)
from finsistant_v2.m3.schema import DEFAULT_M3_REPORTS_DIR


class TestMoreshwarV3Execution(unittest.TestCase):
    def test_historical_tariffs_pre_vs_post_oct_2024_delivery_intraday_futures(self) -> None:
        notional = 1_000_000.0  # INR 10L so exchange/SEBI rates differentiate cleanly

        # 1. DELIVERY (CNC): STT 0.10% Buy & Sell, Stamp 0.015% Buy only, Exch 0.00325% -> 0.00297%, SEBI 0.0001%, GST 18%
        del_buy_pre = calculate_historical_indian_costs(
            product_type="DELIVERY", side="BUY", raw_notional=notional, trade_date="2024-09-30", slippage_bps=0.0
        )
        del_buy_post = calculate_historical_indian_costs(
            product_type="DELIVERY", side="BUY", raw_notional=notional, trade_date="2024-10-01", slippage_bps=0.0
        )
        self.assertEqual(del_buy_pre["brokerage"], 0.0)
        self.assertAlmostEqual(del_buy_pre["stt"], 1000.0, places=2)
        self.assertAlmostEqual(del_buy_pre["stamp_duty"], 150.0, places=2)
        self.assertAlmostEqual(del_buy_pre["exchange_txn"], 32.50, places=2)
        self.assertAlmostEqual(del_buy_post["exchange_txn"], 29.70, places=2)
        self.assertAlmostEqual(del_buy_pre["sebi_fee"], 1.00, places=2)
        self.assertAlmostEqual(del_buy_pre["gst"], round((0.0 + 32.50 + 1.00) * 0.18, 2), places=2)
        self.assertAlmostEqual(del_buy_post["gst"], round((0.0 + 29.70 + 1.00) * 0.18, 2), places=2)
        self.assertEqual(del_buy_pre["dp_charge"], 0.0)

        # 2. INTRADAY (MIS): Brokerage min(20, 0.03%), STT 0.025% Sell only, Stamp 0.003% Buy only
        mis_buy = calculate_historical_indian_costs(
            product_type="INTRADAY", side="BUY", raw_notional=notional, trade_date="2024-10-01", slippage_bps=0.0
        )
        mis_sell = calculate_historical_indian_costs(
            product_type="INTRADAY", side="SELL", raw_notional=notional, trade_date="2024-10-01", slippage_bps=0.0
        )
        self.assertEqual(mis_buy["brokerage"], 20.0)
        self.assertEqual(mis_buy["stt"], 0.0)
        self.assertAlmostEqual(mis_buy["stamp_duty"], 30.0, places=2)
        self.assertEqual(mis_buy["dp_charge"], 0.0)
        self.assertEqual(mis_sell["brokerage"], 20.0)
        self.assertAlmostEqual(mis_sell["stt"], 250.0, places=2)
        self.assertEqual(mis_sell["stamp_duty"], 0.0)
        self.assertAlmostEqual(mis_sell["gst"], round((20.0 + 29.70 + 1.00) * 0.18, 2), places=2)

        # Small intraday order checks 0.03% < INR 20 cap
        mis_small = calculate_historical_indian_costs(
            product_type="INTRADAY", side="BUY", raw_notional=10_000.0, trade_date="2024-10-01", slippage_bps=0.0
        )
        self.assertAlmostEqual(mis_small["brokerage"], 3.0, places=2)

        # 3. FUTURES (NRML): STT 0.0125% pre-Oct-24 -> 0.02% post-Oct-24 Sell only, Exch 0.00173%, Stamp 0.002% Buy
        fut_buy = calculate_historical_indian_costs(
            product_type="FUTURES", side="BUY", raw_notional=notional, trade_date="2024-10-01", slippage_bps=0.0
        )
        fut_sell_pre = calculate_historical_indian_costs(
            product_type="FUTURES", side="SELL", raw_notional=notional, trade_date="2024-09-30", slippage_bps=0.0
        )
        fut_sell_post = calculate_historical_indian_costs(
            product_type="FUTURES", side="SELL", raw_notional=notional, trade_date="2024-10-01", slippage_bps=0.0
        )
        self.assertEqual(fut_buy["stt"], 0.0)
        self.assertAlmostEqual(fut_buy["stamp_duty"], 20.0, places=2)
        self.assertAlmostEqual(fut_buy["exchange_txn"], 17.30, places=2)
        self.assertAlmostEqual(fut_sell_pre["stt"], 125.0, places=2)
        self.assertAlmostEqual(fut_sell_post["stt"], 200.0, places=2)

    def test_dp_charge_symmetry_and_same_day_dedup(self) -> None:
        buy = calculate_historical_indian_costs(
            product_type="DELIVERY", side="BUY", raw_notional=50_000.0, trade_date="2025-01-01", slippage_bps=0.0
        )
        sell_first = calculate_historical_indian_costs(
            product_type="DELIVERY", side="SELL", raw_notional=50_000.0, trade_date="2025-01-01", slippage_bps=0.0, dp_charged_today=False
        )
        sell_second = calculate_historical_indian_costs(
            product_type="DELIVERY", side="SELL", raw_notional=50_000.0, trade_date="2025-01-01", slippage_bps=0.0, dp_charged_today=True
        )
        sell_zero = calculate_historical_indian_costs(
            product_type="DELIVERY", side="SELL", raw_notional=0.0, trade_date="2025-01-01", slippage_bps=0.0, dp_charged_today=False
        )
        self.assertEqual(buy["dp_charge"], 0.0)
        self.assertAlmostEqual(sell_first["dp_charge"], 15.93, places=2)
        self.assertEqual(sell_second["dp_charge"], 0.0)
        self.assertEqual(sell_zero["dp_charge"], 0.0)
        self.assertEqual(sell_first["stamp_duty"], 0.0)

    def test_square_root_market_impact_monotonic_across_sizes_and_adv(self) -> None:
        small_imp = compute_market_impact_and_slippage_bps(order_notional_inr=10_000.0, adv_20d_inr=500_000_000.0)
        large_imp = compute_market_impact_and_slippage_bps(order_notional_inr=5_000_000.0, adv_20d_inr=50_000_000.0)
        self.assertGreaterEqual(small_imp["half_spread_bps"], HistoricalTariffSchedule().base_half_spread_bps)
        self.assertGreater(large_imp["market_impact_bps"], small_imp["market_impact_bps"])
        self.assertGreater(large_imp["total_slippage_bps"], small_imp["total_slippage_bps"])
        self.assertAlmostEqual(large_imp["participation_pct_adv"], 10.0, places=2)

    def test_order_execution_bar_simulation_all_types_and_throttling(self) -> None:
        bar = {"open": 1000.0, "high": 1020.0, "low": 985.0, "close": 1010.0, "volume": 100_000}

        # MARKET: fills at next-bar open + impact/spread
        mkt = simulate_order_execution_bar(
            order_type="MARKET", side="BUY", req_qty=100, bar=bar, max_bar_vol_participation=0.05, max_order_adv_pct=5.0
        )
        self.assertEqual(mkt["status"], "FILLED")
        self.assertEqual(mkt["raw_price"], 1000.0)
        self.assertGreater(mkt["fill_price"], 1000.0)

        # LIMIT: unfilled when price never reaches limit
        lim_unfilled = simulate_order_execution_bar(
            order_type="LIMIT", side="BUY", req_qty=100, limit_price=980.0, bar=bar
        )
        self.assertEqual(lim_unfilled["status"], "UNFILLED")
        self.assertEqual(lim_unfilled["filled_qty"], 0)
        self.assertIsNone(lim_unfilled["fill_price"])

        # LIMIT: filled passively at limit when bar trades through limit (0 crossing spread)
        lim_filled = simulate_order_execution_bar(
            order_type="LIMIT", side="BUY", req_qty=100, limit_price=990.0, bar=bar, max_bar_vol_participation=0.05
        )
        self.assertEqual(lim_filled["status"], "FILLED")
        self.assertEqual(lim_filled["fill_price"], 990.0)
        self.assertEqual(lim_filled["slippage_bps"], 0.0)

        # LIMIT: partial fill capped by bar participation
        lim_partial = simulate_order_execution_bar(
            order_type="LIMIT",
            side="BUY",
            req_qty=1000,
            limit_price=990.0,
            bar={"open": 1000.0, "high": 1010.0, "low": 988.0, "close": 1002.0, "volume": 10_000},
            adv_20d_inr=200_000_000.0,
            max_bar_vol_participation=0.05,
            max_order_adv_pct=5.0,
        )
        self.assertEqual(lim_partial["status"], "PARTIAL_FILL")
        self.assertEqual(lim_partial["filled_qty"], 500)

        # SL_M: gap-through stop fills at actual gap open (not phantom trigger price)
        gap_bar = {"open": 940.0, "high": 955.0, "low": 930.0, "close": 948.0, "volume": 100_000}
        slm_gap = simulate_order_execution_bar(
            order_type="SL_M", side="SELL", req_qty=100, trigger_price=970.0, bar=gap_bar, max_bar_vol_participation=0.05
        )
        self.assertEqual(slm_gap["status"], "FILLED")
        self.assertTrue(slm_gap["is_gap_fill"])
        self.assertEqual(slm_gap["reason"], "GAP_THROUGH_OPEN_FILL")
        self.assertEqual(slm_gap["raw_price"], 940.0)
        self.assertLess(slm_gap["fill_price"], 940.0)

        # SL_L: gap beyond stop-limit remains realistically UNFILLED
        sll_gap = simulate_order_execution_bar(
            order_type="SL_L", side="SELL", req_qty=100, trigger_price=970.0, limit_price=955.0, bar=gap_bar
        )
        self.assertEqual(sll_gap["status"], "UNFILLED")
        self.assertEqual(sll_gap["reason"], "SL_LIMIT_GAP_THROUGH_UNFILLED")
        self.assertEqual(sll_gap["filled_qty"], 0)

        # ADV 5% participation throttling: req_qty=5000 (INR 50L) vs ADV 5 Cr -> capped to 5% (2500), then bar 5% of 40k = 2000
        throttled = simulate_order_execution_bar(
            order_type="MARKET",
            side="BUY",
            req_qty=5000,
            bar={"open": 1000.0, "high": 1015.0, "low": 990.0, "close": 1005.0, "volume": 40_000},
            adv_20d_inr=50_000_000.0,
            max_bar_vol_participation=0.05,
            max_order_adv_pct=5.0,
        )
        self.assertEqual(throttled["status"], "PARTIAL_FILL")
        self.assertEqual(throttled["requested_qty"], 2500)
        self.assertEqual(throttled["filled_qty"], 2000)

    def test_zerodha_broker_adapter_security_isolation(self) -> None:
        adapter = ZerodhaBrokerAdapter()
        posture = adapter.get_security_posture()
        self.assertFalse(posture["live_execution_enabled"])
        self.assertTrue(posture["paper_mode"])
        self.assertFalse(posture["secrets_exposed_to_agents"])
        self.assertEqual(posture["capabilities"]["live_order_submission"], "DISABLED_BY_DEFAULT")

        ticket = adapter.construct_order(symbol="INFY", side="BUY", qty=10, order_type="LIMIT", product="CNC", price=1500.0)
        with self.assertRaises(PermissionError):
            adapter.submit_order(ticket)

        adapter.approve_order(ticket, approved_by="RISK_OFFICER")
        receipt = adapter.submit_order(ticket)
        self.assertEqual(receipt["mode"], "PAPER_SIMULATION")
        self.assertEqual(receipt["status"], "PAPER_ACCEPTED")
        self.assertTrue(receipt["order_id"].startswith("PAPER-"))

        # Even if constructed with live_execution_enabled=True, paper_mode=False, live placement is blocked
        live_adapter = ZerodhaBrokerAdapter(live_execution_enabled=True, paper_mode=False)
        live_ticket = live_adapter.construct_order(symbol="INFY", side="SELL", qty=5, order_type="MARKET", product="CNC")
        live_adapter.approve_order(live_ticket, approved_by="RISK_OFFICER")
        with self.assertRaises(PermissionError):
            live_adapter.submit_order(live_ticket)

    def test_generate_execution_and_security_report_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            report = generate_execution_and_security_report(reports_dir=out)
            json_path = out / "execution_quality_report.json"
            md_path = out / "SECURITY_AND_TARIFF_VALIDATION.md"
            self.assertTrue(json_path.exists())
            self.assertTrue(md_path.exists())

            loaded = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertTrue(loaded["security_audit"]["all_security_invariants_passed"])
            self.assertEqual(len(loaded["market_impact_matrix"]), 20)  # 5 sizes x 4 ADV tiers
            self.assertGreaterEqual(len(loaded["execution_simulations"]), 8)

            combined_text = (json_path.read_text(encoding="utf-8") + md_path.read_text(encoding="utf-8")).lower()
            for marker in FORBIDDEN_CREDENTIAL_MARKERS:
                self.assertNotIn(marker, combined_text)

        # Also check canonical Prompt 8 deliverables if already generated
        canon_dir = DEFAULT_M3_REPORTS_DIR / "execution"
        if (canon_dir / "execution_quality_report.json").exists():
            canon_json = json.loads((canon_dir / "execution_quality_report.json").read_text(encoding="utf-8"))
            self.assertTrue(canon_json["security_audit"]["all_security_invariants_passed"])
            self.assertTrue((canon_dir / "SECURITY_AND_TARIFF_VALIDATION.md").exists())


if __name__ == "__main__":
    unittest.main()
