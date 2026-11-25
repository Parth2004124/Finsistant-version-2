"""Adversarial & Bloomberg-Standard Trade Automation Tests (RBLD + EMSX Blotter + WHLS + BTCA)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class TestBloombergTradeAutomation(unittest.TestCase):
    def test_rbld_compliance_blocks_bad_data_illiquid_spread_drift_and_allows_exits(self) -> None:
        from finsistant_v2.trade_automation import (
            COMPLIANCE_BLOCKED,
            ROUTED,
            EMSOrderBlotter,
            RuleBuilderEngine,
        )

        rbld = RuleBuilderEngine()
        blotter = EMSOrderBlotter()

        # 1. Fail-closed data integrity
        v_bad = rbld.evaluate_order(
            symbol="BADCO", side="BUY", arrival_price=100.0, order_notional_inr=10_000.0,
            nav_inr=100_000.0, adv_20d_inr=100_000_000.0, hl_range_pct=2.0, daily_vol_frac=0.015,
            fundamentals_status="unavailable",
        )
        self.assertFalse(v_bad["compliance_passed"])
        self.assertEqual(v_bad["lifecycle_state"], COMPLIANCE_BLOCKED)
        self.assertEqual(v_bad["blocked_rules"][0]["rule_id"], "RBLD_01_DATA_INTEGRITY")
        blotter.record_compliance_result(v_bad, qty=100)

        # 2. Price drift > 5% blocked
        v_drift = rbld.evaluate_order(
            symbol="DRIFTCO", side="BUY", arrival_price=100.0, order_notional_inr=10_000.0,
            nav_inr=100_000.0, adv_20d_inr=100_000_000.0, hl_range_pct=2.0, daily_vol_frac=0.015,
            price_discrepancy_pct=6.5,
        )
        self.assertFalse(v_drift["compliance_passed"])
        self.assertEqual(v_drift["blocked_rules"][0]["rule_id"], "RBLD_03_PRICE_BAND_DRIFT")

        # 3. Illiquid ADV < INR 2 Cr blocked
        v_illiq = rbld.evaluate_order(
            symbol="ILLIQ", side="BUY", arrival_price=100.0, order_notional_inr=10_000.0,
            nav_inr=100_000.0, adv_20d_inr=5_000_000.0, hl_range_pct=2.0, daily_vol_frac=0.015,
        )
        self.assertFalse(v_illiq["compliance_passed"])
        self.assertEqual(v_illiq["blocked_rules"][0]["rule_id"], "RBLD_04_MIN_LIQUIDITY_ADV")

        # 4. Wide spread (>7.5%) blocked for BUY, but protective SELL still routes via NEXT_SESSION_OPEN!
        v_spread = rbld.evaluate_order(
            symbol="WIDECO", side="BUY", arrival_price=100.0, order_notional_inr=10_000.0,
            nav_inr=100_000.0, adv_20d_inr=100_000_000.0, hl_range_pct=9.2, daily_vol_frac=0.015,
        )
        self.assertFalse(v_spread["compliance_passed"])
        self.assertEqual(v_spread["blocked_rules"][0]["rule_id"], "RBLD_05_SPREAD_VOL_LIMIT")

        v_exit = rbld.evaluate_order(
            symbol="WIDECO", side="SELL", arrival_price=100.0, order_notional_inr=10_000.0,
            nav_inr=100_000.0, adv_20d_inr=5_000_000.0, hl_range_pct=9.2, daily_vol_frac=0.045,
            desk_halted=True, desk_halt_reasons=["DRAWDOWN"],
        )
        self.assertTrue(v_exit["compliance_passed"])
        self.assertEqual(v_exit["lifecycle_state"], ROUTED)
        self.assertEqual(v_exit["selected_algo"], "NEXT_SESSION_OPEN")
        self.assertEqual(blotter.to_payload(rules=rbld.get_rules())["counts"]["exceptions"], 1)

    def test_algo_wheel_routing_tiers_and_kill_switch(self) -> None:
        from finsistant_v2.trade_automation import RuleBuilderEngine, compute_pre_trade_tca

        rbld = RuleBuilderEngine()
        halted, reasons = rbld.evaluate_desk_kill_switch(drawdown_pct=10.5, recon_diff_inr=0.05, btca_drift_breached=True)
        self.assertTrue(halted)
        self.assertEqual(len(reasons), 3)

        # Low -> NEXT_SESSION_OPEN, Moderate -> LIMIT_ORDER_SIM, High vol -> VOLATILITY_ADJUSTED, Wide spread -> SPREAD_AWARE
        self.assertEqual(
            compute_pre_trade_tca(side="BUY", arrival_price=100.0, order_notional_inr=10_000.0, adv_20d_inr=500_000_000.0, hl_range_pct=1.2, daily_vol_frac=0.012)["selected_algo"],
            "NEXT_SESSION_OPEN",
        )
        self.assertEqual(
            compute_pre_trade_tca(side="BUY", arrival_price=100.0, order_notional_inr=10_000.0, adv_20d_inr=200_000_000.0, hl_range_pct=3.1, daily_vol_frac=0.018)["selected_algo"],
            "LIMIT_ORDER_SIM",
        )
        self.assertEqual(
            compute_pre_trade_tca(side="BUY", arrival_price=100.0, order_notional_inr=10_000.0, adv_20d_inr=200_000_000.0, hl_range_pct=3.5, daily_vol_frac=0.035)["selected_algo"],
            "VOLATILITY_ADJUSTED",
        )
        self.assertEqual(
            compute_pre_trade_tca(side="BUY", arrival_price=100.0, order_notional_inr=10_000.0, adv_20d_inr=200_000_000.0, hl_range_pct=5.2, daily_vol_frac=0.018)["selected_algo"],
            "SPREAD_AWARE",
        )

    def test_post_trade_btca_implementation_shortfall_and_m32_engine_whls_integration(self) -> None:
        from finsistant_v2.m3_2.orchestrated_session import ORCH_STRATEGY_ID, OrchestratedSessionRunner
        from finsistant_v2.trade_automation import summarize_btca_and_feedback

        # Direct BTCA feedback calibration check when slippage runs hot
        hot_fills = [
            {"algo_strategy": "NEXT_SESSION_OPEN", "fill_rate_pct": 100.0, "implementation_shortfall_bps": 120.0, "delay_cost_bps": 20.0, "execution_slippage_bps": 72.0, "statutory_cost_bps": 28.0, "slippage_surprise_bps": 85.0}
            for _ in range(5)
        ]
        fb = summarize_btca_and_feedback(hot_fills)
        self.assertTrue(fb["kill_switch_drift_breached"])
        self.assertGreater(fb["tca_hurdle_adder_bps"], 0.0)

        # End-to-end M3.2 execution routing with per-order execution_mode -> BTCA sync
        with tempfile.TemporaryDirectory() as tmp:
            tdir = Path(tmp)
            runner = OrchestratedSessionRunner(
                db_path=tdir / "m32.db",
                learning_db_path=tdir / "learn.db",
                reports_dir=tdir / "rep",
                scan_dir=tdir / "scan",
            )
            dates = ["2026-11-20", "2026-11-23"]
            for i, d in enumerate(dates):
                px_nb = 290.0 + i
                px_a = 100.0 + i * 1.5
                cands = (
                    [{
                        "sec_id": "SEC_W", "symbol": "WHLSCO", "strategy_id": ORCH_STRATEGY_ID,
                        "score": 0.90, "stop_ratio": 0.09, "execution_mode": "SPREAD_AWARE",
                        "pre_trade_cost_bps": 42.0, "matched_rules": ["RBLD_01_DATA_INTEGRITY"],
                    }]
                    if i == 0 else []
                )
                snap = {
                    "trading_date": d,
                    "data_timestamp_utc": f"{d}T17:00:00+05:30",
                    "niftybees_bar": {"open": px_nb, "high": px_nb + 1, "low": px_nb - 1, "close": px_nb, "prev_close": px_nb - 0.5, "volume": 4_000_000.0, "atr_14": 2.5, "vol_20d": 0.01, "valid": True},
                    "liquidbees_bar": {"open": 1000.0, "high": 1000.1, "low": 999.9, "close": 1000.0, "prev_close": 1000.0, "volume": 500_000.0, "atr_14": 0.2, "vol_20d": 0.001, "valid": True},
                    "niftybees_close_history": [px_nb - 20.0] * 210 + [px_nb],
                    "equity_bars": {
                        "SEC_W": {
                            "sec_id": "SEC_W", "symbol": "WHLSCO", "sector": "IT", "industry": "Software",
                            "is_unknown_sector": False, "open": px_a, "high": px_a + 1, "low": px_a - 1,
                            "close": px_a, "prev_close": px_a - 1.0, "volume": 1_000_000.0, "atr_14": 2.0,
                            "vol_20d": 0.014, "hl_range_pct": 2.0, "adv_20d_inr": 100_000_000.0,
                            "div_yield_ttm": 0.01, "valid": True,
                        }
                    },
                    "family_e_candidates": cands,
                    "pairwise_corr_map": {},
                    "diagnostic_index_available": False,
                }
                res = runner.orch.run_daily_cycle(
                    trading_date=d, market_snapshot=snap, observation_mode="PROSPECTIVE_PAPER", export_reports=False
                )
                orders = (res.get("execution_result") or {}).get("orders") or []
                if i == 1:
                    added = runner.ledger.sync_btca_executions(orders)
                    self.assertGreaterEqual(added, 1)
                    btca = runner.ledger.btca_report()
                    self.assertGreaterEqual(btca["filled_orders"], 1)
                    self.assertIn("SPREAD_AWARE", btca["algo_breakdown"])
                    self.assertIsNotNone(btca["mean_implementation_shortfall_bps"])

    def test_negative_pe_loss_company_scores_cleanly_and_server_isolates_failed_symbol(self) -> None:
        from finsistant_v2.scoring_engine import calculate_fundamental_score
        import server

        loss_meta = {
            "name": "Distressed Media Ltd",
            "sector": "Media",
            "pe": None,  # loss-making company genuinely has no positive P/E
            "sales_growth": 4.0,
            "profit_growth": -25.0,
            "roe": -4.0,
            "roce": 2.0,
            "opm": 9.0,
            "mcap_cr": 12000.0,
            "debt_to_equity": None,
        }
        sc = calculate_fundamental_score(loss_meta, strict=True)
        self.assertIsNone(sc["ratios"]["pe"])
        self.assertIn("Negative or unavailable TTM earnings", sc["nlp"])

        f_real = calculate_fundamental_score({
            "name": "Reliance Mock", "sector": "Energy", "pe": 20.0, "roe": 22.0,
            "roce": 24.0, "opm": 22.0, "sales_growth": 18.0, "profit_growth": 20.0, "mcap_cr": 50000.0,
        })
        from finsistant_v2.scoring_engine import calculate_moreshwar_levels, calculate_porters_score
        p_real = calculate_porters_score({
            "name": "Reliance Mock", "sector": "Energy", "roe": 22.0,
            "roce": 24.0, "opm": 22.0, "sales_growth": 18.0, "profit_growth": 20.0, "mcap_cr": 50000.0,
        })
        m_real = calculate_moreshwar_levels(250.0, f_real["total"], p_real["total"], is_holding=False)

        def _fake_analysis(sym: str, is_holding: bool = False, force_refresh: bool = False):
            if sym == "TATAMOTORS":
                raise RuntimeError("Fundamental data unavailable for TMCV")
            candles = [
                {"date": f"2026-10-{d + 1:02d}", "open": 100 + d, "high": 102 + d, "low": 99 + d, "close": 101 + d, "volume": 1_000_000}
                for d in range(25)
            ]
            return {
                "symbol": sym, "yahoo_symbol": f"{sym}.NS", "name": sym, "exchange": "NSE",
                "currency": "INR", "sector": "IT", "asset_type": "STOCK", "setup_type": "Breakout",
                "price": 250.0, "change_pct": 1.2, "last_candle_date": candles[-1]["date"],
                "last_candle_close": 250.0, "rsi_14": 58.0, "sma_20": 240.0, "ema_12": 245.0,
                "ema_26": 238.0, "macd": 2.5, "macd_signal": 1.8, "bb_upper": 260.0, "bb_lower": 225.0,
                "projected_30d_target": 275.0, "regression_curve": "Upward Curve", "regression_std_dev": 3.0,
                "fundamentals_status": "live",
                "fundamental_score": f_real,
                "porters_score": p_real,
                "moreshwar_levels": m_real,
                "decision_matrix": {"action": "BUY NOW", "rationale": "Strong fundamentals"},
                "candles": candles,
            }

        with (
            patch.object(server, "fetch_kite_portfolio", return_value={"connected": False, "status": "Paper", "cash_balance": 100_000.0, "holdings": []}),
            patch.object(server, "_load_dynamic_orchestrator_symbols", return_value=(["RELIANCE", "TATAMOTORS"], "TEST_UNIVERSE")),
            patch.object(server, "_load_extra_watchlist_symbols", return_value=[]),
            patch.object(server, "get_asset_full_analysis", side_effect=_fake_analysis),
            patch.object(server, "fetch_live_market_snapshot", side_effect=RuntimeError("skip hist refresh")),
            patch.object(server, "critique_suggested_trades", return_value={}),
            patch.object(server, "_refresh_analytics_and_audit", return_value=None),
        ):
            server._initialize_platform_state(force_refresh=True)
            self.assertIn("RELIANCE", server.PORTFOLIO_STATE)
            self.assertNotIn("TATAMOTORS", server.PORTFOLIO_STATE)
            blotter = server.EMS_BLOTTER.to_payload(rules=server.RBLD_ENGINE.get_rules())
            exc_syms = [e["symbol"] for e in blotter["exception_queue"]]
            self.assertIn("TATAMOTORS", exc_syms)


if __name__ == "__main__":
    unittest.main()
