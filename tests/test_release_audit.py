"""Independent Release Audit Tests for Moreshwar 2.1, Math Accuracy, Security & APIs."""

from __future__ import annotations

from datetime import date, timedelta
from http.server import ThreadingHTTPServer
import json
import math
from pathlib import Path
import threading
import unittest
import urllib.error
import urllib.request

from finsistant_v2.backtest_engine_v2_1 import (
    INITIAL_CAPITAL,
    RISK_FREE_ANNUAL,
    _finalize_v21_report,
    precompute_causal_features,
    run_monte_carlo_suite,
    simulate_portfolio_v2_1,
)
from finsistant_v2.config import FinsistantSettings
from finsistant_v2.moreshwar_v2_1 import (
    MoreshwarV21Config,
    compute_indian_delivery_costs,
    evaluate_moreshwar_v2_1,
)
import server


def _date_seq(n: int, start: date = date(2024, 1, 1)) -> list[str]:
    out = []
    d = start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d.isoformat())
        d += timedelta(days=1)
    return out


class TestMoreshwarReleaseAudit(unittest.TestCase):
    def test_manual_indian_delivery_cost_math_exact(self) -> None:
        buy = compute_indian_delivery_costs(side="BUY", notional=50_000.0, currency="INR", slippage_bps=7.5)
        self.assertAlmostEqual(buy["brokerage"], 15.0, places=4)
        self.assertAlmostEqual(buy["stt"], 50.0, places=4)
        self.assertAlmostEqual(buy["exchange_txn"], 1.485, places=4)
        self.assertAlmostEqual(buy["sebi_fee"], 0.05, places=4)
        self.assertAlmostEqual(buy["stamp_duty"], 7.5, places=4)
        self.assertAlmostEqual(buy["gst"], round(0.18 * (15.0 + 1.485 + 0.05), 4), places=4)
        self.assertEqual(buy["dp_charge"], 0.0)
        self.assertAlmostEqual(buy["slippage_cost"], 37.5, places=4)

        sell = compute_indian_delivery_costs(side="SELL", notional=50_000.0, currency="INR", slippage_bps=7.5)
        self.assertEqual(sell["stamp_duty"], 0.0)
        self.assertAlmostEqual(sell["dp_charge"], 15.93, places=4)

    def test_manual_position_sizing_sector_corr_and_dust_limits(self) -> None:
        dates = _date_seq(70)
        # Steady +1% trend with constant true range = 4.0 -> ATR = 4.0
        candles = [
            {
                "date": dates[i],
                "open": 200.0 + i * 1.5,
                "high": 202.0 + i * 1.5,
                "low": 198.0 + i * 1.5,
                "close": 200.0 + i * 1.5,
                "volume": 500_000,
            }
            for i in range(70)
        ]
        bench = [
            {
                "date": dates[i],
                "open": 20000.0 + i * 40.0,
                "high": 20050.0 + i * 40.0,
                "low": 19950.0 + i * 40.0,
                "close": 20020.0 + i * 40.0,
                "volume": 0,
            }
            for i in range(70)
        ]
        cfg = MoreshwarV21Config()
        res = evaluate_moreshwar_v2_1(
            symbol="TESTA",
            sector="IT Services",
            candles=candles,
            spot_price=candles[-1]["close"],
            tech_signal="BULLISH",
            benchmark_candles=bench,
            decision_date=dates[-1],
            capital=100_000.0,
            current_deployed_capital=0.0,
            open_positions_context=[],
            currency="INR",
            cfg=cfg,
        )
        self.assertTrue(res["accepted"])
        self.assertGreater(res["position_qty"], 0)
        self.assertLessEqual(res["position_qty"] * res["entry_price"], 100_000.0 * (cfg.max_single_stock_pct / 100.0))

        # 2 open same-sector holdings -> 3rd must be blocked
        open_same_sector = [
            {"symbol": "S1", "sector": "IT Services", "entry_price": 100.0, "sl": 95.0, "qty": 10, "recent_closes": []},
            {"symbol": "S2", "sector": "IT Services", "entry_price": 100.0, "sl": 95.0, "qty": 10, "recent_closes": []},
        ]
        res_sec = evaluate_moreshwar_v2_1(
            symbol="TESTB",
            sector="IT Services",
            candles=candles,
            spot_price=candles[-1]["close"],
            tech_signal="BULLISH",
            benchmark_candles=bench,
            decision_date=dates[-1],
            open_positions_context=open_same_sector,
            cfg=cfg,
        )
        self.assertFalse(res_sec["accepted"])
        self.assertEqual(res_sec["rejection_category"], "PORTFOLIO_CAP_OR_CORR")

        # Low-price tiny cap order below ₹1,500 minimum viable notional -> rejected cleanly
        dust_candles = [
            {
                "date": dates[i],
                "open": 10.0 + i * 0.05,
                "high": 10.3 + i * 0.05,
                "low": 9.7 + i * 0.05,
                "close": 10.1 + i * 0.05,
                "volume": 500_000,
            }
            for i in range(70)
        ]
        res_dust = evaluate_moreshwar_v2_1(
            symbol="DUST",
            sector="Energy",
            candles=dust_candles,
            spot_price=dust_candles[-1]["close"],
            tech_signal="BULLISH",
            benchmark_candles=bench,
            decision_date=dates[-1],
            capital=5_000.0,  # 1% NAV = ₹50 -> qty tiny -> below ₹1,500
            cfg=cfg,
        )
        self.assertFalse(res_dust["accepted"])
        self.assertEqual(res_dust["position_qty"], 0)

    def test_manual_gap_stop_zero_vol_deferral_and_metrics_math(self) -> None:
        # Build synthetic 70-bar uptrend with small pullbacks (keeping RSI in 42..70) followed by:
        # - bar 65 triggers entry at bar 66 open
        # - bar 67 zero-volume halt (even if low hits SL, exit must defer on non-final day)
        # - bar 68 massive gap-down open below SL -> fills at gap open - slippage
        dates = _date_seq(70)
        candles = []
        px = 200.0
        for i in range(66):
            delta = -1.5 if (i % 2 == 1) else 2.1
            px = round(px + delta, 2)
            candles.append({"date": dates[i], "open": px - 0.5, "high": px + 1.5, "low": px - 1.5, "close": px, "volume": 300_000})
        # Bar 66: open fill bar near ~220
        candles.append({"date": dates[66], "open": px, "high": px + 2.0, "low": px - 1.5, "close": px + 0.5, "volume": 300_000})
        # Bar 67: zero volume halted gap-down bar (must defer)
        candles.append({"date": dates[67], "open": 170.0, "high": 170.0, "low": 170.0, "close": 170.0, "volume": 0})
        # Bar 68: resumes trading at gap-down open 160.0 < SL -> must mark gap_through_stop=True at open ~160
        candles.append({"date": dates[68], "open": 160.0, "high": 162.0, "low": 158.0, "close": 161.0, "volume": 300_000})
        candles.append({"date": dates[69], "open": 162.0, "high": 164.0, "low": 160.0, "close": 163.0, "volume": 300_000})

        bench = [{"date": dates[i], "open": 20000.0 + i * 35.0, "high": 20040.0 + i * 35.0, "low": 19960.0 + i * 35.0, "close": 20010.0 + i * 35.0, "volume": 0} for i in range(70)]
        universe = {
            "SYNTH": {
                "symbol": "SYNTH",
                "currency": "INR",
                "sector": "IT Services",
                "cap_tier": "Large Cap",
                "candles": candles,
                "missing_fields": [],
                "fundamental_score": {"total": 60, "ratios": {}},
                "porters_score": {"total": 60},
            }
        }
        pre = precompute_causal_features(universe, bench)
        res = simulate_portfolio_v2_1(
            universe,
            bench,
            pre,
            model_name="AuditSynth",
            model_family="M2_1",
            cfg_v21=MoreshwarV21Config(),
        )
        trades = res["trades"]
        self.assertGreaterEqual(len(trades), 1)
        gap_trades = [t for t in trades if t["gap_through_stop"]]
        self.assertTrue(gap_trades, "Expected gap-through-stop fill to trigger on bar 68 open")
        self.assertEqual(res["metrics"]["execution_realism"]["liquidity_delayed_exits"], 1)
        self.assertGreaterEqual(res["metrics"]["execution_realism"]["gap_through_stops"], 1)
        # Gap trade fill must be near 240 open minus slippage, not idealized stop (~285+)
        self.assertLess(gap_trades[-1]["exit_price"], 245.0)

    def test_manual_cagr_sharpe_sortino_pf_maxdd_formulas(self) -> None:
        # 5-day known NAV curve -> verify closed-form metrics
        navs = [
            {"date": "2026-01-01", "nav": 100_000.0, "regime6": "STRONG_BULLISH", "regime4": "BULLISH"},
            {"date": "2026-01-02", "nav": 110_000.0, "regime6": "STRONG_BULLISH", "regime4": "BULLISH"},
            {"date": "2026-01-03", "nav": 99_000.0, "regime6": "STRONG_BULLISH", "regime4": "BULLISH"},
            {"date": "2026-01-04", "nav": 105_000.0, "regime6": "STRONG_BULLISH", "regime4": "BULLISH"},
            {"date": "2026-01-05", "nav": 108_000.0, "regime6": "STRONG_BULLISH", "regime4": "BULLISH"},
        ]
        snaps = [{"date": n["date"], "open_positions": 1, "gross_exposure_pct": 50.0, "sl_risk_pct": 1.0, "max_sector_pct": 50.0, "avg_pairwise_corr": 0.0, "daily_ret_pct": 0.0, "bench_ret_pct": 0.0} for n in navs]
        trades = [
            {"symbol": "A", "sector": "IT", "signal_regime6": "STRONG_BULLISH", "holding_days": 2, "pnl_amount": 10_000.0, "net_return_pct": 10.0, "exit_reason": "TARGET_HIT"},
            {"symbol": "B", "sector": "IT", "signal_regime6": "STRONG_BULLISH", "holding_days": 2, "pnl_amount": -2_000.0, "net_return_pct": -2.0, "exit_reason": "STOP_LOSS"},
        ]
        rep = _finalize_v21_report(
            model_name="ManualCheck",
            model_family="M2_1",
            fund_mode="historical_unavailable",
            closed_trades=trades,
            daily_navs=navs,
            daily_risk_snapshots=snaps,
            initial_capital=100_000.0,
            cost_totals={"total_cost": 500.0},
            exec_stats={"gap_through_stops": 0, "rejection_categories": {}},
            rejected_log=[],
        )
        m = rep["metrics"]
        self.assertAlmostEqual(m["total_return_pct"], 8.0, places=2)
        expected_cagr = (((108_000.0 / 100_000.0) ** (252.0 / 5.0)) - 1.0) * 100.0
        self.assertAlmostEqual(m["cagr_pct"], round(expected_cagr, 2), places=2)
        self.assertAlmostEqual(m["max_drawdown_pct"], 10.0, places=2)  # 110k -> 99k = -10.0%
        self.assertAlmostEqual(m["profit_factor"], 5.0, places=2)  # 10,000 / 2,000 = 5.0

        # Recompute Sharpe & Sortino manually
        rets = [0.0, 0.10, -0.10, (105_000.0 - 99_000.0) / 99_000.0, (108_000.0 - 105_000.0) / 105_000.0]
        rf_d = RISK_FREE_ANNUAL / 252.0
        ex = [r - rf_d for r in rets]
        m_ex = sum(ex) / len(ex)
        sd_ex = math.sqrt(sum((x - m_ex) ** 2 for x in ex) / (len(ex) - 1))
        dn_ex = math.sqrt(sum(min(0.0, x) ** 2 for x in ex) / len(ex))
        self.assertAlmostEqual(m["sharpe_ratio"], round((m_ex / sd_ex) * math.sqrt(252.0), 2), places=2)
        self.assertAlmostEqual(m["sortino_ratio"], round((m_ex / dn_ex) * math.sqrt(252.0), 2), places=2)

    def test_monte_carlo_reproducibility_and_limitations(self) -> None:
        trades = [{"pnl_amount": 800.0 if i % 2 == 0 else -500.0} for i in range(40)]
        rets = [0.004 if i % 2 == 0 else -0.0025 for i in range(60)]
        mc1 = run_monte_carlo_suite(trades, rets, n_sims=1000)
        mc2 = run_monte_carlo_suite(trades, rets, n_sims=1000)
        self.assertEqual(mc1, mc2)
        for k in ("trade_bootstrap", "daily_block_bootstrap"):
            self.assertEqual(mc1[k]["simulations"], 1000)
            self.assertLessEqual(mc1[k]["p05_return_pct"], mc1[k]["median_return_pct"])
            self.assertLessEqual(mc1[k]["median_return_pct"], mc1[k]["p95_return_pct"])
            self.assertTrue(len(mc1[k]["limitation"]) > 15)

    def test_api_integration_and_secret_masking(self) -> None:
        # Inject clean test state into server module (zero external network calls)
        orig_port = dict(server.PORTFOLIO_STATE)
        orig_anal = dict(server.ANALYSIS_CACHE)
        orig_pipe = dict(server.PIPELINE_CACHE)
        orig_crit = dict(server.GEMINI_CRITIQUES)
        orig_set = server.runner.settings
        orig_v21 = dict(server.MORESHWAR_V2_1_RESEARCH)
        orig_prov = dict(server.PROVENANCE_CACHE)
        orig_bt = dict(server.BACKTEST_CACHE)
        orig_aud = dict(server.AUDIT_LEDGER_CACHE)
        orig_bench = dict(server.BENCHMARK_CACHE)

        try:
            server.runner.settings = FinsistantSettings(
                kite_api_key="test_secret_key_123",
                kite_api_secret="test_secret_456",
                gemini_api_key="test_gem_789",
            )
            server.BENCHMARK_CACHE.clear()
            server.BENCHMARK_CACHE.update({
                "symbol": "^NSEI",
                "source": "Synthetic Offline Benchmark",
                "candles": [
                    {"date": f"2026-11-{10 + i:02d}", "open": 24000.0 + i * 20, "high": 24050.0 + i * 20, "low": 23950.0 + i * 20, "close": 24010.0 + i * 20, "volume": 0}
                    for i in range(9)
                ],
            })
            server.PROVENANCE_CACHE.clear()
            server.BACKTEST_CACHE.clear()
            server.AUDIT_LEDGER_CACHE.clear()
            server.ANALYSIS_CACHE.clear()
            server.ANALYSIS_CACHE["RELIANCE"] = {
                "symbol": "RELIANCE",
                "name": "Reliance Industries",
                "exchange": "NSE",
                "currency": "INR",
                "sector": "Energy",
                "asset_type": "EQUITY",
                "price": 1300.0,
                "change_pct": 0.5,
                "candles": [{"date": "2026-11-18", "open": 1295, "high": 1305, "low": 1290, "close": 1300, "volume": 1000000}],
                "fundamental_score": {"total": 50, "ratios": {"sales_growth": 10, "profit_growth": 10}, "industry": "Energy"},
                "porters_score": {"total": 60},
                "moreshwar_levels": {"entry_price": 1300.0, "target": 1440.0, "stop_loss": 1220.0},
                "decision_matrix": {"action": "HOLD", "rationale": "Test"},
                "projected_30d_target": 1380.0,
                "regression_curve": "Upward Curve",
                "provenance": {"spot_source": "Yahoo Live", "is_stale": False},
            }
            server.PORTFOLIO_STATE.clear()
            server.PORTFOLIO_STATE["RELIANCE"] = {
                "symbol": "RELIANCE",
                "name": "Reliance Industries",
                "exchange": "NSE",
                "currency": "INR",
                "sector": "Energy",
                "asset_type": "EQUITY",
                "qty": 0,
                "avg_price": 0.0,
                "ltp": 1300.0,
                "change_pct": 0.5,
                "fundamental_score": 50,
                "porters_score": 60,
                "decision": "HOLD",
                "moreshwar": {"entry_price": 1300.0, "target": 1440.0, "stop_loss": 1220.0},
            }
            server.PIPELINE_CACHE.clear()
            server.PIPELINE_CACHE["RELIANCE"] = {"final_action": "HOLD", "conviction": 65.0, "agents": []}
            server.GEMINI_CRITIQUES["RELIANCE"] = {"symbol": "RELIANCE", "status": "unavailable", "stance": "HOLD"}
            v21_path = Path(__file__).resolve().parent.parent / "data" / "moreshwar_v2_1_experiment_report.json"
            server.MORESHWAR_V2_1_RESEARCH.clear()
            server.MORESHWAR_V2_1_RESEARCH.update(json.loads(v21_path.read_text(encoding="utf-8")))

            httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.FinsistantV2HTTPHandler)
            port = httpd.server_address[1]
            th = threading.Thread(target=httpd.serve_forever, daemon=True)
            th.start()

            def _get(ep: str) -> dict:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}{ep}", timeout=15) as r:
                    self.assertEqual(r.status, 200)
                    return json.loads(r.read().decode("utf-8"))

            try:
                h = _get("/api/health")
                self.assertEqual(h["pipeline_status"], "HEALTHY")

                st = _get("/api/state")
                self.assertEqual(st["config"]["kite_api_key"], "***")
                self.assertEqual(st["config"]["kite_api_secret"], "***")
                self.assertEqual(st["config"]["gemini_api_key"], "***")
                self.assertFalse(st["moreshwar_v2_1_research"]["promoted_to_production"])

                rv21 = _get("/api/research/v2_1")
                self.assertEqual(rv21["research_version"], "2.1.0")
                self.assertFalse(rv21["promoted_to_production"])

                prov = _get("/api/provenance")
                self.assertIn("taxonomy", prov)
                self.assertIn("assets", prov)
                self.assertIn("RELIANCE", prov["assets"])

                bt = _get("/api/backtest")
                self.assertIn("tracks", bt)
                self.assertIn("benchmark", bt)

                aud = _get("/api/audit")
                self.assertIn("records", aud)
                self.assertIn("engine_versions", aud)

                self.assertIn("kite_auth", _get("/api/kite/status"))
                self.assertEqual(_get("/api/analyze/RELIANCE")["active_analysis"]["symbol"], "RELIANCE")

                with self.assertRaises(urllib.error.HTTPError) as cm:
                    urllib.request.urlopen(f"http://127.0.0.1:{port}/api/does_not_exist", timeout=5)
                self.assertEqual(cm.exception.code, 404)
            finally:
                httpd.shutdown()
                httpd.server_close()
        finally:
            server.PORTFOLIO_STATE.clear()
            server.PORTFOLIO_STATE.update(orig_port)
            server.ANALYSIS_CACHE.clear()
            server.ANALYSIS_CACHE.update(orig_anal)
            server.PIPELINE_CACHE.clear()
            server.PIPELINE_CACHE.update(orig_pipe)
            server.GEMINI_CRITIQUES.clear()
            server.GEMINI_CRITIQUES.update(orig_crit)
            server.runner.settings = orig_set
            server.MORESHWAR_V2_1_RESEARCH.clear()
            server.MORESHWAR_V2_1_RESEARCH.update(orig_v21)
            server.PROVENANCE_CACHE.clear()
            server.PROVENANCE_CACHE.update(orig_prov)
            server.BACKTEST_CACHE.clear()
            server.BACKTEST_CACHE.update(orig_bt)
            server.AUDIT_LEDGER_CACHE.clear()
            server.AUDIT_LEDGER_CACHE.update(orig_aud)
            server.BENCHMARK_CACHE.clear()
            server.BENCHMARK_CACHE.update(orig_bench)


if __name__ == "__main__":
    unittest.main()
