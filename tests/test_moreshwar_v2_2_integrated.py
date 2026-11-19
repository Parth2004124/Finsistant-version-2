"""Unit & Integration Tests for Moreshwar 2.2 Prompt 3 (Integrated Optimization & Validation)."""

from __future__ import annotations

from datetime import date, timedelta
import http.client
import json
from pathlib import Path
import threading
import unittest

from finsistant_v2.backtest_engine_v2_2 import precompute_v22_universe
from finsistant_v2.backtest_engine_v2_2_integrated import (
    REPORTS_DIR,
    compute_statistical_confidence,
    simulate_integrated_v2_2,
)
from finsistant_v2.moreshwar_v2_2 import DeliveryCostConfig
from finsistant_v2.moreshwar_v2_2_integrated import (
    IntegratedM22Config,
    PortfolioRiskAllocator,
    SelectionAndEntryEngine,
)


def _dates(n: int, start: date = date(2023, 1, 2)) -> list[str]:
    out = []
    d = start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d.isoformat())
        d += timedelta(days=1)
    return out


def _make_synthetic_universe(n_bars: int = 260) -> tuple[dict, list[dict]]:
    dates = _dates(n_bars)
    bench = []
    b_px = 18000.0
    for dt in dates:
        b_px *= 1.0008
        bench.append({
            "date": dt,
            "open": round(b_px * 0.999, 2),
            "high": round(b_px * 1.004, 2),
            "low": round(b_px * 0.997, 2),
            "close": round(b_px, 2),
            "volume": 100_000_000,
        })

    def _stock_series(sym: str, sector: str, drift: float, pullback_at: int | None = None, gap_drop_at: int | None = None) -> dict:
        candles = []
        px = 500.0
        for i, dt in enumerate(dates):
            px *= (1.0 + drift)
            o_px = px * 0.998
            h_px = px * 1.012
            l_px = px * 0.992
            c_px = px
            if pullback_at is not None and i == pullback_at:
                # Pullback into rising SMA20 with healthy close > open bounce
                o_px = px * 0.980
                l_px = px * 0.975
                c_px = px * 0.990
                h_px = px * 0.995
            if gap_drop_at is not None and i == gap_drop_at:
                # Sharp overnight gap down well below stop-loss
                o_px = px * 0.85
                l_px = px * 0.83
                c_px = px * 0.86
                h_px = px * 0.87
                px = c_px
            candles.append({
                "date": dt,
                "open": round(o_px, 2),
                "high": round(h_px, 2),
                "low": round(l_px, 2),
                "close": round(c_px, 2),
                "volume": 800_000,
            })
        return {
            "symbol": sym,
            "currency": "INR",
            "sector": sector,
            "candles": candles,
            "fundamentals": {"pe_ratio": 22.0, "roe": 18.0, "roce": 20.0, "debt_to_equity": 0.25},
        }

    uni = {
        "ALPHA": _stock_series("ALPHA", "IT", 0.0022, pullback_at=240, gap_drop_at=248),
        "BETA": _stock_series("BETA", "Banking", 0.0019, pullback_at=240),
        "GAMMA": _stock_series("GAMMA", "Auto", 0.0017, pullback_at=241),
    }
    return uni, bench


class TestMoreshwarV22Integrated(unittest.TestCase):
    def test_selection_engine_zero_lookahead(self) -> None:
        uni1, bench1 = _make_synthetic_universe(260)
        uni2 = json.loads(json.dumps(uni1))
        # Corrupt all future bars after index 240 in uni2
        for sym in uni2:
            for bar in uni2[sym]["candles"][241:]:
                bar["open"] = 10.0
                bar["high"] = 12.0
                bar["low"] = 8.0
                bar["close"] = 9.0

        pre1 = precompute_v22_universe(uni1, bench1, currency_filter="INR")
        pre2 = precompute_v22_universe(uni2, bench1, currency_filter="INR")
        dt = bench1[240]["date"]
        cfg = IntegratedM22Config(max_open_positions=8)
        tariff = DeliveryCostConfig()

        cands1 = SelectionAndEntryEngine.evaluate_universe_day(
            valid_syms=sorted(pre1["features"].keys()),
            open_positions={},
            cooldown_until_idx={},
            d_idx=240,
            dt=dt,
            features=pre1["features"],
            uni=pre1["universe"],
            regime6_obj=pre1["regime6_by_date"][dt],
            cfg=cfg,
            tariff=tariff,
        )
        cands2 = SelectionAndEntryEngine.evaluate_universe_day(
            valid_syms=sorted(pre2["features"].keys()),
            open_positions={},
            cooldown_until_idx={},
            d_idx=240,
            dt=dt,
            features=pre2["features"],
            uni=pre2["universe"],
            regime6_obj=pre2["regime6_by_date"][dt],
            cfg=cfg,
            tariff=tariff,
        )
        self.assertEqual(
            [(c["symbol"], c["rank_score"], c["spot"], c["sl_ratio"]) for c in cands1],
            [(c["symbol"], c["rank_score"], c["spot"], c["sl_ratio"]) for c in cands2],
        )

    def test_portfolio_risk_allocator_sizing_regime_and_dd_throttle(self) -> None:
        base_closes = [100.0 + i * 0.5 for i in range(61)]
        low_vol_cand = {
            "symbol": "LOWVOL",
            "sector": "IT",
            "spot": 500.0,
            "sl_ratio": 0.04,
            "atr_pct": 1.4,
            "recent_closes": base_closes,
            "rank_score": 82.0,
            "trend_strength": 75.0,
            "market_regime": "STRONG_BULLISH",
            "archetype": "PULL_CONT",
            "signal_date": "2024-06-01",
            "atr_entry": 7.0,
            "slippage_bps": 7.5,
        }
        high_vol_cand = {
            **low_vol_cand,
            "symbol": "HIVOL",
            "sector": "Auto",
            "sl_ratio": 0.08,
            "atr_pct": 3.4,
            "recent_closes": [100.0 + ((i % 2) * 3.0) + i * 0.2 for i in range(61)],
            "rank_score": 80.0,
        }

        # Dynamic regime gate must block entries in STRONG_BEARISH / HIGH_VOLATILITY
        cfg_reg = IntegratedM22Config(sizing_mode="RISK_BUDGET", max_open_positions=6, use_dynamic_regime_exposure=True)
        blocked = PortfolioRiskAllocator.allocate(
            candidates=[low_vol_cand, high_vol_cand],
            open_positions={},
            cash=100_000.0,
            nav=100_000.0,
            current_dd_pct=0.0,
            dd_throttled=False,
            regime6_name="STRONG_BEARISH",
            cfg=cfg_reg,
        )
        self.assertEqual(blocked, [])

        # INV_VOL allocates higher notional to lower-ATR stock than high-ATR stock
        cfg_inv = IntegratedM22Config(sizing_mode="INV_VOL", max_open_positions=6, max_pairwise_corr=0.99)
        inv_orders = PortfolioRiskAllocator.allocate(
            candidates=[low_vol_cand, high_vol_cand],
            open_positions={},
            cash=100_000.0,
            nav=100_000.0,
            current_dd_pct=0.0,
            dd_throttled=False,
            regime6_name="STRONG_BULLISH",
            cfg=cfg_inv,
        )
        self.assertEqual(len(inv_orders), 2)
        self.assertGreater(inv_orders[0]["req_qty"] * inv_orders[0]["spot"], inv_orders[1]["req_qty"] * inv_orders[1]["spot"])

        # DD throttle cuts allocation size at >= dd_throttle_level2_pct (12%)
        cfg_dd = IntegratedM22Config(sizing_mode="RISK_BUDGET", max_open_positions=8, use_drawdown_throttle=True)
        normal_ord = PortfolioRiskAllocator.allocate(
            candidates=[low_vol_cand], open_positions={}, cash=100_000.0, nav=100_000.0,
            current_dd_pct=0.0, dd_throttled=False, regime6_name="STRONG_BULLISH", cfg=cfg_dd,
        )
        throttled_ord = PortfolioRiskAllocator.allocate(
            candidates=[low_vol_cand], open_positions={}, cash=100_000.0, nav=100_000.0,
            current_dd_pct=13.0, dd_throttled=True, regime6_name="STRONG_BULLISH", cfg=cfg_dd,
        )
        self.assertGreater(normal_ord[0]["req_qty"], throttled_ord[0]["req_qty"])

        # Sector cap limits at most max_sector_positions=2 per sector
        same_sec = [{**low_vol_cand, "symbol": f"IT{i}", "sector": "IT"} for i in range(4)]
        sec_ord = PortfolioRiskAllocator.allocate(
            candidates=same_sec, open_positions={}, cash=100_000.0, nav=100_000.0,
            current_dd_pct=0.0, dd_throttled=False, regime6_name="STRONG_BULLISH",
            cfg=IntegratedM22Config(sizing_mode="EQUAL_WEIGHT", max_open_positions=6, max_sector_positions=2, max_pairwise_corr=0.99),
        )
        self.assertLessEqual(len(sec_ord), 2)

    def test_delayed_entry_gap_shock_and_cost_reconciliation(self) -> None:
        uni, bench = _make_synthetic_universe(260)
        pre = precompute_v22_universe(uni, bench, currency_filter="INR")
        tariff = DeliveryCostConfig()

        base_cfg = IntegratedM22Config(max_open_positions=6, require_pit_filters=False, reentry_cooldown_days=0)
        r_base = simulate_integrated_v2_2(pre, experiment_id="T_BASE", label="base", cfg=base_cfg, tariff=tariff)
        r_cost15 = simulate_integrated_v2_2(pre, experiment_id="T_15X", label="1.5x", cfg=base_cfg, tariff=tariff, cost_multiplier=1.5)
        r_delay2 = simulate_integrated_v2_2(
            pre,
            experiment_id="T_D2",
            label="t+2",
            cfg=IntegratedM22Config(max_open_positions=6, require_pit_filters=False, reentry_cooldown_days=0, entry_delay_bars=2),
            tariff=tariff,
        )
        r_gap = simulate_integrated_v2_2(
            pre,
            experiment_id="T_GAP",
            label="gap",
            cfg=IntegratedM22Config(max_open_positions=6, require_pit_filters=False, reentry_cooldown_days=0, gap_shock_extra_pct=1.0),
            tariff=tariff,
        )

        # Accounting identity holds within sub-rupee rounding on both 1.0x and 1.5x stressed costs
        self.assertLessEqual(abs(r_base["metrics"]["reconciliation_diff"]), 1.0)
        self.assertLessEqual(abs(r_cost15["metrics"]["reconciliation_diff"]), 1.0)
        if r_base["trades"]:
            self.assertGreater(r_cost15["metrics"]["total_cost"], r_base["metrics"]["total_cost"])
            self.assertLessEqual(r_cost15["metrics"]["net_pnl_total"], r_base["metrics"]["net_pnl_total"])
            # Delayed t+2 entry occurs at strictly later entry_date than t+1
            b0 = r_base["trades"][0]
            d0 = next((t for t in r_delay2["trades"] if t["symbol"] == b0["symbol"] and t["signal_date"] == b0["signal_date"]), None)
            if d0:
                self.assertGreater(d0["entry_date"], b0["entry_date"])
        # Gap shock increases loss or equals on any gap-through stops
        self.assertLessEqual(r_gap["metrics"]["total_return_pct"], r_base["metrics"]["total_return_pct"] + 0.01)

    def test_bootstrap_determinism_and_ordering(self) -> None:
        trades = [{"net_return_pct": (3.2 if i % 2 == 0 else -1.9), "pnl_amount": (960.0 if i % 2 == 0 else -570.0)} for i in range(40)]
        daily_rets = [0.0012 if i % 3 != 0 else -0.0007 for i in range(120)]
        sc1 = compute_statistical_confidence(trades, daily_rets, n_sims=200)
        sc2 = compute_statistical_confidence(trades, daily_rets, n_sims=200)
        self.assertEqual(sc1, sc2)
        for key in ("trade_bootstrap", "block_bootstrap_10d", "block_bootstrap_20d"):
            p05, p50, p95 = sc1[key]["return_p05_p50_p95"]
            self.assertLessEqual(p05, p50)
            self.assertLessEqual(p50, p95)

    def test_prompt3_reports_integrity_and_unpromoted_status(self) -> None:
        json_path = REPORTS_DIR / "integrated_final_report.json"
        reg_path = REPORTS_DIR / "experiment_registry.json"
        md_path = REPORTS_DIR / "MORESHWAR_2_2_FINAL_REPORT.md"
        eq_svg = REPORTS_DIR / "equity_curve_comparison.svg"
        dd_svg = REPORTS_DIR / "drawdown_comparison.svg"

        for p in (json_path, reg_path, md_path, eq_svg, dd_svg):
            self.assertTrue(p.exists(), f"Missing deliverable artifact: {p}")

        data = json.loads(json_path.read_text(encoding="utf-8"))
        self.assertEqual(data["research_version"], "2.2.0-integrated-final")
        self.assertFalse(data["promoted_to_production"])
        self.assertIn(data["decision"], ("NO-GO", "CONDITIONAL GO", "GO: Research candidate"))
        self.assertEqual(len(data["walk_forward_4fold"]), 4)
        self.assertGreaterEqual(len(data["stress_tests"]), 10)
        for row in data["optimization_registry_is_only"]:
            self.assertLessEqual(abs(row["reconciliation_diff"]), 1.0)
        for row in data["stress_tests"]:
            self.assertLessEqual(abs(row.get("reconciliation_diff", 0.0)), 1.0)

        md_txt = md_path.read_text(encoding="utf-8")
        self.assertIn(data["decision"], md_txt)
        self.assertIn(data["selected_config_id"], md_txt)

    def test_server_v2_2_research_endpoint_and_contract_preservation(self) -> None:
        from http.server import ThreadingHTTPServer
        import server

        server.MORESHWAR_V2_1_RESEARCH.clear()
        server.MORESHWAR_V2_2_RESEARCH.clear()
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.FinsistantV2HTTPHandler)
        port = httpd.server_address[1]
        t = threading.Thread(target=httpd.serve_forever, daemon=True)
        t.start()
        conn = None
        try:
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
            conn.request("GET", "/api/research/v2_2")
            r22 = conn.getresponse()
            self.assertEqual(r22.status, 200)
            body22 = json.loads(r22.read().decode("utf-8"))
            self.assertEqual(body22["research_version"], "2.2.0-integrated-final")
            self.assertFalse(body22["promoted_to_production"])

            # Verify frozen v2_1 endpoint and full state contract are preserved
            conn.request("GET", "/api/research/v2_1")
            r21 = conn.getresponse()
            self.assertEqual(r21.status, 200)
            body21 = json.loads(r21.read().decode("utf-8"))
            self.assertIn("models", body21)

            state_body = server._build_full_state_payload("RELIANCE")
            self.assertIn("moreshwar_v2_1_research", state_body)
            self.assertIn("moreshwar_v2_2_research", state_body)
            self.assertIn("assets", state_body)
        finally:
            if conn is not None:
                conn.close()
            httpd.shutdown()
            httpd.server_close()


if __name__ == "__main__":
    unittest.main()
