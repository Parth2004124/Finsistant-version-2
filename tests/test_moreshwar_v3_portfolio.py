"""Automated Validation Tests for Moreshwar 3.0 Prompt 5 (Portfolio Construction & Capital Allocation Engine)."""

from __future__ import annotations

import json
import unittest

import numpy as np

from finsistant_v2.m3.portfolio_construction import (
    PortfolioConstructionConfig,
    _causal_pairwise_corr_60d,
    _drawdown_governor_multiplier,
    compute_euler_risk_contributions,
    simulate_shared_capital_portfolio_m3,
)
from finsistant_v2.m3.regime_intelligence import classify_rule_based_9state, compute_causal_market_features
from finsistant_v2.m3.schema import DEFAULT_M3_REPORTS_DIR
from finsistant_v2.m3.strategy_families import prepare_strategy_signal_workspace
from tests.test_moreshwar_v3_strategies_and_regime import _make_synthetic_panel


class TestMoreshwarV3PortfolioConstruction(unittest.TestCase):
    def test_all_five_sizing_methods_and_shared_capital_invariants(self) -> None:
        panel = _make_synthetic_panel(t_len=310, n_len=24)
        ws = prepare_strategy_signal_workspace(panel)
        feats = compute_causal_market_features(panel)
        _, rule5 = classify_rule_based_9state(feats, confirm_days=3)

        for sm in (
            "EQUAL_WEIGHT",
            "VOLATILITY_ADJUSTED",
            "RISK_BUDGETED",
            "MAX_LOSS_BASED",
            "CONFIDENCE_VALIDATED",
        ):
            cfg = PortfolioConstructionConfig(
                portfolio_id=f"TEST_{sm}",
                portfolio_name=f"Test {sm}",
                sizing_method=sm,
                max_open_positions=8,
                max_sector_positions=2,
                settlement_lag_days=1,
            )
            res = simulate_shared_capital_portfolio_m3(
                panel=panel,
                ws=ws,
                cfg=cfg,
                regime_labels=rule5,
                date_start=str(panel["dates"][252]),
                date_end=str(panel["dates"][-1]),
            )
            m = res["metrics"]
            self.assertEqual(float(m["reconciliation_diff"]), 0.0, f"Reconciliation diff non-zero for {sm}")
            for d in res["daily_navs"]:
                # Never double-allocate capital or use leverage (> 1.0x)
                self.assertGreaterEqual(d["settled_cash"], -1e-4)
                self.assertGreaterEqual(d["unsettled_cash"], -1e-4)
                self.assertLessEqual(d["deployed_mkt"], d["nav"] * 1.0001)
                self.assertLessEqual(d["open_positions"], cfg.max_open_positions)

    def test_euler_risk_contributions_and_correlation_gate(self) -> None:
        rng = np.random.default_rng(99)
        daily_ret = rng.normal(0.0005, 0.015, size=(120, 6))
        # Make col 1 almost identical to col 0 (correlation ~ 0.99)
        daily_ret[:, 1] = daily_ret[:, 0] * 0.98 + rng.normal(0.0, 0.001, size=120)
        corr_01 = _causal_pairwise_corr_60d(daily_ret, 100, 0, 1)
        corr_02 = _causal_pairwise_corr_60d(daily_ret, 100, 0, 2)
        self.assertGreater(corr_01, 0.90)
        self.assertLess(abs(corr_02), 0.45)

        sectors = np.array(["IT", "IT", "Banking", "Pharma", "Auto", "Energy"], dtype=object)
        rc = compute_euler_risk_contributions(
            daily_ret=daily_ret,
            ti=100,
            cols=[0, 2, 3, 4],
            weights=np.array([0.25, 0.25, 0.25, 0.25]),
            sectors=sectors,
            strategies=["STRAT_A", "STRAT_B", "STRAT_D", "STRAT_E"],
        )
        self.assertGreater(rc["ex_ante_ann_vol_pct"], 0.0)
        self.assertGreaterEqual(rc["diversification_ratio"], 1.0)
        self.assertAlmostEqual(sum(rc["stock_rc_pct"].values()), 100.0, delta=0.2)
        self.assertAlmostEqual(sum(rc["sector_rc_pct"].values()), 100.0, delta=0.2)
        self.assertAlmostEqual(sum(rc["strategy_rc_pct"].values()), 100.0, delta=0.2)

        # Drawdown governor monotonically reduces exposure ceiling without ever exceeding 1.0
        self.assertEqual(_drawdown_governor_multiplier(2.0, True), 1.0)
        self.assertEqual(_drawdown_governor_multiplier(7.5, True), 0.82)
        self.assertEqual(_drawdown_governor_multiplier(11.0, True), 0.65)
        self.assertEqual(_drawdown_governor_multiplier(16.0, True), 0.45)

    def test_prompt5_deliverables_exist_and_valid(self) -> None:
        rep_dir = DEFAULT_M3_REPORTS_DIR / "portfolio_construction"
        json_p = rep_dir / "portfolio_construction_report.json"
        util_p = rep_dir / "capital_utilisation_and_bottlenecks.json"
        md_p = rep_dir / "PORTFOLIO_CONSTRUCTION_REPORT.md"

        self.assertTrue(json_p.exists(), "Missing portfolio_construction_report.json")
        self.assertTrue(util_p.exists(), "Missing capital_utilisation_and_bottlenecks.json")
        self.assertTrue(md_p.exists(), "Missing PORTFOLIO_CONSTRUCTION_REPORT.md")

        data = json.loads(json_p.read_text(encoding="utf-8"))
        self.assertEqual(len(data["sizing_comparison"]), 5)
        for sm, obj in data["sizing_comparison"].items():
            self.assertEqual(float(obj["full_metrics"]["reconciliation_diff"]), 0.0, sm)
        self.assertIn("correlation_spike_stress", data["stress_tests"])
        self.assertIn("sector_crash_stress", data["stress_tests"])
        self.assertIn("consecutive_losses_stress", data["stress_tests"])
        self.assertIn("liquidity_deterioration_oos_stress", data["stress_tests"])


if __name__ == "__main__":
    unittest.main()
