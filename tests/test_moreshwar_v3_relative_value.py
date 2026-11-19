"""Automated Validation Tests for Moreshwar 3.0 Prompt 6 (Relative-Value, Pairs, Market-Neutral & Futures-Hedged Research)."""

from __future__ import annotations

import json
import unittest

import numpy as np

from finsistant_v2.m3.relative_value_hedged import (
    _engle_granger_adf_1lag,
    discover_cointegrated_pairs_in_sample,
    simulate_cointegrated_pairs_strategy,
    simulate_nifty_futures_hedged_overlay,
)
from finsistant_v2.m3.schema import DEFAULT_M3_REPORTS_DIR
from tests.test_moreshwar_v3_strategies_and_regime import _make_synthetic_panel


class TestMoreshwarV3RelativeValueHedged(unittest.TestCase):
    def test_cointegration_adf_distinguishes_mean_reversion_from_correlated_random_walk(self) -> None:
        rng = np.random.default_rng(123)
        n = 500
        # 1. Stationary Ornstein-Uhlenbeck mean-reverting spread (phi = 0.88 -> half-life ~ 5.4 days)
        ou = np.zeros(n, dtype=np.float64)
        for t in range(1, n):
            ou[t] = 0.88 * ou[t - 1] + rng.normal(0.0, 0.02)
        t_ou, p_ou, hl_ou = _engle_granger_adf_1lag(ou)
        self.assertLess(t_ou, -3.12, f"Expected strong ADF rejection for OU spread, got {t_ou}")
        self.assertLess(p_ou, 0.05)
        self.assertTrue(3.0 <= hl_ou <= 12.0, f"Unexpected half-life {hl_ou}")

        # 2. High-correlation non-cointegrated random walk residual (unit root)
        rw = np.cumsum(rng.normal(0.0005, 0.015, size=n))
        t_rw, p_rw, _ = _engle_granger_adf_1lag(rw)
        self.assertGreater(t_rw, -2.50, f"Unit-root random walk should not pass cointegration: {t_rw}")
        self.assertGreater(p_rw, 0.05)

    def test_in_sample_pair_discovery_has_zero_oos_lookahead(self) -> None:
        panel = _make_synthetic_panel(t_len=360, n_len=16, seed=42)
        # Inject a true cointegrated pair into cols 0 & 4 (both IT sector)
        rng = np.random.default_rng(42)
        ou = np.zeros(360, dtype=np.float32)
        for t in range(1, 360):
            ou[t] = 0.85 * ou[t - 1] + float(rng.normal(0.0, 0.012))
        panel["adj_close"][:, 4] = panel["adj_close"][:, 0] * np.exp(ou)
        panel["adj_open"][:, 4] = panel["adj_open"][:, 0] * np.exp(ou)

        cutoff_dt = str(panel["dates"][250])
        disc_base = discover_cointegrated_pairs_in_sample(
            panel, date_start=str(panel["dates"][0]), is_cutoff=cutoff_dt
        )
        self.assertGreaterEqual(disc_base["total_cointegrated_found"], 1)

        # Corrupt all OOS bars after index 250 by 10x -> IS pair discovery must be 100% identical!
        panel_shock = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in panel.items()}
        panel_shock["adj_close"][251:, :] *= 10.0
        disc_shock = discover_cointegrated_pairs_in_sample(
            panel_shock, date_start=str(panel["dates"][0]), is_cutoff=cutoff_dt
        )
        self.assertEqual(
            [(p.pair_id, p.is_adf_t_stat, p.is_half_life_days) for p in disc_base["selected_pairs"]],
            [(p.pair_id, p.is_adf_t_stat, p.is_half_life_days) for p in disc_shock["selected_pairs"]],
        )

        # Simulate pairs trading and verify SLB borrow & transaction cost accounting
        sim = simulate_cointegrated_pairs_strategy(
            panel=panel,
            pairs=disc_base["selected_pairs"],
            date_start=str(panel["dates"][0]),
            date_end=str(panel["dates"][-1]),
        )
        self.assertEqual(len(sim["daily_rets"]), len(panel["dates"]))
        self.assertGreaterEqual(sim["total_cost_drag_pct"], 0.0)
        self.assertGreaterEqual(sim["total_borrow_drag_pct"], 0.0)

    def test_nifty_futures_lot_quantization_blocks_100k_and_hedges_25l(self) -> None:
        panel = _make_synthetic_panel(t_len=200, n_len=12, seed=7)
        navs = [
            {
                "date": str(panel["dates"][i]),
                "nav": 100_000.0,
                "deployed_mkt": 85_000.0,
                "ret": -0.004,
            }
            for i in range(200)
        ]
        regimes = ["STRESS_BEAR"] * 200

        # At Rs 100,000 capital, 1 Nifty futures lot (~Rs 9L-18L) cannot be opened without >1.25x over-hedging
        res_100k = simulate_nifty_futures_hedged_overlay(
            panel=panel, long_daily_navs=navs, regime_labels=regimes, capital_base_inr=100_000.0, lot_quantized=True
        )
        self.assertEqual(res_100k["roll_events"], 0)
        self.assertGreater(res_100k["lot_granularity_blocked_days"], 150)

        # At Rs 25,00,000 capital, lot-quantized Nifty futures opens and rolls monthly
        res_25l = simulate_nifty_futures_hedged_overlay(
            panel=panel, long_daily_navs=navs, regime_labels=regimes, capital_base_inr=2_500_000.0, lot_quantized=True
        )
        self.assertGreater(res_25l["roll_events"], 5)
        self.assertGreater(res_25l["total_futures_cost_drag_pct"], 0.0)

    def test_prompt6_deliverables_exist_and_isolated_from_long_portfolio(self) -> None:
        rep_dir = DEFAULT_M3_REPORTS_DIR / "relative_value_hedged"
        rep_p = rep_dir / "relative_value_hedged_report.json"
        pair_p = rep_dir / "pair_selection_cointegration_registry.json"
        md_p = rep_dir / "RELATIVE_VALUE_AND_HEDGED_REPORT.md"

        self.assertTrue(rep_p.exists(), "Missing relative_value_hedged_report.json")
        self.assertTrue(pair_p.exists(), "Missing pair_selection_cointegration_registry.json")
        self.assertTrue(md_p.exists(), "Missing RELATIVE_VALUE_AND_HEDGED_REPORT.md")

        data = json.loads(rep_p.read_text(encoding="utf-8"))
        self.assertTrue(data["isolated_research_branch"])
        self.assertFalse(data["mixed_into_long_only_portfolio"])
        self.assertGreater(len(data["pair_discovery_summary"]["selected_cointegrated_pairs"]), 0)
        self.assertGreater(len(data["pair_discovery_summary"]["high_corr_non_cointegrated_pairs"]), 0)
        self.assertIn("STRAT_RV1_COINT_PAIRS", data["branches"])
        self.assertIn("STRAT_RV3_MARKET_NEUTRAL_LS", data["branches"])
        self.assertIn("STRAT_RV4A_NIFTY_FUT_HEDGED_25L_LOT", data["branches"])


if __name__ == "__main__":
    unittest.main()
