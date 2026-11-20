"""Automated Validation Tests for Moreshwar 3.0 Prompt 9 (Independent Robustness, Statistical & Stress Validation Auditor)."""

from __future__ import annotations

import json
import unittest

import numpy as np

from finsistant_v2.m3.independent_audit_p9 import (
    compute_statistical_and_bootstrap_audit,
    run_concentration_and_outlier_removal_audit,
)
from finsistant_v2.m3.schema import DEFAULT_M3_REPORTS_DIR


class TestMoreshwarV3IndependentAuditP9(unittest.TestCase):
    def test_statistical_bootstrap_monte_carlo_and_deflated_sharpe_math(self) -> None:
        rng = np.random.default_rng(77)
        n_days = 300
        nav = 100_000.0
        daily_navs = [{"date": "2023-01-01", "nav": nav, "deployed_mkt": 50_000.0, "ret": 0.0}]
        for i in range(1, n_days):
            r = float(rng.normal(0.0003, 0.008))
            nav *= (1.0 + r)
            daily_navs.append({"date": f"2023-{(i // 25) + 1:02d}-{(i % 25) + 1:02d}", "nav": nav, "deployed_mkt": nav * 0.50, "ret": r})
        trades = [
            {"symbol": f"S{i % 8}", "strategy_id": "STRAT_A", "pnl_amount": float(rng.normal(250.0, 1200.0)), "net_return_pct": float(rng.normal(1.2, 6.0))}
            for i in range(60)
        ]
        bench_rets = rng.normal(0.0004, 0.009, size=n_days)

        st = compute_statistical_and_bootstrap_audit(
            daily_navs=daily_navs, trades=trades, bench_rets=bench_rets, total_trials_tested=185, n_bootstrap=400, block_size=20
        )
        mm = st["return_distribution_moments"]
        self.assertLessEqual(mm["historical_cvar_95_daily_pct"], mm["historical_var_95_daily_pct"])
        self.assertLessEqual(mm["historical_cvar_99_daily_pct"], mm["historical_var_99_daily_pct"])

        bb = st["block_bootstrap_20d_2000_sims"]
        ci_r = bb["uninvested_total_return_ci95_pct"]
        self.assertTrue(ci_r[0] <= ci_r[1] <= ci_r[2])
        ci_dd = bb["max_drawdown_ci95_pct"]
        self.assertTrue(0.0 <= ci_dd[0] <= ci_dd[1] <= ci_dd[2])

        mc = st["monte_carlo_trade_sequence_2000_sims"]
        self.assertTrue(mc["mc_drawdown_p50_pct"] <= mc["mc_drawdown_p95_pct"] <= mc["mc_drawdown_p99_pct"])

        ds = st["multiple_testing_and_deflated_sharpe"]
        self.assertTrue(0.0 <= ds["deflated_sharpe_ratio_prob"] <= 1.0)
        self.assertGreater(ds["expected_max_sharpe_under_null"], 0.5)

    def test_concentration_and_outlier_removal_fragility(self) -> None:
        daily_navs = [
            {"date": "2023-01-01", "nav": 100_000.0, "ret": 0.0},
            {"date": "2023-05-10", "nav": 106_000.0, "ret": 0.06},
            {"date": "2024-02-10", "nav": 108_000.0, "ret": 0.0188679},
        ]
        trades = [
            {"symbol": "WIN1", "strategy_id": "STRAT_A", "pnl_amount": 5000.0, "net_return_pct": 25.0},
            {"symbol": "WIN2", "strategy_id": "STRAT_A", "pnl_amount": 4000.0, "net_return_pct": 20.0},
            {"symbol": "LOSS1", "strategy_id": "STRAT_E", "pnl_amount": -1000.0, "net_return_pct": -5.0},
        ]
        conc = run_concentration_and_outlier_removal_audit(daily_navs=daily_navs, trades=trades)
        self.assertEqual(conc["baseline_uninvested_return_pct"], 8.0)
        self.assertEqual(conc["remove_largest_winning_stock"]["symbol"], "WIN1")
        self.assertEqual(conc["remove_largest_winning_stock"]["remaining_return_pct"], 3.0)
        self.assertEqual(conc["remove_top_5_winning_trades"]["remaining_return_pct"], -1.0)
        self.assertTrue(conc["remove_top_5_winning_trades"]["turns_negative_without_top5_trades"])

    def test_prompt9_deliverables_reconciliation_and_no_go_verdict(self) -> None:
        rep_dir = DEFAULT_M3_REPORTS_DIR / "independent_audit_p9"
        json_p = rep_dir / "independent_validation_report.json"
        reg_p = rep_dir / "reproducibility_and_failure_registry.json"
        md_p = rep_dir / "INDEPENDENT_VALIDATION_REPORT.md"

        self.assertTrue(json_p.exists(), "Missing independent_validation_report.json")
        self.assertTrue(reg_p.exists(), "Missing reproducibility_and_failure_registry.json")
        self.assertTrue(md_p.exists(), "Missing INDEPENDENT_VALIDATION_REPORT.md")

        data = json.loads(json_p.read_text(encoding="utf-8"))
        self.assertEqual(data["final_verdict"]["deployment_decision"], "NO-GO")
        self.assertFalse(data["final_verdict"]["approved_for_live_capital"])
        self.assertTrue(data["reproducibility_and_ledger_reconciliation"]["exact_reconciliation_passed"])
        self.assertEqual(len(data["walk_forward_6fold_audit"]), 6)
        self.assertEqual(len(data["forensic_bias_and_failure_registry"]), 8)


if __name__ == "__main__":
    unittest.main()
