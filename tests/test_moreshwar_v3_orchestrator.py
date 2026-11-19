"""Automated Validation Tests for Moreshwar 3.0 Prompt 7 (Adaptive Strategy Orchestration, Suspension & Explainability Engine)."""

from __future__ import annotations

import json
import unittest

import numpy as np

from finsistant_v2.m3.schema import DEFAULT_M3_REPORTS_DIR
from finsistant_v2.m3.strategy_orchestrator import (
    SleeveRuntimeState,
    build_m3_strategy_registry,
    compute_orchestrator_weights_at_date,
    simulate_strategy_orchestration_m3,
    update_sleeve_suspension_state,
)


class TestMoreshwarV3Orchestrator(unittest.TestCase):
    def test_strategy_registry_completeness_and_is_only_baselines(self) -> None:
        reg = build_m3_strategy_registry()
        self.assertEqual(len(reg), 9)
        sids = {e.strategy_id for e in reg}
        self.assertIn("STRAT_A_CS_MOMENTUM", sids)
        self.assertIn("STRAT_E_DEFENSIVE_YIELD_LOWVOL", sids)
        self.assertIn("STRAT_RV1_COINT_PAIRS", sids)
        self.assertIn("STRAT_RV4A_NIFTY_FUT_HEDGED_25L_LOT", sids)
        for e in reg:
            self.assertTrue(e.version.startswith("3.0.0"))
            self.assertGreater(len(e.required_data), 0)
            self.assertGreater(len(e.validated_regimes), 0)
            self.assertGreater(e.max_allowed_sleeve_dd_pct, e.throttle_dd_pct)
            self.assertGreater(e.throttle_dd_pct, e.recovery_dd_pct)

    def test_sleeve_suspension_and_documented_recovery_state_machine(self) -> None:
        reg_entry = build_m3_strategy_registry()[0]  # STRAT_A_CS_MOMENTUM (throttle=14%, suspend=21%, recover=11%)
        st = SleeveRuntimeState(strategy_id=reg_entry.strategy_id)
        self.assertEqual(st.status, "ACTIVE")

        # 1. Moderate drawdown (-15%) triggers THROTTLED_50PCT
        update_sleeve_suspension_state(
            state=st,
            reg_entry=reg_entry,
            day_ret=-0.15,
            trailing_20d_ret=-0.06,
            trailing_60d_ret=-0.06,
            dq_ok=True,
            current_regime="STRONG_BULL",
        )
        self.assertEqual(st.status, "THROTTLED_50PCT")

        # 2. Deeper drawdown (-10% further -> >21% cumulative DD) triggers SUSPENDED
        update_sleeve_suspension_state(
            state=st,
            reg_entry=reg_entry,
            day_ret=-0.10,
            trailing_20d_ret=-0.12,
            trailing_60d_ret=-0.15,
            dq_ok=True,
            current_regime="STRONG_BULL",
        )
        self.assertEqual(st.status, "SUSPENDED")
        self.assertIn("SLEEVE_MAX_DD_BREACH", st.suspension_reason)

        # 3. Before 15-day cool-off, even a strong 20D rebound must NOT exit SUSPENDED
        for _ in range(10):
            update_sleeve_suspension_state(
                state=st,
                reg_entry=reg_entry,
                day_ret=0.002,
                trailing_20d_ret=0.04,
                trailing_60d_ret=0.01,
                dq_ok=True,
                current_regime="STRONG_BULL",
            )
        self.assertEqual(st.status, "SUSPENDED")

        # 4. In an incompatible regime (STRESS_BEAR), even after 15 days it stays SUSPENDED
        for _ in range(6):
            update_sleeve_suspension_state(
                state=st,
                reg_entry=reg_entry,
                day_ret=0.002,
                trailing_20d_ret=0.04,
                trailing_60d_ret=0.01,
                dq_ok=True,
                current_regime="STRESS_BEAR",
            )
        self.assertEqual(st.status, "SUSPENDED")

        # 5. Once in validated regime (STRONG_BULL) with >=15d cooloff and +4% 20D shadow recovery -> THROTTLED_50PCT probation
        update_sleeve_suspension_state(
            state=st,
            reg_entry=reg_entry,
            day_ret=0.005,
            trailing_20d_ret=0.04,
            trailing_60d_ret=0.02,
            dq_ok=True,
            current_regime="STRONG_BULL",
        )
        self.assertEqual(st.status, "THROTTLED_50PCT")
        self.assertIn("RECOVERED_TO_PROBATION_50PCT", st.recovery_note)

    def test_orchestrator_explainability_and_zero_lookahead(self) -> None:
        reg = build_m3_strategy_registry()
        long_sids = [e.strategy_id for e in reg if not e.requires_slbm_or_futures]
        rng = np.random.default_rng(55)
        n_days = 160
        dates = [f"2024-{(i // 22) + 1:02d}-{(i % 22) + 1:02d}" for i in range(n_days)]
        sleeve_rets = {s: rng.normal(0.0005, 0.011, size=n_days) for s in long_sids}
        bench_rets = rng.normal(0.0004, 0.010, size=n_days)
        regimes = ["WEAK_BULL"] * 80 + ["STRESS_BEAR"] * 40 + ["STRONG_BULL"] * 40
        dq_cov = np.full(n_days, 98.0, dtype=np.float64)

        sim_base = simulate_strategy_orchestration_m3(
            mode="CONSTRAINED_ADAPTIVE_ORCHESTRATOR",
            dates=dates,
            sleeve_daily_rets=sleeve_rets,
            bench_rets=bench_rets,
            regime_labels=regimes,
            dq_coverage_series=dq_cov,
            registry=reg,
            rebalance_interval_days=15,
        )
        self.assertGreater(len(sim_base["explanation_logs"]), 5)
        for expl in sim_base["explanation_logs"]:
            self.assertAlmostEqual(sum(expl["target_weights"].values()), 1.0, places=3)
            self.assertIn("STRAT_RV1_COINT_PAIRS", expl["rejected_strategies"])

        # Perturb returns after day 105 -> all NAVs and explanations up to day 105 must be identical
        rets_shock = {s: v.copy() for s, v in sleeve_rets.items()}
        for s in long_sids:
            rets_shock[s][106:] = -0.05
        sim_shock = simulate_strategy_orchestration_m3(
            mode="CONSTRAINED_ADAPTIVE_ORCHESTRATOR",
            dates=dates,
            sleeve_daily_rets=rets_shock,
            bench_rets=bench_rets,
            regime_labels=regimes,
            dq_coverage_series=dq_cov,
            registry=reg,
            rebalance_interval_days=15,
        )
        self.assertEqual(
            [d["nav"] for d in sim_base["daily_navs"][:106]],
            [d["nav"] for d in sim_shock["daily_navs"][:106]],
        )

    def test_prompt7_deliverables_exist_and_valid(self) -> None:
        rep_dir = DEFAULT_M3_REPORTS_DIR / "orchestration"
        reg_p = rep_dir / "strategy_registry.json"
        expl_p = rep_dir / "allocation_explanations_log.json"
        rep_p = rep_dir / "strategy_orchestration_report.json"
        md_p = rep_dir / "STRATEGY_ORCHESTRATION_REPORT.md"

        self.assertTrue(reg_p.exists(), "Missing strategy_registry.json")
        self.assertTrue(expl_p.exists(), "Missing allocation_explanations_log.json")
        self.assertTrue(rep_p.exists(), "Missing strategy_orchestration_report.json")
        self.assertTrue(md_p.exists(), "Missing STRATEGY_ORCHESTRATION_REPORT.md")

        data = json.loads(rep_p.read_text(encoding="utf-8"))
        self.assertEqual(data["strategy_registry_count"], 9)
        for mode in (
            "STATIC",
            "REGIME_BASED",
            "VOLATILITY_ADJUSTED",
            "RISK_BUDGETED",
            "CONSTRAINED_ADAPTIVE_ORCHESTRATOR",
        ):
            self.assertIn(mode, data["allocators_comparison"])


if __name__ == "__main__":
    unittest.main()
