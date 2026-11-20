"""Unit and Verification Tests for Moreshwar 3.0 Prompt 3 (Strategy Families A..E) & Prompt 4 (Regime Intelligence)."""

from __future__ import annotations

import json
import unittest

import numpy as np

from finsistant_v2.m3.regime_intelligence import (
    REGIME_ALLOCATION_POLICY,
    classify_rule_based_9state,
    compute_causal_market_features,
    fit_predict_causal_hmm_3state,
    fit_predict_causal_kmeans,
)
from finsistant_v2.m3.schema import DEFAULT_M3_REPORTS_DIR
from finsistant_v2.m3.strategy_base import StrategySimConfig, simulate_independent_strategy_m3
from finsistant_v2.m3.strategy_families import (
    build_candidate_generator,
    get_strategy_family_configs,
    prepare_strategy_signal_workspace,
)


def _make_synthetic_panel(t_len: int = 320, n_len: int = 24, seed: int = 17) -> dict:
    rng = np.random.default_rng(seed)
    dates = np.array([f"2023-{(i // 22) + 1:02d}-{(i % 22) + 1:02d}" for i in range(t_len)], dtype=object)
    # Build monotonic dates across 2021-2026
    base_dt = np.datetime64("2021-07-01")
    dates = np.array([str(base_dt + np.timedelta64(i * 2, "D")) for i in range(t_len)], dtype=object)
    sec_ids = np.array([f"SEC_{j:03d}" for j in range(n_len)], dtype=object)
    syms = np.array([f"SYM{j:03d}" for j in range(n_len)], dtype=object)
    secs = np.array(["IT", "Banking", "Pharma", "Auto"][j % 4] for j in range(n_len))
    sectors = np.array([["IT", "Banking", "Pharma", "Auto"][j % 4] for j in range(n_len)], dtype=object)

    rets = rng.normal(0.0007, 0.016, size=(t_len, n_len))
    adj_c = 200.0 * np.exp(np.cumsum(rets, axis=0)).astype(np.float32)
    adj_o = (adj_c * (1.0 + rng.normal(0.0, 0.003, size=(t_len, n_len)))).astype(np.float32)
    adj_h = (np.maximum(adj_o, adj_c) * (1.0 + np.abs(rng.normal(0.006, 0.004, size=(t_len, n_len))))).astype(np.float32)
    adj_l = (np.minimum(adj_o, adj_c) * (1.0 - np.abs(rng.normal(0.006, 0.004, size=(t_len, n_len))))).astype(np.float32)
    vol = rng.uniform(200_000, 1_500_000, size=(t_len, n_len)).astype(np.float64)
    t_inr = vol * adj_c
    deliv = rng.uniform(35.0, 75.0, size=(t_len, n_len)).astype(np.float32)
    div_ex = np.zeros((t_len, n_len), dtype=np.float32)
    div_ex[50::60, :] = 4.0

    n50 = 18000.0 * np.exp(np.cumsum(np.mean(rets, axis=1))).astype(np.float64)
    return {
        "dates": dates,
        "sec_ids": sec_ids,
        "current_symbols": syms,
        "sectors": sectors,
        "industries": sectors,
        "statuses": np.array(["ACTIVE"] * n_len, dtype=object),
        "listing_dates": np.array(["2018-01-01"] * n_len, dtype=object),
        "delisting_dates": np.array([""] * n_len, dtype=object),
        "nifty50_open": n50,
        "nifty50_high": n50 * 1.005,
        "nifty50_low": n50 * 0.995,
        "nifty50_close": n50,
        "nifty500_close": n50,
        "midcap150_close": n50,
        "smallcap250_close": n50,
        "open_px": adj_o,
        "high_px": adj_h,
        "low_px": adj_l,
        "close_px": adj_c,
        "adj_open": adj_o,
        "adj_high": adj_h,
        "adj_low": adj_l,
        "adj_close": adj_c,
        "volume": vol,
        "turnover_inr": t_inr,
        "delivery_pct": deliv,
        "mcap_inr": np.full((t_len, n_len), 5e10, dtype=np.float64),
        "adv_20d_inr": np.full((t_len, n_len), 1.5e8, dtype=np.float64),
        "adv_60d_inr": np.full((t_len, n_len), 1.5e8, dtype=np.float64),
        "trading_freq_60d": np.ones((t_len, n_len), dtype=np.float32),
        "realized_vol_20d": np.full((t_len, n_len), 0.22, dtype=np.float32),
        "hl_spread_proxy": np.full((t_len, n_len), 0.012, dtype=np.float32),
        "div_inr_ex": div_ex,
        "is_pit_elig": np.ones((t_len, n_len), dtype=bool),
        "is_valid_eq": np.ones((t_len, n_len), dtype=bool),
    }


class TestMoreshwarV3StrategiesAndRegimes(unittest.TestCase):
    def test_all_five_strategy_families_reconcile_and_avoid_lookahead(self) -> None:
        panel = _make_synthetic_panel(t_len=310, n_len=24)
        ws = prepare_strategy_signal_workspace(panel)
        cfgs = get_strategy_family_configs()
        self.assertEqual(len(cfgs), 5)

        for cfg in cfgs:
            cand_fn = build_candidate_generator(cfg, ws)
            res = simulate_independent_strategy_m3(
                panel=panel,
                cfg=cfg,
                candidate_fn=cand_fn,
                date_start=str(panel["dates"][255]),
                date_end=str(panel["dates"][-1]),
            )
            m = res["metrics"]
            self.assertIn("total_return_pct", m)
            # Exact Cash-to-NAV Reconciliation within Rs 0.05 rounding tolerance
            self.assertLessEqual(abs(float(m["reconciliation_diff"])), 0.05, msg=f"Reconciliation failed for {cfg.strategy_id}")

            # Verify zero lookahead in candidate generator at decision index t_test = 265
            t_test = 265
            cands_base = cand_fn(t_test, panel, {})
            panel_shock = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in panel.items()}
            panel_shock["adj_close"][t_test + 1 :, :] *= 3.5
            panel_shock["adj_high"][t_test + 1 :, :] *= 3.5
            ws_shock = prepare_strategy_signal_workspace(panel_shock)
            cands_shock = build_candidate_generator(cfg, ws_shock)(t_test, panel_shock, {})
            self.assertEqual(
                [(c["col"], round(c["score"], 6)) for c in cands_base],
                [(c["col"], round(c["score"], 6)) for c in cands_shock],
                msg=f"Look-ahead detected in {cfg.strategy_id}",
            )

    def test_causal_regime_classifiers_and_allocation_policy_invariants(self) -> None:
        panel = _make_synthetic_panel(t_len=310, n_len=24)
        feats = compute_causal_market_features(panel)
        is_mask = np.arange(len(panel["dates"])) < 220

        fine9, rule5 = classify_rule_based_9state(feats, confirm_days=3)
        km4 = fit_predict_causal_kmeans(feats, is_mask=is_mask, k=4)
        hmm3, hmm_probs = fit_predict_causal_hmm_3state(feats, is_mask=is_mask)

        self.assertEqual(len(rule5), len(panel["dates"]))
        self.assertEqual(len(km4), len(panel["dates"]))
        self.assertEqual(len(hmm3), len(panel["dates"]))
        self.assertTrue(np.allclose(np.sum(hmm_probs, axis=1), 1.0, atol=1e-5))

        # Verify causal forward-only filtering: shocking t > 240 must NOT change HMM probs or rule5 at t <= 240
        panel_future = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in panel.items()}
        panel_future["nifty50_close"][241:] *= 0.45
        panel_future["adj_close"][241:, :] *= 0.45
        feats_future = compute_causal_market_features(panel_future)
        _, rule5_f = classify_rule_based_9state(feats_future, confirm_days=3)
        _, hmm_probs_f = fit_predict_causal_hmm_3state(feats_future, is_mask=is_mask)

        self.assertEqual(rule5[:241], rule5_f[:241])
        self.assertTrue(np.allclose(hmm_probs[:241], hmm_probs_f[:241], atol=1e-7))

        # Verify every regime allocation policy sums strictly to 1.0 (100% capital accounted for)
        for reg_lbl, weights in REGIME_ALLOCATION_POLICY.items():
            self.assertAlmostEqual(sum(weights.values()), 1.0, places=6, msg=f"Weights for {reg_lbl} do not sum to 1.0")

    def test_prompt3_and_prompt4_deliverables_exist_and_valid(self) -> None:
        strat_rep_path = DEFAULT_M3_REPORTS_DIR / "strategies" / "strategy_families_report.json"
        strat_md_path = DEFAULT_M3_REPORTS_DIR / "strategies" / "STRATEGY_FAMILIES_REPORT.md"
        reg_rep_path = DEFAULT_M3_REPORTS_DIR / "regime_intelligence" / "adaptive_allocation_report.json"
        reg_md_path = DEFAULT_M3_REPORTS_DIR / "regime_intelligence" / "REGIME_INTELLIGENCE_REPORT.md"

        self.assertTrue(strat_rep_path.exists(), "Missing strategy_families_report.json")
        self.assertTrue(strat_md_path.exists(), "Missing STRATEGY_FAMILIES_REPORT.md")
        self.assertTrue(reg_rep_path.exists(), "Missing adaptive_allocation_report.json")
        self.assertTrue(reg_md_path.exists(), "Missing REGIME_INTELLIGENCE_REPORT.md")

        s_data = json.loads(strat_rep_path.read_text(encoding="utf-8"))
        self.assertEqual(len(s_data["strategies"]), 5)
        for sid, s_obj in s_data["strategies"].items():
            self.assertLessEqual(abs(float(s_obj["full_metrics"]["reconciliation_diff"])), 0.05, sid)

        r_data = json.loads(reg_rep_path.read_text(encoding="utf-8"))
        self.assertIn("M3_RULE5_REGIME_ADAPTIVE", r_data["portfolio_comparison"])
        self.assertIn("M3_HMM3_CAUSAL_ADAPTIVE", r_data["portfolio_comparison"])
        self.assertIn("M3_STATIC_EQUAL_WEIGHT_5FAM", r_data["portfolio_comparison"])


if __name__ == "__main__":
    unittest.main()
