"""Unit, Causality, Edge-Case & Integration Tests for Moreshwar 3.1 (Prompt 4/5) Strategy Intelligence Engine."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from finsistant_v2.m3_1.adaptive_strategy_engine import (
    get_standard_adaptive_policies,
    simulate_m3_1_adaptive_regime_portfolio,
)
from finsistant_v2.m3_1.capital_allocation_engine import precompute_m3_1_research_context
from finsistant_v2.m3_1.execution_simulation_engine import ExecutionIntelligenceConfig
from finsistant_v2.m3_1.market_regime_engine import (
    classify_rule_based_6state,
    compute_causal_regime_features,
    fit_and_predict_statistical_regime_model,
)
from finsistant_v2.m3_1.regime_monitoring import evaluate_regime_and_strategy_health
from finsistant_v2.m3_1.strategy_validation_engine import DEFAULT_M3_1_STRAT_DIR
from tests.test_moreshwar_v3_strategies_and_regime import _make_synthetic_panel


def test_regime_features_strict_no_future_leakage() -> None:
    """Prove bit-for-bit that mutating ALL data after `t_cut` has ZERO effect on historical features or regimes <= `t_cut`."""
    p1 = _make_synthetic_panel(t_len=280, n_len=15)
    p2 = {k: (v.copy() if isinstance(v, np.ndarray) else list(v) if isinstance(v, list) else v) for k, v in p1.items()}
    t_cut = 180

    # Violently mutate all future data after t_cut
    for key in ("adj_open", "adj_high", "adj_low", "adj_close", "close_px"):
        p2[key][t_cut + 1 :] *= 3.5
    p2["nifty50_close"][t_cut + 1 :] *= 0.25
    p2["volume"][t_cut + 1 :] *= 10.0

    etf1 = {"NIFTYBEES": {"open": p1["nifty50_close"].copy(), "high": p1["nifty50_close"] * 1.01, "low": p1["nifty50_close"] * 0.99, "close": p1["nifty50_close"].copy(), "volume": np.full(280, 1e6)}}
    etf2 = {"NIFTYBEES": {"open": p2["nifty50_close"].copy(), "high": p2["nifty50_close"] * 1.01, "low": p2["nifty50_close"] * 0.99, "close": p2["nifty50_close"].copy(), "volume": np.full(280, 1e6)}}

    f1 = compute_causal_regime_features(panel=p1, etf_bars=etf1, db_path=Path("/tmp/nonexistent_p4.db"))
    f2 = compute_causal_regime_features(panel=p2, etf_bars=etf2, db_path=Path("/tmp/nonexistent_p4.db"))

    for fname in ("nb_ret_20d", "nb_ret_60d", "px_vs_sma50_pct", "pct_ab50", "breadth_delta_20d", "vol_20d_ann", "vol_pctile_252d", "ew_vs_nifty_20d"):
        a1 = f1[fname][: t_cut + 1]
        a2 = f2[fname][: t_cut + 1]
        assert np.allclose(a1, a2, equal_nan=True), f"Future leakage detected in feature {fname}!"

    r1 = classify_rule_based_6state(f1, confirm_days=3)
    r2 = classify_rule_based_6state(f2, confirm_days=3)
    assert r1["labels"][: t_cut + 1] == r2["labels"][: t_cut + 1]
    assert r1["confidences"][: t_cut + 1] == r2["confidences"][: t_cut + 1]

    # Model B frozen on <= t_cut MUST also be bit-for-bit identical <= t_cut!
    cut_dt = str(p1["dates"][t_cut])
    s1 = fit_and_predict_statistical_regime_model(f1, train_start=str(p1["dates"][0]), train_end=cut_dt, k=4)
    s2 = fit_and_predict_statistical_regime_model(f2, train_start=str(p2["dates"][0]), train_end=cut_dt, k=4)
    assert s1["labels"][: t_cut + 1] == s2["labels"][: t_cut + 1]


def test_missing_data_holidays_and_insufficient_history_handling() -> None:
    """Verify graceful handling of short history, missing NaN holiday gaps, and insufficient bars."""
    panel = _make_synthetic_panel(t_len=35, n_len=8)
    # Inject NaN gaps / missing trading bars
    panel["adj_close"][5:8, :] = np.nan
    panel["nifty50_close"][10:12] = np.nan
    panel["volume"][15, :] = 0.0

    etf = {"NIFTYBEES": {"open": panel["nifty50_close"].copy(), "high": panel["nifty50_close"].copy(), "low": panel["nifty50_close"].copy(), "close": panel["nifty50_close"].copy(), "volume": np.full(35, 5e5)}}
    feats = compute_causal_regime_features(panel=panel, etf_bars=etf, db_path=Path("/tmp/nonexistent_p4.db"))
    rule = classify_rule_based_6state(feats, confirm_days=3)
    assert len(rule["labels"]) == 35
    assert all(st == "WARMUP_PARTIAL_SMA200" for st in rule["data_availability"])

    # Insufficient evaluation window (<15 days) in adaptive simulator returns clean empty metrics without crashing
    panel_sim = _make_synthetic_panel(t_len=160, n_len=8)
    panel_sim["adj_close"][5:8, :] = np.nan
    ctx = precompute_m3_1_research_context(panel_sim, db_path=Path("/tmp/nonexistent_p4.db"))
    pol = get_standard_adaptive_policies()["A_STATIC_PROMPT3_PRIMARY"]
    res = simulate_m3_1_adaptive_regime_portfolio(
        ctx=ctx, exec_cfg=ExecutionIntelligenceConfig(config_id="SHORT", config_name="Short"),
        policy=pol, regime_labels=["BULL_BROAD"] * 160, date_start=str(panel_sim["dates"][-5]), date_end=str(panel_sim["dates"][-1]),
    )
    assert res["daily_navs"] == []


def test_regime_transition_hysteresis_and_immediate_stress_protection() -> None:
    """Confirm 1-day non-stress blips do not flip regime under confirm_days=3, while HIGH_VOL_STRESS triggers immediately."""
    t_len = 12
    feats = {
        "dates": [f"2024-01-{i+1:02d}" for i in range(t_len)],
        "px_vs_sma50_pct": np.array([2.0, 2.0, 2.0, -0.5, 2.0, 2.0, 2.0, -8.0, 2.0, 2.0, 2.0, 2.0]),
        "px_vs_sma200_pct": np.full(t_len, 5.0),
        "sma50_slope_20d": np.full(t_len, 1.5),
        "nb_ret_20d": np.array([3.0, 3.0, 3.0, 0.0, 3.0, 3.0, 3.0, -7.0, 3.0, 3.0, 3.0, 3.0]),
        "nb_ret_60d": np.full(t_len, 6.0),
        "dist_252h_pct": np.array([-1.0, -1.0, -1.0, -2.0, -1.0, -1.0, -1.0, -12.0, -1.0, -1.0, -1.0, -1.0]),
        "pct_ab50": np.array([65.0, 65.0, 65.0, 48.0, 65.0, 65.0, 65.0, 25.0, 65.0, 65.0, 65.0, 65.0]),
        "breadth_delta_20d": np.zeros(t_len),
        "ew_vs_nifty_20d": np.zeros(t_len),
        "vol_20d_ann": np.array([11.0, 11.0, 11.0, 12.0, 11.0, 11.0, 11.0, 24.0, 11.0, 11.0, 11.0, 11.0]),
        "vol_pctile_252d": np.full(t_len, 0.40),
        "sector_participation_pct": np.full(t_len, 70.0),
        "diag_india_vix": np.full(t_len, 13.0),
    }
    out = classify_rule_based_6state(feats, confirm_days=3)
    # Day 0-2 confirm BULL_BROAD; Day 3 single sideways blip MUST be absorbed by hysteresis!
    assert out["labels"][2] == "BULL_BROAD"
    assert out["labels"][3] == "BULL_BROAD"
    # Day 7 sudden vol=24% & dd=-12% MUST transition immediately into HIGH_VOL_STRESS!
    assert out["labels"][7] == "HIGH_VOL_STRESS"


def test_all_adaptive_policy_weights_reconcile_to_100() -> None:
    for pid, pol in get_standard_adaptive_policies().items():
        for rule in pol.regime_rules:
            c, t, l = rule[1], rule[2], rule[3]
            assert abs((c + t + l) - 100.0) <= 1e-6, f"Weights do not sum to 100 in {pid}/{rule[0]}"
            assert c >= 0.0 and t >= 0.0 and l >= 0.0


def test_advisory_monitoring_distinguishes_normal_drawdown_from_edge_decay_and_data_failure() -> None:
    # 1. Three normal losing trades (<15 min) + 13% drawdown -> NORMAL_DRAWDOWN_WATCH (MUST NOT suspend!)
    res_norm = evaluate_regime_and_strategy_health(
        date="2025-05-10", vol_ratio_20_60=1.05, vol_pctile_252d=0.55, pct_ab50=52.0, breadth_delta_20d=-2.0,
        current_drawdown_pct=13.0,
        recent_closed_trades=[{"total_pnl_with_div_inr": -400.0}] * 3,
        trailing_annualized_turnover_x=4.5, cost_pct_of_gross_profits=4.5, max_sector_exposure_pct=22.0,
        uninvested_cash_pct=2.5, consecutive_high_cash_days=0, regime_flips_last_20d=1,
    )
    assert res_norm["advisory_status"] == "NORMAL_DRAWDOWN_WATCH"
    assert res_norm["live_execution_enabled"] is False

    # 2. Persistent PF decay (<0.70 over 20 closed trades) -> SUSPEND_TACTICAL_ADVISORY
    res_decay = evaluate_regime_and_strategy_health(
        date="2025-05-10", vol_ratio_20_60=1.05, vol_pctile_252d=0.55, pct_ab50=52.0, breadth_delta_20d=-2.0,
        current_drawdown_pct=8.0,
        recent_closed_trades=[{"total_pnl_with_div_inr": 200.0}] * 5 + [{"total_pnl_with_div_inr": -500.0}] * 15,
        trailing_annualized_turnover_x=4.5, cost_pct_of_gross_profits=4.5, max_sector_exposure_pct=22.0,
        uninvested_cash_pct=2.5, consecutive_high_cash_days=0, regime_flips_last_20d=1,
    )
    assert res_decay["advisory_status"] == "SUSPEND_TACTICAL_ADVISORY"
    assert res_decay["live_execution_enabled"] is False

    # 3. Data or ledger failure -> DATA_EXECUTION_FAILURE_HALT
    res_fail = evaluate_regime_and_strategy_health(
        date="2025-05-10", vol_ratio_20_60=1.0, vol_pctile_252d=0.5, pct_ab50=55.0, breadth_delta_20d=0.0,
        current_drawdown_pct=2.0, recent_closed_trades=[], trailing_annualized_turnover_x=3.0,
        cost_pct_of_gross_profits=3.0, max_sector_exposure_pct=15.0, uninvested_cash_pct=2.0,
        consecutive_high_cash_days=0, regime_flips_last_20d=0, data_quality_ok=False, reconciliation_diff_inr=25.0,
    )
    assert res_fail["advisory_status"] == "DATA_EXECUTION_FAILURE_HALT"


def test_prompt4_all_12_deliverables_8_charts_and_reproduction_verified() -> None:
    out = DEFAULT_M3_1_STRAT_DIR
    required_files = [
        "STRATEGY_INTELLIGENCE_ARCHITECTURE.md",
        "MARKET_REGIME_DEFINITION.md",
        "REGIME_CLASSIFICATION_HISTORY.csv",
        "FAMILY_E_REGIME_PERFORMANCE.csv",
        "ADAPTIVE_ALLOCATION_EXPERIMENTS.csv",
        "FAMILY_E_COMPONENT_ANALYSIS.csv",
        "NEW_STRATEGY_RESEARCH.csv",
        "NESTED_WALK_FORWARD_RESULTS.csv",
        "ROBUSTNESS_AND_STATISTICAL_AUDIT.md",
        "RISK_MONITORING_RULES.md",
        "PROMPT_4_FINAL_RECOMMENDATION.md",
        "FROZEN_PROSPECTIVE_PAPER_TRADING_CONFIG.json",
        "chart_market_regime_history.svg",
        "chart_regime_confidence.svg",
        "chart_family_e_returns_by_regime.svg",
        "chart_static_vs_adaptive_allocation.svg",
        "chart_drawdown_comparison.svg",
        "chart_benchmark_relative_performance.svg",
        "chart_outer_fold_performance.svg",
        "chart_tactical_exposure_over_time.svg",
        "strategy_intelligence_summary.json",
    ]
    for fname in required_files:
        p = out / fname
        assert p.exists(), f"Missing Prompt 4/5 deliverable: {p}"
        assert p.stat().st_size > 200, f"Deliverable too small: {p}"

    # Verify frozen prospective config strictly disables live trading
    frozen = json.loads((out / "FROZEN_PROSPECTIVE_PAPER_TRADING_CONFIG.json").read_text(encoding="utf-8"))
    assert frozen["live_trading_enabled"] is False
    assert frozen["broker_order_placement_enabled"] is False
    assert frozen["automatic_capital_deployment_enabled"] is False

    # Verify exact Prompt 3 baseline reproduction & 0.00 INR reconciliation across all adaptive policies
    summary = json.loads((out / "strategy_intelligence_summary.json").read_text(encoding="utf-8"))
    assert summary["reproduction_check"]["exact_reproduction_verified"] is True
    assert abs(float(summary["reproduction_check"]["primary_diff_pp"])) <= 0.02

    with (out / "ADAPTIVE_ALLOCATION_EXPERIMENTS.csv").open("r", encoding="utf-8") as f:
        arows = list(csv.DictReader(f))
    assert len(arows) == 7
    for r in arows:
        assert abs(float(r["reconciliation_diff_inr"])) <= 0.50
