"""Unit & Integration Tests for Moreshwar 3.1 (Prompt 2/5) Intelligent Capital Allocation & Portfolio Engine."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from finsistant_v2.m3_1.allocation_experiments import DEFAULT_M3_1_ALLOC_DIR
from finsistant_v2.m3_1.capital_allocation_engine import (
    CapitalAllocationConfig,
    _compute_dynamic_pillar_targets,
    precompute_m3_1_research_context,
    simulate_m3_1_capital_allocation,
)
from finsistant_v2.m3_1.dividend_accounting import (
    apply_ex_dividend_accounting_and_stop_adjustment,
    compute_split_adjusted_div_yield_ttm,
)
from finsistant_v2.m3_1.sector_hierarchy import (
    UNKNOWN_SECTOR_LABEL,
    build_4tier_sector_classification,
)
from tests.test_moreshwar_v3_strategies_and_regime import _make_synthetic_panel


def test_split_adjusted_dividend_yield_and_ex_date_stop_protection() -> None:
    # Construct a 300-day panel with a 1:5 stock split on day 200 and a pre-split Rs 25 dividend on day 100
    t_len = 300
    raw_c = np.full((t_len, 1), 1000.0, dtype=np.float64)
    raw_c[200:, 0] = 200.0  # 1:5 split on day 200
    adj_c = np.full((t_len, 1), 200.0, dtype=np.float64)  # split-adjusted price is flat at Rs 200
    div_ex = np.zeros((t_len, 1), dtype=np.float64)
    div_ex[100, 0] = 25.0  # Rs 25 pre-split = 2.5% yield (Rs 5 post-split)

    dy = compute_split_adjusted_div_yield_ttm({"close_px": raw_c, "adj_close": adj_c, "div_inr_ex": div_ex})
    # On day 260 (within 252D of day 100), naive div_252/raw_c would report 25/200 = 12.5%!
    # Split-adjusted must report exact 5/200 = 2.5%!
    assert abs(float(dy[260, 0]) - 0.025) < 1e-4

    # Test ex-dividend cash credit & stop-loss protection
    pos = {
        "symbol": "IDFC",
        "qty": 100,
        "entry_ti": 50,
        "entry_px": 85.0,
        "sl_px": 78.0,
        "max_px": 90.0,
        "dividends_received_inr": 0.0,
    }
    ev = apply_ex_dividend_accounting_and_stop_adjustment(pos=pos, ti=65, dt="2023-02-13", div_per_adj_share=11.0)
    assert ev["credited_cash_inr"] == 1100.0
    assert ev["adjusted_sl"] is True
    assert abs(pos["sl_px"] - 67.0) < 1e-6  # 78 - 11 = 67 prevents artificial ex-div stop-out
    assert pos["dividends_received_inr"] == 1100.0

    # Same-day entry (ti == entry_ti) MUST NOT receive ex-date dividend (bought ex-dividend!)
    ev_same = apply_ex_dividend_accounting_and_stop_adjustment(pos=pos, ti=50, dt="2023-02-13", div_per_adj_share=11.0)
    assert ev_same["credited_cash_inr"] == 0.0


def test_4tier_sector_hierarchy_resolves_unclassified_stocks() -> None:
    panel = _make_synthetic_panel(t_len=30, n_len=6)
    panel["current_symbols"] = np.array(["TCS", "HDFC", "BIKAJI", "AVANTEL", "XYZ PHARMA", "MYSTERYCORP"], dtype=object)
    panel["sectors"] = np.array(
        [
            "Information Technology",
            "Unclassified (Historical/Non-Total-Market)",
            "Unclassified (Historical/Non-Total-Market)",
            "Unclassified (Historical/Non-Total-Market)",
            "Unclassified (Historical/Non-Total-Market)",
            "Unclassified (Historical/Non-Total-Market)",
        ],
        dtype=object,
    )
    res = build_4tier_sector_classification(panel, db_path=Path("/tmp/nonexistent_m3_test.db"))
    assert res["sectors"][0] == "Information Technology"
    assert res["tiers"][0] == "TIER_1_VALIDATED_NSE_SECTOR"
    assert res["sectors"][1] == "Financial Services"
    assert res["tiers"][1] == "TIER_2_REFERENCE_MAPPING"
    assert res["sectors"][2] == "Fast Moving Consumer Goods"
    assert res["sectors"][4] == "Healthcare"
    assert res["sectors"][5] == UNKNOWN_SECTOR_LABEL
    assert bool(res["is_unknown"][5]) is True


def test_all_six_allocation_modes_sum_to_100_and_zero_leverage() -> None:
    for mode in ("FIXED", "VOLATILITY_TARGETED", "DRAWDOWN_AWARE", "REGIME_AWARE", "RISK_BUDGETED", "CONFIDENCE_WEIGHTED"):
        cfg = CapitalAllocationConfig(
            config_id=f"T_{mode}",
            config_name=mode,
            core_target_pct=60.0,
            tactical_target_pct=30.0,
            liquid_target_pct=10.0,
            allocation_mode=mode,
        )
        for reg in ("STRONG_BULL", "SIDEWAYS_CHOP", "STRESS_BEAR"):
            c, t, l = _compute_dynamic_pillar_targets(
                cfg=cfg,
                cur_regime=reg,
                cur_dd_pct=12.0,
                port_vol_20d_ann=18.0,
                core_vol_20d_ann=14.0,
                tact_vol_20d_ann=17.0,
                avg_signal_score=0.72,
                signal_count=6,
            )
            assert abs((c + t + l) - 100.0) <= 0.05, f"Weights did not sum to 100 for {mode}/{reg}: {c},{t},{l}"
            assert c >= 0.0 and t >= 0.0 and l >= 0.0


def test_m3_1_synthetic_simulation_invariants() -> None:
    panel = _make_synthetic_panel(t_len=310, n_len=20)
    ctx = precompute_m3_1_research_context(panel, db_path=Path("/tmp/nonexistent_m3_test.db"))
    cfg = CapitalAllocationConfig(
        config_id="SYNTH_TEST",
        config_name="Synthetic Test M3.1",
        core_target_pct=60.0,
        tactical_target_pct=30.0,
        liquid_target_pct=10.0,
        allocation_mode="REGIME_AWARE",
        event_driven_replenishment=True,
    )
    sim = simulate_m3_1_capital_allocation(
        ctx=ctx,
        cfg=cfg,
        date_start=str(panel["dates"][252]),
        date_end=str(panel["dates"][-1]),
        initial_capital=100_000.0,
    )
    m = sim["metrics"]
    assert m["avg_productive_capital_pct"] >= 85.0
    assert m["avg_uninvested_cash_pct"] <= 15.0
    assert abs(float(m["reconciliation_check_diff_inr"])) <= 5.0
    for d in sim["daily_navs"]:
        assert d["nav"] > 0.0
        assert d["equity_exposure_pct"] <= 101.0


def test_prompt2_deliverables_and_6_charts_generated() -> None:
    out = DEFAULT_M3_1_ALLOC_DIR
    required_files = [
        "CAPITAL_ALLOCATION_ARCHITECTURE.md",
        "CAPITAL_ALLOCATION_EXPERIMENTS.csv",
        "PORTFOLIO_NAV_COMPARISON.csv",
        "CAPITAL_UTILIZATION_ANALYSIS.csv",
        "SECTOR_CLASSIFICATION_AUDIT.csv",
        "TRADE_REPLENISHMENT_ANALYSIS.csv",
        "PORTFOLIO_RISK_REPORT.md",
        "DIVIDEND_ACCOUNTING_VALIDATION.md",
        "MORESHWAR_3_1_ALLOCATION_RECOMMENDATIONS.md",
        "chart_nav_comparison.svg",
        "chart_passive_vs_tactical_contribution.svg",
        "chart_capital_utilization.svg",
        "chart_portfolio_drawdowns.svg",
        "chart_allocation_over_time.svg",
        "chart_sector_exposure.svg",
        "capital_allocation_summary.json",
    ]
    for fname in required_files:
        p = out / fname
        assert p.exists(), f"Missing Prompt 2/5 deliverable: {p}"
        assert p.stat().st_size > 200, f"Deliverable too small: {p}"

    with (out / "CAPITAL_ALLOCATION_EXPERIMENTS.csv").open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    exp_ids = {r["experiment_id"] for r in rows}
    # Verify all 9 experiments A..I exist in both FULL and OOS
    assert len(exp_ids) == 9
    assert len(rows) == 18

    summary = json.loads((out / "capital_allocation_summary.json").read_text(encoding="utf-8"))
    assert summary["exp_I_full_metrics"]["avg_productive_capital_pct"] >= 94.0
    assert summary["sector_hierarchy_summary"]["remaining_unknown_pct"] < 35.0
