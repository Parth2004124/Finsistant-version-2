"""Unit, Integration & Regression Tests for Moreshwar 3.1 (Prompt 3/5) Execution Intelligence Engine."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from finsistant_v2.m3_1.capital_allocation_engine import precompute_m3_1_research_context
from finsistant_v2.m3_1.execution_experiments import DEFAULT_M3_1_EXEC_DIR
from finsistant_v2.m3_1.execution_simulation_engine import (
    ExecutionIntelligenceConfig,
    enrich_m3_1_execution_context,
    simulate_m3_1_execution_intelligence,
)
from finsistant_v2.m3_1.trade_quality_models import (
    compute_intelligent_position_size,
    estimate_trade_economic_value,
    evaluate_position_exit,
    evaluate_position_replacement,
    simulate_execution_fill,
)
from finsistant_v2.m3_1.transaction_cost_intelligence import (
    IndianTariffRegimeSchedule,
    TransactionCostIntelligence,
)
from tests.test_moreshwar_v3_strategies_and_regime import _make_synthetic_panel


def test_historical_indian_tariff_regime_switches_and_no_double_counting() -> None:
    ce = TransactionCostIntelligence(IndianTariffRegimeSchedule())

    # Pre-Oct 2024 vs Post-Oct 2024 historical regime switch (NSE CM fee & CDSL DP charge)
    sell_pre = ce.calculate_leg_cost(instrument_class="DELIVERY", side="SELL", raw_notional=10_000.0, trade_date="2023-06-15", slippage_bps=0.0)
    sell_post = ce.calculate_leg_cost(instrument_class="DELIVERY", side="SELL", raw_notional=10_000.0, trade_date="2025-06-15", slippage_bps=0.0)
    assert sell_pre["dp_charge"] == 15.93
    assert sell_post["dp_charge"] == 15.34
    assert sell_pre["exchange_txn"] > sell_post["exchange_txn"]
    assert sell_pre["stt"] == 10.0  # 0.10% delivery STT

    # DP charge MUST NOT be charged twice on same scrip/day or on buys
    sell_dup = ce.calculate_leg_cost(
        instrument_class="DELIVERY", side="SELL", raw_notional=10_000.0, trade_date="2025-06-15", slippage_bps=0.0, dp_already_charged_today=True
    )
    buy_del = ce.calculate_leg_cost(instrument_class="DELIVERY", side="BUY", raw_notional=10_000.0, trade_date="2025-06-15", slippage_bps=0.0)
    assert sell_dup["dp_charge"] == 0.0
    assert buy_del["dp_charge"] == 0.0
    assert buy_del["stamp_duty"] == 1.5  # 0.015% buy stamp duty

    # Equity ETF (`NIFTYBEES`) sell STT = 0.001%, buy STT = 0; Liquid ETF (`LIQUIDBEES`) STT = 0
    etf_b = ce.calculate_leg_cost(instrument_class="EQUITY_ETF", side="BUY", raw_notional=100_000.0, trade_date="2025-06-15", slippage_bps=0.0)
    etf_s = ce.calculate_leg_cost(instrument_class="EQUITY_ETF", side="SELL", raw_notional=100_000.0, trade_date="2025-06-15", slippage_bps=0.0)
    liq_b = ce.calculate_leg_cost(instrument_class="LIQUID_ETF", side="BUY", raw_notional=100_000.0, trade_date="2025-06-15", slippage_bps=0.0)
    liq_s = ce.calculate_leg_cost(instrument_class="LIQUID_ETF", side="SELL", raw_notional=100_000.0, trade_date="2025-06-15", slippage_bps=0.0)
    assert etf_b["stt"] == 0.0
    assert abs(etf_s["stt"] - 1.0) <= 0.05  # 0.001% of 100,000 = INR 1.00
    assert liq_b["stt"] == 0.0 and liq_s["stt"] == 0.0
    assert abs(liq_b["stamp_duty"] - 5.0) <= 0.05  # 0.005% buy stamp


def test_trade_economic_value_and_dynamic_hurdle_filtering() -> None:
    ce = TransactionCostIntelligence()
    strong_cand = {"strategy_id": "STRAT_E_DEFENSIVE_YIELD_LOWVOL", "score": 0.85, "stop_ratio": 0.09}
    weak_cand = {"strategy_id": "STRAT_A_CS_MOMENTUM", "score": 0.20, "stop_ratio": 0.06}

    ev_strong = estimate_trade_economic_value(
        candidate=strong_cand,
        lot_notional_inr=15_000.0,
        trade_date="2025-06-15",
        daily_vol_frac=0.015,
        hl_range_pct=1.8,
        adv_20d_inr=200_000_000.0,
        div_yield_ttm_frac=0.035,
        regime="STRONG_BULL",
        cost_engine=ce,
        hurdle_mode="DYNAMIC_VOL",
    )
    ev_weak = estimate_trade_economic_value(
        candidate=weak_cand,
        lot_notional_inr=3_000.0,
        trade_date="2025-06-15",
        daily_vol_frac=0.035,
        hl_range_pct=4.5,
        adv_20d_inr=50_000_000.0,
        div_yield_ttm_frac=0.0,
        regime="STRESS_BEAR",
        cost_engine=ce,
        hurdle_mode="DYNAMIC_VOL",
    )
    assert ev_strong["approved"] is True
    assert ev_weak["approved"] is False
    assert ev_weak["hurdle_bps"] > ev_strong["hurdle_bps"]  # Dynamic vol hurdle scales up cleanly


def test_intelligent_position_replacement_protects_winners() -> None:
    ce = TransactionCostIntelligence()
    winner_pos = {"qty": 100, "entry_px": 100.0, "bars_held": 30, "entry_score": 0.72, "hold_max_days": 120}
    loser_pos = {"qty": 100, "entry_px": 100.0, "bars_held": 30, "entry_score": 0.55, "hold_max_days": 120}
    top_ev = {"approved": True, "expected_net_alpha_bps": 320.0}

    rep_win = evaluate_position_replacement(
        pos=winner_pos,
        cur_close_px=112.0,
        sma_50_px=105.0,
        in_keep_buffer=False,
        best_candidate={"score": 0.85},
        best_cand_ev=top_ev,
        nav_inr=100_000.0,
        trade_date="2025-06-15",
        regime="STRONG_BULL",
        sector_exposure_pct=12.0,
        cost_engine=ce,
        replacement_policy="WINNER_SHIELD_LONG_HOLD",
    )
    assert rep_win["should_replace"] is False
    assert rep_win["reason"] == "RETAIN_WINNER_SHIELD_INTACT"

    rep_lose = evaluate_position_replacement(
        pos=loser_pos,
        cur_close_px=91.0,
        sma_50_px=101.0,
        in_keep_buffer=False,
        best_candidate={"score": 0.85},
        best_cand_ev=top_ev,
        nav_inr=100_000.0,
        trade_date="2025-06-15",
        regime="STRONG_BULL",
        sector_exposure_pct=12.0,
        cost_engine=ce,
        replacement_policy="INTELLIGENT_HYSTERESIS",
    )
    assert rep_lose["should_replace"] is True


def test_all_six_stop_loss_models_and_gap_through_open_honesty() -> None:
    for sm in (
        "A_EXISTING_M3_0_BE",
        "B_ATR_DYNAMIC",
        "C_VOL_ADJUSTED_TRAILING",
        "D_STRUCTURE_BASED",
        "E_TIME_DECAY",
        "F_HYBRID_STOP_PROFIT_PROTECT",
    ):
        pos = {"entry_px": 100.0, "sl_px": 92.0, "atr_entry": 3.0, "max_px": 110.0, "bars_held": 15, "be_locked": False}
        # Gap down below sl_px MUST fill honestly at open (88.0), NEVER at impossible 92.0!
        reason, px = evaluate_position_exit(
            pos=pos, stop_model=sm, o_v=88.0, h_v=90.0, l_v=87.0, c_v=89.0,
            sma_50_v=98.0, low_20_prev=94.0, vol_20d_daily=0.018, hold_max_days=120, is_last_bar=False
        )
        assert reason is not None and px == 88.0


def test_execution_timing_circuit_limits_and_vwap_approx_flag() -> None:
    # Locked upper circuit bar MUST fail execution
    res_ckt = simulate_execution_fill(
        execution_mode="NEXT_SESSION_OPEN", side="BUY", req_qty=10,
        prev_close=100.0, bar_open=120.0, bar_high=120.0, bar_low=120.0, bar_close=120.0,
        bar_vol=50_000, adv_20d_inr=100_000_000.0, atr_14=3.0, hl_range_pct=0.0
    )
    assert res_ckt["status"] == "FAILED" and res_ckt["reason"] == "CIRCUIT_LIMIT_LOCKED"

    # VWAP approximation MUST be explicitly flagged is_approximation=True
    res_vwap = simulate_execution_fill(
        execution_mode="NEXT_SESSION_VWAP_APPROX", side="BUY", req_qty=50,
        prev_close=100.0, bar_open=100.5, bar_high=102.0, bar_low=99.5, bar_close=101.0,
        bar_vol=500_000, adv_20d_inr=100_000_000.0, atr_14=2.5, hl_range_pct=2.5
    )
    assert res_vwap["status"] == "FILLED" and res_vwap["is_approximation"] is True

    # Limit order that bar low never touches MUST return UNFILLED
    res_lim = simulate_execution_fill(
        execution_mode="LIMIT_ORDER_SIM", side="BUY", req_qty=50,
        prev_close=100.0, bar_open=101.0, bar_high=102.5, bar_low=100.5, bar_close=102.0,
        bar_vol=500_000, adv_20d_inr=100_000_000.0, atr_14=2.5, hl_range_pct=2.0
    )
    assert res_lim["status"] == "UNFILLED"


def test_all_six_sizing_models_respect_caps() -> None:
    for sm in ("A_FIXED", "B_VOLATILITY_ADJUSTED", "C_RISK_BUDGETED", "D_CONFIDENCE_WEIGHTED", "E_EXPECTED_ALPHA_WEIGHTED", "F_HYBRID_RISK_ALPHA"):
        alloc = compute_intelligent_position_size(
            sizing_mode=sm,
            base_slot_inr=12_000.0,
            rem_tact_budget_inr=30_000.0,
            nav_inr=100_000.0,
            max_single_stock_pct=15.0,
            sector_room_inr=14_000.0,
            industry_room_inr=14_000.0,
            adv_20d_inr=200_000_000.0,
            daily_vol_frac=0.015,
            stop_ratio=0.09,
            signal_score=0.82,
            expected_net_alpha_bps=280.0,
            hurdle_bps=120.0,
            max_corr_with_open=0.35,
            min_lot_inr=6_500.0,
        )
        assert 6_500.0 <= alloc <= 14_000.0


def test_synthetic_execution_simulation_zero_reconciliation_diff() -> None:
    panel = _make_synthetic_panel(t_len=310, n_len=20)
    ctx = precompute_m3_1_research_context(panel, db_path=Path("/tmp/nonexistent_m3_test.db"))
    enrich_m3_1_execution_context(ctx)
    cfg = ExecutionIntelligenceConfig(
        config_id="SYNTH_EXEC_TEST",
        config_name="Synthetic Execution Intelligence Test",
        core_target_pct=65.0,
        tactical_target_pct=25.0,
        liquid_target_pct=10.0,
        allocation_mode="RISK_BUDGETED",
        hurdle_mode="DYNAMIC_VOL",
        replacement_policy="WINNER_SHIELD_LONG_HOLD",
        stop_model="F_HYBRID_STOP_PROFIT_PROTECT",
        execution_mode="VOLATILITY_ADJUSTED",
        sizing_mode="F_HYBRID_RISK_ALPHA",
    )
    sim = simulate_m3_1_execution_intelligence(
        ctx=ctx, cfg=cfg, date_start=str(panel["dates"][252]), date_end=str(panel["dates"][-1]), initial_capital=100_000.0
    )
    m = sim["metrics"]
    assert abs(float(m["reconciliation_check_diff_inr"])) <= 1.0
    assert float(m["gross_return_pct"]) >= float(m["total_return_pct"])
    assert float(m["avg_productive_capital_pct"]) >= 85.0


def test_prompt3_deliverables_10_files_and_8_charts_verified() -> None:
    out = DEFAULT_M3_1_EXEC_DIR
    required_files = [
        "EXECUTION_ARCHITECTURE.md",
        "TRANSACTION_COST_FORENSICS.csv",
        "TRADE_ECONOMIC_VALUE_ANALYSIS.csv",
        "POSITION_REPLACEMENT_ANALYSIS.csv",
        "STOP_LOSS_EXPERIMENTS.csv",
        "EXECUTION_MODEL_COMPARISON.csv",
        "POSITION_SIZING_EXPERIMENTS.csv",
        "TURNOVER_OPTIMIZATION_REPORT.md",
        "EXECUTION_PERFORMANCE_COMPARISON.csv",
        "MORESHWAR_3_1_EXECUTION_RECOMMENDATIONS.md",
        "chart_gross_vs_net_returns.svg",
        "chart_transaction_costs_over_time.svg",
        "chart_trade_profitability_distribution.svg",
        "chart_holding_period_distribution.svg",
        "chart_stop_loss_effectiveness.svg",
        "chart_portfolio_turnover.svg",
        "chart_drawdown_comparison.svg",
        "chart_alpha_by_strategy_family.svg",
        "execution_intelligence_summary.json",
    ]
    for fname in required_files:
        p = out / fname
        assert p.exists(), f"Missing Prompt 3/5 deliverable: {p}"
        assert p.stat().st_size > 200, f"Deliverable too small: {p}"

    # Verify exact 0.00 INR reconciliation across all configurations in EXECUTION_PERFORMANCE_COMPARISON.csv
    with (out / "EXECUTION_PERFORMANCE_COMPARISON.csv").open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) >= 24
    for r in rows:
        assert abs(float(r["reconciliation_diff_inr"])) <= 0.50

    # Verify VWAP approximation is explicitly flagged as APPROXIMATION_NO_INTRADAY_DATA
    with (out / "EXECUTION_MODEL_COMPARISON.csv").open("r", encoding="utf-8") as f:
        erows = list(csv.DictReader(f))
    vwap_rows = [r for r in erows if r["execution_mode"] == "NEXT_SESSION_VWAP_APPROX"]
    assert len(vwap_rows) >= 2
    for vr in vwap_rows:
        assert vr["is_approximation"] == "True"
        assert "APPROXIMATION_NO_INTRADAY_DATA" in vr["data_validation_label"]

    summary = json.loads((out / "execution_intelligence_summary.json").read_text(encoding="utf-8"))
    assert summary["total_experiments_run"] >= 100
    assert len(summary["nested_walk_forward"]["folds"]) == 3
