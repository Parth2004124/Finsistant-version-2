"""Automated Verification Suite for Moreshwar 3.1 (Prompt 1/5) Independent Forensic Performance Audit."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from finsistant_v2.m3_1.forensic_audit import (
    DEFAULT_M3_1_AUDIT_DIR,
    _compute_comprehensive_metrics,
    _run_extended_outlier_forensics,
)
import numpy as np


def test_forensic_audit_deliverables_exist_and_non_empty() -> None:
    audit_dir = DEFAULT_M3_1_AUDIT_DIR
    required_files = [
        "FORENSIC_AUDIT_REPORT.md",
        "BASELINE_REPRODUCTION.csv",
        "TRADE_LEVEL_ATTRIBUTION.csv",
        "TRANSACTION_COST_BREAKDOWN.csv",
        "BENCHMARK_RECONSTRUCTION.csv",
        "STRATEGY_PERFORMANCE_METRICS.csv",
        "DATA_INTEGRITY_FINDINGS.md",
        "CAPITAL_EFFICIENCY_ANALYSIS.md",
        "MORESHWAR_3_1_RECOMMENDATIONS.md",
        "forensic_audit_summary.json",
    ]
    for fname in required_files:
        fpath = audit_dir / fname
        assert fpath.exists(), f"Missing required forensic audit deliverable: {fpath}"
        assert fpath.stat().st_size > 200, f"Deliverable {fpath} is unexpectedly small"


def test_baseline_reproduction_and_cost_forensics_accuracy() -> None:
    summary_path = DEFAULT_M3_1_AUDIT_DIR / "forensic_audit_summary.json"
    data = json.loads(summary_path.read_text(encoding="utf-8"))

    # Verify dataset version, seed, and initial capital
    assert data["dataset_version"] == "m3_20261119T191710Z"
    assert data["random_seed"] == 20261119
    assert data["initial_capital_inr"] == 100_000.0

    by_id = {r["entity_id"]: r for r in data["baseline_reproduction"]}
    # Exact reproduction of PORT_RISK_BUDGETED (with & without sweep) and CONSTRAINED_ADAPTIVE_ORCHESTRATOR
    assert by_id["PORT_RISK_BUDGETED_CASH_SWEEP"]["reproduced_full_return_pct"] == 29.76
    assert by_id["PORT_RISK_BUDGETED_CASH_SWEEP"]["diff_vs_frozen_json_report_pp"] == 0.0
    assert by_id["PORT_RISK_BUDGETED_UNINVESTED"]["reproduced_full_return_pct"] == 7.43
    assert by_id["PORT_RISK_BUDGETED_UNINVESTED"]["diff_vs_frozen_json_report_pp"] == 0.0
    assert by_id["CONSTRAINED_ADAPTIVE_ORCHESTRATOR"]["reproduced_full_return_pct"] == 14.83
    assert by_id["CONSTRAINED_ADAPTIVE_ORCHESTRATOR"]["diff_vs_frozen_json_report_pp"] == 0.0

    # Discrepancy identification for Family E (+26.15% actual code vs +45.47% chat summary) and Family A (+10.97% actual code vs +11.98% chat summary)
    assert by_id["STRAT_E_DEFENSIVE_YIELD_LOWVOL"]["reproduced_full_return_pct"] == 26.15
    assert by_id["STRAT_E_DEFENSIVE_YIELD_LOWVOL"]["diff_vs_frozen_json_report_pp"] == 0.0
    assert by_id["STRAT_E_DEFENSIVE_YIELD_LOWVOL"]["diff_vs_chat_reported_pp"] == -19.32

    assert by_id["STRAT_A_CS_MOMENTUM"]["reproduced_full_return_pct"] == 10.97
    assert by_id["STRAT_A_CS_MOMENTUM"]["diff_vs_frozen_json_report_pp"] == 0.0
    assert by_id["STRAT_A_CS_MOMENTUM"]["diff_vs_chat_reported_pp"] == -1.01

    # Verify transaction cost breakdown invariants
    cf_port = data["transaction_cost_forensics"]["PORT_RISK_BUDGETED"]
    assert cf_port["total_trades"] == 238
    assert cf_port["brokerage_inr"] == 0.0
    assert abs(
        (cf_port["statutory_charges_total_inr"] + cf_port["slippage_total_inr"]) - cf_port["total_all_costs_inr"]
    ) <= 0.05
    assert cf_port["uncredited_cash_dividends_inr"] > 0.0


def test_csv_schemas_and_extended_outlier_counterfactuals() -> None:
    audit_dir = DEFAULT_M3_1_AUDIT_DIR
    with (audit_dir / "STRATEGY_PERFORMANCE_METRICS.csv").open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) >= 45  # 16 entities x 3 windows
    req_cols = {
        "cagr_pct",
        "total_return_pct",
        "annualized_vol_pct",
        "max_drawdown_pct",
        "sharpe_vs_65_rf",
        "sortino_vs_65_rf",
        "calmar_ratio",
        "profit_factor",
        "win_rate_pct",
        "expectancy_pct",
        "avg_exposure_pct",
        "turnover_multiple_x",
        "time_in_market_pct",
        "max_recovery_duration_days",
        "rolling_12m_median_pct",
    }
    assert req_cols.issubset(set(rows[0].keys()))

    # Test synthetic extended outlier helper
    navs = [{"date": f"2024-01-{i+1:02d}", "nav": 100000.0 + i * 100.0, "ret": 0.001} for i in range(20)]
    trades = [
        {"symbol": "A", "entry_date": "2024-01-01", "exit_date": "2024-01-05", "pnl_amount": 5000.0, "net_return_pct": 10.0},
        {"symbol": "B", "entry_date": "2024-01-02", "exit_date": "2024-01-06", "pnl_amount": 3000.0, "net_return_pct": 6.0},
        {"symbol": "C", "entry_date": "2024-01-03", "exit_date": "2024-01-07", "pnl_amount": -2000.0, "net_return_pct": -4.0},
    ]
    out = _run_extended_outlier_forensics(strategy_id="TEST", daily_navs=navs, trades=trades)
    assert out["baseline_net_pnl_inr"] == 6000.0
    assert out["remove_top_1_trade"]["remaining_pnl_inr"] == 1000.0
    assert out["remove_top_3_trades"]["remaining_pnl_inr"] == -2000.0
