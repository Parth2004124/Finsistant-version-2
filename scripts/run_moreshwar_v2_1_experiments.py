"""Reproducible 5-Year Research Runner for Moreshwar 2.1 (Never Auto-Promotes to Production).

Runs:
1. Expanded 5Y OHLCV dataset load/refresh across 40 instruments + Nifty 50 + delisted survivorship probes.
2. Full benchmark comparison (M1.0, M2.0, M2.1, M2.1 Permissive, E-Ref Leak Control, SMA-20, Fixed 5/10, Nifty 50, Cash).
3. Rolling Walk-Forward Validation + Completely Untouched 25% Out-of-Sample Test Split.
4. 1,200-run Monte Carlo (Trade Bootstrap + Daily Circular Block Bootstrap).
5. Strategy & Feature Attribution Investigation (including Accepted vs Rejected Counterfactuals).
6. 6-State Market Regime Evaluation & Portfolio-Level Risk Report.
7. Saves versioned report to `data/moreshwar_v2_1_experiment_report.json` (while preserving `data/moreshwar_v2_experiment_report.json` untouched).
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from finsistant_v2.backtest_engine_v2_1 import run_moreshwar_v2_1_research_suite
from finsistant_v2.historical_data import load_or_build_expanded_dataset
from finsistant_v2.moreshwar_v2_1 import MoreshwarV21Config


def main() -> None:
    t0 = time.time()
    force_refresh = "--refresh" in sys.argv
    dataset = load_or_build_expanded_dataset(force_refresh=force_refresh, include_fundamentals=True)
    cfg21 = MoreshwarV21Config()
    report = run_moreshwar_v2_1_research_suite(dataset, cfg_v21=cfg21, n_mc_sims=1200)

    out_path = ROOT_DIR / "data" / "moreshwar_v2_1_experiment_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # Compact console summary preserving all negative/positive findings
    elapsed = round(time.time() - t0, 2)
    ds = report["dataset_summary"]
    dq = report["data_quality_report"]
    print(f"=== Moreshwar 2.1 Research Suite Complete ({elapsed}s) ===")
    print(f"Window: {ds['start_date']} -> {ds['end_date']} ({ds['trading_days']} trading days) | Instruments: {ds['instruments_count']} | OOS: {ds['oos_start']} -> {ds['oos_end']}")
    print(f"Data Quality: 5Y={dq['complete_multi_year_count']}, Distressed={dq['distressed_or_illiquid_count']}, Partial={dq['partial_history_count']}, Delisted 404s={dq['delisted_probes_unavailable']}/{dq['delisted_probes_attempted']}")
    print(f"Promoted to Production: {report['promoted_to_production']}")
    print("\n--- 5Y Benchmark Comparison (Indian Delivery Costs + Gap/Liquidity Sim) ---")
    for m in report["benchmark_comparison"]:
        print(
            f"  {m['model']:46s} | Tot {m.get('total_return_pct', 0):+7.2f}% | CAGR {m.get('cagr_pct', 0):+6.2f}% | "
            f"Win {m.get('win_rate_pct', 0):5.1f}% | PF {str(m.get('profit_factor')):>4s} | Sh {str(m.get('sharpe_ratio')):>5s} | "
            f"DD {m.get('max_drawdown_pct', 0):5.2f}% | Trd {str(m.get('total_trades', 0)):>4s} | Exp {str(m.get('expectancy_pct')):>6s}% | p={m.get('p_value_vs_zero', '--')}"
        )

    print("\n--- Untouched Out-of-Sample Split ---")
    for m in report["out_of_sample"]["comparison"]:
        print(
            f"  {m['model']:36s} | Tot {m.get('total_return_pct', 0):+6.2f}% | CAGR {m.get('cagr_pct', 0):+6.2f}% | "
            f"Sh {str(m.get('sharpe_ratio')):>5s} | DD {m.get('max_drawdown_pct', 0):5.2f}% | Trd {str(m.get('total_trades', 0)):>3s}"
        )

    mc = report["monte_carlo"]["moreshwar_2_1"]
    print("\n--- Monte Carlo (1,200 Sims) ---")
    print("  Trade Bootstrap:", mc["trade_bootstrap"])
    print("  Block Bootstrap:", mc["daily_block_bootstrap"])

    inv = report["strategy_investigation"]
    print("\n--- Selectivity & Counterfactuals ---")
    print(" ", inv["selectivity_counterfactuals"])
    print("  E-Ref Leak Corr (FundScore vs Past 4Y):", inv["eref_leakage_audit"]["static_score_vs_past_4y_corr"])

    print("\n--- Final Classification ---")
    print(" ", report["final_decision"]["classification"])
    print(" ", report["final_decision"]["rationale"])


if __name__ == "__main__":
    main()
