#!/usr/bin/env python3
"""Unified Verification Script for Finsistant Version 2 (All-in-One Platform)."""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from finsistant_v2 import (
    AgentRunner,
    ExecutionAgent,
    FundamentalAnalyst,
    TechnicalAnalyst,
    get_asset_full_analysis,
)


def main() -> None:
    tech_agent = TechnicalAnalyst(name="TechBot_1")
    fund_agent = FundamentalAnalyst(name="FundBot_1")
    exec_agent = ExecutionAgent(name="ExecBot_1")

    runner = AgentRunner(agents=[tech_agent, fund_agent, exec_agent])

    print("Verifying Finsistant Version 2 (All-in-One) platform status...")
    status = runner.health_check()
    print(f"Health Status:\n{json.dumps(status, indent=2)}")

    snap = get_asset_full_analysis("RELIANCE", is_holding=True)
    run = runner.run_pipeline(symbol="RELIANCE", timeframe="1D", capital=500000.0, active_trades=2, is_holding=True)
    print(
        f"\nUnified Pipeline Verification [RELIANCE]:\n"
        f"  - StockSight V2 Fundamental Score : {snap['fundamental_score']['total']}/100\n"
        f"  - Porter's 5 Forces Moat Score    : {snap['porters_score']['total']}/100\n"
        f"  - Moreshwar Target / Stop-Loss    : Target ₹{snap['moreshwar_levels']['target']} | SL ₹{snap['moreshwar_levels']['stop_loss']}\n"
        f"  - Finsistant 30D Regression Target: ₹{snap['projected_30d_target']} ({snap['regression_curve']})\n"
        f"  - Multi-Agent Final Verdict       : {run.final_action} ({run.decision_badge}) | Conviction {run.conviction_score*100:.1f}% | Order {run.order_receipt.get('order_id')}"
    )


if __name__ == "__main__":
    main()
