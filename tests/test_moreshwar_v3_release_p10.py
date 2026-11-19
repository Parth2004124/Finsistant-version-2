"""Unit, Integration, Security & UI Parity Tests for Moreshwar 3.0 Prompt 10 (Final Integration, Paper Trading & Release Candidate)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from finsistant_v2.m3.execution_cost_engine import PaperOnlyZerodhaBrokerAdapter
from finsistant_v2.m3.paper_trading_engine import (
    MoreshwarV3PaperTradingEngine,
    RELEASE_VERSION_M3,
    run_operational_health_and_monitoring_checks,
)
from tests.test_moreshwar_v3_strategies_and_regime import _make_synthetic_panel


def test_paper_only_zerodha_broker_adapter_blocks_live_orders() -> None:
    adapter = PaperOnlyZerodhaBrokerAdapter()
    assert adapter.LIVE_ORDER_SUBMISSION_ENABLED is False
    posture = adapter.security_audit_snapshot()
    assert posture["paper_mode"] is True
    assert posture["live_execution_enabled"] is False
    assert posture["LIVE_ORDER_SUBMISSION_ENABLED"] is False

    paper_order = adapter.construct_paper_order(
        symbol="RELIANCE",
        side="BUY",
        qty=5,
        order_type="MARKET",
        product="CNC",
        reference_price=1300.0,
        strategy_id="FAM_A_BREAKOUT_TREND",
        signal_date="2025-10-15",
    )
    assert paper_order["mode"] == "PAPER_SIMULATION"
    assert paper_order["status"] == "PAPER_ACCEPTED"
    assert paper_order["paper_order_id"].startswith("PAPER-")

    with pytest.raises(PermissionError, match="HARD_SECURITY_BOUNDARY"):
        adapter.submit_live_order({"symbol": "RELIANCE", "qty": 5, "side": "BUY"})


def test_paper_trading_engine_synthetic_cycle_and_operational_checks(tmp_path: Path) -> None:
    panel = _make_synthetic_panel(t_len=310, n_len=20)
    engine = MoreshwarV3PaperTradingEngine(panel=panel, state_dir=tmp_path)
    state = engine.run_paper_trading_cycle(initial_capital=100_000.0)

    assert state["release_version"] == RELEASE_VERSION_M3
    assert "state_checksum_sha256" in state and len(state["state_checksum_sha256"]) == 64
    assert (tmp_path / "paper_trading_state_snapshot.json").exists()

    ledger = state["paper_account_ledger"]
    assert ledger["account_mode"] == "PAPER_TRADING_SIMULATED"
    assert abs(float(ledger["reconciliation_discrepancy_inr"])) <= 0.05


    op = run_operational_health_and_monitoring_checks(state, engine.broker_adapter)
    assert op["data_freshness_check"]["status"] == "HEALTHY"
    assert op["four_ledger_reconciliation_check"]["status"] == "PASS"
    assert op["zerodha_security_boundary_check"]["status"] == "PASS"
    assert op["zerodha_security_boundary_check"]["live_submission_raises_permission_error"] is True



def test_release_candidate_artifacts_and_dashboard_ui_parity() -> None:
    root = Path(__file__).resolve().parent.parent
    rc_path = root / "reports" / "moreshwar_3_0" / "moreshwar_v3_release_candidate.json"
    assert rc_path.exists(), "moreshwar_v3_release_candidate.json must exist"
    rc = json.loads(rc_path.read_text(encoding="utf-8"))

    assert rc["release_version"] == RELEASE_VERSION_M3
    rd = rc["release_decision"]
    assert "CONDITIONAL GO" in rd["release_outcome"]
    assert rd["paper_trading_authorized"] is True
    assert rd["live_capital_deployment_authorized"] is False
    assert rd["promoted_to_production"] is False

    for key in (
        "research_dashboard",
        "portfolio_dashboard",
        "execution_dashboard",
        "risk_dashboard",
        "validation_dashboard",
    ):
        assert key in rc, f"Missing dashboard section {key}"

    # Verify index.html and app.js contain Moreshwar 3.0 dashboard hooks across root, web/, and mobile-app/
    root_html = (root / "index.html").read_text(encoding="utf-8")
    web_html = (root / "web" / "index.html").read_text(encoding="utf-8")
    mob_html = (root / "mobile-app" / "app" / "src" / "main" / "assets" / "index.html").read_text(encoding="utf-8")
    assert "intelSectionV30" in root_html
    assert root_html == web_html == mob_html

    root_js = (root / "app.js").read_text(encoding="utf-8")
    web_js = (root / "web" / "app.js").read_text(encoding="utf-8")
    mob_js = (root / "mobile-app" / "app" / "src" / "main" / "assets" / "app.js").read_text(encoding="utf-8")
    assert "renderMoreshwarV30Dashboard" in root_js
    assert root_js == web_js == mob_js

