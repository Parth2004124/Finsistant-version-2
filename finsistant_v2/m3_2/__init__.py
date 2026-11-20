"""Moreshwar 3.2 — Orchestrator-Driven Prospective Paper Trading.

The multi-agent orchestrator (TechBot_1 + FundBot_1 + ExecBot_1) is the scanner: it scans the full
PIT-eligible NSE universe every session, its picks are the paper trades, and realised outcomes feed
back into the agents (see `orchestrator_scanner` and `learning_ledger`). Paper trading exists to
perfect the strategy that the orchestrator will eventually run with real money, so the two are never
isolated from each other.
"""

from .orchestrator_scanner import (  # noqa: F401
    ORCH_STRATEGY_ID,
    SCANNER_VERSION,
    OrchestratorScanConfig,
    OrchestratorUniverseScanner,
)
from .learning_ledger import OrchestratorLearningLedger  # noqa: F401
