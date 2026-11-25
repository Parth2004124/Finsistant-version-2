"""Finsistant Version 2 — Unified All-in-One Platform.

Merges:
1. StockSight Version 2 (4-Pillar Fundamental Engine, Porter's 5 Forces, Moreshwar Levels, Decision Matrix)
2. Finsistant (Execution Terminal, Trade Queue, Auto-Execution Rules, Polynomial Regression, Stocky AI)
3. Multi-Agent Trading Orchestrator (AgentRunner: TechBot_1, FundBot_1, ExecBot_1)
"""

from .adversarial_prosecutor import DEFAULT_MIN_SURVIVAL_SCORE, prosecute_trade_candidate
from .agents import (
    AdversarialProsecutorAgent,
    AgentMessage,
    AgentRunner,
    ExecutionAgent,
    FundamentalAnalyst,
    PipelineResult,
    TechnicalAnalyst,
)
from .audit_ledger import (
    ENGINE_VERSIONS,
    build_audit_dashboard_view,
    load_audit_ledger,
    record_broker_execution,
    record_scan_decisions,
)
from .backtest_engine import run_backtest_and_attribution
from .backtest_engine_v2 import run_moreshwar_v2_research_suite
from .backtest_engine_v2_1 import run_moreshwar_v2_1_research_suite
from .config import FinsistantSettings, load_settings
from .gemini_critic import analyze_trade_with_gemini, apply_trade_recalibration, ask_gemini_stocky, critique_suggested_trades
from .historical_data import EXPANDED_RESEARCH_UNIVERSE, load_or_build_expanded_dataset
from .kite_auth import (
    auto_fetch_kite_access_token,
    exchange_request_token,
    get_kite_auth_status,
    resolve_kite_session,
)
from .live_data import DEFAULT_SYMBOL_CONFIG, fetch_benchmark_candles, fetch_kite_portfolio, fetch_live_market_snapshot
from .moreshwar_v2 import (
    MoreshwarV2Config,
    classify_market_regime,
    compute_atr,
    compute_confidence_adjustment,
    evaluate_moreshwar_v2,
)
from .moreshwar_v2_1 import (
    MoreshwarV21Config,
    classify_market_regime_6,
    compute_indian_delivery_costs,
    evaluate_moreshwar_v2_1,
)
from .nse_mcp import (
    fetch_nse_corporate_actions,
    fetch_nse_index_1y,
    fetch_nse_stock_history_1y,
    fetch_nse_stock_quote,
)
from .provenance import build_asset_provenance, build_data_integrity_report
from .quant_engine import ASSET_UNIVERSE, get_asset_full_analysis
from .scoring_engine import (
    calculate_decision_matrix,
    calculate_fundamental_score,
    calculate_moreshwar_levels,
    calculate_porters_score,
)
from .stocky_ai import StockyAssistant

__all__ = [
    "AdversarialProsecutorAgent",
    "DEFAULT_MIN_SURVIVAL_SCORE",
    "prosecute_trade_candidate",
    "AgentRunner",
    "AgentMessage",
    "PipelineResult",
    "TechnicalAnalyst",
    "FundamentalAnalyst",
    "ExecutionAgent",
    "ENGINE_VERSIONS",
    "build_audit_dashboard_view",
    "load_audit_ledger",
    "record_broker_execution",
    "record_scan_decisions",
    "run_backtest_and_attribution",
    "run_moreshwar_v2_research_suite",
    "run_moreshwar_v2_1_research_suite",
    "EXPANDED_RESEARCH_UNIVERSE",
    "load_or_build_expanded_dataset",
    "build_asset_provenance",
    "build_data_integrity_report",
    "MoreshwarV2Config",
    "classify_market_regime",
    "compute_atr",
    "compute_confidence_adjustment",
    "evaluate_moreshwar_v2",
    "MoreshwarV21Config",
    "classify_market_regime_6",
    "compute_indian_delivery_costs",
    "evaluate_moreshwar_v2_1",
    "fetch_benchmark_candles",
    "fetch_nse_corporate_actions",
    "fetch_nse_index_1y",
    "fetch_nse_stock_history_1y",
    "fetch_nse_stock_quote",
    "FinsistantSettings",
    "load_settings",
    "analyze_trade_with_gemini",
    "apply_trade_recalibration",
    "ask_gemini_stocky",
    "critique_suggested_trades",
    "auto_fetch_kite_access_token",
    "exchange_request_token",
    "get_kite_auth_status",
    "resolve_kite_session",
    "DEFAULT_SYMBOL_CONFIG",
    "fetch_kite_portfolio",
    "fetch_live_market_snapshot",
    "ASSET_UNIVERSE",
    "get_asset_full_analysis",
    "calculate_fundamental_score",
    "calculate_porters_score",
    "calculate_moreshwar_levels",
    "calculate_decision_matrix",
    "StockyAssistant",
]
