"""Finsistant Version 2 — Unified All-in-One Platform.

Merges:
1. StockSight Version 2 (4-Pillar Fundamental Engine, Porter's 5 Forces, Moreshwar Levels, Decision Matrix)
2. Finsistant (Execution Terminal, Trade Queue, Auto-Execution Rules, Polynomial Regression, Stocky AI)
3. Multi-Agent Trading Orchestrator (AgentRunner: TechBot_1, FundBot_1, ExecBot_1)
"""

from .agents import AgentMessage, AgentRunner, ExecutionAgent, FundamentalAnalyst, PipelineResult, TechnicalAnalyst
from .config import FinsistantSettings, load_settings
from .gemini_critic import analyze_trade_with_gemini, apply_trade_recalibration, ask_gemini_stocky, critique_suggested_trades
from .kite_auth import (
    auto_fetch_kite_access_token,
    exchange_request_token,
    get_kite_auth_status,
    resolve_kite_session,
)
from .live_data import DEFAULT_SYMBOL_CONFIG, fetch_kite_portfolio, fetch_live_market_snapshot
from .quant_engine import ASSET_UNIVERSE, get_asset_full_analysis
from .scoring_engine import (
    calculate_decision_matrix,
    calculate_fundamental_score,
    calculate_moreshwar_levels,
    calculate_porters_score,
)
from .stocky_ai import StockyAssistant

__all__ = [
    "AgentRunner",
    "AgentMessage",
    "PipelineResult",
    "TechnicalAnalyst",
    "FundamentalAnalyst",
    "ExecutionAgent",
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
