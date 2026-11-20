"""Moreshwar 3.1 — Forensic Audit & Intelligent Capital Allocation Research Package."""

from .allocation_experiments import (
    DEFAULT_M3_1_ALLOC_DIR,
    run_moreshwar_v3_1_capital_allocation_suite,
)
from .capital_allocation_engine import (
    CapitalAllocationConfig,
    precompute_m3_1_research_context,
    simulate_m3_1_capital_allocation,
)
from .dividend_accounting import (
    apply_ex_dividend_accounting_and_stop_adjustment,
    compute_split_adjusted_div_per_adj_share,
    compute_split_adjusted_div_yield_ttm,
    load_real_nse_etf_bars,
    prepare_m3_1_signal_workspace,
)
from .execution_simulation_engine import (
    ExecutionIntelligenceConfig,
    enrich_m3_1_execution_context,
    simulate_m3_1_execution_intelligence,
)
from .forensic_audit import run_moreshwar_v3_1_forensic_audit
from .sector_hierarchy import (
    UNKNOWN_INDUSTRY_LABEL,
    UNKNOWN_SECTOR_LABEL,
    build_4tier_sector_classification,
)
from .adaptive_strategy_engine import (
    AdaptiveRegimePolicy,
    analyze_family_e_by_regime,
    get_standard_adaptive_policies,
    simulate_m3_1_adaptive_regime_portfolio,
)
from .market_regime_engine import (
    REGIME_STATES_6,
    classify_rule_based_6state,
    classify_simple_trend_baseline,
    compute_causal_regime_features,
    fit_and_predict_statistical_regime_model,
)
from .regime_monitoring import (
    MonitoringThresholds,
    evaluate_regime_and_strategy_health,
)
from .transaction_cost_intelligence import (
    IndianTariffRegimeSchedule,
    TransactionCostIntelligence,
)
from .paper_trading_orchestrator import (
    DEFAULT_PROSPECTIVE_REPORTS_DIR,
    PaperTradingOrchestrator,
    build_pit_snapshot_from_context,
    validate_market_data_snapshot,
)
from .paper_trading_risk_monitor import (
    PaperRiskAlertThresholds,
    PaperTradingRiskMonitor,
)
from .paper_trading_scheduler import (
    DEFAULT_NSE_HOLIDAYS,
    PaperTradingScheduler,
    is_eligible_nse_trading_day,
)
from .prospective_paper_trading_engine import (
    ConfigIntegrityError,
    DEFAULT_FROZEN_CONFIG_PATH,
    DEFAULT_PAPER_DB_PATH,
    EXPECTED_CONFIG_SHA256_PREFIX,
    PaperOnlyExecutionGuard,
    PaperTradingSafetyError,
    ProspectivePaperTradingEngine,
    RELEASE_VERSION_M3_1,
    ReconciliationFailureError,
    VerifiedFrozenConfig,
    compute_config_hashes,
    load_and_verify_frozen_config,
)

__all__ = [
    "AdaptiveRegimePolicy",
    "CapitalAllocationConfig",
    "ConfigIntegrityError",
    "DEFAULT_FROZEN_CONFIG_PATH",
    "DEFAULT_M3_1_ALLOC_DIR",
    "DEFAULT_NSE_HOLIDAYS",
    "DEFAULT_PAPER_DB_PATH",
    "DEFAULT_PROSPECTIVE_REPORTS_DIR",
    "EXPECTED_CONFIG_SHA256_PREFIX",
    "ExecutionIntelligenceConfig",
    "IndianTariffRegimeSchedule",
    "MonitoringThresholds",
    "PaperOnlyExecutionGuard",
    "PaperRiskAlertThresholds",
    "PaperTradingOrchestrator",
    "PaperTradingRiskMonitor",
    "PaperTradingScheduler",
    "PaperTradingSafetyError",
    "ProspectivePaperTradingEngine",
    "REGIME_STATES_6",
    "RELEASE_VERSION_M3_1",
    "ReconciliationFailureError",
    "TransactionCostIntelligence",
    "UNKNOWN_INDUSTRY_LABEL",
    "UNKNOWN_SECTOR_LABEL",
    "VerifiedFrozenConfig",
    "analyze_family_e_by_regime",
    "apply_ex_dividend_accounting_and_stop_adjustment",
    "build_4tier_sector_classification",
    "build_pit_snapshot_from_context",
    "classify_rule_based_6state",
    "classify_simple_trend_baseline",
    "compute_causal_regime_features",
    "compute_config_hashes",
    "compute_split_adjusted_div_per_adj_share",
    "compute_split_adjusted_div_yield_ttm",
    "enrich_m3_1_execution_context",
    "evaluate_regime_and_strategy_health",
    "fit_and_predict_statistical_regime_model",
    "get_standard_adaptive_policies",
    "is_eligible_nse_trading_day",
    "load_and_verify_frozen_config",
    "load_real_nse_etf_bars",
    "precompute_m3_1_research_context",
    "prepare_m3_1_signal_workspace",
    "run_moreshwar_v3_1_capital_allocation_suite",
    "run_moreshwar_v3_1_forensic_audit",
    "simulate_m3_1_adaptive_regime_portfolio",
    "simulate_m3_1_capital_allocation",
    "simulate_m3_1_execution_intelligence",
    "validate_market_data_snapshot",
]
