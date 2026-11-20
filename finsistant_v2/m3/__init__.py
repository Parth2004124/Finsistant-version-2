"""Moreshwar 3.0 — Point-in-Time Indian (NSE) Quantitative Research Data & Strategy Infrastructure.

Completely isolated from frozen Moreshwar 2.1 and 2.2 engines/datasets.
Provides:
- Prompt 1: Authoritative NSE security master, SCD Type-2 symbol history, PIT daily OHLCV + delivery metrics + corporate actions, DQ auditing, and causal liquidity eligibility.
- Prompt 2: 31-factor causal discovery lab with Benjamini-Hochberg FDR, Newey-West HAC t-stats, Fama-MacBeth, block bootstrap, and strict IS/OOS validation.
- Prompt 3: Five independent strategy families (A..E) on a stateless fair-comparison portfolio harness with exact cash-to-NAV reconciliation.
- Prompt 4: Five causal market regime intelligence models (Rule-Based 5-Core with Hysteresis, Rule-9Fine, K-Means-4, Causal Forward-Filtered 3-State HMM, Vol-State, CUSUM) and adaptive regime-conditioned strategy allocation.
- Prompt 8: Historical Zerodha/NSE/SEBI statutory tariff & square-root market impact execution engine with paper-only broker isolation.
"""

from .builder import build_m3_pit_universe
from .execution_cost_engine import (
    HistoricalTariffSchedule,
    calculate_historical_indian_costs,
    compute_market_impact_and_slippage_bps,
)
from .execution_report import generate_execution_and_security_report
from .factor_research import run_moreshwar_v3_factor_research
from .factors import FACTOR_SPECS, REJECTED_NON_PIT_FACTORS, compute_all_m3_factors
from .independent_audit_p9 import (
    compute_statistical_and_bootstrap_audit,
    run_concentration_and_outlier_removal_audit,
    run_moreshwar_v3_independent_audit_p9,
)
from .liquidity_engine import PITUniverseStore
from .panel_loader import build_or_load_m3_panel
from .paper_trading_engine import (
    MoreshwarV3PaperTradingEngine,
    RELEASE_VERSION_M3,
    build_moreshwar_v3_dashboard_and_release_bundle,
    run_operational_health_and_monitoring_checks,
)
from .portfolio_construction import (
    PortfolioConstructionConfig,
    compute_euler_risk_contributions,
    run_moreshwar_v3_portfolio_construction,
    simulate_shared_capital_portfolio_m3,
)
from .regime_intelligence import (
    REGIME_ALLOCATION_POLICY,
    compute_causal_market_features,
    run_adaptive_regime_allocation,
    run_regime_detection_comparison,
)
from .relative_value_hedged import (
    PairCointegrationRecord,
    discover_cointegrated_pairs_in_sample,
    run_moreshwar_v3_relative_value_research,
    simulate_cointegrated_pairs_strategy,
    simulate_nifty_futures_hedged_overlay,
    simulate_sector_relative_and_market_neutral,
)
from .schema import (
    DEFAULT_M3_DATA_DIR,
    DEFAULT_M3_DB_PATH,
    DEFAULT_M3_REPORTS_DIR,
    connect_m3_db,
    init_m3_schema,
)
from .strategy_base import StrategySimConfig, simulate_independent_strategy_m3
from .strategy_families import (
    get_strategy_family_configs,
    prepare_strategy_signal_workspace,
    run_moreshwar_v3_strategy_families,
)
from .strategy_orchestrator import (
    SleeveRuntimeState,
    StrategyRegistryEntry,
    build_m3_strategy_registry,
    compute_orchestrator_weights_at_date,
    run_moreshwar_v3_strategy_orchestrator,
    simulate_strategy_orchestration_m3,
    update_sleeve_suspension_state,
)

__all__ = [
    "DEFAULT_M3_DATA_DIR",
    "DEFAULT_M3_DB_PATH",
    "DEFAULT_M3_REPORTS_DIR",
    "FACTOR_SPECS",
    "HistoricalTariffSchedule",
    "MoreshwarV3PaperTradingEngine",
    "PITUniverseStore",
    "PairCointegrationRecord",
    "PortfolioConstructionConfig",
    "REGIME_ALLOCATION_POLICY",
    "REJECTED_NON_PIT_FACTORS",
    "RELEASE_VERSION_M3",
    "SleeveRuntimeState",
    "StrategyRegistryEntry",
    "StrategySimConfig",
    "build_m3_pit_universe",
    "build_m3_strategy_registry",
    "build_moreshwar_v3_dashboard_and_release_bundle",
    "build_or_load_m3_panel",
    "calculate_historical_indian_costs",
    "compute_all_m3_factors",
    "compute_causal_market_features",
    "compute_euler_risk_contributions",
    "compute_market_impact_and_slippage_bps",
    "compute_orchestrator_weights_at_date",
    "compute_statistical_and_bootstrap_audit",
    "connect_m3_db",
    "discover_cointegrated_pairs_in_sample",
    "generate_execution_and_security_report",
    "get_strategy_family_configs",
    "init_m3_schema",
    "prepare_strategy_signal_workspace",
    "run_adaptive_regime_allocation",
    "run_concentration_and_outlier_removal_audit",
    "run_moreshwar_v3_factor_research",
    "run_moreshwar_v3_independent_audit_p9",
    "run_moreshwar_v3_portfolio_construction",
    "run_moreshwar_v3_relative_value_research",
    "run_moreshwar_v3_strategy_families",
    "run_moreshwar_v3_strategy_orchestrator",
    "run_operational_health_and_monitoring_checks",
    "run_regime_detection_comparison",
    "simulate_cointegrated_pairs_strategy",
    "simulate_independent_strategy_m3",
    "simulate_nifty_futures_hedged_overlay",
    "simulate_sector_relative_and_market_neutral",
    "simulate_shared_capital_portfolio_m3",
    "simulate_strategy_orchestration_m3",
    "update_sleeve_suspension_state",
]

