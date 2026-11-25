"""Moreshwar 3.1 (Prompt 5/5) — Prospective Forward-Only Paper-Trading Engine & Durable SQLite Ledger.

Converts the frozen Moreshwar 3.1 research artifacts (Prompts 1–4) into a deterministic,
restart-safe, idempotent, auditable PAPER-TRADING engine with multi-layer fail-closed safety guards.

Non-negotiable architectural invariants:
1. Immutable Frozen Configuration:
   - Authoritative config: `reports/moreshwar_3_1/strategy_intelligence/FROZEN_PROSPECTIVE_PAPER_TRADING_CONFIG.json`
   - Verified canonical payload SHA256 prefix: `47a30f4bbd004c5f`
   - Any hash mismatch or parameter tampering halts execution and triggers `SAFE_MODE`.
2. Hard Paper-Only Execution Boundary:
   - `live_trading_enabled = False`
   - `broker_order_placement_enabled = False`
   - `automatic_capital_deployment_enabled = False`
   - Any attempt to enable live execution, load a live broker adapter, or activate suspended
     strategies (`STRAT_A_CS_MOMENTUM`) fails closed with `PermissionError`.
3. Strict Separation of Observation Modes:
   - `HISTORICAL_RESEARCH`: frozen Prompt 1–4 archive (`<= 2026-11-19`) — NEVER counts toward prospective target.
   - `HISTORICAL_VALIDATION`: synthetic/replay verification — NEVER counts toward prospective target.
   - `PROSPECTIVE_PAPER`: strictly unseen sessions AFTER `prospective_start_date` — ONLY these count toward
     the mandatory `60` minimum / `120` target unseen trading sessions.
4. Exact Point-in-Time Causality & Accounting:
   - Signals computed after session `T` close execute strictly at session `T+1` open (`NEXT_SESSION_OPEN`).
   - Exact Indian statutory & CDSL DP tariffs via `TransactionCostIntelligence` (charged once per order/day).
   - T+1 settlement, split-adjusted ex-dividend crediting, Liquid BeES yield accrual, and `<= INR 0.01`
     four-ledger reconciliation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sqlite3
from types import MappingProxyType
from typing import Any

import numpy as np

from finsistant_v2.m3_1.adaptive_strategy_engine import AdaptiveRegimePolicy
from finsistant_v2.m3_1.dividend_accounting import (
    LIQUID_BEES_DAILY_YIELD,
    apply_ex_dividend_accounting_and_stop_adjustment,
)
from finsistant_v2.m3_1.execution_simulation_engine import ExecutionIntelligenceConfig
from finsistant_v2.m3_1.market_regime_engine import classify_simple_trend_baseline
from finsistant_v2.m3_1.trade_quality_models import (
    compute_intelligent_position_size,
    estimate_trade_economic_value,
    evaluate_position_exit,
    simulate_execution_fill,
)
from finsistant_v2.m3_1.transaction_cost_intelligence import (
    IndianTariffRegimeSchedule,
    TransactionCostIntelligence,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_FROZEN_CONFIG_PATH = (
    REPO_ROOT
    / "reports"
    / "moreshwar_3_1"
    / "strategy_intelligence"
    / "FROZEN_PROSPECTIVE_PAPER_TRADING_CONFIG.json"
)
DEFAULT_PAPER_DATA_DIR = REPO_ROOT / "data" / "m3_1_paper_trading"
DEFAULT_PAPER_DB_PATH = DEFAULT_PAPER_DATA_DIR / "prospective_paper_ledger.db"

EXPECTED_CONFIG_SHA256_PREFIX = "47a30f4bbd004c5f"
HISTORICAL_DB_END_DATE = "2026-11-19"
DIAGNOSTIC_INDEX_DB_END_DATE = "2025-10-10"
INITIAL_PAPER_CAPITAL_INR = 100_000.0
MIN_PROSPECTIVE_SESSIONS = 60
TARGET_PROSPECTIVE_SESSIONS = 120
RECONCILIATION_TOLERANCE_INR = 0.01
RELEASE_VERSION_M3_1 = "3.1.0-rc1"

ALLOWED_OBSERVATION_MODES = frozenset({
    "HISTORICAL_RESEARCH",
    "HISTORICAL_VALIDATION",
    "PROSPECTIVE_PAPER",
})

ALLOWED_ENGINE_STATES = frozenset({
    "IDLE",
    "RUNNING",
    "PAUSED",
    "STOPPED",
    "SAFE_MODE",
})

SUSPENDED_STRATEGIES_HARD = frozenset({
    "STRAT_A_CS_MOMENTUM",
    "FAM_A_BREAKOUT_TREND",
    "FAM_A",
    "FAMILY_A",
})


class PaperTradingSafetyError(PermissionError):
    """Raised whenever any prohibited live-trading, config-tampering, or suspended-strategy action is attempted."""


class ConfigIntegrityError(ValueError):
    """Raised when the frozen prospective paper-trading configuration fails SHA256 or schema verification."""


class ReconciliationFailureError(RuntimeError):
    """Raised when the four-ledger accounting reconciliation exceeds INR 0.01."""


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _freeze_mapping(obj: Any) -> Any:
    """Recursively wrap dicts in read-only MappingProxyType and lists in tuples."""
    if isinstance(obj, dict):
        return MappingProxyType({k: _freeze_mapping(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return tuple(_freeze_mapping(v) for v in obj)
    return obj


def _unfreeze_mapping(obj: Any) -> Any:
    """Convert frozen MappingProxyType/tuple back to plain JSON-serializable Python types."""
    if isinstance(obj, MappingProxyType):
        return {k: _unfreeze_mapping(v) for k, v in obj.items()}
    if isinstance(obj, tuple):
        return [_unfreeze_mapping(v) for v in obj]
    return obj


def compute_config_hashes(config_path: Path = DEFAULT_FROZEN_CONFIG_PATH) -> dict[str, Any]:
    """Compute both canonical payload SHA256 (Prompt 4 spec) and raw file byte SHA256."""
    if not config_path.exists():
        raise ConfigIntegrityError(f"Frozen configuration file not found: {config_path}")
    raw_bytes = config_path.read_bytes()
    raw_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    try:
        parsed = json.loads(raw_bytes.decode("utf-8"))
    except Exception as exc:
        raise ConfigIntegrityError(f"Invalid JSON in frozen config: {exc}") from exc

    payload_without_prefix = {k: v for k, v in parsed.items() if k != "config_sha256_prefix"}
    canonical_sha256 = hashlib.sha256(
        json.dumps(payload_without_prefix, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return {
        "config_path": str(config_path),
        "raw_file_sha256": raw_sha256,
        "canonical_payload_sha256": canonical_sha256,
        "canonical_prefix_16": canonical_sha256[:16],
        "stored_prefix": str(parsed.get("config_sha256_prefix", "")),
        "parsed": parsed,
    }


@dataclass(frozen=True)
class VerifiedFrozenConfig:
    """Immutable, cryptographically verified prospective paper-trading configuration."""

    config_path: str
    canonical_sha256: str
    raw_file_sha256: str
    expected_prefix: str
    verified_at_utc: str
    raw_config: MappingProxyType
    exec_cfg: ExecutionIntelligenceConfig
    regime_policy: AdaptiveRegimePolicy
    tariff_schedule: IndianTariffRegimeSchedule

    # Hard immutable safety constants — NEVER mutable!
    live_trading_enabled: bool = False
    broker_order_placement_enabled: bool = False
    automatic_capital_deployment_enabled: bool = False

    def to_integrity_dict(self) -> dict[str, Any]:
        return {
            "verified": True,
            "config_path": self.config_path,
            "expected_sha256_prefix": self.expected_prefix,
            "canonical_payload_sha256": self.canonical_sha256,
            "canonical_prefix_16": self.canonical_sha256[:16],
            "raw_file_sha256": self.raw_file_sha256,
            "verified_at_utc": self.verified_at_utc,
            "selected_policy_id": self.regime_policy.policy_id,
            "execution_config_id": self.exec_cfg.config_id,
            "admitted_tactical_strategies": list(self.exec_cfg.active_sleeves),
            "suspended_strategies": list(self.raw_config.get("suspended_strategies", ())),
            "live_trading_enabled": False,
            "broker_order_placement_enabled": False,
            "automatic_capital_deployment_enabled": False,
            "instrument_mapping_note": (
                "Frozen config maps G_SIMPLE_TREND_BASELINE to NIFTYBEES (Core 60%/45%), "
                "STRAT_E_DEFENSIVE_YIELD_LOWVOL (Tactical Equity 30%/10%), and LIQUIDBEES (Liquid 10%/45%). "
                "No Gold ETF instrument exists in the frozen configuration."
            ),
        }


def load_and_verify_frozen_config(
    config_path: Path = DEFAULT_FROZEN_CONFIG_PATH,
    *,
    expected_prefix: str = EXPECTED_CONFIG_SHA256_PREFIX,
) -> VerifiedFrozenConfig:
    """Load, hash-verify, and structurally validate the immutable frozen prospective config."""
    info = compute_config_hashes(config_path)
    canon_sha = info["canonical_payload_sha256"]
    raw_sha = info["raw_file_sha256"]
    parsed = info["parsed"]

    if not canon_sha.startswith(expected_prefix):
        raise ConfigIntegrityError(
            f"CONFIG_HASH_MISMATCH: expected canonical SHA256 prefix '{expected_prefix}', "
            f"computed '{canon_sha[:16]}' (full={canon_sha}). Refusing to proceed."
        )
    if str(parsed.get("config_sha256_prefix", "")) != expected_prefix:
        raise ConfigIntegrityError(
            f"CONFIG_STORED_PREFIX_MISMATCH: stored='{parsed.get('config_sha256_prefix')}' != '{expected_prefix}'"
        )

    # Hard multi-layer safety validation
    for flag in (
        "live_trading_enabled",
        "broker_order_placement_enabled",
        "automatic_capital_deployment_enabled",
    ):
        if parsed.get(flag) is not False:
            raise PaperTradingSafetyError(
                f"HARD_PAPER_ONLY_GUARD: Frozen config field '{flag}' must be strictly False."
            )

    if parsed.get("selected_policy_id") != "G_SIMPLE_TREND_BASELINE":
        raise ConfigIntegrityError(
            f"Unexpected selected_policy_id: {parsed.get('selected_policy_id')} != G_SIMPLE_TREND_BASELINE"
        )

    ec_raw = parsed["execution_config"]
    if ec_raw.get("config_id") != "EXP_E_COMBINED_EXEC_INTEL":
        raise ConfigIntegrityError(
            f"Unexpected execution_config.config_id: {ec_raw.get('config_id')} != EXP_E_COMBINED_EXEC_INTEL"
        )

    active_sleeves = tuple(str(s) for s in ec_raw.get("active_sleeves", []))
    if active_sleeves != ("STRAT_E_DEFENSIVE_YIELD_LOWVOL",):
        raise PaperTradingSafetyError(
            f"HARD_STRATEGY_GUARD: Only STRAT_E_DEFENSIVE_YIELD_LOWVOL is admitted; got {active_sleeves}."
        )
    for s in active_sleeves:
        if s in SUSPENDED_STRATEGIES_HARD:
            raise PaperTradingSafetyError(f"HARD_STRATEGY_GUARD: Suspended strategy '{s}' cannot be activated.")

    exec_cfg = ExecutionIntelligenceConfig(
        config_id=str(ec_raw["config_id"]),
        config_name=str(ec_raw["config_name"]),
        core_target_pct=float(ec_raw["core_target_pct"]),
        tactical_target_pct=float(ec_raw["tactical_target_pct"]),
        liquid_target_pct=float(ec_raw["liquid_target_pct"]),
        allocation_mode=str(ec_raw["allocation_mode"]),
        active_sleeves=active_sleeves,
        sleeve_weights=tuple((str(k), float(v)) for k, v in ec_raw["sleeve_weights"]),
        hurdle_mode=str(ec_raw["hurdle_mode"]),
        replacement_policy=str(ec_raw["replacement_policy"]),
        min_replacement_edge_bps=float(ec_raw["min_replacement_edge_bps"]),
        min_hold_days_before_rotate=int(ec_raw["min_hold_days_before_rotate"]),
        protect_winner_pct=float(ec_raw["protect_winner_pct"]),
        stop_model=str(ec_raw["stop_model"]),
        execution_mode=str(ec_raw["execution_mode"]),
        sizing_mode=str(ec_raw["sizing_mode"]),
        event_driven_replenishment=bool(ec_raw["event_driven_replenishment"]),
        periodic_review_interval_days=int(ec_raw["periodic_review_interval_days"]),
        replenish_min_gap_days=int(ec_raw["replenish_min_gap_days"]),
        sweep_unused_tactical_to_core=bool(ec_raw["sweep_unused_tactical_to_core"]),
        core_rebal_band_pct=float(ec_raw["core_rebal_band_pct"]),
        liquid_sweep_min_cash_inr=float(ec_raw["liquid_sweep_min_cash_inr"]),
        min_core_trade_inr=float(ec_raw["min_core_trade_inr"]),
        max_tactical_positions=int(ec_raw["max_tactical_positions"]),
        max_single_stock_pct=float(ec_raw["max_single_stock_pct"]),
        min_tactical_lot_inr=float(ec_raw["min_tactical_lot_inr"]),
        max_sector_exposure_pct=float(ec_raw["max_sector_exposure_pct"]),
        max_sector_positions=int(ec_raw["max_sector_positions"]),
        max_industry_exposure_pct=float(ec_raw["max_industry_exposure_pct"]),
        max_unknown_sector_exposure_pct=float(ec_raw["max_unknown_sector_exposure_pct"]),
        max_pairwise_corr=float(ec_raw["max_pairwise_corr"]),
        target_portfolio_vol_pct=float(ec_raw["target_portfolio_vol_pct"]),
        max_daily_vol_participation=float(ec_raw["max_daily_vol_participation"]),
        settlement_lag_days=int(ec_raw["settlement_lag_days"]),
        slippage_override_bps=ec_raw.get("slippage_override_bps"),
        hold_max_days=int(ec_raw["hold_max_days"]),
        cooldown_days=int(ec_raw["cooldown_days"]),
        enable_dividend_accounting=bool(ec_raw["enable_dividend_accounting"]),
    )

    rp_raw = parsed["regime_policy"]
    regime_policy = AdaptiveRegimePolicy(
        policy_id=str(rp_raw["policy_id"]),
        policy_name=str(rp_raw["policy_name"]),
        regime_source=str(rp_raw["regime_source"]),
        regime_rules=tuple(
            (str(r[0]), float(r[1]), float(r[2]), float(r[3]), bool(r[4]), float(r[5]), float(r[6]))
            for r in rp_raw["regime_rules"]
        ),
        allocation_confirm_days=int(rp_raw["allocation_confirm_days"]),
    )

    ts_raw = parsed["tariff_schedule"]
    tariff_schedule = IndianTariffRegimeSchedule(
        schedule_version=str(ts_raw["schedule_version"]),
        oct_2024_effective_date=str(ts_raw["oct_2024_effective_date"]),
        delivery_brokerage_mode=str(ts_raw["delivery_brokerage_mode"]),
        delivery_stt_buy=float(ts_raw["delivery_stt_buy"]),
        delivery_stt_sell=float(ts_raw["delivery_stt_sell"]),
        delivery_stamp_buy=float(ts_raw["delivery_stamp_buy"]),
        equity_etf_stt_buy=float(ts_raw["equity_etf_stt_buy"]),
        equity_etf_stt_sell=float(ts_raw["equity_etf_stt_sell"]),
        equity_etf_stamp_buy=float(ts_raw["equity_etf_stamp_buy"]),
        equity_etf_tracking_diff_annual_bps=float(ts_raw["equity_etf_tracking_diff_annual_bps"]),
        liquid_etf_stt_buy=float(ts_raw["liquid_etf_stt_buy"]),
        liquid_etf_stt_sell=float(ts_raw["liquid_etf_stt_sell"]),
        liquid_etf_stamp_buy=float(ts_raw["liquid_etf_stamp_buy"]),
        liquid_etf_ter_annual_bps=float(ts_raw["liquid_etf_ter_annual_bps"]),
        nse_cm_exch_pre_202410=float(ts_raw["nse_cm_exch_pre_202410"]),
        nse_cm_exch_post_202410=float(ts_raw["nse_cm_exch_post_202410"]),
        sebi_fee_rate=float(ts_raw["sebi_fee_rate"]),
        gst_rate=float(ts_raw["gst_rate"]),
        dp_charge_pre_202410_inr=float(ts_raw["dp_charge_pre_202410_inr"]),
        dp_charge_post_202410_inr=float(ts_raw["dp_charge_post_202410_inr"]),
        equity_base_half_spread_bps=float(ts_raw["equity_base_half_spread_bps"]),
        equity_etf_half_spread_bps=float(ts_raw["equity_etf_half_spread_bps"]),
        liquid_etf_half_spread_bps=float(ts_raw["liquid_etf_half_spread_bps"]),
        impact_eta_equity=float(ts_raw["impact_eta_equity"]),
        impact_eta_etf=float(ts_raw["impact_eta_etf"]),
    )

    return VerifiedFrozenConfig(
        config_path=str(config_path),
        canonical_sha256=canon_sha,
        raw_file_sha256=raw_sha,
        expected_prefix=expected_prefix,
        verified_at_utc=_utc_now_iso(),
        raw_config=_freeze_mapping(parsed),
        exec_cfg=exec_cfg,
        regime_policy=regime_policy,
        tariff_schedule=tariff_schedule,
    )


class PaperOnlyExecutionGuard:
    """Multi-layer fail-closed safety guard blocking all live execution & config mutations."""

    LIVE_TRADING_ENABLED: bool = False
    BROKER_ORDER_PLACEMENT_ENABLED: bool = False
    AUTOMATIC_CAPITAL_DEPLOYMENT_ENABLED: bool = False

    def __init__(self, extra_admitted_strategy_ids: tuple[str, ...] | frozenset[str] = ()) -> None:
        # Moreshwar 3.2 admits the orchestrator-consensus sleeve; Moreshwar 3.1 admits Family E only.
        self.extra_admitted_strategy_ids = frozenset(str(x) for x in extra_admitted_strategy_ids)

    def assert_paper_only(
        self,
        *,
        live_trading_enabled: bool = False,
        broker_order_placement_enabled: bool = False,
        automatic_capital_deployment_enabled: bool = False,
        broker_adapter: Any = None,
        strategy_ids: list[str] | tuple[str, ...] | None = None,
    ) -> None:
        if live_trading_enabled:
            raise PaperTradingSafetyError("HARD_PAPER_ONLY_GUARD: live_trading_enabled cannot be set to True.")
        if broker_order_placement_enabled:
            raise PaperTradingSafetyError("HARD_PAPER_ONLY_GUARD: broker_order_placement_enabled cannot be set to True.")
        if automatic_capital_deployment_enabled:
            raise PaperTradingSafetyError(
                "HARD_PAPER_ONLY_GUARD: automatic_capital_deployment_enabled cannot be set to True."
            )
        if broker_adapter is not None:
            raise PaperTradingSafetyError(
                "HARD_PAPER_ONLY_GUARD: Live broker adapters cannot be loaded into Moreshwar 3.1 paper engine."
            )
        if strategy_ids:
            for sid in strategy_ids:
                if str(sid) in SUSPENDED_STRATEGIES_HARD:
                    raise PaperTradingSafetyError(
                        f"HARD_STRATEGY_GUARD: Strategy '{sid}' is permanently suspended in Moreshwar 3.1."
                    )
                if str(sid) != "STRAT_E_DEFENSIVE_YIELD_LOWVOL" and str(sid) not in self.extra_admitted_strategy_ids:
                    raise PaperTradingSafetyError(
                        f"HARD_STRATEGY_GUARD: Unadmitted strategy '{sid}' blocked; only Family E "
                        f"and explicitly admitted orchestrator sleeves are admitted."
                    )

    def submit_live_order(self, order: dict[str, Any]) -> None:
        raise PaperTradingSafetyError(
            "HARD_PAPER_ONLY_GUARD: Live broker order placement is permanently disabled in Moreshwar 3.1."
        )

    def deploy_live_capital(self, amount_inr: float) -> None:
        raise PaperTradingSafetyError(
            "HARD_PAPER_ONLY_GUARD: Automatic capital deployment is permanently disabled in Moreshwar 3.1."
        )

    def load_live_execution_adapter(self, adapter_name: str = "zerodha_kite") -> None:
        raise PaperTradingSafetyError(
            f"HARD_PAPER_ONLY_GUARD: Loading live execution adapter '{adapter_name}' is prohibited."
        )

    def security_posture(self) -> dict[str, Any]:
        return {
            "paper_trading_only": True,
            "live_trading_enabled": False,
            "broker_order_placement_enabled": False,
            "automatic_capital_deployment_enabled": False,
            "banner_text": "PAPER TRADING ONLY — LIVE TRADING DISABLED",
            "live_broker_adapter_loaded": False,
        }


class ProspectivePaperTradingEngine:
    """Durable, restart-safe, idempotent forward-only paper-trading engine backed by SQLite."""

    def __init__(
        self,
        *,
        db_path: Path = DEFAULT_PAPER_DB_PATH,
        config_path: Path = DEFAULT_FROZEN_CONFIG_PATH,
        initial_capital_inr: float = INITIAL_PAPER_CAPITAL_INR,
        prospective_start_date: str = HISTORICAL_DB_END_DATE,
        live_trading_enabled: bool = False,
        broker_order_placement_enabled: bool = False,
        automatic_capital_deployment_enabled: bool = False,
        broker_adapter: Any = None,
        extra_admitted_strategy_ids: tuple[str, ...] = (),
        regime_policy_override: AdaptiveRegimePolicy | None = None,
        exec_cfg_override: dict[str, Any] | None = None,
    ) -> None:
        self.safety_guard = PaperOnlyExecutionGuard(extra_admitted_strategy_ids)
        self.safety_guard.assert_paper_only(
            live_trading_enabled=live_trading_enabled,
            broker_order_placement_enabled=broker_order_placement_enabled,
            automatic_capital_deployment_enabled=automatic_capital_deployment_enabled,
            broker_adapter=broker_adapter,
        )

        if (regime_policy_override is not None or exec_cfg_override is not None) and "STRAT_ORCH_AGENT_CONSENSUS" not in extra_admitted_strategy_ids:
            raise PaperTradingSafetyError(
                "HARD_CONFIG_GUARD: Allocation/execution overrides are forbidden for frozen Moreshwar 3.1."
            )

        self.config_path = Path(config_path)
        self.verified_config = load_and_verify_frozen_config(self.config_path)
        self._regime_policy_override = regime_policy_override
        if exec_cfg_override:
            merged_ec = asdict(self.verified_config.exec_cfg)
            merged_ec.update(exec_cfg_override)
            merged_ec["active_sleeves"] = tuple(merged_ec["active_sleeves"])
            merged_ec["sleeve_weights"] = tuple((str(k), float(v)) for k, v in merged_ec["sleeve_weights"])
            self._effective_exec_cfg = ExecutionIntelligenceConfig(**merged_ec)
        else:
            self._effective_exec_cfg = self.verified_config.exec_cfg
        self._effective_regime_policy = regime_policy_override or self.verified_config.regime_policy

        self.cost_engine = TransactionCostIntelligence(self.verified_config.tariff_schedule)
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.initial_capital_inr = float(initial_capital_inr)
        self.prospective_start_date = str(prospective_start_date)

        self._init_schema()
        self._bootstrap_or_verify_metadata()

    # ------------------------------------------------------------------
    # SQLite Connection & Schema Initialization
    # ------------------------------------------------------------------
    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=30.0, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=FULL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        return conn

    def _init_schema(self) -> None:
        conn = self._connect()
        try:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS engine_metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at_utc TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS session_checkpoints (
                    trading_date TEXT PRIMARY KEY,
                    observation_mode TEXT NOT NULL,
                    phase TEXT NOT NULL,
                    idempotency_key TEXT NOT NULL,
                    data_quality_state TEXT NOT NULL,
                    reconciliation_diff_inr REAL NOT NULL DEFAULT 0.0,
                    error_message TEXT DEFAULT '',
                    started_at_utc TEXT NOT NULL,
                    completed_at_utc TEXT DEFAULT ''
                );

                CREATE TABLE IF NOT EXISTS portfolio_state (
                    singleton_id INTEGER PRIMARY KEY CHECK (singleton_id = 1),
                    as_of_date TEXT NOT NULL,
                    settled_cash_inr REAL NOT NULL,
                    unsettled_cash_inr REAL NOT NULL,
                    core_qty INTEGER NOT NULL DEFAULT 0,
                    core_cost_basis_inr REAL NOT NULL DEFAULT 0.0,
                    core_last_px REAL NOT NULL DEFAULT 0.0,
                    liquid_units REAL NOT NULL DEFAULT 0.0,
                    liquid_cost_basis_inr REAL NOT NULL DEFAULT 0.0,
                    liquid_last_px REAL NOT NULL DEFAULT 1000.0,
                    realized_core_pnl_inr REAL NOT NULL DEFAULT 0.0,
                    realized_liquid_pnl_inr REAL NOT NULL DEFAULT 0.0,
                    realized_tactical_pnl_inr REAL NOT NULL DEFAULT 0.0,
                    cum_dividends_inr REAL NOT NULL DEFAULT 0.0,
                    cum_liquid_yield_inr REAL NOT NULL DEFAULT 0.0,
                    cum_costs_inr REAL NOT NULL DEFAULT 0.0,
                    cum_turnover_inr REAL NOT NULL DEFAULT 0.0,
                    peak_nav_inr REAL NOT NULL,
                    last_nav_inr REAL NOT NULL,
                    raw_regime TEXT NOT NULL DEFAULT 'BULL_TREND',
                    confirmed_regime TEXT NOT NULL DEFAULT 'BULL_TREND',
                    regime_confirm_streak INTEGER NOT NULL DEFAULT 3,
                    target_core_pct REAL NOT NULL DEFAULT 60.0,
                    target_tactical_pct REAL NOT NULL DEFAULT 30.0,
                    target_liquid_pct REAL NOT NULL DEFAULT 10.0,
                    allow_new_entries INTEGER NOT NULL DEFAULT 1,
                    consecutive_high_cash_days INTEGER NOT NULL DEFAULT 0,
                    session_counter INTEGER NOT NULL DEFAULT 0,
                    last_replenish_session INTEGER NOT NULL DEFAULT -999,
                    bench_niftybees_base_px REAL NOT NULL DEFAULT 0.0,
                    bench_liquid_base_px REAL NOT NULL DEFAULT 0.0,
                    bench_45_55_liquid_units REAL NOT NULL DEFAULT 0.0,
                    bench_45_55_nifty_units REAL NOT NULL DEFAULT 0.0,
                    pending_orders_json TEXT NOT NULL DEFAULT '[]',
                    cooldown_json TEXT NOT NULL DEFAULT '{}',
                    dp_pairs_json TEXT NOT NULL DEFAULT '[]',
                    updated_at_utc TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS positions (
                    position_id TEXT PRIMARY KEY,
                    instrument_type TEXT NOT NULL,
                    sec_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    sector TEXT NOT NULL,
                    industry TEXT NOT NULL,
                    strategy_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    entry_date TEXT NOT NULL,
                    exit_date TEXT DEFAULT '',
                    qty REAL NOT NULL,
                    avg_entry_price REAL NOT NULL,
                    last_price REAL NOT NULL,
                    cost_basis_inr REAL NOT NULL,
                    buy_exec_notional_inr REAL NOT NULL,
                    buy_total_cost_inr REAL NOT NULL,
                    sl_price REAL NOT NULL,
                    atr_entry REAL NOT NULL,
                    max_price REAL NOT NULL,
                    bars_held INTEGER NOT NULL DEFAULT 0,
                    hold_max_days INTEGER NOT NULL DEFAULT 120,
                    be_locked INTEGER NOT NULL DEFAULT 0,
                    mae_pct REAL NOT NULL DEFAULT 0.0,
                    mfe_pct REAL NOT NULL DEFAULT 0.0,
                    dividends_received_inr REAL NOT NULL DEFAULT 0.0,
                    realized_pnl_inr REAL NOT NULL DEFAULT 0.0,
                    exit_reason TEXT DEFAULT '',
                    entry_regime TEXT NOT NULL DEFAULT 'BULL_TREND',
                    exit_regime TEXT DEFAULT ''
                );

                CREATE TABLE IF NOT EXISTS simulated_orders (
                    order_id TEXT PRIMARY KEY,
                    trading_date TEXT NOT NULL,
                    signal_timestamp TEXT NOT NULL,
                    execution_timestamp TEXT NOT NULL,
                    strategy_id TEXT NOT NULL,
                    instrument_class TEXT NOT NULL,
                    sec_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    requested_qty REAL NOT NULL,
                    filled_qty REAL NOT NULL,
                    expected_price REAL NOT NULL,
                    actual_exec_price REAL NOT NULL,
                    gross_value_inr REAL NOT NULL,
                    slippage_bps REAL NOT NULL DEFAULT 0.0,
                    slippage_inr REAL NOT NULL DEFAULT 0.0,
                    brokerage_inr REAL NOT NULL DEFAULT 0.0,
                    stt_inr REAL NOT NULL DEFAULT 0.0,
                    exchange_fee_inr REAL NOT NULL DEFAULT 0.0,
                    sebi_fee_inr REAL NOT NULL DEFAULT 0.0,
                    gst_inr REAL NOT NULL DEFAULT 0.0,
                    stamp_duty_inr REAL NOT NULL DEFAULT 0.0,
                    dp_charge_inr REAL NOT NULL DEFAULT 0.0,
                    total_cost_inr REAL NOT NULL DEFAULT 0.0,
                    net_value_inr REAL NOT NULL DEFAULT 0.0,
                    execution_status TEXT NOT NULL,
                    rejection_reason TEXT DEFAULT '',
                    ev_expected_net_bps REAL DEFAULT 0.0,
                    ev_hurdle_bps REAL DEFAULT 200.0,
                    observation_mode TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS transaction_cost_ledger (
                    cost_id TEXT PRIMARY KEY,
                    order_id TEXT UNIQUE NOT NULL,
                    trading_date TEXT NOT NULL,
                    instrument_class TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    gross_value_inr REAL NOT NULL,
                    brokerage_inr REAL NOT NULL,
                    stt_inr REAL NOT NULL,
                    exchange_fee_inr REAL NOT NULL,
                    sebi_fee_inr REAL NOT NULL,
                    gst_inr REAL NOT NULL,
                    stamp_duty_inr REAL NOT NULL,
                    dp_charge_inr REAL NOT NULL,
                    slippage_inr REAL NOT NULL,
                    regulatory_total_inr REAL NOT NULL,
                    total_cost_inr REAL NOT NULL,
                    observation_mode TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS dividend_ledger (
                    dividend_id TEXT PRIMARY KEY,
                    trading_date TEXT NOT NULL,
                    sec_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    qty_eligible REAL NOT NULL,
                    rate_per_unit_inr REAL NOT NULL,
                    credited_cash_inr REAL NOT NULL,
                    stop_adjusted_from REAL DEFAULT 0.0,
                    stop_adjusted_to REAL DEFAULT 0.0,
                    observation_mode TEXT NOT NULL,
                    UNIQUE(trading_date, sec_id, event_type)
                );

                CREATE TABLE IF NOT EXISTS settlement_ledger (
                    settlement_id TEXT PRIMARY KEY,
                    order_id TEXT UNIQUE NOT NULL,
                    trade_date TEXT NOT NULL,
                    settlement_date TEXT NOT NULL,
                    due_session_counter INTEGER NOT NULL,
                    symbol TEXT NOT NULL,
                    amount_inr REAL NOT NULL,
                    status TEXT NOT NULL,
                    settled_on_date TEXT DEFAULT '',
                    observation_mode TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS daily_nav_ledger (
                    trading_date TEXT PRIMARY KEY,
                    observation_mode TEXT NOT NULL,
                    nav_inr REAL NOT NULL,
                    settled_cash_inr REAL NOT NULL,
                    unsettled_cash_inr REAL NOT NULL,
                    core_value_inr REAL NOT NULL,
                    tactical_value_inr REAL NOT NULL,
                    liquid_value_inr REAL NOT NULL,
                    realized_pnl_inr REAL NOT NULL,
                    unrealized_pnl_inr REAL NOT NULL,
                    cum_dividends_inr REAL NOT NULL,
                    cum_liquid_yield_inr REAL NOT NULL,
                    cum_costs_inr REAL NOT NULL,
                    daily_pnl_inr REAL NOT NULL,
                    daily_return_pct REAL NOT NULL,
                    cumulative_return_pct REAL NOT NULL,
                    drawdown_pct REAL NOT NULL,
                    raw_regime TEXT NOT NULL,
                    confirmed_regime TEXT NOT NULL,
                    regime_confirm_streak INTEGER NOT NULL,
                    target_core_pct REAL NOT NULL,
                    target_tactical_pct REAL NOT NULL,
                    target_liquid_pct REAL NOT NULL,
                    allow_new_entries INTEGER NOT NULL,
                    open_tactical_positions INTEGER NOT NULL,
                    benchmark_niftybees_nav REAL NOT NULL,
                    benchmark_45_55_nav REAL NOT NULL,
                    reconciliation_diff_inr REAL NOT NULL,
                    data_quality_state TEXT NOT NULL,
                    engine_state TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS decision_audit_log (
                    log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trading_date TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    payload_sha256 TEXT NOT NULL,
                    prev_chain_sha256 TEXT NOT NULL,
                    chain_sha256 TEXT NOT NULL,
                    created_at_utc TEXT NOT NULL
                );

                CREATE TRIGGER IF NOT EXISTS prevent_audit_update
                BEFORE UPDATE ON decision_audit_log
                BEGIN
                    SELECT RAISE(ABORT, 'APPEND_ONLY_AUDIT_LOG: UPDATE prohibited');
                END;

                CREATE TRIGGER IF NOT EXISTS prevent_audit_delete
                BEFORE DELETE ON decision_audit_log
                BEGIN
                    SELECT RAISE(ABORT, 'APPEND_ONLY_AUDIT_LOG: DELETE prohibited');
                END;
                """
            )
        finally:
            conn.close()

    def _get_meta(self, conn: sqlite3.Connection, key: str, default: str = "") -> str:
        row = conn.execute("SELECT value FROM engine_metadata WHERE key = ?", (key,)).fetchone()
        return str(row["value"]) if row is not None else default

    def _set_meta(self, conn: sqlite3.Connection, key: str, value: str) -> None:
        conn.execute(
            """
            INSERT INTO engine_metadata(key, value, updated_at_utc)
            VALUES (?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at_utc = excluded.updated_at_utc
            """,
            (key, str(value), _utc_now_iso()),
        )

    def _append_audit_event(
        self,
        conn: sqlite3.Connection,
        *,
        trading_date: str,
        event_type: str,
        payload: dict[str, Any],
    ) -> str:
        last = conn.execute(
            "SELECT chain_sha256 FROM decision_audit_log ORDER BY log_id DESC LIMIT 1"
        ).fetchone()
        prev_hash = str(last["chain_sha256"]) if last is not None else "0" * 64
        payload_str = json.dumps(payload, sort_keys=True, default=str)
        payload_sha = hashlib.sha256(payload_str.encode("utf-8")).hexdigest()
        chain_sha = hashlib.sha256(f"{prev_hash}:{trading_date}:{event_type}:{payload_sha}".encode("utf-8")).hexdigest()
        conn.execute(
            """
            INSERT INTO decision_audit_log(
                trading_date, event_type, payload_json, payload_sha256,
                prev_chain_sha256, chain_sha256, created_at_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (trading_date, event_type, payload_str, payload_sha, prev_hash, chain_sha, _utc_now_iso()),
        )
        return chain_sha

    def _bootstrap_or_verify_metadata(self) -> None:
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            existing_hash = self._get_meta(conn, "config_canonical_sha256", "")
            now = _utc_now_iso()
            if not existing_hash:
                self._set_meta(conn, "release_version", RELEASE_VERSION_M3_1)
                self._set_meta(conn, "initial_capital_inr", f"{self.initial_capital_inr:.2f}")
                self._set_meta(conn, "prospective_start_date", self.prospective_start_date)
                self._set_meta(conn, "historical_db_end_date", HISTORICAL_DB_END_DATE)
                self._set_meta(conn, "config_canonical_sha256", self.verified_config.canonical_sha256)
                self._set_meta(conn, "config_raw_sha256", self.verified_config.raw_file_sha256)
                self._set_meta(conn, "config_expected_prefix", self.verified_config.expected_prefix)
                self._set_meta(conn, "engine_state", "IDLE")
                self._set_meta(conn, "safe_mode_reason", "")
                self._set_meta(conn, "live_trading_enabled", "false")
                self._set_meta(conn, "broker_order_placement_enabled", "false")
                self._set_meta(conn, "automatic_capital_deployment_enabled", "false")

                conn.execute(
                    """
                    INSERT OR IGNORE INTO portfolio_state(
                        singleton_id, as_of_date, settled_cash_inr, unsettled_cash_inr,
                        peak_nav_inr, last_nav_inr, updated_at_utc
                    ) VALUES (1, ?, ?, 0.0, ?, ?, ?)
                    """,
                    (
                        self.prospective_start_date,
                        self.initial_capital_inr,
                        self.initial_capital_inr,
                        self.initial_capital_inr,
                        now,
                    ),
                )
                self._append_audit_event(
                    conn,
                    trading_date=self.prospective_start_date,
                    event_type="ENGINE_INITIALIZED",
                    payload={
                        "initial_capital_inr": self.initial_capital_inr,
                        "prospective_start_date": self.prospective_start_date,
                        "config_canonical_sha256": self.verified_config.canonical_sha256,
                        "config_raw_sha256": self.verified_config.raw_file_sha256,
                        "live_trading_enabled": False,
                    },
                )
            else:
                if existing_hash != self.verified_config.canonical_sha256:
                    self._set_meta(conn, "engine_state", "SAFE_MODE")
                    self._set_meta(
                        conn,
                        "safe_mode_reason",
                        f"CONFIG_HASH_DRIFT: stored={existing_hash[:16]} vs loaded={self.verified_config.canonical_sha256[:16]}",
                    )
                    conn.execute("COMMIT")
                    raise ConfigIntegrityError(
                        f"Stored ledger config hash {existing_hash[:16]} does not match "
                        f"loaded config hash {self.verified_config.canonical_sha256[:16]}"
                    )
                self.prospective_start_date = self._get_meta(conn, "prospective_start_date", self.prospective_start_date)
                self.initial_capital_inr = float(self._get_meta(conn, "initial_capital_inr", str(self.initial_capital_inr)))
            conn.execute("COMMIT")
        except Exception:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            raise
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Engine Operational State & Safe-Mode Controls (Paper-Only)
    # ------------------------------------------------------------------
    def get_engine_state(self) -> dict[str, Any]:
        conn = self._connect()
        try:
            return {
                "engine_state": self._get_meta(conn, "engine_state", "IDLE"),
                "safe_mode_reason": self._get_meta(conn, "safe_mode_reason", ""),
                "release_version": self._get_meta(conn, "release_version", RELEASE_VERSION_M3_1),
                "historical_cutoff_date": self._get_meta(conn, "historical_db_end_date", HISTORICAL_DB_END_DATE),
                "prospective_start_date": self._get_meta(conn, "prospective_start_date", self.prospective_start_date),
                "first_eligible_prospective_session": self._get_meta(conn, "first_eligible_prospective_session", ""),
                "deployment_timestamp_utc": self._get_meta(conn, "deployment_timestamp_utc", ""),
                "database_identifier": self._get_meta(conn, "database_identifier", ""),
                "database_path": str(self.db_path),
                "git_commit_sha": self._get_meta(conn, "git_commit_sha", ""),
                "initial_capital_inr": float(self._get_meta(conn, "initial_capital_inr", str(self.initial_capital_inr))),
                "initial_capital_classification": self._get_meta(
                    conn, "initial_capital_classification", "SIMULATED_CASH_DEPOSIT_NOT_PROFIT"
                ),
                "config_canonical_sha256": self._get_meta(conn, "config_canonical_sha256", ""),
                "config_raw_sha256": self._get_meta(conn, "config_raw_sha256", ""),
                "live_trading_enabled": False,
                "broker_order_placement_enabled": False,
                "automatic_capital_deployment_enabled": False,
            }
        finally:
            conn.close()

    def record_official_deployment_metadata(
        self,
        *,
        first_eligible_prospective_session: str,
        git_commit_sha: str = "",
        deployment_timestamp_utc: str | None = None,
    ) -> dict[str, Any]:
        """Durably record the official prospective paper-trading deployment anchor (idempotent on first launch)."""
        if str(first_eligible_prospective_session) <= self.prospective_start_date:
            raise PaperTradingSafetyError(
                f"PROSPECTIVE_BOUNDARY_VIOLATION: first_eligible_prospective_session "
                f"'{first_eligible_prospective_session}' must be strictly after '{self.prospective_start_date}'."
            )
        ts = deployment_timestamp_utc or _utc_now_iso()
        db_id = f"M31-PAPER-DB-{hashlib.sha256(str(self.db_path.resolve()).encode('utf-8')).hexdigest()[:12].upper()}"
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            existing_ts = self._get_meta(conn, "deployment_timestamp_utc", "")
            effective_ts = existing_ts or ts
            self._set_meta(conn, "deployment_timestamp_utc", effective_ts)
            self._set_meta(conn, "last_activation_timestamp_utc", ts)
            self._set_meta(conn, "first_eligible_prospective_session", str(first_eligible_prospective_session))
            self._set_meta(conn, "database_identifier", db_id)
            self._set_meta(conn, "git_commit_sha", str(git_commit_sha))
            self._set_meta(conn, "initial_capital_classification", "SIMULATED_CASH_DEPOSIT_NOT_PROFIT")
            self._append_audit_event(
                conn,
                trading_date=self.prospective_start_date,
                event_type="PROSPECTIVE_DEPLOYMENT_ACTIVATED",
                payload={
                    "deployment_timestamp_utc": effective_ts,
                    "activation_timestamp_utc": ts,
                    "historical_cutoff_date": HISTORICAL_DB_END_DATE,
                    "prospective_start_date": self.prospective_start_date,
                    "first_eligible_prospective_session": str(first_eligible_prospective_session),
                    "database_identifier": db_id,
                    "software_version": RELEASE_VERSION_M3_1,
                    "git_commit_sha": str(git_commit_sha),
                    "initial_capital_inr": self.initial_capital_inr,
                    "initial_capital_classification": "SIMULATED_CASH_DEPOSIT_NOT_PROFIT",
                    "config_canonical_sha256": self.verified_config.canonical_sha256,
                    "config_raw_sha256": self.verified_config.raw_file_sha256,
                    "live_trading_enabled": False,
                    "broker_order_placement_enabled": False,
                    "automatic_capital_deployment_enabled": False,
                },
            )
            conn.execute("COMMIT")
            return self.get_engine_state()
        except Exception:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            raise
        finally:
            conn.close()

    def enter_safe_mode(self, reason: str, *, trading_date: str = "") -> dict[str, Any]:
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            self._set_meta(conn, "engine_state", "SAFE_MODE")
            self._set_meta(conn, "safe_mode_reason", str(reason))
            self._append_audit_event(
                conn,
                trading_date=trading_date or self.prospective_start_date,
                event_type="SAFE_MODE_ENTERED",
                payload={"reason": str(reason)},
            )
            conn.execute("COMMIT")
            return self.get_engine_state()
        except Exception:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            raise
        finally:
            conn.close()

    def set_operational_state(self, new_state: str, *, reason: str = "") -> dict[str, Any]:
        """Paper-only operational transitions. Refuses to exit SAFE_MODE without passing recovery validation."""
        st = str(new_state).upper().strip()
        if st not in ALLOWED_ENGINE_STATES:
            raise ValueError(f"Unsupported engine state: {new_state}")
        cur = self.get_engine_state()
        if cur["engine_state"] == "SAFE_MODE" and st in ("RUNNING", "IDLE"):
            raise PaperTradingSafetyError(
                "HARD_SAFE_MODE_GUARD: Cannot bypass SAFE_MODE directly; call recover_from_safe_mode() after validation."
            )
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            self._set_meta(conn, "engine_state", st)
            if st == "SAFE_MODE":
                self._set_meta(conn, "safe_mode_reason", reason or "MANUAL_SAFE_MODE")
            self._append_audit_event(
                conn,
                trading_date=self._current_as_of_date(conn),
                event_type=f"STATE_TRANSITION_{st}",
                payload={"previous": cur["engine_state"], "new": st, "reason": reason},
            )
            conn.execute("COMMIT")
            return self.get_engine_state()
        except Exception:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            raise
        finally:
            conn.close()

    def recover_from_safe_mode(self, *, operator_note: str = "verified_recovery") -> dict[str, Any]:
        """Validate config integrity, audit chain, checkpoints, and four-ledger reconciliation before leaving SAFE_MODE."""
        # 1. Re-verify frozen configuration hash from disk
        fresh_cfg = load_and_verify_frozen_config(self.config_path)
        if fresh_cfg.canonical_sha256 != self.verified_config.canonical_sha256:
            raise ConfigIntegrityError("Config hash changed on disk; cannot leave SAFE_MODE.")

        # 2. Verify ledger reconciliation and check incomplete checkpoints
        recon = self.reconcile_portfolio()
        if not recon["passed"]:
            raise ReconciliationFailureError(
                f"Cannot exit SAFE_MODE while reconciliation fails: diff={recon['reconciliation_diff_inr']:.4f} INR"
            )

        chain = self.verify_audit_chain()
        if not chain["valid"]:
            raise RuntimeError(f"Audit chain corrupted at log_id={chain.get('failed_log_id')}; cannot exit SAFE_MODE.")

        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            # Mark any stale STARTED/PROCESSING checkpoints as FAILED_RECOVERED so they can be cleanly re-run
            conn.execute(
                """
                UPDATE session_checkpoints
                SET phase = 'FAILED', error_message = COALESCE(error_message, '') || ' [Recovered from crash]'
                WHERE phase IN ('STARTED', 'DATA_VALIDATED', 'EXECUTED')
                """
            )
            self._set_meta(conn, "engine_state", "RUNNING")
            self._set_meta(conn, "safe_mode_reason", "")
            self._append_audit_event(
                conn,
                trading_date=self._current_as_of_date(conn),
                event_type="SAFE_MODE_RECOVERED",
                payload={"operator_note": operator_note, "reconciliation_diff_inr": recon["reconciliation_diff_inr"]},
            )
            conn.execute("COMMIT")
            return {"recovered": True, "reconciliation": recon, "state": self.get_engine_state()}
        except Exception:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            raise
        finally:
            conn.close()

    def _current_as_of_date(self, conn: sqlite3.Connection) -> str:
        row = conn.execute("SELECT as_of_date FROM portfolio_state WHERE singleton_id = 1").fetchone()
        return str(row["as_of_date"]) if row is not None else self.prospective_start_date

    # ------------------------------------------------------------------
    # Restart Reconstruction & Four-Ledger Reconciliation
    # ------------------------------------------------------------------
    def verify_audit_chain(self) -> dict[str, Any]:
        """Verify cryptographic hash chain integrity of the append-only decision_audit_log."""
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT log_id, trading_date, event_type, payload_json, payload_sha256, prev_chain_sha256, chain_sha256 "
                "FROM decision_audit_log ORDER BY log_id ASC"
            ).fetchall()
            prev = "0" * 64
            for r in rows:
                calc_payload_sha = hashlib.sha256(str(r["payload_json"]).encode("utf-8")).hexdigest()
                calc_chain_sha = hashlib.sha256(
                    f"{prev}:{r['trading_date']}:{r['event_type']}:{calc_payload_sha}".encode("utf-8")
                ).hexdigest()
                if (
                    str(r["prev_chain_sha256"]) != prev
                    or str(r["payload_sha256"]) != calc_payload_sha
                    or str(r["chain_sha256"]) != calc_chain_sha
                ):
                    return {"valid": False, "entries": len(rows), "failed_log_id": int(r["log_id"])}
                prev = calc_chain_sha
            return {"valid": True, "entries": len(rows), "head_chain_sha256": prev}
        finally:
            conn.close()

    def reconcile_portfolio(self) -> dict[str, Any]:
        """Reconstruct portfolio from durable SQLite ledgers and verify <= INR 0.01 discrepancy."""
        conn = self._connect()
        try:
            ps = conn.execute("SELECT * FROM portfolio_state WHERE singleton_id = 1").fetchone()
            if ps is None:
                return {"passed": False, "reconciliation_diff_inr": 999999.0, "reason": "MISSING_PORTFOLIO_STATE"}

            open_tact = conn.execute(
                "SELECT * FROM positions WHERE instrument_type = 'TACTICAL_EQUITY' AND status = 'OPEN'"
            ).fetchall()
            closed_tact = conn.execute(
                "SELECT * FROM positions WHERE instrument_type = 'TACTICAL_EQUITY' AND status = 'CLOSED'"
            ).fetchall()
            pending_settle = conn.execute(
                "SELECT COALESCE(SUM(amount_inr), 0.0) AS s FROM settlement_ledger WHERE status = 'PENDING'"
            ).fetchone()
            cost_sum = conn.execute(
                "SELECT COALESCE(SUM(total_cost_inr), 0.0) AS s FROM transaction_cost_ledger"
            ).fetchone()
            div_sum = conn.execute(
                "SELECT COALESCE(SUM(credited_cash_inr), 0.0) AS s FROM dividend_ledger WHERE event_type = 'EQUITY_DIVIDEND'"
            ).fetchone()
            liq_yield_sum = conn.execute(
                "SELECT COALESCE(SUM(credited_cash_inr), 0.0) AS s FROM dividend_ledger WHERE event_type = 'LIQUID_BEES_YIELD'"
            ).fetchone()
            incomplete_cp = conn.execute(
                "SELECT trading_date, phase FROM session_checkpoints WHERE phase IN ('STARTED', 'DATA_VALIDATED', 'EXECUTED')"
            ).fetchall()

            settled_cash = float(ps["settled_cash_inr"])
            unsettled_cash_state = float(ps["unsettled_cash_inr"])
            unsettled_cash_ledger = float(pending_settle["s"])

            core_val = float(ps["core_qty"]) * float(ps["core_last_px"])
            liquid_val = float(ps["liquid_units"]) * float(ps["liquid_last_px"])
            tact_val = sum(float(p["qty"]) * float(p["last_price"]) for p in open_tact)
            tact_cost_basis = sum(float(p["cost_basis_inr"]) for p in open_tact)

            nav_from_assets = settled_cash + unsettled_cash_ledger + core_val + liquid_val + tact_val

            # Unrealized P&L across all open holdings
            unrealized_core = core_val - float(ps["core_cost_basis_inr"])
            # Liquid BeES yield accrues as additional units -> subtract accrued yield from unrealized price delta
            unrealized_liquid = liquid_val - float(ps["liquid_cost_basis_inr"]) - float(ps["cum_liquid_yield_inr"])
            unrealized_tactical = tact_val - tact_cost_basis
            total_unrealized = unrealized_core + unrealized_liquid + unrealized_tactical

            realized_tact_ledger = sum(float(p["realized_pnl_inr"]) for p in closed_tact)
            total_realized = (
                float(ps["realized_core_pnl_inr"])
                + float(ps["realized_liquid_pnl_inr"])
                + realized_tact_ledger
            )
            cum_div_ledger = float(div_sum["s"])
            cum_liq_yield_ledger = float(liq_yield_sum["s"])
            cum_cost_ledger = float(cost_sum["s"])

            # Accounting identity:
            # NAV == Initial Capital + Realized Price PnL + Unrealized Price PnL + Equity Dividends + Liquid Yield
            # (Note: all transaction costs are embedded inside cost_basis_inr & realized proceeds!)
            nav_from_equity = (
                self.initial_capital_inr
                + total_realized
                + total_unrealized
                + cum_div_ledger
                + cum_liq_yield_ledger
            )

            nav_diff = abs(nav_from_assets - nav_from_equity)
            settle_diff = abs(unsettled_cash_state - unsettled_cash_ledger)
            cost_diff = abs(float(ps["cum_costs_inr"]) - cum_cost_ledger)
            div_diff = abs(float(ps["cum_dividends_inr"]) - cum_div_ledger)
            cash_negative = settled_cash < -0.01

            max_diff = max(nav_diff, settle_diff, cost_diff, div_diff)
            passed = (max_diff <= RECONCILIATION_TOLERANCE_INR) and (not cash_negative) and (len(incomplete_cp) == 0)

            reasons: list[str] = []
            if nav_diff > RECONCILIATION_TOLERANCE_INR:
                reasons.append(f"NAV_MISMATCH({nav_diff:.4f})")
            if settle_diff > RECONCILIATION_TOLERANCE_INR:
                reasons.append(f"SETTLEMENT_MISMATCH({settle_diff:.4f})")
            if cost_diff > RECONCILIATION_TOLERANCE_INR:
                reasons.append(f"COST_LEDGER_MISMATCH({cost_diff:.4f})")
            if div_diff > RECONCILIATION_TOLERANCE_INR:
                reasons.append(f"DIVIDEND_LEDGER_MISMATCH({div_diff:.4f})")
            if cash_negative:
                reasons.append(f"NEGATIVE_CASH({settled_cash:.2f})")
            if incomplete_cp:
                reasons.append(f"INCOMPLETE_CHECKPOINTS({[r['trading_date'] for r in incomplete_cp]})")

            return {
                "passed": passed,
                "as_of_date": str(ps["as_of_date"]),
                "nav_from_assets_inr": round(nav_from_assets, 2),
                "nav_from_equity_inr": round(nav_from_equity, 2),
                "reconciliation_diff_inr": round(max_diff, 4),
                "nav_diff_inr": round(nav_diff, 4),
                "settlement_diff_inr": round(settle_diff, 4),
                "cost_ledger_diff_inr": round(cost_diff, 4),
                "dividend_ledger_diff_inr": round(div_diff, 4),
                "settled_cash_inr": round(settled_cash, 2),
                "unsettled_cash_inr": round(unsettled_cash_ledger, 2),
                "core_value_inr": round(core_val, 2),
                "tactical_value_inr": round(tact_val, 2),
                "liquid_value_inr": round(liquid_val, 2),
                "realized_pnl_inr": round(total_realized, 2),
                "unrealized_pnl_inr": round(total_unrealized, 2),
                "cum_dividends_inr": round(cum_div_ledger, 2),
                "cum_liquid_yield_inr": round(cum_liq_yield_ledger, 2),
                "cum_costs_inr": round(cum_cost_ledger, 2),
                "incomplete_checkpoints": [dict(r) for r in incomplete_cp],
                "failure_reasons": reasons,
            }
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Single-Session Point-in-Time Paper Execution Cycle
    # ------------------------------------------------------------------
    def _classify_observation_mode(self, trading_date: str, requested_mode: str | None) -> str:
        """Enforce strict separation between historical research/validation and genuine prospective sessions."""
        if requested_mode is not None:
            mode = str(requested_mode).upper().strip()
            if mode not in ALLOWED_OBSERVATION_MODES:
                raise ValueError(f"Invalid observation_mode: {requested_mode}")
            if mode == "PROSPECTIVE_PAPER" and str(trading_date) <= self.prospective_start_date:
                raise PaperTradingSafetyError(
                    f"PROSPECTIVE_BOUNDARY_VIOLATION: trading_date '{trading_date}' <= prospective_start_date "
                    f"'{self.prospective_start_date}'. Historical dates CANNOT be labelled PROSPECTIVE_PAPER."
                )
            return mode
        return "PROSPECTIVE_PAPER" if str(trading_date) > self.prospective_start_date else "HISTORICAL_VALIDATION"

    def record_missed_session(self, trading_date: str, *, reason: str, observation_mode: str | None = None) -> dict[str, Any]:
        """Explicitly record a missed NSE trading session instead of fabricating fills."""
        mode = self._classify_observation_mode(trading_date, observation_mode)
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            existing = conn.execute(
                "SELECT phase FROM session_checkpoints WHERE trading_date = ?", (trading_date,)
            ).fetchone()
            if existing is not None and str(existing["phase"]) == "COMPLETED":
                conn.execute("COMMIT")
                return {"trading_date": trading_date, "status": "ALREADY_COMPLETED"}
            now = _utc_now_iso()
            conn.execute(
                """
                INSERT INTO session_checkpoints(
                    trading_date, observation_mode, phase, idempotency_key,
                    data_quality_state, reconciliation_diff_inr, error_message, started_at_utc, completed_at_utc
                ) VALUES (?, ?, 'MISSED', ?, 'MISSING_DATA', 0.0, ?, ?, ?)
                ON CONFLICT(trading_date) DO UPDATE SET
                    phase = 'MISSED',
                    data_quality_state = 'MISSING_DATA',
                    error_message = excluded.error_message,
                    completed_at_utc = excluded.completed_at_utc
                """,
                (trading_date, mode, f"missed:{trading_date}", str(reason), now, now),
            )
            self._append_audit_event(
                conn,
                trading_date=trading_date,
                event_type="SESSION_MISSED",
                payload={"reason": str(reason), "observation_mode": mode},
            )
            conn.execute("COMMIT")
            return {"trading_date": trading_date, "status": "MISSED", "reason": str(reason)}
        except Exception:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            raise
        finally:
            conn.close()

    def process_trading_session(
        self,
        *,
        trading_date: str,
        market_snapshot: dict[str, Any],
        observation_mode: str | None = None,
        allow_entries_override: bool | None = None,
        simulate_db_failure_after_checkpoint: bool = False,
    ) -> dict[str, Any]:
        """Execute ONE complete, atomic, idempotent point-in-time paper-trading session.

        `market_snapshot` must contain ONLY information available for `trading_date`:
        - `niftybees_bar`: {open, high, low, close, volume, valid}
        - `liquidbees_bar`: {open, high, low, close, volume, valid}
        - `niftybees_close_history`: causal list/array of NIFTYBEES closes up to & including `trading_date`
        - `equity_bars`: dict[sec_id -> {symbol, sector, industry, is_unknown_sector, open, high, low, close, prev_close, volume, adv_20d_inr, vol_20d, hl_range_pct, atr_14, sma_50, low_20_prev, div_per_adj_share, div_yield_ttm, valid}]
        - `family_e_candidates`: list[dict] ranked strictly from session `trading_date` close (NO T+1 open!)
        - `data_quality_state`: str (`OK` | `PARTIAL_DIAGNOSTIC_ONLY` | `DEGRADED` | `INVALID`)
        - `data_timestamp_utc`: str
        - `pairwise_corr_map`: optional dict[(sec_id_a, sec_id_b) -> float]
        """
        dt = str(trading_date)
        obs_mode = self._classify_observation_mode(dt, observation_mode)

        # Re-verify frozen configuration hash from disk before every cycle!
        try:
            fresh_cfg = load_and_verify_frozen_config(self.config_path)
            if fresh_cfg.canonical_sha256 != self.verified_config.canonical_sha256:
                raise ConfigIntegrityError("Frozen config hash drifted on disk.")
        except Exception as exc:
            self.enter_safe_mode(f"CONFIG_INTEGRITY_FAILURE: {exc}", trading_date=dt)
            raise

        # Check operational state
        eng_state_info = self.get_engine_state()
        eng_state = eng_state_info["engine_state"]
        if eng_state == "STOPPED":
            return {
                "trading_date": dt,
                "status": "SKIPPED_STOPPED",
                "engine_state": eng_state,
                "idempotent_skip": False,
            }

        # Idempotency check: if date already COMPLETED, never duplicate orders/costs/dividends!
        conn = self._connect()
        try:
            cp = conn.execute(
                "SELECT * FROM session_checkpoints WHERE trading_date = ?", (dt,)
            ).fetchone()
            if cp is not None and str(cp["phase"]) == "COMPLETED":
                nav_row = conn.execute(
                    "SELECT * FROM daily_nav_ledger WHERE trading_date = ?", (dt,)
                ).fetchone()
                return {
                    "trading_date": dt,
                    "status": "IDEMPOTENT_ALREADY_COMPLETED",
                    "observation_mode": str(cp["observation_mode"]),
                    "idempotent_skip": True,
                    "reconciliation_diff_inr": float(cp["reconciliation_diff_inr"]),
                    "nav_snapshot": dict(nav_row) if nav_row is not None else {},
                }
            if cp is not None and str(cp["phase"]) in ("STARTED", "DATA_VALIDATED", "EXECUTED"):
                conn.close()
                self.enter_safe_mode(
                    f"INCOMPLETE_PRIOR_CHECKPOINT on {dt} (phase={cp['phase']}); recovery required.",
                    trading_date=dt,
                )
                return {
                    "trading_date": dt,
                    "status": "SAFE_MODE_INCOMPLETE_CHECKPOINT",
                    "engine_state": "SAFE_MODE",
                    "idempotent_skip": False,
                }
        finally:
            try:
                conn.close()
            except Exception:
                pass

        # Validate essential market data before modifying state
        dq_state = str(market_snapshot.get("data_quality_state", "OK")).upper()
        nb_bar = market_snapshot.get("niftybees_bar") or {}
        lb_bar = market_snapshot.get("liquidbees_bar") or {}
        nb_valid = bool(nb_bar.get("valid", True)) and float(nb_bar.get("open", 0.0) or 0.0) > 0 and float(nb_bar.get("close", 0.0) or 0.0) > 0
        lb_valid = bool(lb_bar.get("valid", True)) and float(lb_bar.get("open", 0.0) or 0.0) > 0 and float(lb_bar.get("close", 0.0) or 0.0) > 0
        if dq_state == "INVALID" or (not nb_valid) or (not lb_valid):
            self.enter_safe_mode(
                f"INVALID_MARKET_DATA on {dt}: dq={dq_state}, nb_valid={nb_valid}, lb_valid={lb_valid}",
                trading_date=dt,
            )
            return {
                "trading_date": dt,
                "status": "SAFE_MODE_INVALID_DATA",
                "engine_state": "SAFE_MODE",
                "data_quality_state": "INVALID",
                "idempotent_skip": False,
            }

        # Validate strategy IDs in candidates (fail closed on Family A or any unadmitted sleeve!)
        candidates = list(market_snapshot.get("family_e_candidates") or [])
        for cand in candidates:
            sid = str(cand.get("strategy_id", "STRAT_E_DEFENSIVE_YIELD_LOWVOL"))
            self.safety_guard.assert_paper_only(strategy_ids=[sid])

        # Enforce SAFE_MODE or PAUSED entry halt
        entries_permitted_by_state = (eng_state not in ("SAFE_MODE", "PAUSED"))
        if allow_entries_override is False:
            entries_permitted_by_state = False

        ecfg = self._effective_exec_cfg
        policy = self._effective_regime_policy
        policy_map = {
            r[0]: (float(r[1]), float(r[2]), float(r[3]), bool(r[4]), float(r[5]), float(r[6]))
            for r in policy.regime_rules
        }

        eq_bars: dict[str, dict[str, Any]] = market_snapshot.get("equity_bars") or {}
        corr_map: dict[tuple[str, str], float] = market_snapshot.get("pairwise_corr_map") or {}
        nb_px_o = float(nb_bar["open"])
        nb_px_c = float(nb_bar["close"])
        lb_px_o = float(lb_bar["open"])
        lb_px_c = float(lb_bar["close"])
        data_ts = str(market_snapshot.get("data_timestamp_utc") or _utc_now_iso())

        # Begin atomic SQLite transaction for this session
        conn = self._connect()
        orders_today: list[dict[str, Any]] = []
        signals_today: list[dict[str, Any]] = []
        dividends_today: list[dict[str, Any]] = []
        try:
            conn.execute("BEGIN IMMEDIATE")
            now = _utc_now_iso()
            idemp_key = hashlib.sha256(f"{dt}:{obs_mode}:{self.verified_config.canonical_sha256}".encode()).hexdigest()[:24]
            conn.execute(
                """
                INSERT INTO session_checkpoints(
                    trading_date, observation_mode, phase, idempotency_key,
                    data_quality_state, started_at_utc
                ) VALUES (?, ?, 'STARTED', ?, ?, ?)
                ON CONFLICT(trading_date) DO UPDATE SET
                    phase = 'STARTED',
                    observation_mode = excluded.observation_mode,
                    idempotency_key = excluded.idempotency_key,
                    data_quality_state = excluded.data_quality_state,
                    error_message = '',
                    started_at_utc = excluded.started_at_utc
                """,
                (dt, obs_mode, idemp_key, dq_state, now),
            )

            if simulate_db_failure_after_checkpoint:
                # Deliberately commit the STARTED checkpoint and fail — tests crash recovery!
                conn.execute("COMMIT")
                raise RuntimeError("SIMULATED_DB_WRITE_INTERRUPTION_AFTER_CHECKPOINT")

            ps = dict(conn.execute("SELECT * FROM portfolio_state WHERE singleton_id = 1").fetchone())
            session_k = int(ps["session_counter"]) + 1
            settled_cash = float(ps["settled_cash_inr"])
            core_qty = int(ps["core_qty"])
            core_cost_basis = float(ps["core_cost_basis_inr"])
            liquid_units = float(ps["liquid_units"])
            liquid_cost_basis = float(ps["liquid_cost_basis_inr"])
            realized_core_pnl = float(ps["realized_core_pnl_inr"])
            realized_liquid_pnl = float(ps["realized_liquid_pnl_inr"])
            realized_tact_pnl = float(ps["realized_tactical_pnl_inr"])
            cum_dividends = float(ps["cum_dividends_inr"])
            cum_liquid_yield = float(ps["cum_liquid_yield_inr"])
            cum_costs = float(ps["cum_costs_inr"])
            cum_turnover = float(ps["cum_turnover_inr"])
            peak_nav = float(ps["peak_nav_inr"])
            prev_nav = float(ps["last_nav_inr"])
            prev_raw_regime = str(ps["raw_regime"])
            confirmed_regime = str(ps["confirmed_regime"])
            regime_streak = int(ps["regime_confirm_streak"])
            cur_tgt_c = float(ps["target_core_pct"])
            cur_tgt_t = float(ps["target_tactical_pct"])
            cur_tgt_l = float(ps["target_liquid_pct"])
            cur_allow_entries = bool(ps["allow_new_entries"])
            consec_high_cash = int(ps["consecutive_high_cash_days"])
            last_replenish_session = int(ps["last_replenish_session"])

            pending_orders: list[dict[str, Any]] = json.loads(ps["pending_orders_json"] or "[]")
            cooldown_map: dict[str, int] = {str(k): int(v) for k, v in json.loads(ps["cooldown_json"] or "{}").items()}
            dp_pairs: set[tuple[str, str]] = {tuple(x) for x in json.loads(ps["dp_pairs_json"] or "[]")}

            # Load open tactical positions from DB
            open_rows = conn.execute(
                "SELECT * FROM positions WHERE instrument_type = 'TACTICAL_EQUITY' AND status = 'OPEN'"
            ).fetchall()
            open_pos: dict[str, dict[str, Any]] = {str(r["sec_id"]): dict(r) for r in open_rows}

            order_seq = 0

            def _next_order_id(prefix: str) -> str:
                nonlocal order_seq
                order_seq += 1
                return f"M31P-{dt.replace('-', '')}-{prefix}-{order_seq:03d}"

            def _record_order_and_cost(
                *,
                order_id: str,
                signal_ts: str,
                strategy_id: str,
                instrument_class: str,
                sec_id: str,
                symbol: str,
                side: str,
                req_qty: float,
                fill_qty: float,
                expected_px: float,
                exec_px: float,
                cost_dict: dict[str, Any] | None,
                status: str,
                rejection_reason: str = "",
                ev_net_bps: float = 0.0,
                ev_hurdle_bps: float = 200.0,
            ) -> None:
                nonlocal cum_costs, cum_turnover
                gross_val = float(cost_dict["raw_notional"]) if cost_dict else (fill_qty * exec_px)
                exec_notional = float(cost_dict["exec_notional"]) if cost_dict else gross_val
                reg_total = float(cost_dict["regulatory_total"]) if cost_dict else 0.0
                total_cost = float(cost_dict["total_cost"]) if cost_dict else 0.0
                net_val = (
                    (exec_notional + reg_total) if side == "BUY" else max(0.0, exec_notional - reg_total)
                ) if status in ("FILLED", "PARTIAL") else 0.0

                row_dict = {
                    "order_id": order_id,
                    "trading_date": dt,
                    "signal_timestamp": signal_ts,
                    "execution_timestamp": f"{dt}T09:15:00+05:30" if status in ("FILLED", "PARTIAL") else "",
                    "strategy_id": strategy_id,
                    "instrument_class": instrument_class,
                    "sec_id": sec_id,
                    "symbol": symbol,
                    "side": side,
                    "requested_qty": round(float(req_qty), 6),
                    "filled_qty": round(float(fill_qty), 6),
                    "expected_price": round(float(expected_px), 4),
                    "actual_exec_price": round(float(exec_px), 4),
                    "gross_value_inr": round(gross_val, 2),
                    "slippage_bps": round(float(cost_dict["slippage_bps"]), 3) if cost_dict else 0.0,
                    "slippage_inr": round(float(cost_dict["slippage_cost"]), 2) if cost_dict else 0.0,
                    "brokerage_inr": round(float(cost_dict["brokerage"]), 2) if cost_dict else 0.0,
                    "stt_inr": round(float(cost_dict["stt"]), 2) if cost_dict else 0.0,
                    "exchange_fee_inr": round(float(cost_dict["exchange_txn"]), 2) if cost_dict else 0.0,
                    "sebi_fee_inr": round(float(cost_dict["sebi_fee"]), 2) if cost_dict else 0.0,
                    "gst_inr": round(float(cost_dict["gst"]), 2) if cost_dict else 0.0,
                    "stamp_duty_inr": round(float(cost_dict["stamp_duty"]), 2) if cost_dict else 0.0,
                    "dp_charge_inr": round(float(cost_dict["dp_charge"]), 2) if cost_dict else 0.0,
                    "total_cost_inr": round(total_cost, 2),
                    "net_value_inr": round(net_val, 2),
                    "execution_status": status,
                    "rejection_reason": rejection_reason,
                    "ev_expected_net_bps": round(float(ev_net_bps), 2),
                    "ev_hurdle_bps": round(float(ev_hurdle_bps), 2),
                    "observation_mode": obs_mode,
                }
                orders_today.append(row_dict)
                conn.execute(
                    """
                    INSERT INTO simulated_orders(
                        order_id, trading_date, signal_timestamp, execution_timestamp, strategy_id,
                        instrument_class, sec_id, symbol, side, requested_qty, filled_qty,
                        expected_price, actual_exec_price, gross_value_inr, slippage_bps, slippage_inr,
                        brokerage_inr, stt_inr, exchange_fee_inr, sebi_fee_inr, gst_inr,
                        stamp_duty_inr, dp_charge_inr, total_cost_inr, net_value_inr,
                        execution_status, rejection_reason, ev_expected_net_bps, ev_hurdle_bps, observation_mode
                    ) VALUES (
                        :order_id, :trading_date, :signal_timestamp, :execution_timestamp, :strategy_id,
                        :instrument_class, :sec_id, :symbol, :side, :requested_qty, :filled_qty,
                        :expected_price, :actual_exec_price, :gross_value_inr, :slippage_bps, :slippage_inr,
                        :brokerage_inr, :stt_inr, :exchange_fee_inr, :sebi_fee_inr, :gst_inr,
                        :stamp_duty_inr, :dp_charge_inr, :total_cost_inr, :net_value_inr,
                        :execution_status, :rejection_reason, :ev_expected_net_bps, :ev_hurdle_bps, :observation_mode
                    )
                    """,
                    row_dict,
                )

                if status in ("FILLED", "PARTIAL") and cost_dict is not None:
                    cum_costs += total_cost
                    cum_turnover += exec_notional
                    conn.execute(
                        """
                        INSERT INTO transaction_cost_ledger(
                            cost_id, order_id, trading_date, instrument_class, symbol, side,
                            gross_value_inr, brokerage_inr, stt_inr, exchange_fee_inr, sebi_fee_inr,
                            gst_inr, stamp_duty_inr, dp_charge_inr, slippage_inr,
                            regulatory_total_inr, total_cost_inr, observation_mode
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            f"COST-{order_id}",
                            order_id,
                            dt,
                            instrument_class,
                            symbol,
                            side,
                            round(gross_val, 2),
                            round(float(cost_dict["brokerage"]), 4),
                            round(float(cost_dict["stt"]), 4),
                            round(float(cost_dict["exchange_txn"]), 4),
                            round(float(cost_dict["sebi_fee"]), 4),
                            round(float(cost_dict["gst"]), 4),
                            round(float(cost_dict["stamp_duty"]), 4),
                            round(float(cost_dict["dp_charge"]), 4),
                            round(float(cost_dict["slippage_cost"]), 4),
                            round(reg_total, 4),
                            round(total_cost, 4),
                            obs_mode,
                        ),
                    )

            # --------------------------------------------------------------
            # STEP 0A: Release T+1 Settled Cash due on session `session_k`
            # --------------------------------------------------------------
            due_settlements = conn.execute(
                "SELECT settlement_id, amount_inr FROM settlement_ledger WHERE status = 'PENDING' AND due_session_counter <= ?",
                (session_k,),
            ).fetchall()
            for s_row in due_settlements:
                settled_cash += float(s_row["amount_inr"])
                conn.execute(
                    "UPDATE settlement_ledger SET status = 'SETTLED', settled_on_date = ? WHERE settlement_id = ?",
                    (dt, str(s_row["settlement_id"])),
                )

            # --------------------------------------------------------------
            # STEP 0B: Overnight Liquid BeES Yield Accrual
            # --------------------------------------------------------------
            if session_k > 1 and liquid_units > 1e-6:
                liq_yield_inr = liquid_units * lb_px_c * LIQUID_BEES_DAILY_YIELD
                if liq_yield_inr > 0.0:
                    liquid_units += liq_yield_inr / max(1.0, lb_px_c)
                    cum_liquid_yield += liq_yield_inr
                    div_id = f"LIQY-{dt}-LIQUIDBEES"
                    conn.execute(
                        """
                        INSERT INTO dividend_ledger(
                            dividend_id, trading_date, sec_id, symbol, event_type,
                            qty_eligible, rate_per_unit_inr, credited_cash_inr, observation_mode
                        ) VALUES (?, ?, 'LIQUIDBEES', 'LIQUIDBEES', 'LIQUID_BEES_YIELD', ?, ?, ?, ?)
                        """,
                        (div_id, dt, round(liquid_units, 6), round(lb_px_c * LIQUID_BEES_DAILY_YIELD, 6), round(liq_yield_inr, 6), obs_mode),
                    )
                    dividends_today.append({"dividend_id": div_id, "symbol": "LIQUIDBEES", "event_type": "LIQUID_BEES_YIELD", "credited_cash_inr": round(liq_yield_inr, 4)})

            # --------------------------------------------------------------
            # STEP 0C: Split-Adjusted Ex-Dividend Cash Crediting & Stop Protection
            # --------------------------------------------------------------
            if ecfg.enable_dividend_accounting and open_pos:
                for sec_id, pos in open_pos.items():
                    bar = eq_bars.get(sec_id)
                    if not bar:
                        continue
                    d_adj = float(bar.get("div_per_adj_share", 0.0) or 0.0)
                    if d_adj > 1e-8 and str(pos["entry_date"]) < dt:
                        pos_adapter = {
                            "symbol": str(pos["symbol"]),
                            "entry_ti": session_k - 1,
                            "qty": int(pos["qty"]),
                            "sl_px": float(pos["sl_price"]),
                            "max_px": float(pos["max_price"]),
                            "dividends_received_inr": float(pos["dividends_received_inr"]),
                            "dividend_events": [],
                        }
                        ev = apply_ex_dividend_accounting_and_stop_adjustment(
                            pos=pos_adapter, ti=session_k, dt=dt, div_per_adj_share=d_adj
                        )
                        if ev["credited_cash_inr"] > 0.0:
                            settled_cash += float(ev["credited_cash_inr"])
                            cum_dividends += float(ev["credited_cash_inr"])
                            pos["sl_price"] = float(pos_adapter["sl_px"])
                            pos["max_price"] = float(pos_adapter["max_px"])
                            pos["dividends_received_inr"] = float(pos_adapter["dividends_received_inr"])
                            div_id = f"DIV-{dt}-{sec_id}"
                            conn.execute(
                                """
                                INSERT INTO dividend_ledger(
                                    dividend_id, trading_date, sec_id, symbol, event_type,
                                    qty_eligible, rate_per_unit_inr, credited_cash_inr,
                                    stop_adjusted_from, stop_adjusted_to, observation_mode
                                ) VALUES (?, ?, ?, ?, 'EQUITY_DIVIDEND', ?, ?, ?, ?, ?, ?)
                                """,
                                (
                                    div_id,
                                    dt,
                                    sec_id,
                                    str(pos["symbol"]),
                                    int(pos["qty"]),
                                    round(d_adj, 6),
                                    round(float(ev["credited_cash_inr"]), 6),
                                    round(float(ev["old_sl_px"]), 4),
                                    round(float(ev["new_sl_px"]), 4),
                                    obs_mode,
                                ),
                            )
                            dividends_today.append({"dividend_id": div_id, "symbol": str(pos["symbol"]), "event_type": "EQUITY_DIVIDEND", "credited_cash_inr": round(float(ev["credited_cash_inr"]), 2)})

            # Helper: Redeem LIQUIDBEES at Open to fund buys
            def _redeem_liquid(needed_inr: float, sig_ts: str) -> None:
                nonlocal liquid_units, liquid_cost_basis, realized_liquid_pnl, settled_cash
                if needed_inr <= 500.0 or liquid_units * lb_px_o < 500.0:
                    return
                sell_amt = min(needed_inr * 1.005, liquid_units * lb_px_o)
                units_sell = min(liquid_units, sell_amt / max(1.0, lb_px_o))
                lc = self.cost_engine.calculate_leg_cost(
                    instrument_class="LIQUID_ETF",
                    side="SELL",
                    raw_notional=units_sell * lb_px_o,
                    trade_date=dt,
                    dp_already_charged_today=((dt, "LIQUIDBEES") in dp_pairs),
                )
                if lc["dp_charge"] > 0:
                    dp_pairs.add((dt, "LIQUIDBEES"))
                net_cash = float(lc["exec_notional"]) - float(lc["regulatory_total"])
                if net_cash <= 0.0:
                    return
                avg_u_cost = liquid_cost_basis / max(1e-8, liquid_units)
                sold_basis = avg_u_cost * units_sell
                realized_liquid_pnl += net_cash - sold_basis
                liquid_cost_basis = max(0.0, liquid_cost_basis - sold_basis)
                liquid_units = max(0.0, liquid_units - units_sell)
                settled_cash += net_cash
                oid = _next_order_id("LIQS")
                _record_order_and_cost(
                    order_id=oid,
                    signal_ts=sig_ts,
                    strategy_id=policy.policy_id,
                    instrument_class="LIQUID_ETF",
                    sec_id="LIQUIDBEES",
                    symbol="LIQUIDBEES",
                    side="SELL",
                    req_qty=units_sell,
                    fill_qty=units_sell,
                    expected_px=lb_px_o,
                    exec_px=float(lc["exec_notional"]) / max(1e-8, units_sell),
                    cost_dict=lc,
                    status="FILLED",
                )

            # Helper: Rebalance Core NIFTYBEES at Open
            def _rebalance_core(target_core_inr: float, sig_ts: str, *, force_trim: bool = False) -> None:
                nonlocal core_qty, core_cost_basis, realized_core_pnl, settled_cash
                cur_core_val = core_qty * nb_px_o
                unsettled_now = float(
                    conn.execute("SELECT COALESCE(SUM(amount_inr), 0.0) FROM settlement_ledger WHERE status = 'PENDING'").fetchone()[0]
                )
                cur_total_est = (
                    settled_cash
                    + unsettled_now
                    + cur_core_val
                    + (liquid_units * lb_px_o)
                    + sum(float(p["last_price"]) * float(p["qty"]) for p in open_pos.values())
                )
                band_inr = cur_total_est * (ecfg.core_rebal_band_pct / 100.0)
                diff_inr = target_core_inr - cur_core_val
                if (not force_trim) and abs(diff_inr) < max(ecfg.min_core_trade_inr, band_inr):
                    return
                if diff_inr > ecfg.min_core_trade_inr:
                    if not entries_permitted_by_state:
                        return
                    if settled_cash < diff_inr:
                        _redeem_liquid(diff_inr - settled_cash, sig_ts)
                    avail_core = min(diff_inr, settled_cash * 0.996)
                    buy_qty = int(avail_core // max(1.0, nb_px_o * 1.001))
                    if buy_qty < 1 or (buy_qty * nb_px_o) < ecfg.min_core_trade_inr:
                        return
                    cc = self.cost_engine.calculate_leg_cost(
                        instrument_class="EQUITY_ETF",
                        side="BUY",
                        raw_notional=buy_qty * nb_px_o,
                        trade_date=dt,
                        adv_20d_inr=1_500_000_000.0,
                        daily_vol_frac=0.008,
                    )
                    tot_out = float(cc["exec_notional"]) + float(cc["regulatory_total"])
                    if tot_out > settled_cash:
                        buy_qty = int((settled_cash * 0.995) // max(1.0, nb_px_o * 1.001))
                        if buy_qty < 1:
                            return
                        cc = self.cost_engine.calculate_leg_cost(
                            instrument_class="EQUITY_ETF",
                            side="BUY",
                            raw_notional=buy_qty * nb_px_o,
                            trade_date=dt,
                            adv_20d_inr=1_500_000_000.0,
                            daily_vol_frac=0.008,
                        )
                        tot_out = float(cc["exec_notional"]) + float(cc["regulatory_total"])
                    if tot_out <= settled_cash:
                        settled_cash -= tot_out
                        core_qty += buy_qty
                        core_cost_basis += tot_out
                        oid = _next_order_id("CORB")
                        _record_order_and_cost(
                            order_id=oid,
                            signal_ts=sig_ts,
                            strategy_id=policy.policy_id,
                            instrument_class="EQUITY_ETF",
                            sec_id="NIFTYBEES",
                            symbol="NIFTYBEES",
                            side="BUY",
                            req_qty=buy_qty,
                            fill_qty=buy_qty,
                            expected_px=nb_px_o,
                            exec_px=float(cc["exec_notional"]) / max(1, buy_qty),
                            cost_dict=cc,
                            status="FILLED",
                        )
                elif diff_inr < -ecfg.min_core_trade_inr and core_qty > 0:
                    sell_qty = core_qty if target_core_inr <= 0.0 else min(core_qty, int(abs(diff_inr) // max(1.0, nb_px_o)))
                    if sell_qty < 1:
                        return
                    cc = self.cost_engine.calculate_leg_cost(
                        instrument_class="EQUITY_ETF",
                        side="SELL",
                        raw_notional=sell_qty * nb_px_o,
                        trade_date=dt,
                        adv_20d_inr=1_500_000_000.0,
                        daily_vol_frac=0.008,
                        dp_already_charged_today=((dt, "NIFTYBEES") in dp_pairs),
                    )
                    if cc["dp_charge"] > 0:
                        dp_pairs.add((dt, "NIFTYBEES"))
                    net_in = float(cc["exec_notional"]) - float(cc["regulatory_total"])
                    avg_basis = core_cost_basis / max(1, core_qty)
                    sold_basis = avg_basis * sell_qty
                    realized_core_pnl += net_in - sold_basis
                    core_cost_basis = max(0.0, core_cost_basis - sold_basis)
                    core_qty -= sell_qty
                    oid = _next_order_id("CORS")
                    _record_order_and_cost(
                        order_id=oid,
                        signal_ts=sig_ts,
                        strategy_id=policy.policy_id,
                        instrument_class="EQUITY_ETF",
                        sec_id="NIFTYBEES",
                        symbol="NIFTYBEES",
                        side="SELL",
                        req_qty=sell_qty,
                        fill_qty=sell_qty,
                        expected_px=nb_px_o,
                        exec_px=float(cc["exec_notional"]) / max(1, sell_qty),
                        cost_dict=cc,
                        status="FILLED",
                    )
                    if ecfg.settlement_lag_days <= 0:
                        settled_cash += net_in
                    else:
                        conn.execute(
                            """
                            INSERT INTO settlement_ledger(
                                settlement_id, order_id, trade_date, settlement_date,
                                due_session_counter, symbol, amount_inr, status, observation_mode
                            ) VALUES (?, ?, ?, ?, ?, 'NIFTYBEES', ?, 'PENDING', ?)
                            """,
                            (f"SET-{oid}", oid, dt, f"T+{ecfg.settlement_lag_days}", session_k + ecfg.settlement_lag_days, round(net_in, 6), obs_mode),
                        )

            # --------------------------------------------------------------
            # STEP 1: Execute Queued Orders from Prior Session Close at Today's Open
            # --------------------------------------------------------------
            vacated_slot = False
            queued_core_order = next((o for o in pending_orders if o.get("kind") == "CORE_REBALANCE"), None)
            queued_liquid_sweep = next((o for o in pending_orders if o.get("kind") == "LIQUID_SWEEP"), None)
            queued_tact_sells = [o for o in pending_orders if o.get("kind") == "TACTICAL_SELL"]
            queued_tact_buys = [o for o in pending_orders if o.get("kind") == "TACTICAL_BUY"]
            carry_pending: list[dict[str, Any]] = []

            for order in queued_tact_sells:
                sec_id = str(order["sec_id"])
                pos = open_pos.get(sec_id)
                if pos is None:
                    continue
                bar = eq_bars.get(sec_id)
                if not bar or not bool(bar.get("valid", True)) or float(bar.get("open", 0.0) or 0.0) <= 0:
                    carry_pending.append(order)
                    continue
                sym = str(pos["symbol"])
                o_px = float(bar["open"])
                qty_sell = int(pos["qty"])
                sc = self.cost_engine.calculate_leg_cost(
                    instrument_class="DELIVERY",
                    side="SELL",
                    raw_notional=qty_sell * o_px,
                    trade_date=dt,
                    slippage_bps=ecfg.slippage_override_bps,
                    adv_20d_inr=float(bar.get("adv_20d_inr") or 80_000_000.0),
                    daily_vol_frac=float(bar.get("vol_20d") or 0.018),
                    hl_range_pct=float(bar.get("hl_range_pct") or 2.0),
                    execution_mode=ecfg.execution_mode,
                    dp_already_charged_today=((dt, sym) in dp_pairs),
                )
                if sc["dp_charge"] > 0:
                    dp_pairs.add((dt, sym))
                net_proc = float(sc["exec_notional"]) - float(sc["regulatory_total"])
                oid = _next_order_id("SELL")
                _record_order_and_cost(
                    order_id=oid,
                    signal_ts=str(order["signal_timestamp"]),
                    strategy_id=str(pos["strategy_id"]),
                    instrument_class="DELIVERY",
                    sec_id=sec_id,
                    symbol=sym,
                    side="SELL",
                    req_qty=qty_sell,
                    fill_qty=qty_sell,
                    expected_px=o_px,
                    exec_px=float(sc["exec_notional"]) / max(1, qty_sell),
                    cost_dict=sc,
                    status="FILLED",
                )
                if ecfg.settlement_lag_days <= 0:
                    settled_cash += net_proc
                else:
                    conn.execute(
                        """
                        INSERT INTO settlement_ledger(
                            settlement_id, order_id, trade_date, settlement_date,
                            due_session_counter, symbol, amount_inr, status, observation_mode
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, 'PENDING', ?)
                        """,
                        (f"SET-{oid}", oid, dt, f"T+{ecfg.settlement_lag_days}", session_k + ecfg.settlement_lag_days, sym, round(net_proc, 6), obs_mode),
                    )
                pnl_price = net_proc - float(pos["cost_basis_inr"])
                realized_tact_pnl += pnl_price
                conn.execute(
                    """
                    UPDATE positions SET
                        status = 'CLOSED', exit_date = ?, last_price = ?, bars_held = ?,
                        dividends_received_inr = ?, realized_pnl_inr = ?, exit_reason = ?, exit_regime = ?
                    WHERE position_id = ?
                    """,
                    (
                        dt,
                        o_px,
                        int(pos["bars_held"]),
                        float(pos["dividends_received_inr"]),
                        pnl_price,
                        str(order.get("reason") or "ORCHESTRATOR_EXIT"),
                        confirmed_regime,
                        str(pos["position_id"]),
                    ),
                )
                del open_pos[sec_id]
                cooldown_map[sec_id] = session_k + ecfg.cooldown_days
                vacated_slot = True

            if queued_tact_buys and entries_permitted_by_state:
                if queued_core_order is not None:
                    override_core_zero = (
                        self._regime_policy_override is not None
                        and policy_map.get(confirmed_regime, (1.0,))[0] <= 0.0
                    )
                    tgt_c_inr = 0.0 if override_core_zero else float(queued_core_order["target_core_inr"])
                    if (core_qty * nb_px_o) > tgt_c_inr + ecfg.min_core_trade_inr:
                        _rebalance_core(tgt_c_inr, str(queued_core_order["signal_timestamp"]), force_trim=True)
                est_buy = 0.0
                for o in queued_tact_buys:
                    b = eq_bars.get(str(o["sec_id"]))
                    if b and bool(b.get("valid", True)) and float(b.get("open", 0.0) or 0.0) > 0:
                        est_buy += int(o["req_qty"]) * float(b["open"])
                if est_buy > settled_cash:
                    _redeem_liquid(est_buy - settled_cash, str(queued_tact_buys[0]["signal_timestamp"]))

            for order in queued_tact_buys:
                sec_id = str(order["sec_id"])
                sym = str(order["symbol"])
                sig_ts = str(order["signal_timestamp"])
                strat_id = str(order.get("strategy_id", "STRAT_E_DEFENSIVE_YIELD_LOWVOL"))
                self.safety_guard.assert_paper_only(strategy_ids=[strat_id])

                if not entries_permitted_by_state:
                    _record_order_and_cost(
                        order_id=_next_order_id("REJ"),
                        signal_ts=sig_ts,
                        strategy_id=strat_id,
                        instrument_class="DELIVERY",
                        sec_id=sec_id,
                        symbol=sym,
                        side="BUY",
                        req_qty=int(order["req_qty"]),
                        fill_qty=0,
                        expected_px=float(order.get("expected_price", 0.0)),
                        exec_px=0.0,
                        cost_dict=None,
                        status="REJECTED",
                        rejection_reason=f"ENGINE_{eng_state}_ENTRIES_HALTED",
                        ev_net_bps=float(order.get("ev_net_bps", 0.0)),
                        ev_hurdle_bps=float(order.get("ev_hurdle_bps", 200.0)),
                    )
                    continue

                if sec_id in open_pos or len(open_pos) >= ecfg.max_tactical_positions:
                    _record_order_and_cost(
                        order_id=_next_order_id("REJ"),
                        signal_ts=sig_ts,
                        strategy_id=strat_id,
                        instrument_class="DELIVERY",
                        sec_id=sec_id,
                        symbol=sym,
                        side="BUY",
                        req_qty=int(order["req_qty"]),
                        fill_qty=0,
                        expected_px=float(order.get("expected_price", 0.0)),
                        exec_px=0.0,
                        cost_dict=None,
                        status="REJECTED",
                        rejection_reason="DUPLICATE_OR_MAX_POSITIONS",
                        ev_net_bps=float(order.get("ev_net_bps", 0.0)),
                        ev_hurdle_bps=float(order.get("ev_hurdle_bps", 200.0)),
                    )
                    continue

                bar = eq_bars.get(sec_id)
                if not bar or (not bool(bar.get("valid", True))) or float(bar.get("open", 0.0) or 0.0) <= 0:
                    _record_order_and_cost(
                        order_id=_next_order_id("FAIL"),
                        signal_ts=sig_ts,
                        strategy_id=strat_id,
                        instrument_class="DELIVERY",
                        sec_id=sec_id,
                        symbol=sym,
                        side="BUY",
                        req_qty=int(order["req_qty"]),
                        fill_qty=0,
                        expected_px=float(order.get("expected_price", 0.0)),
                        exec_px=0.0,
                        cost_dict=None,
                        status="FAILED",
                        rejection_reason="MISSING_OR_INVALID_EXECUTION_BAR",
                        ev_net_bps=float(order.get("ev_net_bps", 0.0)),
                        ev_hurdle_bps=float(order.get("ev_hurdle_bps", 200.0)),
                    )
                    continue

                o_px = float(bar["open"])
                prev_c = float(bar.get("prev_close") or o_px)
                atr_v = float(bar.get("atr_14") or (o_px * 0.025))
                adv_v = float(bar.get("adv_20d_inr") or 80_000_000.0)
                hl_v = float(bar.get("hl_range_pct") or 2.0)
                vol_v = float(bar.get("vol_20d") or 0.018)

                fill = simulate_execution_fill(
                    execution_mode=ecfg.execution_mode,
                    side="BUY",
                    req_qty=int(order["req_qty"]),
                    prev_close=prev_c,
                    bar_open=o_px,
                    bar_high=float(bar.get("high", o_px)),
                    bar_low=float(bar.get("low", o_px)),
                    bar_close=float(bar.get("close", o_px)),
                    bar_vol=float(bar.get("volume", 100_000.0)),
                    adv_20d_inr=adv_v,
                    atr_14=atr_v,
                    hl_range_pct=hl_v,
                    max_bar_vol_part=ecfg.max_daily_vol_participation,
                    min_viable_notional_inr=ecfg.min_tactical_lot_inr * 0.70,
                )
                if fill["status"] in ("FAILED", "UNFILLED") or int(fill["fill_qty"]) < 1:
                    _record_order_and_cost(
                        order_id=_next_order_id("FAIL"),
                        signal_ts=sig_ts,
                        strategy_id=strat_id,
                        instrument_class="DELIVERY",
                        sec_id=sec_id,
                        symbol=sym,
                        side="BUY",
                        req_qty=int(order["req_qty"]),
                        fill_qty=0,
                        expected_px=o_px,
                        exec_px=0.0,
                        cost_dict=None,
                        status=str(fill["status"]),
                        rejection_reason=str(fill.get("reason") or "EXECUTION_FILL_FAILED"),
                        ev_net_bps=float(order.get("ev_net_bps", 0.0)),
                        ev_hurdle_bps=float(order.get("ev_hurdle_bps", 200.0)),
                    )
                    continue

                raw_fill_px = float(fill["raw_fill_px"])
                fill_qty = min(int(fill["fill_qty"]), int((settled_cash * 0.996) // max(1e-6, raw_fill_px * 1.003)))
                if fill_qty < 1 or (fill_qty * raw_fill_px) < ecfg.min_tactical_lot_inr * 0.70:
                    _record_order_and_cost(
                        order_id=_next_order_id("REJ"),
                        signal_ts=sig_ts,
                        strategy_id=strat_id,
                        instrument_class="DELIVERY",
                        sec_id=sec_id,
                        symbol=sym,
                        side="BUY",
                        req_qty=int(order["req_qty"]),
                        fill_qty=0,
                        expected_px=raw_fill_px,
                        exec_px=0.0,
                        cost_dict=None,
                        status="REJECTED",
                        rejection_reason="INSUFFICIENT_SETTLED_CASH_OR_BELOW_MIN_LOT",
                        ev_net_bps=float(order.get("ev_net_bps", 0.0)),
                        ev_hurdle_bps=float(order.get("ev_hurdle_bps", 200.0)),
                    )
                    continue

                bc = self.cost_engine.calculate_leg_cost(
                    instrument_class="DELIVERY",
                    side="BUY",
                    raw_notional=fill_qty * raw_fill_px,
                    trade_date=dt,
                    slippage_bps=ecfg.slippage_override_bps,
                    adv_20d_inr=adv_v,
                    daily_vol_frac=vol_v,
                    hl_range_pct=hl_v,
                    execution_mode=ecfg.execution_mode,
                )
                tot_out = float(bc["exec_notional"]) + float(bc["regulatory_total"])
                if tot_out > settled_cash:
                    _record_order_and_cost(
                        order_id=_next_order_id("REJ"),
                        signal_ts=sig_ts,
                        strategy_id=strat_id,
                        instrument_class="DELIVERY",
                        sec_id=sec_id,
                        symbol=sym,
                        side="BUY",
                        req_qty=int(order["req_qty"]),
                        fill_qty=0,
                        expected_px=raw_fill_px,
                        exec_px=0.0,
                        cost_dict=None,
                        status="REJECTED",
                        rejection_reason="TOTAL_OUTFLOW_EXCEEDS_SETTLED_CASH",
                        ev_net_bps=float(order.get("ev_net_bps", 0.0)),
                        ev_hurdle_bps=float(order.get("ev_hurdle_bps", 200.0)),
                    )
                    continue

                settled_cash -= tot_out
                oid = _next_order_id("BUY")
                eff_entry = float(bc["exec_notional"]) / fill_qty
                fill_status = "PARTIAL" if fill_qty < int(order["req_qty"]) else "FILLED"
                _record_order_and_cost(
                    order_id=oid,
                    signal_ts=sig_ts,
                    strategy_id=strat_id,
                    instrument_class="DELIVERY",
                    sec_id=sec_id,
                    symbol=sym,
                    side="BUY",
                    req_qty=int(order["req_qty"]),
                    fill_qty=fill_qty,
                    expected_px=raw_fill_px,
                    exec_px=eff_entry,
                    cost_dict=bc,
                    status=fill_status,
                    ev_net_bps=float(order.get("ev_net_bps", 0.0)),
                    ev_hurdle_bps=float(order.get("ev_hurdle_bps", 200.0)),
                )

                sl_r = float(order.get("stop_ratio") or 0.09)
                pos_id = f"POS-{dt.replace('-', '')}-{sec_id}-{order_seq:03d}"
                new_pos = {
                    "position_id": pos_id,
                    "instrument_type": "TACTICAL_EQUITY",
                    "sec_id": sec_id,
                    "symbol": sym,
                    "sector": str(bar.get("sector", "UNKNOWN_SECTOR")),
                    "industry": str(bar.get("industry", "UNKNOWN_INDUSTRY")),
                    "strategy_id": strat_id,
                    "status": "OPEN",
                    "entry_date": dt,
                    "exit_date": "",
                    "qty": float(fill_qty),
                    "avg_entry_price": eff_entry,
                    "last_price": eff_entry,
                    "cost_basis_inr": tot_out,
                    "buy_exec_notional_inr": float(bc["exec_notional"]),
                    "buy_total_cost_inr": float(bc["total_cost"]),
                    "sl_price": eff_entry * (1.0 - sl_r),
                    "atr_entry": max(eff_entry * 0.015, eff_entry * sl_r / 2.6),
                    "max_price": eff_entry,
                    "bars_held": 0,
                    "hold_max_days": int(order.get("hold_max_days") or ecfg.hold_max_days),
                    "be_locked": 0,
                    "mae_pct": 0.0,
                    "mfe_pct": 0.0,
                    "dividends_received_inr": 0.0,
                    "realized_pnl_inr": 0.0,
                    "exit_reason": "",
                    "entry_regime": confirmed_regime,
                    "exit_regime": "",
                }
                open_pos[sec_id] = new_pos
                conn.execute(
                    """
                    INSERT INTO positions(
                        position_id, instrument_type, sec_id, symbol, sector, industry, strategy_id,
                        status, entry_date, exit_date, qty, avg_entry_price, last_price,
                        cost_basis_inr, buy_exec_notional_inr, buy_total_cost_inr, sl_price,
                        atr_entry, max_price, bars_held, hold_max_days, be_locked, mae_pct, mfe_pct,
                        dividends_received_inr, realized_pnl_inr, exit_reason, entry_regime, exit_regime
                    ) VALUES (
                        :position_id, :instrument_type, :sec_id, :symbol, :sector, :industry, :strategy_id,
                        :status, :entry_date, :exit_date, :qty, :avg_entry_price, :last_price,
                        :cost_basis_inr, :buy_exec_notional_inr, :buy_total_cost_inr, :sl_price,
                        :atr_entry, :max_price, :bars_held, :hold_max_days, :be_locked, :mae_pct, :mfe_pct,
                        :dividends_received_inr, :realized_pnl_inr, :exit_reason, :entry_regime, :exit_regime
                    )
                    """,
                    new_pos,
                )

            # Execute queued Core Rebalance & Liquid Sweep at today's Open
            override_core_zero_post = (
                self._regime_policy_override is not None
                and policy_map.get(confirmed_regime, (1.0,))[0] <= 0.0
            )
            if queued_core_order is not None:
                tgt_post = 0.0 if override_core_zero_post else float(queued_core_order["target_core_inr"])
                _rebalance_core(tgt_post, str(queued_core_order["signal_timestamp"]), force_trim=override_core_zero_post)
            elif override_core_zero_post and core_qty > 0:
                _rebalance_core(0.0, f"{dt}T09:15:00+05:30", force_trim=True)

            if queued_liquid_sweep is not None and entries_permitted_by_state:
                excess_cash = settled_cash - ecfg.liquid_sweep_min_cash_inr
                if excess_cash >= max(2_500.0, ecfg.liquid_sweep_min_cash_inr * 0.75):
                    buy_liq_amt = excess_cash * 0.997
                    lc = self.cost_engine.calculate_leg_cost(
                        instrument_class="LIQUID_ETF", side="BUY", raw_notional=buy_liq_amt, trade_date=dt
                    )
                    tot_out = float(lc["exec_notional"]) + float(lc["regulatory_total"])
                    if tot_out <= settled_cash:
                        settled_cash -= tot_out
                        bought_units = buy_liq_amt / max(1.0, lb_px_o)
                        liquid_units += bought_units
                        liquid_cost_basis += tot_out
                        oid = _next_order_id("LIQB")
                        _record_order_and_cost(
                            order_id=oid,
                            signal_ts=str(queued_liquid_sweep["signal_timestamp"]),
                            strategy_id=policy.policy_id,
                            instrument_class="LIQUID_ETF",
                            sec_id="LIQUIDBEES",
                            symbol="LIQUIDBEES",
                            side="BUY",
                            req_qty=bought_units,
                            fill_qty=bought_units,
                            expected_px=lb_px_o,
                            exec_px=float(lc["exec_notional"]) / max(1e-8, bought_units),
                            cost_dict=lc,
                            status="FILLED",
                        )

            # --------------------------------------------------------------
            # STEP 2: Existing Risk Management & Stop-Loss Exits (ALWAYS active, even in Bear/SAFE_MODE!)
            # --------------------------------------------------------------
            for sec_id in list(open_pos.keys()):
                pos = open_pos[sec_id]
                pos["bars_held"] = int(pos["bars_held"]) + 1
                bar = eq_bars.get(sec_id)
                if not bar or not bool(bar.get("valid", True)):
                    continue
                o_v, h_v, l_v, c_v = float(bar["open"]), float(bar["high"]), float(bar["low"]), float(bar["close"])
                pos["last_price"] = c_v
                ex_lo = ((l_v / max(1e-6, float(pos["avg_entry_price"]))) - 1.0) * 100.0
                ex_hi = ((h_v / max(1e-6, float(pos["avg_entry_price"]))) - 1.0) * 100.0
                pos["mae_pct"] = min(float(pos["mae_pct"]), ex_lo)
                pos["mfe_pct"] = max(float(pos["mfe_pct"]), ex_hi)

                pos_eval_adapter = {
                    "entry_px": float(pos["avg_entry_price"]),
                    "sl_px": float(pos["sl_price"]),
                    "atr_entry": float(pos["atr_entry"]),
                    "max_px": float(pos["max_price"]),
                    "bars_held": int(pos["bars_held"]),
                    "be_locked": bool(pos["be_locked"]),
                }
                s50 = bar.get("sma_50")
                l20 = bar.get("low_20_prev")
                exit_reason, exit_px = evaluate_position_exit(
                    pos=pos_eval_adapter,
                    stop_model=ecfg.stop_model,
                    o_v=o_v,
                    h_v=h_v,
                    l_v=l_v,
                    c_v=c_v,
                    sma_50_v=float(s50) if s50 is not None and not np.isnan(float(s50)) else None,
                    low_20_prev=float(l20) if l20 is not None and not np.isnan(float(l20)) else None,
                    vol_20d_daily=float(bar.get("vol_20d") or 0.018),
                    hold_max_days=int(pos["hold_max_days"]),
                    is_last_bar=False,
                )
                pos["sl_price"] = float(pos_eval_adapter["sl_px"])
                pos["max_price"] = float(pos_eval_adapter["max_px"])
                pos["be_locked"] = 1 if pos_eval_adapter["be_locked"] else 0

                if exit_reason is not None and exit_px is not None:
                    sym = str(pos["symbol"])
                    sc = self.cost_engine.calculate_leg_cost(
                        instrument_class="DELIVERY",
                        side="SELL",
                        raw_notional=int(pos["qty"]) * float(exit_px),
                        trade_date=dt,
                        slippage_bps=ecfg.slippage_override_bps,
                        adv_20d_inr=float(bar.get("adv_20d_inr") or 80_000_000.0),
                        daily_vol_frac=float(bar.get("vol_20d") or 0.018),
                        hl_range_pct=float(bar.get("hl_range_pct") or 2.0),
                        execution_mode=ecfg.execution_mode,
                        dp_already_charged_today=((dt, sym) in dp_pairs),
                    )
                    if sc["dp_charge"] > 0:
                        dp_pairs.add((dt, sym))
                    net_proc = float(sc["exec_notional"]) - float(sc["regulatory_total"])
                    oid = _next_order_id("SELL")
                    _record_order_and_cost(
                        order_id=oid,
                        signal_ts=f"{dt}T15:30:00+05:30",
                        strategy_id=str(pos["strategy_id"]),
                        instrument_class="DELIVERY",
                        sec_id=sec_id,
                        symbol=sym,
                        side="SELL",
                        req_qty=int(pos["qty"]),
                        fill_qty=int(pos["qty"]),
                        expected_px=float(exit_px),
                        exec_px=float(sc["exec_notional"]) / max(1, int(pos["qty"])),
                        cost_dict=sc,
                        status="FILLED",
                    )
                    if ecfg.settlement_lag_days <= 0:
                        settled_cash += net_proc
                    else:
                        conn.execute(
                            """
                            INSERT INTO settlement_ledger(
                                settlement_id, order_id, trade_date, settlement_date,
                                due_session_counter, symbol, amount_inr, status, observation_mode
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, 'PENDING', ?)
                            """,
                            (f"SET-{oid}", oid, dt, f"T+{ecfg.settlement_lag_days}", session_k + ecfg.settlement_lag_days, sym, round(net_proc, 6), obs_mode),
                        )
                    pnl_price = net_proc - float(pos["cost_basis_inr"])
                    realized_tact_pnl += pnl_price
                    conn.execute(
                        """
                        UPDATE positions SET
                            status = 'CLOSED', exit_date = ?, last_price = ?, sl_price = ?,
                            max_price = ?, bars_held = ?, be_locked = ?, mae_pct = ?, mfe_pct = ?,
                            dividends_received_inr = ?, realized_pnl_inr = ?, exit_reason = ?, exit_regime = ?
                        WHERE position_id = ?
                        """,
                        (
                            dt,
                            c_v,
                            pos["sl_price"],
                            pos["max_price"],
                            pos["bars_held"],
                            pos["be_locked"],
                            pos["mae_pct"],
                            pos["mfe_pct"],
                            pos["dividends_received_inr"],
                            pnl_price,
                            str(exit_reason),
                            confirmed_regime,
                            str(pos["position_id"]),
                        ),
                    )
                    del open_pos[sec_id]
                    cooldown_map[sec_id] = session_k + ecfg.cooldown_days
                    vacated_slot = True
                else:
                    conn.execute(
                        """
                        UPDATE positions SET
                            last_price = ?, sl_price = ?, max_price = ?, bars_held = ?,
                            be_locked = ?, mae_pct = ?, mfe_pct = ?, dividends_received_inr = ?
                        WHERE position_id = ?
                        """,
                        (
                            pos["last_price"],
                            pos["sl_price"],
                            pos["max_price"],
                            pos["bars_held"],
                            pos["be_locked"],
                            pos["mae_pct"],
                            pos["mfe_pct"],
                            pos["dividends_received_inr"],
                            str(pos["position_id"]),
                        ),
                    )

            # --------------------------------------------------------------
            # STEP 3: Strictly Causal End-of-Day Regime Classification (Session T Close Only!)
            # --------------------------------------------------------------
            nb_hist = [float(x) for x in (market_snapshot.get("niftybees_close_history") or []) if x is not None and not np.isnan(float(x)) and float(x) > 0]
            if not nb_hist:
                nb_hist = [nb_px_c]
            # Use exact M3.1 `classify_simple_trend_baseline` logic on strictly causal history <= dt
            sma_window = 200
            if len(nb_hist) >= sma_window:
                sma200_val = float(np.mean(nb_hist[-sma_window:]))
                px_vs_sma200_pct = ((nb_hist[-1] / sma200_val) - 1.0) * 100.0
                raw_regime = classify_simple_trend_baseline(
                    {"dates": [dt], "px_vs_sma200_pct": np.array([px_vs_sma200_pct])},
                    sma_window=200,
                )[0]
            else:
                sma200_val = float(np.mean(nb_hist))
                raw_regime = "BULL_TREND"

            if raw_regime == prev_raw_regime:
                regime_streak += 1
            else:
                prev_raw_regime = raw_regime
                regime_streak = 1

            if regime_streak >= policy.allocation_confirm_days and raw_regime in policy_map:
                confirmed_regime = raw_regime
                cur_tgt_c, cur_tgt_t, cur_tgt_l, cur_allow_entries, cur_hurdle_adder, cur_size_mult = policy_map[confirmed_regime]
            else:
                _, _, _, _, cur_hurdle_adder, cur_size_mult = policy_map.get(
                    confirmed_regime, (cur_tgt_c, cur_tgt_t, cur_tgt_l, cur_allow_entries, 0.0, 1.0)
                )

            # --------------------------------------------------------------
            # STEP 4: End-of-Day NAV & Benchmark Valuation (Strictly T Close)
            # --------------------------------------------------------------
            unsettled_now = float(
                conn.execute("SELECT COALESCE(SUM(amount_inr), 0.0) FROM settlement_ledger WHERE status = 'PENDING'").fetchone()[0]
            )
            core_val_c = core_qty * nb_px_c
            liquid_val_c = liquid_units * lb_px_c
            tact_val_c = sum(float(p["qty"]) * float(p["last_price"]) for p in open_pos.values())
            nav_c = settled_cash + unsettled_now + core_val_c + liquid_val_c + tact_val_c
            peak_nav = max(peak_nav, nav_c)
            dd_pct = ((peak_nav - nav_c) / max(1e-6, peak_nav)) * 100.0
            daily_pnl = nav_c - prev_nav if session_k > 1 else (nav_c - self.initial_capital_inr)
            daily_ret_pct = ((nav_c / max(1e-6, prev_nav)) - 1.0) * 100.0 if session_k > 1 else ((nav_c / self.initial_capital_inr) - 1.0) * 100.0
            cum_ret_pct = ((nav_c / self.initial_capital_inr) - 1.0) * 100.0

            # Aligned executable Benchmarks (100% NIFTYBEES & 45/55 NIFTYBEES/LIQUIDBEES)
            b_nb_base = float(ps["bench_niftybees_base_px"])
            b_lb_base = float(ps["bench_liquid_base_px"])
            b45_nb_u = float(ps["bench_45_55_nifty_units"])
            b45_lb_u = float(ps["bench_45_55_liquid_units"])
            if b_nb_base <= 0.0:
                b_nb_base = nb_px_c
                b_lb_base = lb_px_c
                b45_nb_u = (self.initial_capital_inr * 0.45) / max(1.0, nb_px_c)
                b45_lb_u = (self.initial_capital_inr * 0.55) / max(1.0, lb_px_c)
            else:
                b45_lb_u += b45_lb_u * LIQUID_BEES_DAILY_YIELD
            bench_nb_nav = self.initial_capital_inr * (nb_px_c / max(1.0, b_nb_base))
            bench_45_55_nav = (b45_nb_u * nb_px_c) + (b45_lb_u * lb_px_c)

            uninvested_cash_pct = (settled_cash / max(1.0, nav_c)) * 100.0
            consec_high_cash = (consec_high_cash + 1) if uninvested_cash_pct > 12.0 else 0

            # --------------------------------------------------------------
            # STEP 5: End-of-Day Signal Generation for NEXT SESSION OPEN (T+1)
            # --------------------------------------------------------------
            next_pending_orders: list[dict[str, Any]] = list(carry_pending)
            sig_ts_eod = f"{dt}T15:30:00+05:30"

            # Queue active orchestrator exit signals for NEXT SESSION OPEN (T+1)
            exiting_sec_ids: set[str] = set()
            for ex in (market_snapshot.get("tactical_exit_signals") or []):
                ex_sid = str(ex.get("sec_id") or "")
                if not ex_sid or ex_sid not in open_pos or ex_sid in exiting_sec_ids:
                    continue
                pos = open_pos[ex_sid]
                exiting_sec_ids.add(ex_sid)
                next_pending_orders.append({
                    "kind": "TACTICAL_SELL",
                    "sec_id": ex_sid,
                    "symbol": str(pos["symbol"]),
                    "strategy_id": str(pos["strategy_id"]),
                    "req_qty": int(pos["qty"]),
                    "reason": str(ex.get("reason") or "ORCHESTRATOR_EXIT"),
                    "signal_timestamp": sig_ts_eod,
                })
                signals_today.append({
                    "sec_id": ex_sid,
                    "symbol": str(pos["symbol"]),
                    "decision": "QUEUED_EXIT_NEXT_OPEN",
                    "reason": str(ex.get("reason") or "ORCHESTRATOR_EXIT"),
                })

            # Target Core ETF rebalance queued for T+1 open
            unused_tact = max(0.0, (nav_c * (cur_tgt_t / 100.0)) - tact_val_c)
            eff_core_target_inr = nav_c * (cur_tgt_c / 100.0)
            if ecfg.sweep_unused_tactical_to_core and cur_allow_entries:
                eff_core_target_inr += unused_tact * 0.45
            next_pending_orders.append({
                "kind": "CORE_REBALANCE",
                "target_core_inr": round(eff_core_target_inr, 2),
                "signal_timestamp": sig_ts_eod,
            })

            # Candidate evaluation (ONLY if Bull regime allows new entries AND engine not halted/paused)
            can_replenish = (
                (session_k == 1)
                or (ecfg.event_driven_replenishment and vacated_slot and (session_k - last_replenish_session) >= ecfg.replenish_min_gap_days)
                or ((session_k - last_replenish_session) >= ecfg.periodic_review_interval_days)
            )
            if cur_allow_entries and entries_permitted_by_state and can_replenish and len(open_pos) < ecfg.max_tactical_positions:
                tact_budget_inr = nav_c * (cur_tgt_t / 100.0)
                rem_tact_inr = max(0.0, tact_budget_inr - tact_val_c)
                slots_open = ecfg.max_tactical_positions - len(open_pos)
                if rem_tact_inr >= ecfg.min_tactical_lot_inr and slots_open > 0:
                    sec_exp: dict[str, float] = {}
                    sec_cnt: dict[str, int] = {}
                    ind_exp: dict[str, float] = {}
                    for p in open_pos.values():
                        pv = float(p["qty"]) * float(p["last_price"])
                        s_nm = str(p["sector"])
                        i_nm = str(p["industry"])
                        sec_exp[s_nm] = sec_exp.get(s_nm, 0.0) + pv
                        sec_cnt[s_nm] = sec_cnt.get(s_nm, 0) + 1
                        ind_exp[i_nm] = ind_exp.get(i_nm, 0.0) + pv

                    queued_this_eod = 0
                    for cand in candidates:
                        if queued_this_eod >= slots_open or rem_tact_inr < ecfg.min_tactical_lot_inr:
                            break
                        sec_id = str(cand["sec_id"])
                        sym = str(cand["symbol"])
                        if sec_id in open_pos or sec_id in exiting_sec_ids or session_k < cooldown_map.get(sec_id, -1):
                            continue
                        bar = eq_bars.get(sec_id)
                        if not bar or not bool(bar.get("valid", True)):
                            continue
                        c_px = float(bar["close"])
                        if c_px <= 0:
                            continue
                        s_nm = str(bar.get("sector", "UNKNOWN_SECTOR"))
                        i_nm = str(bar.get("industry", "UNKNOWN_INDUSTRY"))
                        is_unk = bool(bar.get("is_unknown_sector", s_nm == "UNKNOWN_SECTOR"))

                        max_sec_pct = ecfg.max_unknown_sector_exposure_pct if is_unk else ecfg.max_sector_exposure_pct
                        sec_room = max(0.0, nav_c * (max_sec_pct / 100.0) - sec_exp.get(s_nm, 0.0))
                        ind_room = max(0.0, nav_c * (ecfg.max_industry_exposure_pct / 100.0) - ind_exp.get(i_nm, 0.0))
                        if sec_cnt.get(s_nm, 0) >= ecfg.max_sector_positions or sec_room < ecfg.min_tactical_lot_inr:
                            signals_today.append({"sec_id": sec_id, "symbol": sym, "decision": "REJECTED", "reason": "SECTOR_CAP"})
                            continue

                        max_corr = 0.0
                        for held_sid in open_pos:
                            c_val = corr_map.get((sec_id, held_sid), corr_map.get((held_sid, sec_id), 0.0))
                            max_corr = max(max_corr, float(c_val))
                        if max_corr > ecfg.max_pairwise_corr:
                            signals_today.append({"sec_id": sec_id, "symbol": sym, "decision": "REJECTED", "reason": "CORRELATION_CAP"})
                            continue

                        base_slot = (tact_budget_inr / max(1, ecfg.max_tactical_positions)) * cur_size_mult
                        est_lot = min(base_slot, rem_tact_inr, sec_room, ind_room, nav_c * (ecfg.max_single_stock_pct / 100.0))
                        ev = estimate_trade_economic_value(
                            candidate=cand,
                            lot_notional_inr=est_lot,
                            trade_date=dt,
                            daily_vol_frac=float(bar.get("vol_20d") or 0.018),
                            hl_range_pct=float(bar.get("hl_range_pct") or 2.0),
                            adv_20d_inr=float(bar.get("adv_20d_inr") or 80_000_000.0),
                            div_yield_ttm_frac=float(bar.get("div_yield_ttm") or 0.015),
                            regime=confirmed_regime,
                            cost_engine=self.cost_engine,
                            execution_mode=ecfg.execution_mode,
                            hurdle_mode=ecfg.hurdle_mode,
                            slippage_override_bps=ecfg.slippage_override_bps,
                        )
                        eff_hurdle = float(ev["hurdle_bps"]) + cur_hurdle_adder
                        if (not ev["approved"]) or float(ev["expected_net_alpha_bps"]) < eff_hurdle:
                            signals_today.append({
                                "sec_id": sec_id,
                                "symbol": sym,
                                "decision": "REJECTED",
                                "reason": "BELOW_EV_HURDLE",
                                "ev_net_bps": float(ev["expected_net_alpha_bps"]),
                                "hurdle_bps": eff_hurdle,
                            })
                            continue

                        stop_ratio = float(cand.get("stop_ratio") or 0.09)
                        sized_inr = compute_intelligent_position_size(
                            sizing_mode=ecfg.sizing_mode,
                            base_slot_inr=base_slot,
                            rem_tact_budget_inr=rem_tact_inr,
                            nav_inr=nav_c,
                            max_single_stock_pct=ecfg.max_single_stock_pct,
                            sector_room_inr=sec_room,
                            industry_room_inr=ind_room,
                            adv_20d_inr=float(bar.get("adv_20d_inr") or 80_000_000.0),
                            daily_vol_frac=float(bar.get("vol_20d") or 0.018),
                            stop_ratio=stop_ratio,
                            signal_score=float(cand.get("score") or 0.7),
                            expected_net_alpha_bps=float(ev["expected_net_alpha_bps"]),
                            hurdle_bps=eff_hurdle,
                            max_corr_with_open=max_corr,
                            min_lot_inr=ecfg.min_tactical_lot_inr,
                        )
                        req_qty = int(sized_inr // max(1e-6, c_px))
                        if req_qty < 1 or (req_qty * c_px) < ecfg.min_tactical_lot_inr:
                            signals_today.append({"sec_id": sec_id, "symbol": sym, "decision": "REJECTED", "reason": "SIZE_BELOW_MIN_LOT"})
                            continue

                        next_pending_orders.append({
                            "kind": "TACTICAL_BUY",
                            "sec_id": sec_id,
                            "symbol": sym,
                            "strategy_id": str(cand.get("strategy_id") or "STRAT_E_DEFENSIVE_YIELD_LOWVOL"),
                            "req_qty": req_qty,
                            "expected_price": round(c_px, 4),
                            "stop_ratio": stop_ratio,
                            "hold_max_days": ecfg.hold_max_days,
                            "score": float(cand.get("score") or 0.7),
                            "ev_net_bps": float(ev["expected_net_alpha_bps"]),
                            "ev_hurdle_bps": eff_hurdle,
                            "signal_timestamp": sig_ts_eod,
                        })
                        signals_today.append({
                            "sec_id": sec_id,
                            "symbol": sym,
                            "decision": "QUEUED_FOR_NEXT_OPEN",
                            "req_qty": req_qty,
                            "ref_close": round(c_px, 2),
                            "ev_net_bps": round(float(ev["expected_net_alpha_bps"]), 2),
                            "hurdle_bps": round(eff_hurdle, 2),
                        })
                        rem_tact_inr -= req_qty * c_px
                        sec_exp[s_nm] = sec_exp.get(s_nm, 0.0) + req_qty * c_px
                        sec_cnt[s_nm] = sec_cnt.get(s_nm, 0) + 1
                        ind_exp[i_nm] = ind_exp.get(i_nm, 0.0) + req_qty * c_px
                        queued_this_eod += 1
                    if queued_this_eod > 0:
                        last_replenish_session = session_k

            # Queue liquid sweep if excess cash exceeds floor
            if entries_permitted_by_state and settled_cash > ecfg.liquid_sweep_min_cash_inr + 2_500.0:
                next_pending_orders.append({"kind": "LIQUID_SWEEP", "signal_timestamp": sig_ts_eod})

            # --------------------------------------------------------------
            # STEP 6: Persist Updated Portfolio State & Daily NAV Ledger
            # --------------------------------------------------------------
            unrealized_total = (
                (core_val_c - core_cost_basis)
                + (liquid_val_c - liquid_cost_basis - cum_liquid_yield)
                + sum((float(p["qty"]) * float(p["last_price"])) - float(p["cost_basis_inr"]) for p in open_pos.values())
            )
            realized_total = realized_core_pnl + realized_liquid_pnl + realized_tact_pnl

            # Inline exact reconciliation check BEFORE committing!
            nav_equity_check = self.initial_capital_inr + realized_total + unrealized_total + cum_dividends + cum_liquid_yield
            recon_diff = abs(nav_c - nav_equity_check)

            conn.execute(
                """
                UPDATE portfolio_state SET
                    as_of_date = ?, settled_cash_inr = ?, unsettled_cash_inr = ?,
                    core_qty = ?, core_cost_basis_inr = ?, core_last_px = ?,
                    liquid_units = ?, liquid_cost_basis_inr = ?, liquid_last_px = ?,
                    realized_core_pnl_inr = ?, realized_liquid_pnl_inr = ?, realized_tactical_pnl_inr = ?,
                    cum_dividends_inr = ?, cum_liquid_yield_inr = ?, cum_costs_inr = ?, cum_turnover_inr = ?,
                    peak_nav_inr = ?, last_nav_inr = ?,
                    raw_regime = ?, confirmed_regime = ?, regime_confirm_streak = ?,
                    target_core_pct = ?, target_tactical_pct = ?, target_liquid_pct = ?,
                    allow_new_entries = ?, consecutive_high_cash_days = ?, session_counter = ?,
                    last_replenish_session = ?,
                    bench_niftybees_base_px = ?, bench_liquid_base_px = ?,
                    bench_45_55_liquid_units = ?, bench_45_55_nifty_units = ?,
                    pending_orders_json = ?, cooldown_json = ?, dp_pairs_json = ?, updated_at_utc = ?
                WHERE singleton_id = 1
                """,
                (
                    dt,
                    settled_cash,
                    unsettled_now,
                    core_qty,
                    core_cost_basis,
                    nb_px_c,
                    liquid_units,
                    liquid_cost_basis,
                    lb_px_c,
                    realized_core_pnl,
                    realized_liquid_pnl,
                    realized_tact_pnl,
                    cum_dividends,
                    cum_liquid_yield,
                    cum_costs,
                    cum_turnover,
                    peak_nav,
                    nav_c,
                    raw_regime,
                    confirmed_regime,
                    regime_streak,
                    cur_tgt_c,
                    cur_tgt_t,
                    cur_tgt_l,
                    1 if cur_allow_entries else 0,
                    consec_high_cash,
                    session_k,
                    last_replenish_session,
                    b_nb_base,
                    b_lb_base,
                    b45_lb_u,
                    b45_nb_u,
                    json.dumps(next_pending_orders),
                    json.dumps(cooldown_map),
                    json.dumps([list(x) for x in sorted(dp_pairs)]),
                    _utc_now_iso(),
                ),
            )

            next_engine_state = eng_state if eng_state != "IDLE" else "RUNNING"
            if recon_diff > RECONCILIATION_TOLERANCE_INR or settled_cash < -0.01:
                next_engine_state = "SAFE_MODE"
                self._set_meta(conn, "engine_state", "SAFE_MODE")
                self._set_meta(
                    conn,
                    "safe_mode_reason",
                    f"RECONCILIATION_BREACH on {dt}: diff={recon_diff:.4f} INR, cash={settled_cash:.2f}",
                )
            elif eng_state == "IDLE":
                self._set_meta(conn, "engine_state", "RUNNING")

            daily_row = {
                "trading_date": dt,
                "observation_mode": obs_mode,
                "nav_inr": round(nav_c, 2),
                "settled_cash_inr": round(settled_cash, 2),
                "unsettled_cash_inr": round(unsettled_now, 2),
                "core_value_inr": round(core_val_c, 2),
                "tactical_value_inr": round(tact_val_c, 2),
                "liquid_value_inr": round(liquid_val_c, 2),
                "realized_pnl_inr": round(realized_total, 2),
                "unrealized_pnl_inr": round(unrealized_total, 2),
                "cum_dividends_inr": round(cum_dividends, 2),
                "cum_liquid_yield_inr": round(cum_liquid_yield, 2),
                "cum_costs_inr": round(cum_costs, 2),
                "daily_pnl_inr": round(daily_pnl, 2),
                "daily_return_pct": round(daily_ret_pct, 4),
                "cumulative_return_pct": round(cum_ret_pct, 4),
                "drawdown_pct": round(dd_pct, 4),
                "raw_regime": raw_regime,
                "confirmed_regime": confirmed_regime,
                "regime_confirm_streak": regime_streak,
                "target_core_pct": cur_tgt_c,
                "target_tactical_pct": cur_tgt_t,
                "target_liquid_pct": cur_tgt_l,
                "allow_new_entries": 1 if cur_allow_entries else 0,
                "open_tactical_positions": len(open_pos),
                "benchmark_niftybees_nav": round(bench_nb_nav, 2),
                "benchmark_45_55_nav": round(bench_45_55_nav, 2),
                "reconciliation_diff_inr": round(recon_diff, 4),
                "data_quality_state": dq_state,
                "engine_state": next_engine_state,
            }
            conn.execute(
                """
                INSERT INTO daily_nav_ledger(
                    trading_date, observation_mode, nav_inr, settled_cash_inr, unsettled_cash_inr,
                    core_value_inr, tactical_value_inr, liquid_value_inr, realized_pnl_inr, unrealized_pnl_inr,
                    cum_dividends_inr, cum_liquid_yield_inr, cum_costs_inr, daily_pnl_inr,
                    daily_return_pct, cumulative_return_pct, drawdown_pct,
                    raw_regime, confirmed_regime, regime_confirm_streak,
                    target_core_pct, target_tactical_pct, target_liquid_pct, allow_new_entries,
                    open_tactical_positions, benchmark_niftybees_nav, benchmark_45_55_nav,
                    reconciliation_diff_inr, data_quality_state, engine_state
                ) VALUES (
                    :trading_date, :observation_mode, :nav_inr, :settled_cash_inr, :unsettled_cash_inr,
                    :core_value_inr, :tactical_value_inr, :liquid_value_inr, :realized_pnl_inr, :unrealized_pnl_inr,
                    :cum_dividends_inr, :cum_liquid_yield_inr, :cum_costs_inr, :daily_pnl_inr,
                    :daily_return_pct, :cumulative_return_pct, :drawdown_pct,
                    :raw_regime, :confirmed_regime, :regime_confirm_streak,
                    :target_core_pct, :target_tactical_pct, :target_liquid_pct, :allow_new_entries,
                    :open_tactical_positions, :benchmark_niftybees_nav, :benchmark_45_55_nav,
                    :reconciliation_diff_inr, :data_quality_state, :engine_state
                )
                """,
                daily_row,
            )

            conn.execute(
                """
                UPDATE session_checkpoints
                SET phase = 'COMPLETED', reconciliation_diff_inr = ?, completed_at_utc = ?
                WHERE trading_date = ?
                """,
                (round(recon_diff, 4), _utc_now_iso(), dt),
            )
            audit_sha = self._append_audit_event(
                conn,
                trading_date=dt,
                event_type="DAILY_SESSION_COMPLETED",
                payload={
                    "observation_mode": obs_mode,
                    "data_timestamp_utc": data_ts,
                    "config_sha256": self.verified_config.canonical_sha256,
                    "raw_regime": raw_regime,
                    "confirmed_regime": confirmed_regime,
                    "regime_confirm_streak": regime_streak,
                    "sma200_niftybees": round(sma200_val, 4),
                    "niftybees_close": round(nb_px_c, 4),
                    "allocation": {"core_pct": cur_tgt_c, "tactical_pct": cur_tgt_t, "liquid_pct": cur_tgt_l},
                    "orders_count": len(orders_today),
                    "signals_count": len(signals_today),
                    "nav_inr": daily_row["nav_inr"],
                    "reconciliation_diff_inr": daily_row["reconciliation_diff_inr"],
                    "engine_state": next_engine_state,
                },
            )
            conn.execute("COMMIT")
            return {
                "trading_date": dt,
                "status": "COMPLETED" if next_engine_state != "SAFE_MODE" else "COMPLETED_WITH_SAFE_MODE",
                "observation_mode": obs_mode,
                "idempotent_skip": False,
                "engine_state": next_engine_state,
                "reconciliation_diff_inr": daily_row["reconciliation_diff_inr"],
                "nav_snapshot": daily_row,
                "orders": orders_today,
                "signals": signals_today,
                "dividends": dividends_today,
                "pending_next_session_orders": next_pending_orders,
                "audit_chain_sha256": audit_sha,
            }
        except Exception:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            raise
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Read-Only Queries for Risk Monitor, Reporting & Dashboard
    # ------------------------------------------------------------------
    def get_full_ledger_snapshot(self) -> dict[str, Any]:
        """Return complete durable state, ledgers, and prospective validation progress."""
        conn = self._connect()
        try:
            ps = dict(conn.execute("SELECT * FROM portfolio_state WHERE singleton_id = 1").fetchone())
            open_positions = [
                dict(r)
                for r in conn.execute(
                    "SELECT * FROM positions WHERE status = 'OPEN' ORDER BY entry_date ASC"
                ).fetchall()
            ]
            closed_positions = [
                dict(r)
                for r in conn.execute(
                    "SELECT * FROM positions WHERE status = 'CLOSED' ORDER BY exit_date ASC"
                ).fetchall()
            ]
            orders = [
                dict(r) for r in conn.execute("SELECT * FROM simulated_orders ORDER BY trading_date ASC, order_id ASC").fetchall()
            ]
            costs = [
                dict(r) for r in conn.execute("SELECT * FROM transaction_cost_ledger ORDER BY trading_date ASC, cost_id ASC").fetchall()
            ]
            dividends = [
                dict(r) for r in conn.execute("SELECT * FROM dividend_ledger ORDER BY trading_date ASC, dividend_id ASC").fetchall()
            ]
            settlements = [
                dict(r) for r in conn.execute("SELECT * FROM settlement_ledger ORDER BY trade_date ASC, settlement_id ASC").fetchall()
            ]
            daily_navs = [
                dict(r) for r in conn.execute("SELECT * FROM daily_nav_ledger ORDER BY trading_date ASC").fetchall()
            ]
            checkpoints = [
                dict(r) for r in conn.execute("SELECT * FROM session_checkpoints ORDER BY trading_date ASC").fetchall()
            ]

            prospective_navs = [r for r in daily_navs if r["observation_mode"] == "PROSPECTIVE_PAPER"]
            prospective_orders = [o for o in orders if o["observation_mode"] == "PROSPECTIVE_PAPER"]

            # Build unified holdings list including Core NIFTYBEES and Liquid LIQUIDBEES
            all_open_holdings: list[dict[str, Any]] = []
            if int(ps["core_qty"]) > 0:
                cv = int(ps["core_qty"]) * float(ps["core_last_px"])
                cb = float(ps["core_cost_basis_inr"])
                all_open_holdings.append({
                    "position_id": "CORE-NIFTYBEES",
                    "instrument_type": "CORE_ETF",
                    "sec_id": "NIFTYBEES",
                    "symbol": "NIFTYBEES",
                    "sector": "Index Core ETF",
                    "industry": "Nifty 50 ETF",
                    "strategy_id": self.verified_config.regime_policy.policy_id,
                    "qty": int(ps["core_qty"]),
                    "avg_entry_price": round(cb / max(1, int(ps["core_qty"])), 4),
                    "last_price": round(float(ps["core_last_px"]), 4),
                    "market_value_inr": round(cv, 2),
                    "cost_basis_inr": round(cb, 2),
                    "unrealized_pnl_inr": round(cv - cb, 2),
                    "sl_price": 0.0,
                    "entry_date": str(ps["as_of_date"]),
                })
            if float(ps["liquid_units"]) > 1e-4:
                lv = float(ps["liquid_units"]) * float(ps["liquid_last_px"])
                lb = float(ps["liquid_cost_basis_inr"])
                all_open_holdings.append({
                    "position_id": "LIQUID-LIQUIDBEES",
                    "instrument_type": "LIQUID_ETF",
                    "sec_id": "LIQUIDBEES",
                    "symbol": "LIQUIDBEES",
                    "sector": "Liquid Cash ETF",
                    "industry": "Overnight TREPS ETF",
                    "strategy_id": self.verified_config.regime_policy.policy_id,
                    "qty": round(float(ps["liquid_units"]), 4),
                    "avg_entry_price": round(lb / max(1e-6, float(ps["liquid_units"])), 4),
                    "last_price": round(float(ps["liquid_last_px"]), 4),
                    "market_value_inr": round(lv, 2),
                    "cost_basis_inr": round(lb, 2),
                    "unrealized_pnl_inr": round(lv - lb - float(ps["cum_liquid_yield_inr"]), 2),
                    "sl_price": 0.0,
                    "entry_date": str(ps["as_of_date"]),
                })
            for p in open_positions:
                mv = float(p["qty"]) * float(p["last_price"])
                all_open_holdings.append({
                    **p,
                    "market_value_inr": round(mv, 2),
                    "unrealized_pnl_inr": round(mv - float(p["cost_basis_inr"]), 2),
                })

            recon = self.reconcile_portfolio()
            chain = self.verify_audit_chain()
            eng_state = self.get_engine_state()

            return {
                "release_version": RELEASE_VERSION_M3_1,
                "engine_state": eng_state,
                "safety_posture": self.safety_guard.security_posture(),
                "configuration_integrity": self.verified_config.to_integrity_dict(),
                "portfolio_state": ps,
                "open_holdings": all_open_holdings,
                "open_tactical_positions": open_positions,
                "closed_tactical_positions": closed_positions,
                "pending_next_session_orders": json.loads(ps["pending_orders_json"] or "[]"),
                "simulated_orders": orders,
                "transaction_cost_ledger": costs,
                "dividend_ledger": dividends,
                "settlement_ledger": settlements,
                "daily_nav_ledger": daily_navs,
                "prospective_daily_navs": prospective_navs,
                "prospective_orders": prospective_orders,
                "session_checkpoints": checkpoints,
                "reconciliation": recon,
                "audit_chain": chain,
                "prospective_validation": {
                    "prospective_start_date": self.prospective_start_date,
                    "completed_unseen_sessions": len(prospective_navs),
                    "minimum_required_sessions": MIN_PROSPECTIVE_SESSIONS,
                    "target_sessions": TARGET_PROSPECTIVE_SESSIONS,
                    "progress_to_min_pct": round(min(100.0, (len(prospective_navs) / MIN_PROSPECTIVE_SESSIONS) * 100.0), 2),
                    "progress_to_target_pct": round(min(100.0, (len(prospective_navs) / TARGET_PROSPECTIVE_SESSIONS) * 100.0), 2),
                    "validation_complete": len(prospective_navs) >= MIN_PROSPECTIVE_SESSIONS,
                    "status_label": (
                        "VALIDATION_MINIMUM_MET_REVIEW_READY"
                        if len(prospective_navs) >= MIN_PROSPECTIVE_SESSIONS
                        else (
                            "AWAITING_FIRST_PROSPECTIVE_SESSION"
                            if len(prospective_navs) == 0
                            else "IN_PROGRESS_INCOMPLETE_SAMPLE"
                        )
                    ),
                    "prospective_paper_trades": sum(1 for o in prospective_orders if o["execution_status"] in ("FILLED", "PARTIAL")),
                    "prospective_rejected_orders": sum(1 for o in prospective_orders if o["execution_status"] in ("REJECTED", "FAILED", "UNFILLED")),
                },
            }
        finally:
            conn.close()
