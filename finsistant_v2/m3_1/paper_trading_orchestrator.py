"""Moreshwar 3.1 (Prompt 5/5) — Unified Paper-Trading Orchestrator & Daily Audit Reporter.

Central integration layer connecting all validated Moreshwar 3.1 engines without duplicating
strategy math or modifying frozen M3.0 / Prompt 1–4 modules:

    Market Data
      ↓
    Data Validation & Point-in-Time Controls (`validate_market_data_snapshot`)
      ↓
    Market Regime Engine (`market_regime_engine.classify_simple_trend_baseline`)
      ↓
    Frozen Capital Allocation Policy (`G_SIMPLE_TREND_BASELINE` 60/30/10 vs 45/10/45)
      ↓
    Family E Strategy Eligibility (`STRAT_E_DEFENSIVE_YIELD_LOWVOL` only; Family A hard-blocked)
      ↓
    Trade Quality & Execution Intelligence (`EXP_E_COMBINED_EXEC_INTEL`, `trade_quality_models`)
      ↓
    Prospective Paper Execution Engine (`ProspectivePaperTradingEngine` + `TransactionCostIntelligence`)
      ↓
    Durable Portfolio & Settlement SQLite Ledger
      ↓
    Independent Risk Monitor (`PaperTradingRiskMonitor` + `regime_monitoring`)
      ↓
    Dashboard Payload & 12-File Daily Audit Trail (`reports/moreshwar_3_1/prospective_paper_trading/`)
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from finsistant_v2.m3_1.paper_trading_risk_monitor import PaperTradingRiskMonitor
from finsistant_v2.m3_1.prospective_paper_trading_engine import (
    DEFAULT_FROZEN_CONFIG_PATH,
    DEFAULT_PAPER_DB_PATH,
    DIAGNOSTIC_INDEX_DB_END_DATE,
    HISTORICAL_DB_END_DATE,
    INITIAL_PAPER_CAPITAL_INR,
    PaperOnlyExecutionGuard,
    PaperTradingSafetyError,
    ProspectivePaperTradingEngine,
    RELEASE_VERSION_M3_1,
    REPO_ROOT,
    load_and_verify_frozen_config,
)

DEFAULT_PROSPECTIVE_REPORTS_DIR = (
    REPO_ROOT / "reports" / "moreshwar_3_1" / "prospective_paper_trading"
)
PROMPT4_SUMMARY_PATH = (
    REPO_ROOT / "reports" / "moreshwar_3_1" / "strategy_intelligence" / "strategy_intelligence_summary.json"
)
PROMPT4_REGIME_CSV_PATH = (
    REPO_ROOT / "reports" / "moreshwar_3_1" / "strategy_intelligence" / "REGIME_CLASSIFICATION_HISTORY.csv"
)

ALLOWED_CONTROL_ACTIONS = frozenset({
    "start_monitoring",
    "pause_entries",
    "resume_after_validation",
    "stop_engine",
    "run_reconciliation",
    "export_reports",
})


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def validate_market_data_snapshot(
    *,
    trading_date: str,
    market_snapshot: dict[str, Any],
) -> dict[str, Any]:
    """Validate daily market data freshness, integrity, and known database boundaries (no fabrication!)."""
    dt = str(trading_date)
    issues: list[str] = []
    warnings: list[str] = []

    nb = market_snapshot.get("niftybees_bar") or {}
    lb = market_snapshot.get("liquidbees_bar") or {}
    eq_bars = market_snapshot.get("equity_bars") or {}

    def _check_ohlc(name: str, bar: dict[str, Any]) -> bool:
        if not bar or not bool(bar.get("valid", True)):
            issues.append(f"{name}_BAR_MISSING_OR_INVALID")
            return False
        o, h, l, c = (
            float(bar.get("open", 0.0) or 0.0),
            float(bar.get("high", 0.0) or 0.0),
            float(bar.get("low", 0.0) or 0.0),
            float(bar.get("close", 0.0) or 0.0),
        )
        if any(math_bad(v) or v <= 0.0 for v in (o, h, l, c)):
            issues.append(f"{name}_NON_POSITIVE_OR_NAN_PRICE")
            return False
        if h < l or h < max(o, c) * 0.999 or l > min(o, c) * 1.001:
            issues.append(f"{name}_INCONSISTENT_OHLC(o={o},h={h},l={l},c={c})")
            return False
        return True

    def math_bad(v: float) -> bool:
        return bool(np.isnan(v) or np.isinf(v))

    nb_ok = _check_ohlc("NIFTYBEES", nb)
    lb_ok = _check_ohlc("LIQUIDBEES", lb)

    # Check equity bars
    valid_eq_count = 0
    invalid_eq_count = 0
    for sid, b in eq_bars.items():
        if bool(b.get("valid", True)) and float(b.get("close", 0.0) or 0.0) > 0 and float(b.get("open", 0.0) or 0.0) > 0:
            valid_eq_count += 1
        else:
            invalid_eq_count += 1

    # Explicitly enforce known database boundary for `daily_index_bars` (ends 2025-10-10)
    diagnostic_index_available = bool(market_snapshot.get("diagnostic_index_available", dt <= DIAGNOSTIC_INDEX_DB_END_DATE))
    if dt > DIAGNOSTIC_INDEX_DB_END_DATE and not diagnostic_index_available:
        warnings.append(
            f"DIAGNOSTIC_INDEX_BARS_UNAVAILABLE_POST_{DIAGNOSTIC_INDEX_DB_END_DATE}: "
            "India VIX / GS 10Yr / sector indices are unavailable and NOT fabricated."
        )

    # Check for forbidden synthetic/fabricated flags
    if market_snapshot.get("contains_fabricated_data") is True:
        issues.append("FORBIDDEN_FABRICATED_DATA_FLAG")

    if issues:
        overall = "INVALID"
    elif warnings:
        overall = "PARTIAL_DIAGNOSTIC_ONLY"
    else:
        overall = "OK"

    return {
        "trading_date": dt,
        "checked_at_utc": _utc_now_iso(),
        "data_timestamp_utc": str(market_snapshot.get("data_timestamp_utc") or _utc_now_iso()),
        "overall_status": overall,
        "niftybees_valid": nb_ok,
        "liquidbees_valid": lb_ok,
        "valid_equity_bars": valid_eq_count,
        "invalid_equity_bars": invalid_eq_count,
        "diagnostic_index_bars_available": diagnostic_index_available,
        "known_boundaries": {
            "historical_daily_bars_coverage": f"2021-01-01 -> {HISTORICAL_DB_END_DATE} (1,454 sessions, 2,832 PIT equities + NIFTYBEES + LIQUIDBEES)",
            "diagnostic_index_bars_end_date": DIAGNOSTIC_INDEX_DB_END_DATE,
            "intraday_tick_minute_data": "UNAVAILABLE (NEXT_SESSION_OPEN daily bars only)",
            "pit_macro_timestamps": "UNAVAILABLE (excluded)",
            "corporate_earnings_financials": "UNAVAILABLE (excluded)",
        },
        "stale_price_days": int(market_snapshot.get("stale_price_days", 0) or 0),
        "issues": issues,
        "warnings": warnings,
        "summary": "; ".join(issues) if issues else ("; ".join(warnings) if warnings else "All required PIT daily bars validated."),
    }


def build_pit_snapshot_from_context(
    *,
    ctx: dict[str, Any],
    ti: int,
    max_candidates: int = 25,
) -> dict[str, Any]:
    """Construct a strictly causal single-session `market_snapshot` from a precomputed M3.1 context at index `ti`.

    Guarantees ZERO future leakage: uses ONLY slices `<= ti` (never `ti + 1`!).
    """
    panel = ctx["panel"]
    ws = ctx["ws_m31"]
    ef = ctx["exec_features"]
    etf_bars = ctx["etf_bars"]
    dates = [str(d) for d in panel["dates"]]
    dt = dates[ti]
    prev_ti = max(0, ti - 1)

    nb = etf_bars["NIFTYBEES"]
    lb = etf_bars["LIQUIDBEES"]

    nb_bar = {
        "open": float(nb["open"][ti]),
        "high": float(nb["high"][ti]),
        "low": float(nb["low"][ti]),
        "close": float(nb["close"][ti]),
        "volume": float(nb.get("volume", np.full(len(dates), 1_000_000.0))[ti]),
        "valid": bool((not np.isnan(nb["open"][ti])) and nb["open"][ti] > 0 and (not np.isnan(nb["close"][ti])) and nb["close"][ti] > 0),
    }
    lb_bar = {
        "open": float(lb["open"][ti]),
        "high": float(lb["high"][ti]),
        "low": float(lb["low"][ti]),
        "close": float(lb["close"][ti]),
        "volume": float(lb.get("volume", np.full(len(dates), 500_000.0))[ti]),
        "valid": bool((not np.isnan(lb["open"][ti])) and lb["open"][ti] > 0 and (not np.isnan(lb["close"][ti])) and lb["close"][ti] > 0),
    }

    # Strictly causal NIFTYBEES close history <= ti
    hist_start = max(0, ti - 250)
    nb_hist = [float(x) for x in nb["close"][hist_start : ti + 1] if not np.isnan(x) and x > 0]

    # Family E candidates strictly at session `ti` close
    raw_cands = list(ctx["precomputed_cands"].get("STRAT_E_DEFENSIVE_YIELD_LOWVOL", {}).get(ti, []))[:max_candidates]
    cand_cols = {int(c["col"]) for c in raw_cands}

    # Include all valid PIT equities needed for candidates & any active holdings
    sec_ids = panel["sec_ids"]
    syms = panel["current_symbols"]
    sectors = panel["sectors"]
    industries = panel["industries"]
    is_unk = panel.get("is_unknown_sector", np.zeros(len(sectors), dtype=bool))
    is_valid_eq = panel["is_valid_eq"]
    open_p, high_p, low_p, close_p, vol_p = (
        panel["adj_open"],
        panel["adj_high"],
        panel["adj_low"],
        panel["adj_close"],
        panel["volume"],
    )
    div_per_adj = ws["div_per_adj_share"]
    div_yield_ttm = ws["factors"]["DIV_YIELD_TTM_PIT"]
    sma_50d = ws["sma_50d"]

    eq_bars: dict[str, dict[str, Any]] = {}
    n_sec = len(sec_ids)
    # Populate equity bars for all valid rows on session `ti` (vectorized check)
    valid_cols = set(np.where(is_valid_eq[ti])[0].tolist()) | cand_cols
    for col in valid_cols:
        if col < 0 or col >= n_sec:
            continue
        sid = str(sec_ids[col])
        v_ok = bool(is_valid_eq[ti, col])
        c_val = float(close_p[ti, col]) if not np.isnan(close_p[ti, col]) else 0.0
        o_val = float(open_p[ti, col]) if not np.isnan(open_p[ti, col]) else c_val
        h_val = float(high_p[ti, col]) if not np.isnan(high_p[ti, col]) else max(o_val, c_val)
        l_val = float(low_p[ti, col]) if not np.isnan(low_p[ti, col]) else min(o_val, c_val)
        prev_c = float(close_p[prev_ti, col]) if not np.isnan(close_p[prev_ti, col]) else o_val
        eq_bars[sid] = {
            "sec_id": sid,
            "symbol": str(syms[col]),
            "sector": str(sectors[col]),
            "industry": str(industries[col]),
            "is_unknown_sector": bool(is_unk[col]),
            "open": o_val,
            "high": h_val,
            "low": l_val,
            "close": c_val,
            "prev_close": prev_c,
            "volume": float(vol_p[ti, col]) if not np.isnan(vol_p[ti, col]) else 0.0,
            "adv_20d_inr": float(ef["adv_20d_inr"][ti, col]) if not np.isnan(ef["adv_20d_inr"][ti, col]) else 80_000_000.0,
            "vol_20d": float(ef["vol_20d"][ti, col]) if not np.isnan(ef["vol_20d"][ti, col]) else 0.018,
            "hl_range_pct": float(ef["hl_range_pct"][ti, col]) if not np.isnan(ef["hl_range_pct"][ti, col]) else 2.0,
            "atr_14": float(ef["atr_14"][ti, col]) if not np.isnan(ef["atr_14"][ti, col]) else o_val * 0.025,
            "sma_50": float(sma_50d[ti, col]) if not np.isnan(sma_50d[ti, col]) else None,
            "low_20_prev": float(ef["low_20d_prev"][ti, col]) if not np.isnan(ef["low_20d_prev"][ti, col]) else None,
            "div_per_adj_share": float(div_per_adj[ti, col]) if not np.isnan(div_per_adj[ti, col]) else 0.0,
            "div_yield_ttm": float(div_yield_ttm[ti, col]) if not np.isnan(div_yield_ttm[ti, col]) else 0.0,
            "valid": v_ok and o_val > 0 and c_val > 0,
        }

    formatted_cands = [
        {
            "sec_id": str(sec_ids[int(c["col"])]),
            "symbol": str(syms[int(c["col"])]),
            "strategy_id": "STRAT_E_DEFENSIVE_YIELD_LOWVOL",
            "score": float(c.get("score", 0.7)),
            "stop_ratio": float(c.get("stop_ratio", 0.09)),
        }
        for c in raw_cands
    ]

    return {
        "trading_date": dt,
        "data_timestamp_utc": f"{dt}T16:15:00+05:30",
        "niftybees_bar": nb_bar,
        "liquidbees_bar": lb_bar,
        "niftybees_close_history": nb_hist,
        "equity_bars": eq_bars,
        "family_e_candidates": formatted_cands,
        "diagnostic_index_available": dt <= DIAGNOSTIC_INDEX_DB_END_DATE,
        "data_quality_state": "PARTIAL_DIAGNOSTIC_ONLY" if dt > DIAGNOSTIC_INDEX_DB_END_DATE else "OK",
    }


class PaperTradingOrchestrator:
    """Central Moreshwar 3.1 Paper-Trading Orchestrator, Reporter & Dashboard Integrator."""

    def __init__(
        self,
        *,
        db_path: Path = DEFAULT_PAPER_DB_PATH,
        config_path: Path = DEFAULT_FROZEN_CONFIG_PATH,
        reports_dir: Path = DEFAULT_PROSPECTIVE_REPORTS_DIR,
        initial_capital_inr: float = INITIAL_PAPER_CAPITAL_INR,
        prospective_start_date: str = HISTORICAL_DB_END_DATE,
        live_trading_enabled: bool = False,
        broker_order_placement_enabled: bool = False,
        automatic_capital_deployment_enabled: bool = False,
        broker_adapter: Any = None,
        extra_admitted_strategy_ids: tuple[str, ...] = (),
        regime_policy_override: Any = None,
        exec_cfg_override: dict[str, Any] | None = None,
    ) -> None:
        self.safety_guard = PaperOnlyExecutionGuard(extra_admitted_strategy_ids)
        self.safety_guard.assert_paper_only(
            live_trading_enabled=live_trading_enabled,
            broker_order_placement_enabled=broker_order_placement_enabled,
            automatic_capital_deployment_enabled=automatic_capital_deployment_enabled,
            broker_adapter=broker_adapter,
        )
        self.config_path = Path(config_path)
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.engine = ProspectivePaperTradingEngine(
            db_path=db_path,
            config_path=self.config_path,
            initial_capital_inr=initial_capital_inr,
            prospective_start_date=prospective_start_date,
            extra_admitted_strategy_ids=extra_admitted_strategy_ids,
            regime_policy_override=regime_policy_override,
            exec_cfg_override=exec_cfg_override,
        )
        self.risk_monitor = PaperTradingRiskMonitor()
        self._last_data_quality_report: dict[str, Any] = {
            "trading_date": self.engine.prospective_start_date,
            "checked_at_utc": _utc_now_iso(),
            "overall_status": "PARTIAL_DIAGNOSTIC_ONLY",
            "niftybees_valid": True,
            "liquidbees_valid": True,
            "diagnostic_index_bars_available": False,
            "known_boundaries": {
                "historical_daily_bars_coverage": f"2021-01-01 -> {HISTORICAL_DB_END_DATE} (1,454 sessions)",
                "diagnostic_index_bars_end_date": DIAGNOSTIC_INDEX_DB_END_DATE,
                "intraday_tick_minute_data": "UNAVAILABLE",
                "pit_macro_timestamps": "UNAVAILABLE",
                "corporate_earnings_financials": "UNAVAILABLE",
            },
            "issues": [],
            "warnings": [
                f"Awaiting first prospective NSE session after {self.engine.prospective_start_date}; "
                f"daily_index_bars ends {DIAGNOSTIC_INDEX_DB_END_DATE} (diagnostic-only)."
            ],
            "summary": "Initialized cleanly; awaiting unseen prospective daily market bars.",
        }
        dq_persisted = self.reports_dir / "data_quality_report.json"
        if dq_persisted.exists():
            try:
                loaded_dq = json.loads(dq_persisted.read_text(encoding="utf-8"))
                if isinstance(loaded_dq, dict) and loaded_dq.get("overall_status"):
                    self._last_data_quality_report = loaded_dq
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Daily Orchestration Cycle
    # ------------------------------------------------------------------
    def run_daily_cycle(
        self,
        *,
        trading_date: str,
        market_snapshot: dict[str, Any],
        observation_mode: str | None = None,
        regime_context: dict[str, Any] | None = None,
        export_reports: bool = True,
    ) -> dict[str, Any]:
        """Run one complete daily orchestration cycle: validate -> execute -> reconcile -> monitor -> report."""
        dt = str(trading_date)

        # 1. Verify frozen config hash
        cfg_ok = True
        try:
            fresh = load_and_verify_frozen_config(self.config_path)
            cfg_ok = fresh.canonical_sha256 == self.engine.verified_config.canonical_sha256
        except Exception as exc:
            cfg_ok = False
            self.engine.enter_safe_mode(f"CONFIG_INTEGRITY_FAILED: {exc}", trading_date=dt)
            snap = self.engine.get_full_ledger_snapshot()
            risk = self.risk_monitor.evaluate_risk_and_alerts(snap, config_verified=False, data_quality_report=self._last_data_quality_report)
            if export_reports:
                self.export_all_daily_reports(risk_report=risk, data_quality_report=self._last_data_quality_report)
            raise

        # 2. Validate PIT market data
        dq = validate_market_data_snapshot(trading_date=dt, market_snapshot=market_snapshot)
        self._last_data_quality_report = dq
        enriched_snapshot = {**market_snapshot, "data_quality_state": dq["overall_status"]}

        # 3. Execute daily paper-trading cycle (idempotent & atomic)
        exec_res = self.engine.process_trading_session(
            trading_date=dt,
            market_snapshot=enriched_snapshot,
            observation_mode=observation_mode,
        )

        # 4. Run independent portfolio risk monitor & fail-safe check
        snap = self.engine.get_full_ledger_snapshot()
        risk = self.risk_monitor.evaluate_risk_and_alerts(
            snap,
            config_verified=cfg_ok,
            data_quality_report=dq,
            regime_context=regime_context,
        )
        if risk["should_enter_safe_mode"] and snap["engine_state"]["engine_state"] != "SAFE_MODE":
            self.engine.enter_safe_mode("; ".join(risk["safe_mode_triggers"]), trading_date=dt)
            snap = self.engine.get_full_ledger_snapshot()

        # 5. Export all 12 daily audit reports + append-only trail
        report_paths: dict[str, str] = {}
        if export_reports:
            report_paths = self.export_all_daily_reports(
                risk_report=risk,
                data_quality_report=dq,
                latest_cycle_result=exec_res,
            )

        return {
            "trading_date": dt,
            "execution_result": exec_res,
            "data_quality": dq,
            "risk_assessment": risk,
            "reconciliation": snap["reconciliation"],
            "engine_state": snap["engine_state"],
            "report_paths": report_paths,
        }

    # ------------------------------------------------------------------
    # Section 8: Export All 12 Mandatory Daily Reports + Append-Only Trail
    # ------------------------------------------------------------------
    def export_all_daily_reports(
        self,
        *,
        risk_report: dict[str, Any] | None = None,
        data_quality_report: dict[str, Any] | None = None,
        latest_cycle_result: dict[str, Any] | None = None,
    ) -> dict[str, str]:
        """Generate all 12 Section 8 daily reporting artifacts under `self.reports_dir`."""
        snap = self.engine.get_full_ledger_snapshot()
        dq = data_quality_report or self._last_data_quality_report
        risk = risk_report or self.risk_monitor.evaluate_risk_and_alerts(snap, config_verified=True, data_quality_report=dq)

        ps = snap["portfolio_state"]
        recon = snap["reconciliation"]
        cfg_int = snap["configuration_integrity"]
        daily_navs = snap["daily_nav_ledger"]
        latest_nav = daily_navs[-1] if daily_navs else None
        as_of = str(ps["as_of_date"])

        # 1. paper_trading_daily_report.json
        daily_report_payload = {
            "schema_version": "m3_1_paper_daily_report_v1",
            "release_version": RELEASE_VERSION_M3_1,
            "generated_at_utc": _utc_now_iso(),
            "trading_date": as_of,
            "data_availability_timestamp": dq.get("data_timestamp_utc", _utc_now_iso()),
            "configuration_sha256": cfg_int["canonical_payload_sha256"],
            "configuration_raw_sha256": cfg_int["raw_file_sha256"],
            "strategy_identifier": f"{cfg_int['selected_policy_id']} + {cfg_int['execution_config_id']}",
            "admitted_tactical_strategies": cfg_int["admitted_tactical_strategies"],
            "suspended_strategies": cfg_int["suspended_strategies"],
            "safety_posture": snap["safety_posture"],
            "regime_determination": {
                "raw_regime": str(ps["raw_regime"]),
                "confirmed_regime": str(ps["confirmed_regime"]),
                "confirmation_streak_days": int(ps["regime_confirm_streak"]),
                "confirmation_required_days": int(self.engine.verified_config.regime_policy.allocation_confirm_days),
                "allow_new_family_e_entries": bool(ps["allow_new_entries"]),
            },
            "allocation": {
                "target_core_niftybees_pct": float(ps["target_core_pct"]),
                "target_tactical_family_e_pct": float(ps["target_tactical_pct"]),
                "target_liquid_liquidbees_pct": float(ps["target_liquid_pct"]),
                "target_gold_pct": 0.0,
                "actual_core_pct": risk["metrics"]["core_etf_exposure_pct"],
                "actual_tactical_pct": risk["metrics"]["tactical_equity_exposure_pct"],
                "actual_liquid_pct": risk["metrics"]["liquid_allocation_pct"],
                "actual_cash_pct": risk["metrics"]["cash_pct"],
            },
            "signals_generated": (latest_cycle_result or {}).get("signals", []),
            "simulated_orders_latest_session": [o for o in snap["simulated_orders"] if str(o["trading_date"]) == as_of],
            "pending_next_session_orders": snap["pending_next_session_orders"],
            "execution_summary": {
                "total_orders_all_sessions": len(snap["simulated_orders"]),
                "filled_orders": sum(1 for o in snap["simulated_orders"] if o["execution_status"] in ("FILLED", "PARTIAL")),
                "rejected_orders": sum(1 for o in snap["simulated_orders"] if o["execution_status"] == "REJECTED"),
                "failed_orders": sum(1 for o in snap["simulated_orders"] if o["execution_status"] in ("FAILED", "UNFILLED")),
            },
            "transaction_costs": {
                "cumulative_costs_inr": recon["cum_costs_inr"],
                "cost_pct_of_gross_traded": risk["metrics"]["cost_pct_of_gross_traded_value"],
            },
            "portfolio_nav": {
                "initial_capital_inr": risk["metrics"]["initial_capital_inr"],
                "current_nav_inr": risk["metrics"]["current_nav_inr"],
                "daily_pnl_inr": risk["metrics"]["daily_pnl_inr"],
                "daily_return_pct": risk["metrics"]["daily_return_pct"],
                "cumulative_return_pct": risk["metrics"]["cumulative_return_pct"],
                "benchmark_niftybees_nav": float(latest_nav["benchmark_niftybees_nav"]) if latest_nav else risk["metrics"]["initial_capital_inr"],
                "benchmark_45_55_nav": float(latest_nav["benchmark_45_55_nav"]) if latest_nav else risk["metrics"]["initial_capital_inr"],
            },
            "risk_metrics": risk["metrics"],
            "data_quality_state": dq["overall_status"],
            "reconciliation_result": recon,
            "engine_status": snap["engine_state"]["engine_state"],
            "safe_mode_reason": snap["engine_state"]["safe_mode_reason"],
            "audit_chain": snap["audit_chain"],
            "prospective_validation": snap["prospective_validation"],
        }
        p1 = self.reports_dir / "paper_trading_daily_report.json"
        _write_json(p1, daily_report_payload)

        # 2. paper_trading_daily_report.csv
        p2 = self.reports_dir / "paper_trading_daily_report.csv"
        daily_csv_fields = [
            "trading_date", "observation_mode", "nav_inr", "settled_cash_inr", "unsettled_cash_inr",
            "core_value_inr", "tactical_value_inr", "liquid_value_inr", "realized_pnl_inr", "unrealized_pnl_inr",
            "cum_dividends_inr", "cum_liquid_yield_inr", "cum_costs_inr", "daily_pnl_inr",
            "daily_return_pct", "cumulative_return_pct", "drawdown_pct", "raw_regime", "confirmed_regime",
            "regime_confirm_streak", "target_core_pct", "target_tactical_pct", "target_liquid_pct",
            "allow_new_entries", "open_tactical_positions", "benchmark_niftybees_nav", "benchmark_45_55_nav",
            "reconciliation_diff_inr", "data_quality_state", "engine_state",
        ]
        _write_csv(p2, daily_csv_fields, daily_navs)

        # 3. portfolio_snapshot.json
        p3 = self.reports_dir / "portfolio_snapshot.json"
        _write_json(
            p3,
            {
                "generated_at_utc": _utc_now_iso(),
                "as_of_date": as_of,
                "engine_state": snap["engine_state"],
                "safety_posture": snap["safety_posture"],
                "reconciliation": recon,
                "risk_metrics": risk["metrics"],
                "open_holdings": snap["open_holdings"],
                "pending_next_session_orders": snap["pending_next_session_orders"],
            },
        )

        # 4. open_positions.csv
        p4 = self.reports_dir / "open_positions.csv"
        _write_csv(
            p4,
            [
                "position_id", "instrument_type", "sec_id", "symbol", "sector", "industry", "strategy_id",
                "entry_date", "qty", "avg_entry_price", "last_price", "cost_basis_inr", "market_value_inr",
                "unrealized_pnl_inr", "sl_price",
            ],
            snap["open_holdings"],
        )

        # 5. simulated_orders.csv
        p5 = self.reports_dir / "simulated_orders.csv"
        _write_csv(
            p5,
            [
                "order_id", "trading_date", "signal_timestamp", "execution_timestamp", "strategy_id",
                "instrument_class", "sec_id", "symbol", "side", "requested_qty", "filled_qty",
                "expected_price", "actual_exec_price", "gross_value_inr", "slippage_bps", "slippage_inr",
                "brokerage_inr", "stt_inr", "exchange_fee_inr", "sebi_fee_inr", "gst_inr", "stamp_duty_inr",
                "dp_charge_inr", "total_cost_inr", "net_value_inr", "execution_status", "rejection_reason",
                "ev_expected_net_bps", "ev_hurdle_bps", "observation_mode",
            ],
            snap["simulated_orders"],
        )

        # 6. transaction_cost_ledger.csv
        p6 = self.reports_dir / "transaction_cost_ledger.csv"
        _write_csv(
            p6,
            [
                "cost_id", "order_id", "trading_date", "instrument_class", "symbol", "side",
                "gross_value_inr", "brokerage_inr", "stt_inr", "exchange_fee_inr", "sebi_fee_inr",
                "gst_inr", "stamp_duty_inr", "dp_charge_inr", "slippage_inr",
                "regulatory_total_inr", "total_cost_inr", "observation_mode",
            ],
            snap["transaction_cost_ledger"],
        )

        # 7. dividend_ledger.csv
        p7 = self.reports_dir / "dividend_ledger.csv"
        _write_csv(
            p7,
            [
                "dividend_id", "trading_date", "sec_id", "symbol", "event_type",
                "qty_eligible", "rate_per_unit_inr", "credited_cash_inr",
                "stop_adjusted_from", "stop_adjusted_to", "observation_mode",
            ],
            snap["dividend_ledger"],
        )

        # 8. settlement_ledger.csv
        p8 = self.reports_dir / "settlement_ledger.csv"
        _write_csv(
            p8,
            [
                "settlement_id", "order_id", "trade_date", "settlement_date", "due_session_counter",
                "symbol", "amount_inr", "status", "settled_on_date", "observation_mode",
            ],
            snap["settlement_ledger"],
        )

        # 9. risk_alerts.json
        p9 = self.reports_dir / "risk_alerts.json"
        _write_json(
            p9,
            {
                "generated_at_utc": _utc_now_iso(),
                "as_of_date": as_of,
                "overall_health": risk["overall_health"],
                "should_enter_safe_mode": risk["should_enter_safe_mode"],
                "safe_mode_triggers": risk["safe_mode_triggers"],
                "alert_counts": risk["alert_counts"],
                "alerts": risk["alerts"],
                "prompt4_advisory_health": risk["prompt4_advisory_health"],
            },
        )

        # 10. data_quality_report.json
        p10 = self.reports_dir / "data_quality_report.json"
        _write_json(p10, dq)

        # 11. configuration_integrity.json
        p11 = self.reports_dir / "configuration_integrity.json"
        _write_json(p11, cfg_int)

        # 12. prospective_validation_progress.json
        p12 = self.reports_dir / "prospective_validation_progress.json"
        pv = {
            **snap["prospective_validation"],
            "generated_at_utc": _utc_now_iso(),
            "configuration_verified": cfg_int["verified"],
            "configuration_sha256_prefix": cfg_int["canonical_prefix_16"],
            "data_quality_status": dq["overall_status"],
            "live_trading_enabled": False,
            "benchmark_relative_return_pp": (
                round(
                    ((float(snap["prospective_daily_navs"][-1]["nav_inr"]) / float(snap["engine_state"]["initial_capital_inr"]) - 1.0)
                     - (float(snap["prospective_daily_navs"][-1]["benchmark_niftybees_nav"]) / float(snap["engine_state"]["initial_capital_inr"]) - 1.0)) * 100.0,
                    2,
                )
                if snap["prospective_daily_navs"]
                else None
            ),
            "prospective_max_drawdown_pct": (
                round(max(float(r["drawdown_pct"]) for r in snap["prospective_daily_navs"]), 2)
                if snap["prospective_daily_navs"]
                else None
            ),
            "prospective_rolling_sharpe": risk["metrics"]["rolling_sharpe"] if snap["prospective_daily_navs"] else None,
            "sample_adequacy_note": (
                "Prospective validation has 0 completed unseen sessions. "
                "Historical backtests (2021-07 -> 2026-11-19) are strictly excluded from prospective counts."
                if len(snap["prospective_daily_navs"]) == 0
                else (
                    f"{len(snap['prospective_daily_navs'])} unseen prospective sessions completed "
                    f"(minimum 60, target 120)."
                )
            ),
        }
        _write_json(p12, pv)

        # Append-only hash-chained JSONL audit log on disk
        audit_jsonl = self.reports_dir / "daily_audit_trail.jsonl"
        prev_digest = "0" * 64
        if audit_jsonl.exists():
            lines = [ln for ln in audit_jsonl.read_text(encoding="utf-8").splitlines() if ln.strip()]
            if lines:
                try:
                    prev_digest = str(json.loads(lines[-1]).get("entry_chain_sha256", prev_digest))
                except Exception:
                    pass
        record_core = {
            "recorded_at_utc": _utc_now_iso(),
            "trading_date": as_of,
            "config_sha256": cfg_int["canonical_payload_sha256"],
            "engine_state": snap["engine_state"]["engine_state"],
            "nav_inr": risk["metrics"]["current_nav_inr"],
            "reconciliation_diff_inr": recon["reconciliation_diff_inr"],
            "prospective_sessions": snap["prospective_validation"]["completed_unseen_sessions"],
            "prev_chain_sha256": prev_digest,
        }
        entry_sha = hashlib.sha256(json.dumps(record_core, sort_keys=True).encode("utf-8")).hexdigest()
        record_core["entry_chain_sha256"] = entry_sha
        with audit_jsonl.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record_core, sort_keys=True) + "\n")

        return {
            "paper_trading_daily_report_json": str(p1),
            "paper_trading_daily_report_csv": str(p2),
            "portfolio_snapshot_json": str(p3),
            "open_positions_csv": str(p4),
            "simulated_orders_csv": str(p5),
            "transaction_cost_ledger_csv": str(p6),
            "dividend_ledger_csv": str(p7),
            "settlement_ledger_csv": str(p8),
            "risk_alerts_json": str(p9),
            "data_quality_report_json": str(p10),
            "configuration_integrity_json": str(p11),
            "prospective_validation_progress_json": str(p12),
            "daily_audit_trail_jsonl": str(audit_jsonl),
        }

    # ------------------------------------------------------------------
    # Paper-Only Operational Controls (Section 6.5)
    # ------------------------------------------------------------------
    def execute_paper_control(self, action: str, *, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        """Execute strictly paper-only operational controls; fail closed on any live-execution attempt."""
        body = payload or {}
        # Multi-layer guard against live-trading override flags in request payload
        self.safety_guard.assert_paper_only(
            live_trading_enabled=bool(body.get("live_trading_enabled", False)),
            broker_order_placement_enabled=bool(body.get("broker_order_placement_enabled", False)),
            automatic_capital_deployment_enabled=bool(body.get("automatic_capital_deployment_enabled", False)),
            broker_adapter=body.get("broker_adapter"),
            strategy_ids=body.get("strategy_ids"),
        )
        act = str(action).lower().strip()
        if act in ("enable_live", "activate_live", "place_live_order", "deploy_capital", "activate_family_a", "override_config"):
            raise PaperTradingSafetyError(f"HARD_PAPER_ONLY_GUARD: Prohibited control action '{action}'.")
        if act not in ALLOWED_CONTROL_ACTIONS:
            raise ValueError(f"Unsupported paper control action: '{action}'. Allowed: {sorted(ALLOWED_CONTROL_ACTIONS)}")

        if act == "start_monitoring":
            st = self.engine.set_operational_state("RUNNING", reason="operator_start_paper_monitoring")
            return {"action": act, "status": "OK", "engine_state": st, "paper_only": True}
        if act == "pause_entries":
            st = self.engine.set_operational_state("PAUSED", reason="operator_pause_new_entries")
            return {"action": act, "status": "OK", "engine_state": st, "paper_only": True}
        if act == "stop_engine":
            st = self.engine.set_operational_state("STOPPED", reason="operator_stop_paper_engine")
            return {"action": act, "status": "OK", "engine_state": st, "paper_only": True}
        if act == "resume_after_validation":
            cur = self.engine.get_engine_state()
            if cur["engine_state"] == "SAFE_MODE":
                rec = self.engine.recover_from_safe_mode(operator_note=str(body.get("note", "dashboard_validated_resume")))
                return {"action": act, "status": "RECOVERED_AND_RUNNING", **rec, "paper_only": True}
            recon = self.engine.reconcile_portfolio()
            if not recon["passed"]:
                self.engine.enter_safe_mode("RECONCILIATION_FAILED_ON_RESUME")
                raise PaperTradingSafetyError("Cannot resume: portfolio reconciliation failed; entered SAFE_MODE.")
            st = self.engine.set_operational_state("RUNNING", reason="operator_validated_resume")
            return {"action": act, "status": "OK", "engine_state": st, "reconciliation": recon, "paper_only": True}
        if act == "run_reconciliation":
            recon = self.engine.reconcile_portfolio()
            if not recon["passed"]:
                self.engine.enter_safe_mode(f"MANUAL_RECON_FAILED: {recon['failure_reasons']}")
            return {"action": act, "status": "PASS" if recon["passed"] else "FAIL_SAFE_MODE", "reconciliation": recon, "paper_only": True}
        if act == "export_reports":
            paths = self.export_all_daily_reports()
            return {"action": act, "status": "EXPORTED", "report_paths": paths, "paper_only": True}
        raise ValueError(f"Unhandled action: {act}")

    # ------------------------------------------------------------------
    # Unified Section 6 Dashboard Payload (Strict Historical vs Prospective Separation)
    # ------------------------------------------------------------------
    def _load_historical_research_baseline(self) -> dict[str, Any]:
        """Load frozen Prompt 1–4 historical research metrics strictly labeled BACKTESTED_RESEARCH."""
        baseline: dict[str, Any] = {
            "data_classification": "HISTORICAL_BACKTEST_RESEARCH_ONLY",
            "counts_toward_prospective_validation": False,
            "evaluation_window": f"2021-07-01 -> {HISTORICAL_DB_END_DATE}",
            "outer_window": f"2024-07-01 -> {HISTORICAL_DB_END_DATE}",
            "frozen_policy_id": "G_SIMPLE_TREND_BASELINE",
            "execution_profile_id": "EXP_E_COMBINED_EXEC_INTEL",
            "summary": {},
            "regime_history_tail": [],
        }
        if PROMPT4_SUMMARY_PATH.exists():
            try:
                baseline["summary"] = json.loads(PROMPT4_SUMMARY_PATH.read_text(encoding="utf-8"))
            except Exception:
                pass
        if PROMPT4_REGIME_CSV_PATH.exists():
            try:
                with PROMPT4_REGIME_CSV_PATH.open("r", encoding="utf-8") as fh:
                    rows = list(csv.DictReader(fh))
                # Downsample to ~120 evenly spaced historical points for crisp chart rendering
                step = max(1, len(rows) // 120)
                tail = rows[::step]
                if rows and tail[-1] != rows[-1]:
                    tail.append(rows[-1])
                baseline["regime_history_tail"] = [
                    {
                        "date": r.get("date", ""),
                        "niftybees_close": float(r.get("nb_close") or 0.0),
                        "simple_trend_regime": r.get("simple_trend_regime", "BULL_TREND"),
                        "rule_regime": r.get("rule_regime", ""),
                    }
                    for r in tail
                ]
            except Exception:
                pass
        return baseline

    def _build_prospective_benchmark_monitoring(self, snap: dict[str, Any]) -> dict[str, Any]:
        """Compute Section 10 aligned executable benchmark telemetry (100% NIFTYBEES & 45/55 NIFTYBEES/LIQUIDBEES)."""
        init_cap = float(snap["engine_state"]["initial_capital_inr"])
        prosp_navs = snap.get("prospective_daily_navs", [])
        n = len(prosp_navs)
        port_nav = float(prosp_navs[-1]["nav_inr"]) if n else init_cap
        port_ret = ((port_nav / max(1e-6, init_cap)) - 1.0) * 100.0

        def _bench_stats(nav_key: str, bench_id: str, bench_name: str, weights_desc: str, est_entry_cost_inr: float) -> dict[str, Any]:
            if n == 0:
                return {
                    "benchmark_id": bench_id,
                    "benchmark_name": bench_name,
                    "weights": weights_desc,
                    "initial_capital_inr": round(init_cap, 2),
                    "prospective_start_date": snap["prospective_validation"]["prospective_start_date"],
                    "current_nav_inr": round(init_cap, 2),
                    "absolute_return_pct": 0.0,
                    "portfolio_relative_return_pp": 0.0,
                    "cagr_pct": None,
                    "cagr_status": "INSUFFICIENT_SAMPLE (0/60 sessions)",
                    "max_drawdown_pct": 0.0,
                    "annualized_volatility_pct": None,
                    "sharpe_ratio": None,
                    "ratios_status": "INSUFFICIENT_SAMPLE (0/20 sessions)",
                    "estimated_entry_transaction_cost_inr": round(est_entry_cost_inr, 2),
                    "status": "INITIALIZED_AWAITING_FIRST_PROSPECTIVE_SESSION",
                }
            series = [float(r[nav_key]) for r in prosp_navs]
            cur = series[-1]
            abs_ret = ((cur / max(1e-6, init_cap)) - 1.0) * 100.0
            peak = init_cap
            max_dd = 0.0
            for v in series:
                peak = max(peak, v)
                max_dd = max(max_dd, ((peak - v) / max(1e-6, peak)) * 100.0)
            cagr = round(((cur / init_cap) ** (252.0 / n) - 1.0) * 100.0, 2) if n >= 60 else None
            vol_pct = None
            sharpe = None
            if n >= 20:
                arr = np.array([init_cap] + series, dtype=np.float64)
                drets = np.diff(arr) / np.maximum(1e-6, arr[:-1])
                sd = float(np.std(drets, ddof=1))
                vol_pct = round(sd * (252.0 ** 0.5) * 100.0, 2)
                sharpe = round(((float(np.mean(drets)) - 0.065 / 252.0) / max(1e-9, sd)) * (252.0 ** 0.5), 2) if sd > 1e-9 else 0.0
            return {
                "benchmark_id": bench_id,
                "benchmark_name": bench_name,
                "weights": weights_desc,
                "initial_capital_inr": round(init_cap, 2),
                "prospective_start_date": snap["prospective_validation"]["prospective_start_date"],
                "current_nav_inr": round(cur, 2),
                "absolute_return_pct": round(abs_ret, 4),
                "portfolio_relative_return_pp": round(port_ret - abs_ret, 4),
                "cagr_pct": cagr,
                "cagr_status": "SUFFICIENT_SAMPLE" if n >= 60 else f"INSUFFICIENT_SAMPLE ({n}/60 sessions)",
                "max_drawdown_pct": round(max_dd, 2),
                "annualized_volatility_pct": vol_pct,
                "sharpe_ratio": sharpe,
                "ratios_status": "SUFFICIENT_SAMPLE" if n >= 20 else f"INSUFFICIENT_SAMPLE ({n}/20 sessions)",
                "estimated_entry_transaction_cost_inr": round(est_entry_cost_inr, 2),
                "status": "ACTIVE_TRACKING",
            }

        # Exact one-time deployment friction under centralized TransactionCostIntelligence
        dt_ref = snap["prospective_validation"]["prospective_start_date"]
        nb_entry_c = float(
            self.engine.cost_engine.calculate_leg_cost(
                instrument_class="EQUITY_ETF", side="BUY", raw_notional=init_cap, trade_date=dt_ref
            )["total_cost"]
        )
        b45_entry_c = float(
            self.engine.cost_engine.calculate_leg_cost(
                instrument_class="EQUITY_ETF", side="BUY", raw_notional=init_cap * 0.45, trade_date=dt_ref
            )["total_cost"]
        ) + float(
            self.engine.cost_engine.calculate_leg_cost(
                instrument_class="LIQUID_ETF", side="BUY", raw_notional=init_cap * 0.55, trade_date=dt_ref
            )["total_cost"]
        )

        return {
            "methodology_frozen": True,
            "prospective_sessions_observed": n,
            "benchmarks": [
                _bench_stats("benchmark_niftybees_nav", "BENCH_100_NIFTYBEES", "100% NIFTYBEES (Executable)", "100% NIFTYBEES", nb_entry_c),
                _bench_stats("benchmark_45_55_nav", "BENCH_45_55_ETF", "45% NIFTYBEES / 55% LIQUIDBEES (Executable)", "45% NIFTYBEES / 55% LIQUIDBEES", b45_entry_c),
            ],
        }

    def build_dashboard_payload(self, *, scheduler_status: dict[str, Any] | None = None) -> dict[str, Any]:
        """Build complete Section 6 Moreshwar 3.1 monitoring payload with zero historical/prospective mixing."""
        snap = self.engine.get_full_ledger_snapshot()
        dq = self._last_data_quality_report
        risk = self.risk_monitor.evaluate_risk_and_alerts(snap, config_verified=True, data_quality_report=dq)
        ps = snap["portfolio_state"]
        as_of = str(ps["as_of_date"])

        # Compute monthly returns from prospective NAVs only (never blended with historical backtest!)
        prosp_navs = snap["prospective_daily_navs"]
        monthly_returns: list[dict[str, Any]] = []
        if prosp_navs:
            by_month: dict[str, list[dict[str, Any]]] = {}
            for r in prosp_navs:
                ym = str(r["trading_date"])[:7]
                by_month.setdefault(ym, []).append(r)
            prev_end_nav = float(snap["engine_state"]["initial_capital_inr"])
            for ym, mrows in sorted(by_month.items()):
                end_nav = float(mrows[-1]["nav_inr"])
                m_ret = ((end_nav / max(1e-6, prev_end_nav)) - 1.0) * 100.0
                monthly_returns.append({"month": ym, "return_pct": round(m_ret, 2), "sessions": len(mrows)})
                prev_end_nav = end_nav

        return {
            "release_version": RELEASE_VERSION_M3_1,
            "generated_at_utc": _utc_now_iso(),
            "banner": "PAPER TRADING ONLY — LIVE TRADING DISABLED",
            "safety_banner_lines": [
                "MORESHWAR 3.1 — PAPER TRADING ONLY.",
                "LIVE TRADING: DISABLED.",
                "BROKER ORDER PLACEMENT: DISABLED.",
                "AUTOMATIC CAPITAL DEPLOYMENT: DISABLED.",
            ],
            "safety_posture": snap["safety_posture"],
            "configuration_integrity": snap["configuration_integrity"],
            "engine_state": snap["engine_state"],
            "scheduler_status": scheduler_status or {"status": "IDLE", "last_checked_date": as_of},
            # Strictly separated historical baseline (Prompt 1–4)
            "historical_research_baseline": self._load_historical_research_baseline(),
            # Strictly separated prospective paper-trading state
            "prospective_paper_trading": {
                "data_classification": "PROSPECTIVE_PAPER_TRADING_ONLY",
                "boundary_date": snap["prospective_validation"]["prospective_start_date"],
                "portfolio_overview": {
                    "initial_capital_inr": risk["metrics"]["initial_capital_inr"],
                    "current_nav_inr": risk["metrics"]["current_nav_inr"],
                    "total_return_pct": risk["metrics"]["cumulative_return_pct"],
                    "today_pnl_inr": risk["metrics"]["daily_pnl_inr"],
                    "available_cash_inr": snap["reconciliation"]["settled_cash_inr"],
                    "pending_settlements_inr": snap["reconciliation"]["unsettled_cash_inr"],
                    "invested_capital_inr": round(
                        snap["reconciliation"]["core_value_inr"]
                        + snap["reconciliation"]["tactical_value_inr"]
                        + snap["reconciliation"]["liquid_value_inr"],
                        2,
                    ),
                    "open_positions_count": len(snap["open_holdings"]),
                    "open_tactical_positions_count": len(snap["open_tactical_positions"]),
                },
                "strategy_status": {
                    "active_policy": self.engine.verified_config.regime_policy.policy_id,
                    "active_policy_name": self.engine.verified_config.regime_policy.policy_name,
                    "execution_profile": self.engine.verified_config.exec_cfg.config_id,
                    "current_raw_regime": str(ps["raw_regime"]),
                    "confirmed_regime": str(ps["confirmed_regime"]),
                    "confirmation_streak_days": int(ps["regime_confirm_streak"]),
                    "confirmation_required_days": int(self.engine.verified_config.regime_policy.allocation_confirm_days),
                    "target_allocation_pct": {
                        "core_niftybees": float(ps["target_core_pct"]),
                        "tactical_family_e": float(ps["target_tactical_pct"]),
                        "liquid_liquidbees": float(ps["target_liquid_pct"]),
                        "gold": 0.0,
                    },
                    "family_e_new_entries_eligible": bool(ps["allow_new_entries"]) and snap["engine_state"]["engine_state"] not in ("SAFE_MODE", "PAUSED", "STOPPED"),
                    "family_a_status": "SUSPENDED_HARD_BLOCKED",
                    "config_hash_verified": snap["configuration_integrity"]["verified"],
                    "config_sha256_prefix": snap["configuration_integrity"]["canonical_prefix_16"],
                },
                "performance": {
                    "equity_curve": [
                        {
                            "date": r["trading_date"],
                            "portfolio_nav": r["nav_inr"],
                            "niftybees_nav": r["benchmark_niftybees_nav"],
                            "blend_45_55_nav": r["benchmark_45_55_nav"],
                            "drawdown_pct": r["drawdown_pct"],
                            "daily_return_pct": r["daily_return_pct"],
                        }
                        for r in prosp_navs
                    ],
                    "monthly_returns": monthly_returns,
                    "has_prospective_sessions": len(prosp_navs) > 0,
                },
                "benchmark_monitoring": self._build_prospective_benchmark_monitoring(snap),
                "execution_monitoring": {
                    "open_holdings": snap["open_holdings"],
                    "pending_next_session_orders": snap["pending_next_session_orders"],
                    "recent_orders": snap["simulated_orders"][-25:],
                    "settlement_ledger": snap["settlement_ledger"][-20:],
                    "total_transaction_costs_inr": snap["reconciliation"]["cum_costs_inr"],
                },
                "risk_dashboard": {
                    "overall_health": risk["overall_health"],
                    "metrics": risk["metrics"],
                    "alerts": risk["alerts"],
                    "alert_counts": risk["alert_counts"],
                    "data_quality": dq,
                    "reconciliation": snap["reconciliation"],
                    "audit_chain": snap["audit_chain"],
                },
                "validation_tracker": snap["prospective_validation"],
            },
            "operational_controls": {
                "paper_only": True,
                "live_order_button_present": False,
                "allowed_actions": sorted(ALLOWED_CONTROL_ACTIONS),
            },
        }
