"""Moreshwar 3.1 (Prompt 5/5) — Independent Paper-Trading Risk Management & Fail-Safe Monitor.

Independently evaluates the durable paper portfolio ledger without modifying any frozen strategy
parameter or execution rule.

Implements all Section 5 requirements:
- 5.1 Portfolio-level risk metrics with explicit sample-size sufficiency flags (never reports
  misleading annualized CAGR/Sharpe/Sortino on tiny sample sizes!).
- 5.2 Structured operational & portfolio risk alert catalog (with Alert ID, Severity, Timestamp,
  Affected Component, Observed Value, Threshold, Description, Recommended Action, Resolution Status).
- 5.3 Automatic fail-safe detection (`SAFE_MODE` trigger evaluation) on config integrity failure,
  invalid data, reconciliation mismatch, inconsistent cash/settlements, or suspended strategy activation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import math
from typing import Any

import numpy as np

from finsistant_v2.m3_1.prospective_paper_trading_engine import (
    RECONCILIATION_TOLERANCE_INR,
    SUSPENDED_STRATEGIES_HARD,
)
from finsistant_v2.m3_1.regime_monitoring import (
    MonitoringThresholds,
    evaluate_regime_and_strategy_health,
)

MIN_SESSIONS_FOR_ROLLING_RATIOS = 20
MIN_SESSIONS_FOR_CAGR = 60
RISK_FREE_ANNUAL_PAPER = 0.065


@dataclass(frozen=True)
class PaperRiskAlertThresholds:
    """Configurable risk & operational alerting thresholds for independent monitoring."""

    drawdown_warning_pct: float = 12.0
    drawdown_critical_pct: float = 21.0
    allocation_deviation_warning_pp: float = 12.0
    single_stock_concentration_pct: float = 16.0
    sector_concentration_pct: float = 28.0
    abnormal_cost_pct_gross_traded: float = 0.45  # 45 bps one-way average cost warning
    max_failed_orders_session: int = 2
    stale_price_days_warning: int = 3
    reconciliation_tolerance_inr: float = RECONCILIATION_TOLERANCE_INR


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _make_alert(
    *,
    code: str,
    severity: str,  # "INFO" | "WARNING" | "CRITICAL"
    component: str,
    observed_value: Any,
    threshold: Any,
    description: str,
    recommended_action: str,
    as_of_date: str,
    resolution_status: str = "OPEN",
) -> dict[str, Any]:
    raw_key = f"{as_of_date}:{code}:{component}:{observed_value}"
    aid = f"ALT-{hashlib.sha256(raw_key.encode('utf-8')).hexdigest()[:10].upper()}"
    return {
        "alert_id": aid,
        "code": code,
        "severity": severity,
        "timestamp": _utc_now_iso(),
        "as_of_date": as_of_date,
        "affected_component": component,
        "observed_value": observed_value,
        "threshold": threshold,
        "description": description,
        "recommended_operational_action": recommended_action,
        "resolution_status": resolution_status,
    }


class PaperTradingRiskMonitor:
    """Independent portfolio risk calculator, alert generator, and fail-safe evaluator."""

    def __init__(
        self,
        *,
        risk_thresholds: PaperRiskAlertThresholds | None = None,
        regime_thresholds: MonitoringThresholds | None = None,
    ) -> None:
        self.risk_thresholds = risk_thresholds or PaperRiskAlertThresholds()
        self.regime_thresholds = regime_thresholds or MonitoringThresholds()

    def compute_portfolio_risk_metrics(
        self,
        snapshot: dict[str, Any],
        *,
        prospective_only: bool = True,
    ) -> dict[str, Any]:
        """Compute all Section 5.1 risk & exposure metrics with strict sample-size labelling."""
        ps = snapshot["portfolio_state"]
        recon = snapshot["reconciliation"]
        navs_all = snapshot.get("prospective_daily_navs", []) if prospective_only else snapshot.get("daily_nav_ledger", [])
        if not navs_all:
            navs_all = snapshot.get("daily_nav_ledger", [])

        init_cap = float(snapshot["engine_state"]["initial_capital_inr"])
        cur_nav = float(recon["nav_from_assets_inr"])
        settled_cash = float(recon["settled_cash_inr"])
        unsettled_cash = float(recon["unsettled_cash_inr"])
        core_val = float(recon["core_value_inr"])
        tact_val = float(recon["tactical_value_inr"])
        liquid_val = float(recon["liquid_value_inr"])
        invested_val = core_val + tact_val + liquid_val

        realized_pnl = float(recon["realized_pnl_inr"])
        unrealized_pnl = float(recon["unrealized_pnl_inr"])
        cum_dividends = float(recon["cum_dividends_inr"])
        cum_liq_yield = float(recon["cum_liquid_yield_inr"])
        cum_costs = float(recon["cum_costs_inr"])
        cum_turnover = float(ps.get("cum_turnover_inr", 0.0))

        cum_pnl = cur_nav - init_cap
        cum_return_pct = (cum_pnl / max(1e-6, init_cap)) * 100.0
        daily_pnl = float(navs_all[-1]["daily_pnl_inr"]) if navs_all else 0.0
        daily_ret_pct = float(navs_all[-1]["daily_return_pct"]) if navs_all else 0.0

        n_sessions = len(navs_all)
        rets = np.array([float(r["daily_return_pct"]) / 100.0 for r in navs_all], dtype=np.float64) if n_sessions else np.array([], dtype=np.float64)
        dds = [float(r["drawdown_pct"]) for r in navs_all]
        current_dd_pct = float(dds[-1]) if dds else 0.0
        max_dd_pct = float(max(dds)) if dds else 0.0

        # Sample-size gated annualized metrics
        if n_sessions >= MIN_SESSIONS_FOR_CAGR and init_cap > 0 and cur_nav > 0:
            years = n_sessions / 252.0
            cagr_pct: float | None = round(((cur_nav / init_cap) ** (1.0 / max(1e-6, years)) - 1.0) * 100.0, 2)
            cagr_status = "SUFFICIENT_SAMPLE"
        else:
            cagr_pct = None
            cagr_status = f"INSUFFICIENT_SAMPLE ({n_sessions}/{MIN_SESSIONS_FOR_CAGR} sessions)"

        if n_sessions >= MIN_SESSIONS_FOR_ROLLING_RATIOS:
            w = rets[-min(60, n_sessions) :]
            std_d = float(np.std(w, ddof=1))
            rolling_vol_pct: float | None = round(std_d * math.sqrt(252.0) * 100.0, 2)
            mean_d = float(np.mean(w))
            rf_d = RISK_FREE_ANNUAL_PAPER / 252.0
            rolling_sharpe: float | None = round(((mean_d - rf_d) / max(1e-9, std_d)) * math.sqrt(252.0), 2) if std_d > 1e-9 else 0.0
            downside = w[w < rf_d] - rf_d
            down_std = float(np.sqrt(np.mean(downside ** 2))) if len(downside) >= 3 else 0.0
            rolling_sortino: float | None = (
                round(((mean_d - rf_d) / max(1e-9, down_std)) * math.sqrt(252.0), 2) if down_std > 1e-9 else None
            )
            ratios_status = "SUFFICIENT_SAMPLE"
        else:
            rolling_vol_pct = None
            rolling_sharpe = None
            rolling_sortino = None
            ratios_status = f"INSUFFICIENT_SAMPLE ({n_sessions}/{MIN_SESSIONS_FOR_ROLLING_RATIOS} sessions)"

        # Exposure breakdown by instrument & sector
        instrument_exposure: list[dict[str, Any]] = []
        sector_exposure_map: dict[str, float] = {}
        max_single_stock_pct = 0.0
        for h in snapshot.get("open_holdings", []):
            mv = float(h.get("market_value_inr", 0.0))
            wt = (mv / max(1.0, cur_nav)) * 100.0
            instrument_exposure.append({
                "symbol": str(h["symbol"]),
                "instrument_type": str(h["instrument_type"]),
                "sector": str(h.get("sector", "UNKNOWN_SECTOR")),
                "market_value_inr": round(mv, 2),
                "weight_pct": round(wt, 2),
            })
            if str(h["instrument_type"]) == "TACTICAL_EQUITY":
                max_single_stock_pct = max(max_single_stock_pct, wt)
                sec_nm = str(h.get("sector", "UNKNOWN_SECTOR"))
                sector_exposure_map[sec_nm] = sector_exposure_map.get(sec_nm, 0.0) + wt

        sector_exposure = [
            {"sector": s, "weight_pct": round(w, 2)}
            for s, w in sorted(sector_exposure_map.items(), key=lambda kv: kv[1], reverse=True)
        ]
        max_sector_pct = float(sector_exposure[0]["weight_pct"]) if sector_exposure else 0.0

        core_pct = (core_val / max(1.0, cur_nav)) * 100.0
        tact_pct = (tact_val / max(1.0, cur_nav)) * 100.0
        equity_total_pct = core_pct + tact_pct
        liquid_pct = (liquid_val / max(1.0, cur_nav)) * 100.0
        cash_pct = ((settled_cash + unsettled_cash) / max(1.0, cur_nav)) * 100.0
        cash_utilization_pct = (invested_val / max(1.0, cur_nav)) * 100.0
        turnover_x = cum_turnover / max(1.0, init_cap)
        cost_pct_gross_traded = (cum_costs / max(1.0, cum_turnover)) * 100.0 if cum_turnover > 0 else 0.0

        return {
            "as_of_date": str(ps["as_of_date"]),
            "sample_sessions": n_sessions,
            "initial_capital_inr": round(init_cap, 2),
            "current_nav_inr": round(cur_nav, 2),
            "daily_pnl_inr": round(daily_pnl, 2),
            "daily_return_pct": round(daily_ret_pct, 4),
            "cumulative_pnl_inr": round(cum_pnl, 2),
            "cumulative_return_pct": round(cum_return_pct, 4),
            "cagr_pct": cagr_pct,
            "cagr_sample_status": cagr_status,
            "max_drawdown_pct": round(max_dd_pct, 2),
            "current_drawdown_pct": round(current_dd_pct, 2),
            "rolling_volatility_pct": rolling_vol_pct,
            "rolling_sharpe": rolling_sharpe,
            "rolling_sortino": rolling_sortino,
            "rolling_ratios_sample_status": ratios_status,
            "exposure_by_instrument": instrument_exposure,
            "exposure_by_sector": sector_exposure,
            "core_etf_exposure_pct": round(core_pct, 2),
            "tactical_equity_exposure_pct": round(tact_pct, 2),
            "equity_exposure_pct": round(equity_total_pct, 2),
            "gold_exposure_pct": 0.0,
            "gold_exposure_note": "0.0% — Frozen config G_SIMPLE_TREND_BASELINE has no Gold ETF (maps 60/30/10 & 45/10/45 to NIFTYBEES / Family E / LIQUIDBEES).",
            "liquid_allocation_pct": round(liquid_pct, 2),
            "cash_pct": round(cash_pct, 2),
            "cash_utilisation_pct": round(cash_utilization_pct, 2),
            "max_single_stock_pct": round(max_single_stock_pct, 2),
            "max_sector_exposure_pct": round(max_sector_pct, 2),
            "turnover_inr": round(cum_turnover, 2),
            "turnover_x": round(turnover_x, 3),
            "total_transaction_costs_inr": round(cum_costs, 2),
            "cost_pct_of_gross_traded_value": round(cost_pct_gross_traded, 4),
            "realized_pnl_inr": round(realized_pnl, 2),
            "unrealized_pnl_inr": round(unrealized_pnl, 2),
            "cum_dividends_inr": round(cum_dividends, 2),
            "cum_liquid_yield_inr": round(cum_liq_yield, 2),
        }

    def evaluate_risk_and_alerts(
        self,
        snapshot: dict[str, Any],
        *,
        config_verified: bool = True,
        data_quality_report: dict[str, Any] | None = None,
        regime_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run full independent risk audit, generate Section 5.2 alerts, and decide Section 5.3 fail-safe."""
        ps = snapshot["portfolio_state"]
        recon = snapshot["reconciliation"]
        as_of = str(ps["as_of_date"])
        metrics = self.compute_portfolio_risk_metrics(snapshot, prospective_only=False)
        th = self.risk_thresholds

        alerts: list[dict[str, Any]] = []
        safe_mode_triggers: list[str] = []

        # 1. Configuration Integrity Check (CRITICAL)
        cfg_ok = bool(config_verified) and bool(snapshot.get("configuration_integrity", {}).get("verified", False))
        if not cfg_ok:
            safe_mode_triggers.append("CONFIG_HASH_MISMATCH")
            alerts.append(
                _make_alert(
                    code="FROZEN_CONFIG_HASH_MISMATCH",
                    severity="CRITICAL",
                    component="configuration_integrity",
                    observed_value="FAILED",
                    threshold=snapshot.get("configuration_integrity", {}).get("expected_sha256_prefix", "47a30f4bbd004c5f"),
                    description="Frozen prospective configuration SHA256 verification failed.",
                    recommended_action="Halt new paper entries immediately and restore unmodified frozen config.",
                    as_of_date=as_of,
                )
            )

        # 2. Portfolio Reconciliation Check (CRITICAL)
        recon_diff = float(recon.get("reconciliation_diff_inr", 0.0))
        if (not recon.get("passed", False)) or recon_diff > th.reconciliation_tolerance_inr:
            safe_mode_triggers.append(f"RECONCILIATION_FAILURE({recon_diff:.4f} INR)")
            alerts.append(
                _make_alert(
                    code="PORTFOLIO_RECONCILIATION_FAILURE",
                    severity="CRITICAL",
                    component="portfolio_ledger",
                    observed_value=round(recon_diff, 4),
                    threshold=th.reconciliation_tolerance_inr,
                    description=f"Four-ledger reconciliation failed: {recon.get('failure_reasons')}",
                    recommended_action="Enter SAFE_MODE, halt new entries, audit SQLite ledgers, and run verified recovery.",
                    as_of_date=as_of,
                )
            )

        # 3. Negative or Inconsistent Cash Balance (CRITICAL)
        settled_cash = float(recon.get("settled_cash_inr", 0.0))
        if settled_cash < -0.01:
            safe_mode_triggers.append(f"NEGATIVE_CASH({settled_cash:.2f})")
            alerts.append(
                _make_alert(
                    code="NEGATIVE_CASH_BALANCE",
                    severity="CRITICAL",
                    component="cash_ledger",
                    observed_value=round(settled_cash, 2),
                    threshold=0.0,
                    description=f"Settled cash balance is negative: INR {settled_cash:.2f}.",
                    recommended_action="Enter SAFE_MODE and audit recent debit orders.",
                    as_of_date=as_of,
                )
            )

        # 4. Settlement Mismatch (CRITICAL)
        settle_diff = float(recon.get("settlement_diff_inr", 0.0))
        if settle_diff > th.reconciliation_tolerance_inr:
            safe_mode_triggers.append(f"SETTLEMENT_MISMATCH({settle_diff:.4f})")
            alerts.append(
                _make_alert(
                    code="SETTLEMENT_LEDGER_MISMATCH",
                    severity="CRITICAL",
                    component="settlement_ledger",
                    observed_value=round(settle_diff, 4),
                    threshold=th.reconciliation_tolerance_inr,
                    description="Unsettled cash in portfolio_state disagrees with pending settlement_ledger sum.",
                    recommended_action="Enter SAFE_MODE and reconcile pending T+1 settlements.",
                    as_of_date=as_of,
                )
            )

        # 5. Unexpected Strategy or Instrument Activation (CRITICAL)
        for o in snapshot.get("simulated_orders", []):
            sid = str(o.get("strategy_id", ""))
            if sid in SUSPENDED_STRATEGIES_HARD:
                safe_mode_triggers.append(f"SUSPENDED_STRATEGY_ACTIVATED({sid})")
                alerts.append(
                    _make_alert(
                        code="UNEXPECTED_STRATEGY_ACTIVATION",
                        severity="CRITICAL",
                        component="strategy_guard",
                        observed_value=sid,
                        threshold="STRAT_E_DEFENSIVE_YIELD_LOWVOL",
                        description=f"Suspended strategy '{sid}' detected in simulated order history.",
                        recommended_action="Immediate SAFE_MODE halt and forensic code audit.",
                        as_of_date=as_of,
                    )
                )
                break

        # 6. Duplicate Order / Cost Ledger Integrity (CRITICAL)
        orders = snapshot.get("simulated_orders", [])
        cost_rows = snapshot.get("transaction_cost_ledger", [])
        filled_oids = [str(o["order_id"]) for o in orders if o["execution_status"] in ("FILLED", "PARTIAL")]
        cost_oids = [str(c["order_id"]) for c in cost_rows]
        if len(filled_oids) != len(set(filled_oids)) or len(cost_oids) != len(set(cost_oids)) or set(filled_oids) != set(cost_oids):
            safe_mode_triggers.append("ORDER_COST_LEDGER_INCONSISTENCY")
            alerts.append(
                _make_alert(
                    code="DUPLICATE_OR_MISSING_COST_ENTRY",
                    severity="CRITICAL",
                    component="transaction_cost_ledger",
                    observed_value={"filled_orders": len(filled_oids), "cost_entries": len(cost_oids)},
                    threshold="1-to-1 unique match",
                    description="Filled simulated orders and transaction_cost_ledger entries do not match 1-to-1.",
                    recommended_action="Enter SAFE_MODE and verify no order was double-charged or omitted.",
                    as_of_date=as_of,
                )
            )

        # 7. Data Quality & Stale Prices Check
        dq = data_quality_report or {}
        dq_state = str(dq.get("overall_status", "OK")).upper()
        if dq_state == "INVALID":
            safe_mode_triggers.append("INVALID_MARKET_DATA")
            alerts.append(
                _make_alert(
                    code="MISSING_OR_INVALID_MARKET_DATA",
                    severity="CRITICAL",
                    component="market_data_validation",
                    observed_value=dq_state,
                    threshold="OK / PARTIAL_DIAGNOSTIC_ONLY",
                    description=str(dq.get("summary", "Required daily market bars missing or invalid.")),
                    recommended_action="Enter SAFE_MODE; await valid market data without fabricating values.",
                    as_of_date=as_of,
                )
            )
        elif dq_state in ("PARTIAL_DIAGNOSTIC_ONLY", "DEGRADED"):
            alerts.append(
                _make_alert(
                    code="DIAGNOSTIC_INDEX_DATA_BOUNDARY",
                    severity="INFO" if dq_state == "PARTIAL_DIAGNOSTIC_ONLY" else "WARNING",
                    component="market_data_validation",
                    observed_value=dq_state,
                    threshold="OK",
                    description=str(dq.get("summary", "External daily_index_bars ends 2025-10-10; diagnostic-only.")),
                    recommended_action="Continue trading strictly on 100%-coverage daily_bars; do not fabricate VIX/GS10Y.",
                    as_of_date=as_of,
                )
            )

        stale_days = int(dq.get("stale_price_days", 0) or 0)
        if stale_days >= th.stale_price_days_warning:
            alerts.append(
                _make_alert(
                    code="STALE_MARKET_PRICES",
                    severity="WARNING",
                    component="market_data_validation",
                    observed_value=stale_days,
                    threshold=th.stale_price_days_warning,
                    description=f"Market data appears stale by {stale_days} trading sessions.",
                    recommended_action="Verify NSE daily bar ingestion before next scheduled session.",
                    as_of_date=as_of,
                )
            )

        # 8. Portfolio Drawdown Alerts
        cur_dd = float(metrics["current_drawdown_pct"])
        if cur_dd >= th.drawdown_critical_pct:
            alerts.append(
                _make_alert(
                    code="SEVERE_PORTFOLIO_DRAWDOWN",
                    severity="CRITICAL",
                    component="portfolio_risk",
                    observed_value=cur_dd,
                    threshold=th.drawdown_critical_pct,
                    description=f"Current drawdown {cur_dd:.2f}% breached critical threshold {th.drawdown_critical_pct:.1f}%.",
                    recommended_action="Pause new Family E tactical entries and perform independent strategy review.",
                    as_of_date=as_of,
                )
            )
        elif cur_dd >= th.drawdown_warning_pct:
            alerts.append(
                _make_alert(
                    code="ELEVATED_PORTFOLIO_DRAWDOWN",
                    severity="WARNING",
                    component="portfolio_risk",
                    observed_value=cur_dd,
                    threshold=th.drawdown_warning_pct,
                    description=f"Current drawdown {cur_dd:.2f}% exceeded normal watch threshold {th.drawdown_warning_pct:.1f}%.",
                    recommended_action="Monitor closely; do not override frozen strategy during normal drawdowns.",
                    as_of_date=as_of,
                )
            )

        # 9. Allocation Deviation Alert (only once portfolio has deployed capital on >= 2 sessions)
        if int(metrics["sample_sessions"]) >= 2:
            tgt_c = float(ps["target_core_pct"])
            obs_c = float(metrics["core_etf_exposure_pct"])
            dev_c = abs(obs_c - tgt_c)
            if dev_c > th.allocation_deviation_warning_pp:
                alerts.append(
                    _make_alert(
                        code="UNEXPECTED_ALLOCATION_DEVIATION",
                        severity="WARNING",
                        component="capital_allocation",
                        observed_value=round(dev_c, 2),
                        threshold=th.allocation_deviation_warning_pp,
                        description=f"Core NIFTYBEES allocation ({obs_c:.1f}%) deviates by {dev_c:.1f} pp from target ({tgt_c:.1f}%).",
                        recommended_action="Verify 8.5% rebalance band trigger and pending next-session Core rebalance order.",
                        as_of_date=as_of,
                    )
                )

        # 10. Concentration Alerts (Single Stock & Sector)
        if float(metrics["max_single_stock_pct"]) > th.single_stock_concentration_pct:
            alerts.append(
                _make_alert(
                    code="EXCESSIVE_SINGLE_STOCK_CONCENTRATION",
                    severity="WARNING",
                    component="portfolio_concentration",
                    observed_value=metrics["max_single_stock_pct"],
                    threshold=th.single_stock_concentration_pct,
                    description=f"Largest tactical equity position is {metrics['max_single_stock_pct']:.2f}% of NAV.",
                    recommended_action="Verify sizing cap (15%) and monitor price appreciation drift.",
                    as_of_date=as_of,
                )
            )
        if float(metrics["max_sector_exposure_pct"]) > th.sector_concentration_pct:
            alerts.append(
                _make_alert(
                    code="EXCESSIVE_SECTOR_CONCENTRATION",
                    severity="WARNING",
                    component="portfolio_concentration",
                    observed_value=metrics["max_sector_exposure_pct"],
                    threshold=th.sector_concentration_pct,
                    description=f"Largest tactical sector exposure is {metrics['max_sector_exposure_pct']:.2f}% of NAV.",
                    recommended_action="Block additional entries in breached sector until exposure normalizes.",
                    as_of_date=as_of,
                )
            )

        # 11. Abnormal Transaction Costs Alert
        if float(metrics["cost_pct_of_gross_traded_value"]) > th.abnormal_cost_pct_gross_traded:
            alerts.append(
                _make_alert(
                    code="ABNORMAL_TRANSACTION_COSTS",
                    severity="WARNING",
                    component="execution_costs",
                    observed_value=metrics["cost_pct_of_gross_traded_value"],
                    threshold=th.abnormal_cost_pct_gross_traded,
                    description=f"Total costs are {metrics['cost_pct_of_gross_traded_value']:.3f}% of gross traded value.",
                    recommended_action="Audit slippage assumptions, small-lot DP drag, and tariff schedule.",
                    as_of_date=as_of,
                )
            )

        # 12. Failed Simulated Executions Alert
        latest_date = as_of
        failed_today = [
            o for o in orders if str(o["trading_date"]) == latest_date and str(o["execution_status"]) == "FAILED"
        ]
        if len(failed_today) >= th.max_failed_orders_session:
            alerts.append(
                _make_alert(
                    code="FAILED_SIMULATED_EXECUTIONS",
                    severity="WARNING",
                    component="execution_simulator",
                    observed_value=len(failed_today),
                    threshold=th.max_failed_orders_session,
                    description=f"{len(failed_today)} simulated order(s) failed execution on {latest_date}.",
                    recommended_action="Inspect missing bars / circuit-limit / liquidity rejection reasons.",
                    as_of_date=as_of,
                )
            )

        # Reuse Prompt 4 `evaluate_regime_and_strategy_health` advisory layer
        rc = regime_context or {}
        closed_trades = [
            {"total_pnl_with_div_inr": float(p["realized_pnl_inr"]) + float(p["dividends_received_inr"])}
            for p in snapshot.get("closed_tactical_positions", [])
        ]
        gross_profit = max(0.0, float(metrics["cumulative_pnl_inr"]) + float(metrics["total_transaction_costs_inr"]))
        cost_pct_gross_profit = (float(metrics["total_transaction_costs_inr"]) / max(1.0, gross_profit)) * 100.0 if gross_profit > 100.0 else 0.0
        advisory = evaluate_regime_and_strategy_health(
            date=as_of,
            vol_ratio_20_60=rc.get("vol_ratio_20_60"),
            vol_pctile_252d=rc.get("vol_pctile_252d"),
            pct_ab50=rc.get("pct_ab50"),
            breadth_delta_20d=rc.get("breadth_delta_20d"),
            current_drawdown_pct=cur_dd,
            recent_closed_trades=closed_trades,
            trailing_annualized_turnover_x=float(metrics["turnover_x"]),
            cost_pct_of_gross_profits=cost_pct_gross_profit,
            max_sector_exposure_pct=float(metrics["max_sector_exposure_pct"]),
            uninvested_cash_pct=float(metrics["cash_pct"]),
            consecutive_high_cash_days=int(ps.get("consecutive_high_cash_days", 0)),
            regime_flips_last_20d=int(rc.get("regime_flips_last_20d", 0)),
            data_quality_ok=(dq_state != "INVALID"),
            reconciliation_diff_inr=recon_diff,
            thresholds=self.regime_thresholds,
        )

        should_enter_safe_mode = len(safe_mode_triggers) > 0
        overall_health = (
            "CRITICAL_SAFE_MODE"
            if should_enter_safe_mode
            else ("WARNING" if any(a["severity"] in ("WARNING", "CRITICAL") for a in alerts) else "HEALTHY")
        )

        return {
            "as_of_date": as_of,
            "evaluated_at_utc": _utc_now_iso(),
            "overall_health": overall_health,
            "should_enter_safe_mode": should_enter_safe_mode,
            "safe_mode_triggers": safe_mode_triggers,
            "metrics": metrics,
            "alerts": alerts,
            "alert_counts": {
                "CRITICAL": sum(1 for a in alerts if a["severity"] == "CRITICAL"),
                "WARNING": sum(1 for a in alerts if a["severity"] == "WARNING"),
                "INFO": sum(1 for a in alerts if a["severity"] == "INFO"),
            },
            "prompt4_advisory_health": advisory,
            "thresholds": asdict(self.risk_thresholds),
        }
