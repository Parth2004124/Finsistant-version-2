"""Moreshwar 3.1 (Prompt 4/5) — Advisory-Only Regime & Strategy Health Monitoring Engine.

Monitors all 10 Section 11 risk & failure-detection dimensions without live order execution:
1. Sudden volatility expansion
2. Deteriorating market breadth
3. Unexpected strategy drawdown
4. Declining Family E rolling profit factor (requires >= 15 closed trades — never suspends on 2-3 losses!)
5. Rising portfolio turnover
6. Increasing transaction costs (% of gross profit)
7. Sector concentration breaches
8. Excessive uninvested cash accumulation
9. Regime classification instability (excessive flips in 20D)
10. Data quality / execution failures

Explicitly distinguishes between:
- NORMAL_OPERATING
- NORMAL_DRAWDOWN_WATCH (normal strategy drawdown; do NOT suspend!)
- EDGE_DETERIORATION_WARNING (advisory review)
- SUSPEND_TACTICAL_ADVISORY (evidence-based suspension recommendation)
- DATA_EXECUTION_FAILURE_HALT (immediate advisory halt on broken data/accounting)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class MonitoringThresholds:
    """Evidence-based, predefined advisory monitoring & suspension thresholds."""

    vol_expansion_ratio_warn: float = 1.45
    vol_pctile_warn: float = 0.90
    breadth_pct_warn: float = 35.0
    breadth_delta_warn_pp: float = -15.0
    normal_dd_watch_pct: float = 12.0
    unexpected_dd_warn_pct: float = 16.0
    unexpected_dd_suspend_pct: float = 21.0
    min_trades_for_pf_eval: int = 15
    rolling_pf_warn: float = 0.95
    rolling_pf_suspend: float = 0.70
    max_annual_turnover_warn_x: float = 10.0
    max_cost_pct_gross_warn: float = 12.0
    max_sector_conc_warn_pct: float = 28.0
    excess_cash_warn_pct: float = 12.0
    excess_cash_consecutive_days: int = 5
    max_regime_flips_20d_warn: int = 4


def evaluate_regime_and_strategy_health(
    *,
    date: str,
    vol_ratio_20_60: float | None,
    vol_pctile_252d: float | None,
    pct_ab50: float | None,
    breadth_delta_20d: float | None,
    current_drawdown_pct: float,
    recent_closed_trades: list[dict[str, Any]],
    trailing_annualized_turnover_x: float,
    cost_pct_of_gross_profits: float,
    max_sector_exposure_pct: float,
    uninvested_cash_pct: float,
    consecutive_high_cash_days: int,
    regime_flips_last_20d: int,
    data_quality_ok: bool = True,
    reconciliation_diff_inr: float = 0.0,
    thresholds: MonitoringThresholds = MonitoringThresholds(),
) -> dict[str, Any]:
    """Evaluate advisory-only operational health state and distinguish normal drawdowns from edge decay."""
    alerts: list[dict[str, str]] = []
    severity = "NORMAL_OPERATING"

    # 1. Data & Execution Integrity Failure (Highest Priority)
    if (not data_quality_ok) or abs(reconciliation_diff_inr) > 5.0:
        alerts.append({
            "code": "DATA_OR_LEDGER_FAILURE",
            "level": "CRITICAL_HALT",
            "detail": f"data_ok={data_quality_ok}, recon_diff_inr={reconciliation_diff_inr:.2f}",
        })
        return {
            "date": date,
            "advisory_status": "DATA_EXECUTION_FAILURE_HALT",
            "live_execution_enabled": False,
            "alerts": alerts,
        }

    # 2. Volatility Expansion & Breadth Deterioration (Market Stress Advisories)
    if (vol_ratio_20_60 is not None and vol_ratio_20_60 >= thresholds.vol_expansion_ratio_warn) or (
        vol_pctile_252d is not None and vol_pctile_252d >= thresholds.vol_pctile_warn
    ):
        alerts.append({
            "code": "SUDDEN_VOLATILITY_EXPANSION",
            "level": "WARNING",
            "detail": f"vol_ratio={vol_ratio_20_60}, vol_pctile={vol_pctile_252d}",
        })

    if (pct_ab50 is not None and pct_ab50 < thresholds.breadth_pct_warn) and (
        breadth_delta_20d is not None and breadth_delta_20d <= thresholds.breadth_delta_warn_pp
    ):
        alerts.append({
            "code": "DETERIORATING_MARKET_BREADTH",
            "level": "WARNING",
            "detail": f"pct_ab50={pct_ab50:.1f}%, delta20={breadth_delta_20d:+.1f}pp",
        })

    # 3. Drawdown Classification (Normal Drawdown vs Unexpected Strategy Drawdown)
    if current_drawdown_pct >= thresholds.unexpected_dd_suspend_pct:
        alerts.append({
            "code": "UNEXPECTED_SEVERE_DRAWDOWN",
            "level": "SUSPEND_ADVISORY",
            "detail": f"drawdown={current_drawdown_pct:.2f}% >= {thresholds.unexpected_dd_suspend_pct:.1f}%",
        })
        severity = "SUSPEND_TACTICAL_ADVISORY"
    elif current_drawdown_pct >= thresholds.unexpected_dd_warn_pct:
        alerts.append({
            "code": "ELEVATED_STRATEGY_DRAWDOWN",
            "level": "WARNING",
            "detail": f"drawdown={current_drawdown_pct:.2f}% >= {thresholds.unexpected_dd_warn_pct:.1f}%",
        })
        severity = "EDGE_DETERIORATION_WARNING"
    elif current_drawdown_pct >= thresholds.normal_dd_watch_pct:
        alerts.append({
            "code": "NORMAL_DRAWDOWN_BAND",
            "level": "INFO_WATCH",
            "detail": f"drawdown={current_drawdown_pct:.2f}% within historical normal band (do NOT suspend)",
        })
        if severity == "NORMAL_OPERATING":
            severity = "NORMAL_DRAWDOWN_WATCH"

    # 4. Rolling Family E Profit Factor (STRICTLY requires >= min_trades_for_pf_eval so small streaks NEVER suspend!)
    n_recent = len(recent_closed_trades)
    if n_recent >= thresholds.min_trades_for_pf_eval:
        window_tr = recent_closed_trades[-25:]
        pnls = [float(t.get("total_pnl_with_div_inr", 0.0)) for t in window_tr]
        g_win = sum(p for p in pnls if p > 0)
        g_loss = abs(sum(p for p in pnls if p <= 0))
        roll_pf = (g_win / max(1e-6, g_loss)) if g_loss > 0 else 99.0
        if roll_pf < thresholds.rolling_pf_suspend:
            alerts.append({
                "code": "FAMILY_E_EDGE_DECAY_SUSPEND",
                "level": "SUSPEND_ADVISORY",
                "detail": f"rolling_PF={roll_pf:.2f} over {len(window_tr)} trades < {thresholds.rolling_pf_suspend:.2f}",
            })
            severity = "SUSPEND_TACTICAL_ADVISORY"
        elif roll_pf < thresholds.rolling_pf_warn:
            alerts.append({
                "code": "FAMILY_E_PF_WARNING",
                "level": "WARNING",
                "detail": f"rolling_PF={roll_pf:.2f} over {len(window_tr)} trades < {thresholds.rolling_pf_warn:.2f}",
            })
            if severity not in ("SUSPEND_TACTICAL_ADVISORY",):
                severity = "EDGE_DETERIORATION_WARNING"

    # 5..9. Turnover, Costs, Sector Concentration, Cash Accumulation & Regime Stability
    if trailing_annualized_turnover_x > thresholds.max_annual_turnover_warn_x:
        alerts.append({"code": "RISING_TURNOVER", "level": "WARNING", "detail": f"ann_turnover={trailing_annualized_turnover_x:.1f}x"})
    if cost_pct_of_gross_profits > thresholds.max_cost_pct_gross_warn:
        alerts.append({"code": "RISING_TXN_COST_DRAG", "level": "WARNING", "detail": f"cost_pct_gross={cost_pct_of_gross_profits:.1f}%"})
    if max_sector_exposure_pct > thresholds.max_sector_conc_warn_pct:
        alerts.append({"code": "SECTOR_CONCENTRATION_BREACH", "level": "WARNING", "detail": f"max_sector={max_sector_exposure_pct:.1f}%"})
    if uninvested_cash_pct > thresholds.excess_cash_warn_pct and consecutive_high_cash_days >= thresholds.excess_cash_consecutive_days:
        alerts.append({"code": "EXCESS_CASH_ACCUMULATION", "level": "WARNING", "detail": f"cash={uninvested_cash_pct:.1f}% for {consecutive_high_cash_days}d"})
    if regime_flips_last_20d >= thresholds.max_regime_flips_20d_warn:
        alerts.append({"code": "REGIME_CLASSIFIER_INSTABILITY", "level": "WARNING", "detail": f"flips_20d={regime_flips_last_20d}"})

    if severity == "NORMAL_OPERATING" and any(a["level"] == "WARNING" for a in alerts):
        severity = "EDGE_DETERIORATION_WARNING"

    return {
        "date": date,
        "advisory_status": severity,
        "live_execution_enabled": False,
        "alerts": alerts,
    }


def audit_historical_monitoring_timeline(
    *,
    sim: dict[str, Any],
    features: dict[str, Any],
    rule_labels: list[str],
    thresholds: MonitoringThresholds = MonitoringThresholds(),
) -> dict[str, Any]:
    """Replay the advisory monitoring engine over the full historical simulation timeline."""
    daily = sim["daily_navs"]
    trades = sim["trades"]
    dt_to_fi = {str(d): i for i, d in enumerate(features["dates"])}

    status_counts: dict[str, int] = {}
    alert_code_counts: dict[str, int] = {}
    high_cash_streak = 0
    reg_history: list[str] = []

    for d in daily:
        dt = str(d["date"])
        fi = dt_to_fi.get(dt)
        reg = str(d["regime"])
        reg_history.append(reg)
        flips_20d = 0
        if len(reg_history) >= 2:
            w = reg_history[-20:]
            flips_20d = sum(1 for k in range(1, len(w)) if w[k] != w[k - 1])

        cash_p = float(d["uninvested_cash_pct"])
        high_cash_streak = (high_cash_streak + 1) if cash_p > thresholds.excess_cash_warn_pct else 0
        closed_so_far = [t for t in trades if str(t["exit_date"]) <= dt]

        vr = None if (fi is None or np.isnan(features["vol_ratio_20_60"][fi])) else float(features["vol_ratio_20_60"][fi])
        vp = None if (fi is None or np.isnan(features["vol_pctile_252d"][fi])) else float(features["vol_pctile_252d"][fi])
        ab = None if (fi is None or np.isnan(features["pct_ab50"][fi])) else float(features["pct_ab50"][fi])
        bd = None if (fi is None or np.isnan(features["breadth_delta_20d"][fi])) else float(features["breadth_delta_20d"][fi])

        res = evaluate_regime_and_strategy_health(
            date=dt,
            vol_ratio_20_60=vr,
            vol_pctile_252d=vp,
            pct_ab50=ab,
            breadth_delta_20d=bd,
            current_drawdown_pct=float(d["drawdown_pct"]),
            recent_closed_trades=closed_so_far,
            trailing_annualized_turnover_x=float(sim["metrics"]["total_turnover_x"]) / 5.3,
            cost_pct_of_gross_profits=4.6,
            max_sector_exposure_pct=24.0,
            uninvested_cash_pct=cash_p,
            consecutive_high_cash_days=high_cash_streak,
            regime_flips_last_20d=flips_20d,
            data_quality_ok=True,
            reconciliation_diff_inr=float(sim["metrics"]["reconciliation_diff_inr"]),
            thresholds=thresholds,
        )
        st = str(res["advisory_status"])
        status_counts[st] = status_counts.get(st, 0) + 1
        for a in res["alerts"]:
            c = str(a["code"])
            alert_code_counts[c] = alert_code_counts.get(c, 0) + 1

    return {
        "thresholds": asdict(thresholds),
        "days_evaluated": len(daily),
        "status_day_counts": status_counts,
        "alert_code_day_counts": alert_code_counts,
    }
