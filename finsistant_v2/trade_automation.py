"""Bloomberg-Style Trade Automation Engine (RBLD + EMSX Blotter + Pre/Post-Trade BTCA).

Provides institutional trade workflow governance across both Moreshwar 3.2 prospective
paper trading and the live scan desk (with zero fake fallbacks — fail-closed compliance):
1. Declarative Rule Builder (`RuleBuilderEngine` / `RBLD`) with toggleable rules and
   exact per-rule pass/block audit verdicts.
2. Pre-Trade Transaction Cost Analysis (`compute_pre_trade_tca`) scoring order
   difficulty (%ADV participation, intraday spread proxy, daily volatility, market impact)
   and routing compliant orders across execution algos (`WHLS` Broker/Algo Wheel).
3. Order Lifecycle State Machine & Exception-First `EMSOrderBlotter` (`STAGED` ->
   `COMPLIANCE_PASSED` / `COMPLIANCE_BLOCKED` -> `ROUTED` -> `FILLED` / `PARTIAL` /
   `UNFILLED` / `REJECTED`).
4. Post-Trade `BTCA` Implementation Shortfall attribution (`compute_post_trade_tca`)
   comparing `actual_exec_price` vs `arrival_price`, decomposed into delay cost,
   execution slippage, and statutory fees, plus bounded feedback (`tca_hurdle_adder_bps`).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import math
from typing import Any

STAGED = "STAGED"
COMPLIANCE_PASSED = "COMPLIANCE_PASSED"
COMPLIANCE_BLOCKED = "COMPLIANCE_BLOCKED"
ROUTED = "ROUTED"
PARTIAL = "PARTIAL"
FILLED = "FILLED"
UNFILLED = "UNFILLED"
REJECTED = "REJECTED"
CANCELLED = "CANCELLED"

VALID_ALGO_MODES = frozenset({
    "NEXT_SESSION_OPEN",
    "LIMIT_ORDER_SIM",
    "SPREAD_AWARE",
    "VOLATILITY_ADJUSTED",
})


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class RuleSpec:
    rule_id: str
    name: str
    category: str  # DATA_INTEGRITY | COMPLIANCE | LIQUIDITY | EXECUTION_QUALITY | RISK_GUARD
    enabled: bool
    action_on_fail: str  # BLOCK | HALT_DESK | WARN | ROUTE_ALGO
    threshold: float
    description: str


DEFAULT_RBLD_RULES: tuple[RuleSpec, ...] = (
    RuleSpec(
        rule_id="RBLD_01_DATA_INTEGRITY",
        name="Fail-Closed Exchange & Fundamental Integrity",
        category="DATA_INTEGRITY",
        enabled=True,
        action_on_fail="BLOCK",
        threshold=1.0,
        description="Blocks routing when spot price, OHLCV bar, or live fundamentals are unavailable or invalid.",
    ),
    RuleSpec(
        rule_id="RBLD_02_DESK_KILL_SWITCH",
        name="Desk Drawdown, Reconciliation & BTCA Circuit Breaker",
        category="RISK_GUARD",
        enabled=True,
        action_on_fail="HALT_DESK",
        threshold=10.0,
        description="Halts new BUY routing when drawdown >= 10%, recon drift > INR 0.01, SAFE_MODE active, or BTCA slippage drift > 75 bps.",
    ),
    RuleSpec(
        rule_id="RBLD_03_PRICE_BAND_DRIFT",
        name="Live vs EOD Reference Price Band",
        category="COMPLIANCE",
        enabled=True,
        action_on_fail="BLOCK",
        threshold=5.0,
        description="Blocks routing when live price deviates > 5.0% from EOD reference bar.",
    ),
    RuleSpec(
        rule_id="RBLD_04_MIN_LIQUIDITY_ADV",
        name="20D ADV Floor & Participation Cap",
        category="LIQUIDITY",
        enabled=True,
        action_on_fail="BLOCK",
        threshold=20_000_000.0,
        description="Requires >= INR 2.0 Cr 20D average daily turnover and <= 1.5% ADV participation per order.",
    ),
    RuleSpec(
        rule_id="RBLD_05_SPREAD_VOL_LIMIT",
        name="Intraday Spread & Volatility Shield",
        category="EXECUTION_QUALITY",
        enabled=True,
        action_on_fail="BLOCK",
        threshold=7.5,
        description="Blocks entries when H-L spread range > 7.5% or 20D daily vol > 6.0%.",
    ),
    RuleSpec(
        rule_id="RBLD_06_CONCENTRATION_CAPS",
        name="Pre-Trade AIM Single-Stock & Sector Caps",
        category="COMPLIANCE",
        enabled=True,
        action_on_fail="BLOCK",
        threshold=15.0,
        description="Enforces single-stock <= 15% NAV, sector <= 30% NAV, and max open tactical slots.",
    ),
    RuleSpec(
        rule_id="RBLD_07_PRE_TRADE_TCA_EV",
        name="Pre-Trade TCA Net-of-Cost EV Hurdle",
        category="EXECUTION_QUALITY",
        enabled=True,
        action_on_fail="BLOCK",
        threshold=0.0,
        description="Requires expected net alpha (bps) minus Pre-Trade TCA costs >= effective EV hurdle.",
    ),
)


def compute_pre_trade_tca(
    *,
    side: str,
    arrival_price: float,
    order_notional_inr: float,
    adv_20d_inr: float,
    hl_range_pct: float,
    daily_vol_frac: float,
    statutory_cost_bps: float = 28.0,
    prefer_algo_wheel: bool = True,
    forced_execution_mode: str | None = None,
) -> dict[str, Any]:
    """Compute Pre-Trade TCA difficulty profile and select execution algo (`WHLS`)."""
    sd = str(side or "BUY").upper().strip()
    px = max(1e-6, float(arrival_price or 0.0))
    notional = max(0.0, float(order_notional_inr or 0.0))
    adv = max(1.0, float(adv_20d_inr or 0.0))
    hl = max(0.0, float(hl_range_pct or 0.0))
    vol = max(0.001, float(daily_vol_frac or 0.015))

    adv_pct = (notional / adv) * 100.0
    # Market impact scales with sqrt(%ADV) * daily vol; spread proxy ~ 12% of H-L range capped safely
    est_impact_bps = round(min(180.0, 2.0 + 38.0 * math.sqrt(max(0.0, adv_pct / 100.0)) * (vol / 0.018)), 2)
    est_spread_bps = round(min(90.0, max(3.0, hl * 5.0)), 2)
    est_stat_bps = round(max(0.0, float(statutory_cost_bps)), 2)
    est_total_bps = round(est_impact_bps + est_spread_bps + est_stat_bps, 2)

    difficulty_score = round(
        min(
            100.0,
            max(
                0.0,
                (adv_pct / 1.5) * 35.0 + (hl / 7.5) * 35.0 + ((vol * 100.0) / 5.0) * 30.0,
            ),
        ),
        1,
    )
    if hl > 7.5 or adv_pct > 1.5 or vol > 0.06:
        tier = "EXTREME"
    elif difficulty_score >= 60.0 or hl > 4.5 or vol > 0.03:
        tier = "HIGH"
    elif difficulty_score >= 30.0 or hl > 2.5 or adv_pct > 0.35:
        tier = "MODERATE"
    else:
        tier = "LOW"

    if forced_execution_mode and forced_execution_mode in VALID_ALGO_MODES:
        selected_algo = forced_execution_mode
    elif sd == "SELL":
        # Protective risk exits always prioritize guaranteed immediate liquidity at next open
        selected_algo = "NEXT_SESSION_OPEN"
    elif not prefer_algo_wheel:
        selected_algo = "NEXT_SESSION_OPEN"
    elif hl > 4.5:
        selected_algo = "SPREAD_AWARE"
    elif vol > 0.028:
        selected_algo = "VOLATILITY_ADJUSTED"
    elif tier == "MODERATE":
        selected_algo = "LIMIT_ORDER_SIM"
    else:
        selected_algo = "NEXT_SESSION_OPEN"

    return {
        "side": sd,
        "arrival_price": round(px, 4),
        "order_notional_inr": round(notional, 2),
        "adv_20d_inr": round(adv, 2),
        "adv_participation_pct": round(adv_pct, 4),
        "hl_range_pct": round(hl, 2),
        "daily_vol_frac": round(vol, 5),
        "est_impact_bps": est_impact_bps,
        "est_spread_bps": est_spread_bps,
        "est_statutory_bps": est_stat_bps,
        "est_total_cost_bps": est_total_bps,
        "difficulty_score": difficulty_score,
        "difficulty_tier": tier,
        "selected_algo": selected_algo,
    }


def compute_post_trade_tca(
    *,
    order_id: str,
    symbol: str,
    side: str,
    trading_date: str,
    arrival_price: float,
    bar_open: float,
    actual_exec_price: float,
    req_qty: float,
    filled_qty: float,
    gross_value_inr: float,
    total_cost_inr: float,
    slippage_inr: float = 0.0,
    algo_strategy: str = "NEXT_SESSION_OPEN",
    pre_trade_est_cost_bps: float | None = None,
) -> dict[str, Any]:
    """Compute Post-Trade BTCA Implementation Shortfall (bps) vs arrival & open benchmarks."""
    sd = str(side or "BUY").upper().strip()
    sign = 1.0 if sd == "BUY" else -1.0
    arr = float(arrival_price or 0.0)
    opn = float(bar_open or 0.0)
    act = float(actual_exec_price or 0.0)
    rq = max(1e-6, float(req_qty or 0.0))
    fq = max(0.0, float(filled_qty or 0.0))
    gross = max(0.0, float(gross_value_inr or 0.0))

    fill_rate_pct = round(min(100.0, (fq / rq) * 100.0), 2)
    if fq <= 0 or act <= 0 or arr <= 0:
        return {
            "order_id": order_id,
            "symbol": symbol,
            "side": sd,
            "trading_date": trading_date,
            "algo_strategy": algo_strategy,
            "fill_rate_pct": 0.0,
            "arrival_price": round(arr, 4),
            "bar_open": round(opn, 4),
            "actual_exec_price": round(act, 4),
            "delay_cost_bps": None,
            "execution_slippage_bps": None,
            "statutory_cost_bps": None,
            "implementation_shortfall_bps": None,
            "slippage_surprise_bps": None,
        }

    delay_bps = (((opn - arr) / arr) * 10_000.0 * sign) if opn > 0 else 0.0
    exec_slip_bps = (((act - opn) / opn) * 10_000.0 * sign) if opn > 0 else (((act - arr) / arr) * 10_000.0 * sign)
    reg_cost_only = max(0.0, float(total_cost_inr or 0.0) - float(slippage_inr or 0.0))
    stat_bps = ((reg_cost_only / gross) * 10_000.0) if gross > 0 else 0.0
    total_is_bps = (((act - arr) / arr) * 10_000.0 * sign) + stat_bps
    surprise_bps = (exec_slip_bps + stat_bps - float(pre_trade_est_cost_bps)) if pre_trade_est_cost_bps is not None else None

    return {
        "order_id": order_id,
        "symbol": symbol,
        "side": sd,
        "trading_date": trading_date,
        "algo_strategy": algo_strategy,
        "fill_rate_pct": fill_rate_pct,
        "arrival_price": round(arr, 4),
        "bar_open": round(opn, 4),
        "actual_exec_price": round(act, 4),
        "delay_cost_bps": round(delay_bps, 2),
        "execution_slippage_bps": round(exec_slip_bps, 2),
        "statutory_cost_bps": round(stat_bps, 2),
        "implementation_shortfall_bps": round(total_is_bps, 2),
        "slippage_surprise_bps": round(surprise_bps, 2) if surprise_bps is not None else None,
    }


def summarize_btca_and_feedback(executions: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize Post-Trade BTCA stats and emit bounded hurdle/algo feedback."""
    filled = [e for e in executions if e.get("implementation_shortfall_bps") is not None]
    n_total = len(executions)
    n_filled = len(filled)
    if n_filled == 0:
        return {
            "total_orders": n_total,
            "filled_orders": 0,
            "avg_fill_rate_pct": None,
            "mean_implementation_shortfall_bps": None,
            "mean_delay_cost_bps": None,
            "mean_execution_slippage_bps": None,
            "mean_statutory_cost_bps": None,
            "mean_slippage_surprise_bps": None,
            "tca_hurdle_adder_bps": 0.0,
            "kill_switch_drift_breached": False,
            "algo_breakdown": {},
        }

    avg_fill = sum(float(e.get("fill_rate_pct") or 0.0) for e in executions) / max(1, n_total)
    is_vals = [float(e["implementation_shortfall_bps"]) for e in filled]
    delay_vals = [float(e["delay_cost_bps"]) for e in filled if e.get("delay_cost_bps") is not None]
    slip_vals = [float(e["execution_slippage_bps"]) for e in filled if e.get("execution_slippage_bps") is not None]
    stat_vals = [float(e["statutory_cost_bps"]) for e in filled if e.get("statutory_cost_bps") is not None]
    surp_vals = [float(e["slippage_surprise_bps"]) for e in filled if e.get("slippage_surprise_bps") is not None]

    mean_is = sum(is_vals) / len(is_vals)
    mean_surp = (sum(surp_vals) / len(surp_vals)) if surp_vals else 0.0

    # Bounded feedback: if realized implementation slippage exceeds pre-trade expectations, raise EV hurdle [0..100 bps]
    shrink = len(filled) / (len(filled) + 5.0)
    hurdle_adder = round(max(0.0, min(100.0, mean_surp * shrink)), 2) if (len(surp_vals) >= 3 and mean_surp > 10.0) else 0.0
    drift_breached = bool(len(surp_vals) >= 5 and mean_surp > 75.0)

    by_algo: dict[str, list[dict[str, Any]]] = {}
    for e in filled:
        by_algo.setdefault(str(e.get("algo_strategy") or "NEXT_SESSION_OPEN"), []).append(e)
    algo_summary = {
        k: {
            "orders": len(v),
            "mean_is_bps": round(sum(float(x["implementation_shortfall_bps"]) for x in v) / len(v), 2),
            "mean_fill_rate_pct": round(sum(float(x.get("fill_rate_pct") or 0.0) for x in v) / len(v), 2),
        }
        for k, v in by_algo.items()
    }

    return {
        "total_orders": n_total,
        "filled_orders": n_filled,
        "avg_fill_rate_pct": round(avg_fill, 2),
        "mean_implementation_shortfall_bps": round(mean_is, 2),
        "mean_delay_cost_bps": round(sum(delay_vals) / len(delay_vals), 2) if delay_vals else None,
        "mean_execution_slippage_bps": round(sum(slip_vals) / len(slip_vals), 2) if slip_vals else None,
        "mean_statutory_cost_bps": round(sum(stat_vals) / len(stat_vals), 2) if stat_vals else None,
        "mean_slippage_surprise_bps": round(mean_surp, 2) if surp_vals else None,
        "tca_hurdle_adder_bps": hurdle_adder,
        "kill_switch_drift_breached": drift_breached,
        "algo_breakdown": algo_summary,
    }


class RuleBuilderEngine:
    """Declarative Bloomberg RBLD-style pre-trade compliance & smart-routing engine."""

    def __init__(self, rules: tuple[RuleSpec, ...] | list[RuleSpec] = DEFAULT_RBLD_RULES) -> None:
        self._rules: list[RuleSpec] = list(rules)

    def toggle_rule(self, rule_id: str, enabled: bool | None = None) -> dict[str, Any] | None:
        for idx, r in enumerate(self._rules):
            if r.rule_id == rule_id:
                new_state = (not r.enabled) if enabled is None else bool(enabled)
                self._rules[idx] = RuleSpec(
                    rule_id=r.rule_id,
                    name=r.name,
                    category=r.category,
                    enabled=new_state,
                    action_on_fail=r.action_on_fail,
                    threshold=r.threshold,
                    description=r.description,
                )
                return asdict(self._rules[idx])
        return None

    def get_rules(self) -> list[dict[str, Any]]:
        return [asdict(r) for r in self._rules]

    def evaluate_desk_kill_switch(
        self,
        *,
        engine_state: str = "RUNNING",
        drawdown_pct: float = 0.0,
        recon_diff_inr: float = 0.0,
        btca_drift_breached: bool = False,
        max_drawdown_pct: float = 10.0,
    ) -> tuple[bool, list[str]]:
        """Evaluate global desk kill-switch conditions (halting new BUY auto-routing)."""
        reasons: list[str] = []
        if str(engine_state).upper() == "SAFE_MODE":
            reasons.append("ENGINE_IN_SAFE_MODE")
        if float(drawdown_pct or 0.0) >= max_drawdown_pct:
            reasons.append(f"DRAWDOWN_LIMIT_BREACHED ({drawdown_pct:.2f}% >= {max_drawdown_pct:.2f}%)")
        if abs(float(recon_diff_inr or 0.0)) > 0.01:
            reasons.append(f"RECONCILIATION_DRIFT (INR {recon_diff_inr:.4f} > 0.01)")
        if btca_drift_breached:
            reasons.append("BTCA_SLIPPAGE_DRIFT_BREACHED (> 75 bps)")
        return (len(reasons) > 0), reasons

    def evaluate_order(
        self,
        *,
        symbol: str,
        side: str,
        arrival_price: float,
        order_notional_inr: float,
        nav_inr: float,
        adv_20d_inr: float,
        hl_range_pct: float,
        daily_vol_frac: float,
        fundamentals_status: str = "live",
        bar_valid: bool = True,
        price_discrepancy_pct: float = 0.0,
        sector_exposure_after_pct: float = 0.0,
        open_positions_count: int = 0,
        max_positions: int = 8,
        expected_net_alpha_bps: float | None = None,
        effective_hurdle_bps: float | None = None,
        statutory_cost_bps: float = 28.0,
        desk_halted: bool = False,
        desk_halt_reasons: list[str] | None = None,
        prefer_algo_wheel: bool = True,
        require_min_adv: bool = True,
    ) -> dict[str, Any]:
        """Evaluate an order against all enabled RBLD rules and select Pre-Trade TCA algo."""
        sd = str(side or "BUY").upper().strip()
        is_exit = sd == "SELL"
        pre_tca = compute_pre_trade_tca(
            side=sd,
            arrival_price=arrival_price,
            order_notional_inr=order_notional_inr,
            adv_20d_inr=adv_20d_inr,
            hl_range_pct=hl_range_pct,
            daily_vol_frac=daily_vol_frac,
            statutory_cost_bps=statutory_cost_bps,
            prefer_algo_wheel=prefer_algo_wheel,
        )

        rule_audit: list[dict[str, Any]] = []
        matched_rules: list[str] = []
        blocked_rules: list[dict[str, Any]] = []

        rule_map = {r.rule_id: r for r in self._rules}

        def _rec(rule_id: str, passed: bool, observed: Any, detail: str) -> None:
            spec = rule_map.get(rule_id)
            if spec is None or not spec.enabled:
                return
            verdict = "PASS" if passed else spec.action_on_fail
            entry = {
                "rule_id": rule_id,
                "name": spec.name,
                "category": spec.category,
                "passed": passed,
                "verdict": verdict,
                "observed": observed,
                "threshold": spec.threshold,
                "detail": detail,
            }
            rule_audit.append(entry)
            if passed:
                matched_rules.append(rule_id)
            else:
                blocked_rules.append(entry)

        # 1. Data Integrity (fail-closed)
        data_ok = (
            float(arrival_price or 0.0) > 0
            and bool(bar_valid)
            and str(fundamentals_status or "").lower() != "unavailable"
        )
        _rec("RBLD_01_DATA_INTEGRITY", data_ok, f"px={arrival_price},fund={fundamentals_status}", "Live price & fundamental integrity verified" if data_ok else "Missing price/bar/fundamentals")

        # 2. Desk Kill Switch (buys halted; protective sells still permitted!)
        if not is_exit:
            ks_ok = not desk_halted
            _rec(
                "RBLD_02_DESK_KILL_SWITCH",
                ks_ok,
                "HALTED" if desk_halted else "ACTIVE",
                "Desk active" if ks_ok else "; ".join(desk_halt_reasons or ["DESK_HALTED"]),
            )

        # 3. Price Band Drift
        band_lim = rule_map["RBLD_03_PRICE_BAND_DRIFT"].threshold
        band_ok = abs(float(price_discrepancy_pct or 0.0)) <= band_lim
        _rec(
            "RBLD_03_PRICE_BAND_DRIFT",
            band_ok,
            round(float(price_discrepancy_pct or 0.0), 3),
            f"Drift {price_discrepancy_pct:.2f}% <= {band_lim:.1f}%" if band_ok else f"Drift {price_discrepancy_pct:.2f}% > {band_lim:.1f}%",
        )

        # 4. Liquidity & ADV Cap (buys only — never trap an open exit in an illiquid decline)
        if not is_exit:
            min_adv = rule_map["RBLD_04_MIN_LIQUIDITY_ADV"].threshold if require_min_adv else 0.0
            adv_ok = float(adv_20d_inr or 0.0) >= min_adv and pre_tca["adv_participation_pct"] <= 1.5
            _rec(
                "RBLD_04_MIN_LIQUIDITY_ADV",
                adv_ok,
                f"adv_inr={adv_20d_inr:.0f},part={pre_tca['adv_participation_pct']:.3f}%",
                "Liquidity & %ADV cap satisfied" if adv_ok else "Below INR 2 Cr ADV floor or > 1.5% ADV participation",
            )

        # 5. Spread & Vol Guard (buys only)
        if not is_exit:
            spread_lim = rule_map["RBLD_05_SPREAD_VOL_LIMIT"].threshold
            sv_ok = float(hl_range_pct or 0.0) <= spread_lim and float(daily_vol_frac or 0.0) <= 0.06
            _rec(
                "RBLD_05_SPREAD_VOL_LIMIT",
                sv_ok,
                f"hl={hl_range_pct:.2f}%,vol={daily_vol_frac*100:.2f}%",
                "Spread & volatility within execution bounds" if sv_ok else "Extreme spread or volatility spike",
            )

        # 6. Pre-Trade AIM Concentration Caps (buys only)
        if not is_exit:
            single_pct = (float(order_notional_inr or 0.0) / max(1.0, float(nav_inr or 100_000.0))) * 100.0
            single_lim = rule_map["RBLD_06_CONCENTRATION_CAPS"].threshold
            conc_ok = (
                single_pct <= (single_lim + 0.05)
                and float(sector_exposure_after_pct or 0.0) <= 30.05
                and int(open_positions_count) < int(max_positions)
            )
            _rec(
                "RBLD_06_CONCENTRATION_CAPS",
                conc_ok,
                f"stock={single_pct:.2f}%,sector={sector_exposure_after_pct:.2f}%,slots={open_positions_count}/{max_positions}",
                "Concentration limits passed" if conc_ok else "Exceeds single-stock/sector/slot cap",
            )

        # 7. Pre-Trade TCA EV Hurdle (when provided for buys)
        if not is_exit and expected_net_alpha_bps is not None and effective_hurdle_bps is not None:
            ev_ok = float(expected_net_alpha_bps) >= float(effective_hurdle_bps)
            _rec(
                "RBLD_07_PRE_TRADE_TCA_EV",
                ev_ok,
                round(float(expected_net_alpha_bps), 2),
                f"EV {expected_net_alpha_bps:.1f} bps >= {effective_hurdle_bps:.1f} bps" if ev_ok else f"Below hurdle ({expected_net_alpha_bps:.1f} < {effective_hurdle_bps:.1f} bps)",
            )

        passed = len(blocked_rules) == 0
        lifecycle = ROUTED if passed else COMPLIANCE_BLOCKED
        return {
            "symbol": str(symbol).upper(),
            "side": sd,
            "lifecycle_state": lifecycle,
            "compliance_passed": passed,
            "selected_algo": pre_tca["selected_algo"] if passed else "BLOCKED",
            "matched_rules": matched_rules,
            "blocked_rules": blocked_rules,
            "rule_audit": rule_audit,
            "pre_trade_tca": pre_tca,
            "evaluated_at_utc": _utc_now_iso(),
        }


class EMSOrderBlotter:
    """In-session Order Management & Exception-First Blotter."""

    def __init__(self, *, desk_halted: bool = False, kill_switch_reasons: list[str] | None = None) -> None:
        self.desk_halted = bool(desk_halted)
        self.kill_switch_reasons = list(kill_switch_reasons or [])
        self.routed_orders: list[dict[str, Any]] = []
        self.blocked_orders: list[dict[str, Any]] = []
        self.exceptions: list[dict[str, Any]] = []

    def record_compliance_result(self, verdict: dict[str, Any], *, qty: int = 0) -> None:
        rec = {
            "symbol": verdict["symbol"],
            "side": verdict["side"],
            "qty": int(qty),
            "lifecycle_state": verdict["lifecycle_state"],
            "selected_algo": verdict["selected_algo"],
            "matched_rules": verdict["matched_rules"],
            "blocked_rules": verdict["blocked_rules"],
            "pre_trade_tca": verdict["pre_trade_tca"],
            "evaluated_at_utc": verdict["evaluated_at_utc"],
        }
        if verdict["compliance_passed"]:
            self.routed_orders.append(rec)
        else:
            self.blocked_orders.append(rec)
            self.exceptions.append({
                "exception_type": "COMPLIANCE_BLOCKED",
                "symbol": verdict["symbol"],
                "side": verdict["side"],
                "reasons": [b["detail"] for b in verdict["blocked_rules"]],
                "timestamp_utc": verdict["evaluated_at_utc"],
            })

    def record_data_exception(self, symbol: str, error: str, *, source: str = "LIVE_FEED") -> None:
        self.exceptions.append({
            "exception_type": "FAILED_DATA_UNAVAILABLE",
            "symbol": str(symbol).upper(),
            "source": source,
            "reasons": [str(error)[:240]],
            "timestamp_utc": _utc_now_iso(),
        })

    def to_payload(self, *, rules: list[dict[str, Any]] | None = None, btca_summary: dict[str, Any] | None = None) -> dict[str, Any]:
        return {
            "generated_at_utc": _utc_now_iso(),
            "desk_status": "HALTED_KILL_SWITCH" if self.desk_halted else "ACTIVE",
            "kill_switch_reasons": list(self.kill_switch_reasons),
            "counts": {
                "routed": len(self.routed_orders),
                "compliance_blocked": len(self.blocked_orders),
                "exceptions": len(self.exceptions),
            },
            "routed_orders": list(self.routed_orders),
            "compliance_blocked_orders": list(self.blocked_orders),
            "exception_queue": list(self.exceptions),
            "rbld_rules": list(rules or []),
            "post_trade_btca": dict(btca_summary or {}),
        }
