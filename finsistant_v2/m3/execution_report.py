"""Moreshwar 3.0 Prompt 8 — Execution Intelligence, Tariff Benchmarking & Security Audit Report.

Consumes `finsistant_v2.m3.execution_cost_engine` to produce:
1. `execution_quality_report.json`
2. `SECURITY_AND_TARIFF_VALIDATION.md`

Covering:
- Pre-Oct-2024 vs Post-Oct-2024 Zerodha/SEBI statutory tariff validation across CNC, MIS, NRML Futures.
- Round-trip bps & INR cost matrix across 5 order sizes (10k..50L) x 4 ADV tiers (5 Cr..500 Cr).
- Realistic OHLCV bar fill simulation benchmarks (MARKET, LIMIT, SL_M, SL_L, partial/unfilled, 5% ADV throttling).
- ZerodhaBrokerAdapter 6-stage architectural isolation & zero-credential-leak security audit.
"""

from __future__ import annotations

import ast
from datetime import datetime, timezone
import inspect
import json
from pathlib import Path
from typing import Any

from . import execution_cost_engine as ece
from .execution_cost_engine import (
    HistoricalTariffSchedule,
    ZerodhaBrokerAdapter,
    calculate_historical_indian_costs,
    compute_market_impact_and_slippage_bps,
    simulate_order_execution_bar,
)
from .schema import DEFAULT_M3_REPORTS_DIR

DEFAULT_EXECUTION_REPORTS_DIR = DEFAULT_M3_REPORTS_DIR / "execution"

ORDER_SIZES_INR: tuple[tuple[str, float], ...] = (
    ("INR 10k", 10_000.0),
    ("INR 50k", 50_000.0),
    ("INR 2L", 200_000.0),
    ("INR 10L", 1_000_000.0),
    ("INR 50L", 5_000_000.0),
)

ADV_TIERS_INR: tuple[tuple[str, float], ...] = (
    ("INR 5 Cr", 50_000_000.0),
    ("INR 25 Cr", 250_000_000.0),
    ("INR 100 Cr", 1_000_000_000.0),
    ("INR 500 Cr", 5_000_000_000.0),
)

FORBIDDEN_CREDENTIAL_MARKERS: tuple[str, ...] = (
    "api_key",
    "api_secret",
    "access_token",
    "enctoken",
    "totp",
    "kite_secret",
    "password",
)

SIX_STAGE_PIPELINE: tuple[dict[str, str], ...] = (
    {
        "stage": "1. Signal",
        "responsibility": "Causal cross-sectional alpha & regime-aware target scores at close t (no look-ahead).",
        "broker_access": "NONE",
    },
    {
        "stage": "2. PortfolioAllocation",
        "responsibility": "Weight sizing, sector/single-name caps, and turnover-buffered target weights.",
        "broker_access": "NONE",
    },
    {
        "stage": "3. RiskGate",
        "responsibility": "Drawdown circuit breakers, gross exposure cap, PIT liquidity & ADV participation checks.",
        "broker_access": "NONE",
    },
    {
        "stage": "4. PaperOrderPlanner",
        "responsibility": "Converts delta weights into validated OrderTicket objects (symbol, side, qty, order_type, product).",
        "broker_access": "OrderConstructor (pure schema validation only)",
    },
    {
        "stage": "5. ExecutionSimulator",
        "responsibility": "Next-bar open + spread + sqrt market impact + historical Zerodha/SEBI statutory cost simulation.",
        "broker_access": "OFFLINE_SIMULATION_ONLY",
    },
    {
        "stage": "6. BrokerAdapter",
        "responsibility": "Six-port ZerodhaBrokerAdapter enforcing OrderApprovalGate and live_execution_enabled=False lock.",
        "broker_access": "PAPER_SIMULATION / READ_ONLY_MASKED",
    },
)


def _round_trip_tariff_pair(
    *,
    product_type: str,
    notional: float,
    trade_date: str,
    slippage_bps: float = 0.0,
    tariff: HistoricalTariffSchedule = HistoricalTariffSchedule(),
) -> dict[str, Any]:
    buy = calculate_historical_indian_costs(
        product_type=product_type,
        side="BUY",
        raw_notional=notional,
        trade_date=trade_date,
        slippage_bps=slippage_bps,
        tariff=tariff,
    )
    sell = calculate_historical_indian_costs(
        product_type=product_type,
        side="SELL",
        raw_notional=notional,
        trade_date=trade_date,
        slippage_bps=slippage_bps,
        tariff=tariff,
    )
    rt_reg = round(buy["regulatory_total"] + sell["regulatory_total"], 2)
    rt_slip = round(buy["slippage_cost"] + sell["slippage_cost"], 2)
    rt_total = round(buy["total_cost"] + sell["total_cost"], 2)
    rt_reg_bps = round((rt_reg / max(1.0, notional)) * 10_000.0, 2)
    rt_total_bps = round((rt_total / max(1.0, notional)) * 10_000.0, 2)
    return {
        "product_type": product_type,
        "trade_date": trade_date,
        "notional_inr": notional,
        "slippage_bps_each_way": slippage_bps,
        "buy": buy,
        "sell": sell,
        "round_trip_regulatory_inr": rt_reg,
        "round_trip_slippage_inr": rt_slip,
        "round_trip_total_inr": rt_total,
        "round_trip_regulatory_bps": rt_reg_bps,
        "round_trip_total_bps": rt_total_bps,
    }


def benchmark_historical_tariffs(
    tariff: HistoricalTariffSchedule = HistoricalTariffSchedule(),
) -> dict[str, Any]:
    """Compare pre-Oct-2024 vs post-Oct-2024 statutory levies across CNC, MIS, and FUTURES."""
    regimes = (
        ("pre_oct_2024", "2024-09-30"),
        ("post_oct_2024", "2025-01-01"),
    )
    products = ("DELIVERY", "INTRADAY", "FUTURES")
    benchmark_notionals = (100_000.0, 1_000_000.0)

    comparisons: list[dict[str, Any]] = []
    for notional in benchmark_notionals:
        for pt in products:
            by_regime: dict[str, Any] = {}
            for rname, tdate in regimes:
                by_regime[rname] = _round_trip_tariff_pair(
                    product_type=pt,
                    notional=notional,
                    trade_date=tdate,
                    slippage_bps=0.0,
                    tariff=tariff,
                )
            pre = by_regime["pre_oct_2024"]
            post = by_regime["post_oct_2024"]
            delta_inr = round(post["round_trip_regulatory_inr"] - pre["round_trip_regulatory_inr"], 2)
            delta_bps = round(post["round_trip_regulatory_bps"] - pre["round_trip_regulatory_bps"], 2)
            comparisons.append({
                "product_type": pt,
                "notional_inr": notional,
                "pre_oct_2024": pre,
                "post_oct_2024": post,
                "delta_post_minus_pre_inr": delta_inr,
                "delta_post_minus_pre_bps": delta_bps,
            })

    # Verify DP charge symmetry & same-day deduplication on DELIVERY
    dp_buy = calculate_historical_indian_costs(
        product_type="DELIVERY", side="BUY", raw_notional=100_000.0, trade_date="2025-01-01", slippage_bps=0.0, tariff=tariff
    )
    dp_sell_first = calculate_historical_indian_costs(
        product_type="DELIVERY", side="SELL", raw_notional=100_000.0, trade_date="2025-01-01", slippage_bps=0.0, dp_charged_today=False, tariff=tariff
    )
    dp_sell_second = calculate_historical_indian_costs(
        product_type="DELIVERY", side="SELL", raw_notional=100_000.0, trade_date="2025-01-01", slippage_bps=0.0, dp_charged_today=True, tariff=tariff
    )

    return {
        "tariff_version": tariff.tariff_version,
        "schedule_rates": {
            "delivery_brokerage_mode": tariff.delivery_brokerage_flat_or_pct,
            "delivery_stt_buy_and_sell_pct": round(tariff.delivery_stt * 100.0, 5),
            "delivery_stamp_buy_pct": round(tariff.delivery_stamp_buy * 100.0, 5),
            "delivery_dp_sell_inr": tariff.delivery_dp_sell_inr,
            "intraday_brokerage": "min(INR 20, 0.03%) per order",
            "intraday_stt_sell_pct": round(tariff.intraday_stt_sell * 100.0, 5),
            "intraday_stamp_buy_pct": round(tariff.intraday_stamp_buy * 100.0, 5),
            "futures_brokerage": "min(INR 20, 0.03%) per order",
            "futures_stt_sell_pre_202410_pct": round(tariff.futures_stt_sell_pre_202410 * 100.0, 5),
            "futures_stt_sell_post_202410_pct": round(tariff.futures_stt_sell_post_202410 * 100.0, 5),
            "futures_exch_rate_pct": round(tariff.futures_exch_rate * 100.0, 5),
            "futures_stamp_buy_pct": round(tariff.futures_stamp_buy * 100.0, 5),
            "nse_cm_exch_pre_202410_pct": round(tariff.nse_cm_exch_pre_202410 * 100.0, 5),
            "nse_cm_exch_post_202410_pct": round(tariff.nse_cm_exch_post_202410 * 100.0, 5),
            "sebi_fee_rate_pct": round(tariff.sebi_fee_rate * 100.0, 5),
            "gst_rate_pct": round(tariff.gst_rate * 100.0, 2),
        },
        "dp_symmetry_audit": {
            "buy_delivery_dp_inr": dp_buy["dp_charge"],
            "sell_delivery_first_scrip_dp_inr": dp_sell_first["dp_charge"],
            "sell_delivery_subsequent_same_day_dp_inr": dp_sell_second["dp_charge"],
        },
        "regime_comparisons": comparisons,
    }


def benchmark_market_impact_matrix(
    tariff: HistoricalTariffSchedule = HistoricalTariffSchedule(),
) -> list[dict[str, Any]]:
    """Evaluate square-root impact + spread + statutory costs across order sizes x ADV tiers."""
    rows: list[dict[str, Any]] = []
    for size_label, order_inr in ORDER_SIZES_INR:
        for adv_label, adv_inr in ADV_TIERS_INR:
            imp = compute_market_impact_and_slippage_bps(
                order_notional_inr=order_inr,
                adv_20d_inr=adv_inr,
                daily_vol_frac=0.018,
                hl_spread_proxy_pct=2.0,
                tariff=tariff,
            )
            rt = _round_trip_tariff_pair(
                product_type="DELIVERY",
                notional=order_inr,
                trade_date="2025-01-01",
                slippage_bps=imp["total_slippage_bps"],
                tariff=tariff,
            )
            rows.append({
                "order_size_label": size_label,
                "order_notional_inr": order_inr,
                "adv_tier_label": adv_label,
                "adv_20d_inr": adv_inr,
                "participation_pct_adv": imp["participation_pct_adv"],
                "half_spread_bps_one_way": imp["half_spread_bps"],
                "market_impact_bps_one_way": imp["market_impact_bps"],
                "total_slippage_bps_one_way": imp["total_slippage_bps"],
                "round_trip_statutory_inr": rt["round_trip_regulatory_inr"],
                "round_trip_statutory_bps": rt["round_trip_regulatory_bps"],
                "round_trip_slippage_inr": rt["round_trip_slippage_inr"],
                "round_trip_dp_inr": rt["sell"]["dp_charge"],
                "round_trip_total_inr": rt["round_trip_total_inr"],
                "round_trip_total_bps": rt["round_trip_total_bps"],
                "exceeds_5pct_adv_cap": imp["participation_pct_adv"] > 5.0,
            })
    return rows


def run_execution_simulation_benchmarks() -> list[dict[str, Any]]:
    """Run deterministic bar-level fill simulations across all order types and edge cases."""
    std_bar = {"open": 1000.0, "high": 1020.0, "low": 985.0, "close": 1010.0, "volume": 100_000}
    low_vol_bar = {"open": 1000.0, "high": 1012.0, "low": 992.0, "close": 1005.0, "volume": 10_000}
    gap_down_bar = {"open": 940.0, "high": 955.0, "low": 930.0, "close": 948.0, "volume": 150_000}

    cases: list[tuple[str, dict[str, Any]]] = [
        (
            "MARKET_BUY_FULL",
            simulate_order_execution_bar(
                order_type="MARKET",
                side="BUY",
                req_qty=200,
                bar=std_bar,
                adv_20d_inr=100_000_000.0,
                max_bar_vol_participation=0.05,
                max_order_adv_pct=5.0,
                product_type="DELIVERY",
            ),
        ),
        (
            "LIMIT_BUY_PASSIVE_FILLED",
            simulate_order_execution_bar(
                order_type="LIMIT",
                side="BUY",
                req_qty=200,
                limit_price=990.0,
                bar=std_bar,
                adv_20d_inr=100_000_000.0,
                max_bar_vol_participation=0.05,
                max_order_adv_pct=5.0,
                product_type="DELIVERY",
            ),
        ),
        (
            "LIMIT_BUY_PARTIAL_VOL_CAP",
            simulate_order_execution_bar(
                order_type="LIMIT",
                side="BUY",
                req_qty=1000,
                limit_price=995.0,
                bar=low_vol_bar,
                adv_20d_inr=200_000_000.0,
                max_bar_vol_participation=0.05,
                max_order_adv_pct=5.0,
                product_type="DELIVERY",
            ),
        ),
        (
            "LIMIT_BUY_UNFILLED_NOT_REACHED",
            simulate_order_execution_bar(
                order_type="LIMIT",
                side="BUY",
                req_qty=200,
                limit_price=970.0,
                bar=std_bar,
                adv_20d_inr=100_000_000.0,
                max_bar_vol_participation=0.05,
                max_order_adv_pct=5.0,
                product_type="DELIVERY",
            ),
        ),
        (
            "SL_M_SELL_INTRADAY_TRIGGER",
            simulate_order_execution_bar(
                order_type="SL_M",
                side="SELL",
                req_qty=200,
                trigger_price=990.0,
                bar=std_bar,
                adv_20d_inr=100_000_000.0,
                max_bar_vol_participation=0.05,
                max_order_adv_pct=5.0,
                product_type="DELIVERY",
            ),
        ),
        (
            "SL_M_SELL_GAP_THROUGH_OPEN",
            simulate_order_execution_bar(
                order_type="SL_M",
                side="SELL",
                req_qty=200,
                trigger_price=970.0,
                bar=gap_down_bar,
                adv_20d_inr=100_000_000.0,
                max_bar_vol_participation=0.05,
                max_order_adv_pct=5.0,
                product_type="DELIVERY",
            ),
        ),
        (
            "SL_L_SELL_GAP_BEYOND_LIMIT_UNFILLED",
            simulate_order_execution_bar(
                order_type="SL_L",
                side="SELL",
                req_qty=200,
                trigger_price=970.0,
                limit_price=955.0,
                bar=gap_down_bar,
                adv_20d_inr=100_000_000.0,
                max_bar_vol_participation=0.05,
                max_order_adv_pct=5.0,
                product_type="DELIVERY",
            ),
        ),
        (
            "ADV_5PCT_THROTTLE_AND_VOL_CAP",
            simulate_order_execution_bar(
                order_type="MARKET",
                side="BUY",
                req_qty=5000,
                bar={"open": 1000.0, "high": 1015.0, "low": 990.0, "close": 1005.0, "volume": 40_000},
                adv_20d_inr=50_000_000.0,
                max_bar_vol_participation=0.05,
                max_order_adv_pct=5.0,
                product_type="DELIVERY",
            ),
        ),
        (
            "DUST_ORDER_REJECTED",
            simulate_order_execution_bar(
                order_type="MARKET",
                side="BUY",
                req_qty=1,
                bar=std_bar,
                min_viable_notional_inr=1500.0,
                product_type="DELIVERY",
            ),
        ),
    ]

    results: list[dict[str, Any]] = []
    for scenario_id, res in cases:
        results.append({
            "scenario_id": scenario_id,
            **res,
        })
    return results


def audit_broker_security_architecture() -> dict[str, Any]:
    """Audit ZerodhaBrokerAdapter 6-stage isolation, live-order lock, and zero-credential leakage."""
    adapter = ZerodhaBrokerAdapter()
    posture = adapter.get_security_posture()
    ticket = adapter.construct_order(
        symbol="RELIANCE",
        side="BUY",
        qty=10,
        order_type="LIMIT",
        product="CNC",
        price=2500.0,
    )

    # 1. Approval gate must block unapproved orders
    unapproved_blocked = False
    unapproved_error = ""
    try:
        adapter.submit_order(ticket)
    except PermissionError as exc:
        unapproved_blocked = True
        unapproved_error = str(exc)

    # 2. Approved order in default paper mode returns simulated payload with zero HTTP
    th = adapter.approve_order(ticket, approved_by="RISK_COMMITTEE_TEST")
    paper_receipt = adapter.submit_order(ticket)
    monitored = adapter.monitor_orders()

    # 3. Even if live_execution_enabled=True, paper_mode=False, live submission raises PermissionError
    live_adapter = ZerodhaBrokerAdapter(live_execution_enabled=True, paper_mode=False)
    live_ticket = live_adapter.construct_order(symbol="TCS", side="SELL", qty=5, order_type="MARKET", product="CNC")
    live_adapter.approve_order(live_ticket, approved_by="AUDIT_PROBE")
    live_blocked = False
    live_error = ""
    try:
        live_adapter.submit_order(live_ticket)
    except PermissionError as exc:
        live_blocked = True
        live_error = str(exc)

    # 4. Inspect execution_cost_engine.py AST imports for any HTTP / network / broker SDK dependency
    ece_source = Path(inspect.getfile(ece)).read_text(encoding="utf-8")
    tree = ast.parse(ece_source)
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module)
    network_imports = sorted({
        imp for imp in imports
        if any(net in imp.lower() for net in ("requests", "urllib", "http", "httpx", "aiohttp", "socket", "kiteconnect"))
    })

    # 5. Check serialized payloads for credential markers
    serialized_probe = json.dumps({"posture": posture, "paper_receipt": paper_receipt, "monitored": monitored}).lower()
    credential_leaks = [m for m in FORBIDDEN_CREDENTIAL_MARKERS if m in serialized_probe]

    return {
        "six_stage_pipeline": list(SIX_STAGE_PIPELINE),
        "default_posture": posture,
        "checks": {
            "default_live_execution_disabled": posture["live_execution_enabled"] is False,
            "default_paper_mode_enabled": posture["paper_mode"] is True,
            "unapproved_order_blocked": unapproved_blocked,
            "unapproved_error": unapproved_error,
            "approved_ticket_hash": th,
            "paper_submission_mode": paper_receipt.get("mode"),
            "paper_submission_order_id": paper_receipt.get("order_id"),
            "paper_monitor_count": len(monitored),
            "explicit_live_attempt_blocked": live_blocked,
            "live_blocked_error": live_error,
            "network_imports_in_engine": network_imports,
            "zero_network_imports_verified": len(network_imports) == 0,
            "credential_markers_found_in_payloads": credential_leaks,
            "zero_credential_leak_verified": len(credential_leaks) == 0,
        },
        "all_security_invariants_passed": all([
            posture["live_execution_enabled"] is False,
            posture["paper_mode"] is True,
            unapproved_blocked,
            paper_receipt.get("mode") == "PAPER_SIMULATION",
            live_blocked,
            len(network_imports) == 0,
            len(credential_leaks) == 0,
        ]),
    }


def _write_security_and_tariff_markdown(
    *,
    reports_dir: Path,
    report: dict[str, Any],
) -> None:
    tb = report["tariff_benchmarks"]
    rates = tb["schedule_rates"]
    dp = tb["dp_symmetry_audit"]
    sec = report["security_audit"]
    chk = sec["checks"]

    lines = [
        "# MORESHWAR 3.0 (PROMPT 8) — SECURITY, TARIFF VALIDATION & EXECUTION QUALITY REPORT",
        "",
        f"- **Report Version:** `{report['report_version']}`",
        f"- **Tariff Version:** `{tb['tariff_version']}`",
        f"- **Generated At (UTC):** `{report['generated_at']}`",
        f"- **All Security Invariants Passed:** **`{sec['all_security_invariants_passed']}`**",
        "",
        "## 1. Historical Zerodha / NSE / SEBI Statutory Tariff Schedule Validation",
        "",
        "| Segment / Levy | Pre-Oct-2024 (`<= 2024-09-30`) | Post-Oct-2024 (`>= 2024-10-01`) | Incidence |",
        "| :--- | :---: | :---: | :--- |",
        f"| Equity Delivery (`CNC`) Brokerage | `INR 0` (`{rates['delivery_brokerage_mode']}`) | `INR 0` (`{rates['delivery_brokerage_mode']}`) | Buy & Sell |",
        f"| Equity Delivery (`CNC`) STT | `{rates['delivery_stt_buy_and_sell_pct']:.4f}%` | `{rates['delivery_stt_buy_and_sell_pct']:.4f}%` | Buy & Sell |",
        f"| Equity Delivery (`CNC`) Stamp Duty | `{rates['delivery_stamp_buy_pct']:.4f}%` | `{rates['delivery_stamp_buy_pct']:.4f}%` | Buy only (`0` on Sell) |",
        f"| Equity Delivery (`CNC`) DP Charge | `INR {rates['delivery_dp_sell_inr']:.2f}` | `INR {rates['delivery_dp_sell_inr']:.2f}` | Sell >= 1 qty, once/scrip/day (`Buy = INR {dp['buy_delivery_dp_inr']:.2f}`) |",
        f"| NSE Cash Exchange Txn Fee (`CNC`/`MIS`) | `{rates['nse_cm_exch_pre_202410_pct']:.5f}%` | `{rates['nse_cm_exch_post_202410_pct']:.5f}%` | Buy & Sell |",
        f"| Equity Intraday (`MIS`) Brokerage | `{rates['intraday_brokerage']}` | `{rates['intraday_brokerage']}` | Per executed order |",
        f"| Equity Intraday (`MIS`) STT | `{rates['intraday_stt_sell_pct']:.4f}%` | `{rates['intraday_stt_sell_pct']:.4f}%` | Sell only |",
        f"| Equity Intraday (`MIS`) Stamp Duty | `{rates['intraday_stamp_buy_pct']:.4f}%` | `{rates['intraday_stamp_buy_pct']:.4f}%` | Buy only |",
        f"| Futures (`NRML`) Brokerage | `{rates['futures_brokerage']}` | `{rates['futures_brokerage']}` | Per executed order |",
        f"| Futures (`NRML`) STT | `{rates['futures_stt_sell_pre_202410_pct']:.4f}%` | `{rates['futures_stt_sell_post_202410_pct']:.4f}%` | Sell only |",
        f"| Futures (`NRML`) NSE Exchange Fee | `{rates['futures_exch_rate_pct']:.5f}%` | `{rates['futures_exch_rate_pct']:.5f}%` | Buy & Sell |",
        f"| Futures (`NRML`) Stamp Duty | `{rates['futures_stamp_buy_pct']:.4f}%` | `{rates['futures_stamp_buy_pct']:.4f}%` | Buy only |",
        f"| SEBI Turnover Fee | `{rates['sebi_fee_rate_pct']:.4f}%` | `{rates['sebi_fee_rate_pct']:.4f}%` | Buy & Sell |",
        f"| GST | `{rates['gst_rate_pct']:.1f}%` | `{rates['gst_rate_pct']:.1f}%` | On `(Brokerage + Exchange + SEBI)` |",
        "",
        "### Pre-Oct-2024 vs Post-Oct-2024 Round-Trip Statutory Benchmarks (Zero Slippage)",
        "",
        "| Product | Notional (INR) | Pre-Oct-24 RT (INR) | Pre-Oct-24 (bps) | Post-Oct-24 RT (INR) | Post-Oct-24 (bps) | Delta (INR) | Delta (bps) |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for c in tb["regime_comparisons"]:
        pre = c["pre_oct_2024"]
        post = c["post_oct_2024"]
        lines.append(
            f"| `{c['product_type']}` | {c['notional_inr']:,.0f} | "
            f"{pre['round_trip_regulatory_inr']:,.2f} | {pre['round_trip_regulatory_bps']:.2f} | "
            f"{post['round_trip_regulatory_inr']:,.2f} | {post['round_trip_regulatory_bps']:.2f} | "
            f"{c['delta_post_minus_pre_inr']:+.2f} | {c['delta_post_minus_pre_bps']:+.2f} |"
        )

    lines += [
        "",
        "## 2. Equity Delivery (`CNC`) Round-Trip Cost & Square-Root Market Impact Matrix",
        "",
        "| Order Size | ADV Tier | % ADV | Half-Spread (bps) | Impact (bps) | 1-Way Slip (bps) | RT Statutory (INR) | RT Slip (INR) | RT Total (INR) | RT Total (bps) |",
        "| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["market_impact_matrix"]:
        lines.append(
            f"| {row['order_size_label']} | {row['adv_tier_label']} | {row['participation_pct_adv']:.3f}% | "
            f"{row['half_spread_bps_one_way']:.2f} | {row['market_impact_bps_one_way']:.2f} | "
            f"{row['total_slippage_bps_one_way']:.2f} | {row['round_trip_statutory_inr']:,.2f} | "
            f"{row['round_trip_slippage_inr']:,.2f} | {row['round_trip_total_inr']:,.2f} | "
            f"**{row['round_trip_total_bps']:.2f}** |"
        )

    lines += [
        "",
        "## 3. Realistic OHLCV Bar Execution Simulation Scenarios",
        "",
        "| Scenario | Status | Reason | Req Qty | Filled Qty | Raw Price | Exec Fill Price | Slip (bps) | Gap Fill | Total Cost (INR) |",
        "| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | :---: | ---: |",
    ]
    for s in report["execution_simulations"]:
        raw_px = f"{s['raw_price']:.2f}" if s.get("raw_price") is not None else "—"
        fill_px = f"{s['fill_price']:.2f}" if s.get("fill_price") is not None else "—"
        slip = f"{s['slippage_bps']:.2f}" if s.get("slippage_bps") is not None else "—"
        tot_cost = f"{s['costs']['total_cost']:,.2f}" if s.get("costs") else "0.00"
        req_q = s.get("requested_qty", "—")
        gap = str(s.get("is_gap_fill", False))
        lines.append(
            f"| `{s['scenario_id']}` | `{s['status']}` | `{s['reason']}` | {req_q} | {s['filled_qty']} | "
            f"{raw_px} | {fill_px} | {slip} | `{gap}` | {tot_cost} |"
        )

    lines += [
        "",
        "## 4. Six-Stage Execution Isolation & `ZerodhaBrokerAdapter` Security Audit",
        "",
        "| Stage | Responsibility | Broker Capability |",
        "| :--- | :--- | :--- |",
    ]
    for st in sec["six_stage_pipeline"]:
        lines.append(f"| `{st['stage']}` | {st['responsibility']} | `{st['broker_access']}` |")

    lines += [
        "",
        "### Verified Security Controls",
        "",
        f"- **Default `live_execution_enabled`:** `{chk['default_live_execution_disabled']}` (value is `False`)",
        f"- **Default `paper_mode`:** `{chk['default_paper_mode_enabled']}` (value is `True`)",
        f"- **OrderApprovalGate blocks unapproved orders:** `{chk['unapproved_order_blocked']}` (`{chk['unapproved_error']}`)",
        f"- **Dry-run paper submission mode:** `{chk['paper_submission_mode']}` (order id `{chk['paper_submission_order_id']}`)",
        f"- **Explicit live-mode override blocked:** `{chk['explicit_live_attempt_blocked']}` (`{chk['live_blocked_error']}`)",
        f"- **Zero network/HTTP/SDK imports in engine:** `{chk['zero_network_imports_verified']}` (detected: `{chk['network_imports_in_engine']}`)",
        f"- **Zero credential leakage in payloads/logs:** `{chk['zero_credential_leak_verified']}` (markers matched: `{chk['credential_markers_found_in_payloads']}`)",
    ]

    (reports_dir / "SECURITY_AND_TARIFF_VALIDATION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_execution_and_security_report(
    reports_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Run all Prompt 8 tariff, impact, simulation, and security benchmarks and write artifacts."""
    out_dir = Path(reports_dir) if reports_dir is not None else DEFAULT_EXECUTION_REPORTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    tariff = HistoricalTariffSchedule()
    report: dict[str, Any] = {
        "report_version": "3.0.0-p8-execution-intelligence",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "tariff_benchmarks": benchmark_historical_tariffs(tariff),
        "market_impact_matrix": benchmark_market_impact_matrix(tariff),
        "execution_simulations": run_execution_simulation_benchmarks(),
        "security_audit": audit_broker_security_architecture(),
    }

    (out_dir / "execution_quality_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_security_and_tariff_markdown(reports_dir=out_dir, report=report)

    # Final post-write scan to guarantee zero credential strings entered generated reports
    for fname in ("execution_quality_report.json", "SECURITY_AND_TARIFF_VALIDATION.md"):
        txt = (out_dir / fname).read_text(encoding="utf-8").lower()
        leaked = [m for m in FORBIDDEN_CREDENTIAL_MARKERS if m in txt]
        if leaked:
            raise RuntimeError(f"Credential marker(s) {leaked} detected in {fname}")

    return report


if __name__ == "__main__":
    generate_execution_and_security_report()
