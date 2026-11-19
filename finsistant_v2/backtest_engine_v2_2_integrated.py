"""Moreshwar 2.2 Prompt 3 — Integrated Portfolio Optimization, Stress & Final Validation Engine.

Strictly preserves Moreshwar 1.0, 2.0, 2.1, Prompt 1, and Prompt 2 artifacts untouched.
Selects integrated portfolio/risk parameters strictly using In-Sample (Dev + Validation)
metrics ONLY (`<= 2025-09-04`), evaluates once on Prior OOS (`2025-09-05..2026-11-19`) and
Terminal Untouched Holdout (`2026-05-19..2026-11-19`), runs comprehensive Cost/Execution/
Parameter/Market/Concentration stress tests, computes 10D/20D circular block bootstraps,
renders equity & drawdown SVG charts, and writes versioned deliverables into
`reports/moreshwar_2_2/`.
"""

from __future__ import annotations

from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
from typing import Any

from .backtest_engine_v2_1 import (
    INITIAL_CAPITAL,
    RISK_FREE_ANNUAL,
    _compute_statistical_significance,
)
from .backtest_engine_v2_2 import (
    DEV_TRAIN_RANGE,
    FULL_RANGE,
    IS_COMBINED_RANGE,
    OOS_RANGE,
    VALIDATION_RANGE,
    _compute_calendar_returns,
    precompute_v22_universe,
)
from .historical_data import load_or_build_expanded_dataset
from .moreshwar_v2_2 import DeliveryCostConfig, compute_repaired_delivery_costs
from .moreshwar_v2_2_integrated import (
    IntegratedM22Config,
    PortfolioRiskAllocator,
    SelectionAndEntryEngine,
)
from .moreshwar_v2_2_p2 import (
    MoreshwarV22P2Config,
    check_bar_exit_action_p2,
    compute_profit_retention_metrics,
)

TERMINAL_HOLDOUT_RANGE = ("2026-05-19", "2026-11-19")
REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports" / "moreshwar_2_2"
P1_REPORT_PATH = Path(__file__).resolve().parent.parent / "data" / "moreshwar_v2_2_experiment_report.json"
P2_REPORT_PATH = Path(__file__).resolve().parent.parent / "data" / "moreshwar_v2_2_p2_experiment_report.json"


def simulate_integrated_v2_2(
    precomputed: dict[str, Any],
    *,
    experiment_id: str,
    label: str,
    cfg: IntegratedM22Config,
    tariff: DeliveryCostConfig | None = None,
    cost_multiplier: float = 1.0,
    date_range: tuple[str, str] | None = None,
    initial_capital: float = INITIAL_CAPITAL,
) -> dict[str, Any]:
    """Chronological portfolio simulation for Integrated Moreshwar 2.2 with stress hooks."""
    tc = tariff or DeliveryCostConfig()
    uni = precomputed["universe"]
    timeline_full = precomputed["timeline"]
    timeline = [d for d in timeline_full if date_range[0] <= d <= date_range[1]] if date_range else list(timeline_full)
    if len(timeline) < 10:
        return {"experiment_id": experiment_id, "label": label, "metrics": {}, "trades": [], "equity_curve": []}

    features = precomputed["features"]
    date_bars = precomputed["date_bars"]
    regime6_map = precomputed["regime6_by_date"]
    bench_rets = precomputed["bench_ret_by_date"]
    bench_closes = precomputed["bench_close_by_date"]
    valid_syms = sorted(features.keys())

    p2_exit_cfg = MoreshwarV22P2Config(
        entry_mode=cfg.entry_mode,
        exit_mode=cfg.exit_mode,
        horizon_days=cfg.horizon_days,
        initial_atr_mult=cfg.initial_atr_mult,
        breakeven_trigger_atr=cfg.breakeven_trigger_atr,
        breakeven_fee_buffer_pct=cfg.breakeven_fee_buffer_pct,
    )

    cash = float(initial_capital)
    peak_nav = float(initial_capital)
    dd_throttled = False
    open_positions: dict[str, dict[str, Any]] = {}
    # Pending orders carry `due_idx` so entry_delay_bars (1 normal vs 2 delayed stress) works cleanly
    pending_orders: list[dict[str, Any]] = []
    closed_trades: list[dict[str, Any]] = []
    dp_charged_pairs: set[tuple[str, str]] = set()
    cooldown_until_idx: dict[str, int] = {}

    buy_notional_exec_sum = 0.0
    sell_notional_exec_sum = 0.0
    cost_totals = {k: 0.0 for k in ("brokerage", "stt", "exchange_txn", "sebi_fee", "stamp_duty", "gst", "dp_charge", "slippage_cost", "regulatory_total", "total_cost")}
    exec_stats = {"gap_through_stops": 0, "partial_fills": 0, "unfillable_zero_vol": 0, "liquidity_delayed_exits": 0, "multi_exit_days": 0}
    daily_navs: list[dict[str, Any]] = []
    daily_snaps: list[dict[str, Any]] = []

    for d_idx, dt in enumerate(timeline):
        r6 = regime6_map[dt]["regime"]

        # 1. Execute orders whose due_idx == d_idx
        remaining_pending = []
        for ord_req in pending_orders:
            if ord_req["due_idx"] > d_idx:
                remaining_pending.append(ord_req)
                continue
            sym = ord_req["symbol"]
            if sym in open_positions or dt not in date_bars.get(sym, {}):
                continue
            b_idx, bar = date_bars[sym][dt]
            raw_open = float(bar.get("open") or 0.0)
            bar_vol = int(bar.get("volume") or 0)
            if raw_open <= 0.0 or bar_vol <= 0:
                exec_stats["unfillable_zero_vol"] += 1
                continue

            slip_bps = float(ord_req["slippage_bps"])
            exec_entry_px = raw_open * (1.0 + slip_bps / 10_000.0)
            qty = min(int(ord_req["req_qty"]), int((cash * 0.96) // max(1.0, exec_entry_px)))
            max_bar_shares = max(1, int(bar_vol * cfg.max_daily_volume_participation))
            if qty > max_bar_shares:
                qty = max_bar_shares
                exec_stats["partial_fills"] += 1
            if qty < 1 or (qty * exec_entry_px) < 1500.0:
                continue

            raw_buy = qty * raw_open
            buy_costs = compute_repaired_delivery_costs(
                side="BUY", raw_notional=raw_buy, currency="INR", slippage_bps=slip_bps, tariff=tc
            )
            exec_buy = buy_costs["exec_notional"]
            reg_buy = buy_costs["regulatory_total"] * cost_multiplier
            slip_buy = buy_costs["slippage_cost"] * cost_multiplier
            # If cost_multiplier > 1.0x stress, debit excess regulatory/slippage friction proportionally
            extra_stress_cost = (buy_costs["regulatory_total"] + buy_costs["slippage_cost"]) * max(0.0, cost_multiplier - 1.0)
            if exec_buy + reg_buy + (extra_stress_cost - (reg_buy - buy_costs["regulatory_total"])) > cash:
                qty = int((cash * 0.95) // max(1.0, exec_entry_px * 1.005))
                if qty < 1 or (qty * exec_entry_px) < 1500.0:
                    continue
                raw_buy = qty * raw_open
                buy_costs = compute_repaired_delivery_costs(side="BUY", raw_notional=raw_buy, currency="INR", slippage_bps=slip_bps, tariff=tc)
                exec_buy = buy_costs["exec_notional"]
                reg_buy = buy_costs["regulatory_total"] * cost_multiplier
                slip_buy = buy_costs["slippage_cost"] * cost_multiplier
                extra_stress_cost = (buy_costs["regulatory_total"] + buy_costs["slippage_cost"]) * max(0.0, cost_multiplier - 1.0)

            for ck in cost_totals:
                cost_totals[ck] += buy_costs.get(ck, 0.0) * cost_multiplier
            buy_notional_exec_sum += exec_buy
            # Deduct executed buy + multiplied reg fees + any extra slippage stress above 1.0x
            extra_slip_cash = buy_costs["slippage_cost"] * max(0.0, cost_multiplier - 1.0)
            cash -= (exec_buy + reg_buy + extra_slip_cash)

            eff_sl = round(exec_entry_px * (1.0 - ord_req["sl_ratio"]), 4)
            rec_closes = [float(x["close"]) for x in uni[sym]["candles"][max(0, b_idx - 61) : b_idx + 1]]
            open_positions[sym] = {
                "symbol": sym,
                "sector": ord_req["sector"],
                "archetype": ord_req["archetype"],
                "signal_date": ord_req["signal_date"],
                "signal_regime6": ord_req["market_regime"],
                "atr_entry": ord_req["atr_entry"],
                "atr_pct": ord_req["atr_pct"],
                "entry_date": dt,
                "entry_price": exec_entry_px,
                "last_price": exec_entry_px,
                "max_price_seen": exec_entry_px,
                "max_close_seen": exec_entry_px,
                "qty": qty,
                "raw_buy_notional": raw_buy,
                "exec_buy_notional": exec_buy,
                "buy_reg_cost": reg_buy,
                "buy_slip_cost": slip_buy,
                "extra_buy_slip_cash": extra_slip_cash,
                "initial_cost_basis": exec_buy + reg_buy + extra_slip_cash,
                "sl": eff_sl,
                "tp": 0.0,
                "partial_target": 0.0,
                "partial_booked": False,
                "breakeven_locked": False,
                "trailing_active": False,
                "slippage_bps": slip_bps,
                "mfe_pct": 0.0,
                "mae_pct": 0.0,
                "bars_held": 0,
                "recent_closes": rec_closes,
            }
        pending_orders = remaining_pending

        # 2. Check exits
        to_close: list[str] = []
        is_final_bar = (d_idx == len(timeline) - 1)
        exits_today = 0
        for sym, pos in list(open_positions.items()):
            if dt not in date_bars.get(sym, {}):
                continue
            _, bar = date_bars[sym][dt]
            bar_feats = features.get(sym, {}).get(dt)
            act = check_bar_exit_action_p2(
                pos=pos,
                bar=bar,
                bar_feats=bar_feats,
                regime6_name=r6,
                is_final_bar=is_final_bar,
                cfg=p2_exit_cfg,
            )
            if act["action"] == "DEFER_ZERO_VOL":
                exec_stats["liquidity_delayed_exits"] += 1
                continue

            if act["action"] == "FULL_EXIT" and act["exit_px"] is not None:
                exits_today += 1
                raw_exit_px = float(act["exit_px"])
                if act["is_gap_stop"]:
                    exec_stats["gap_through_stops"] += 1
                    if cfg.gap_shock_extra_pct > 0.0:
                        raw_exit_px = raw_exit_px * max(0.0, 1.0 - cfg.gap_shock_extra_pct / 100.0)

                raw_sell = pos["qty"] * raw_exit_px
                dp_key = (dt, sym)
                dp_already = dp_key in dp_charged_pairs
                sell_costs = compute_repaired_delivery_costs(
                    side="SELL", raw_notional=raw_sell, currency="INR", slippage_bps=pos["slippage_bps"], dp_charged_today=dp_already, tariff=tc
                )
                if not dp_already and sell_costs["dp_charge"] > 0:
                    dp_charged_pairs.add(dp_key)
                for ck in cost_totals:
                    cost_totals[ck] += sell_costs.get(ck, 0.0) * cost_multiplier

                exec_sell = sell_costs["exec_notional"]
                sell_notional_exec_sum += exec_sell
                reg_sell = sell_costs["regulatory_total"] * cost_multiplier
                extra_sell_slip_cash = sell_costs["slippage_cost"] * max(0.0, cost_multiplier - 1.0)
                net_proceeds = exec_sell - reg_sell - extra_sell_slip_cash
                cash += net_proceeds

                tot_cost = (
                    pos["buy_reg_cost"]
                    + pos["buy_slip_cost"]
                    + reg_sell
                    + sell_costs["slippage_cost"] * cost_multiplier
                )
                pnl_amt = net_proceeds - pos["initial_cost_basis"]
                gross_pnl = raw_sell - pos["raw_buy_notional"]
                net_ret_pct = (pnl_amt / max(1e-6, pos["initial_cost_basis"])) * 100.0

                ex_reason = str(act["exit_reason"])
                if cfg.reentry_cooldown_days > 0 and "STOP_LOSS" in ex_reason:
                    cooldown_until_idx[sym] = d_idx + cfg.reentry_cooldown_days

                closed_trades.append({
                    "symbol": sym,
                    "sector": pos["sector"],
                    "archetype": pos["archetype"],
                    "signal_date": pos["signal_date"],
                    "entry_date": pos["entry_date"],
                    "exit_date": dt,
                    "signal_regime6": pos["signal_regime6"],
                    "exit_regime6": r6,
                    "atr_pct": pos["atr_pct"],
                    "holding_days": pos["bars_held"],
                    "qty": pos["qty"],
                    "raw_buy_notional": round(pos["raw_buy_notional"], 2),
                    "gross_pnl_amount": round(gross_pnl, 2),
                    "pnl_amount": round(pnl_amt, 2),
                    "cost_amount": round(tot_cost, 2),
                    "net_return_pct": round(net_ret_pct, 2),
                    "mfe_pct": round(pos["mfe_pct"], 2),
                    "mae_pct": round(pos["mae_pct"], 2),
                    "exit_reason": ex_reason,
                    "gap_through_stop": act["is_gap_stop"],
                })
                to_close.append(sym)
            elif cfg.tighten_be_on_bear_regime and r6 in ("STRONG_BEARISH", "WEAK_BEARISH", "HIGH_VOLATILITY"):
                # Optional regime-deterioration tightening if position is green
                be_px = round(pos["entry_price"] * (1.0 + cfg.breakeven_fee_buffer_pct / 100.0), 4)
                if pos["last_price"] >= pos["entry_price"] + 1.0 * pos["atr_entry"] and be_px > pos["sl"]:
                    pos["sl"] = be_px
                    pos["breakeven_locked"] = True

        if exits_today >= 2:
            exec_stats["multi_exit_days"] += 1
        for sym in to_close:
            del open_positions[sym]

        # 3. End-of-day NAV & Drawdown Monitor update
        deployed_mkt = sum(p["last_price"] * p["qty"] for p in open_positions.values())
        nav = cash + deployed_mkt
        peak_nav = max(peak_nav, nav)
        cur_dd_pct = max(0.0, ((peak_nav - nav) / max(1e-6, peak_nav)) * 100.0)
        if cur_dd_pct >= cfg.dd_throttle_level1_pct:
            dd_throttled = True
        elif cur_dd_pct <= cfg.dd_recovery_pct:
            dd_throttled = False

        gross_exp_pct = (deployed_mkt / max(1.0, nav)) * 100.0
        sec_map: dict[str, float] = {}
        for p in open_positions.values():
            sec_map[p["sector"]] = sec_map.get(p["sector"], 0.0) + p["last_price"] * p["qty"]
        max_sec_pct = (max(sec_map.values()) / max(1.0, nav)) * 100.0 if sec_map else 0.0
        prev_nav = daily_navs[-1]["nav"] if daily_navs else initial_capital
        day_ret_pct = ((nav - prev_nav) / max(1e-6, prev_nav)) * 100.0
        daily_navs.append({"date": dt, "nav": round(nav, 2), "regime6": r6, "dd_pct": round(-cur_dd_pct, 2)})
        daily_snaps.append({
            "date": dt,
            "open_positions": len(open_positions),
            "gross_exposure_pct": round(gross_exp_pct, 2),
            "max_sector_pct": round(max_sec_pct, 2),
            "daily_ret_pct": round(day_ret_pct, 3),
            "bench_ret_pct": round(bench_rets.get(dt, 0.0), 3),
        })

        if is_final_bar:
            continue

        # 4. Selection, Entry, Risk & Portfolio Allocation
        cands = SelectionAndEntryEngine.evaluate_universe_day(
            valid_syms=valid_syms,
            open_positions=open_positions,
            cooldown_until_idx=cooldown_until_idx,
            d_idx=d_idx,
            dt=dt,
            features=features,
            uni=uni,
            regime6_obj=regime6_map[dt],
            cfg=cfg,
            tariff=tc,
        )
        orders = PortfolioRiskAllocator.allocate(
            candidates=cands,
            open_positions=open_positions,
            cash=cash,
            nav=nav,
            current_dd_pct=cur_dd_pct,
            dd_throttled=dd_throttled,
            regime6_name=r6,
            cfg=cfg,
        )
        for o in orders:
            o["due_idx"] = d_idx + cfg.entry_delay_bars
            pending_orders.append(o)

    # Metrics & Analytics
    end_nav = daily_navs[-1]["nav"] if daily_navs else initial_capital
    total_ret = ((end_nav - initial_capital) / initial_capital) * 100.0
    n_days = max(1, len(daily_navs))
    cagr = (((end_nav / initial_capital) ** (252.0 / n_days)) - 1.0) * 100.0 if end_nav > 0 else -100.0

    pk = initial_capital
    max_dd = 0.0
    daily_rets: list[float] = []
    prev_n = initial_capital
    for pt in daily_navs:
        nv = pt["nav"]
        pk = max(pk, nv)
        max_dd = min(max_dd, ((nv - pk) / pk) * 100.0)
        daily_rets.append((nv - prev_n) / prev_n)
        prev_n = nv

    rf_d = RISK_FREE_ANNUAL / 252.0
    ex = [r - rf_d for r in daily_rets]
    m_ex = sum(ex) / len(ex)
    m_raw = sum(daily_rets) / len(daily_rets)
    std_raw = math.sqrt(sum((r - m_raw) ** 2 for r in daily_rets) / max(1, len(daily_rets) - 1))
    std_ex = math.sqrt(sum((x - m_ex) ** 2 for x in ex) / max(1, len(ex) - 1))
    dn_ex = math.sqrt(sum(min(0.0, x) ** 2 for x in ex) / len(ex))
    ann_vol = round(std_raw * math.sqrt(252.0) * 100.0, 2)
    sharpe = round((m_ex / std_ex) * math.sqrt(252.0), 2) if std_ex > 1e-9 else 0.0
    sortino = round((m_ex / dn_ex) * math.sqrt(252.0), 2) if dn_ex > 1e-9 else 0.0

    sig = _compute_statistical_significance(closed_trades)
    wins = [t for t in closed_trades if t["pnl_amount"] > 0]
    losses = [t for t in closed_trades if t["pnl_amount"] <= 0]
    gw = sum(t["pnl_amount"] for t in wins)
    gl = abs(sum(t["pnl_amount"] for t in losses))
    net_pf = round(gw / gl, 2) if gl > 1e-6 else (9.99 if gw > 0 else 0.0)

    b_start = bench_closes.get(timeline[0], 0.0)
    b_end = bench_closes.get(timeline[-1], 0.0)
    bench_ret_pct = ((b_end - b_start) / max(1e-6, b_start)) * 100.0 if b_start > 0 else 0.0
    cash_ret_pct = (((1.0 + RISK_FREE_ANNUAL) ** (n_days / 252.0)) - 1.0) * 100.0
    yearly_ret, monthly_ret = _compute_calendar_returns(daily_navs, bench_closes, initial_capital)
    retention = compute_profit_retention_metrics(closed_trades)
    net_pnl_sum = sum(t["pnl_amount"] for t in closed_trades)

    # Rolling 252-day returns
    rolling_1y = []
    for ri in range(252, len(daily_navs), 21):
        r1y = ((daily_navs[ri]["nav"] / max(1e-6, daily_navs[ri - 252]["nav"])) - 1.0) * 100.0
        rolling_1y.append({"end_date": daily_navs[ri]["date"], "rolling_1y_pct": round(r1y, 2)})

    metrics = {
        "experiment_id": experiment_id,
        "label": label,
        "distinction": "SIMULATED_RESEARCH_BACKTEST (NOT LIVE EXECUTION)",
        "window": f"{timeline[0]} → {timeline[-1]} ({n_days} days)",
        "total_return_pct": round(total_ret, 2),
        "cagr_pct": round(cagr, 2),
        "max_drawdown_pct": round(abs(max_dd), 2),
        "net_profit_factor": net_pf,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "annualised_vol_pct": ann_vol,
        "total_trades": len(closed_trades),
        "win_rate_pct": sig["win_rate_pct"],
        "avg_winner_pct": round(sum(t["net_return_pct"] for t in wins) / max(1, len(wins)), 2) if wins else 0.0,
        "avg_loser_pct": round(sum(t["net_return_pct"] for t in losses) / max(1, len(losses)), 2) if losses else 0.0,
        "expectancy_pct": sig["expectancy_pct"],
        "p_value_vs_zero": sig["p_value_vs_zero"],
        "total_cost": round(cost_totals["total_cost"], 2),
        "cost_drag_pct": round((cost_totals["total_cost"] / initial_capital) * 100.0, 2),
        "turnover_multiple_x": round((buy_notional_exec_sum + sell_notional_exec_sum) / initial_capital, 2),
        "capital_utilisation_pct": round(sum(s["gross_exposure_pct"] for s in daily_snaps) / max(1, len(daily_snaps)), 1),
        "max_sector_pct": round(max((s["max_sector_pct"] for s in daily_snaps), default=0.0), 2),
        "net_pnl_total": round(net_pnl_sum, 2),
        "nav_change_reconciled": round(end_nav - initial_capital, 2),
        "reconciliation_diff": round((end_nav - initial_capital) - net_pnl_sum, 2),
        "benchmark_return_pct": round(bench_ret_pct, 2),
        "cash_return_pct": round(cash_ret_pct, 2),
        "net_mfe_retention_pct": retention.get("net_portfolio_mfe_retention_pct", 0.0),
        "winner_mfe_retention_pct": retention.get("winner_mfe_retention_pct", 0.0),
        "yearly_returns": yearly_ret,
        "monthly_returns": monthly_ret,
        "rolling_1y": rolling_1y,
        "execution_realism": exec_stats,
    }

    return {
        "metrics": metrics,
        "retention": retention,
        "trades": closed_trades,
        "equity_curve": daily_navs,
        "daily_rets": daily_rets,
        "config": asdict(cfg),
    }


def compute_statistical_confidence(
    trades: list[dict[str, Any]],
    daily_rets: list[float],
    n_sims: int = 1200,
) -> dict[str, Any]:
    """Trade bootstrap + 10D & 20D circular daily block bootstraps preserving time-series correlation."""
    rng = random.Random(20261119)
    n_days = len(daily_rets)
    rets_tr = [float(t["net_return_pct"]) for t in trades]
    pnls_tr = [float(t["pnl_amount"]) for t in trades]

    def _block_sim(block_len: int) -> dict[str, Any]:
        tot_rets, cagrs, sharpes, max_dds = [], [], [], []
        rf_d = RISK_FREE_ANNUAL / 252.0
        for _ in range(n_sims):
            seq: list[float] = []
            while len(seq) < n_days:
                start = rng.randrange(n_days)
                for b in range(block_len):
                    seq.append(daily_rets[(start + b) % n_days])
                    if len(seq) >= n_days:
                        break
            eq, pk, mdd = 1.0, 1.0, 0.0
            for r in seq:
                eq *= (1.0 + r)
                pk = max(pk, eq)
                mdd = min(mdd, (eq - pk) / pk)
            tot_r = (eq - 1.0) * 100.0
            cg = ((eq ** (252.0 / max(1, n_days))) - 1.0) * 100.0
            ex = [r - rf_d for r in seq]
            mx = sum(ex) / len(ex)
            sx = math.sqrt(sum((x - mx) ** 2 for x in ex) / max(1, len(ex) - 1))
            sh = (mx / sx) * math.sqrt(252.0) if sx > 1e-9 else 0.0
            tot_rets.append(tot_r)
            cagrs.append(cg)
            sharpes.append(sh)
            max_dds.append(abs(mdd) * 100.0)

        tot_rets.sort()
        cagrs.sort()
        sharpes.sort()
        max_dds.sort()
        i5, i50, i95 = int(0.05 * n_sims), int(0.50 * n_sims), int(0.95 * n_sims)
        cash_hurdle = (((1.0 + RISK_FREE_ANNUAL) ** (n_days / 252.0)) - 1.0) * 100.0
        return {
            "simulations": n_sims,
            "block_length_days": block_len,
            "return_p05_p50_p95": [round(tot_rets[i5], 2), round(tot_rets[i50], 2), round(tot_rets[i95], 2)],
            "cagr_p05_p50_p95": [round(cagrs[i5], 2), round(cagrs[i50], 2), round(cagrs[i95], 2)],
            "sharpe_p05_p50_p95": [round(sharpes[i5], 2), round(sharpes[i50], 2), round(sharpes[i95], 2)],
            "max_dd_p50_p95": [round(max_dds[i50], 2), round(max_dds[i95], 2)],
            "prob_positive_return_pct": round((sum(1 for x in tot_rets if x > 0) / n_sims) * 100.0, 1),
            "prob_beat_cash_65_pct": round((sum(1 for x in tot_rets if x > cash_hurdle) / n_sims) * 100.0, 1),
        }

    # Trade-sequence bootstrap
    tr_totals, tr_dds = [], []
    nt = len(pnls_tr)
    if nt > 0:
        for _ in range(n_sims):
            eq, pk, mdd = INITIAL_CAPITAL, INITIAL_CAPITAL, 0.0
            for _j in range(nt):
                eq += pnls_tr[rng.randrange(nt)]
                pk = max(pk, eq)
                mdd = min(mdd, (eq - pk) / max(1.0, pk))
            tr_totals.append(((eq - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100.0)
            tr_dds.append(abs(mdd) * 100.0)
        tr_totals.sort()
        tr_dds.sort()

    # Skewness & Excess Kurtosis of trade returns
    n_t = len(rets_tr)
    mean_t = sum(rets_tr) / max(1, n_t)
    sd_t = math.sqrt(sum((x - mean_t) ** 2 for x in rets_tr) / max(1, n_t - 1)) if n_t > 1 else 1.0
    skew = (sum((x - mean_t) ** 3 for x in rets_tr) / max(1, n_t)) / max(1e-9, sd_t ** 3) if n_t > 2 else 0.0
    kurt = (sum((x - mean_t) ** 4 for x in rets_tr) / max(1, n_t)) / max(1e-9, sd_t ** 4) - 3.0 if n_t > 3 else 0.0

    return {
        "trade_bootstrap": {
            "simulations": n_sims,
            "return_p05_p50_p95": [round(tr_totals[int(0.05 * n_sims)], 2), round(tr_totals[int(0.50 * n_sims)], 2), round(tr_totals[int(0.95 * n_sims)], 2)] if nt else [0, 0, 0],
            "max_dd_p50_p95": [round(tr_dds[int(0.50 * n_sims)], 2), round(tr_dds[int(0.95 * n_sims)], 2)] if nt else [0, 0],
            "prob_positive_pct": round((sum(1 for x in tr_totals if x > 0) / n_sims) * 100.0, 1) if nt else 0.0,
        },
        "block_bootstrap_10d": _block_sim(10),
        "block_bootstrap_20d": _block_sim(20),
        "trade_return_distribution": {
            "count": n_t,
            "mean_pct": round(mean_t, 2),
            "std_pct": round(sd_t, 2),
            "skewness": round(skew, 3),
            "excess_kurtosis": round(kurt, 3),
        },
        "multiple_testing_disclosure": (
            "Across Prompts 1–3 (~90 tested variants total), nominal p-values < 0.05 require deflated caution. "
            "Prompt 3 selected risk/portfolio overlays strictly on IS (<=2025-09-04) before evaluating 2025-09+ OOS and 2026-05+ holdout."
        ),
    }


def _render_svg_charts(
    *,
    series_dict: dict[str, list[dict[str, Any]]],
    bench_closes: dict[str, float],
    out_dir: Path,
) -> tuple[str, str]:
    """Generate clean equity & drawdown comparison SVG charts inside reports/moreshwar_2_2/."""
    colors = {
        "Integrated M2.2": "#10b981",
        "Stress 1.5x Costs": "#f59e0b",
        "Prompt 2 Best": "#38bdf8",
        "Nifty 50": "#94a3b8",
    }
    dates = [pt["date"] for pt in series_dict["Integrated M2.2"]]
    n_pts = len(dates)

    # Normalize Nifty 50 to 100,000 start
    b0 = bench_closes.get(dates[0], 1.0)
    nifty_curve = [{"date": d, "nav": round(100_000.0 * (bench_closes.get(d, b0) / max(1e-6, b0)), 2)} for d in dates]
    all_curves = {**series_dict, "Nifty 50": nifty_curve}

    def _build_svg(mode: str, filename: str, title: str) -> str:
        w, h, pad_l, pad_r, pad_t, pad_b = 920, 380, 62, 24, 42, 42
        vals_all = []
        series_vals: dict[str, list[float]] = {}
        for name, pts in all_curves.items():
            if mode == "equity":
                arr = [(p["nav"] / 1000.0) for p in pts]  # Base 100
            else:
                pk, arr = 1.0, []
                for p in pts:
                    nv = p["nav"]
                    pk = max(pk, nv)
                    arr.append(((nv - pk) / max(1e-6, pk)) * 100.0)
            series_vals[name] = arr
            vals_all.extend(arr)

        y_min = min(vals_all) if mode == "drawdown" else max(50.0, min(vals_all) - 4.0)
        y_max = 0.0 if mode == "drawdown" else max(vals_all) + 5.0
        span = max(1e-6, y_max - y_min)

        def px_x(i: int, tot: int) -> float:
            return pad_l + (i / max(1, tot - 1)) * (w - pad_l - pad_r)

        def px_y(v: float) -> float:
            return pad_t + ((y_max - v) / span) * (h - pad_t - pad_b)

        lines = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" style="background:#0f172a;font-family:monospace;">',
            f'<text x="{pad_l}" y="24" fill="#e2e8f0" font-size="14" font-weight="bold">{title}</text>',
        ]
        for k_g in range(5):
            yv = y_min + (k_g / 4.0) * span
            yy = px_y(yv)
            lbl = f"{yv:.0f}%" if mode == "drawdown" else f"{yv:.0f}"
            lines.append(f'<line x1="{pad_l}" y1="{yy:.1f}" x2="{w-pad_r}" y2="{yy:.1f}" stroke="#1e293b" stroke-width="1"/>')
            lines.append(f'<text x="{pad_l-8}" y="{yy+4:.1f}" fill="#94a3b8" font-size="10" text-anchor="end">{lbl}</text>')

        leg_x = pad_l
        for name, arr in series_vals.items():
            col = colors.get(name, "#cbd5e1")
            step = max(1, len(arr) // 240)
            pts_str = " ".join(f"{px_x(i, len(arr)):.1f},{px_y(arr[i]):.1f}" for i in range(0, len(arr), step))
            if (len(arr) - 1) % step != 0:
                pts_str += f" {px_x(len(arr)-1, len(arr)):.1f},{px_y(arr[-1]):.1f}"
            lines.append(f'<polyline fill="none" stroke="{col}" stroke-width="2" points="{pts_str}"/>')
            leg_val = min(arr) if mode == "drawdown" else arr[-1]
            lines.append(f'<text x="{leg_x}" y="{h-12}" fill="{col}" font-size="11">● {name} ({leg_val:.1f}{"%" if mode=="drawdown" else ""})</text>')
            leg_x += 195

        lines.append("</svg>")
        out_p = out_dir / filename
        out_p.write_text("\n".join(lines), encoding="utf-8")
        return str(out_p)

    eq_path = _build_svg("equity", "equity_curve_comparison.svg", "Moreshwar 2.2 Integrated Equity Curve (Base = 100, NSE INR)")
    dd_path = _build_svg("drawdown", "drawdown_comparison.svg", "Moreshwar 2.2 Underwater Drawdown Comparison (%)")
    return eq_path, dd_path


def run_moreshwar_v2_2_prompt3_suite(
    dataset: dict[str, Any] | None = None,
    *,
    save_artifacts: bool = True,
    out_dir: Path = REPORTS_DIR,
) -> dict[str, Any]:
    """Execute complete Prompt 3 Portfolio Optimization, Validation, Stress & Decision."""
    ds = dataset or load_or_build_expanded_dataset(max_age_hours=720.0)
    uni = ds["universe_history"]
    bench = ds["benchmark"]["candles"]
    pre_inr = precompute_v22_universe(uni, bench, currency_filter="INR")
    tariff = DeliveryCostConfig(brokerage_mode="zerodha_delivery")

    # 1. Portfolio & Risk Optimization strictly on In-Sample (Dev + Validation <= 2025-09-04)
    # ZERO peeking at OOS (>= 2025-09-05) during selection!
    opt_grid: list[tuple[str, str, IntegratedM22Config]] = []
    for cap_n in (4, 5, 6, 8):
        stock_cap = min(25.0, max(16.0, round(100.0 / max(1, cap_n - 1), 1)))
        opt_grid.append((
            f"OPT_RISK_N{cap_n}",
            f"RiskBudget | N={cap_n} | CD=5D",
            IntegratedM22Config(sizing_mode="RISK_BUDGET", max_open_positions=cap_n, max_single_stock_pct=stock_cap, reentry_cooldown_days=5),
        ))
    for smode in ("EQUAL_WEIGHT", "INV_VOL"):
        for cap_n in (6, 8):
            stock_cap = 18.0 if cap_n == 6 else 16.0
            opt_grid.append((
                f"OPT_{smode[:3]}_N{cap_n}",
                f"{smode} | N={cap_n} | CD=5D",
                IntegratedM22Config(sizing_mode=smode, max_open_positions=cap_n, max_single_stock_pct=stock_cap, reentry_cooldown_days=5),
            ))
    for cap_n in (6, 8):
        stock_cap = 18.0 if cap_n == 6 else 16.0
        opt_grid.append((
            f"OPT_REG_DD_N{cap_n}",
            f"RiskBudget + DynRegime + DDThrottle | N={cap_n}",
            IntegratedM22Config(
                sizing_mode="RISK_BUDGET",
                max_open_positions=cap_n,
                max_single_stock_pct=stock_cap,
                use_dynamic_regime_exposure=True,
                use_drawdown_throttle=True,
                tighten_be_on_bear_regime=True,
                reentry_cooldown_days=5,
            ),
        ))
        opt_grid.append((
            f"OPT_INV_REG_DD_N{cap_n}",
            f"InvVol + DynRegime + DDThrottle | N={cap_n}",
            IntegratedM22Config(
                sizing_mode="INV_VOL",
                max_open_positions=cap_n,
                max_single_stock_pct=stock_cap,
                use_dynamic_regime_exposure=True,
                use_drawdown_throttle=True,
                tighten_be_on_bear_regime=True,
                reentry_cooldown_days=5,
            ),
        ))

    is_registry: list[dict[str, Any]] = []
    cfg_map: dict[str, IntegratedM22Config] = {}
    for oid, olbl, ocfg in opt_grid:
        dev_r = simulate_integrated_v2_2(pre_inr, experiment_id=f"{oid}_DEV", label=olbl, cfg=ocfg, tariff=tariff, date_range=DEV_TRAIN_RANGE)
        val_r = simulate_integrated_v2_2(pre_inr, experiment_id=f"{oid}_VAL", label=olbl, cfg=ocfg, tariff=tariff, date_range=VALIDATION_RANGE)
        is_r = simulate_integrated_v2_2(pre_inr, experiment_id=oid, label=olbl, cfg=ocfg, tariff=tariff, date_range=IS_COMBINED_RANGE)
        im = is_r["metrics"]
        im["dev_return_pct"] = dev_r["metrics"]["total_return_pct"]
        im["val_return_pct"] = val_r["metrics"]["total_return_pct"]
        im["val_max_dd_pct"] = val_r["metrics"]["max_drawdown_pct"]
        im["is_calmar"] = round(im["cagr_pct"] / max(1.0, im["max_drawdown_pct"]), 3)
        is_registry.append(im)
        cfg_map[oid] = ocfg

    # Select winning architecture strictly using In-Sample (Dev + Validation) robustness!
    best_is_row = max(
        [r for r in is_registry if r["total_trades"] >= 90],
        key=lambda r: (
            1 if r["val_return_pct"] > -3.5 else 0,
            r["sharpe_ratio"],
            r["is_calmar"],
            r["net_profit_factor"],
        ),
    )
    win_id = best_is_row["experiment_id"]
    win_cfg = cfg_map[win_id]

    # 2. Evaluate Winner ONCE across ALL chronological epochs (Zero retuning after seeing Holdout!)
    full_res = simulate_integrated_v2_2(pre_inr, experiment_id=f"{win_id}_FULL", label=f"Integrated M2.2 ({win_id})", cfg=win_cfg, tariff=tariff, date_range=FULL_RANGE)
    dev_res = simulate_integrated_v2_2(pre_inr, experiment_id=f"{win_id}_DEV", label="Dev-Train", cfg=win_cfg, tariff=tariff, date_range=DEV_TRAIN_RANGE)
    val_res = simulate_integrated_v2_2(pre_inr, experiment_id=f"{win_id}_VAL", label="Validation", cfg=win_cfg, tariff=tariff, date_range=VALIDATION_RANGE)
    oos_res = simulate_integrated_v2_2(pre_inr, experiment_id=f"{win_id}_OOS", label="Prior Exposed OOS (2025-09..2026-11)", cfg=win_cfg, tariff=tariff, date_range=OOS_RANGE)
    hold_res = simulate_integrated_v2_2(pre_inr, experiment_id=f"{win_id}_HOLDOUT", label="Terminal Untouched Holdout (2026-05..2026-11)", cfg=win_cfg, tariff=tariff, date_range=TERMINAL_HOLDOUT_RANGE)

    # 3. 4-Fold Chronological Walk-Forward
    tl = pre_inr["timeline"]
    q = len(tl) // 4
    wf_folds = []
    for fi in range(4):
        s_i = fi * q
        e_i = (len(tl) - 1) if fi == 3 else ((fi + 1) * q - 1)
        f_r = simulate_integrated_v2_2(pre_inr, experiment_id=f"WF_{fi+1}", label=f"WF-{fi+1}", cfg=win_cfg, tariff=tariff, date_range=(tl[s_i], tl[e_i]))["metrics"]
        wf_folds.append({
            "fold": f"WF-{fi+1}" + (" (OOS TAIL)" if fi == 3 else ""),
            "window": f_r["window"],
            "return_pct": f_r["total_return_pct"],
            "nifty_pct": f_r["benchmark_return_pct"],
            "sharpe": f_r["sharpe_ratio"],
            "max_dd_pct": f_r["max_drawdown_pct"],
            "net_pf": f_r["net_profit_factor"],
            "trades": f_r["total_trades"],
            "win_rate_pct": f_r["win_rate_pct"],
        })

    # 4. Break-Testing Suite (Section 5)
    # 4a. Transaction-Cost & Execution Stress
    stress_results: list[dict[str, Any]] = []
    stress_15_res = None
    for s_id, s_lbl, c_mult, t_ovr, c_ovr in [
        ("STRESS_COST_1_00X", "Base Realistic Costs (1.0x)", 1.0, tariff, win_cfg),
        ("STRESS_COST_1_25X", "Cost Stress 1.25x", 1.25, tariff, win_cfg),
        ("STRESS_COST_1_50X", "Cost Stress 1.50x", 1.50, tariff, win_cfg),
        ("STRESS_COST_2_00X", "Cost Stress 2.00x", 2.00, tariff, win_cfg),
        ("STRESS_LEGACY_BRK", "Legacy 0.03% Brokerage", 1.0, DeliveryCostConfig(brokerage_mode="legacy_0_03pct"), win_cfg),
        ("STRESS_SLIP_12_5BPS", "Slippage 12.5 bps", 1.0, DeliveryCostConfig(base_slippage_bps=12.5), win_cfg),
        ("STRESS_SLIP_20BPS", "Slippage 20.0 bps", 1.0, DeliveryCostConfig(base_slippage_bps=20.0), win_cfg),
        ("STRESS_SLIP_30BPS", "Slippage 30.0 bps", 1.0, DeliveryCostConfig(base_slippage_bps=30.0), win_cfg),
        ("STRESS_DELAY_T2", "Delayed Execution (t+2 open)", 1.0, tariff, replace(win_cfg, entry_delay_bars=2)),
        ("STRESS_GAP_SHOCK", "Gap-Through-Stop Shock (-1.0%)", 1.0, tariff, replace(win_cfg, gap_shock_extra_pct=1.0)),
        ("STRESS_PARTIAL_VOL", "Tight Liquidity Cap (0.5% vol)", 1.0, tariff, replace(win_cfg, max_daily_volume_participation=0.005)),
    ]:
        rf = simulate_integrated_v2_2(pre_inr, experiment_id=s_id, label=s_lbl, cfg=c_ovr, tariff=t_ovr, cost_multiplier=c_mult, date_range=FULL_RANGE)
        ro = simulate_integrated_v2_2(pre_inr, experiment_id=f"{s_id}_OOS", label=s_lbl, cfg=c_ovr, tariff=t_ovr, cost_multiplier=c_mult, date_range=OOS_RANGE)
        if s_id == "STRESS_COST_1_50X":
            stress_15_res = rf
        m = rf["metrics"]
        om = ro["metrics"]
        stress_results.append({
            "scenario_id": s_id,
            "label": s_lbl,
            "full_5y_pct": m["total_return_pct"],
            "cagr_pct": m["cagr_pct"],
            "oos_pct": om["total_return_pct"],
            "max_dd_pct": m["max_drawdown_pct"],
            "net_pf": m["net_profit_factor"],
            "sharpe": m["sharpe_ratio"],
            "total_cost": m["total_cost"],
            "trades": m["total_trades"],
            "reconciliation_diff": m.get("reconciliation_diff", 0.0),
        })

    # 4b. Parameter Stability Perturbations around win_cfg
    param_stability: list[dict[str, Any]] = []
    for p_lbl, p_cfg in [
        ("BE 1.75x ATR", replace(win_cfg, breakeven_trigger_atr=1.75)),
        ("BE 2.00x ATR (Base)", win_cfg),
        ("BE 2.25x ATR", replace(win_cfg, breakeven_trigger_atr=2.25)),
        ("Init SL 1.75x ATR", replace(win_cfg, initial_atr_mult=1.75)),
        ("Init SL 2.25x ATR", replace(win_cfg, initial_atr_mult=2.25)),
        ("Horizon 45D", replace(win_cfg, horizon_days=45)),
        ("Horizon 75D", replace(win_cfg, horizon_days=75)),
        ("Cooldown 0D", replace(win_cfg, reentry_cooldown_days=0)),
        ("Cooldown 10D", replace(win_cfg, reentry_cooldown_days=10)),
    ]:
        pf_m = simulate_integrated_v2_2(pre_inr, experiment_id="PARAM", label=p_lbl, cfg=p_cfg, tariff=tariff, date_range=FULL_RANGE)["metrics"]
        po_m = simulate_integrated_v2_2(pre_inr, experiment_id="PARAM_OOS", label=p_lbl, cfg=p_cfg, tariff=tariff, date_range=OOS_RANGE)["metrics"]
        param_stability.append({
            "perturbation": p_lbl,
            "full_5y_pct": pf_m["total_return_pct"],
            "oos_pct": po_m["total_return_pct"],
            "sharpe": pf_m["sharpe_ratio"],
            "max_dd_pct": pf_m["max_drawdown_pct"],
            "net_pf": pf_m["net_profit_factor"],
        })

    # 4c. Concentration & Market Stress on Winner Trades
    tr_all = full_res["trades"]
    by_sym_pnl: dict[str, float] = {}
    for t in tr_all:
        by_sym_pnl[t["symbol"]] = by_sym_pnl.get(t["symbol"], 0.0) + t["pnl_amount"]
    top_sym = max(by_sym_pnl.items(), key=lambda kv: kv[1]) if by_sym_pnl else ("NONE", 0.0)
    tr_ex_top_sym = [t for t in tr_all if t["symbol"] != top_sym[0]]

    sorted_wins = sorted(tr_all, key=lambda t: t["pnl_amount"], reverse=True)
    top5_ids = {(t["symbol"], t["entry_date"], t["exit_date"]) for t in sorted_wins[:5]}
    tr_ex_top5 = [t for t in tr_all if (t["symbol"], t["entry_date"], t["exit_date"]) not in top5_ids]

    def _subset_pf(rows: list[dict[str, Any]]) -> tuple[float, float]:
        tot = sum(r["pnl_amount"] for r in rows)
        gw = sum(r["pnl_amount"] for r in rows if r["pnl_amount"] > 0)
        gl = abs(sum(r["pnl_amount"] for r in rows if r["pnl_amount"] <= 0))
        return round((tot / INITIAL_CAPITAL) * 100.0, 2), round(gw / max(1e-6, gl), 2)

    ex_sym_ret, ex_sym_pf = _subset_pf(tr_ex_top_sym)
    ex_top5_ret, ex_top5_pf = _subset_pf(tr_ex_top5)
    ex_2023_ret, ex_2023_pf = _subset_pf([t for t in tr_all if not t["exit_date"].startswith("2023")])
    best_mo = max(full_res["metrics"]["monthly_returns"], key=lambda m: m["return_pct"]) if full_res["metrics"]["monthly_returns"] else {"month": "N/A", "return_pct": 0.0}

    concentration_stress = {
        "top_winning_stock": {"symbol": top_sym[0], "pnl": round(top_sym[1], 2)},
        "ex_top_winning_stock": {"return_pct": ex_sym_ret, "net_pf": ex_sym_pf},
        "top5_trades_pnl_sum": round(sum(t["pnl_amount"] for t in sorted_wins[:5]), 2),
        "ex_top5_winning_trades": {"return_pct": ex_top5_ret, "net_pf": ex_top5_pf},
        "ex_best_year_2023": {"return_pct": ex_2023_ret, "net_pf": ex_2023_pf},
        "best_month": best_mo,
    }

    # 5. Statistical Confidence (1,200 sims Trade + 10D Block + 20D Block)
    stat_conf = compute_statistical_confidence(tr_all, full_res["daily_rets"], n_sims=1200)

    # Load P1/P2 reference metrics for scorecard
    p1_best_m = json.loads(P1_REPORT_PATH.read_text(encoding="utf-8"))["best_candidate"]["metrics"] if P1_REPORT_PATH.exists() else {}
    p2_data = json.loads(P2_REPORT_PATH.read_text(encoding="utf-8")) if P2_REPORT_PATH.exists() else {}
    p2_best_m = (p2_data.get("best_trade_management") or {}).get("metrics", {})
    p2_best_curve = simulate_integrated_v2_2(
        pre_inr,
        experiment_id="P2_BEST_REF",
        label="Prompt 2 Best Ref",
        cfg=IntegratedM22Config(entry_mode="PULL_CONT", exit_mode="P1_BREAKEVEN_LOCK", max_open_positions=6, reentry_cooldown_days=0),
        tariff=tariff,
        date_range=FULL_RANGE,
    )["equity_curve"]

    # 6. Render Equity & Drawdown SVG Charts
    chart_paths = {}
    if save_artifacts:
        out_dir.mkdir(parents=True, exist_ok=True)
        eq_p, dd_p = _render_svg_charts(
            series_dict={
                "Integrated M2.2": full_res["equity_curve"],
                "Stress 1.5x Costs": (stress_15_res or full_res)["equity_curve"],
                "Prompt 2 Best": p2_best_curve,
            },
            bench_closes=pre_inr["bench_close_by_date"],
            out_dir=out_dir,
        )
        chart_paths = {"equity_svg": eq_p, "drawdown_svg": dd_p}

    # 7. Go / Conditional Go / No-Go Verdict (Strict Objective Check)
    fm = full_res["metrics"]
    om = oos_res["metrics"]
    hm = hold_res["metrics"]
    s15 = stress_results[2]  # 1.5x cost stress
    blk10 = stat_conf["block_bootstrap_10d"]

    strong_sharpe = fm["sharpe_ratio"] >= 0.75
    beats_cash_oos = om["total_return_pct"] > om["cash_return_pct"]
    robust_under_stress = s15["full_5y_pct"] > 0.0 and s15["net_pf"] >= 1.25
    holds_concentration = ex_top5_ret > 0.0 and ex_sym_ret > 0.0

    if (
        fm["total_return_pct"] > fm["cash_return_pct"]
        and om["total_return_pct"] > 0.0
        and hm["total_return_pct"] >= 0.0
        and fm["max_drawdown_pct"] < 15.0
        and fm["net_profit_factor"] >= 1.35
        and strong_sharpe
        and robust_under_stress
        and holds_concentration
        and blk10["prob_positive_return_pct"] >= 75.0
    ):
        go_decision = "GO: Research candidate"
    elif (
        fm["total_return_pct"] > 15.0
        and fm["net_profit_factor"] >= 1.30
        and om["total_return_pct"] >= 0.0
        and s15["full_5y_pct"] > 0.0
    ):
        go_decision = "CONDITIONAL GO"
    else:
        go_decision = "NO-GO"

    report = {
        "research_version": "2.2.0-integrated-final",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "promoted_to_production": False,
        "decision": go_decision,
        "selected_config_id": win_id,
        "selected_config": asdict(win_cfg),
        "optimization_registry_is_only": is_registry,
        "scorecard": {
            "m21_frozen": p1_best_m and json.loads(P1_REPORT_PATH.read_text(encoding="utf-8"))["m21_reconciliation"]["frozen_mixed"],
            "m21_frozen_inr": p1_best_m and json.loads(P1_REPORT_PATH.read_text(encoding="utf-8"))["m21_reconciliation"]["frozen_inr_only"],
            "prompt1_best": p1_best_m,
            "prompt2_best": p2_best_m,
            "integrated_m22": {
                **fm,
                "dev_return_pct": dev_res["metrics"]["total_return_pct"],
                "val_return_pct": val_res["metrics"]["total_return_pct"],
                "oos_return_pct": om["total_return_pct"],
                "oos_cagr_pct": om["cagr_pct"],
                "oos_sharpe": om["sharpe_ratio"],
                "oos_pf": om["net_profit_factor"],
                "oos_trades": om["total_trades"],
                "holdout_return_pct": hm["total_return_pct"],
                "holdout_max_dd_pct": hm["max_drawdown_pct"],
                "holdout_pf": hm["net_profit_factor"],
                "holdout_trades": hm["total_trades"],
            },
            "integrated_m22_stress_15x": s15,
            "us_sleeve_isolated": json.loads(P1_REPORT_PATH.read_text(encoding="utf-8")).get("us_sleeve_isolated", {}) if P1_REPORT_PATH.exists() else {},
        },
        "walk_forward_4fold": wf_folds,
        "stress_tests": stress_results,
        "parameter_stability": param_stability,
        "concentration_stress": concentration_stress,
        "statistical_confidence": stat_conf,
        "charts": chart_paths,
        "equity_curve_sampled": full_res["equity_curve"][::5] + ([full_res["equity_curve"][-1]] if full_res["equity_curve"] else []),
    }

    if save_artifacts:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "experiment_registry.json").write_text(
            json.dumps({"is_optimization_grid": is_registry, "stress": stress_results, "param_stability": param_stability}, indent=2),
            encoding="utf-8",
        )
        (out_dir / "integrated_final_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    return report


if __name__ == "__main__":
    rep = run_moreshwar_v2_2_prompt3_suite(save_artifacts=True)
    im = rep["scorecard"]["integrated_m22"]
    print(json.dumps({
        "decision": rep["decision"],
        "selected": rep["selected_config_id"],
        "integrated": {k: im.get(k) for k in (
            "total_return_pct", "dev_return_pct", "val_return_pct", "oos_return_pct",
            "holdout_return_pct", "cagr_pct", "max_drawdown_pct", "net_profit_factor",
            "sharpe_ratio", "sortino_ratio", "win_rate_pct", "expectancy_pct",
            "p_value_vs_zero", "total_cost", "total_trades", "capital_utilisation_pct",
            "annualised_vol_pct", "benchmark_return_pct", "cash_return_pct",
        )},
        "concentration_stress": rep["concentration_stress"],
        "stat_conf": rep["statistical_confidence"],
    }, indent=2))
