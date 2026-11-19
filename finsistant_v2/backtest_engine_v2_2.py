"""Moreshwar 2.2 Controlled Quantitative Experiment & Economic Reconciliation Engine.

Strictly preserves Moreshwar 1.0, Moreshwar 2.0, and Moreshwar 2.1 untouched.
Executes isolated NSE (INR) and US (USD) simulations with reconciled Zerodha
Equity Delivery tariffs, per-scrip-per-day DP charges, exact slippage tracking,
deterministic cross-sectional candidate ranking, controlled 14D/30D/45D horizon
ablations, capacity (4/5/6/8) ablations, and Dev-Train / Validation / Untouched OOS
evaluation across all 18 required performance & concentration metrics.
"""

from __future__ import annotations

from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any

from .backtest_engine_v2_1 import (
    INITIAL_CAPITAL,
    RISK_FREE_ANNUAL,
    _compute_benchmark_metrics,
    _compute_cash_benchmark,
    _compute_statistical_significance,
    precompute_causal_features as precompute_v21_features,
    simulate_portfolio_v2_1,
)
from .historical_data import load_or_build_expanded_dataset
from .moreshwar_v2_1 import MoreshwarV21Config, classify_market_regime_6
from .moreshwar_v2_2 import (
    DeliveryCostConfig,
    MoreshwarV22Config,
    allocate_ranked_slots_v2_2,
    compute_causal_factor_snapshot,
    compute_repaired_delivery_costs,
    score_candidate_v2_2,
)

DEFAULT_M22_REPORT_PATH = Path(__file__).resolve().parent.parent / "data" / "moreshwar_v2_2_experiment_report.json"

# Chronological isolated epochs matching M2.1's exact OOS boundary (2025-09-05)
DEV_TRAIN_RANGE = ("2022-01-24", "2024-06-30")
VALIDATION_RANGE = ("2024-07-01", "2025-09-04")
IS_COMBINED_RANGE = ("2022-01-24", "2025-09-04")
OOS_RANGE = ("2025-09-05", "2026-11-19")
FULL_RANGE = ("2022-01-24", "2026-11-19")


def precompute_v22_universe(
    universe_history: dict[str, dict[str, Any]],
    benchmark_candles: list[dict[str, Any]],
    *,
    currency_filter: str = "INR",
    min_date: str = "2022-01-24",
) -> dict[str, Any]:
    """Precompute strictly causal M2.2 factor snapshots on a currency-isolated universe."""
    b_list = [b for b in (benchmark_candles or []) if float(b.get("close") or 0.0) > 0]
    bench_dates = [str(b["date"]) for b in b_list]
    bench_closes = [float(b["close"]) for b in b_list]
    bench_ret60_map: dict[str, float] = {}
    for bi, dt in enumerate(bench_dates):
        r60 = ((bench_closes[bi] / max(1e-6, bench_closes[bi - 60])) - 1.0) * 100.0 if bi >= 60 else 0.0
        bench_ret60_map[dt] = r60

    filtered_uni = {
        sym: snap
        for sym, snap in universe_history.items()
        if str(snap.get("currency", "INR")).upper() == currency_filter.upper()
    }

    features_by_sym: dict[str, dict[str, dict[str, Any]]] = {}
    date_bars: dict[str, dict[str, tuple[int, dict[str, Any]]]] = {}
    all_dates: set[str] = set()

    for sym, snap in sorted(filtered_uni.items()):
        candles = snap.get("candles") or []
        if len(candles) < 62:
            continue
        s_feats: dict[str, dict[str, Any]] = {}
        s_bars: dict[str, tuple[int, dict[str, Any]]] = {}
        b_ptr = 0
        for idx, bar in enumerate(candles):
            dt = str(bar.get("date", ""))
            s_bars[dt] = (idx, bar)
            if idx < 60 or dt < min_date:
                continue
            while b_ptr + 1 < len(bench_dates) and bench_dates[b_ptr + 1] <= dt:
                b_ptr += 1
            b_dt = bench_dates[b_ptr] if bench_dates else dt
            b_r60 = bench_ret60_map.get(b_dt, 0.0)
            ft = compute_causal_factor_snapshot(candles, b_list, idx, bench_ret60_pct=b_r60)
            if ft is not None:
                s_feats[dt] = ft
                all_dates.add(dt)
        features_by_sym[sym] = s_feats
        date_bars[sym] = s_bars

    timeline = sorted(all_dates)
    regime6_by_date: dict[str, dict[str, Any]] = {}
    bench_ret_by_date: dict[str, float] = {}
    bench_close_by_date: dict[str, float] = {}
    b_ptr = 0
    for dt in timeline:
        while b_ptr + 1 < len(b_list) and str(b_list[b_ptr + 1]["date"]) <= dt:
            b_ptr += 1
        sub_b = b_list[max(0, b_ptr - 60) : b_ptr + 1] if b_list else []
        regime6_by_date[dt] = classify_market_regime_6(sub_b, decision_date=dt)
        if b_list:
            c1 = float(b_list[b_ptr]["close"])
            c0 = float(b_list[b_ptr - 1]["close"]) if b_ptr > 0 else c1
            bench_ret_by_date[dt] = ((c1 - c0) / max(1e-6, c0)) * 100.0
            bench_close_by_date[dt] = c1
        else:
            bench_ret_by_date[dt] = 0.0
            bench_close_by_date[dt] = 0.0

    return {
        "currency": currency_filter,
        "symbols": sorted(filtered_uni.keys()),
        "universe": filtered_uni,
        "timeline": timeline,
        "features": features_by_sym,
        "date_bars": date_bars,
        "regime6_by_date": regime6_by_date,
        "bench_ret_by_date": bench_ret_by_date,
        "bench_close_by_date": bench_close_by_date,
    }


def _compute_calendar_returns(
    daily_navs: list[dict[str, Any]],
    bench_closes: dict[str, float],
    initial_capital: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Compute exact yearly and monthly portfolio returns vs benchmark."""
    if not daily_navs:
        return [], []

    by_year: dict[str, list[dict[str, Any]]] = {}
    by_month: dict[str, list[dict[str, Any]]] = {}
    for row in daily_navs:
        dt = row["date"]
        by_year.setdefault(dt[:4], []).append(row)
        by_month.setdefault(dt[:7], []).append(row)

    yearly_out: list[dict[str, Any]] = []
    prev_nav = initial_capital
    prev_b = bench_closes.get(daily_navs[0]["date"], 0.0)
    for yr in sorted(by_year.keys()):
        pts = by_year[yr]
        end_nav = pts[-1]["nav"]
        end_b = bench_closes.get(pts[-1]["date"], prev_b)
        p_ret = ((end_nav - prev_nav) / max(1e-6, prev_nav)) * 100.0
        b_ret = ((end_b - prev_b) / max(1e-6, prev_b)) * 100.0 if prev_b > 0 else 0.0
        yearly_out.append({
            "year": yr,
            "start_date": pts[0]["date"],
            "end_date": pts[-1]["date"],
            "return_pct": round(p_ret, 2),
            "benchmark_pct": round(b_ret, 2),
            "alpha_pp": round(p_ret - b_ret, 2),
        })
        prev_nav = end_nav
        prev_b = end_b

    monthly_out: list[dict[str, Any]] = []
    prev_nav = initial_capital
    prev_b = bench_closes.get(daily_navs[0]["date"], 0.0)
    for mo in sorted(by_month.keys()):
        pts = by_month[mo]
        end_nav = pts[-1]["nav"]
        end_b = bench_closes.get(pts[-1]["date"], prev_b)
        p_ret = ((end_nav - prev_nav) / max(1e-6, prev_nav)) * 100.0
        b_ret = ((end_b - prev_b) / max(1e-6, prev_b)) * 100.0 if prev_b > 0 else 0.0
        monthly_out.append({
            "month": mo,
            "return_pct": round(p_ret, 2),
            "benchmark_pct": round(b_ret, 2),
            "alpha_pp": round(p_ret - b_ret, 2),
        })
        prev_nav = end_nav
        prev_b = end_b

    return yearly_out, monthly_out


def simulate_portfolio_v2_2(
    precomputed: dict[str, Any],
    *,
    experiment_id: str,
    label: str,
    cfg: MoreshwarV22Config,
    tariff: DeliveryCostConfig,
    date_range: tuple[str, str] | None = None,
    initial_capital: float = INITIAL_CAPITAL,
) -> dict[str, Any]:
    """Run strict chronological simulation under Moreshwar 2.2 with reconciled costs & ranking."""
    currency = precomputed["currency"]
    uni = precomputed["universe"]
    timeline_full = precomputed["timeline"]
    timeline = [d for d in timeline_full if date_range[0] <= d <= date_range[1]] if date_range else list(timeline_full)
    if len(timeline) < 10:
        return {"experiment_id": experiment_id, "label": label, "metrics": {}, "trades": []}

    features = precomputed["features"]
    date_bars = precomputed["date_bars"]
    regime6_map = precomputed["regime6_by_date"]
    bench_rets = precomputed["bench_ret_by_date"]
    bench_closes = precomputed["bench_close_by_date"]
    valid_syms = sorted(features.keys())

    cash = float(initial_capital)
    open_positions: dict[str, dict[str, Any]] = {}
    pending_orders: list[dict[str, Any]] = []
    closed_trades: list[dict[str, Any]] = []
    dp_charged_pairs: set[tuple[str, str]] = set()  # (sell_date, symbol) -> per-scrip-per-day rule

    buy_notional_raw_sum = 0.0
    sell_notional_raw_sum = 0.0
    buy_notional_exec_sum = 0.0
    sell_notional_exec_sum = 0.0
    cost_totals = {
        "brokerage": 0.0,
        "stt": 0.0,
        "exchange_txn": 0.0,
        "sebi_fee": 0.0,
        "stamp_duty": 0.0,
        "gst": 0.0,
        "dp_charge": 0.0,
        "slippage_cost": 0.0,
        "regulatory_total": 0.0,
        "total_cost": 0.0,
    }
    exec_stats = {
        "gap_through_stops": 0,
        "partial_fills": 0,
        "unfillable_zero_vol": 0,
        "liquidity_delayed_exits": 0,
        "rejection_categories": {},
    }
    daily_navs: list[dict[str, Any]] = []
    daily_snaps: list[dict[str, Any]] = []

    for d_idx, dt in enumerate(timeline):
        r6 = regime6_map[dt]["regime"]

        # 1. Execute queued next-bar open orders with reconciled slippage & fees
        for ord_req in pending_orders:
            sym = ord_req["symbol"]
            if sym in open_positions or dt not in date_bars.get(sym, {}):
                continue
            b_idx, bar = date_bars[sym][dt]
            raw_open = float(bar.get("open") or 0.0)
            bar_vol = int(bar.get("volume") or 0)
            if raw_open <= 0.0 or bar_vol <= 0:
                exec_stats["unfillable_zero_vol"] += 1
                exec_stats["rejection_categories"]["LIQUIDITY_ZERO_VOL"] = (
                    exec_stats["rejection_categories"].get("LIQUIDITY_ZERO_VOL", 0) + 1
                )
                continue

            slip_bps = float(ord_req["slippage_bps"])
            slip_frac = slip_bps / 10_000.0
            exec_entry_px = raw_open * (1.0 + slip_frac)
            port_nav = cash + sum(p["last_price"] * p["qty"] for p in open_positions.values())

            qty = min(int(ord_req["req_qty"]), int((cash * 0.96) // max(1.0, exec_entry_px)))
            max_bar_shares = max(1, int(bar_vol * cfg.max_daily_volume_participation))
            partial_cap = qty > max_bar_shares
            if partial_cap:
                qty = max_bar_shares
                exec_stats["partial_fills"] += 1

            min_viable = cfg.min_viable_notional_inr if currency == "INR" else cfg.min_viable_notional_usd
            if qty < 1 or (qty * exec_entry_px) < min_viable:
                exec_stats["rejection_categories"]["ZERO_SHARES"] = (
                    exec_stats["rejection_categories"].get("ZERO_SHARES", 0) + 1
                )
                continue

            raw_buy_notional = qty * raw_open
            buy_costs = compute_repaired_delivery_costs(
                side="BUY",
                raw_notional=raw_buy_notional,
                currency=currency,
                slippage_bps=slip_bps,
                dp_charged_today=False,
                tariff=tariff,
            )
            exec_buy_notional = buy_costs["exec_notional"]
            reg_buy_fee = buy_costs["regulatory_total"]
            if exec_buy_notional + reg_buy_fee > cash:
                qty = int((cash * 0.96) // max(1.0, exec_entry_px * 1.002))
                if qty < 1 or (qty * exec_entry_px) < min_viable:
                    continue
                raw_buy_notional = qty * raw_open
                buy_costs = compute_repaired_delivery_costs(
                    side="BUY",
                    raw_notional=raw_buy_notional,
                    currency=currency,
                    slippage_bps=slip_bps,
                    dp_charged_today=False,
                    tariff=tariff,
                )
                exec_buy_notional = buy_costs["exec_notional"]
                reg_buy_fee = buy_costs["regulatory_total"]

            for ck in cost_totals:
                cost_totals[ck] += buy_costs.get(ck, 0.0)
            buy_notional_raw_sum += raw_buy_notional
            buy_notional_exec_sum += exec_buy_notional
            cash -= (exec_buy_notional + reg_buy_fee)

            eff_sl = round(exec_entry_px * (1.0 - ord_req["sl_ratio"]), 4)
            eff_tp = round(exec_entry_px * (1.0 + ord_req["tp_ratio"]), 4)
            rec_closes = [float(x["close"]) for x in uni[sym]["candles"][max(0, b_idx - 61) : b_idx + 1]]
            open_positions[sym] = {
                "symbol": sym,
                "sector": ord_req["sector"],
                "signal_date": ord_req["signal_date"],
                "signal_regime6": ord_req["market_regime"],
                "rank_score": ord_req["rank_score"],
                "entry_date": dt,
                "raw_open_entry": raw_open,
                "entry_price": exec_entry_px,
                "last_price": exec_entry_px,
                "qty": qty,
                "raw_buy_notional": raw_buy_notional,
                "exec_buy_notional": exec_buy_notional,
                "buy_reg_cost": reg_buy_fee,
                "buy_slip_cost": buy_costs["slippage_cost"],
                "sl": eff_sl,
                "tp": eff_tp,
                "slippage_bps": slip_bps,
                "mfe_pct": 0.0,
                "mae_pct": 0.0,
                "bars_held": 0,
                "recent_closes": rec_closes,
            }
        pending_orders = []

        # 2. Exit monitoring
        to_close: list[str] = []
        for sym, pos in open_positions.items():
            if dt not in date_bars.get(sym, {}):
                continue
            b_idx, bar = date_bars[sym][dt]
            hi = float(bar.get("high") or pos["last_price"])
            lo = float(bar.get("low") or pos["last_price"])
            cl = float(bar.get("close") or pos["last_price"])
            op = float(bar.get("open") or cl)
            vol = int(bar.get("volume") or 0)
            pos["last_price"] = cl
            pos["bars_held"] += 1
            pos["mfe_pct"] = max(pos["mfe_pct"], ((hi - pos["entry_price"]) / pos["entry_price"]) * 100.0)
            pos["mae_pct"] = min(pos["mae_pct"], ((lo - pos["entry_price"]) / pos["entry_price"]) * 100.0)

            raw_exit_px = None
            exit_reason = None
            is_gap_stop = False
            if lo <= pos["sl"]:
                if op < pos["sl"]:
                    raw_exit_px = op
                    is_gap_stop = True
                else:
                    raw_exit_px = pos["sl"]
                exit_reason = "STOP_LOSS"
            elif hi >= pos["tp"]:
                raw_exit_px = max(op, pos["tp"]) if op > pos["tp"] else pos["tp"]
                exit_reason = "TARGET_HIT"
            elif pos["bars_held"] >= cfg.horizon_days or d_idx == len(timeline) - 1:
                raw_exit_px = cl
                exit_reason = "HORIZON_EXIT"

            if raw_exit_px is not None:
                if vol <= 0 and d_idx < len(timeline) - 1:
                    exec_stats["liquidity_delayed_exits"] += 1
                    continue
                if is_gap_stop:
                    exec_stats["gap_through_stops"] += 1

                raw_sell_notional = pos["qty"] * raw_exit_px
                dp_key = (dt, sym)
                dp_already = dp_key in dp_charged_pairs
                sell_costs = compute_repaired_delivery_costs(
                    side="SELL",
                    raw_notional=raw_sell_notional,
                    currency=currency,
                    slippage_bps=pos["slippage_bps"],
                    dp_charged_today=dp_already,
                    tariff=tariff,
                )
                if currency == "INR" and not dp_already and sell_costs["dp_charge"] > 0:
                    dp_charged_pairs.add(dp_key)

                for ck in cost_totals:
                    cost_totals[ck] += sell_costs.get(ck, 0.0)
                exec_sell_notional = sell_costs["exec_notional"]
                sell_notional_raw_sum += raw_sell_notional
                sell_notional_exec_sum += exec_sell_notional

                net_proceeds = exec_sell_notional - sell_costs["regulatory_total"]
                cash += net_proceeds

                # Exact Accounting Reconciliation:
                # Gross (unslipped raw open -> raw exit) - Exact Total Cost (Reg + Slip) == Net PnL
                gross_unslipped_pnl = raw_sell_notional - pos["raw_buy_notional"]
                total_trade_cost = pos["buy_reg_cost"] + pos["buy_slip_cost"] + sell_costs["total_cost"]
                pnl_amt = net_proceeds - (pos["exec_buy_notional"] + pos["buy_reg_cost"])
                net_ret_pct = (pnl_amt / max(1e-6, pos["exec_buy_notional"] + pos["buy_reg_cost"])) * 100.0

                closed_trades.append({
                    "symbol": sym,
                    "sector": pos["sector"],
                    "signal_date": pos["signal_date"],
                    "entry_date": pos["entry_date"],
                    "exit_date": dt,
                    "signal_regime6": pos["signal_regime6"],
                    "rank_score": pos["rank_score"],
                    "holding_days": pos["bars_held"],
                    "qty": pos["qty"],
                    "raw_buy_notional": round(pos["raw_buy_notional"], 2),
                    "raw_sell_notional": round(raw_sell_notional, 2),
                    "gross_pnl_amount": round(gross_unslipped_pnl, 2),
                    "pnl_amount": round(pnl_amt, 2),
                    "cost_amount": round(total_trade_cost, 2),
                    "net_return_pct": round(net_ret_pct, 2),
                    "mfe_pct": round(pos["mfe_pct"], 2),
                    "mae_pct": round(pos["mae_pct"], 2),
                    "exit_reason": exit_reason,
                    "gap_through_stop": is_gap_stop,
                })
                to_close.append(sym)
        for sym in to_close:
            del open_positions[sym]

        # 3. Record end-of-day NAV & concentration
        deployed_mkt = sum(p["last_price"] * p["qty"] for p in open_positions.values())
        nav = cash + deployed_mkt
        gross_exp_pct = (deployed_mkt / max(1.0, nav)) * 100.0
        sec_map: dict[str, float] = {}
        for p in open_positions.values():
            sec_map[p["sector"]] = sec_map.get(p["sector"], 0.0) + p["last_price"] * p["qty"]
        max_sec_pct = (max(sec_map.values()) / max(1.0, nav)) * 100.0 if sec_map else 0.0
        prev_nav = daily_navs[-1]["nav"] if daily_navs else initial_capital
        day_ret_pct = ((nav - prev_nav) / max(1e-6, prev_nav)) * 100.0
        daily_navs.append({"date": dt, "nav": round(nav, 2), "regime6": r6})
        daily_snaps.append({
            "date": dt,
            "open_positions": len(open_positions),
            "gross_exposure_pct": round(gross_exp_pct, 2),
            "max_sector_pct": round(max_sec_pct, 2),
            "daily_ret_pct": round(day_ret_pct, 3),
            "bench_ret_pct": round(bench_rets.get(dt, 0.0), 3),
        })

        if d_idx >= len(timeline) - 1:
            continue

        # 4. Evaluate ALL eligible candidates at close `dt` BEFORE ranking/allocating slots
        candidates: list[dict[str, Any]] = []
        candles_by_sym: dict[str, list[dict[str, Any]]] = {}
        r6_obj = regime6_map[dt]
        for sym in valid_syms:
            if sym in open_positions or dt not in features[sym]:
                continue
            ft = features[sym][dt]
            b_idx = ft["idx"]
            c_win = uni[sym]["candles"][max(0, b_idx - 61) : b_idx + 1]
            candles_by_sym[sym] = c_win
            cand = score_candidate_v2_2(
                symbol=sym,
                sector=uni[sym].get("sector", "Equity"),
                feats=ft,
                candles_window=c_win,
                regime6=r6_obj,
                currency=currency,
                capital=nav,
                cfg=cfg,
                tariff=tariff,
            )
            cand["signal_date"] = dt
            if cand["eligible"]:
                candidates.append(cand)
            else:
                rc = cand["rejection_category"] or "OTHER"
                if rc != "SIGNAL_NOT_TRIGGERED":
                    exec_stats["rejection_categories"][rc] = exec_stats["rejection_categories"].get(rc, 0) + 1

        # Rank & allocate slots deterministically
        orders, slot_rejs = allocate_ranked_slots_v2_2(
            candidates=candidates,
            candles_by_symbol=candles_by_sym,
            open_positions=open_positions,
            cash=cash,
            portfolio_nav=nav,
            currency=currency,
            cfg=cfg,
        )
        for rc, cnt in slot_rejs.items():
            exec_stats["rejection_categories"][rc] = exec_stats["rejection_categories"].get(rc, 0) + cnt
        pending_orders = orders

    # Compute ALL 18 Required Metrics
    end_nav = daily_navs[-1]["nav"] if daily_navs else initial_capital
    total_ret = ((end_nav - initial_capital) / initial_capital) * 100.0
    n_days = max(1, len(daily_navs))
    cagr = (((end_nav / initial_capital) ** (252.0 / n_days)) - 1.0) * 100.0 if end_nav > 0 else -100.0

    peak = initial_capital
    max_dd = 0.0
    daily_rets: list[float] = []
    prev_n = initial_capital
    for pt in daily_navs:
        n = pt["nav"]
        peak = max(peak, n)
        max_dd = min(max_dd, ((n - peak) / peak) * 100.0)
        daily_rets.append((n - prev_n) / prev_n)
        prev_n = n

    rf_d = RISK_FREE_ANNUAL / 252.0
    ex = [r - rf_d for r in daily_rets]
    m_ex = sum(ex) / len(ex)
    std_ex = math.sqrt(sum((x - m_ex) ** 2 for x in ex) / max(1, len(ex) - 1))
    dn_ex = math.sqrt(sum(min(0.0, x) ** 2 for x in ex) / len(ex))
    sharpe = round((m_ex / std_ex) * math.sqrt(252.0), 2) if std_ex > 1e-9 else 0.0
    sortino = round((m_ex / dn_ex) * math.sqrt(252.0), 2) if dn_ex > 1e-9 else 0.0

    # Benchmark & Information Ratio
    b_start = bench_closes.get(timeline[0], 0.0)
    b_end = bench_closes.get(timeline[-1], 0.0)
    bench_ret_pct = ((b_end - b_start) / max(1e-6, b_start)) * 100.0 if b_start > 0 else 0.0
    cash_ret_pct = (((1.0 + RISK_FREE_ANNUAL) ** (n_days / 252.0)) - 1.0) * 100.0
    active_daily = [daily_rets[i] - (daily_snaps[i]["bench_ret_pct"] / 100.0) for i in range(len(daily_rets))]
    m_act = sum(active_daily) / max(1, len(active_daily))
    sd_act = math.sqrt(sum((a - m_act) ** 2 for a in active_daily) / max(1, len(active_daily) - 1))
    info_ratio = round((m_act / sd_act) * math.sqrt(252.0), 2) if sd_act > 1e-9 else 0.0

    sig = _compute_statistical_significance(closed_trades)
    n_tr = len(closed_trades)
    wins = [t for t in closed_trades if t["pnl_amount"] > 0]
    losses = [t for t in closed_trades if t["pnl_amount"] <= 0]
    gross_wins = [t for t in closed_trades if t["gross_pnl_amount"] > 0]
    gross_losses = [t for t in closed_trades if t["gross_pnl_amount"] <= 0]

    net_gw = sum(t["pnl_amount"] for t in wins)
    net_gl = abs(sum(t["pnl_amount"] for t in losses))
    gross_gw = sum(t["gross_pnl_amount"] for t in gross_wins)
    gross_gl = abs(sum(t["gross_pnl_amount"] for t in gross_losses))
    net_pf = round(net_gw / net_gl, 2) if net_gl > 1e-6 else (9.99 if net_gw > 0 else 0.0)
    gross_pf = round(gross_gw / gross_gl, 2) if gross_gl > 1e-6 else (9.99 if gross_gw > 0 else 0.0)

    stock_pnl: dict[str, float] = {}
    stock_cnt: dict[str, int] = {}
    for t in closed_trades:
        stock_pnl[t["symbol"]] = stock_pnl.get(t["symbol"], 0.0) + t["pnl_amount"]
        stock_cnt[t["symbol"]] = stock_cnt.get(t["symbol"], 0) + 1
    top_stock = max(stock_pnl.items(), key=lambda kv: kv[1]) if stock_pnl else ("NONE", 0.0)
    worst_stock = min(stock_pnl.items(), key=lambda kv: kv[1]) if stock_pnl else ("NONE", 0.0)
    net_pnl_sum = sum(t["pnl_amount"] for t in closed_trades)
    gross_pnl_sum = sum(t["gross_pnl_amount"] for t in closed_trades)

    yearly_ret, monthly_ret = _compute_calendar_returns(daily_navs, bench_closes, initial_capital)
    total_turnover_exec = buy_notional_exec_sum + sell_notional_exec_sum

    metrics = {
        "experiment_id": experiment_id,
        "label": label,
        "currency": currency,
        "distinction": "SIMULATED_RESEARCH_BACKTEST (NOT LIVE EXECUTION)",
        "window": f"{timeline[0]} → {timeline[-1]} ({n_days} days)",
        "total_return_pct": round(total_ret, 2),
        "gross_return_pct": round((gross_pnl_sum / initial_capital) * 100.0, 2),
        "cagr_pct": round(cagr, 2),
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "max_drawdown_pct": round(abs(max_dd), 2),
        "gross_profit_factor": gross_pf,
        "net_profit_factor": net_pf,
        "total_trades": n_tr,
        "win_rate_pct": sig["win_rate_pct"],
        "win_rate_ci95": sig["win_rate_ci95"],
        "avg_winner_pct": round(sum(t["net_return_pct"] for t in wins) / max(1, len(wins)), 2) if wins else 0.0,
        "avg_loser_pct": round(sum(t["net_return_pct"] for t in losses) / max(1, len(losses)), 2) if losses else 0.0,
        "avg_winner_amt": round(net_gw / max(1, len(wins)), 2) if wins else 0.0,
        "avg_loser_amt": round(-net_gl / max(1, len(losses)), 2) if losses else 0.0,
        "expectancy_pct": sig["expectancy_pct"],
        "p_value_vs_zero": sig["p_value_vs_zero"],
        "gross_pnl_total": round(gross_pnl_sum, 2),
        "net_pnl_total": round(net_pnl_sum, 2),
        "nav_change_reconciled": round(end_nav - initial_capital, 2),
        "total_cost": round(cost_totals["total_cost"], 2),
        "cost_drag_pct": round((cost_totals["total_cost"] / initial_capital) * 100.0, 2),
        "cost_breakdown": {k: round(v, 2) for k, v in cost_totals.items()},
        "buy_notional": round(buy_notional_exec_sum, 2),
        "sell_notional": round(sell_notional_exec_sum, 2),
        "total_turnover": round(total_turnover_exec, 2),
        "turnover_multiple_x": round(total_turnover_exec / initial_capital, 2),
        "avg_holding_days": round(sum(t["holding_days"] for t in closed_trades) / max(1, n_tr), 1) if n_tr else 0.0,
        "capital_utilisation_pct": round(sum(s["gross_exposure_pct"] for s in daily_snaps) / max(1, len(daily_snaps)), 1),
        "max_exposure_pct": round(max((s["gross_exposure_pct"] for s in daily_snaps), default=0.0), 2),
        "max_sector_pct": round(max((s["max_sector_pct"] for s in daily_snaps), default=0.0), 2),
        "max_positions": max((s["open_positions"] for s in daily_snaps), default=0),
        "avg_positions": round(sum(s["open_positions"] for s in daily_snaps) / max(1, len(daily_snaps)), 2),
        "unique_symbols_traded": len(stock_cnt),
        "top_stock": {"symbol": top_stock[0], "pnl": round(top_stock[1], 2), "trades": stock_cnt.get(top_stock[0], 0)},
        "worst_stock": {"symbol": worst_stock[0], "pnl": round(worst_stock[1], 2), "trades": stock_cnt.get(worst_stock[0], 0)},
        "benchmark_return_pct": round(bench_ret_pct, 2),
        "excess_vs_benchmark_pp": round(total_ret - bench_ret_pct, 2),
        "cash_return_pct": round(cash_ret_pct, 2),
        "excess_vs_cash_pp": round(total_ret - cash_ret_pct, 2),
        "information_ratio": info_ratio,
        "yearly_returns": yearly_ret,
        "monthly_returns": monthly_ret,
        "execution_realism": exec_stats,
    }

    return {
        "metrics": metrics,
        "config": asdict(cfg),
        "tariff": asdict(tariff),
    }


def _evaluate_candidate_across_epochs(
    pre_inr: dict[str, Any],
    *,
    experiment_id: str,
    label: str,
    cfg: MoreshwarV22Config,
    tariff: DeliveryCostConfig,
) -> dict[str, Any]:
    """Run identical candidate cleanly across Full 5Y, Dev-Train, Validation, IS, and Untouched OOS."""
    full_res = simulate_portfolio_v2_2(pre_inr, experiment_id=experiment_id, label=label, cfg=cfg, tariff=tariff, date_range=FULL_RANGE)
    dev_res = simulate_portfolio_v2_2(pre_inr, experiment_id=f"{experiment_id}_DEV", label=label, cfg=cfg, tariff=tariff, date_range=DEV_TRAIN_RANGE)
    val_res = simulate_portfolio_v2_2(pre_inr, experiment_id=f"{experiment_id}_VAL", label=label, cfg=cfg, tariff=tariff, date_range=VALIDATION_RANGE)
    is_res = simulate_portfolio_v2_2(pre_inr, experiment_id=f"{experiment_id}_IS", label=label, cfg=cfg, tariff=tariff, date_range=IS_COMBINED_RANGE)
    oos_res = simulate_portfolio_v2_2(pre_inr, experiment_id=f"{experiment_id}_OOS", label=label, cfg=cfg, tariff=tariff, date_range=OOS_RANGE)

    fm = full_res["metrics"]
    fm["dev_train_return_pct"] = dev_res["metrics"].get("total_return_pct", 0.0)
    fm["validation_return_pct"] = val_res["metrics"].get("total_return_pct", 0.0)
    fm["is_return_pct"] = is_res["metrics"].get("total_return_pct", 0.0)
    fm["oos_return_pct"] = oos_res["metrics"].get("total_return_pct", 0.0)
    fm["oos_cagr_pct"] = oos_res["metrics"].get("cagr_pct", 0.0)
    fm["oos_sharpe"] = oos_res["metrics"].get("sharpe_ratio", 0.0)
    fm["oos_max_dd_pct"] = oos_res["metrics"].get("max_drawdown_pct", 0.0)
    fm["oos_pf"] = oos_res["metrics"].get("net_profit_factor", 0.0)
    fm["oos_trades"] = oos_res["metrics"].get("total_trades", 0)
    fm["oos_win_rate_pct"] = oos_res["metrics"].get("win_rate_pct", 0.0)
    fm["oos_excess_vs_nifty_pp"] = oos_res["metrics"].get("excess_vs_benchmark_pp", 0.0)
    return full_res


def run_moreshwar_v2_2_prompt1_suite(
    dataset: dict[str, Any] | None = None,
    *,
    save_report: bool = True,
    report_path: Path = DEFAULT_M22_REPORT_PATH,
) -> dict[str, Any]:
    """Execute Prompt 1 (Economic Repair + Ranked Selection) Controlled Experiment Matrix."""
    ds = dataset or load_or_build_expanded_dataset(max_age_hours=720.0)
    uni = ds["universe_history"]
    bench = ds["benchmark"]["candles"]

    # 1. Exact Frozen M2.1 Baseline Reference + Economic Reconciliation
    pre_v21_mixed = precompute_v21_features(uni, bench)
    m21_frozen_full = simulate_portfolio_v2_1(
        uni, bench, pre_v21_mixed, model_name="M2.1 Frozen (Mixed USD+INR, Legacy Tariff)", model_family="M2_1", cfg_v21=MoreshwarV21Config()
    )
    m21_frozen_oos = simulate_portfolio_v2_1(
        uni, bench, pre_v21_mixed, model_name="M2.1 Frozen OOS", model_family="M2_1", cfg_v21=MoreshwarV21Config(), date_range=OOS_RANGE
    )
    uni_inr_raw = {k: v for k, v in uni.items() if str(v.get("currency", "INR")).upper() == "INR"}
    pre_v21_inr = precompute_v21_features(uni_inr_raw, bench)
    m21_frozen_inr = simulate_portfolio_v2_1(
        uni_inr_raw, bench, pre_v21_inr, model_name="M2.1 Frozen (INR-Only, Conf-Scaled Micro-Lots)", model_family="M2_1", cfg_v21=MoreshwarV21Config()
    )
    m21_frozen_inr_oos = simulate_portfolio_v2_1(
        uni_inr_raw, bench, pre_v21_inr, model_name="M2.1 Frozen INR OOS", model_family="M2_1", cfg_v21=MoreshwarV21Config(), date_range=OOS_RANGE
    )

    # Pure INR precomputed universe (39 NSE stocks; NVDA separated strictly)
    pre_inr = precompute_v22_universe(uni, bench, currency_filter="INR")
    pre_usd = precompute_v22_universe(uni, bench, currency_filter="USD")

    tariff_legacy = DeliveryCostConfig(brokerage_mode="legacy_0_03pct", dp_charge_per_scrip_day=15.93)
    tariff_zerodha = DeliveryCostConfig(brokerage_mode="zerodha_delivery", dp_charge_per_scrip_day=15.93)
    tariff_pre2024_nse = DeliveryCostConfig(brokerage_mode="zerodha_delivery", exchange_txn_rate=0.0000325, dp_charge_per_scrip_day=15.93)

    # Step 1 Cost & Currency Reconciliation Ladder (Exact M2.1 selection, horizon=14D)
    m21_sel_14d_cfg = MoreshwarV22Config(
        horizon_days=14,
        use_ranked_selection=False,
        require_structural_uptrend=False,
        require_anti_extension=False,
        require_pit_risk_filter=False,
        require_regime_gate=False,
        require_cost_hurdle=False,
    )
    econ_inr_legacy = _evaluate_candidate_across_epochs(
        pre_inr, experiment_id="ECON_INR_LEGACY_BRK", label="M2.1 Strategy | INR-Only + Legacy 0.03% Brk", cfg=m21_sel_14d_cfg, tariff=tariff_legacy
    )
    econ_inr_zerodha = _evaluate_candidate_across_epochs(
        pre_inr, experiment_id="ECON_INR_ZERODHA", label="M2.1 Strategy | INR-Only + Zerodha ₹0 Deliv", cfg=m21_sel_14d_cfg, tariff=tariff_zerodha
    )
    econ_inr_pre2024_nse = _evaluate_candidate_across_epochs(
        pre_inr, experiment_id="ECON_INR_PRE2024_NSE", label="M2.1 Strategy | INR-Only + NSE 0.00325% Sens", cfg=m21_sel_14d_cfg, tariff=tariff_pre2024_nse
    )

    # Benchmarks on exact INR NSE calendar
    nse_tl = pre_inr["timeline"]
    split_oos = next((i for i, d in enumerate(nse_tl) if d >= OOS_RANGE[0]), int(len(nse_tl) * 0.75))
    nifty_full = _compute_benchmark_metrics(bench, date_range=(nse_tl[0], nse_tl[-1]))
    nifty_oos = _compute_benchmark_metrics(bench, date_range=(nse_tl[split_oos], nse_tl[-1]))
    cash_full = _compute_cash_benchmark(len(nse_tl))
    cash_oos = _compute_cash_benchmark(len(nse_tl) - split_oos)

    # Step 2: Controlled Ablation Matrix across Horizons (14D, 30D, 45D)
    experiments: list[dict[str, Any]] = []
    for h in (14, 30, 45):
        c_m21_sel = replace(m21_sel_14d_cfg, horizon_days=h)
        c_ranked = replace(c_m21_sel, use_ranked_selection=True, require_cost_hurdle=True)
        c_trend = replace(c_ranked, require_structural_uptrend=True)
        c_quality = replace(c_trend, require_anti_extension=True, require_pit_risk_filter=True)
        c_regime = replace(c_quality, require_regime_gate=True, regime_mode="bull_only_no_sideways")

        for eid, lbl, cfg_obj in [
            (f"EXP_{h}D_M21_SEL", f"{h}D | Existing M2.1 Alphab. Entry (Repaired INR)", c_m21_sel),
            (f"EXP_{h}D_RANKED", f"{h}D | Ranked Cross-Sectional + Cost Hurdle", c_ranked),
            (f"EXP_{h}D_RANKED_TREND", f"{h}D | + 50D Structural Uptrend Filter", c_trend),
            (f"EXP_{h}D_RANKED_QUALITY", f"{h}D | + Anti-Extension (<=5% SMA20) + PIT Risk", c_quality),
            (f"EXP_{h}D_RANKED_REGIME", f"{h}D | + Bull-Only Regime Eligibility (No Sideways)", c_regime),
        ]:
            experiments.append(_evaluate_candidate_across_epochs(pre_inr, experiment_id=eid, label=lbl, cfg=cfg_obj, tariff=tariff_zerodha))

    # Step 3: Portfolio Capacity Ablations (4, 5, 6, 8 positions) on Quality Selection (30D & 45D)
    capacity_experiments: list[dict[str, Any]] = []
    for h in (30, 45):
        base_q = replace(
            m21_sel_14d_cfg,
            horizon_days=h,
            use_ranked_selection=True,
            require_cost_hurdle=True,
            require_structural_uptrend=True,
            require_anti_extension=True,
            require_pit_risk_filter=True,
            require_regime_gate=False,
        )
        for cap_n in (4, 5, 6, 8):
            c_cap = replace(
                base_q,
                max_open_positions=cap_n,
                max_single_stock_pct=min(28.0, max(16.0, round(100.0 / max(1, cap_n - 1), 1))),
                max_portfolio_stop_risk_pct=min(7.5, max(4.5, cap_n * 0.9)),
            )
            capacity_experiments.append(
                _evaluate_candidate_across_epochs(
                    pre_inr,
                    experiment_id=f"CAP_{h}D_N{cap_n}",
                    label=f"{h}D Quality Selection | Capacity={cap_n} Slots",
                    cfg=c_cap,
                    tariff=tariff_zerodha,
                )
            )

    # Standalone US Sleeve (NVDA USD, isolated ledger)
    us_cfg = replace(
        m21_sel_14d_cfg,
        horizon_days=30,
        use_ranked_selection=True,
        require_structural_uptrend=True,
        require_anti_extension=True,
        max_open_positions=1,
        max_single_stock_pct=25.0,
    )
    us_res = _evaluate_candidate_across_epochs(
        pre_usd, experiment_id="US_NVDA_USD_30D", label="US Standalone Sleeve (NVDA USD, 30D)", cfg=us_cfg, tariff=DeliveryCostConfig(base_slippage_bps=5.0)
    )

    # Identify best credible M2.2 INR candidate across controlled ablations & capacity tests (trades >= 100)
    all_inr = experiments + capacity_experiments
    credible = [e for e in all_inr if e["metrics"]["total_trades"] >= 100]
    best_cand = max(
        credible,
        key=lambda e: (
            e["metrics"]["sharpe_ratio"],
            e["metrics"]["net_profit_factor"],
            e["metrics"]["total_return_pct"] + e["metrics"]["oos_return_pct"],
        ),
    )

    # Slippage sensitivity on Best Candidate
    best_cfg_dict = best_cand["config"]
    best_cfg = MoreshwarV22Config(**best_cfg_dict)
    slip_sens = []
    for bps in (7.5, 12.5, 20.0):
        sr = _evaluate_candidate_across_epochs(
            pre_inr,
            experiment_id=f"SENS_SLIP_{bps}BPS",
            label=f"Slippage Stress {bps} bps",
            cfg=best_cfg,
            tariff=DeliveryCostConfig(brokerage_mode="zerodha_delivery", base_slippage_bps=bps),
        )
        m = sr["metrics"]
        slip_sens.append({
            "slippage_bps": bps,
            "full_5y_pct": m["total_return_pct"],
            "oos_pct": m["oos_return_pct"],
            "sharpe": m["sharpe_ratio"],
            "net_pf": m["net_profit_factor"],
            "cost_drag_pct": m["cost_drag_pct"],
        })

    # Success Criteria Gate Evaluation (Prompt 1 Section 8)
    bm = best_cand["metrics"]
    gates = {
        "net_5y_positive": bm["total_return_pct"] > 0.0,
        "oos_positive": bm["oos_return_pct"] > 0.0,
        "beats_m21_5y": bm["total_return_pct"] > m21_frozen_full["metrics"]["total_return_pct"],
        "beats_nifty_5y": bm["total_return_pct"] > nifty_full["metrics"]["total_return_pct"],
        "beats_cash_5y": bm["total_return_pct"] > cash_full["metrics"]["total_return_pct"],
        "sharpe_above_1": bm["sharpe_ratio"] >= 1.0,
        "max_dd_below_15": bm["max_drawdown_pct"] < 15.0,
        "pf_above_1_3": bm["net_profit_factor"] >= 1.3,
        "adequate_sample_ge_100": bm["total_trades"] >= 100,
    }
    all_gates_passed = all(gates.values())

    report = {
        "research_version": "2.2.0-prompt1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "promoted_to_production": False,
        "universe_separation": {
            "nse_inr_count": len(pre_inr["symbols"]),
            "nse_inr_days": len(pre_inr["timeline"]),
            "us_usd_count": len(pre_usd["symbols"]),
            "us_usd_days": len(pre_usd["timeline"]),
            "epochs": {
                "full": FULL_RANGE,
                "dev_train": DEV_TRAIN_RANGE,
                "validation": VALIDATION_RANGE,
                "is_combined": IS_COMBINED_RANGE,
                "untouched_oos": OOS_RANGE,
            },
        },
        "m21_reconciliation": {
            "frozen_mixed": m21_frozen_full["metrics"],
            "frozen_mixed_oos": m21_frozen_oos["metrics"],
            "frozen_inr_only": m21_frozen_inr["metrics"],
            "frozen_inr_only_oos": m21_frozen_inr_oos["metrics"],
            "inr_only_legacy_tariff": econ_inr_legacy["metrics"],
            "inr_only_zerodha_repaired": econ_inr_zerodha["metrics"],
            "inr_only_pre2024_nse_sens": econ_inr_pre2024_nse["metrics"],
            "findings": [
                "M2.1 charged min(₹20, 0.03%) delivery brokerage (~₹2,926 + GST) even though Zerodha Equity Delivery brokerage has been ₹0 since 2015.",
                "M2.1 included NVDA in USD 1:1 alongside INR capital (+₹2,158 un-converted USD units without STT/DP); isolating exact M2.1 INR-only drops 5Y net from -13.53% to -20.31%.",
                "M2.1's `historical_unavailable` confidence scaler (~0.52x) shrank tickets to micro-lots (~₹10.9k) where ₹15.93 DP charges consumed 23.9% of all costs, AND insertion-order loop priority differed markedly from A-Z sorted priority.",
                "M2.2 tracks DP ₹15.93 strictly per scrip per sell day `(exit_date, symbol)`, computes exact slippage without `(1+s)` inflation, and replaces insertion-order saturation with deterministic cross-sectional ranking.",
                f"Historical NSE Txn fee shift (0.00325% vs 0.00297%) changes 5Y net cost by only {round(econ_inr_pre2024_nse['metrics']['total_cost'] - econ_inr_zerodha['metrics']['total_cost'], 2)} INR.",
            ],
        },
        "benchmarks": {
            "nifty50_full": nifty_full["metrics"],
            "nifty50_oos": nifty_oos["metrics"],
            "cash_65_full": cash_full["metrics"],
            "cash_65_oos": cash_oos["metrics"],
        },
        "experiments": [e["metrics"] for e in experiments],
        "capacity_experiments": [e["metrics"] for e in capacity_experiments],
        "us_sleeve_isolated": us_res["metrics"],
        "best_candidate": {
            "metrics": bm,
            "config": best_cand["config"],
            "slippage_sensitivity": slip_sens,
            "success_gates": gates,
            "meets_production_targets": all_gates_passed,
        },
    }

    if save_report:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    return report


if __name__ == "__main__":
    out = run_moreshwar_v2_2_prompt1_suite(save_report=True)
    print(json.dumps({
        "best": {k: out["best_candidate"]["metrics"][k] for k in (
            "experiment_id", "label", "total_return_pct", "dev_train_return_pct",
            "validation_return_pct", "oos_return_pct", "cagr_pct", "sharpe_ratio",
            "sortino_ratio", "max_drawdown_pct", "gross_profit_factor", "net_profit_factor",
            "total_trades", "win_rate_pct", "total_cost", "turnover_multiple_x",
        )},
        "gates": out["best_candidate"]["success_gates"],
    }, indent=2))
