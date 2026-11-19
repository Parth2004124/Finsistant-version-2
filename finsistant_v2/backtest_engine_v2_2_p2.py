"""Moreshwar 2.2 Prompt 2 Experiment Engine — Entry, Exit & Profit Retention Evaluation.

Preserves Moreshwar 1.0, 2.0, 2.1, and Prompt 1 outputs completely untouched.
Simulates independent NSE (INR) portfolios with chronological signal->next-open
entries, intra-bar gap-through-stop fills, partial exit accounting, re-entry
cooldown checks, Dev-Train / Validation / Untouched OOS isolation, 4-fold
walk-forward checks, slippage stress tests, and full profit-retention analytics.
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
from .moreshwar_v2_1 import compute_pairwise_return_correlation
from .moreshwar_v2_2 import DeliveryCostConfig, compute_repaired_delivery_costs
from .moreshwar_v2_2_p2 import (
    MoreshwarV22P2Config,
    check_bar_exit_action_p2,
    compute_profit_retention_metrics,
    evaluate_entry_archetypes_p2,
)

DEFAULT_P2_REPORT_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "moreshwar_v2_2_p2_experiment_report.json"
)
DEFAULT_P1_REPORT_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "moreshwar_v2_2_experiment_report.json"
)


def simulate_portfolio_v2_2_p2(
    precomputed: dict[str, Any],
    *,
    experiment_id: str,
    label: str,
    cfg: MoreshwarV22P2Config,
    tariff: DeliveryCostConfig | None = None,
    date_range: tuple[str, str] | None = None,
    initial_capital: float = INITIAL_CAPITAL,
) -> dict[str, Any]:
    """Run strict chronological Prompt 2 trade-management simulation on pure NSE INR."""
    tc = tariff or DeliveryCostConfig()
    uni = precomputed["universe"]
    timeline_full = precomputed["timeline"]
    timeline = [d for d in timeline_full if date_range[0] <= d <= date_range[1]] if date_range else list(timeline_full)
    if len(timeline) < 10:
        return {"experiment_id": experiment_id, "label": label, "metrics": {}, "retention": {}, "trades": []}

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
    dp_charged_pairs: set[tuple[str, str]] = set()
    cooldown_until_idx: dict[str, int] = {}

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
        "partial_exits_booked": 0,
        "unfillable_zero_vol": 0,
        "liquidity_delayed_exits": 0,
        "multi_exit_days": 0,
        "cooldown_skips": 0,
    }
    daily_navs: list[dict[str, Any]] = []
    daily_snaps: list[dict[str, Any]] = []

    for d_idx, dt in enumerate(timeline):
        r6 = regime6_map[dt]["regime"]

        # 1. Execute queued next-bar open orders (`t` signal -> `t+1` open fill)
        for ord_req in pending_orders:
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

            if qty < 1 or (qty * exec_entry_px) < cfg.min_viable_notional_inr:
                continue

            raw_buy = qty * raw_open
            buy_costs = compute_repaired_delivery_costs(
                side="BUY", raw_notional=raw_buy, currency="INR", slippage_bps=slip_bps, tariff=tc
            )
            exec_buy = buy_costs["exec_notional"]
            reg_buy = buy_costs["regulatory_total"]
            if exec_buy + reg_buy > cash:
                qty = int((cash * 0.96) // max(1.0, exec_entry_px * 1.002))
                if qty < 1 or (qty * exec_entry_px) < cfg.min_viable_notional_inr:
                    continue
                raw_buy = qty * raw_open
                buy_costs = compute_repaired_delivery_costs(
                    side="BUY", raw_notional=raw_buy, currency="INR", slippage_bps=slip_bps, tariff=tc
                )
                exec_buy = buy_costs["exec_notional"]
                reg_buy = buy_costs["regulatory_total"]

            for ck in cost_totals:
                cost_totals[ck] += buy_costs.get(ck, 0.0)
            buy_notional_exec_sum += exec_buy
            cash -= (exec_buy + reg_buy)

            eff_sl = round(exec_entry_px * (1.0 - ord_req["sl_ratio"]), 4)
            eff_tp = round(exec_entry_px * (1.0 + ord_req["tp_ratio"]), 4) if ord_req["use_fixed_tp"] else 0.0
            risk_dist = max(0.01, exec_entry_px - eff_sl)
            partial_target = round(exec_entry_px + cfg.partial_book_rr * risk_dist, 4)
            rec_closes = [float(x["close"]) for x in uni[sym]["candles"][max(0, b_idx - 61) : b_idx + 1]]

            open_positions[sym] = {
                "symbol": sym,
                "sector": ord_req["sector"],
                "archetype": ord_req["archetype"],
                "signal_date": ord_req["signal_date"],
                "signal_regime6": ord_req["market_regime"],
                "rank_score": ord_req["rank_score"],
                "atr_entry": ord_req["atr_entry"],
                "atr_pct": ord_req["atr_pct"],
                "entry_date": dt,
                "entry_price": exec_entry_px,
                "last_price": exec_entry_px,
                "max_price_seen": exec_entry_px,
                "max_close_seen": exec_entry_px,
                "qty": qty,
                "initial_qty": qty,
                "raw_buy_notional": raw_buy,
                "initial_raw_buy": raw_buy,
                "exec_buy_notional": exec_buy,
                "buy_reg_cost": reg_buy,
                "buy_slip_cost": buy_costs["slippage_cost"],
                "initial_cost_basis": exec_buy + reg_buy,
                "sl": eff_sl,
                "tp": eff_tp,
                "partial_target": partial_target,
                "partial_booked": False,
                "realized_partial_pnl": 0.0,
                "realized_partial_gross": 0.0,
                "realized_partial_cost": 0.0,
                "breakeven_locked": False,
                "trailing_active": False,
                "slippage_bps": slip_bps,
                "mfe_pct": 0.0,
                "mae_pct": 0.0,
                "bars_held": 0,
                "recent_closes": rec_closes,
            }
        pending_orders = []

        # 2. Check chronological exits & post-close stop ratchets across open positions
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
                cfg=cfg,
            )

            if act["action"] == "DEFER_ZERO_VOL":
                exec_stats["liquidity_delayed_exits"] += 1
                continue

            # Partial exit booking (does not close position completely)
            if act["partial_qty"] > 0 and act["partial_px"] is not None:
                p_qty = int(act["partial_qty"])
                p_raw_px = float(act["partial_px"])
                frac = p_qty / max(1, pos["qty"])
                dp_key = (dt, sym)
                dp_already = dp_key in dp_charged_pairs
                raw_part_sell = p_qty * p_raw_px
                part_sell_costs = compute_repaired_delivery_costs(
                    side="SELL",
                    raw_notional=raw_part_sell,
                    currency="INR",
                    slippage_bps=pos["slippage_bps"],
                    dp_charged_today=dp_already,
                    tariff=tc,
                )
                if not dp_already and part_sell_costs["dp_charge"] > 0:
                    dp_charged_pairs.add(dp_key)
                for ck in cost_totals:
                    cost_totals[ck] += part_sell_costs.get(ck, 0.0)
                exec_part_sell = part_sell_costs["exec_notional"]
                sell_notional_exec_sum += exec_part_sell
                net_part_proceeds = exec_part_sell - part_sell_costs["regulatory_total"]
                cash += net_part_proceeds

                # Proportional allocation of original buy basis & costs
                prop_raw_buy = pos["raw_buy_notional"] * frac
                prop_exec_buy = pos["exec_buy_notional"] * frac
                prop_buy_reg = pos["buy_reg_cost"] * frac
                prop_buy_slip = pos["buy_slip_cost"] * frac

                pos["qty"] -= p_qty
                pos["raw_buy_notional"] -= prop_raw_buy
                pos["exec_buy_notional"] -= prop_exec_buy
                pos["buy_reg_cost"] -= prop_buy_reg
                pos["buy_slip_cost"] -= prop_buy_slip

                part_cost = prop_buy_reg + prop_buy_slip + part_sell_costs["total_cost"]
                part_gross = raw_part_sell - prop_raw_buy
                part_pnl = net_part_proceeds - (prop_exec_buy + prop_buy_reg)

                pos["realized_partial_pnl"] += part_pnl
                pos["realized_partial_gross"] += part_gross
                pos["realized_partial_cost"] += part_cost
                exec_stats["partial_exits_booked"] += 1

            if act["action"] == "FULL_EXIT" and act["exit_px"] is not None:
                exits_today += 1
                if act["is_gap_stop"]:
                    exec_stats["gap_through_stops"] += 1
                raw_exit_px = float(act["exit_px"])
                rem_qty = pos["qty"]
                raw_sell = rem_qty * raw_exit_px
                dp_key = (dt, sym)
                dp_already = dp_key in dp_charged_pairs
                sell_costs = compute_repaired_delivery_costs(
                    side="SELL",
                    raw_notional=raw_sell,
                    currency="INR",
                    slippage_bps=pos["slippage_bps"],
                    dp_charged_today=dp_already,
                    tariff=tc,
                )
                if not dp_already and sell_costs["dp_charge"] > 0:
                    dp_charged_pairs.add(dp_key)
                for ck in cost_totals:
                    cost_totals[ck] += sell_costs.get(ck, 0.0)
                exec_sell = sell_costs["exec_notional"]
                sell_notional_exec_sum += exec_sell
                net_proceeds = exec_sell - sell_costs["regulatory_total"]
                cash += net_proceeds

                rem_gross = raw_sell - pos["raw_buy_notional"]
                rem_cost = pos["buy_reg_cost"] + pos["buy_slip_cost"] + sell_costs["total_cost"]
                rem_pnl = net_proceeds - (pos["exec_buy_notional"] + pos["buy_reg_cost"])

                tot_pnl = pos["realized_partial_pnl"] + rem_pnl
                tot_gross = pos["realized_partial_gross"] + rem_gross
                tot_cost = pos["realized_partial_cost"] + rem_cost
                net_ret_pct = (tot_pnl / max(1e-6, pos["initial_cost_basis"])) * 100.0

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
                    "rank_score": pos["rank_score"],
                    "holding_days": pos["bars_held"],
                    "qty": pos["initial_qty"],
                    "raw_buy_notional": round(pos["initial_raw_buy"], 2),
                    "gross_pnl_amount": round(tot_gross, 2),
                    "pnl_amount": round(tot_pnl, 2),
                    "cost_amount": round(tot_cost, 2),
                    "net_return_pct": round(net_ret_pct, 2),
                    "mfe_pct": round(pos["mfe_pct"], 2),
                    "mae_pct": round(pos["mae_pct"], 2),
                    "exit_reason": ex_reason,
                    "gap_through_stop": act["is_gap_stop"],
                    "partial_booked": pos["partial_booked"],
                })
                to_close.append(sym)

        if exits_today >= 2:
            exec_stats["multi_exit_days"] += 1
        for sym in to_close:
            del open_positions[sym]

        # 3. Record end-of-day NAV & exposure
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

        if is_final_bar:
            continue

        # 4. Evaluate ALL entry candidates across valid universe, rank & allocate
        candidates: list[dict[str, Any]] = []
        r6_obj = regime6_map[dt]
        for sym in valid_syms:
            if sym in open_positions or dt not in features[sym]:
                continue
            if d_idx < cooldown_until_idx.get(sym, -1):
                exec_stats["cooldown_skips"] += 1
                continue
            ft = features[sym][dt]
            b_idx = ft["idx"]
            c_win = uni[sym]["candles"][max(0, b_idx - 61) : b_idx + 1]
            ev = evaluate_entry_archetypes_p2(symbol=sym, feats=ft, candles_window=c_win, regime6=r6_obj, cfg=cfg, tariff=tc)
            if ev["matched"]:
                ev["sector"] = uni[sym].get("sector", "Equity")
                ev["signal_date"] = dt
                ev["recent_closes"] = [float(b["close"]) for b in c_win[-61:]]
                candidates.append(ev)

        if cfg.use_ranked_selection:
            candidates.sort(key=lambda x: (-x["rank_score"], -x["trend_strength"], x["symbol"]))

        tentative_open = [
            {
                "symbol": s,
                "sector": p["sector"],
                "entry_price": p["entry_price"],
                "sl": p["sl"],
                "qty": p["qty"],
                "recent_closes": p.get("recent_closes", []),
                "notional": p["last_price"] * p["qty"],
            }
            for s, p in open_positions.items()
        ]
        deployed_amt = sum(p["notional"] for p in tentative_open)
        avail_cash = cash
        for cand in candidates:
            if len(tentative_open) >= cfg.max_open_positions:
                break
            sec = cand["sector"]
            if sum(1 for p in tentative_open if p["sector"] == sec) >= cfg.max_sector_positions:
                continue
            max_corr = 0.0
            for op_p in tentative_open:
                if op_p["recent_closes"] and cand["recent_closes"]:
                    max_corr = max(max_corr, compute_pairwise_return_correlation(cand["recent_closes"], op_p["recent_closes"], window=60))
            if max_corr > cfg.max_pairwise_corr:
                continue

            existing_sl = sum(max(0.0, (p["entry_price"] - p["sl"]) * p["qty"]) for p in tentative_open)
            rem_sl = max(0.0, nav * (cfg.max_portfolio_stop_risk_pct / 100.0) - existing_sl)
            spot = cand["spot"]
            risk_sh = max(0.01, spot * cand["sl_ratio"])
            if rem_sl < risk_sh:
                continue

            regime_mult = 1.0 if r6 == "STRONG_BULLISH" else (0.80 if r6 == "WEAK_BULLISH" else 0.60)
            corr_mult = 0.65 if max_corr >= 0.65 else 1.0
            eff_risk = min(nav * (cfg.risk_per_trade_pct / 100.0) * regime_mult * corr_mult, rem_sl)

            rem_cap = max(0.0, nav * (cfg.max_portfolio_exposure_pct / 100.0) - deployed_amt)
            allowed_notional = min(nav * (cfg.max_single_stock_pct / 100.0), rem_cap, avail_cash * 0.96)
            qty = max(0, min(int(eff_risk // risk_sh), int(allowed_notional // max(1e-6, spot))))
            if qty < 1 or (qty * spot) < cfg.min_viable_notional_inr:
                continue

            deployed_amt += qty * spot
            avail_cash -= qty * spot
            tentative_open.append({
                "symbol": cand["symbol"],
                "sector": sec,
                "entry_price": spot,
                "sl": spot * (1.0 - cand["sl_ratio"]),
                "qty": qty,
                "recent_closes": cand["recent_closes"],
                "notional": qty * spot,
            })
            pending_orders.append({**cand, "req_qty": qty})

    # Reconcile NAV vs Closed Trades
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

    sig = _compute_statistical_significance(closed_trades)
    wins = [t for t in closed_trades if t["pnl_amount"] > 0]
    losses = [t for t in closed_trades if t["pnl_amount"] <= 0]
    gw = sum(t["pnl_amount"] for t in wins)
    gl = abs(sum(t["pnl_amount"] for t in losses))
    net_pf = round(gw / gl, 2) if gl > 1e-6 else (9.99 if gw > 0 else 0.0)

    ggw = sum(t["gross_pnl_amount"] for t in closed_trades if t["gross_pnl_amount"] > 0)
    ggl = abs(sum(t["gross_pnl_amount"] for t in closed_trades if t["gross_pnl_amount"] <= 0))
    gross_pf = round(ggw / ggl, 2) if ggl > 1e-6 else (9.99 if ggw > 0 else 0.0)

    b_start = bench_closes.get(timeline[0], 0.0)
    b_end = bench_closes.get(timeline[-1], 0.0)
    bench_ret_pct = ((b_end - b_start) / max(1e-6, b_start)) * 100.0 if b_start > 0 else 0.0
    yearly_ret, _ = _compute_calendar_returns(daily_navs, bench_closes, initial_capital)
    retention = compute_profit_retention_metrics(closed_trades)

    net_pnl_sum = sum(t["pnl_amount"] for t in closed_trades)
    metrics = {
        "experiment_id": experiment_id,
        "label": label,
        "entry_mode": cfg.entry_mode,
        "exit_mode": cfg.exit_mode,
        "distinction": "SIMULATED_RESEARCH_BACKTEST (NOT LIVE EXECUTION)",
        "window": f"{timeline[0]} → {timeline[-1]} ({n_days} days)",
        "total_return_pct": round(total_ret, 2),
        "cagr_pct": round(cagr, 2),
        "max_drawdown_pct": round(abs(max_dd), 2),
        "gross_profit_factor": gross_pf,
        "net_profit_factor": net_pf,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "total_trades": len(closed_trades),
        "win_rate_pct": sig["win_rate_pct"],
        "avg_winner_pct": round(sum(t["net_return_pct"] for t in wins) / max(1, len(wins)), 2) if wins else 0.0,
        "avg_loser_pct": round(sum(t["net_return_pct"] for t in losses) / max(1, len(losses)), 2) if losses else 0.0,
        "expectancy_pct": sig["expectancy_pct"],
        "p_value_vs_zero": sig["p_value_vs_zero"],
        "total_cost": round(cost_totals["total_cost"], 2),
        "cost_drag_pct": round((cost_totals["total_cost"] / initial_capital) * 100.0, 2),
        "turnover_multiple_x": round((buy_notional_exec_sum + sell_notional_exec_sum) / initial_capital, 2),
        "avg_holding_days": retention.get("avg_holding_days", 0.0),
        "capital_utilisation_pct": round(sum(s["gross_exposure_pct"] for s in daily_snaps) / max(1, len(daily_snaps)), 1),
        "max_sector_pct": round(max((s["max_sector_pct"] for s in daily_snaps), default=0.0), 2),
        "net_pnl_total": round(net_pnl_sum, 2),
        "nav_change_reconciled": round(end_nav - initial_capital, 2),
        "reconciliation_diff": round((end_nav - initial_capital) - net_pnl_sum, 2),
        "benchmark_return_pct": round(bench_ret_pct, 2),
        "excess_vs_nifty_pp": round(total_ret - bench_ret_pct, 2),
        "winner_mfe_retention_pct": retention.get("winner_mfe_retention_pct", 0.0),
        "net_portfolio_mfe_retention_pct": retention.get("net_portfolio_mfe_retention_pct", 0.0),
        "mean_pos_mfe_capture_pct": retention.get("mean_pos_mfe_capture_pct", 0.0),
        "profitable_1_5pct_closed_loss": retention.get("profitable_1_5pct_closed_loss", 0),
        "premature_horizon_exits": retention.get("premature_horizon_exits", 0),
        "avg_mfe_pct": retention.get("avg_mfe_pct", 0.0),
        "avg_mae_pct": retention.get("avg_mae_pct", 0.0),
        "yearly_returns": yearly_ret,
        "execution_realism": exec_stats,
    }

    return {
        "metrics": metrics,
        "retention": retention,
        "config": asdict(cfg),
    }


def _eval_p2_epochs(
    pre_inr: dict[str, Any],
    *,
    experiment_id: str,
    label: str,
    cfg: MoreshwarV22P2Config,
    tariff: DeliveryCostConfig | None = None,
) -> dict[str, Any]:
    full = simulate_portfolio_v2_2_p2(pre_inr, experiment_id=experiment_id, label=label, cfg=cfg, tariff=tariff, date_range=FULL_RANGE)
    dev = simulate_portfolio_v2_2_p2(pre_inr, experiment_id=f"{experiment_id}_DEV", label=label, cfg=cfg, tariff=tariff, date_range=DEV_TRAIN_RANGE)
    val = simulate_portfolio_v2_2_p2(pre_inr, experiment_id=f"{experiment_id}_VAL", label=label, cfg=cfg, tariff=tariff, date_range=VALIDATION_RANGE)
    is_r = simulate_portfolio_v2_2_p2(pre_inr, experiment_id=f"{experiment_id}_IS", label=label, cfg=cfg, tariff=tariff, date_range=IS_COMBINED_RANGE)
    oos = simulate_portfolio_v2_2_p2(pre_inr, experiment_id=f"{experiment_id}_OOS", label=label, cfg=cfg, tariff=tariff, date_range=OOS_RANGE)

    fm = full["metrics"]
    fm["dev_train_return_pct"] = dev["metrics"].get("total_return_pct", 0.0)
    fm["validation_return_pct"] = val["metrics"].get("total_return_pct", 0.0)
    fm["is_return_pct"] = is_r["metrics"].get("total_return_pct", 0.0)
    fm["oos_return_pct"] = oos["metrics"].get("total_return_pct", 0.0)
    fm["oos_cagr_pct"] = oos["metrics"].get("cagr_pct", 0.0)
    fm["oos_sharpe"] = oos["metrics"].get("sharpe_ratio", 0.0)
    fm["oos_max_dd_pct"] = oos["metrics"].get("max_drawdown_pct", 0.0)
    fm["oos_pf"] = oos["metrics"].get("net_profit_factor", 0.0)
    fm["oos_trades"] = oos["metrics"].get("total_trades", 0)
    fm["oos_win_rate_pct"] = oos["metrics"].get("win_rate_pct", 0.0)
    fm["oos_expectancy_pct"] = oos["metrics"].get("expectancy_pct", 0.0)
    fm["oos_mfe_retention_pct"] = oos["metrics"].get("net_portfolio_mfe_retention_pct", 0.0)
    return full


def run_moreshwar_v2_2_prompt2_suite(
    dataset: dict[str, Any] | None = None,
    *,
    save_report: bool = True,
    report_path: Path = DEFAULT_P2_REPORT_PATH,
) -> dict[str, Any]:
    """Execute complete Prompt 2 (Entries, Exits, Profit Retention & Risk Interaction) Suite."""
    ds = dataset or load_or_build_expanded_dataset(max_age_hours=720.0)
    uni = ds["universe_history"]
    bench = ds["benchmark"]["candles"]
    pre_inr = precompute_v22_universe(uni, bench, currency_filter="INR")
    tariff = DeliveryCostConfig(brokerage_mode="zerodha_delivery")

    # Reference P1 report summary (read-only)
    p1_ref: dict[str, Any] = {}
    if DEFAULT_P1_REPORT_PATH.exists():
        p1_data = json.loads(DEFAULT_P1_REPORT_PATH.read_text(encoding="utf-8"))
        p1_ref = {
            "frozen_m21": p1_data["m21_reconciliation"]["frozen_mixed"],
            "frozen_m21_inr": p1_data["m21_reconciliation"]["frozen_inr_only"],
            "best_p1_selection": p1_data["best_candidate"]["metrics"],
            "benchmarks": p1_data["benchmarks"],
        }

    # --- Phase A: Independent Entry Archetype Evaluation (Exit = P0_EXISTING, 45D) ---
    entry_runs: list[dict[str, Any]] = []
    for eid, lbl, emode, pit_req in [
        ("P2A_M21_BASE", "Entry Control: M21_BASE + PIT Qual (P0 45D)", "M21_BASE", True),
        ("P2A_BREAKOUT", "Entry A: 20D Breakout + Trend + RS60 (P0 45D)", "BREAKOUT", True),
        ("P2A_PULLBACK", "Entry B: Controlled Pullback in Uptrend (P0 45D)", "PULLBACK", True),
        ("P2A_CONTINUATION", "Entry C: Non-Extended Momentum Continuation (P0 45D)", "CONTINUATION", True),
        ("P2A_BREAK_PULL", "Entry A+B: Pullback + Breakout (P0 45D)", "BREAK_PULL", True),
        ("P2A_ALL_THREE", "Entry A+B+C: All Three Causal Archetypes (P0 45D)", "ALL_THREE", True),
    ]:
        c_a = MoreshwarV22P2Config(entry_mode=emode, exit_mode="P0_EXISTING", horizon_days=45, require_pit_filters=pit_req)
        entry_runs.append(_eval_p2_epochs(pre_inr, experiment_id=eid, label=lbl, cfg=c_a, tariff=tariff))

    # --- Phase B: Independent Exit & Profit-Retention Ablation (Fixed Base Entry = M21_BASE + PIT) ---
    exit_runs: list[dict[str, Any]] = []
    for eid, lbl, ex_mode, kw in [
        ("P2B_P0_EXISTING", "Exit P0: Static SL 2.0xATR + TP + 45D Cliff", "P0_EXISTING", {"horizon_days": 45}),
        ("P2B_P1_BE_1_5ATR", "Exit P1: Cost-Covered Breakeven @ +1.5x ATR (60D)", "P1_BREAKEVEN_LOCK", {"horizon_days": 60, "breakeven_trigger_atr": 1.5}),
        ("P2B_P1_BE_2_0ATR", "Exit P1: Cost-Covered Breakeven @ +2.0x ATR (60D)", "P1_BREAKEVEN_LOCK", {"horizon_days": 60, "breakeven_trigger_atr": 2.0}),
        ("P2B_P2_TRAIL_2_0", "Exit P2: 2.0x ATR Trailing Stop (No TP, 60D)", "P2_ATR_TRAILING", {"horizon_days": 60, "trailing_atr_mult": 2.0}),
        ("P2B_P2_TRAIL_2_5", "Exit P2: 2.5x ATR Trailing Stop (No TP, 60D)", "P2_ATR_TRAILING", {"horizon_days": 60, "trailing_atr_mult": 2.5}),
        ("P2B_P2_TRAIL_3_0", "Exit P2: 3.0x ATR Trailing Stop (No TP, 60D)", "P2_ATR_TRAILING", {"horizon_days": 60, "trailing_atr_mult": 3.0}),
        ("P2B_P3_PARTIAL_15R", "Exit P3: 50% Partial @ 1.5R + BE + 2.5x Trail", "P3_PARTIAL_BOOKING", {"horizon_days": 60, "trailing_atr_mult": 2.5, "partial_book_rr": 1.5}),
        ("P2B_P4_ADAPTIVE", "Exit P4: Regime & Momentum Adaptive Trailing", "P4_ADAPTIVE_TRAILING", {"horizon_days": 60, "trailing_atr_mult": 2.4}),
        ("P2B_P5_TIME_MOM", "Exit P5: 15D Stall Cut + 75D Trend Extension", "P5_TIME_MOMENTUM", {"horizon_days": 45, "stagnant_check_days": 15, "max_extended_horizon_days": 75}),
    ]:
        c_b = MoreshwarV22P2Config(entry_mode="M21_BASE", exit_mode=ex_mode, require_pit_filters=True, **kw)
        exit_runs.append(_eval_p2_epochs(pre_inr, experiment_id=eid, label=lbl, cfg=c_b, tariff=tariff))

    # --- Phase C: Explainable Entry x Exit Combinations (Prompt 2 Section 5) ---
    combo_runs: list[dict[str, Any]] = []
    for eid, lbl, emode, ex_mode, kw in [
        ("P2C_BRK_ADAPTIVE", "Strong Sel + Breakout + P4 Adaptive Trail", "BREAKOUT", "P4_ADAPTIVE_TRAILING", {"horizon_days": 60, "trailing_atr_mult": 2.4}),
        ("P2C_PULL_ADAPTIVE", "Strong Sel + Pullback + P4 Adaptive Trail", "PULLBACK", "P4_ADAPTIVE_TRAILING", {"horizon_days": 60, "trailing_atr_mult": 2.4}),
        ("P2C_PULL_TRAIL25", "Strong Sel + Pullback + P2 2.5x ATR Trail", "PULLBACK", "P2_ATR_TRAILING", {"horizon_days": 60, "trailing_atr_mult": 2.5}),
        ("P2C_PULL_TIME_MOM", "Strong Sel + Pullback + P5 Time/Momentum", "PULLBACK", "P5_TIME_MOMENTUM", {"horizon_days": 45, "stagnant_check_days": 15, "max_extended_horizon_days": 75}),
        ("P2C_CONT_TIME_MOM", "Strong Sel + Continuation + P5 Time/Momentum", "CONTINUATION", "P5_TIME_MOMENTUM", {"horizon_days": 45, "stagnant_check_days": 15, "max_extended_horizon_days": 75}),
        ("P2C_CONT_ADAPTIVE", "Strong Sel + Continuation + P4 Adaptive Trail", "CONTINUATION", "P4_ADAPTIVE_TRAILING", {"horizon_days": 60, "trailing_atr_mult": 2.4}),
        ("P2C_ALL3_PARTIAL", "Strong Sel + All-3 Entries + P3 Partial+Trail", "ALL_THREE", "P3_PARTIAL_BOOKING", {"horizon_days": 60, "trailing_atr_mult": 2.5}),
        ("P2C_ALL3_ADAPTIVE", "Strong Sel + All-3 Entries + P4 Adaptive Trail", "ALL_THREE", "P4_ADAPTIVE_TRAILING", {"horizon_days": 60, "trailing_atr_mult": 2.4}),
        ("P2C_ALL3_TIME_MOM", "Strong Sel + All-3 Entries + P5 Time/Momentum", "ALL_THREE", "P5_TIME_MOMENTUM", {"horizon_days": 45, "stagnant_check_days": 15, "max_extended_horizon_days": 75}),
        ("P2C_BRKPULL_TIME", "Strong Sel + Break+Pull + P5 Time/Momentum", "BREAK_PULL", "P5_TIME_MOMENTUM", {"horizon_days": 45, "stagnant_check_days": 15, "max_extended_horizon_days": 75}),
        ("P2C_ALL3_BULL_ADAPT", "Strong Sel + All-3 + BullOnly + P4 Adaptive", "ALL_THREE", "P4_ADAPTIVE_TRAILING", {"horizon_days": 60, "trailing_atr_mult": 2.4, "require_bull_regime_entry": True}),
        ("P2C_PULL_BE20", "Strong Sel + Pullback + P1 BE @ 2.0x ATR", "PULLBACK", "P1_BREAKEVEN_LOCK", {"horizon_days": 60, "breakeven_trigger_atr": 2.0}),
        ("P2C_CONT_BE20", "Strong Sel + Continuation + P1 BE @ 2.0x ATR", "CONTINUATION", "P1_BREAKEVEN_LOCK", {"horizon_days": 60, "breakeven_trigger_atr": 2.0}),
        ("P2C_PULLCONT_BE20", "Strong Sel + Pull+Cont + P1 BE @ 2.0x ATR", "PULL_CONT", "P1_BREAKEVEN_LOCK", {"horizon_days": 60, "breakeven_trigger_atr": 2.0}),
        ("P2C_PULLCONT_ADAPT", "Strong Sel + Pull+Cont + P4 Adaptive Trail", "PULL_CONT", "P4_ADAPTIVE_TRAILING", {"horizon_days": 60, "trailing_atr_mult": 2.4}),
    ]:
        c_c = MoreshwarV22P2Config(entry_mode=emode, exit_mode=ex_mode, require_pit_filters=True, **kw)
        combo_runs.append(_eval_p2_epochs(pre_inr, experiment_id=eid, label=lbl, cfg=c_c, tariff=tariff))

    # Select strongest overall candidate balancing 5Y Return, Sharpe, NetPF, MaxDD & Untouched OOS (>= 120 trades)
    all_candidates = entry_runs + exit_runs + combo_runs
    credible = [c for c in all_candidates if c["metrics"]["total_trades"] >= 120]
    best_p2 = max(
        credible,
        key=lambda c: (
            1 if c["metrics"]["oos_return_pct"] > 0.0 else 0,
            round(c["metrics"]["net_profit_factor"], 2),
            c["metrics"]["sharpe_ratio"],
            c["metrics"]["total_return_pct"] - 0.5 * c["metrics"]["max_drawdown_pct"],
        ),
    )
    best_cfg = MoreshwarV22P2Config(**best_p2["config"])

    # --- Phase D: Portfolio Interaction, Re-entry Cooldown, Capacity, Slippage & 4-Fold Walk-Forward ---
    portfolio_interactions: list[dict[str, Any]] = []
    for cap_n in (4, 5, 6, 8):
        c_cap = replace(
            best_cfg,
            max_open_positions=cap_n,
            max_single_stock_pct=min(25.0, max(15.0, round(100.0 / max(1, cap_n - 1), 1))),
        )
        portfolio_interactions.append(
            _eval_p2_epochs(pre_inr, experiment_id=f"P2D_CAP_{cap_n}", label=f"Best P2 | Capacity={cap_n} Slots", cfg=c_cap, tariff=tariff)["metrics"]
        )

    for cd in (0, 5, 10):
        c_cd = replace(best_cfg, reentry_cooldown_days=cd)
        portfolio_interactions.append(
            _eval_p2_epochs(pre_inr, experiment_id=f"P2D_COOLDOWN_{cd}D", label=f"Best P2 | Re-entry Cooldown={cd}D", cfg=c_cd, tariff=tariff)["metrics"]
        )

    slippage_stress = []
    for bps in (7.5, 12.5, 20.0):
        sr = _eval_p2_epochs(
            pre_inr,
            experiment_id=f"P2D_SLIP_{bps}BPS",
            label=f"Best P2 | Slippage={bps} bps",
            cfg=best_cfg,
            tariff=DeliveryCostConfig(brokerage_mode="zerodha_delivery", base_slippage_bps=bps),
        )["metrics"]
        slippage_stress.append({
            "slippage_bps": bps,
            "full_5y_pct": sr["total_return_pct"],
            "is_pct": sr["is_return_pct"],
            "val_pct": sr["validation_return_pct"],
            "oos_pct": sr["oos_return_pct"],
            "sharpe": sr["sharpe_ratio"],
            "net_pf": sr["net_profit_factor"],
            "max_dd_pct": sr["max_drawdown_pct"],
        })

    # Rolling 4-Fold Chronological Walk-Forward Check on Best P2 Candidate
    tl = pre_inr["timeline"]
    n_t = len(tl)
    q = n_t // 4
    wf_folds = []
    for fi in range(4):
        s_i = fi * q
        e_i = (n_t - 1) if fi == 3 else ((fi + 1) * q - 1)
        f_range = (tl[s_i], tl[e_i])
        f_sim = simulate_portfolio_v2_2_p2(pre_inr, experiment_id=f"WF_{fi+1}", label=f"WF-{fi+1}", cfg=best_cfg, tariff=tariff, date_range=f_range)
        fm = f_sim["metrics"]
        wf_folds.append({
            "fold": f"WF-{fi+1}" + (" (UNTOUCHED OOS TAIL)" if fi == 3 else ""),
            "window": f"{f_range[0]} → {f_range[1]}",
            "return_pct": fm.get("total_return_pct", 0.0),
            "nifty_pct": fm.get("benchmark_return_pct", 0.0),
            "sharpe": fm.get("sharpe_ratio", 0.0),
            "max_dd_pct": fm.get("max_drawdown_pct", 0.0),
            "net_pf": fm.get("net_profit_factor", 0.0),
            "trades": fm.get("total_trades", 0),
            "win_rate_pct": fm.get("win_rate_pct", 0.0),
            "mfe_retention_pct": fm.get("net_portfolio_mfe_retention_pct", 0.0),
        })

    report = {
        "research_version": "2.2.0-prompt2",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "promoted_to_production": False,
        "prompt1_reference": p1_ref,
        "phase_a_entries": [r["metrics"] for r in entry_runs],
        "phase_b_exits": [r["metrics"] for r in exit_runs],
        "phase_c_combinations": [r["metrics"] for r in combo_runs],
        "phase_d_portfolio_interactions": portfolio_interactions,
        "slippage_stress": slippage_stress,
        "walk_forward_4fold": wf_folds,
        "best_trade_management": {
            "metrics": best_p2["metrics"],
            "retention": best_p2["retention"],
            "config": best_p2["config"],
        },
    }

    if save_report:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    return report


if __name__ == "__main__":
    out = run_moreshwar_v2_2_prompt2_suite(save_report=True)
    bm = out["best_trade_management"]["metrics"]
    print(json.dumps({
        "best_p2": {k: bm.get(k) for k in (
            "experiment_id", "label", "total_return_pct", "dev_train_return_pct",
            "validation_return_pct", "oos_return_pct", "cagr_pct", "max_drawdown_pct",
            "net_profit_factor", "sharpe_ratio", "sortino_ratio", "win_rate_pct",
            "expectancy_pct", "total_cost", "total_trades", "winner_mfe_retention_pct",
            "net_portfolio_mfe_retention_pct", "profitable_1_5pct_closed_loss",
        )},
        "wf_folds": out["walk_forward_4fold"],
    }, indent=2))
