"""Moreshwar 3.0 Prompt 3 — Common Fair-Comparison Portfolio Simulator & Cost Engine.

Provides an isolated, stateless-per-run daily portfolio simulation harness so all
five independent strategy families (A..E) execute on the EXACT same:
- Prompt 1 Point-in-Time NSE Equity Universe (`is_pit_elig`, ETFs excluded)
- Trading calendar (`2021-01-01 -> 2026-11-19`) & IS/OOS splits
- Capital assumptions (`initial_capital = Rs 100,000`)
- Zerodha Equity Delivery transaction costs + 7.5 bps one-way slippage
- Strictly causal signal t close -> entry at t+1 open
- Daily mark-to-market equity curve, drawdown, Nifty 50 benchmark, and trade ledger
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np

from finsistant_v2.moreshwar_v2_2 import DeliveryCostConfig, compute_repaired_delivery_costs

IS_CUTOFF_DATE = "2025-03-31"
OOS_START_DATE = "2025-04-01"
EVAL_START_DATE = "2021-07-01"
INITIAL_CAPITAL_M3 = 100_000.0
RISK_FREE_ANNUAL_M3 = 0.065


@dataclass(frozen=True)
class StrategySimConfig:
    strategy_id: str
    strategy_name: str
    family: str
    max_positions: int = 10
    max_single_weight_pct: float = 15.0
    max_sector_positions: int = 3
    max_portfolio_exposure_pct: float = 95.0
    risk_per_trade_pct: float = 1.0
    rebalance_mode: str = "DAILY_EVENT"  # "DAILY_EVENT" | "PERIODIC"
    rebalance_interval_days: int = 20
    hold_max_days: int = 40
    initial_atr_stop_mult: float = 2.0
    trailing_atr_mult: float = 0.0
    breakeven_trigger_atr: float = 0.0
    target_atr_mult: float = 0.0
    max_daily_vol_participation: float = 0.01
    slippage_bps: float = 7.5
    cooldown_days: int = 5


def compute_perf_metrics(
    *,
    strategy_id: str,
    strategy_name: str,
    family: str,
    daily_navs: list[dict[str, Any]],
    closed_trades: list[dict[str, Any]],
    total_cost: float,
    turnover_buy_sell: float,
    avg_util_pct: float,
    bench_rets: list[float],
    initial_capital: float,
    reconciliation_diff: float,
) -> dict[str, Any]:
    n_days = len(daily_navs)
    if n_days < 2:
        return {"strategy_id": strategy_id, "error": "Insufficient days"}
    navs = np.array([d["nav"] for d in daily_navs], dtype=np.float64)
    rets = np.zeros(n_days, dtype=np.float64)
    rets[1:] = (navs[1:] / np.maximum(1.0, navs[:-1])) - 1.0
    tot_ret_pct = ((navs[-1] / initial_capital) - 1.0) * 100.0
    years = max(0.05, n_days / 252.0)
    cagr_pct = (((navs[-1] / initial_capital) ** (1.0 / years)) - 1.0) * 100.0

    pk = np.maximum.accumulate(navs)
    dds = (navs - pk) / np.maximum(1.0, pk)
    max_dd_pct = abs(float(np.min(dds))) * 100.0

    rf_d = RISK_FREE_ANNUAL_M3 / 252.0
    ex = rets[1:] - rf_d
    mu_e = float(np.mean(ex))
    sd_e = float(np.std(ex, ddof=1)) if len(ex) > 1 else 1.0
    sharpe = (mu_e / sd_e) * math.sqrt(252.0) if sd_e > 1e-10 else 0.0
    down = ex[ex < 0.0]
    ds_sd = math.sqrt(float(np.mean(down ** 2))) if len(down) > 0 else 1e-9
    sortino = (mu_e / ds_sd) * math.sqrt(252.0) if ds_sd > 1e-10 else 0.0
    ann_vol = float(np.std(rets[1:], ddof=1)) * math.sqrt(252.0) * 100.0 if len(rets) > 2 else 0.0

    # Benchmark return over identical daily dates
    b_arr = np.array(bench_rets[1:], dtype=np.float64)
    b_cum = (float(np.prod(1.0 + np.where(np.isnan(b_arr), 0.0, b_arr))) - 1.0) * 100.0

    n_tr = len(closed_trades)
    if n_tr > 0:
        pnls = [float(t["pnl_amount"]) for t in closed_trades]
        r_pcts = [float(t["net_return_pct"]) for t in closed_trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        net_pf = (sum(wins) / max(1e-6, abs(sum(losses)))) if losses else (99.0 if wins else 0.0)
        win_rate = 100.0 * len(wins) / n_tr
        exp_pct = sum(r_pcts) / n_tr
        avg_hold = sum(int(t["holding_days"]) for t in closed_trades) / n_tr
    else:
        net_pf, win_rate, exp_pct, avg_hold = 0.0, 0.0, 0.0, 0.0

    cash_ret_pct = (((1.0 + RISK_FREE_ANNUAL_M3) ** years) - 1.0) * 100.0
    return {
        "strategy_id": strategy_id,
        "strategy_name": strategy_name,
        "family": family,
        "window": f"{daily_navs[0]['date']} -> {daily_navs[-1]['date']} ({n_days}d)",
        "total_return_pct": round(tot_ret_pct, 2),
        "cagr_pct": round(cagr_pct, 2),
        "max_drawdown_pct": round(max_dd_pct, 2),
        "sharpe_ratio": round(sharpe, 2),
        "sortino_ratio": round(sortino, 2),
        "annualised_vol_pct": round(ann_vol, 2),
        "net_profit_factor": round(min(99.0, net_pf), 2),
        "total_trades": n_tr,
        "win_rate_pct": round(win_rate, 2),
        "expectancy_pct": round(exp_pct, 3),
        "avg_holding_days": round(avg_hold, 1),
        "total_cost_inr": round(total_cost, 2),
        "cost_drag_pct": round((total_cost / initial_capital) * 100.0, 2),
        "turnover_multiple_x": round(turnover_buy_sell / initial_capital, 2),
        "capital_utilisation_pct": round(avg_util_pct, 1),
        "benchmark_return_pct": round(b_cum, 2),
        "excess_vs_nifty_pp": round(tot_ret_pct - b_cum, 2),
        "cash_65_return_pct": round(cash_ret_pct, 2),
        "reconciliation_diff": round(reconciliation_diff, 2),
    }


def simulate_independent_strategy_m3(
    *,
    panel: dict[str, Any],
    cfg: StrategySimConfig,
    candidate_fn: Callable[[int, dict[str, Any], dict[int, dict[str, Any]]], list[dict[str, Any]]],
    date_start: str = EVAL_START_DATE,
    date_end: str | None = None,
    initial_capital: float = INITIAL_CAPITAL_M3,
    tariff: DeliveryCostConfig | None = None,
) -> dict[str, Any]:
    """Run stateless simulation for one strategy family over PIT NumPy panel matrices.

    `candidate_fn(t_idx, panel, open_positions)` returns a ranked list of candidate dicts:
    `{"col": int, "score": float, "atr": float, "stop_ratio": float | None, "target_ratio": float | None}`
    Signals evaluated at day `t` close strictly fill at `t+1` open.
    """
    tc = tariff or DeliveryCostConfig(brokerage_mode="zerodha_delivery", base_slippage_bps=cfg.slippage_bps)
    dates = panel["dates"]
    end_dt = date_end or str(dates[-1])
    t_indices = [i for i, d in enumerate(dates) if date_start <= str(d) <= end_dt]
    if len(t_indices) < 20:
        return {"config": asdict(cfg), "metrics": {}, "trades": [], "daily_navs": []}

    open_p = panel["adj_open"]
    high_p = panel["adj_high"]
    low_p = panel["adj_low"]
    close_p = panel["adj_close"]
    vol_p = panel["volume"]
    sectors = panel["sectors"]
    sec_ids = panel["sec_ids"]
    cur_syms = panel["current_symbols"]
    n50_c = panel["nifty50_close"]

    cash = float(initial_capital)
    open_pos: dict[int, dict[str, Any]] = {}  # keyed by panel column index `col`
    pending_buys: list[dict[str, Any]] = []
    pending_rebal_sells: set[int] = set()
    cooldown_until: dict[int, int] = {}
    dp_pairs: set[tuple[str, str]] = set()

    closed_trades: list[dict[str, Any]] = []
    daily_navs: list[dict[str, Any]] = []
    daily_rets: list[float] = []
    bench_rets: list[float] = []
    sim_signals_log: list[dict[str, Any]] = []

    total_cost = 0.0
    turnover_bs = 0.0
    util_samples: list[float] = []

    t_first, t_last = t_indices[0], t_indices[-1]
    prev_nav = initial_capital

    for step_k, ti in enumerate(t_indices):
        dt = str(dates[ti])
        prev_ti = ti - 1

        # Benchmark daily return
        if ti > 0 and not np.isnan(n50_c[ti]) and not np.isnan(n50_c[prev_ti]) and n50_c[prev_ti] > 0:
            bench_rets.append(float(n50_c[ti] / n50_c[prev_ti] - 1.0))
        else:
            bench_rets.append(0.0)

        # 1. Process Rebalance Exits + Entry Fills strictly at Day `ti` Open
        for col in list(pending_rebal_sells):
            if col not in open_pos or not panel["is_valid_eq"][ti, col]:
                continue
            pos = open_pos[col]
            raw_open = float(open_p[ti, col])
            bar_vol = float(vol_p[ti, col])
            if np.isnan(raw_open) or raw_open <= 0 or bar_vol <= 0:
                continue
            raw_sell = pos["qty"] * raw_open
            sym = str(cur_syms[col])
            sc = compute_repaired_delivery_costs(
                side="SELL", raw_notional=raw_sell, currency="INR",
                slippage_bps=cfg.slippage_bps, dp_charged_today=((dt, sym) in dp_pairs), tariff=tc,
            )
            if sc["dp_charge"] > 0:
                dp_pairs.add((dt, sym))
            net_proc = sc["exec_notional"] - sc["regulatory_total"]
            cash += net_proc
            total_cost += sc["regulatory_total"] + sc["slippage_cost"]
            turnover_bs += sc["exec_notional"]
            pnl_amt = net_proc - pos["cost_basis"]
            closed_trades.append({
                "sec_id": str(sec_ids[col]),
                "symbol": sym,
                "sector": str(sectors[col]),
                "entry_date": pos["entry_date"],
                "exit_date": dt,
                "holding_days": pos["bars_held"],
                "pnl_amount": round(pnl_amt, 2),
                "net_return_pct": round((pnl_amt / max(1e-6, pos["cost_basis"])) * 100.0, 2),
                "exit_reason": "REBALANCE_ROTATE",
            })
            del open_pos[col]
        pending_rebal_sells.clear()

        # Execute pending BUY fills at open_p[ti, col]
        for ord_req in pending_buys:
            col = int(ord_req["col"])
            if col in open_pos or not panel["is_valid_eq"][ti, col]:
                continue
            if not panel["is_pit_elig"][ti, col]:
                continue
            raw_open = float(open_p[ti, col])
            bar_vol = float(vol_p[ti, col])
            if np.isnan(raw_open) or raw_open <= 0 or bar_vol <= 0:
                continue
            exec_px = raw_open * (1.0 + cfg.slippage_bps / 10_000.0)
            req_qty = min(int(ord_req["req_qty"]), int((cash * 0.96) // max(1.0, exec_px)), max(1, int(bar_vol * cfg.max_daily_vol_participation)))
            if req_qty < 1 or (req_qty * exec_px) < 1500.0:
                continue
            raw_buy = req_qty * raw_open
            bc = compute_repaired_delivery_costs(
                side="BUY", raw_notional=raw_buy, currency="INR", slippage_bps=cfg.slippage_bps, tariff=tc
            )
            tot_buy = bc["exec_notional"] + bc["regulatory_total"]
            if tot_buy > cash:
                continue
            cash -= tot_buy
            total_cost += bc["regulatory_total"] + bc["slippage_cost"]
            turnover_bs += bc["exec_notional"]
            sl_r = float(ord_req.get("stop_ratio") or 0.06)
            tp_r = ord_req.get("target_ratio")
            open_pos[col] = {
                "col": col,
                "entry_date": dt,
                "entry_px": exec_px,
                "last_px": exec_px,
                "max_px": exec_px,
                "qty": req_qty,
                "cost_basis": tot_buy,
                "sl_px": exec_px * (1.0 - sl_r) if sl_r > 0 else 0.0,
                "tp_px": exec_px * (1.0 + float(tp_r)) if tp_r else 0.0,
                "atr_entry": float(ord_req.get("atr") or (exec_px * 0.025)),
                "be_locked": False,
                "bars_held": 0,
            }
        pending_buys.clear()

        # 2. Check Intraday/EOD Exit Conditions on Open Positions (Causal!)
        is_last = (ti == t_last)
        for col in list(open_pos.keys()):
            if not panel["is_valid_eq"][ti, col]:
                if is_last:
                    # Force mark/exit at last known price cleanly
                    pos = open_pos[col]
                    raw_sell = pos["qty"] * pos["last_px"]
                    sym = str(cur_syms[col])
                    sc = compute_repaired_delivery_costs(side="SELL", raw_notional=raw_sell, currency="INR", slippage_bps=cfg.slippage_bps, dp_charged_today=((dt, sym) in dp_pairs), tariff=tc)
                    net_proc = sc["exec_notional"] - sc["regulatory_total"]
                    cash += net_proc
                    total_cost += sc["regulatory_total"] + sc["slippage_cost"]
                    turnover_bs += sc["exec_notional"]
                    pnl_amt = net_proc - pos["cost_basis"]
                    closed_trades.append({
                        "sec_id": str(sec_ids[col]), "symbol": sym, "sector": str(sectors[col]),
                        "entry_date": pos["entry_date"], "exit_date": dt, "holding_days": pos["bars_held"],
                        "pnl_amount": round(pnl_amt, 2), "net_return_pct": round((pnl_amt / max(1e-6, pos["cost_basis"])) * 100.0, 2),
                        "exit_reason": "FINAL_BAR_CLOSE",
                    })
                    del open_pos[col]
                continue

            pos = open_pos[col]
            o_v, h_v, l_v, c_v, v_v = float(open_p[ti, col]), float(high_p[ti, col]), float(low_p[ti, col]), float(close_p[ti, col]), float(vol_p[ti, col])
            if np.isnan(c_v) or c_v <= 0:
                continue
            pos["bars_held"] += 1
            pos["last_px"] = c_v
            if v_v <= 0:
                continue

            exit_px: float | None = None
            reason = ""
            # Gap-through stop fills strictly at open
            if pos["sl_px"] > 0 and o_v <= pos["sl_px"]:
                exit_px = o_v
                reason = "STOP_LOSS_BE_GAP" if pos["be_locked"] else "STOP_LOSS_GAP"
            elif pos["sl_px"] > 0 and l_v <= pos["sl_px"]:
                exit_px = pos["sl_px"]
                reason = "STOP_LOSS_BE" if pos["be_locked"] else "STOP_LOSS"
            elif pos["tp_px"] > 0 and o_v >= pos["tp_px"]:
                exit_px = o_v
                reason = "TARGET_GAP"
            elif pos["tp_px"] > 0 and h_v >= pos["tp_px"]:
                exit_px = pos["tp_px"]
                reason = "TARGET"
            elif cfg.hold_max_days > 0 and pos["bars_held"] >= cfg.hold_max_days:
                exit_px = c_v
                reason = "HORIZON_EXIT"
            elif is_last:
                exit_px = c_v
                reason = "FINAL_BAR_CLOSE"

            if exit_px is not None:
                raw_sell = pos["qty"] * exit_px
                sym = str(cur_syms[col])
                sc = compute_repaired_delivery_costs(
                    side="SELL", raw_notional=raw_sell, currency="INR",
                    slippage_bps=cfg.slippage_bps, dp_charged_today=((dt, sym) in dp_pairs), tariff=tc,
                )
                if sc["dp_charge"] > 0:
                    dp_pairs.add((dt, sym))
                net_proc = sc["exec_notional"] - sc["regulatory_total"]
                cash += net_proc
                total_cost += sc["regulatory_total"] + sc["slippage_cost"]
                turnover_bs += sc["exec_notional"]
                pnl_amt = net_proc - pos["cost_basis"]
                closed_trades.append({
                    "sec_id": str(sec_ids[col]), "symbol": sym, "sector": str(sectors[col]),
                    "entry_date": pos["entry_date"], "exit_date": dt, "holding_days": pos["bars_held"],
                    "pnl_amount": round(pnl_amt, 2),
                    "net_return_pct": round((pnl_amt / max(1e-6, pos["cost_basis"])) * 100.0, 2),
                    "exit_reason": reason,
                })
                if "STOP_LOSS" in reason and cfg.cooldown_days > 0:
                    cooldown_until[col] = ti + cfg.cooldown_days
                del open_pos[col]
            else:
                # Causal end-of-bar trailing stop / breakeven update (NEVER affects same bar!)
                pos["max_px"] = max(pos["max_px"], h_v)
                if cfg.breakeven_trigger_atr > 0 and not pos["be_locked"]:
                    if c_v >= pos["entry_px"] + cfg.breakeven_trigger_atr * pos["atr_entry"]:
                        be_px = pos["entry_px"] * 1.0065
                        if be_px > pos["sl_px"]:
                            pos["sl_px"] = be_px
                            pos["be_locked"] = True
                if cfg.trailing_atr_mult > 0:
                    tr_px = pos["max_px"] - cfg.trailing_atr_mult * pos["atr_entry"]
                    if tr_px > pos["sl_px"]:
                        pos["sl_px"] = tr_px

        # 3. End-of-day NAV & Utilisation
        dep_mkt = sum(p["last_px"] * p["qty"] for p in open_pos.values())
        nav = cash + dep_mkt
        util_samples.append((dep_mkt / max(1.0, nav)) * 100.0)
        d_ret = (nav / max(1.0, prev_nav)) - 1.0 if daily_navs else 0.0
        daily_navs.append({"date": dt, "nav": round(nav, 2), "cash": round(cash, 2), "open_positions": len(open_pos), "ret": d_ret})
        daily_rets.append(d_ret)
        prev_nav = nav

        # 4. Generate Next-Day Entry Signals at Day `ti` Close (skip last day)
        if is_last:
            continue
        is_rebal_day = (step_k % max(1, cfg.rebalance_interval_days) == 0)
        if not is_rebal_day:
            continue

        cands = candidate_fn(ti, panel, open_pos)
        sim_signals_log.append({
            "date": dt,
            "raw_candidates": len(cands),
            "top_sec_ids": [str(sec_ids[c["col"]]) for c in cands[: cfg.max_positions]],
        })

        if cfg.rebalance_mode == "PERIODIC":
            keep_n = max(cfg.max_positions * 2, min(len(cands), 25))
            target_cols = [c["col"] for c in cands[:keep_n]]
            target_set = set(target_cols)
            for col in open_pos:
                if col not in target_set:
                    pending_rebal_sells.add(col)

        # Allocate available slots & capital respecting limits
        sec_counts: dict[str, int] = {}
        for col in open_pos:
            if col not in pending_rebal_sells:
                s_nm = str(sectors[col])
                sec_counts[s_nm] = sec_counts.get(s_nm, 0) + 1
        cur_open_cnt = len(open_pos) - len(pending_rebal_sells)
        projected_cash = cash + sum(open_pos[c]["last_px"] * open_pos[c]["qty"] * 0.99 for c in pending_rebal_sells)
        projected_dep = sum(p["last_px"] * p["qty"] for c, p in open_pos.items() if c not in pending_rebal_sells)
        max_exp_amt = nav * (cfg.max_portfolio_exposure_pct / 100.0)
        per_slot_cap = min(nav * (cfg.max_single_weight_pct / 100.0), max_exp_amt / max(1, cfg.max_positions))

        for c in cands:
            if cur_open_cnt >= cfg.max_positions:
                break
            col = int(c["col"])
            if col in open_pos or ti < cooldown_until.get(col, -1):
                continue
            s_nm = str(sectors[col])
            if sec_counts.get(s_nm, 0) >= cfg.max_sector_positions:
                continue
            spot = float(close_p[ti, col])
            if np.isnan(spot) or spot <= 0:
                continue
            sl_r = float(c.get("stop_ratio") or 0.06)
            # Risk-budget & max-slot allocation cap
            risk_notional = (nav * (cfg.risk_per_trade_pct / 100.0)) / max(0.02, sl_r)
            rem_exp = max(0.0, max_exp_amt - projected_dep)
            alloc_amt = min(per_slot_cap, risk_notional, rem_exp, projected_cash * 0.96)
            qty = int(alloc_amt // spot)
            if qty < 1 or (qty * spot) < 1500.0:
                continue
            pending_buys.append({**c, "req_qty": qty})
            cur_open_cnt += 1
            sec_counts[s_nm] = sec_counts.get(s_nm, 0) + 1
            projected_cash -= qty * spot
            projected_dep += qty * spot

    # Cash-to-NAV Reconciliation (at end of simulation all positions are closed so final_nav == cash)
    final_nav = cash + sum(p["last_px"] * p["qty"] for p in open_pos.values())
    if daily_navs:
        daily_navs[-1]["nav"] = round(final_nav, 2)
    rec_diff = round(final_nav, 2) - (daily_navs[-1]["nav"] if daily_navs else round(initial_capital, 2))

    metrics = compute_perf_metrics(
        strategy_id=cfg.strategy_id,
        strategy_name=cfg.strategy_name,
        family=cfg.family,
        daily_navs=daily_navs,
        closed_trades=closed_trades,
        total_cost=total_cost,
        turnover_buy_sell=turnover_bs,
        avg_util_pct=float(np.mean(util_samples)) if util_samples else 0.0,
        bench_rets=bench_rets,
        initial_capital=initial_capital,
        reconciliation_diff=rec_diff,
    )
    return {
        "config": asdict(cfg),
        "metrics": metrics,
        "trades": closed_trades,
        "daily_navs": daily_navs,
        "signals_log": sim_signals_log,
    }
