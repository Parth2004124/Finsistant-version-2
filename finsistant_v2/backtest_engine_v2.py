"""Experimental Quantitative Backtesting Engine for Moreshwar 2.0 (Phases 4–7).

Features:
- Strictly chronological, portfolio-level daily simulation (capital constraint, integer shares
  where enabled, 5 bps entry/exit slippage + 25 bps statutory costs, zero overlapping same-symbol trades).
- Fixes Historical Data Integrity (Phase 4): explicitly refuses to leak today's static fundamentals
  into past dates for Moreshwar 2.0 historical tests (`historical_unavailable` neutral treatment +
  confidence haircut), while preserving an explicit Model A baseline and static-reference control for comparison.
- Executes controlled Phase 5 isolation suite (Models A..F + Benchmarks), Phase 6 Sensitivity Sweep +
  Train / Validation / Out-of-Sample + Rolling Walk-Forward Validation, and full Phase 7 metrics.
"""

from __future__ import annotations

from datetime import datetime, timezone
import math
from typing import Any

from .moreshwar_v2 import (
    MoreshwarV2Config,
    classify_market_regime,
    compute_atr,
    evaluate_moreshwar_v2,
)
from .quant_engine import compute_ohlcv_and_regression_from_candles
from .scoring_engine import calculate_decision_matrix, calculate_moreshwar_levels

ENTRY_SLIPPAGE_BPS = 5.0
EXIT_SLIPPAGE_BPS = 5.0
STATUTORY_COST_BPS = 25.0  # Round-trip brokerage + STT + Exchange + SEBI + Stamp + GST
RISK_FREE_ANNUAL = 0.065
INITIAL_CAPITAL = 100_000.0


def _causal_tech_eval(quant: dict[str, Any]) -> tuple[str, float]:
    latest = quant["candles"][-1]
    price = float(latest["close"])
    sma_20 = float(latest["sma_20"])
    ema_12 = float(latest["ema_12"])
    ema_26 = float(latest["ema_26"])
    macd = float(latest["macd"])
    macd_sig = float(latest["macd_signal"])
    rsi = float(latest["rsi_14"])
    proj_30d = float(quant["projected_30d_target"])

    score = 0.50
    score += 0.14 if price >= sma_20 else -0.12
    score += 0.12 if ema_12 >= ema_26 else -0.10
    score += 0.08 if macd >= macd_sig else -0.08
    score += 0.10 if proj_30d > price else -0.10
    if 46.0 <= rsi <= 70.0:
        score += 0.07
    conf = round(max(0.18, min(0.96, score)), 3)
    sig = "BULLISH" if conf >= 0.58 else ("BEARISH" if conf <= 0.44 else "NEUTRAL")
    return sig, conf


def _simulate_chronological_portfolio(
    universe_history: dict[str, dict[str, Any]],
    benchmark_candles: list[dict[str, Any]] | None,
    *,
    model_name: str,
    entry_style: str,  # "matrix_static" | "causal_momentum" | "sma_fixed"
    exit_style: str,  # "moreshwar_1" | "fixed_5_10" | "moreshwar_2"
    cfg: MoreshwarV2Config | None = None,
    use_atr_stops: bool = False,
    use_vol_sizing: bool = False,
    use_regime_filter: bool = False,
    use_conf_targets: bool = False,
    fundamental_mode: str = "point_in_time_static_warning",  # "historical_unavailable" | "point_in_time_static_warning"
    date_range: tuple[str, str] | None = None,
    initial_capital: float = INITIAL_CAPITAL,
) -> dict[str, Any]:
    """Run a strict day-by-day chronological portfolio simulation across all stocks."""
    c = cfg or MoreshwarV2Config()

    # Build aligned sorted trading calendar across universe bars (after 45-bar indicator warmup)
    date_bars_by_sym: dict[str, dict[str, tuple[int, dict[str, Any]]]] = {}
    all_dates_set: set[str] = set()
    valid_syms: list[str] = []

    for sym, snap in universe_history.items():
        candles = snap.get("candles") or []
        if len(candles) < 62:
            continue
        valid_syms.append(sym)
        sym_map: dict[str, tuple[int, dict[str, Any]]] = {}
        for idx, bar in enumerate(candles):
            dt = str(bar.get("date", ""))
            sym_map[dt] = (idx, bar)
            if idx >= 45:
                all_dates_set.add(dt)
        date_bars_by_sym[sym] = sym_map

    timeline = sorted(all_dates_set)
    if date_range:
        lo_dt, hi_dt = date_range
        timeline = [d for d in timeline if lo_dt <= d <= hi_dt]

    if len(timeline) < 5:
        return {
            "model": model_name,
            "metrics": _empty_phase7_metrics(model_name),
            "regimes": {},
            "by_stock": {},
            "by_sector": {},
            "equity_curve": [],
        }

    # Pre-compute causal market regime once per date (strictly using benchmark bars <= d)
    regime_by_date: dict[str, dict[str, Any]] = {
        d: classify_market_regime(benchmark_candles, decision_date=d) for d in timeline
    }

    cash = float(initial_capital)
    open_positions: dict[str, dict[str, Any]] = {}
    pending_orders: list[dict[str, Any]] = []
    closed_trades: list[dict[str, Any]] = []

    rejected_count = 0
    regime_blocked_count = 0
    total_cost_amount = 0.0
    daily_navs: list[dict[str, Any]] = []
    daily_utilizations: list[float] = []
    sl_count = 0
    tp_count = 0
    horizon_count = 0

    for d_idx, dt in enumerate(timeline):
        day_regime = regime_by_date[dt]["regime"]

        # 1. Execute pending orders at today's open (strict next-bar execution)
        next_pending: list[dict[str, Any]] = []
        for ord_req in pending_orders:
            sym = ord_req["symbol"]
            if sym in open_positions or dt not in date_bars_by_sym[sym]:
                continue
            b_idx, bar = date_bars_by_sym[sym][dt]
            raw_open = float(bar.get("open") or bar.get("close") or 0.0)
            if raw_open <= 0:
                continue
            fill_px = round(raw_open * (1.0 + ENTRY_SLIPPAGE_BPS / 10000.0), 4)

            # Compute size based on portfolio cash & model rules
            if use_vol_sizing:
                deployed = sum(p["entry_price"] * p["qty"] for p in open_positions.values())
                eval_res = evaluate_moreshwar_v2(
                    symbol=sym,
                    candles=universe_history[sym]["candles"][max(0, b_idx - 50) : b_idx + 1],
                    spot_price=fill_px,
                    fundamental_score=ord_req.get("f_tot"),
                    porters_score=ord_req.get("p_tot"),
                    missing_fields=ord_req.get("missing_fields"),
                    fundamental_mode=fundamental_mode,
                    tech_signal=ord_req["tech_sig"],
                    regression_curve=ord_req["reg_curve"],
                    benchmark_candles=benchmark_candles,
                    decision_date=ord_req["signal_date"],
                    capital=cash + sum(p["last_price"] * p["qty"] for p in open_positions.values()),
                    current_deployed_capital=deployed,
                    currency=universe_history[sym].get("currency", "INR"),
                    cfg=c,
                    enable_atr_stops=use_atr_stops,
                    enable_vol_sizing=True,
                    enable_regime_filter=False,  # already evaluated at signal date
                    enable_confidence_and_targets=use_conf_targets,
                )
                qty = int(eval_res["position_qty"])
                # Scale to float cash safely
                max_cash_qty = int((cash * 0.98) // max(1.0, fill_px))
                qty = min(qty, max_cash_qty)
                if qty < 1:
                    rejected_count += 1
                    continue
                notional = qty * fill_px
                eff_sl = round(fill_px * (1.0 - ord_req["sl_ratio"]), 4)
                eff_tp = round(fill_px * (1.0 + ord_req["tp_ratio"]), 4)
            else:
                # Baseline Moreshwar 1.0 / Fixed comparison: fractional 25% of portfolio capital if available
                port_val = cash + sum(p["last_price"] * p["qty"] for p in open_positions.values())
                alloc_target = min(cash * 0.98, port_val * 0.25)
                if alloc_target < fill_px * 0.5 or alloc_target < 500.0:
                    rejected_count += 1
                    continue
                notional = alloc_target
                qty = notional / fill_px
                eff_sl = round(fill_px * (1.0 - ord_req["sl_ratio"]), 4)
                eff_tp = round(fill_px * (1.0 + ord_req["tp_ratio"]), 4)

            cash -= notional
            open_positions[sym] = {
                "symbol": sym,
                "sector": universe_history[sym].get("sector", "Equity"),
                "signal_date": ord_req["signal_date"],
                "signal_regime": ord_req["signal_regime"],
                "entry_date": dt,
                "entry_nav_idx": d_idx,
                "entry_price": fill_px,
                "last_price": fill_px,
                "qty": qty,
                "notional": notional,
                "sl": eff_sl,
                "tp": eff_tp,
                "mfe_pct": 0.0,
                "mae_pct": 0.0,
                "bars_held": 0,
            }
        pending_orders = next_pending

        # 2. Mark to market & check exits on open positions
        to_close: list[str] = []
        for sym, pos in open_positions.items():
            if dt not in date_bars_by_sym[sym]:
                continue
            _, bar = date_bars_by_sym[sym][dt]
            hi = float(bar.get("high") or bar.get("close") or pos["last_price"])
            lo = float(bar.get("low") or bar.get("close") or pos["last_price"])
            cl = float(bar.get("close") or pos["last_price"])
            op = float(bar.get("open") or cl)
            pos["last_price"] = cl
            pos["bars_held"] += 1

            pos["mfe_pct"] = max(pos["mfe_pct"], ((hi - pos["entry_price"]) / pos["entry_price"]) * 100.0)
            pos["mae_pct"] = min(pos["mae_pct"], ((lo - pos["entry_price"]) / pos["entry_price"]) * 100.0)

            exit_px = None
            exit_reason = None
            if lo <= pos["sl"]:
                exit_px = min(op, pos["sl"]) if pos["bars_held"] > 1 else pos["sl"]
                exit_reason = "STOP_LOSS"
                sl_count += 1
            elif hi >= pos["tp"]:
                exit_px = pos["tp"]
                exit_reason = "TARGET_HIT"
                tp_count += 1
            elif pos["bars_held"] >= c.horizon_days or d_idx == len(timeline) - 1:
                exit_px = cl
                exit_reason = "HORIZON_EXIT"
                horizon_count += 1

            if exit_px is not None:
                exit_after_slip = exit_px * (1.0 - EXIT_SLIPPAGE_BPS / 10000.0)
                proceeds = pos["qty"] * exit_after_slip
                statutory_fee = proceeds * (STATUTORY_COST_BPS / 10000.0)
                slip_cost = pos["qty"] * (
                    (pos["entry_price"] * (ENTRY_SLIPPAGE_BPS / 10000.0))
                    + (exit_px * (EXIT_SLIPPAGE_BPS / 10000.0))
                )
                total_cost_amount += statutory_fee + slip_cost
                cash += proceeds - statutory_fee

                gross_pct = ((exit_after_slip - pos["entry_price"]) / pos["entry_price"]) * 100.0
                net_pct = (((proceeds - statutory_fee) - pos["notional"]) / max(1e-6, pos["notional"])) * 100.0
                pnl_amt = (proceeds - statutory_fee) - pos["notional"]

                closed_trades.append({
                    "symbol": sym,
                    "sector": pos["sector"],
                    "signal_date": pos["signal_date"],
                    "entry_date": pos["entry_date"],
                    "exit_date": dt,
                    "signal_regime": pos["signal_regime"],
                    "holding_days": pos["bars_held"],
                    "entry_price": round(pos["entry_price"], 2),
                    "exit_price": round(exit_after_slip, 2),
                    "qty": round(pos["qty"], 2) if not use_vol_sizing else int(pos["qty"]),
                    "notional": round(pos["notional"], 2),
                    "pnl_amount": round(pnl_amt, 2),
                    "gross_return_pct": round(gross_pct, 2),
                    "net_return_pct": round(net_pct, 2),
                    "cost_amount": round(statutory_fee + slip_cost, 2),
                    "mfe_pct": round(pos["mfe_pct"], 2),
                    "mae_pct": round(pos["mae_pct"], 2),
                    "exit_reason": exit_reason,
                })
                to_close.append(sym)

        for sym in to_close:
            del open_positions[sym]

        # Record end-of-day NAV & capital utilization
        deployed_mkt = sum(p["last_price"] * p["qty"] for p in open_positions.values())
        nav = cash + deployed_mkt
        util_pct = (deployed_mkt / max(1.0, nav)) * 100.0
        daily_utilizations.append(util_pct)
        daily_navs.append({"date": dt, "nav": round(nav, 2), "regime": day_regime})

        # Skip generating new entries on the final bar
        if d_idx >= len(timeline) - 1:
            continue

        # 3. Generate new causal end-of-day signals for tomorrow's open (zero look-ahead)
        for sym in valid_syms:
            if sym in open_positions or dt not in date_bars_by_sym[sym]:
                continue
            b_idx, bar = date_bars_by_sym[sym][dt]
            if b_idx < 45:
                continue

            candles_up_to_t = universe_history[sym]["candles"][max(0, b_idx - 50) : b_idx + 1]
            if len(candles_up_to_t) < 46:
                continue
            try:
                quant = compute_ohlcv_and_regression_from_candles(candles_up_to_t, periods=45)
            except Exception:
                continue

            latest = quant["candles"][-1]
            spot = float(latest["close"])
            sma20 = float(latest["sma_20"])
            rsi14 = float(latest["rsi_14"])
            tech_sig, tech_conf = _causal_tech_eval(quant)

            snap = universe_history[sym]
            f_score_obj = snap.get("fundamental_score") or {}
            p_score_obj = snap.get("porters_score") or {}
            if fundamental_mode == "historical_unavailable":
                f_tot: float | None = None
                p_tot: float | None = None
                missing_fields = list(snap.get("missing_fields") or [])
            else:
                f_tot = float(f_score_obj.get("total", 55))
                p_tot = float(p_score_obj.get("total", 55))
                missing_fields = list(snap.get("missing_fields") or [])

            # Entry eligibility gate
            fire = False
            if entry_style == "matrix_static":
                meta = {
                    "sales_growth": (f_score_obj.get("ratios") or {}).get("sales_growth", 10.0),
                    "profit_growth": (f_score_obj.get("ratios") or {}).get("profit_growth", 10.0),
                }
                dec = calculate_decision_matrix(
                    int(f_tot or 55), int(p_tot or 55), meta=meta, price_above_sma=(spot >= sma20), is_holding=False
                )
                fire = (dec["action"] in ("BUY NOW", "ADD")) and (tech_sig != "BEARISH")
            elif entry_style == "causal_momentum":
                # Purely causal point-in-time TechBot_1 entry (NO future fundamentals leaked):
                # Trend above 20D SMA + TechBot BULLISH confluence without buying RSI > 70 exhaustion
                fire = (
                    spot >= sma20
                    and 42.0 <= rsi14 <= 70.0
                    and tech_sig == "BULLISH"
                )
            elif entry_style == "sma_fixed":
                fire = (spot >= sma20) and (rsi14 < 72.0)

            if not fire:
                continue

            # Exit levels & risk filters
            if exit_style == "moreshwar_1":
                m1 = calculate_moreshwar_levels(spot, int(f_tot or 55), int(p_tot or 55), is_holding=False)
                sl_px, tp_px = float(m1["stop_loss"]), float(m1["target"])
                # Apply optional independent Phase 5 regime / ATR upgrades when testing B/C/D on Model A entries
                if use_atr_stops or use_regime_filter or use_conf_targets:
                    v2_chk = evaluate_moreshwar_v2(
                        symbol=sym,
                        candles=candles_up_to_t,
                        spot_price=spot,
                        fundamental_score=f_tot,
                        porters_score=p_tot,
                        missing_fields=missing_fields,
                        fundamental_mode=fundamental_mode,
                        tech_signal=tech_sig,
                        regression_curve=quant["regression_curve"],
                        sma_20=sma20,
                        rsi_14=rsi14,
                        benchmark_candles=benchmark_candles,
                        decision_date=dt,
                        capital=nav,
                        current_deployed_capital=deployed_mkt,
                        currency=snap.get("currency", "INR"),
                        cfg=c,
                        enable_atr_stops=use_atr_stops,
                        enable_vol_sizing=use_vol_sizing,
                        enable_regime_filter=use_regime_filter,
                        enable_confidence_and_targets=use_conf_targets,
                    )
                    if not v2_chk["accepted"]:
                        rejected_count += 1
                        if v2_chk["blocked_by_regime"]:
                            regime_blocked_count += 1
                        continue
                    if use_atr_stops:
                        sl_px = float(v2_chk["stop_loss"])
                    if use_conf_targets:
                        tp_px = float(v2_chk["target"])
                    elif use_atr_stops:
                        # Isolate stop improvement before changing target formula: 1:2 on ATR stop or original target
                        tp_px = float(m1["target"])
            elif exit_style == "moreshwar_2":
                v2 = evaluate_moreshwar_v2(
                    symbol=sym,
                    candles=candles_up_to_t,
                    spot_price=spot,
                    fundamental_score=f_tot,
                    porters_score=p_tot,
                    missing_fields=missing_fields,
                    fundamental_mode=fundamental_mode,
                    tech_signal=tech_sig,
                    regression_curve=quant["regression_curve"],
                    sma_20=sma20,
                    rsi_14=rsi14,
                    benchmark_candles=benchmark_candles,
                    decision_date=dt,
                    capital=nav,
                    current_deployed_capital=deployed_mkt,
                    currency=snap.get("currency", "INR"),
                    cfg=c,
                    enable_atr_stops=True,
                    enable_vol_sizing=use_vol_sizing,
                    enable_regime_filter=use_regime_filter,
                    enable_confidence_and_targets=use_conf_targets,
                )
                if not v2["accepted"]:
                    rejected_count += 1
                    if v2["blocked_by_regime"]:
                        regime_blocked_count += 1
                    continue
                sl_px, tp_px = float(v2["stop_loss"]), float(v2["target"])
            else:  # fixed_5_10
                sl_px = round(spot * 0.95, 2)
                tp_px = round(spot * 1.10, 2)

            sl_ratio = max(0.01, (spot - sl_px) / max(1e-6, spot))
            tp_ratio = max(0.015, (tp_px - spot) / max(1e-6, spot))
            pending_orders.append({
                "symbol": sym,
                "signal_date": dt,
                "signal_regime": day_regime,
                "sl_ratio": sl_ratio,
                "tp_ratio": tp_ratio,
                "f_tot": f_tot,
                "p_tot": p_tot,
                "missing_fields": missing_fields,
                "tech_sig": tech_sig,
                "reg_curve": quant["regression_curve"],
            })

    # Compute full Phase 7 metrics
    metrics = _compute_phase7_metrics(
        model_name=model_name,
        closed_trades=closed_trades,
        daily_navs=daily_navs,
        daily_utilizations=daily_utilizations,
        initial_capital=initial_capital,
        total_cost_amount=total_cost_amount,
        sl_count=sl_count,
        tp_count=tp_count,
        horizon_count=horizon_count,
        rejected_count=rejected_count,
        regime_blocked_count=regime_blocked_count,
        fundamental_mode=fundamental_mode,
    )

    # Regime breakdown
    regimes_out: dict[str, dict[str, Any]] = {}
    for r_name in ("BULLISH", "NEUTRAL", "BEARISH", "HIGH_VOLATILITY"):
        sub = [t for t in closed_trades if t["signal_regime"] == r_name]
        regimes_out[r_name] = _summarize_trade_cohort(sub)

    # Stock & Sector breakdowns
    by_stock: dict[str, dict[str, Any]] = {}
    for t in closed_trades:
        by_stock.setdefault(t["symbol"], []).append(t)
    stock_summary = {k: _summarize_trade_cohort(v) for k, v in sorted(by_stock.items())}

    by_sector: dict[str, dict[str, Any]] = {}
    for t in closed_trades:
        by_sector.setdefault(t["sector"], []).append(t)
    sector_summary = {k: _summarize_trade_cohort(v) for k, v in sorted(by_sector.items())}

    return {
        "model": model_name,
        "metrics": metrics,
        "regimes": regimes_out,
        "by_stock": stock_summary,
        "by_sector": sector_summary,
        "equity_curve": daily_navs,
        "trades": closed_trades,
    }


def _summarize_trade_cohort(trades: list[dict[str, Any]]) -> dict[str, Any]:
    if not trades:
        return {"trades": 0, "win_rate_pct": 0.0, "avg_return_pct": 0.0, "total_pnl": 0.0}
    wins = [t for t in trades if t["net_return_pct"] > 0]
    return {
        "trades": len(trades),
        "win_rate_pct": round((len(wins) / len(trades)) * 100.0, 2),
        "avg_return_pct": round(sum(t["net_return_pct"] for t in trades) / len(trades), 2),
        "total_pnl": round(sum(t["pnl_amount"] for t in trades), 2),
    }


def _empty_phase7_metrics(model_name: str) -> dict[str, Any]:
    return {
        "model": model_name,
        "total_return_pct": 0.0,
        "annualised_return_pct": 0.0,
        "win_rate_pct": 0.0,
        "profit_factor": 0.0,
        "sharpe_ratio": 0.0,
        "sortino_ratio": 0.0,
        "max_drawdown_pct": 0.0,
        "avg_holding_days": 0.0,
        "avg_winning_trade_pct": 0.0,
        "avg_losing_trade_pct": 0.0,
        "avg_mae_pct": 0.0,
        "avg_mfe_pct": 0.0,
        "total_trades": 0,
        "transaction_costs_pct": 0.0,
        "capital_utilisation_pct": 0.0,
        "stop_loss_freq_pct": 0.0,
        "target_hit_freq_pct": 0.0,
        "horizon_exit_freq_pct": 0.0,
        "rejected_trades": 0,
        "regime_blocked_trades": 0,
        "fundamental_integrity": "UNKNOWN",
    }


def _compute_phase7_metrics(
    *,
    model_name: str,
    closed_trades: list[dict[str, Any]],
    daily_navs: list[dict[str, Any]],
    daily_utilizations: list[float],
    initial_capital: float,
    total_cost_amount: float,
    sl_count: int,
    tp_count: int,
    horizon_count: int,
    rejected_count: int,
    regime_blocked_count: int,
    fundamental_mode: str,
) -> dict[str, Any]:
    end_nav = daily_navs[-1]["nav"] if daily_navs else initial_capital
    total_ret = ((end_nav - initial_capital) / max(1.0, initial_capital)) * 100.0
    n_days = max(1, len(daily_navs))
    ann_ret = (((end_nav / max(1.0, initial_capital)) ** (252.0 / n_days)) - 1.0) * 100.0 if end_nav > 0 else -100.0

    # Daily portfolio Sharpe, Sortino & Max Drawdown
    peak = initial_capital
    max_dd = 0.0
    daily_rets: list[float] = []
    prev_n = initial_capital
    for pt in daily_navs:
        n = pt["nav"]
        peak = max(peak, n)
        dd = ((n - peak) / max(1.0, peak)) * 100.0
        max_dd = min(max_dd, dd)
        daily_rets.append((n - prev_n) / max(1.0, prev_n))
        prev_n = n

    rf_daily = RISK_FREE_ANNUAL / 252.0
    ex = [r - rf_daily for r in daily_rets]
    if len(ex) > 2:
        m_ex = sum(ex) / len(ex)
        var_ex = sum((x - m_ex) ** 2 for x in ex) / (len(ex) - 1)
        std_ex = math.sqrt(max(0.0, var_ex))
        down_ex = math.sqrt(sum(min(0.0, x) ** 2 for x in ex) / len(ex))
        sharpe = round((m_ex / std_ex) * math.sqrt(252.0), 2) if std_ex > 1e-8 else 0.0
        sortino = round((m_ex / down_ex) * math.sqrt(252.0), 2) if down_ex > 1e-8 else (sharpe if m_ex > 0 else 0.0)
    else:
        sharpe, sortino = 0.0, 0.0

    n_trades = len(closed_trades)
    if n_trades > 0:
        win_rets = [t["net_return_pct"] for t in closed_trades if t["net_return_pct"] > 0]
        loss_rets = [t["net_return_pct"] for t in closed_trades if t["net_return_pct"] <= 0]
        gross_win_amt = sum(t["pnl_amount"] for t in closed_trades if t["pnl_amount"] > 0)
        gross_loss_amt = abs(sum(t["pnl_amount"] for t in closed_trades if t["pnl_amount"] < 0))
        pf = round(gross_win_amt / gross_loss_amt, 2) if gross_loss_amt > 1e-6 else (9.99 if gross_win_amt > 0 else 0.0)
        win_rate = round((len(win_rets) / n_trades) * 100.0, 2)
        avg_win = round(sum(win_rets) / len(win_rets), 2) if win_rets else 0.0
        avg_loss = round(sum(loss_rets) / len(loss_rets), 2) if loss_rets else 0.0
        avg_hold = round(sum(t["holding_days"] for t in closed_trades) / n_trades, 1)
        avg_mae = round(sum(t["mae_pct"] for t in closed_trades) / n_trades, 2)
        avg_mfe = round(sum(t["mfe_pct"] for t in closed_trades) / n_trades, 2)
        sl_freq = round((sl_count / n_trades) * 100.0, 1)
        tp_freq = round((tp_count / n_trades) * 100.0, 1)
        hz_freq = round((horizon_count / n_trades) * 100.0, 1)
    else:
        pf, win_rate, avg_win, avg_loss, avg_hold, avg_mae, avg_mfe = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        sl_freq, tp_freq, hz_freq = 0.0, 0.0, 0.0

    return {
        "model": model_name,
        "total_return_pct": round(total_ret, 2),
        "annualised_return_pct": round(ann_ret, 2),
        "win_rate_pct": win_rate,
        "profit_factor": pf,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "max_drawdown_pct": round(abs(max_dd), 2),
        "avg_holding_days": avg_hold,
        "avg_winning_trade_pct": avg_win,
        "avg_losing_trade_pct": avg_loss,
        "avg_mae_pct": avg_mae,
        "avg_mfe_pct": avg_mfe,
        "total_trades": n_trades,
        "transaction_costs_pct": round((total_cost_amount / max(1.0, initial_capital)) * 100.0, 2),
        "capital_utilisation_pct": round(sum(daily_utilizations) / max(1, len(daily_utilizations)), 1),
        "stop_loss_freq_pct": sl_freq,
        "target_hit_freq_pct": tp_freq,
        "horizon_exit_freq_pct": hz_freq,
        "rejected_trades": rejected_count,
        "regime_blocked_trades": regime_blocked_count,
        "fundamental_integrity": (
            "CAUSAL_CLEAN (Historical Fundamentals Marked Unavailable)"
            if fundamental_mode == "historical_unavailable"
            else "INCOMPLETE_STATIC_WARNING (Current Snapshot Reused Across Past Bars)"
        ),
    }


def _compute_nifty_buy_and_hold(
    benchmark_candles: list[dict[str, Any]] | None,
    date_range: tuple[str, str] | None = None,
) -> dict[str, Any]:
    if not benchmark_candles:
        return {"model": "Nifty 50 Buy & Hold", "metrics": _empty_phase7_metrics("Nifty 50 Buy & Hold")}
    bars = [c for c in benchmark_candles if float(c.get("close") or 0.0) > 0]
    if date_range:
        bars = [c for c in bars if date_range[0] <= str(c["date"]) <= date_range[1]]
    if len(bars) < 5:
        return {"model": "Nifty 50 Buy & Hold", "metrics": _empty_phase7_metrics("Nifty 50 Buy & Hold")}

    navs = [{"date": b["date"], "nav": round((float(b["close"]) / float(bars[0]["close"])) * INITIAL_CAPITAL, 2)} for b in bars]
    tot_ret = round(((navs[-1]["nav"] - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100.0, 2)
    ann_ret = round((((navs[-1]["nav"] / INITIAL_CAPITAL) ** (252.0 / len(bars))) - 1.0) * 100.0, 2)
    peak = INITIAL_CAPITAL
    max_dd = 0.0
    rets: list[float] = []
    prev = INITIAL_CAPITAL
    for pt in navs:
        peak = max(peak, pt["nav"])
        max_dd = min(max_dd, ((pt["nav"] - peak) / peak) * 100.0)
        rets.append((pt["nav"] - prev) / prev)
        prev = pt["nav"]
    rf_d = RISK_FREE_ANNUAL / 252.0
    ex = [r - rf_d for r in rets[1:]]
    m_ex = sum(ex) / max(1, len(ex))
    s_ex = math.sqrt(sum((x - m_ex) ** 2 for x in ex) / max(1, len(ex) - 1)) if len(ex) > 1 else 0.0
    d_ex = math.sqrt(sum(min(0.0, x) ** 2 for x in ex) / max(1, len(ex))) if ex else 0.0
    return {
        "model": "Nifty 50 (^NSEI) Buy & Hold",
        "metrics": {
            "model": "Nifty 50 (^NSEI) Buy & Hold",
            "total_return_pct": tot_ret,
            "annualised_return_pct": ann_ret,
            "win_rate_pct": 100.0 if tot_ret > 0 else 0.0,
            "profit_factor": None,
            "sharpe_ratio": round((m_ex / s_ex) * math.sqrt(252.0), 2) if s_ex > 1e-8 else 0.0,
            "sortino_ratio": round((m_ex / d_ex) * math.sqrt(252.0), 2) if d_ex > 1e-8 else 0.0,
            "max_drawdown_pct": round(abs(max_dd), 2),
            "avg_holding_days": float(len(bars)),
            "avg_winning_trade_pct": tot_ret if tot_ret > 0 else 0.0,
            "avg_losing_trade_pct": tot_ret if tot_ret <= 0 else 0.0,
            "avg_mae_pct": round(max_dd, 2),
            "avg_mfe_pct": None,
            "total_trades": 1,
            "transaction_costs_pct": 0.0,
            "capital_utilisation_pct": 100.0,
            "stop_loss_freq_pct": 0.0,
            "target_hit_freq_pct": 0.0,
            "horizon_exit_freq_pct": 0.0,
            "rejected_trades": 0,
            "regime_blocked_trades": 0,
            "fundamental_integrity": "MARKET_INDEX_BENCHMARK",
        },
        "equity_curve": navs,
    }


def run_moreshwar_v2_research_suite(
    universe_history: dict[str, dict[str, Any]],
    benchmark_data: dict[str, Any] | None = None,
    cfg: MoreshwarV2Config | None = None,
    run_sensitivity: bool = True,
) -> dict[str, Any]:
    """Execute Phases 5, 6 & 7: Independent Models A–F + Benchmarks + Train/Val/OOS & Sensitivity."""
    c = cfg or MoreshwarV2Config()
    bench_candles = (benchmark_data or {}).get("candles") if benchmark_data else None

    # Determine canonical timeline after 45-bar warmup
    all_dates = sorted({
        str(b["date"])
        for snap in universe_history.values()
        for idx, b in enumerate(snap.get("candles") or [])
        if idx >= 45
    })
    start_dt = all_dates[0] if all_dates else None
    end_dt = all_dates[-1] if all_dates else None

    # Phase 5: Independent Ablation Matrix (identical universe, timeline & cost model)
    model_a = _simulate_chronological_portfolio(
        universe_history, bench_candles,
        model_name="Model A: Original Baseline (Moreshwar 1.0)",
        entry_style="matrix_static", exit_style="moreshwar_1",
        cfg=c, use_atr_stops=False, use_vol_sizing=False, use_regime_filter=False, use_conf_targets=False,
        fundamental_mode="point_in_time_static_warning",
    )
    model_b = _simulate_chronological_portfolio(
        universe_history, bench_candles,
        model_name="Model B: ATR Stops Only",
        entry_style="matrix_static", exit_style="moreshwar_1",
        cfg=c, use_atr_stops=True, use_vol_sizing=False, use_regime_filter=False, use_conf_targets=False,
        fundamental_mode="point_in_time_static_warning",
    )
    model_c = _simulate_chronological_portfolio(
        universe_history, bench_candles,
        model_name="Model C: ATR Stops + Vol Sizing",
        entry_style="matrix_static", exit_style="moreshwar_1",
        cfg=c, use_atr_stops=True, use_vol_sizing=True, use_regime_filter=False, use_conf_targets=False,
        fundamental_mode="point_in_time_static_warning",
    )
    model_d = _simulate_chronological_portfolio(
        universe_history, bench_candles,
        model_name="Model D: ATR + Sizing + Nifty Regime",
        entry_style="matrix_static", exit_style="moreshwar_1",
        cfg=c, use_atr_stops=True, use_vol_sizing=True, use_regime_filter=True, use_conf_targets=False,
        fundamental_mode="point_in_time_static_warning",
    )
    model_e_causal = _simulate_chronological_portfolio(
        universe_history, bench_candles,
        model_name="Model E: Complete Moreshwar 2.0 (Clean Causal)",
        entry_style="causal_momentum", exit_style="moreshwar_2",
        cfg=c, use_atr_stops=True, use_vol_sizing=True, use_regime_filter=True, use_conf_targets=True,
        fundamental_mode="historical_unavailable",
    )
    model_e_static_ref = _simulate_chronological_portfolio(
        universe_history, bench_candles,
        model_name="Model E-Ref: Moreshwar 2.0 + Static Fund Warning",
        entry_style="matrix_static", exit_style="moreshwar_2",
        cfg=c, use_atr_stops=True, use_vol_sizing=True, use_regime_filter=True, use_conf_targets=True,
        fundamental_mode="point_in_time_static_warning",
    )
    model_f = _simulate_chronological_portfolio(
        universe_history, bench_candles,
        model_name="Model F: Fixed 5% / 10% Benchmark",
        entry_style="matrix_static", exit_style="fixed_5_10",
        cfg=c, use_atr_stops=False, use_vol_sizing=False, use_regime_filter=False, use_conf_targets=False,
        fundamental_mode="point_in_time_static_warning",
    )
    sma_bench = _simulate_chronological_portfolio(
        universe_history, bench_candles,
        model_name="Simple SMA-20 + Fixed 1:2 Benchmark",
        entry_style="sma_fixed", exit_style="fixed_5_10",
        cfg=c, use_atr_stops=False, use_vol_sizing=False, use_regime_filter=False, use_conf_targets=False,
        fundamental_mode="historical_unavailable",
    )
    nifty_bh = _compute_nifty_buy_and_hold(
        bench_candles,
        date_range=(start_dt, end_dt) if start_dt and end_dt else None,
    )

    models_list = [model_a, model_b, model_c, model_d, model_e_causal, model_e_static_ref, model_f, sma_bench, nifty_bh]

    # Phase 6: Parameter Sensitivity & Out-of-Sample Split (Train / Val / OOS)
    sensitivity_rows: list[dict[str, Any]] = []
    oos_comparison: dict[str, Any] = {}
    rolling_wf: list[dict[str, Any]] = []

    if run_sensitivity and len(all_dates) >= 90:
        n_dt = len(all_dates)
        s1 = int(n_dt * 0.50)
        s2 = int(n_dt * 0.75)
        train_rng = (all_dates[0], all_dates[s1 - 1])
        val_rng = (all_dates[s1], all_dates[s2 - 1])
        oos_rng = (all_dates[s2], all_dates[-1])

        for mult in (1.5, 2.0, 2.5, 3.0):
            for pol in ("permissive", "defensive_bear_block", "bear_and_highvol_block", "strict_bull_only"):
                sweep_cfg = MoreshwarV2Config(atr_multiplier=mult, regime_policy=pol, min_confidence=c.min_confidence)
                tr_res = _simulate_chronological_portfolio(
                    universe_history, bench_candles,
                    model_name=f"M2_atr{mult}_{pol}",
                    entry_style="causal_momentum", exit_style="moreshwar_2",
                    cfg=sweep_cfg, use_atr_stops=True, use_vol_sizing=True, use_regime_filter=True, use_conf_targets=True,
                    fundamental_mode="historical_unavailable", date_range=train_rng,
                )["metrics"]
                va_res = _simulate_chronological_portfolio(
                    universe_history, bench_candles,
                    model_name=f"M2_atr{mult}_{pol}",
                    entry_style="causal_momentum", exit_style="moreshwar_2",
                    cfg=sweep_cfg, use_atr_stops=True, use_vol_sizing=True, use_regime_filter=True, use_conf_targets=True,
                    fundamental_mode="historical_unavailable", date_range=val_rng,
                )["metrics"]
                sensitivity_rows.append({
                    "atr_multiplier": mult,
                    "regime_policy": pol,
                    "min_confidence": c.min_confidence,
                    "train_return_pct": tr_res["total_return_pct"],
                    "train_sharpe": tr_res["sharpe_ratio"],
                    "train_max_dd_pct": tr_res["max_drawdown_pct"],
                    "train_trades": tr_res["total_trades"],
                    "val_return_pct": va_res["total_return_pct"],
                    "val_sharpe": va_res["sharpe_ratio"],
                    "val_max_dd_pct": va_res["max_drawdown_pct"],
                    "val_trades": va_res["total_trades"],
                })

        # Select median disciplined config avoiding pure curve-fit (default ATR 2.0, defensive_bear_block) vs Moreshwar 1.0 strictly on OOS
        oos_m1 = _simulate_chronological_portfolio(
            universe_history, bench_candles,
            model_name="OOS Moreshwar 1.0 Baseline",
            entry_style="matrix_static", exit_style="moreshwar_1",
            cfg=c, use_atr_stops=False, use_vol_sizing=False, use_regime_filter=False, use_conf_targets=False,
            fundamental_mode="point_in_time_static_warning", date_range=oos_rng,
        )["metrics"]
        oos_m2 = _simulate_chronological_portfolio(
            universe_history, bench_candles,
            model_name="OOS Complete Moreshwar 2.0 (Clean Causal)",
            entry_style="causal_momentum", exit_style="moreshwar_2",
            cfg=c, use_atr_stops=True, use_vol_sizing=True, use_regime_filter=True, use_conf_targets=True,
            fundamental_mode="historical_unavailable", date_range=oos_rng,
        )["metrics"]
        oos_nifty = _compute_nifty_buy_and_hold(bench_candles, date_range=oos_rng)["metrics"]

        oos_comparison = {
            "splits": {
                "train": {"start": train_rng[0], "end": train_rng[1], "bars": s1},
                "validation": {"start": val_rng[0], "end": val_rng[1], "bars": s2 - s1},
                "out_of_sample": {"start": oos_rng[0], "end": oos_rng[1], "bars": n_dt - s2},
            },
            "moreshwar_1_oos": oos_m1,
            "moreshwar_2_oos": oos_m2,
            "nifty50_oos": oos_nifty,
        }

        # 3 Rolling walk-forward windows
        fold_size = max(30, n_dt // 3)
        for f_idx in range(3):
            lo = f_idx * fold_size
            hi = min(n_dt - 1, (f_idx + 1) * fold_size - 1) if f_idx < 2 else (n_dt - 1)
            if hi - lo < 20:
                continue
            f_rng = (all_dates[lo], all_dates[hi])
            f_m1 = _simulate_chronological_portfolio(
                universe_history, bench_candles,
                model_name=f"Fold{f_idx+1}_M1",
                entry_style="matrix_static", exit_style="moreshwar_1",
                cfg=c, fundamental_mode="point_in_time_static_warning", date_range=f_rng,
            )["metrics"]
            f_m2 = _simulate_chronological_portfolio(
                universe_history, bench_candles,
                model_name=f"Fold{f_idx+1}_M2",
                entry_style="causal_momentum", exit_style="moreshwar_2",
                cfg=c, use_atr_stops=True, use_vol_sizing=True, use_regime_filter=True, use_conf_targets=True,
                fundamental_mode="historical_unavailable", date_range=f_rng,
            )["metrics"]
            rolling_wf.append({
                "fold": f"WF-{f_idx + 1}",
                "start": f_rng[0],
                "end": f_rng[1],
                "m1_return_pct": f_m1["total_return_pct"],
                "m1_max_dd_pct": f_m1["max_drawdown_pct"],
                "m2_return_pct": f_m2["total_return_pct"],
                "m2_max_dd_pct": f_m2["max_drawdown_pct"],
                "m2_win_rate_pct": f_m2["win_rate_pct"],
                "m2_trades": f_m2["total_trades"],
            })

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "promoted_to_production": False,
        "promotion_status": "SHADOW_RESEARCH_ONLY (Moreshwar 1.0 remains active pending user review)",
        "date_span": {"start": start_dt, "end": end_dt, "trading_days": len(all_dates)},
        "config": {
            "atr_period": c.atr_period,
            "atr_multiplier": c.atr_multiplier,
            "min_stop_pct": c.min_stop_pct,
            "max_stop_pct": c.max_stop_pct,
            "risk_per_trade_pct": c.risk_per_trade_pct,
            "max_single_stock_pct": c.max_single_stock_pct,
            "regime_policy": c.regime_policy,
            "min_rr_ratio": c.min_rr_ratio,
        },
        "models_comparison": [m["metrics"] for m in models_list],
        "model_details": {
            "model_a": model_a,
            "model_b": model_b,
            "model_c": model_c,
            "model_d": model_d,
            "model_e_causal": model_e_causal,
            "model_e_static_ref": model_e_static_ref,
            "model_f": model_f,
            "sma_bench": sma_bench,
            "nifty_bh": nifty_bh,
        },
        "sensitivity_sweep": sensitivity_rows,
        "out_of_sample": oos_comparison,
        "rolling_walk_forward": rolling_wf,
    }
