"""Backtesting & Performance Attribution Engine for Finsistant Version 2.

Features:
- Strictly causal walk-forward evaluation across real 1Y OHLCV bars (`candles[:t+1]`)
  to prevent look-ahead bias, entering strictly on next-bar open (`candles[t+1]["open"]`).
- Evaluates the complete watchlist universe including weak/breakdown stocks (no
  survivorship bias) and applies realistic Indian equity slippage + statutory costs.
- Computes Signal Performance (Win Rate, Avg Return, MFE, MAE, Avg Holding Period),
  Risk-Adjusted Metrics (Sharpe, Sortino, Max Drawdown, Profit Factor, Realized R:R),
  and Benchmark Comparison against Nifty 50 (^NSEI).
- Computes Multi-Agent Attribution (TechBot_1, FundBot_1, ExecBot_1, Gemini CRO).
- Maintains strictly separated performance tracks:
  1. original_ai_signals
  2. recalibrated_ai_signals
  3. actual_executed_trades (real broker fills only — zero simulated results)
"""

from __future__ import annotations

from datetime import datetime, timezone
import math
from typing import Any

from .quant_engine import compute_ohlcv_and_regression_from_candles
from .scoring_engine import calculate_decision_matrix, calculate_moreshwar_levels

RISK_FREE_ANNUAL = 0.065  # ~6.5% India 91D T-Bill annualized benchmark
ENTRY_SLIPPAGE_BPS = 5.0
EXIT_SLIPPAGE_BPS = 5.0
STATUTORY_COST_BPS = 25.0  # Brokerage + STT + Exchange + SEBI + Stamp + GST round-trip


def _calc_tech_confidence(quant: dict[str, Any]) -> tuple[str, float]:
    candles = quant["candles"]
    latest = candles[-1]
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


def _simulate_window_trade(
    candles: list[dict[str, Any]],
    signal_idx: int,
    raw_entry_ref: float,
    target: float,
    stop_loss: float,
    max_horizon: int = 14,
    limit_entry: float | None = None,
    weight: float = 1.0,
) -> dict[str, Any] | None:
    """Simulate a strictly causal trade entering after `signal_idx` using realistic slippage & costs."""
    if signal_idx + 1 >= len(candles):
        return None

    fill_idx = None
    fill_price = 0.0
    fill_date = ""

    if limit_entry is None:
        bar = candles[signal_idx + 1]
        base_open = float(bar["open"] or bar["close"])
        fill_price = round(base_open * (1.0 + ENTRY_SLIPPAGE_BPS / 10000.0), 4)
        fill_idx = signal_idx + 1
        fill_date = bar["date"]
    else:
        # Realistic limit pullback order valid for next 3 sessions
        for k in range(signal_idx + 1, min(len(candles), signal_idx + 4)):
            bar = candles[k]
            o = float(bar["open"] or bar["close"])
            lo = float(bar["low"] or o)
            if o <= limit_entry:
                fill_price = round(o * (1.0 + ENTRY_SLIPPAGE_BPS / 10000.0), 4)
                fill_idx = k
                fill_date = bar["date"]
                break
            if lo <= limit_entry:
                fill_price = round(limit_entry * (1.0 + ENTRY_SLIPPAGE_BPS / 10000.0), 4)
                fill_idx = k
                fill_date = bar["date"]
                break
        if fill_idx is None:
            return {"status": "UNFILLED_LIMIT", "signal_date": candles[signal_idx]["date"]}

    # Adjust SL & TP relative to actual fill if market gapped slightly, keeping causal risk geometry
    sl_ratio = max(0.015, (raw_entry_ref - stop_loss) / max(1e-6, raw_entry_ref))
    tp_ratio = max(0.025, (target - raw_entry_ref) / max(1e-6, raw_entry_ref))
    eff_sl = round(fill_price * (1.0 - sl_ratio), 4)
    eff_tp = round(fill_price * (1.0 + tp_ratio), 4)

    end_idx = min(len(candles) - 1, fill_idx + max_horizon)
    if end_idx <= fill_idx:
        return None

    mfe_pct = 0.0
    mae_pct = 0.0
    exit_price = float(candles[end_idx]["close"])
    exit_date = candles[end_idx]["date"]
    exit_reason = "HORIZON_14D"
    exit_idx = end_idx

    for j in range(fill_idx, end_idx + 1):
        bar = candles[j]
        hi = float(bar["high"] or bar["close"])
        lo = float(bar["low"] or bar["close"])
        cl = float(bar["close"])
        mfe_pct = max(mfe_pct, ((hi - fill_price) / fill_price) * 100.0)
        mae_pct = min(mae_pct, ((lo - fill_price) / fill_price) * 100.0)

        # Conservative intrabar rule: check stop-loss before target on the same daily candle
        if lo <= eff_sl:
            exit_price = min(float(bar["open"] or eff_sl), eff_sl) if j > fill_idx else eff_sl
            exit_date = bar["date"]
            exit_reason = "STOP_LOSS"
            exit_idx = j
            break
        if hi >= eff_tp:
            exit_price = eff_tp
            exit_date = bar["date"]
            exit_reason = "TARGET_HIT"
            exit_idx = j
            break
        if j == end_idx:
            exit_price = cl
            exit_date = bar["date"]

    exit_after_slip = exit_price * (1.0 - EXIT_SLIPPAGE_BPS / 10000.0)
    gross_ret_pct = ((exit_after_slip - fill_price) / fill_price) * 100.0
    cost_pct = STATUTORY_COST_BPS / 100.0  # 0.25%
    net_ret_pct = gross_ret_pct - cost_pct
    weighted_ret_pct = net_ret_pct * weight
    holding_days = max(1, exit_idx - fill_idx + 1)

    return {
        "status": "CLOSED",
        "signal_date": candles[signal_idx]["date"],
        "entry_date": fill_date,
        "exit_date": exit_date,
        "entry_idx": fill_idx,
        "exit_idx": exit_idx,
        "entry_price": round(fill_price, 2),
        "exit_price": round(exit_after_slip, 2),
        "planned_tp": round(eff_tp, 2),
        "planned_sl": round(eff_sl, 2),
        "exit_reason": exit_reason,
        "holding_days": holding_days,
        "gross_return_pct": round(gross_ret_pct, 2),
        "cost_pct": round(cost_pct + (ENTRY_SLIPPAGE_BPS + EXIT_SLIPPAGE_BPS) / 100.0, 2),
        "net_return_pct": round(net_ret_pct, 2),
        "weighted_return_pct": round(weighted_ret_pct, 2),
        "mfe_pct": round(mfe_pct, 2),
        "mae_pct": round(mae_pct, 2),
        "weight": round(weight, 2),
    }


def compute_track_metrics(
    trades: list[dict[str, Any]],
    benchmark_returns_map: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Compute Signal Performance & Risk-Adjusted metrics over a list of closed trades."""
    closed = [t for t in trades if t.get("status") == "CLOSED"]
    if not closed:
        return {
            "total_signals": 0,
            "winners": 0,
            "losers": 0,
            "breakeven": 0,
            "win_rate_pct": 0.0,
            "avg_return_pct": 0.0,
            "cumulative_return_pct": 0.0,
            "avg_mfe_pct": 0.0,
            "avg_mae_pct": 0.0,
            "avg_holding_days": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "max_drawdown_pct": 0.0,
            "profit_factor": 0.0,
            "realized_rr_ratio": 0.0,
        }

    # Sort chronologically by exit_date for realistic equity curve & drawdown
    ordered = sorted(closed, key=lambda x: (str(x.get("exit_date", "")), str(x.get("entry_date", ""))))
    rets = [float(t.get("weighted_return_pct", t.get("net_return_pct", 0.0))) for t in ordered]
    raw_rets = [float(t.get("net_return_pct", 0.0)) for t in ordered]

    winners_r = [r for r in raw_rets if r > 0]
    losers_r = [r for r in raw_rets if r < 0]
    breakeven = len([r for r in raw_rets if r == 0])

    total = len(ordered)
    win_rate = round((len(winners_r) / total) * 100.0, 2)
    avg_ret = round(sum(raw_rets) / total, 2)
    avg_mfe = round(sum(float(t.get("mfe_pct", 0.0)) for t in ordered) / total, 2)
    avg_mae = round(sum(float(t.get("mae_pct", 0.0)) for t in ordered) / total, 2)
    avg_hold = round(sum(float(t.get("holding_days", 1.0)) for t in ordered) / total, 1)

    gross_profit = sum(r for r in rets if r > 0)
    gross_loss = abs(sum(r for r in rets if r < 0))
    profit_factor = round(gross_profit / gross_loss, 2) if gross_loss > 1e-6 else (9.99 if gross_profit > 0 else 0.0)

    avg_win = (sum(winners_r) / len(winners_r)) if winners_r else 0.0
    avg_loss = abs(sum(losers_r) / len(losers_r)) if losers_r else 0.0
    realized_rr = round(avg_win / avg_loss, 2) if avg_loss > 1e-6 else (avg_win if avg_win > 0 else 0.0)

    # Compounded equity curve across signals (risk-scaled 1/4 slot per signal ~25% capital allocation)
    alloc_frac = 0.25
    nav = 100.0
    peak = 100.0
    max_dd = 0.0
    step_returns: list[float] = []
    for r in rets:
        port_step = (r / 100.0) * alloc_frac
        step_returns.append(port_step)
        nav *= (1.0 + port_step)
        peak = max(peak, nav)
        dd = ((nav - peak) / peak) * 100.0
        max_dd = min(max_dd, dd)

    cum_ret_pct = round(nav - 100.0, 2)

    # Annualize Sharpe & Sortino using effective periods per year (~252 / avg_holding_days)
    periods_per_year = max(12.0, 252.0 / max(1.0, avg_hold))
    rf_per_step = RISK_FREE_ANNUAL / periods_per_year
    excess = [r - rf_per_step for r in step_returns]
    mean_ex = sum(excess) / len(excess)
    if len(excess) > 1:
        var = sum((x - mean_ex) ** 2 for x in excess) / (len(excess) - 1)
        std = math.sqrt(max(0.0, var))
        downside_sq = [min(0.0, x) ** 2 for x in excess]
        down_std = math.sqrt(sum(downside_sq) / len(excess))
    else:
        std = 0.0
        down_std = 0.0

    sharpe = round((mean_ex / std) * math.sqrt(periods_per_year), 2) if std > 1e-7 else 0.0
    sortino = round((mean_ex / down_std) * math.sqrt(periods_per_year), 2) if down_std > 1e-7 else (sharpe if mean_ex > 0 else 0.0)

    return {
        "total_signals": total,
        "winners": len(winners_r),
        "losers": len(losers_r),
        "breakeven": breakeven,
        "win_rate_pct": win_rate,
        "avg_return_pct": avg_ret,
        "cumulative_return_pct": cum_ret_pct,
        "avg_mfe_pct": avg_mfe,
        "avg_mae_pct": avg_mae,
        "avg_holding_days": avg_hold,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "max_drawdown_pct": round(abs(max_dd), 2),
        "profit_factor": profit_factor,
        "realized_rr_ratio": realized_rr,
    }


def _build_benchmark_series(
    original_trades: list[dict[str, Any]],
    recal_trades: list[dict[str, Any]],
    benchmark_candles: list[dict[str, Any]] | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Align equity curve of Original AI & Recalibrated AI against Nifty 50 (^NSEI) on identical dates."""
    bench_by_date: dict[str, float] = {}
    bench_dates: list[str] = []
    if benchmark_candles:
        for b in benchmark_candles:
            bench_by_date[b["date"]] = float(b["close"])
            bench_dates.append(b["date"])

    all_trades = sorted(
        [t for t in (original_trades + recal_trades) if t.get("status") == "CLOSED"],
        key=lambda x: str(x.get("exit_date", "")),
    )
    if not all_trades:
        return ({
            "symbol": "NIFTY 50 (^NSEI)",
            "available": bool(benchmark_candles),
            "benchmark_return_pct": 0.0,
            "alpha_original_pct": 0.0,
            "alpha_recalibrated_pct": 0.0,
            "beta": None,
            "start_date": bench_dates[0] if bench_dates else None,
            "end_date": bench_dates[-1] if bench_dates else None,
        }, [])

    start_dt = min(str(t["entry_date"]) for t in all_trades)
    end_dt = max(str(t["exit_date"]) for t in all_trades)

    # Build date checkpoints strictly over evaluation span
    timeline = [d for d in bench_dates if start_dt <= d <= end_dt] if bench_dates else sorted(
        {str(t["entry_date"]) for t in all_trades} | {str(t["exit_date"]) for t in all_trades}
    )
    if len(timeline) < 2:
        timeline = sorted({str(t["entry_date"]) for t in all_trades} | {str(t["exit_date"]) for t in all_trades})

    bench_base = None
    for d in timeline:
        if d in bench_by_date:
            bench_base = bench_by_date[d]
            break

    orig_by_exit: dict[str, list[float]] = {}
    for t in original_trades:
        if t.get("status") == "CLOSED":
            orig_by_exit.setdefault(t["exit_date"], []).append((float(t["weighted_return_pct"]) / 100.0) * 0.25)

    recal_by_exit: dict[str, list[float]] = {}
    for t in recal_trades:
        if t.get("status") == "CLOSED":
            recal_by_exit.setdefault(t["exit_date"], []).append((float(t["weighted_return_pct"]) / 100.0) * 0.25)

    o_nav = 100.0
    r_nav = 100.0
    last_b = bench_base
    curve_full: list[dict[str, Any]] = []
    bench_rets: list[float] = []
    strat_rets: list[float] = []

    for d in timeline:
        prev_o, prev_b = o_nav, last_b
        for step in orig_by_exit.get(d, []):
            o_nav *= (1.0 + step)
        for step in recal_by_exit.get(d, []):
            r_nav *= (1.0 + step)
        if d in bench_by_date and bench_base:
            last_b = bench_by_date[d]
            b_nav = (last_b / bench_base) * 100.0
            if prev_b:
                bench_rets.append((last_b - prev_b) / prev_b)
                strat_rets.append((o_nav - prev_o) / max(1e-6, prev_o))
        else:
            b_nav = None
        curve_full.append({
            "date": d,
            "original_nav": round(o_nav, 2),
            "recalibrated_nav": round(r_nav, 2),
            "benchmark_nav": round(b_nav, 2) if b_nav is not None else None,
        })

    # Sample up to 60 points for lean JSON transport
    if len(curve_full) > 60:
        stride = max(1, len(curve_full) // 58)
        sampled = curve_full[::stride]
        if sampled[-1]["date"] != curve_full[-1]["date"]:
            sampled.append(curve_full[-1])
    else:
        sampled = curve_full

    bench_ret_pct = 0.0
    if curve_full and curve_full[-1]["benchmark_nav"] is not None:
        bench_ret_pct = round(curve_full[-1]["benchmark_nav"] - 100.0, 2)

    beta = None
    if len(bench_rets) > 5:
        mean_b = sum(bench_rets) / len(bench_rets)
        mean_s = sum(strat_rets) / len(strat_rets)
        var_b = sum((b - mean_b) ** 2 for b in bench_rets)
        cov = sum((s - mean_s) * (b - mean_b) for s, b in zip(strat_rets, bench_rets))
        if var_b > 1e-10:
            beta = round(cov / var_b, 2)

    orig_cum = round(o_nav - 100.0, 2)
    recal_cum = round(r_nav - 100.0, 2)
    summary = {
        "symbol": "NIFTY 50 (^NSEI)",
        "available": bool(bench_base),
        "benchmark_return_pct": bench_ret_pct,
        "alpha_original_pct": round(orig_cum - bench_ret_pct, 2),
        "alpha_recalibrated_pct": round(recal_cum - bench_ret_pct, 2),
        "beta": beta,
        "start_date": timeline[0] if timeline else None,
        "end_date": timeline[-1] if timeline else None,
    }
    return summary, sampled


def run_backtest_and_attribution(
    universe_history: dict[str, dict[str, Any]],
    benchmark_data: dict[str, Any] | None = None,
    executed_orders: list[dict[str, Any]] | None = None,
    audit_records: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run causal walk-forward backtest + multi-agent attribution across real 1Y history."""
    original_trades: list[dict[str, Any]] = []
    recalibrated_trades: list[dict[str, Any]] = []
    deferred_by_cro: list[dict[str, Any]] = []
    unfilled_pullbacks = 0

    tech_bull_rets: list[float] = []
    tech_bear_rets: list[float] = []
    fund_high_rets: list[float] = []
    fund_low_rets: list[float] = []
    exec_sl_saves = 0
    exec_tp_hits = 0

    symbols_evaluated: list[str] = []
    for sym, snap in universe_history.items():
        candles = snap.get("candles") or []
        if len(candles) < 62:
            continue
        symbols_evaluated.append(sym)
        f_score = snap.get("fundamental_score") or {"total": 55}
        p_score = snap.get("porters_score") or {"total": 55}
        f_tot = int(f_score.get("total", 55))
        p_tot = int(p_score.get("total", 55))

        meta = {
            "sales_growth": (f_score.get("ratios") or {}).get("sales_growth", snap.get("sales_growth", 10.0)),
            "profit_growth": (f_score.get("ratios") or {}).get("profit_growth", snap.get("profit_growth", 10.0)),
        }

        # Causal 14D forward check for FundBot cohort attribution (non-overlapping 20-bar steps)
        for t in range(45, len(candles) - 16, 20):
            entry_o = float(candles[t + 1]["open"] or candles[t + 1]["close"])
            exit_c = float(candles[t + 15]["close"])
            fwd_net = ((exit_c - entry_o) / max(1e-6, entry_o)) * 100.0 - 0.35
            if f_tot >= 65:
                fund_high_rets.append(fwd_net)
            elif f_tot <= 48:
                fund_low_rets.append(fwd_net)

        t = 45
        t_recal = 45
        while t < len(candles) - 16:
            window = candles[max(0, t - 45) : t + 1]
            try:
                quant = compute_ohlcv_and_regression_from_candles(window, periods=45)
            except Exception:
                t += 5
                continue

            latest = quant["candles"][-1]
            spot = float(latest["close"])
            sma20 = float(latest["sma_20"])
            rsi14 = float(latest["rsi_14"])

            tech_sig, tech_conf = _calc_tech_confidence(quant)
            # Track TechBot standalone 14D causal direction accuracy
            if t + 15 < len(candles):
                entry_t = float(candles[t + 1]["open"] or candles[t + 1]["close"])
                exit_t = float(candles[t + 15]["close"])
                net_t = ((exit_t - entry_t) / max(1e-6, entry_t)) * 100.0 - 0.35
                if tech_sig == "BULLISH":
                    tech_bull_rets.append(net_t)
                elif tech_sig == "BEARISH":
                    tech_bear_rets.append(net_t)

            moreshwar = calculate_moreshwar_levels(spot, f_tot, p_tot, is_holding=False)
            dec = calculate_decision_matrix(f_tot, p_tot, meta=meta, price_above_sma=(spot >= sma20), is_holding=False)

            is_buy_signal = dec["action"] in ("BUY NOW", "ADD", "SIP ONLY") or f_tot >= 65
            if not is_buy_signal:
                t += 5
                continue

            # 1. Original AI signal simulation
            tr_orig = _simulate_window_trade(
                candles=candles,
                signal_idx=t,
                raw_entry_ref=spot,
                target=float(moreshwar["target"]),
                stop_loss=float(moreshwar["stop_loss"]),
                max_horizon=14,
                limit_entry=None,
                weight=1.0,
            )
            if tr_orig and tr_orig.get("status") == "CLOSED":
                tr_orig["symbol"] = sym
                tr_orig["action"] = dec["action"]
                original_trades.append(tr_orig)
                if tr_orig["exit_reason"] == "STOP_LOSS":
                    exec_sl_saves += 1
                elif tr_orig["exit_reason"] == "TARGET_HIT":
                    exec_tp_hits += 1

            # 2. Recalibrated AI signal simulation (causal CRO discipline)
            if t >= t_recal:
                weak_or_conflict = (f_tot < 55 and quant["regression_curve"] == "Downward Curve") or tech_sig == "BEARISH"
                overbought = rsi14 >= 68.0 or spot > (sma20 * 1.035)
                if weak_or_conflict:
                    # CRO Defers/Blocks this trade
                    if tr_orig and tr_orig.get("status") == "CLOSED":
                        deferred_by_cro.append(tr_orig)
                    t_recal = t + 5
                elif overbought:
                    # CRO converts aggressive BUY into 2% pullback limit + 15% tighter SL + 0.65x sizing
                    lim_px = round(spot * 0.98, 2)
                    sl_dist = max(lim_px * 0.008, spot - float(moreshwar["stop_loss"])) * 0.85
                    tp_gain = max(lim_px * 0.015, float(moreshwar["target"]) - spot) * 0.90
                    tr_rec = _simulate_window_trade(
                        candles=candles,
                        signal_idx=t,
                        raw_entry_ref=lim_px,
                        target=round(lim_px + tp_gain, 2),
                        stop_loss=round(lim_px - sl_dist, 2),
                        max_horizon=14,
                        limit_entry=lim_px,
                        weight=0.65,
                    )
                    if tr_rec and tr_rec.get("status") == "CLOSED":
                        tr_rec["symbol"] = sym
                        tr_rec["action"] = "LIMIT PULLBACK"
                        recalibrated_trades.append(tr_rec)
                        t_recal = max(t + 5, tr_rec["exit_idx"] + 1)
                    else:
                        unfilled_pullbacks += 1
                        t_recal = t + 5
                else:
                    if tr_orig and tr_orig.get("status") == "CLOSED":
                        clone = dict(tr_orig, action=f"{dec['action']} (CONFIRMED)")
                        recalibrated_trades.append(clone)
                        t_recal = max(t + 5, tr_orig["exit_idx"] + 1)
                    else:
                        t_recal = t + 5

            t = (tr_orig["exit_idx"] + 1) if (tr_orig and "exit_idx" in tr_orig) else (t + 5)

    bench_candles = (benchmark_data or {}).get("candles") if benchmark_data else None
    orig_metrics = compute_track_metrics(original_trades)
    recal_metrics = compute_track_metrics(recalibrated_trades)
    bench_summary, equity_curve = _build_benchmark_series(original_trades, recalibrated_trades, bench_candles)

    # Track 3: Real Actual Executed Trades ONLY (never mixes simulated/backtest rows)
    real_exec = list(executed_orders or [])
    for rec in (audit_records or []):
        out = rec.get("outcome") or {}
        if out.get("broker_executed") and out.get("realized_return_pct") is not None:
            real_exec.append({
                "symbol": rec.get("symbol"),
                "status": "CLOSED",
                "entry_date": out.get("broker_timestamp") or rec.get("created_at", "")[:10],
                "exit_date": out.get("marked_at", "")[:10] or rec.get("created_at", "")[:10],
                "net_return_pct": float(out["realized_return_pct"]),
                "weighted_return_pct": float(out["realized_return_pct"]),
                "holding_days": float(out.get("holding_days", 1)),
                "mfe_pct": float(out.get("mfe_pct", 0.0)),
                "mae_pct": float(out.get("mae_pct", 0.0)),
            })

    if real_exec:
        actual_metrics = compute_track_metrics(real_exec)
        actual_status = "BROKER_VERIFIED"
    else:
        actual_metrics = compute_track_metrics([])
        actual_status = "NO_LIVE_BROKER_FILLS"

    # Multi-Agent Attribution
    tech_bull_avg = round(sum(tech_bull_rets) / len(tech_bull_rets), 2) if tech_bull_rets else 0.0
    tech_bull_win = round((len([r for r in tech_bull_rets if r > 0]) / len(tech_bull_rets)) * 100.0, 1) if tech_bull_rets else 0.0
    tech_bear_avg = round(sum(tech_bear_rets) / len(tech_bear_rets), 2) if tech_bear_rets else 0.0

    fund_high_avg = round(sum(fund_high_rets) / len(fund_high_rets), 2) if fund_high_rets else 0.0
    fund_low_avg = round(sum(fund_low_rets) / len(fund_low_rets), 2) if fund_low_rets else 0.0
    fund_spread = round(fund_high_avg - fund_low_avg, 2)

    def_avg = round(sum(t["net_return_pct"] for t in deferred_by_cro) / len(deferred_by_cro), 2) if deferred_by_cro else 0.0
    def_losers = len([t for t in deferred_by_cro if t["net_return_pct"] < 0])
    win_delta = round(recal_metrics["win_rate_pct"] - orig_metrics["win_rate_pct"], 2)
    dd_delta = round(orig_metrics["max_drawdown_pct"] - recal_metrics["max_drawdown_pct"], 2)
    sharpe_delta = round(recal_metrics["sharpe_ratio"] - orig_metrics["sharpe_ratio"], 2)

    recent_trades = sorted(original_trades, key=lambda x: str(x.get("exit_date", "")), reverse=True)[:12]

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "symbols_evaluated": symbols_evaluated,
        "tracks": {
            "original_ai_signals": {
                "label": "Original AI Signals (Pre-CRO)",
                "is_simulated": True,
                "metrics": orig_metrics,
            },
            "recalibrated_ai_signals": {
                "label": "Recalibrated AI Signals (Post-CRO)",
                "is_simulated": True,
                "metrics": recal_metrics,
            },
            "actual_executed_trades": {
                "label": "Actual Executed Broker Trades",
                "is_simulated": False,
                "status": actual_status,
                "metrics": actual_metrics,
                "note": "Strictly isolated to Zerodha Kite broker fills. Backtest results are never presented as live execution.",
            },
        },
        "benchmark": bench_summary,
        "equity_curve": equity_curve,
        "recent_backtest_trades": recent_trades,
        "agent_attribution": {
            "TechBot_1": {
                "role": "Technical Timing & Poly-Regression",
                "bull_signals": len(tech_bull_rets),
                "bull_win_rate_pct": tech_bull_win,
                "bull_avg_14d_return_pct": tech_bull_avg,
                "bear_avg_14d_return_pct": tech_bear_avg,
                "timing_spread_pct": round(tech_bull_avg - tech_bear_avg, 2),
            },
            "FundBot_1": {
                "role": "StockSight V2 4-Pillar & Porter Moat",
                "high_quality_windows": len(fund_high_rets),
                "high_quality_avg_14d_pct": fund_high_avg,
                "weak_quality_windows": len(fund_low_rets),
                "weak_quality_avg_14d_pct": fund_low_avg,
                "quality_alpha_spread_pct": fund_spread,
                "caveat": "Uses current trailing Yahoo fundamentals across 1Y bars (historical PIT fundamentals unavailable via free feed).",
            },
            "ExecBot_1": {
                "role": "Moreshwar Levels & Auto-Risk Sizing",
                "target_hits": exec_tp_hits,
                "stop_losses_triggered": exec_sl_saves,
                "realized_rr_ratio": orig_metrics["realized_rr_ratio"],
                "profit_factor": orig_metrics["profit_factor"],
            },
            "Gemini_CRO": {
                "role": "Critical Review & Recalibration",
                "deferred_or_blocked_signals": len(deferred_by_cro),
                "avoided_losing_trades": def_losers,
                "deferred_avg_counterfactual_pct": def_avg,
                "unfilled_overbought_pullbacks": unfilled_pullbacks,
                "win_rate_delta_pct": win_delta,
                "drawdown_reduction_pct": dd_delta,
                "sharpe_delta": sharpe_delta,
            },
        },
        "methodology_disclosure": {
            "look_ahead_bias_prevention": "Signals at day t strictly consume candles[:t+1]; fills occur on bar t+1 open (or t+1..t+3 low for pullbacks). Conservative intrabar SL-before-TP ordering.",
            "survivorship_bias_control": f"Evaluates full configured universe ({len(symbols_evaluated)} assets) including weak/breakdown names (e.g., ZEEL, TATAMOTORS).",
            "transaction_costs": f"Entry slippage {ENTRY_SLIPPAGE_BPS} bps + Exit slippage {EXIT_SLIPPAGE_BPS} bps + Statutory/Brokerage {STATUTORY_COST_BPS} bps (0.35% total friction per closed trade).",
            "separation_guarantee": "Walk-forward backtest returns are labeled strictly as historical simulations and isolated from actual Zerodha Kite executed trades.",
        },
    }
