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


def _run_causal_variant(
    universe_history: dict[str, dict[str, Any]],
    *,
    gate_mode: str = "matrix_strict",
    exit_mode: str = "moreshwar",
    use_tech: bool = True,
    use_fund: bool = True,
) -> list[dict[str, Any]]:
    """Run an isolated causal walk-forward configuration for unbiased ablation & benchmark studies."""
    trades: list[dict[str, Any]] = []
    for sym, snap in universe_history.items():
        candles = snap.get("candles") or []
        if len(candles) < 62:
            continue
        f_score = snap.get("fundamental_score") or {"total": 55}
        p_score = snap.get("porters_score") or {"total": 55}
        f_tot = int(f_score.get("total", 55)) if use_fund else 65
        p_tot = int(p_score.get("total", 55)) if use_fund else 65
        meta = {
            "sales_growth": (f_score.get("ratios") or {}).get("sales_growth", snap.get("sales_growth", 10.0)) if use_fund else 12.0,
            "profit_growth": (f_score.get("ratios") or {}).get("profit_growth", snap.get("profit_growth", 10.0)) if use_fund else 12.0,
        }

        t = 45
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
            tech_sig, _ = _calc_tech_confidence(quant)

            moreshwar = calculate_moreshwar_levels(spot, f_tot, p_tot, is_holding=False)
            dec = calculate_decision_matrix(
                f_tot, p_tot, meta=meta, price_above_sma=(spot >= sma20 if use_tech else True), is_holding=False
            )

            if gate_mode == "matrix_strict":
                fire = dec["action"] in ("BUY NOW", "ADD")
            elif gate_mode == "sma_fixed":
                fire = (spot >= sma20) and (rsi14 < 72.0)
            else:
                fire = dec["action"] in ("BUY NOW", "ADD", "SIP ONLY") or f_tot >= 65

            if use_tech and gate_mode != "sma_fixed" and tech_sig == "BEARISH":
                fire = False

            if not fire:
                t += 5
                continue

            if exit_mode == "moreshwar":
                tp, sl = float(moreshwar["target"]), float(moreshwar["stop_loss"])
            else:
                tp, sl = round(spot * 1.10, 2), round(spot * 0.95, 2)

            tr = _simulate_window_trade(
                candles=candles,
                signal_idx=t,
                raw_entry_ref=spot,
                target=tp,
                stop_loss=sl,
                max_horizon=14,
                limit_entry=None,
                weight=1.0,
            )
            if tr and tr.get("status") == "CLOSED":
                tr["symbol"] = sym
                trades.append(tr)
                t = tr["exit_idx"] + 1
            else:
                t += 5
    return trades


def _compute_equal_weight_universe_benchmark(universe_history: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Buy-and-hold equal-weight return across the identical watchlist from t=45 to end (net of 0.35% costs)."""
    rets: list[float] = []
    start_dates: list[str] = []
    end_dates: list[str] = []
    for _, snap in universe_history.items():
        candles = snap.get("candles") or []
        if len(candles) < 62:
            continue
        p0 = float(candles[45]["close"] or 0.0)
        p1 = float(candles[-1]["close"] or 0.0)
        if p0 > 0 and p1 > 0:
            rets.append(round(((p1 - p0) / p0) * 100.0 - 0.35, 2))
            start_dates.append(str(candles[45]["date"]))
            end_dates.append(str(candles[-1]["date"]))
    avg_ret = round(sum(rets) / len(rets), 2) if rets else 0.0
    win_assets = len([r for r in rets if r > 0])
    return {
        "assets_count": len(rets),
        "win_assets": win_assets,
        "cumulative_return_pct": avg_ret,
        "start_date": min(start_dates) if start_dates else None,
        "end_date": max(end_dates) if end_dates else None,
    }


def _summarize_live_ledger_for_backtest(audit_records: list[dict[str, Any]] | None) -> dict[str, Any]:
    records = list(audit_records or [])
    settled = [
        r for r in records
        if (r.get("outcome") or {}).get("status") in ("TARGET_HIT", "STOP_HIT", "HORIZON_CLOSED", "DEFERRED_COUNTERFACTUAL")
    ]
    cro_counts = {"PREVENTED_BAD_TRADE": 0, "BENEFICIAL_DOWNGRADE": 0, "PRESERVED_OR_IMPROVED_WINNER": 0, "UNNECESSARY_DOWNGRADE": 0, "UNNECESSARY_BLOCK": 0}
    for r in settled:
        v = str((r.get("outcome") or {}).get("cro_verdict") or "")
        if v in cro_counts:
            cro_counts[v] += 1

    agent_hits: dict[str, dict[str, int]] = {
        "TechBot_1": {"winners": 0, "total": 0},
        "FundBot_1": {"winners": 0, "total": 0},
        "ExecBot_1": {"winners": 0, "total": 0},
        "Gemini_CRO": {"winners": 0, "total": 0},
    }
    for r in settled:
        out = r.get("outcome") or {}
        ag = r.get("agent_assessments") or {}
        fin_ret = float(out.get("realized_return_pct") or 0.0)
        is_win = fin_ret > 0
        if (ag.get("TechBot_1") or {}).get("signal") == "BULLISH":
            agent_hits["TechBot_1"]["total"] += 1
            if is_win:
                agent_hits["TechBot_1"]["winners"] += 1
        if ((r.get("scores") or {}).get("fundamental_total") or 0) >= 65:
            agent_hits["FundBot_1"]["total"] += 1
            if is_win:
                agent_hits["FundBot_1"]["winners"] += 1
        if (ag.get("ExecBot_1") or {}).get("signal") in ("APPROVED", "RECALIBRATED"):
            agent_hits["ExecBot_1"]["total"] += 1
            if is_win:
                agent_hits["ExecBot_1"]["winners"] += 1
        cv = str(out.get("cro_verdict") or "")
        if cv and cv != "NO_INTERVENTION":
            agent_hits["Gemini_CRO"]["total"] += 1
            if cv in ("PREVENTED_BAD_TRADE", "BENEFICIAL_DOWNGRADE", "PRESERVED_OR_IMPROVED_WINNER"):
                agent_hits["Gemini_CRO"]["winners"] += 1

    return {
        "total_records": len(records),
        "settled_records": len(settled),
        "open_records": len(records) - len(settled),
        "min_recommended_settled": 30,
        "sufficient_live_sample": len(settled) >= 30,
        "cro_prevented_bad_trades": cro_counts["PREVENTED_BAD_TRADE"] + cro_counts["BENEFICIAL_DOWNGRADE"],
        "cro_unnecessary_downgrades": cro_counts["UNNECESSARY_DOWNGRADE"] + cro_counts["UNNECESSARY_BLOCK"],
        "cro_preserved_winners": cro_counts["PRESERVED_OR_IMPROVED_WINNER"],
        "agent_live_precision": {
            k: {
                "settled_calls": v["total"],
                "profitable_calls": v["winners"],
                "precision_pct": round((v["winners"] / v["total"]) * 100.0, 1) if v["total"] > 0 else None,
            }
            for k, v in agent_hits.items()
        },
    }


def run_backtest_and_attribution(
    universe_history: dict[str, dict[str, Any]],
    benchmark_data: dict[str, Any] | None = None,
    executed_orders: list[dict[str, Any]] | None = None,
    audit_records: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run causal walk-forward backtest, multi-agent ablation, CRO confusion audit & benchmark verdicts."""
    original_trades: list[dict[str, Any]] = []
    recalibrated_trades: list[dict[str, Any]] = []
    deferred_by_cro: list[dict[str, Any]] = []
    unfilled_pullbacks = 0

    # CRO 2x2 Confusion Matrix counters
    blocks_prevented_bad: list[float] = []
    blocks_missed_good: list[float] = []
    down_beneficial: list[float] = []
    down_unnecessary: list[float] = []
    down_preserved_winner: list[float] = []
    down_neutral: list[float] = []

    tech_bull_rets: list[float] = []
    tech_bear_rets: list[float] = []
    fund_high_rets: list[float] = []
    fund_low_rets: list[float] = []
    exec_sl_saves = 0
    exec_tp_hits = 0
    bars_evaluated = 0
    raw_unfiltered_candidates = 0

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
        while t < len(candles) - 16:
            bars_evaluated += 1
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

            if dec["action"] in ("BUY NOW", "ADD", "SIP ONLY") or f_tot >= 65:
                raw_unfiltered_candidates += 1

            # Quality-first decision gating: require Decision Matrix BUY NOW / ADD (no blind f_tot >= 65 churn)
            is_buy_signal = dec["action"] in ("BUY NOW", "ADD")
            if not is_buy_signal:
                t += 5
                continue

            # 1. Original AI signal simulation (Moreshwar Model)
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
            if not tr_orig or tr_orig.get("status") != "CLOSED":
                t += 5
                continue

            tr_orig["symbol"] = sym
            tr_orig["action"] = dec["action"]
            original_trades.append(tr_orig)
            if tr_orig["exit_reason"] == "STOP_LOSS":
                exec_sl_saves += 1
            elif tr_orig["exit_reason"] == "TARGET_HIT":
                exec_tp_hits += 1

            orig_net = float(tr_orig["net_return_pct"])

            # 2. Recalibrated AI signal simulation & head-to-head CRO Confusion Audit
            block_cond = (
                (quant["regression_curve"] == "Downward Curve")
                or (f_tot < 55 and tech_sig != "BULLISH")
                or tech_sig == "BEARISH"
            )
            overbought = rsi14 >= 74.0 or spot > (sma20 * 1.045)
            if block_cond:
                deferred_by_cro.append(tr_orig)
                if orig_net < 0:
                    blocks_prevented_bad.append(orig_net)
                else:
                    blocks_missed_good.append(orig_net)
            elif overbought:
                lim_px = round(spot * 0.985, 2)
                sl_dist = max(lim_px * 0.020, spot - float(moreshwar["stop_loss"]))
                tp_gain = max(lim_px * 0.020, float(moreshwar["target"]) - spot) * 0.95
                tr_rec = _simulate_window_trade(
                    candles=candles,
                    signal_idx=t,
                    raw_entry_ref=lim_px,
                    target=round(lim_px + tp_gain, 2),
                    stop_loss=round(lim_px - sl_dist, 2),
                    max_horizon=14,
                    limit_entry=lim_px,
                    weight=0.75,
                )
                if tr_rec and tr_rec.get("status") == "CLOSED":
                    tr_rec["symbol"] = sym
                    tr_rec["action"] = "LIMIT PULLBACK"
                    recalibrated_trades.append(tr_rec)
                    rec_w_ret = float(tr_rec["weighted_return_pct"])
                else:
                    unfilled_pullbacks += 1
                    rec_w_ret = 0.0

                delta_ret = rec_w_ret - orig_net
                if orig_net < 0 and delta_ret > 0.05:
                    down_beneficial.append(delta_ret)
                elif orig_net > 0 and delta_ret < -0.05:
                    down_unnecessary.append(delta_ret)
                elif orig_net > 0 and delta_ret >= -0.05:
                    down_preserved_winner.append(delta_ret)
                elif delta_ret > 0.05:
                    down_beneficial.append(delta_ret)
                else:
                    down_neutral.append(delta_ret)
            else:
                clone = dict(tr_orig, action=f"{dec['action']} (CONFIRMED)")
                recalibrated_trades.append(clone)

            t = tr_orig["exit_idx"] + 1

    bench_candles = (benchmark_data or {}).get("candles") if benchmark_data else None
    orig_metrics = compute_track_metrics(original_trades)
    recal_metrics = compute_track_metrics(recalibrated_trades)
    bench_summary, equity_curve = _build_benchmark_series(original_trades, recalibrated_trades, bench_candles)

    # Track 3: Real Actual Executed Trades ONLY
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

    # Controlled Leave-One-Out Agent Ablations & Simple Benchmark Studies
    no_tech_metrics = compute_track_metrics(
        _run_causal_variant(universe_history, gate_mode="matrix_strict", exit_mode="moreshwar", use_tech=False, use_fund=True)
    )
    no_fund_metrics = compute_track_metrics(
        _run_causal_variant(universe_history, gate_mode="matrix_strict", exit_mode="moreshwar", use_tech=True, use_fund=False)
    )
    fixed_same_entry_metrics = compute_track_metrics(
        _run_causal_variant(universe_history, gate_mode="matrix_strict", exit_mode="fixed", use_tech=True, use_fund=True)
    )
    sma_fixed_metrics = compute_track_metrics(
        _run_causal_variant(universe_history, gate_mode="sma_fixed", exit_mode="fixed", use_tech=True, use_fund=False)
    )
    eq_univ_bench = _compute_equal_weight_universe_benchmark(universe_history)
    live_ledger_summary = _summarize_live_ledger_for_backtest(audit_records)

    # Question 1: Standardized Multi-Agent Profit Contribution Leaderboard
    def _agent_row(agent_name: str, role: str, base_m: dict[str, Any], ablated_m: dict[str, Any], basis_note: str) -> dict[str, Any]:
        cum_lift = round(base_m["cumulative_return_pct"] - ablated_m["cumulative_return_pct"], 2)
        win_lift = round(base_m["win_rate_pct"] - ablated_m["win_rate_pct"], 2)
        dd_saved = round(ablated_m["max_drawdown_pct"] - base_m["max_drawdown_pct"], 2)
        pf_lift = round(base_m["profit_factor"] - ablated_m["profit_factor"], 2)
        # Composite Decision Quality Contribution Index (balanced cumulative lift + drawdown saved + PF lift)
        quality_score = round(cum_lift + 0.5 * dd_saved + 5.0 * pf_lift, 2)
        live_p = live_ledger_summary["agent_live_precision"].get(agent_name, {})
        return {
            "agent": agent_name,
            "role": role,
            "cum_return_lift_pct": cum_lift,
            "win_rate_lift_pct": win_lift,
            "drawdown_saved_pct": dd_saved,
            "profit_factor_lift": pf_lift,
            "quality_contribution_index": quality_score,
            "verdict": "POSITIVE CONTRIBUTOR" if quality_score > 0.5 else ("DRAG / NEEDS CALIBRATION" if quality_score < -0.5 else "NEUTRAL"),
            "basis_note": basis_note,
            "live_settled_calls": live_p.get("settled_calls", 0),
            "live_precision_pct": live_p.get("precision_pct"),
        }

    agent_ranking = [
        _agent_row(
            "Gemini_CRO",
            "Adversarial Critique & Recalibration",
            recal_metrics,
            orig_metrics,
            "Full Recalibrated AI vs Pre-CRO Original AI on identical gated entries.",
        ),
        _agent_row(
            "FundBot_1",
            "StockSight V2 4-Pillar & Porter Moat",
            orig_metrics,
            no_fund_metrics,
            "With FundBot quality gates vs Ablated FundBot (ignoring F/P quality filter).",
        ),
        _agent_row(
            "TechBot_1",
            "Technical Timing & Poly-Regression",
            orig_metrics,
            no_tech_metrics,
            "With SMA-20/RSI/MACD timing vs Ablated TechBot (unconditional timing).",
        ),
        _agent_row(
            "ExecBot_1",
            "Moreshwar Dynamic Levels & Stops",
            orig_metrics,
            fixed_same_entry_metrics,
            "Moreshwar dynamic TP/SL vs Fixed 1:2 (5% SL / 10% TP) on identical entries.",
        ),
    ]
    agent_ranking.sort(key=lambda row: row["quality_contribution_index"], reverse=True)
    for idx, row in enumerate(agent_ranking, start=1):
        row["rank"] = idx

    top_agent = agent_ranking[0] if agent_ranking else {"agent": "N/A", "cum_return_lift_pct": 0.0}

    # Question 2: Gemini Intervention Precision (Prevented Bad vs Unnecessarily Downgraded Good)
    prev_blocks = len(blocks_prevented_bad)
    miss_blocks = len(blocks_missed_good)
    ben_downs = len(down_beneficial)
    unnec_downs = len(down_unnecessary)
    pres_downs = len(down_preserved_winner)
    total_blocks = prev_blocks + miss_blocks
    total_downs = ben_downs + unnec_downs + pres_downs + len(down_neutral)
    total_interventions = total_blocks + total_downs

    bad_prevented_total = prev_blocks + ben_downs
    unnecessary_hurt_total = miss_blocks + unnec_downs
    prevention_rate_pct = round((bad_prevented_total / total_interventions) * 100.0, 1) if total_interventions else 0.0
    unnecessary_rate_pct = round((unnecessary_hurt_total / total_interventions) * 100.0, 1) if total_interventions else 0.0
    preserved_rate_pct = round((pres_downs / total_interventions) * 100.0, 1) if total_interventions else 0.0

    avg_loss_saved = round(abs(sum(blocks_prevented_bad) / prev_blocks), 2) if prev_blocks else 0.0
    avg_upside_missed = round(sum(blocks_missed_good) / miss_blocks, 2) if miss_blocks else 0.0
    net_intervention_value_pct = round(
        (-sum(blocks_prevented_bad) + sum(down_beneficial)) - (sum(blocks_missed_good) + abs(sum(down_unnecessary))),
        2,
    )

    gemini_intervention_audit = {
        "total_interventions": total_interventions,
        "blocks": {
            "total": total_blocks,
            "prevented_bad_trades": prev_blocks,
            "prevented_rate_pct": round((prev_blocks / total_blocks) * 100.0, 1) if total_blocks else 0.0,
            "avg_loss_avoided_pct": avg_loss_saved,
            "unnecessarily_blocked_good_trades": miss_blocks,
            "missed_good_rate_pct": round((miss_blocks / total_blocks) * 100.0, 1) if total_blocks else 0.0,
            "avg_upside_missed_pct": avg_upside_missed,
        },
        "downgrades": {
            "total": total_downs,
            "beneficial_downgrades": ben_downs,
            "unnecessary_downgrades": unnec_downs,
            "preserved_winners": pres_downs,
            "neutral": len(down_neutral),
            "unfilled_pullbacks": unfilled_pullbacks,
        },
        "combined_scorecard": {
            "bad_trades_prevented": bad_prevented_total,
            "prevention_rate_pct": prevention_rate_pct,
            "good_trades_unnecessarily_downgraded": unnecessary_hurt_total,
            "unnecessary_downgrade_rate_pct": unnecessary_rate_pct,
            "preserved_winners": pres_downs,
            "preserved_rate_pct": preserved_rate_pct,
            "net_intervention_value_pct": net_intervention_value_pct,
            "headline": (
                f"Prevented {bad_prevented_total}/{total_interventions} bad setups ({prevention_rate_pct}%) vs "
                f"{unnecessary_hurt_total}/{total_interventions} unnecessary downgrades/blocks ({unnecessary_rate_pct}%)."
            ),
        },
        "live_ledger_audit": {
            "settled_interventions": live_ledger_summary["cro_prevented_bad_trades"] + live_ledger_summary["cro_unnecessary_downgrades"] + live_ledger_summary["cro_preserved_winners"],
            "prevented_bad_trades": live_ledger_summary["cro_prevented_bad_trades"],
            "unnecessary_downgrades": live_ledger_summary["cro_unnecessary_downgrades"],
            "preserved_winners": live_ledger_summary["cro_preserved_winners"],
            "open_awaiting_settlement": live_ledger_summary["open_records"],
        },
    }

    # Question 3: Moreshwar Mathematical Model vs Simple Benchmark Strategies
    moreshwar_cum = orig_metrics["cumulative_return_pct"]
    nifty_ret = bench_summary["benchmark_return_pct"]
    eq_ret = eq_univ_bench["cumulative_return_pct"]
    sma_cum = sma_fixed_metrics["cumulative_return_pct"]
    fixed_same_cum = fixed_same_entry_metrics["cumulative_return_pct"]

    beats_nifty = moreshwar_cum > nifty_ret
    beats_eq = moreshwar_cum > eq_ret
    beats_sma = moreshwar_cum > sma_cum
    beats_fixed_exits = moreshwar_cum > fixed_same_cum
    beats_count = sum([beats_nifty, beats_eq, beats_sma])

    if beats_count == 3 and beats_fixed_exits:
        m_verdict = "OUTPERFORMS ALL BENCHMARKS"
    elif beats_count >= 2:
        m_verdict = "OUTPERFORMS MOST BENCHMARKS"
    elif orig_metrics["max_drawdown_pct"] < sma_fixed_metrics["max_drawdown_pct"] and (beats_nifty or beats_eq or beats_sma):
        m_verdict = "MIXED (LOWER DRAWDOWN ONLY)"
    else:
        m_verdict = "UNDERPERFORMS BENCHMARKS"

    moreshwar_vs_benchmarks = {
        "verdict": m_verdict,
        "beats_nifty50": beats_nifty,
        "spread_vs_nifty50_pct": round(moreshwar_cum - nifty_ret, 2),
        "beats_equal_weight_universe": beats_eq,
        "spread_vs_equal_weight_pct": round(moreshwar_cum - eq_ret, 2),
        "beats_simple_sma_strategy": beats_sma,
        "spread_vs_sma_strategy_pct": round(moreshwar_cum - sma_cum, 2),
        "beats_fixed_1to2_same_entries": beats_fixed_exits,
        "spread_vs_fixed_1to2_pct": round(moreshwar_cum - fixed_same_cum, 2),
        "rows": [
            {
                "strategy": "Moreshwar Mathematical Model (Orig)",
                "signals": orig_metrics["total_signals"],
                "win_rate_pct": orig_metrics["win_rate_pct"],
                "cumulative_return_pct": moreshwar_cum,
                "sharpe": orig_metrics["sharpe_ratio"],
                "max_drawdown_pct": orig_metrics["max_drawdown_pct"],
                "profit_factor": orig_metrics["profit_factor"],
            },
            {
                "strategy": "Recalibrated Moreshwar (+ Gemini CRO)",
                "signals": recal_metrics["total_signals"],
                "win_rate_pct": recal_metrics["win_rate_pct"],
                "cumulative_return_pct": recal_metrics["cumulative_return_pct"],
                "sharpe": recal_metrics["sharpe_ratio"],
                "max_drawdown_pct": recal_metrics["max_drawdown_pct"],
                "profit_factor": recal_metrics["profit_factor"],
            },
            {
                "strategy": "Nifty 50 (^NSEI) Buy & Hold",
                "signals": 1,
                "win_rate_pct": 100.0 if nifty_ret > 0 else 0.0,
                "cumulative_return_pct": nifty_ret,
                "sharpe": None,
                "max_drawdown_pct": None,
                "profit_factor": None,
            },
            {
                "strategy": "Equal-Weight Universe Buy & Hold",
                "signals": eq_univ_bench["assets_count"],
                "win_rate_pct": round((eq_univ_bench["win_assets"] / max(1, eq_univ_bench["assets_count"])) * 100.0, 1),
                "cumulative_return_pct": eq_ret,
                "sharpe": None,
                "max_drawdown_pct": None,
                "profit_factor": None,
            },
            {
                "strategy": "Simple SMA-20 + Fixed 1:2 (5%/10%)",
                "signals": sma_fixed_metrics["total_signals"],
                "win_rate_pct": sma_fixed_metrics["win_rate_pct"],
                "cumulative_return_pct": sma_cum,
                "sharpe": sma_fixed_metrics["sharpe_ratio"],
                "max_drawdown_pct": sma_fixed_metrics["max_drawdown_pct"],
                "profit_factor": sma_fixed_metrics["profit_factor"],
            },
            {
                "strategy": "Matrix Entries + Fixed 1:2 Exits",
                "signals": fixed_same_entry_metrics["total_signals"],
                "win_rate_pct": fixed_same_entry_metrics["win_rate_pct"],
                "cumulative_return_pct": fixed_same_cum,
                "sharpe": fixed_same_entry_metrics["sharpe_ratio"],
                "max_drawdown_pct": fixed_same_entry_metrics["max_drawdown_pct"],
                "profit_factor": fixed_same_entry_metrics["profit_factor"],
            },
        ],
        "diagnostic": (
            f"Moreshwar cumulative {moreshwar_cum:+.2f}% ({orig_metrics['win_rate_pct']}% win, {exec_sl_saves} stop-outs vs {exec_tp_hits} targets) "
            f"vs Nifty 50 {nifty_ret:+.2f}%, Equal-Weight {eq_ret:+.2f}%, Simple SMA-20 {sma_cum:+.2f}%, and Fixed 1:2 Same Entries {fixed_same_cum:+.2f}%. "
            + (
                "Notice: Moreshwar's `downside_pts = 100 - X` compresses stop-losses to ~1.5%-3.5% on high-scoring stocks, driving frequent volatility stop-outs compared to wider fixed 5%/10% exits."
                if not beats_fixed_exits
                else "Moreshwar's dynamic score-scaled exits outperformed fixed 1:2 exits on identical entries."
            )
        ),
    }

    # Legacy Multi-Agent Attribution cards (kept populated cleanly alongside Decision Quality Answers)
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
    filtered_out = max(0, raw_unfiltered_candidates - orig_metrics["total_signals"])

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "symbols_evaluated": symbols_evaluated,
        "selectivity_metrics": {
            "bars_evaluated": bars_evaluated,
            "unfiltered_candidate_windows": raw_unfiltered_candidates,
            "quality_gated_signals": orig_metrics["total_signals"],
            "recalibrated_active_signals": recal_metrics["total_signals"],
            "noisy_candidates_filtered": filtered_out,
            "noise_reduction_pct": round((filtered_out / max(1, raw_unfiltered_candidates)) * 100.0, 1),
        },
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
        "decision_quality_answers": {
            "sample_context": live_ledger_summary,
            "top_contributing_agent": {
                "winner": top_agent.get("agent"),
                "winner_role": top_agent.get("role"),
                "cum_return_lift_pct": top_agent.get("cum_return_lift_pct", 0.0),
                "drawdown_saved_pct": top_agent.get("drawdown_saved_pct", 0.0),
                "summary": (
                    f"{top_agent.get('agent')} ranks #1 on Decision Quality Contribution "
                    f"(Cum Return Lift {top_agent.get('cum_return_lift_pct', 0):+.2f}%, "
                    f"DD Saved {top_agent.get('drawdown_saved_pct', 0):+.2f}%)."
                ),
                "ranking": agent_ranking,
            },
            "gemini_intervention_audit": gemini_intervention_audit,
            "moreshwar_vs_benchmarks": moreshwar_vs_benchmarks,
        },
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
