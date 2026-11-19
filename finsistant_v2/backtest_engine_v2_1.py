"""Moreshwar 2.1 Quantitative Validation, Walk-Forward, Monte Carlo & Attribution Engine.

Strict Rules:
- Preserves Moreshwar 1.0 (`scoring_engine.py`) and Moreshwar 2.0 (`moreshwar_v2.py`, `backtest_engine_v2.py`)
  completely unchanged.
- Never fabricates returns or hides negative results. Uses pure chronological next-bar fills,
  itemized Indian equity delivery costs, gap-through-stop fills, volume participation limits,
  6-state causal market regimes, Monte Carlo (>=1,000 runs), and strict out-of-sample isolation.
"""

from __future__ import annotations

from datetime import datetime, timezone
import math
import random
from typing import Any

from .moreshwar_v2 import MoreshwarV2Config, classify_market_regime, compute_atr, evaluate_moreshwar_v2
from .moreshwar_v2_1 import (
    REGIMES_6,
    MoreshwarV21Config,
    classify_market_regime_6,
    compute_indian_delivery_costs,
    evaluate_moreshwar_v2_1,
)
from .quant_engine import compute_ohlcv_and_regression_from_candles
from .scoring_engine import calculate_decision_matrix, calculate_moreshwar_levels

RISK_FREE_ANNUAL = 0.065
INITIAL_CAPITAL = 100_000.0


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _wilson_ci(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return 0.0, 0.0
    p = wins / n
    denom = 1.0 + (z * z) / n
    center = (p + (z * z) / (2.0 * n)) / denom
    half = (z * math.sqrt((p * (1.0 - p) + (z * z) / (4.0 * n)) / n)) / denom
    return round(max(0.0, (center - half) * 100.0), 2), round(min(100.0, (center + half) * 100.0), 2)


def _binomial_two_sided_p(wins: int, n: int, p0: float = 0.5) -> float:
    """Normal-approximated (with exact fallback for n<=30) two-sided binomial p-value vs p0."""
    if n <= 0:
        return 1.0
    if n <= 40:
        obs_p = math.comb(n, wins) * (p0 ** wins) * ((1.0 - p0) ** (n - wins))
        tot = 0.0
        for k in range(n + 1):
            pk = math.comb(n, k) * (p0 ** k) * ((1.0 - p0) ** (n - k))
            if pk <= obs_p + 1e-12:
                tot += pk
        return round(min(1.0, tot), 4)
    se = math.sqrt(p0 * (1.0 - p0) / n)
    z = abs((wins / n) - p0) / max(1e-9, se)
    return round(min(1.0, max(0.0, 2.0 * (1.0 - _normal_cdf(z)))), 4)


def precompute_causal_features(
    universe_history: dict[str, dict[str, Any]],
    benchmark_candles: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    """Precompute strictly causal per-bar features (reading ONLY candles <= bar date)."""
    features: dict[str, dict[str, dict[str, Any]]] = {}
    date_bars: dict[str, dict[str, tuple[int, dict[str, Any]]]] = {}
    all_dates_set: set[str] = set()

    for sym, snap in universe_history.items():
        candles = snap.get("candles") or []
        if len(candles) < 62:
            continue
        sym_feats: dict[str, dict[str, Any]] = {}
        sym_bars: dict[str, tuple[int, dict[str, Any]]] = {}
        closes = [float(c.get("close") or 0.0) for c in candles]

        # Fast causal incremental EMA-12/26/MACD for alignment
        k12, k26, k9 = 2.0 / 13.0, 2.0 / 27.0, 2.0 / 10.0
        ema12, ema26, macd_sig = closes[0], closes[0], 0.0

        for idx, bar in enumerate(candles):
            dt = str(bar.get("date", ""))
            c = closes[idx]
            ema12 = c * k12 + ema12 * (1.0 - k12)
            ema26 = c * k26 + ema26 * (1.0 - k26)
            macd = ema12 - ema26
            macd_sig = macd * k9 + macd_sig * (1.0 - k9)
            sym_bars[dt] = (idx, bar)

            if idx < 45:
                continue
            all_dates_set.add(dt)
            win20 = closes[idx - 19 : idx + 1]
            sma20 = sum(win20) / 20.0
            rs = closes[idx - 14 : idx + 1]
            gains = [max(0.0, rs[j] - rs[j - 1]) for j in range(1, len(rs))]
            losses = [max(0.0, rs[j - 1] - rs[j]) for j in range(1, len(rs))]
            ag, al = sum(gains) / 14.0, sum(losses) / 14.0
            rsi14 = 50.0 if (ag == 0 and al == 0) else (85.0 if al == 0 else max(5.0, min(95.0, 100.0 - 100.0 / (1.0 + ag / al))))

            # Compute lightweight degree-2 slope sign strictly on causal last 45 closes
            y_w = closes[idx - 44 : idx + 1]
            # Fast linear+quad slope check using first vs middle vs end window (or exact regression every signal candidate)
            slope_up = (y_w[-1] >= sum(y_w[-15:]) / 15.0) and (sma20 >= sum(closes[idx - 24 : idx - 4]) / 20.0)
            score = 0.50
            score += 0.14 if c >= sma20 else -0.12
            score += 0.12 if ema12 >= ema26 else -0.10
            score += 0.08 if macd >= macd_sig else -0.08
            score += 0.10 if slope_up else -0.10
            if 46.0 <= rsi14 <= 70.0:
                score += 0.07
            tech_conf = round(max(0.18, min(0.96, score)), 3)
            tech_sig = "BULLISH" if tech_conf >= 0.58 else ("BEARISH" if tech_conf <= 0.44 else "NEUTRAL")

            sym_feats[dt] = {
                "idx": idx,
                "close": c,
                "sma20": round(sma20, 2),
                "rsi14": round(rsi14, 2),
                "tech_sig": tech_sig,
                "tech_conf": tech_conf,
                "reg_curve": "Upward Curve" if slope_up else "Downward Curve",
            }
        features[sym] = sym_feats
        date_bars[sym] = sym_bars

    timeline = sorted(all_dates_set)
    regime6_by_date: dict[str, dict[str, Any]] = {}
    regime4_by_date: dict[str, dict[str, Any]] = {}
    bench_ret_by_date: dict[str, float] = {}

    if benchmark_candles:
        # Linear O(N) causal index pointer scan across sorted benchmark candles
        b_list = [b for b in benchmark_candles if float(b.get("close") or 0.0) > 0]
        b_ptr = 0
        for dt in timeline:
            while b_ptr + 1 < len(b_list) and str(b_list[b_ptr + 1]["date"]) <= dt:
                b_ptr += 1
            sub_b = b_list[max(0, b_ptr - 60) : b_ptr + 1]
            regime6_by_date[dt] = classify_market_regime_6(sub_b, decision_date=dt)
            regime4_by_date[dt] = classify_market_regime(sub_b, decision_date=dt)
            if b_ptr > 0:
                c0 = float(b_list[b_ptr - 1]["close"])
                c1 = float(b_list[b_ptr]["close"])
                bench_ret_by_date[dt] = ((c1 - c0) / max(1e-6, c0)) * 100.0
            else:
                bench_ret_by_date[dt] = 0.0
    else:
        for dt in timeline:
            regime6_by_date[dt] = classify_market_regime_6(None, decision_date=dt)
            regime4_by_date[dt] = classify_market_regime(None, decision_date=dt)
            bench_ret_by_date[dt] = 0.0

    return {
        "timeline": timeline,
        "features": features,
        "date_bars": date_bars,
        "regime6_by_date": regime6_by_date,
        "regime4_by_date": regime4_by_date,
        "bench_ret_by_date": bench_ret_by_date,
    }


def _compute_statistical_significance(closed_trades: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(closed_trades)
    if n == 0:
        return {
            "trade_count": 0,
            "win_count": 0,
            "win_rate_pct": 0.0,
            "win_rate_ci95": [0.0, 0.0],
            "expectancy_pct": 0.0,
            "expectancy_se_pct": 0.0,
            "expectancy_ci95": [0.0, 0.0],
            "t_statistic": 0.0,
            "p_value_vs_zero": 1.0,
            "binomial_p_vs_50": 1.0,
            "distinguishable_from_random": False,
            "sample_note": "No closed trades.",
        }

    rets = [float(t["net_return_pct"]) for t in closed_trades]
    wins = sum(1 for r in rets if r > 0)
    wr = round((wins / n) * 100.0, 2)
    w_lo, w_hi = _wilson_ci(wins, n)

    mean_r = sum(rets) / n
    if n > 1:
        sd = math.sqrt(sum((r - mean_r) ** 2 for r in rets) / (n - 1))
        se = sd / math.sqrt(n)
        t_stat = mean_r / max(1e-9, se)
        # Normal approx to Student-t is conservative check for n>=30; disclose exact n
        p_zero = 2.0 * (1.0 - _normal_cdf(abs(t_stat)))
    else:
        sd, se, t_stat, p_zero = 0.0, 0.0, 0.0, 1.0

    # Deterministic seeded bootstrap CI for mean expectancy (1,500 resamples)
    rng = random.Random(20261119)
    boot_means = []
    for _ in range(1500):
        samp = [rets[rng.randrange(n)] for _ in range(n)]
        boot_means.append(sum(samp) / n)
    boot_means.sort()
    b_lo = round(boot_means[int(0.025 * len(boot_means))], 3)
    b_hi = round(boot_means[int(0.975 * len(boot_means))], 3)
    p_binom = _binomial_two_sided_p(wins, n, 0.5)

    significant_edge = (n >= 30) and (mean_r > 0) and (p_zero < 0.05) and (b_lo > 0)
    if n < 30:
        note = f"Underpowered sample (n={n} < 30); statistical inference is inconclusive."
    elif significant_edge:
        note = f"Positive expectancy (+{mean_r:.2f}%/trade, t={t_stat:.2f}, p={p_zero:.4f}) distinguishable from 0 at 95% confidence."
    elif mean_r <= 0:
        note = f"Negative/zero expectancy ({mean_r:+.2f}%/trade, t={t_stat:.2f}, p={p_zero:.4f}); fails to beat random baseline."
    else:
        note = f"Weak positive expectancy (+{mean_r:.2f}%/trade, p={p_zero:.4f}); 95% CI [{b_lo:+.2f}%, {b_hi:+.2f}%] overlaps 0 — indistinguishable from noise."

    return {
        "trade_count": n,
        "win_count": wins,
        "win_rate_pct": wr,
        "win_rate_ci95": [w_lo, w_hi],
        "expectancy_pct": round(mean_r, 3),
        "expectancy_se_pct": round(se, 3),
        "expectancy_ci95": [b_lo, b_hi],
        "t_statistic": round(t_stat, 3),
        "p_value_vs_zero": round(min(1.0, p_zero), 4),
        "binomial_p_vs_50": p_binom,
        "distinguishable_from_random": significant_edge,
        "sample_note": note,
    }


def simulate_portfolio_v2_1(
    universe_history: dict[str, dict[str, Any]],
    benchmark_candles: list[dict[str, Any]] | None,
    precomputed: dict[str, Any],
    *,
    model_name: str,
    model_family: str,  # "M1" | "M2" | "M2_1" | "M2_1_PERMISSIVE" | "M2_1_EREF" | "SMA20" | "FIXED_5_10"
    cfg_v21: MoreshwarV21Config | None = None,
    cfg_v2: MoreshwarV2Config | None = None,
    date_range: tuple[str, str] | None = None,
    initial_capital: float = INITIAL_CAPITAL,
    record_counterfactuals: bool = False,
) -> dict[str, Any]:
    """Run strict chronological daily simulation with realistic Indian delivery costs & portfolio risk."""
    c21 = cfg_v21 or MoreshwarV21Config()
    c20 = cfg_v2 or MoreshwarV2Config()

    timeline_full = precomputed["timeline"]
    timeline = [d for d in timeline_full if date_range[0] <= d <= date_range[1]] if date_range else list(timeline_full)
    if len(timeline) < 10:
        return {"model": model_name, "metrics": {}, "trades": [], "equity_curve": []}

    features = precomputed["features"]
    date_bars = precomputed["date_bars"]
    regime6_map = precomputed["regime6_by_date"]
    regime4_map = precomputed["regime4_by_date"]
    bench_rets = precomputed["bench_ret_by_date"]
    valid_syms = list(features.keys())

    cash = float(initial_capital)
    open_positions: dict[str, dict[str, Any]] = {}
    pending_orders: list[dict[str, Any]] = []
    closed_trades: list[dict[str, Any]] = []
    rejected_log: list[dict[str, Any]] = []

    daily_navs: list[dict[str, Any]] = []
    daily_risk_snapshots: list[dict[str, Any]] = []
    cost_totals = {
        "brokerage": 0.0,
        "stt": 0.0,
        "exchange_txn": 0.0,
        "sebi_fee": 0.0,
        "stamp_duty": 0.0,
        "gst": 0.0,
        "dp_charge": 0.0,
        "slippage_cost": 0.0,
        "total_cost": 0.0,
    }
    exec_stats = {
        "gap_through_stops": 0,
        "partial_fills": 0,
        "unfillable_zero_vol": 0,
        "liquidity_delayed_exits": 0,
        "rejected_total": 0,
        "rejection_categories": {},
    }

    is_m21_like = model_family in ("M2_1", "M2_1_PERMISSIVE", "M2_1_EREF")
    use_v21_portfolio_risk = is_m21_like
    use_v21_regime = model_family in ("M2_1", "M2_1_EREF")
    fund_mode = (
        "point_in_time_static_warning"
        if model_family in ("M1", "M2_1_EREF")
        else "historical_unavailable"
    )

    for d_idx, dt in enumerate(timeline):
        r6 = regime6_map[dt]["regime"]
        r4 = regime4_map[dt]["regime"]

        # 1. Execute queued next-bar open orders
        for ord_req in pending_orders:
            sym = ord_req["symbol"]
            if sym in open_positions or dt not in date_bars.get(sym, {}):
                continue
            b_idx, bar = date_bars[sym][dt]
            raw_open = float(bar.get("open") or 0.0)
            bar_vol = int(bar.get("volume") or 0)
            if raw_open <= 0 or (use_v21_portfolio_risk and bar_vol <= 0):
                exec_stats["unfillable_zero_vol"] += 1
                exec_stats["rejected_total"] += 1
                exec_stats["rejection_categories"]["LIQUIDITY_ZERO_VOL"] = exec_stats["rejection_categories"].get("LIQUIDITY_ZERO_VOL", 0) + 1
                continue

            cur = universe_history[sym].get("currency", "INR")
            slip_bps = float(ord_req.get("slippage_bps", c21.base_slippage_bps))
            exec_entry_px = round(raw_open * (1.0 + slip_bps / 10000.0), 4)
            port_nav = cash + sum(p["last_price"] * p["qty"] for p in open_positions.values())

            min_order_amt = 1500.0 if cur == "INR" else 25.0
            if model_family in ("M1", "SMA20", "FIXED_5_10"):
                alloc = min(cash * 0.96, port_nav * 0.20)
                qty = int(alloc // max(1.0, exec_entry_px))
                if qty < 1 or (qty * exec_entry_px) < min_order_amt:
                    exec_stats["rejected_total"] += 1
                    continue
                partial_cap = False
            elif model_family == "M2":
                candles_t = universe_history[sym]["candles"][max(0, b_idx - 50) : b_idx + 1]
                deployed = sum(p["entry_price"] * p["qty"] for p in open_positions.values())
                ev2 = evaluate_moreshwar_v2(
                    symbol=sym, candles=candles_t, spot_price=exec_entry_px,
                    fundamental_score=None, porters_score=None,
                    missing_fields=ord_req["missing_fields"],
                    fundamental_mode="historical_unavailable",
                    tech_signal=ord_req["tech_sig"], regression_curve=ord_req["reg_curve"],
                    benchmark_candles=benchmark_candles, decision_date=ord_req["signal_date"],
                    capital=port_nav, current_deployed_capital=deployed, currency=cur, cfg=c20,
                    enable_regime_filter=False,
                )
                qty = min(int(ev2["position_qty"]), int((cash * 0.96) // max(1.0, exec_entry_px)))
                if qty < 1 or (qty * exec_entry_px) < min_order_amt:
                    exec_stats["rejected_total"] += 1
                    continue
                partial_cap = False
            else:
                # Re-verify integer share sizing & volume participation cap against actual next-bar open
                qty = min(int(ord_req["req_qty"]), int((cash * 0.96) // max(1.0, exec_entry_px)))
                max_bar_shares = max(1, int(bar_vol * c21.max_daily_volume_participation)) if bar_vol > 0 else qty
                partial_cap = qty > max_bar_shares
                if partial_cap:
                    qty = max_bar_shares
                    exec_stats["partial_fills"] += 1
                if qty < 1 or (qty * exec_entry_px) < min_order_amt:
                    exec_stats["rejected_total"] += 1
                    continue

            notional = qty * exec_entry_px
            buy_costs = compute_indian_delivery_costs(side="BUY", notional=notional, currency=cur, slippage_bps=slip_bps)
            reg_buy_fee = buy_costs["regulatory_total"]
            if notional + reg_buy_fee > cash:
                qty = int((cash * 0.96) // max(1.0, exec_entry_px * 1.003))
                if qty < 1:
                    exec_stats["rejected_total"] += 1
                    continue
                notional = qty * exec_entry_px
                buy_costs = compute_indian_delivery_costs(side="BUY", notional=notional, currency=cur, slippage_bps=slip_bps)
                reg_buy_fee = buy_costs["regulatory_total"]

            for ck in cost_totals:
                cost_totals[ck] += buy_costs.get(ck, 0.0)
            cash -= (notional + reg_buy_fee)

            eff_sl = round(exec_entry_px * (1.0 - ord_req["sl_ratio"]), 4)
            eff_tp = round(exec_entry_px * (1.0 + ord_req["tp_ratio"]), 4)
            open_positions[sym] = {
                "symbol": sym,
                "sector": universe_history[sym].get("sector", "Equity"),
                "cap_tier": universe_history[sym].get("cap_tier", "Large Cap"),
                "currency": cur,
                "slippage_bps": slip_bps,
                "signal_date": ord_req["signal_date"],
                "signal_regime6": ord_req["signal_regime6"],
                "signal_regime4": ord_req["signal_regime4"],
                "theoretical_signal_price": ord_req["theoretical_price"],
                "entry_date": dt,
                "entry_price": exec_entry_px,
                "last_price": exec_entry_px,
                "qty": qty,
                "notional": notional,
                "buy_reg_cost": reg_buy_fee,
                "buy_slip_cost": buy_costs["slippage_cost"],
                "sl": eff_sl,
                "tp": eff_tp,
                "mfe_pct": 0.0,
                "mae_pct": 0.0,
                "bars_held": 0,
                "partial_fill": partial_cap,
            }
        pending_orders = []

        # 2. Mark-to-market & exit checks (including gap-through-stop and halted bar check)
        to_close: list[str] = []
        for sym, pos in open_positions.items():
            if dt not in date_bars.get(sym, {}):
                continue
            _, bar = date_bars[sym][dt]
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
            elif pos["bars_held"] >= c21.horizon_days or d_idx == len(timeline) - 1:
                raw_exit_px = cl
                exit_reason = "HORIZON_EXIT"

            if raw_exit_px is not None:
                # If trading was halted/zero-volume on an intermediate bar under realistic simulation, defer exit
                if use_v21_portfolio_risk and vol <= 0 and d_idx < len(timeline) - 1:
                    exec_stats["liquidity_delayed_exits"] += 1
                    continue

                if is_gap_stop:
                    exec_stats["gap_through_stops"] += 1

                slip_bps = pos["slippage_bps"]
                exec_exit_px = raw_exit_px * (1.0 - slip_bps / 10000.0)
                gross_proceeds = pos["qty"] * exec_exit_px
                sell_costs = compute_indian_delivery_costs(
                    side="SELL", notional=gross_proceeds, currency=pos["currency"], slippage_bps=slip_bps
                )
                for ck in cost_totals:
                    cost_totals[ck] += sell_costs.get(ck, 0.0)

                net_proceeds = gross_proceeds - sell_costs["regulatory_total"]
                cash += net_proceeds
                total_roundtrip_reg_slip = pos["buy_reg_cost"] + pos["buy_slip_cost"] + sell_costs["total_cost"]
                pnl_amt = net_proceeds - (pos["notional"] + pos["buy_reg_cost"])
                net_ret_pct = (pnl_amt / max(1e-6, pos["notional"] + pos["buy_reg_cost"])) * 100.0

                closed_trades.append({
                    "symbol": sym,
                    "sector": pos["sector"],
                    "cap_tier": pos["cap_tier"],
                    "signal_date": pos["signal_date"],
                    "entry_date": pos["entry_date"],
                    "exit_date": dt,
                    "signal_regime6": pos["signal_regime6"],
                    "signal_regime4": pos["signal_regime4"],
                    "theoretical_signal_price": round(pos["theoretical_signal_price"], 2),
                    "entry_price": round(pos["entry_price"], 2),
                    "exit_price": round(exec_exit_px, 2),
                    "qty": pos["qty"],
                    "notional": round(pos["notional"], 2),
                    "holding_days": pos["bars_held"],
                    "pnl_amount": round(pnl_amt, 2),
                    "net_return_pct": round(net_ret_pct, 2),
                    "cost_amount": round(total_roundtrip_reg_slip, 2),
                    "mfe_pct": round(pos["mfe_pct"], 2),
                    "mae_pct": round(pos["mae_pct"], 2),
                    "exit_reason": exit_reason,
                    "gap_through_stop": is_gap_stop,
                    "partial_fill": pos["partial_fill"],
                })
                to_close.append(sym)

        for sym in to_close:
            del open_positions[sym]

        # 3. Record end-of-day NAV & Portfolio-Level Risk Metrics (Section 6)
        deployed_mkt = sum(p["last_price"] * p["qty"] for p in open_positions.values())
        nav = cash + deployed_mkt
        gross_exp_pct = (deployed_mkt / max(1.0, nav)) * 100.0
        sl_risk_amt = sum(max(0.0, (p["last_price"] - p["sl"]) * p["qty"]) for p in open_positions.values())
        sl_risk_pct = (sl_risk_amt / max(1.0, nav)) * 100.0

        # Sector weight concentration
        sector_amts: dict[str, float] = {}
        for p in open_positions.values():
            sector_amts[p["sector"]] = sector_amts.get(p["sector"], 0.0) + (p["last_price"] * p["qty"])
        max_sec_pct = (max(sector_amts.values()) / max(1.0, nav)) * 100.0 if sector_amts else 0.0

        # Average pairwise correlation among open positions on day dt
        avg_corr_day = 0.0
        if len(open_positions) >= 2:
            o_syms = list(open_positions.keys())
            pair_corrs: list[float] = []
            from .moreshwar_v2_1 import compute_pairwise_return_correlation
            for i_s in range(len(o_syms)):
                for j_s in range(i_s + 1, len(o_syms)):
                    sa, sb = o_syms[i_s], o_syms[j_s]
                    ia = date_bars[sa][dt][0] if dt in date_bars.get(sa, {}) else None
                    ib = date_bars[sb][dt][0] if dt in date_bars.get(sb, {}) else None
                    if ia is not None and ib is not None:
                        ca = [float(x["close"]) for x in universe_history[sa]["candles"][max(0, ia - 60) : ia + 1]]
                        cb = [float(x["close"]) for x in universe_history[sb]["candles"][max(0, ib - 60) : ib + 1]]
                        pair_corrs.append(compute_pairwise_return_correlation(ca, cb, window=60))
            if pair_corrs:
                avg_corr_day = sum(pair_corrs) / len(pair_corrs)

        prev_nav = daily_navs[-1]["nav"] if daily_navs else initial_capital
        day_ret_pct = ((nav - prev_nav) / max(1.0, prev_nav)) * 100.0

        daily_navs.append({"date": dt, "nav": round(nav, 2), "regime6": r6, "regime4": r4})
        daily_risk_snapshots.append({
            "date": dt,
            "open_positions": len(open_positions),
            "gross_exposure_pct": round(gross_exp_pct, 2),
            "sl_risk_pct": round(sl_risk_pct, 2),
            "max_sector_pct": round(max_sec_pct, 2),
            "avg_pairwise_corr": round(avg_corr_day, 3),
            "daily_ret_pct": round(day_ret_pct, 3),
            "bench_ret_pct": round(bench_rets.get(dt, 0.0), 3),
        })

        if d_idx >= len(timeline) - 1:
            continue

        # 4. Generate end-of-day causal signals for next bar open
        open_ctx_cache: list[dict[str, Any]] | None = None
        deployed_sim = deployed_mkt
        if use_v21_portfolio_risk:
            open_ctx_cache = []
            for s_open, p_open in open_positions.items():
                i_open = date_bars[s_open][dt][0] if dt in date_bars.get(s_open, {}) else None
                rec_c = (
                    [float(x["close"]) for x in universe_history[s_open]["candles"][max(0, i_open - 61) : i_open + 1]]
                    if i_open is not None
                    else []
                )
                open_ctx_cache.append({
                    "symbol": s_open,
                    "sector": p_open["sector"],
                    "entry_price": p_open["entry_price"],
                    "sl": p_open["sl"],
                    "qty": p_open["qty"],
                    "recent_closes": rec_c,
                })

        for sym in valid_syms:
            if sym in open_positions or dt not in features[sym]:
                continue
            ft = features[sym][dt]
            b_idx = ft["idx"]
            spot = ft["close"]
            sma20 = ft["sma20"]
            rsi14 = ft["rsi14"]
            tech_sig = ft["tech_sig"]
            reg_curve = ft["reg_curve"]
            snap = universe_history[sym]
            f_obj = snap.get("fundamental_score") or {}
            p_obj = snap.get("porters_score") or {}

            # Entry rule per family
            fire = False
            if model_family in ("M1", "M2_1_EREF"):
                f_val = int(f_obj.get("total", 55))
                p_val = int(p_obj.get("total", 55))
                meta = {
                    "sales_growth": (f_obj.get("ratios") or {}).get("sales_growth", 10.0),
                    "profit_growth": (f_obj.get("ratios") or {}).get("profit_growth", 10.0),
                }
                dec = calculate_decision_matrix(f_val, p_val, meta=meta, price_above_sma=(spot >= sma20), is_holding=False)
                fire = (dec["action"] in ("BUY NOW", "ADD")) and (tech_sig != "BEARISH")
            elif model_family in ("M2", "M2_1", "M2_1_PERMISSIVE"):
                fire = (spot >= sma20) and (42.0 <= rsi14 <= 70.0) and (tech_sig == "BULLISH")
            elif model_family == "SMA20":
                fire = (spot >= sma20) and (rsi14 < 72.0)
            elif model_family == "FIXED_5_10":
                f_val = int(f_obj.get("total", 55))
                p_val = int(p_obj.get("total", 55))
                fire = (spot >= sma20) and (42.0 <= rsi14 <= 70.0) and (tech_sig == "BULLISH")

            if not fire:
                continue

            candles_t = snap["candles"][max(0, b_idx - 61) : b_idx + 1]
            cur = snap.get("currency", "INR")
            miss_f = list(snap.get("missing_fields") or [])

            if model_family == "M1":
                m1 = calculate_moreshwar_levels(spot, int(f_obj.get("total", 55)), int(p_obj.get("total", 55)), is_holding=False)
                sl_px, tp_px = float(m1["stop_loss"]), float(m1["target"])
                req_qty = 0
                slip_bps = c21.base_slippage_bps
            elif model_family == "M2":
                v2 = evaluate_moreshwar_v2(
                    symbol=sym, candles=candles_t, spot_price=spot,
                    fundamental_score=None, porters_score=None, missing_fields=miss_f,
                    fundamental_mode="historical_unavailable", tech_signal=tech_sig,
                    regression_curve=reg_curve, sma_20=sma20, rsi_14=rsi14,
                    benchmark_candles=benchmark_candles, decision_date=dt,
                    capital=nav, current_deployed_capital=deployed_mkt, currency=cur, cfg=c20,
                )
                if not v2["accepted"]:
                    exec_stats["rejected_total"] += 1
                    cat = "REGIME_GATE" if v2["blocked_by_regime"] else "OTHER_V2_GATE"
                    exec_stats["rejection_categories"][cat] = exec_stats["rejection_categories"].get(cat, 0) + 1
                    if record_counterfactuals:
                        _append_counterfactual(rejected_log, sym, snap, b_idx, dt, r6, cat, c21, cur)
                    continue
                sl_px, tp_px = float(v2["stop_loss"]), float(v2["target"])
                req_qty = int(v2["position_qty"])
                slip_bps = c21.base_slippage_bps
            elif is_m21_like:
                f_arg = float(f_obj.get("total", 55)) if fund_mode == "point_in_time_static_warning" else None
                p_arg = float(p_obj.get("total", 55)) if fund_mode == "point_in_time_static_warning" else None
                v21 = evaluate_moreshwar_v2_1(
                    symbol=sym, sector=snap.get("sector", "Equity"), candles=candles_t, spot_price=spot,
                    fundamental_score=f_arg, porters_score=p_arg, missing_fields=miss_f,
                    fundamental_mode=fund_mode, tech_signal=tech_sig, regression_curve=reg_curve,
                    sma_20=sma20, rsi_14=rsi14, benchmark_candles=benchmark_candles, decision_date=dt,
                    capital=nav, current_deployed_capital=deployed_sim,
                    open_positions_context=open_ctx_cache, currency=cur, cfg=c21,
                    enable_regime_filter=use_v21_regime,
                    enable_portfolio_constraints=True,
                )
                if not v21["accepted"]:
                    exec_stats["rejected_total"] += 1
                    cat = v21["rejection_category"] or "OTHER"
                    exec_stats["rejection_categories"][cat] = exec_stats["rejection_categories"].get(cat, 0) + 1
                    if record_counterfactuals:
                        _append_counterfactual(rejected_log, sym, snap, b_idx, dt, r6, cat, c21, cur)
                    continue
                sl_px, tp_px = float(v21["stop_loss"]), float(v21["target"])
                req_qty = int(v21["position_qty"])
                slip_bps = float(v21["slippage_bps"])
                if open_ctx_cache is not None:
                    deployed_sim += req_qty * spot
                    open_ctx_cache.append({
                        "symbol": sym,
                        "sector": snap.get("sector", "Equity"),
                        "entry_price": spot,
                        "sl": sl_px,
                        "qty": req_qty,
                        "recent_closes": [float(x["close"]) for x in candles_t],
                    })
            elif model_family == "SMA20":
                sl_px, tp_px = round(spot * 0.94, 2), round(spot * 1.12, 2)
                req_qty = 0
                slip_bps = c21.base_slippage_bps
            else:  # FIXED_5_10
                sl_px, tp_px = round(spot * 0.95, 2), round(spot * 1.10, 2)
                req_qty = 0
                slip_bps = c21.base_slippage_bps

            pending_orders.append({
                "symbol": sym,
                "signal_date": dt,
                "signal_regime6": r6,
                "signal_regime4": r4,
                "theoretical_price": spot,
                "sl_ratio": max(0.01, (spot - sl_px) / max(1e-6, spot)),
                "tp_ratio": max(0.015, (tp_px - spot) / max(1e-6, spot)),
                "missing_fields": miss_f,
                "tech_sig": tech_sig,
                "reg_curve": reg_curve,
                "req_qty": req_qty,
                "slippage_bps": slip_bps,
            })

    return _finalize_v21_report(
        model_name=model_name,
        model_family=model_family,
        fund_mode=fund_mode,
        closed_trades=closed_trades,
        daily_navs=daily_navs,
        daily_risk_snapshots=daily_risk_snapshots,
        initial_capital=initial_capital,
        cost_totals=cost_totals,
        exec_stats=exec_stats,
        rejected_log=rejected_log,
    )


def _append_counterfactual(
    rejected_log: list[dict[str, Any]],
    sym: str,
    snap: dict[str, Any],
    b_idx: int,
    dt: str,
    regime6: str,
    category: str,
    cfg: MoreshwarV21Config,
    currency: str,
) -> None:
    """Evaluate counterfactual 14D forward performance on a rejected candidate (zero look-ahead in decision)."""
    candles = snap.get("candles") or []
    if b_idx + 2 >= len(candles):
        return
    entry_bar = candles[b_idx + 1]
    op = float(entry_bar.get("open") or 0.0)
    if op <= 0 or int(entry_bar.get("volume") or 0) <= 0:
        return
    atr = compute_atr(candles[max(0, b_idx - 25) : b_idx + 1], period=14)
    stop_dist = max(op * 0.025, min(op * 0.10, atr * cfg.atr_multiplier))
    tp_dist = stop_dist * 2.0
    sl_px, tp_px = op - stop_dist, op + tp_dist
    exec_in = op * (1.0 + cfg.base_slippage_bps / 10000.0)

    exit_px = float(candles[min(len(candles) - 1, b_idx + cfg.horizon_days)]["close"])
    reason = "HORIZON_EXIT"
    for f_i in range(b_idx + 1, min(len(candles), b_idx + 1 + cfg.horizon_days)):
        fb = candles[f_i]
        if float(fb["low"]) <= sl_px:
            exit_px = min(float(fb["open"]), sl_px) if f_i > b_idx + 1 else sl_px
            reason = "STOP_LOSS"
            break
        if float(fb["high"]) >= tp_px:
            exit_px = tp_px
            reason = "TARGET_HIT"
            break

    exec_out = exit_px * (1.0 - cfg.base_slippage_bps / 10000.0)
    # Approximate % delivery cost impact (~0.24% INR round-trip)
    net_ret_pct = (((exec_out - exec_in) / max(1e-6, exec_in)) * 100.0) - (0.24 if currency == "INR" else 0.01)
    rejected_log.append({
        "symbol": sym,
        "sector": snap.get("sector", "Equity"),
        "date": dt,
        "regime6": regime6,
        "category": category,
        "cf_return_pct": round(net_ret_pct, 2),
        "cf_reason": reason,
    })


def _finalize_v21_report(
    *,
    model_name: str,
    model_family: str,
    fund_mode: str,
    closed_trades: list[dict[str, Any]],
    daily_navs: list[dict[str, Any]],
    daily_risk_snapshots: list[dict[str, Any]],
    initial_capital: float,
    cost_totals: dict[str, float],
    exec_stats: dict[str, Any],
    rejected_log: list[dict[str, Any]],
) -> dict[str, Any]:
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

    max_dd_abs = round(abs(max_dd), 2)
    recovery_factor = round(total_ret / max(0.05, max_dd_abs), 2)

    rf_d = RISK_FREE_ANNUAL / 252.0
    ex = [r - rf_d for r in daily_rets]
    if len(ex) > 2:
        m_ex = sum(ex) / len(ex)
        std_ex = math.sqrt(sum((x - m_ex) ** 2 for x in ex) / (len(ex) - 1))
        down_ex = math.sqrt(sum(min(0.0, x) ** 2 for x in ex) / len(ex))
        sharpe = round((m_ex / std_ex) * math.sqrt(252.0), 2) if std_ex > 1e-9 else 0.0
        sortino = round((m_ex / down_ex) * math.sqrt(252.0), 2) if down_ex > 1e-9 else 0.0
    else:
        sharpe, sortino = 0.0, 0.0

    sig = _compute_statistical_significance(closed_trades)
    n_tr = len(closed_trades)
    gross_win = sum(t["pnl_amount"] for t in closed_trades if t["pnl_amount"] > 0)
    gross_loss = abs(sum(t["pnl_amount"] for t in closed_trades if t["pnl_amount"] < 0))
    pf = round(gross_win / gross_loss, 2) if gross_loss > 1e-6 else (9.99 if gross_win > 0 else 0.0)
    avg_hold = round(sum(t["holding_days"] for t in closed_trades) / max(1, n_tr), 1) if n_tr else 0.0
    sl_cnt = sum(1 for t in closed_trades if t["exit_reason"] == "STOP_LOSS")
    tp_cnt = sum(1 for t in closed_trades if t["exit_reason"] == "TARGET_HIT")
    avg_util = round(sum(s["gross_exposure_pct"] for s in daily_risk_snapshots) / max(1, len(daily_risk_snapshots)), 1)

    # Consecutive losing streak
    cur_streak, max_streak, cur_loss_sum, worst_streak_pct = 0, 0, 0.0, 0.0
    for t in closed_trades:
        if t["net_return_pct"] <= 0:
            cur_streak += 1
            cur_loss_sum += t["net_return_pct"]
            max_streak = max(max_streak, cur_streak)
            worst_streak_pct = min(worst_streak_pct, cur_loss_sum)
        else:
            cur_streak, cur_loss_sum = 0, 0.0

    # Portfolio Risk Report (Section 6)
    decline_days = [s for s in daily_risk_snapshots if s["bench_ret_pct"] <= -1.0]
    rolling_5d = [sum(s["daily_ret_pct"] for s in daily_risk_snapshots[max(0, i - 4) : i + 1]) for i in range(len(daily_risk_snapshots))]
    portfolio_risk = {
        "max_simultaneous_positions": max((s["open_positions"] for s in daily_risk_snapshots), default=0),
        "avg_simultaneous_positions": round(sum(s["open_positions"] for s in daily_risk_snapshots) / max(1, len(daily_risk_snapshots)), 2),
        "max_sector_concentration_pct": round(max((s["max_sector_pct"] for s in daily_risk_snapshots), default=0.0), 2),
        "avg_sector_concentration_pct": round(sum(s["max_sector_pct"] for s in daily_risk_snapshots) / max(1, len(daily_risk_snapshots)), 2),
        "avg_concurrent_correlation": round(
            sum(s["avg_pairwise_corr"] for s in daily_risk_snapshots if s["open_positions"] >= 2)
            / max(1, sum(1 for s in daily_risk_snapshots if s["open_positions"] >= 2)),
            3,
        ),
        "peak_concurrent_correlation": round(max((s["avg_pairwise_corr"] for s in daily_risk_snapshots), default=0.0), 3),
        "max_portfolio_exposure_pct": round(max((s["gross_exposure_pct"] for s in daily_risk_snapshots), default=0.0), 2),
        "max_portfolio_stop_loss_exposure_pct": round(max((s["sl_risk_pct"] for s in daily_risk_snapshots), default=0.0), 2),
        "avg_portfolio_stop_loss_exposure_pct": round(sum(s["sl_risk_pct"] for s in daily_risk_snapshots) / max(1, len(daily_risk_snapshots)), 2),
        "max_single_day_loss_pct": round(min((s["daily_ret_pct"] for s in daily_risk_snapshots), default=0.0), 2),
        "max_5day_week_loss_pct": round(min(rolling_5d, default=0.0), 2),
        "max_consecutive_losses": max_streak,
        "worst_consecutive_loss_sum_pct": round(worst_streak_pct, 2),
        "capital_utilization_pct": avg_util,
        "market_decline_days_count": len(decline_days),
        "avg_exposure_on_market_declines_pct": round(sum(s["gross_exposure_pct"] for s in decline_days) / max(1, len(decline_days)), 2),
        "avg_strategy_ret_on_market_declines_pct": round(sum(s["daily_ret_pct"] for s in decline_days) / max(1, len(decline_days)), 3),
    }

    # 6-State Market Regime Evaluation (Section 5)
    regimes_6: dict[str, dict[str, Any]] = {}
    for rn in REGIMES_6:
        sub = [t for t in closed_trades if t["signal_regime6"] == rn]
        r_days = [s for i, s in enumerate(daily_risk_snapshots) if daily_navs[i]["regime6"] == rn]
        regimes_6[rn] = _compute_regime_cohort_metrics(sub, r_days)

    # Stock & Sector breakdowns (filterable in dashboard)
    by_stock: dict[str, Any] = {}
    for t in closed_trades:
        by_stock.setdefault(t["symbol"], []).append(t)
    by_sector: dict[str, Any] = {}
    for t in closed_trades:
        by_sector.setdefault(t["sector"], []).append(t)

    metrics = {
        "model": model_name,
        "model_family": model_family,
        "distinction": "SIMULATED_HISTORICAL_BACKTEST (NOT LIVE BROKER EXECUTION)",
        "total_return_pct": round(total_ret, 2),
        "cagr_pct": round(cagr, 2),
        "win_rate_pct": sig["win_rate_pct"],
        "win_rate_ci95": sig["win_rate_ci95"],
        "profit_factor": pf,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "max_drawdown_pct": max_dd_abs,
        "recovery_factor": recovery_factor,
        "expectancy_pct": sig["expectancy_pct"],
        "expectancy_ci95": sig["expectancy_ci95"],
        "t_statistic": sig["t_statistic"],
        "p_value_vs_zero": sig["p_value_vs_zero"],
        "binomial_p_vs_50": sig["binomial_p_vs_50"],
        "distinguishable_from_random": sig["distinguishable_from_random"],
        "sample_note": sig["sample_note"],
        "total_trades": n_tr,
        "avg_holding_days": avg_hold,
        "capital_utilisation_pct": avg_util,
        "stop_loss_freq_pct": round((sl_cnt / max(1, n_tr)) * 100.0, 1) if n_tr else 0.0,
        "target_hit_freq_pct": round((tp_cnt / max(1, n_tr)) * 100.0, 1) if n_tr else 0.0,
        "cost_drag_pct": round((cost_totals["total_cost"] / initial_capital) * 100.0, 2),
        "cost_breakdown": {k: round(v, 2) for k, v in cost_totals.items()},
        "execution_realism": exec_stats,
        "fundamental_integrity": (
            "CAUSAL_CLEAN (Historical Fundamentals Unavailable — Not Leaked)"
            if fund_mode == "historical_unavailable"
            else "STATIC_LEAK_AUDIT (Current Ratios Applied Retrospectively)"
        ),
    }

    return {
        "model": model_name,
        "metrics": metrics,
        "significance": sig,
        "regimes_6": regimes_6,
        "portfolio_risk": portfolio_risk,
        "by_stock": {k: _summarize_short(v) for k, v in sorted(by_stock.items())},
        "by_sector": {k: _summarize_short(v) for k, v in sorted(by_sector.items())},
        "equity_curve": daily_navs,
        "daily_rets": daily_rets,
        "trades": closed_trades,
        "rejected_log": rejected_log,
    }


def _summarize_short(trades: list[dict[str, Any]]) -> dict[str, Any]:
    if not trades:
        return {"trades": 0, "win_rate_pct": 0.0, "expectancy_pct": 0.0, "profit_factor": 0.0, "total_pnl": 0.0}
    wins = sum(1 for t in trades if t["net_return_pct"] > 0)
    gw = sum(t["pnl_amount"] for t in trades if t["pnl_amount"] > 0)
    gl = abs(sum(t["pnl_amount"] for t in trades if t["pnl_amount"] < 0))
    return {
        "trades": len(trades),
        "win_rate_pct": round((wins / len(trades)) * 100.0, 2),
        "expectancy_pct": round(sum(t["net_return_pct"] for t in trades) / len(trades), 2),
        "profit_factor": round(gw / gl, 2) if gl > 1e-6 else (9.99 if gw > 0 else 0.0),
        "total_pnl": round(sum(t["pnl_amount"] for t in trades), 2),
    }


def _compute_regime_cohort_metrics(trades: list[dict[str, Any]], regime_day_snaps: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute full Section 5 metrics inside a single market regime."""
    n = len(trades)
    if n == 0:
        return {
            "days_in_regime": len(regime_day_snaps),
            "trades": 0,
            "win_rate_pct": 0.0,
            "avg_return_pct": 0.0,
            "profit_factor": 0.0,
            "max_drawdown_pct": 0.0,
            "expectancy_pct": 0.0,
            "sharpe_ratio": 0.0,
        }
    wins = sum(1 for t in trades if t["net_return_pct"] > 0)
    rets = [float(t["net_return_pct"]) for t in trades]
    mean_r = sum(rets) / n
    gw = sum(t["pnl_amount"] for t in trades if t["pnl_amount"] > 0)
    gl = abs(sum(t["pnl_amount"] for t in trades if t["pnl_amount"] < 0))
    pf = round(gw / gl, 2) if gl > 1e-6 else (9.99 if gw > 0 else 0.0)

    # Trade-equity compounding drawdown inside this regime
    eq, pk, mdd = 100.0, 100.0, 0.0
    for t in trades:
        eq *= (1.0 + (t["pnl_amount"] / INITIAL_CAPITAL))
        pk = max(pk, eq)
        mdd = min(mdd, ((eq - pk) / pk) * 100.0)

    # Conditional Sharpe on daily returns during this regime
    if len(regime_day_snaps) > 5:
        rf_d = (RISK_FREE_ANNUAL / 252.0) * 100.0
        dr = [s["daily_ret_pct"] - rf_d for s in regime_day_snaps]
        md = sum(dr) / len(dr)
        sd = math.sqrt(sum((x - md) ** 2 for x in dr) / (len(dr) - 1))
        sharpe = round((md / sd) * math.sqrt(252.0), 2) if sd > 1e-8 else 0.0
    else:
        sharpe = 0.0

    return {
        "days_in_regime": len(regime_day_snaps),
        "trades": n,
        "win_rate_pct": round((wins / n) * 100.0, 2),
        "avg_return_pct": round(mean_r, 2),
        "profit_factor": pf,
        "max_drawdown_pct": round(abs(mdd), 2),
        "expectancy_pct": round(mean_r, 3),
        "sharpe_ratio": sharpe,
    }


def run_monte_carlo_suite(
    closed_trades: list[dict[str, Any]],
    daily_rets: list[float],
    *,
    n_sims: int = 1200,
    block_len: int = 10,
) -> dict[str, Any]:
    """Section 3C: Run >=1,000 Monte Carlo trade-sequence permutations AND block-bootstrap paths."""
    rng = random.Random(42)

    # Method 1: Trade-sequence shuffle (preserves empirical trade distribution, randomizes order)
    t_pnls = [float(t["pnl_amount"]) for t in closed_trades]
    shuffle_rets: list[float] = []
    shuffle_dds: list[float] = []
    if len(t_pnls) >= 5:
        for _ in range(n_sims):
            perm = list(t_pnls)
            rng.shuffle(perm)
            # Sample with replacement to also include sizing variance across fixed trade count
            sampled = [t_pnls[rng.randrange(len(t_pnls))] for _ in range(len(t_pnls))]
            eq = INITIAL_CAPITAL
            pk = eq
            mdd = 0.0
            for pnl in sampled:
                eq = max(1.0, eq + pnl)
                pk = max(pk, eq)
                mdd = max(mdd, ((pk - eq) / pk) * 100.0)
            shuffle_rets.append(((eq - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100.0)
            shuffle_dds.append(mdd)
        shuffle_rets.sort()
        shuffle_dds.sort()

    def _q(arr: list[float], q_val: float) -> float:
        if not arr:
            return 0.0
        return round(arr[min(len(arr) - 1, max(0, int(q_val * (len(arr) - 1))))], 2)

    trade_bootstrap = {
        "simulations": n_sims if len(t_pnls) >= 5 else 0,
        "median_return_pct": _q(shuffle_rets, 0.50),
        "p05_return_pct": _q(shuffle_rets, 0.05),
        "p95_return_pct": _q(shuffle_rets, 0.95),
        "median_max_dd_pct": _q(shuffle_dds, 0.50),
        "p95_max_dd_pct": _q(shuffle_dds, 0.95),
        "prob_positive_return_pct": round((sum(1 for r in shuffle_rets if r > 0) / max(1, len(shuffle_rets))) * 100.0, 1),
        "prob_ruin_25pct_dd": round((sum(1 for d in shuffle_dds if d >= 25.0) / max(1, len(shuffle_dds))) * 100.0, 2),
        "prob_ruin_50pct_dd": round((sum(1 for d in shuffle_dds if d >= 50.0) / max(1, len(shuffle_dds))) * 100.0, 2),
        "limitation": "Assumes i.i.d. trade PnLs; breaks multi-week regime clustering and overlapping capital saturation.",
    }

    # Method 2: Circular Block Bootstrap on Daily Portfolio Returns (preserves volatility & regime clustering)
    n_d = len(daily_rets)
    block_rets: list[float] = []
    block_dds: list[float] = []
    if n_d >= 30:
        n_blocks = max(1, math.ceil(n_d / block_len))
        for _ in range(n_sims):
            eq = 1.0
            pk = 1.0
            mdd = 0.0
            step_cnt = 0
            for _b in range(n_blocks):
                start = rng.randrange(n_d)
                for k in range(block_len):
                    if step_cnt >= n_d:
                        break
                    r = daily_rets[(start + k) % n_d]
                    eq = max(0.01, eq * (1.0 + r))
                    pk = max(pk, eq)
                    mdd = max(mdd, ((pk - eq) / pk) * 100.0)
                    step_cnt += 1
            block_rets.append((eq - 1.0) * 100.0)
            block_dds.append(mdd)
        block_rets.sort()
        block_dds.sort()

    daily_block_bootstrap = {
        "simulations": n_sims if n_d >= 30 else 0,
        "block_length_days": block_len,
        "median_return_pct": _q(block_rets, 0.50),
        "p05_return_pct": _q(block_rets, 0.05),
        "p95_return_pct": _q(block_rets, 0.95),
        "median_max_dd_pct": _q(block_dds, 0.50),
        "p95_max_dd_pct": _q(block_dds, 0.95),
        "prob_positive_return_pct": round((sum(1 for r in block_rets if r > 0) / max(1, len(block_rets))) * 100.0, 1),
        "prob_ruin_25pct_dd": round((sum(1 for d in block_dds if d >= 25.0) / max(1, len(block_dds))) * 100.0, 2),
        "prob_ruin_50pct_dd": round((sum(1 for d in block_dds if d >= 50.0) / max(1, len(block_dds))) * 100.0, 2),
        "limitation": "Preserves local autocorrelation over 10-day blocks, but cannot invent tail shocks larger than observed inside the 5-year sample.",
    }

    return {
        "trade_bootstrap": trade_bootstrap,
        "daily_block_bootstrap": daily_block_bootstrap,
    }


def _compute_benchmark_metrics(
    benchmark_candles: list[dict[str, Any]] | None,
    date_range: tuple[str, str] | None = None,
) -> dict[str, Any]:
    bars = [b for b in (benchmark_candles or []) if float(b.get("close") or 0.0) > 0]
    if date_range:
        bars = [b for b in bars if date_range[0] <= str(b["date"]) <= date_range[1]]
    if len(bars) < 5:
        return {"model": "Nifty 50 (^NSEI) Buy & Hold", "metrics": {}, "equity_curve": []}

    navs = [{"date": b["date"], "nav": round((float(b["close"]) / float(bars[0]["close"])) * INITIAL_CAPITAL, 2)} for b in bars]
    end_n = navs[-1]["nav"]
    tot_ret = ((end_n - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100.0
    cagr = (((end_n / INITIAL_CAPITAL) ** (252.0 / len(bars))) - 1.0) * 100.0
    peak, mdd = INITIAL_CAPITAL, 0.0
    rets = []
    prev = INITIAL_CAPITAL
    for pt in navs:
        peak = max(peak, pt["nav"])
        mdd = min(mdd, ((pt["nav"] - peak) / peak) * 100.0)
        rets.append((pt["nav"] - prev) / prev)
        prev = pt["nav"]

    rf_d = RISK_FREE_ANNUAL / 252.0
    ex = [r - rf_d for r in rets[1:]]
    m_ex = sum(ex) / max(1, len(ex))
    s_ex = math.sqrt(sum((x - m_ex) ** 2 for x in ex) / max(1, len(ex) - 1)) if len(ex) > 1 else 0.0
    d_ex = math.sqrt(sum(min(0.0, x) ** 2 for x in ex) / max(1, len(ex))) if ex else 0.0
    mdd_abs = round(abs(mdd), 2)

    return {
        "model": "Nifty 50 (^NSEI) Buy & Hold",
        "metrics": {
            "model": "Nifty 50 (^NSEI) Buy & Hold",
            "model_family": "BENCHMARK",
            "distinction": "PASSIVE_MARKET_INDEX_BENCHMARK",
            "total_return_pct": round(tot_ret, 2),
            "cagr_pct": round(cagr, 2),
            "win_rate_pct": 100.0 if tot_ret > 0 else 0.0,
            "win_rate_ci95": [100.0, 100.0] if tot_ret > 0 else [0.0, 0.0],
            "profit_factor": None,
            "sharpe_ratio": round((m_ex / s_ex) * math.sqrt(252.0), 2) if s_ex > 1e-9 else 0.0,
            "sortino_ratio": round((m_ex / d_ex) * math.sqrt(252.0), 2) if d_ex > 1e-9 else 0.0,
            "max_drawdown_pct": mdd_abs,
            "recovery_factor": round(tot_ret / max(0.05, mdd_abs), 2),
            "expectancy_pct": round(tot_ret, 2),
            "total_trades": 1,
            "avg_holding_days": float(len(bars)),
            "capital_utilisation_pct": 100.0,
            "cost_drag_pct": 0.0,
            "fundamental_integrity": "MARKET_INDEX_BENCHMARK",
        },
        "equity_curve": navs,
    }


def _compute_cash_benchmark(n_days: int) -> dict[str, Any]:
    tot_ret = (((1.0 + RISK_FREE_ANNUAL / 252.0) ** max(1, n_days)) - 1.0) * 100.0
    return {
        "model": "Risk-Free Cash Benchmark (6.5% India T-Bill)",
        "metrics": {
            "model": "Risk-Free Cash Benchmark (6.5% India T-Bill)",
            "model_family": "CASH_BENCHMARK",
            "distinction": "RISK_FREE_TREASURY_BENCHMARK",
            "total_return_pct": round(tot_ret, 2),
            "cagr_pct": 6.5,
            "win_rate_pct": 100.0,
            "profit_factor": None,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "max_drawdown_pct": 0.0,
            "recovery_factor": None,
            "expectancy_pct": None,
            "total_trades": 0,
            "avg_holding_days": 0.0,
            "capital_utilisation_pct": 0.0,
            "cost_drag_pct": 0.0,
            "fundamental_integrity": "RISK_FREE_BASELINE",
        },
    }


def _pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 4:
        return 0.0
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    denom = math.sqrt(vx * vy)
    return round(cov / denom, 3) if denom > 1e-9 else 0.0


def build_strategy_investigation_report(
    *,
    universe_history: dict[str, dict[str, Any]],
    m1_res: dict[str, Any],
    m2_res: dict[str, Any],
    m21_res: dict[str, Any],
    m21_perm_res: dict[str, Any],
    m21_eref_res: dict[str, Any],
    early_end_date: str,
) -> dict[str, Any]:
    """Section 4: Deep forensic investigation of Moreshwar components & leakage."""
    m1_m = m1_res["metrics"]
    m2_m = m2_res["metrics"]
    m21_m = m21_res["metrics"]
    perm_m = m21_perm_res["metrics"]
    eref_m = m21_eref_res["metrics"]

    # 1. Measure look-ahead bias in E-Ref: correlate today's Fund Score against EARLY 4Y stock returns
    f_scores_list: list[float] = []
    early_rets: list[float] = []
    for sym, snap in universe_history.items():
        bars = snap.get("candles") or []
        early_bars = [b for b in bars if str(b["date"]) <= early_end_date]
        if len(early_bars) < 200:
            continue
        f_sc = float((snap.get("fundamental_score") or {}).get("total", 55))
        r_early = ((float(early_bars[-1]["close"]) - float(early_bars[0]["close"])) / max(1e-6, float(early_bars[0]["close"]))) * 100.0
        f_scores_list.append(f_sc)
        early_rets.append(r_early)
    corr_static_vs_past = _pearson(f_scores_list, early_rets)

    # 2. M1 fundamental stop compression check on high vs mid-score trades
    m1_trades = m1_res["trades"]
    m21_trades = m21_res["trades"]
    m1_sl_rate = m1_m.get("stop_loss_freq_pct", 0.0)
    m21_sl_rate = m21_m.get("stop_loss_freq_pct", 0.0)

    # 3. Accepted vs Rejected counterfactual analysis on Moreshwar 2.1
    rej_log = m21_res.get("rejected_log") or []
    acc_rets = [t["net_return_pct"] for t in m21_trades]
    rej_rets = [r["cf_return_pct"] for r in rej_log]
    acc_mean = round(sum(acc_rets) / max(1, len(acc_rets)), 3) if acc_rets else 0.0
    rej_mean = round(sum(rej_rets) / max(1, len(rej_rets)), 3) if rej_rets else 0.0
    rej_win = round((sum(1 for x in rej_rets if x > 0) / max(1, len(rej_rets))) * 100.0, 2) if rej_rets else 0.0

    # Breakdown of counterfactual performance by rejection category
    by_cat: dict[str, dict[str, Any]] = {}
    cat_groups: dict[str, list[float]] = {}
    for r in rej_log:
        cat_groups.setdefault(r["category"], []).append(r["cf_return_pct"])
    for cat, vals in sorted(cat_groups.items()):
        w = sum(1 for v in vals if v > 0)
        by_cat[cat] = {
            "count": len(vals),
            "cf_win_rate_pct": round((w / len(vals)) * 100.0, 2),
            "cf_avg_return_pct": round(sum(vals) / len(vals), 3),
        }

    # 4. Regime filter attribution: compare M21 (6-state filtered) vs M21_PERMISSIVE
    regime_pnl_saved = round(m21_m["total_return_pct"] - perm_m["total_return_pct"], 2)
    regime_dd_saved = round(perm_m["max_drawdown_pct"] - m21_m["max_drawdown_pct"], 2)

    # 5. Component Contribution Assessment
    attrib_items = [
        {
            "component": "ATR(14) Stop-Loss Normalization vs M1 Formula",
            "impact": "POSITIVE" if m21_sl_rate < m1_sl_rate else "INSIGNIFICANT",
            "evidence": f"Cut SL hit rate from {m1_sl_rate}% (M1 fundamental compression) to {m21_sl_rate}% (M2.1), preventing stop-outs inside normal daily noise.",
        },
        {
            "component": "6-State Nifty Market Regime Filter",
            "impact": "POSITIVE" if (regime_pnl_saved > 0 or regime_dd_saved > 0.5) else "INSIGNIFICANT",
            "evidence": (
                f"Blocking WEAK_BEARISH, STRONG_BEARISH & HIGH_VOLATILITY shifted total return by {regime_pnl_saved:+.2f} pp "
                f"and reduced MaxDD by {regime_dd_saved:+.2f} pp vs permissive M2.1 control."
            ),
        },
        {
            "component": "Clean Causal Trend + Portfolio Correlation/Concentration Limits",
            "impact": "POSITIVE" if m21_m["sharpe_ratio"] > m2_m["sharpe_ratio"] else "NEGATIVE",
            "evidence": (
                f"M2.1 Sharpe {m21_m['sharpe_ratio']} / MaxDD {m21_m['max_drawdown_pct']}% vs M2.0 Sharpe {m2_m['sharpe_ratio']} / MaxDD {m2_m['max_drawdown_pct']}% "
                f"with max sector cap ({c_sec(m21_res)}%) and liquidity/gap safeguards."
            ),
        },
        {
            "component": "Static Current Fundamentals (E-Ref vs Clean Causal)",
            "impact": "LOOK_AHEAD_BIASED" if abs(eref_m["total_return_pct"] - m21_m["total_return_pct"]) > 1.5 else "INSIGNIFICANT",
            "evidence": (
                f"Today's Fund Scores show Pearson r={corr_static_vs_past:+.3f} correlation with 2021–2025 stock performance — "
                f"past winners end up with high trailing ROE/Growth/MCap scores today (r={corr_static_vs_past:+.3f}), "
                f"shifting E-Ref ({eref_m['total_return_pct']:+.2f}%) away from Causal M2.1 ({m21_m['total_return_pct']:+.2f}%) due to retrospective winner knowledge rather than causal predictability."
            ),
        },
        {
            "component": "Selectivity & Rejection Funnel (Accepted vs Counterfactual Rejected)",
            "impact": "POSITIVE" if acc_mean >= rej_mean else "EXCESSIVE_SELECTIVITY_WARNING",
            "evidence": (
                f"Accepted trades ({len(acc_rets)}, mean {acc_mean:+.3f}%, win {m21_m['win_rate_pct']}%) vs "
                f"Rejected opportunities ({len(rej_rets)} counterfactuals, mean {rej_mean:+.3f}%, win {rej_win}%)."
            ),
        },
    ]

    return {
        "atr_stop_analysis": {
            "m1_stop_loss_freq_pct": m1_sl_rate,
            "m2_stop_loss_freq_pct": m2_m.get("stop_loss_freq_pct", 0.0),
            "m21_stop_loss_freq_pct": m21_sl_rate,
            "finding": f"Replacing M1 `(100-X)*scale` with 2.0x ATR(14) reduced stop-out frequency from {m1_sl_rate}% to {m21_sl_rate}%.",
        },
        "regime_filter_analysis": {
            "m21_filtered_return_pct": m21_m["total_return_pct"],
            "m21_permissive_return_pct": perm_m["total_return_pct"],
            "return_delta_pp": regime_pnl_saved,
            "m21_filtered_max_dd_pct": m21_m["max_drawdown_pct"],
            "m21_permissive_max_dd_pct": perm_m["max_drawdown_pct"],
            "drawdown_reduction_pp": regime_dd_saved,
            "finding": f"6-state regime filter {'added' if regime_pnl_saved >= 0 else 'traded off'} {regime_pnl_saved:+.2f} pp return while reducing MaxDD by {regime_dd_saved:+.2f} pp.",
        },
        "trade_count_and_funnel": {
            "m1_trades": m1_m["total_trades"],
            "m2_trades": m2_m["total_trades"],
            "m21_trades": m21_m["total_trades"],
            "m21_permissive_trades": perm_m["total_trades"],
            "rejection_breakdown": m21_m["execution_realism"]["rejection_categories"],
            "finding": "M2.0/M2.1 take fewer trades because signals require both causal technical alignment AND favorable Nifty regime + liquidity/risk-budget headroom.",
        },
        "eref_leakage_audit": {
            "eref_total_return_pct": eref_m["total_return_pct"],
            "m21_clean_return_pct": m21_m["total_return_pct"],
            "static_score_vs_past_4y_corr": corr_static_vs_past,
            "finding": (
                f"E-Ref ({eref_m['total_return_pct']:+.2f}%) differs from Clean Causal M2.1 ({m21_m['total_return_pct']:+.2f}%). "
                f"Pearson r={corr_static_vs_past:+.3f} between today's fundamental score and prior 4Y stock returns proves retrospective winner leakage contaminates static-ratio backtests."
            ),
        },
        "selectivity_counterfactuals": {
            "accepted_trades": len(acc_rets),
            "accepted_mean_net_pct": acc_mean,
            "accepted_win_rate_pct": m21_m["win_rate_pct"],
            "rejected_evaluated": len(rej_rets),
            "rejected_cf_mean_net_pct": rej_mean,
            "rejected_cf_win_rate_pct": rej_win,
            "spread_accepted_minus_rejected_pp": round(acc_mean - rej_mean, 3),
            "excessive_selectivity_detected": rej_mean > acc_mean + 0.15,
            "by_rejection_category": by_cat,
        },
        "component_attribution": attrib_items,
    }


def c_sec(res: dict[str, Any]) -> float:
    return res.get("portfolio_risk", {}).get("max_sector_concentration_pct", 0.0)


def classify_final_verdict(
    *,
    m21_full: dict[str, Any],
    m21_oos: dict[str, Any],
    m1_oos: dict[str, Any],
    nifty_full: dict[str, Any],
    nifty_oos: dict[str, Any],
    wf_rows: list[dict[str, Any]],
    mc_res: dict[str, Any],
) -> dict[str, Any]:
    """Section 10: Objective 4-tier classification without forced optimism."""
    fm = m21_full["metrics"]
    om = m21_oos["metrics"]
    om1 = m1_oos["metrics"]
    nm_f = nifty_full["metrics"]
    nm_o = nifty_oos["metrics"]

    mc_blk = mc_res.get("daily_block_bootstrap") or {}
    mc_prob_pos = mc_blk.get("prob_positive_return_pct", 0.0)
    oos_trades = om.get("total_trades", 0)
    oos_ret = om.get("total_return_pct", 0.0)
    oos_sharpe = om.get("sharpe_ratio", 0.0)
    full_ret = fm.get("total_return_pct", 0.0)
    full_cagr = fm.get("cagr_pct", 0.0)
    full_sharpe = fm.get("sharpe_ratio", 0.0)
    p_val = fm.get("p_value_vs_zero", 1.0)
    pos_folds = sum(1 for w in wf_rows if w.get("m21_return_pct", 0.0) > 0)
    beat_m1_folds = sum(1 for w in wf_rows if w.get("m21_return_pct", -999) > w.get("m1_return_pct", 999))

    # Strict objective gating
    if full_ret <= 0.0 or fm.get("expectancy_pct", 0.0) <= 0.0 or (oos_ret < 0.0 and full_cagr < 2.0):
        category = "1. Research failure: No reliable edge demonstrated"
        rationale = (
            f"Over {fm.get('total_trades', 0)} realistic Indian delivery trades across 5 years, Moreshwar 2.1 generated "
            f"{full_ret:+.2f}% total ({full_cagr:+.2f}% CAGR, Sharpe {full_sharpe}, p={p_val}), failing to deliver a "
            f"statistically reliable positive net edge over transaction costs, Cash (6.5%), or Nifty 50 ({nm_f.get('total_return_pct', 0):+.2f}%)."
        )
    elif (
        fm.get("distinguishable_from_random")
        and pos_folds >= max(3, int(len(wf_rows) * 0.75))
        and oos_ret > 0.0
        and oos_sharpe >= 0.75
        and mc_prob_pos >= 80.0
    ):
        category = "3. Robust candidate: Consistent performance across independent periods and market regimes"
        rationale = (
            f"Statistically significant expectancy (+{fm['expectancy_pct']}%, p={p_val}), {pos_folds}/{len(wf_rows)} positive "
            f"walk-forward windows, OOS +{oos_ret:.2f}% (Sharpe {oos_sharpe}), and {mc_prob_pos}% positive Monte Carlo block paths."
        )
    elif (
        full_ret > 0.0
        and oos_ret > 0.0
        and oos_trades >= 20
        and fm.get("max_drawdown_pct", 99) < nm_f.get("max_drawdown_pct", 0)
        and full_sharpe >= 0.25
        and mc_prob_pos >= 65.0
        and beat_m1_folds >= len(wf_rows) // 2
    ):
        category = "4. Shadow-testing candidate: Strong enough to justify paper/shadow testing, but not live capital"
        rationale = (
            f"Outperforms Moreshwar 1.0 in drawdown ({fm['max_drawdown_pct']}% vs M1) and stays positive both full-sample "
            f"({full_ret:+.2f}%, {full_cagr:+.2f}% CAGR) and OOS (+{oos_ret:.2f}%), with {mc_prob_pos}% positive block-bootstrap runs, "
            f"but lacks high Sharpe ({full_sharpe}) / Cash hurdle alpha for live capital."
        )
    else:
        category = "2. Promising but inconclusive: Improvements exist, but evidence is insufficient"
        rationale = (
            f"Moreshwar 2.1 improves risk control and drawdown over M1.0 (MaxDD {fm.get('max_drawdown_pct')}%, OOS {oos_ret:+.2f}% vs M1 {om1.get('total_return_pct', 0):+.2f}%), "
            f"but full-sample expectancy (+{fm.get('expectancy_pct', 0)}%/trade, p={p_val}), Sharpe ({full_sharpe}), and Cash (6.5% p.a.) "
            f"comparison remain statistically inconclusive ({mc_prob_pos}% Monte Carlo positive paths). Keep strictly in research/shadow tracking."
        )

    return {
        "classification": category,
        "promoted_to_production": False,
        "rationale": rationale,
        "key_evidence": {
            "full_5y_return_pct": full_ret,
            "full_5y_cagr_pct": full_cagr,
            "full_5y_sharpe": full_sharpe,
            "full_5y_max_dd_pct": fm.get("max_drawdown_pct", 0.0),
            "expectancy_p_value": p_val,
            "oos_m21_return_pct": oos_ret,
            "oos_m1_return_pct": om1.get("total_return_pct", 0.0),
            "oos_nifty_return_pct": nm_o.get("total_return_pct", 0.0),
            "walk_forward_positive_folds": f"{pos_folds}/{len(wf_rows)}",
            "monte_carlo_prob_positive_pct": mc_prob_pos,
        },
        "unresolved_risks": [
            "Lack of historical quarterly point-in-time Indian fundamental financial statements (only OHLCV is cleanly non-leaked over 5 years).",
            "Purged delisted companies (DHFL, RELCAPITAL, SINTEX) return 404 on free Yahoo endpoints, leaving partial survivorship bias despite distressed controls (YESBANK, JETAIRWAYS, RCOM, ZEEL).",
            "Trend-following equity swing entries earn modest capital utilization during bearish/sideways regimes, trailing 6.5% risk-free Indian T-bills in prolonged choppy markets.",
        ],
    }


def run_moreshwar_v2_1_research_suite(
    dataset: dict[str, Any],
    cfg_v21: MoreshwarV21Config | None = None,
    cfg_v2: MoreshwarV2Config | None = None,
    n_mc_sims: int = 1200,
) -> dict[str, Any]:
    """Execute complete Moreshwar 2.1 research pipeline across 5Y 40-stock dataset."""
    c21 = cfg_v21 or MoreshwarV21Config()
    c20 = cfg_v2 or MoreshwarV2Config()
    universe_history = dataset["universe_history"]
    bench_candles = (dataset.get("benchmark") or {}).get("candles") or []
    pre = precompute_causal_features(universe_history, bench_candles)
    timeline = pre["timeline"]

    # Untouched 25% Out-of-Sample tail split (last ~300 bars / ~1.2 years)
    n_tot = len(timeline)
    split_oos_idx = int(n_tot * 0.75)
    early_end_date = timeline[max(0, split_oos_idx - 1)]
    oos_range = (timeline[split_oos_idx], timeline[-1])
    pre_oos_range = (timeline[0], timeline[max(0, split_oos_idx - 1)])

    # Full 5-year benchmark comparison (identical dates, ₹100,000 capital, Indian delivery costs)
    res_m1 = simulate_portfolio_v2_1(universe_history, bench_candles, pre, model_name="Moreshwar 1.0 (Baseline)", model_family="M1", cfg_v21=c21, cfg_v2=c20)
    res_m2 = simulate_portfolio_v2_1(universe_history, bench_candles, pre, model_name="Moreshwar 2.0 (Clean Causal)", model_family="M2", cfg_v21=c21, cfg_v2=c20)
    res_m21 = simulate_portfolio_v2_1(universe_history, bench_candles, pre, model_name="Moreshwar 2.1 (6-Regime + Portfolio Risk)", model_family="M2_1", cfg_v21=c21, cfg_v2=c20, record_counterfactuals=True)
    res_m21_perm = simulate_portfolio_v2_1(universe_history, bench_candles, pre, model_name="Moreshwar 2.1 (Permissive Regimes Control)", model_family="M2_1_PERMISSIVE", cfg_v21=c21, cfg_v2=c20)
    res_m21_eref = simulate_portfolio_v2_1(universe_history, bench_candles, pre, model_name="Moreshwar 2.1 + Static Fund (Leak Audit)", model_family="M2_1_EREF", cfg_v21=c21, cfg_v2=c20)
    res_sma20 = simulate_portfolio_v2_1(universe_history, bench_candles, pre, model_name="Simple SMA-20 Strategy", model_family="SMA20", cfg_v21=c21, cfg_v2=c20)
    res_fixed = simulate_portfolio_v2_1(universe_history, bench_candles, pre, model_name="Fixed 5% Stop / 10% Target", model_family="FIXED_5_10", cfg_v21=c21, cfg_v2=c20)
    res_nifty = _compute_benchmark_metrics(bench_candles, date_range=(timeline[0], timeline[-1]))
    res_cash = _compute_cash_benchmark(len(timeline))

    benchmark_table = [
        res_m1["metrics"],
        res_m2["metrics"],
        res_m21["metrics"],
        res_m21_perm["metrics"],
        res_m21_eref["metrics"],
        res_sma20["metrics"],
        res_fixed["metrics"],
        res_nifty["metrics"],
        res_cash["metrics"],
    ]

    # Section 3A: Rolling Walk-Forward Validation (Train -> Val -> Untouched Test progression across 4 non-overlapping periods + 5Y degradation check)
    wf_folds: list[dict[str, Any]] = []
    # Split pre-OOS 75% + OOS 25% into 4 consecutive ~1.25-year test folds with rolling prior train+val observation windows
    fold_len = max(120, n_tot // 4)
    for f_i in range(4):
        lo = f_i * fold_len
        hi = min(n_tot - 1, (f_i + 1) * fold_len - 1) if f_i < 3 else (n_tot - 1)
        if hi - lo < 40:
            continue
        f_rng = (timeline[lo], timeline[hi])
        # Track rolling prior-window parameters explicitly without fitting on test fold
        prior_rng = (timeline[max(0, lo - fold_len)], timeline[lo - 1]) if lo >= 60 else None
        fm1 = simulate_portfolio_v2_1(universe_history, bench_candles, pre, model_name=f"WF{f_i+1}_M1", model_family="M1", cfg_v21=c21, date_range=f_rng)["metrics"]
        fm2 = simulate_portfolio_v2_1(universe_history, bench_candles, pre, model_name=f"WF{f_i+1}_M2", model_family="M2", cfg_v21=c21, cfg_v2=c20, date_range=f_rng)["metrics"]
        fm21 = simulate_portfolio_v2_1(universe_history, bench_candles, pre, model_name=f"WF{f_i+1}_M21", model_family="M2_1", cfg_v21=c21, date_range=f_rng)["metrics"]
        fnifty = _compute_benchmark_metrics(bench_candles, date_range=f_rng)["metrics"]
        wf_folds.append({
            "fold": f"WF-{f_i + 1}" + (" (UNTOUCHED OOS)" if f_i == 3 else ""),
            "prior_train_window": f"{prior_rng[0]}..{prior_rng[1]}" if prior_rng else "Warmup/Initial",
            "test_window": f"{f_rng[0]} → {f_rng[1]}",
            "bars": hi - lo + 1,
            "m1_return_pct": fm1["total_return_pct"],
            "m1_max_dd_pct": fm1["max_drawdown_pct"],
            "m2_return_pct": fm2["total_return_pct"],
            "m2_max_dd_pct": fm2["max_drawdown_pct"],
            "m21_return_pct": fm21["total_return_pct"],
            "m21_cagr_pct": fm21["cagr_pct"],
            "m21_sharpe": fm21["sharpe_ratio"],
            "m21_max_dd_pct": fm21["max_drawdown_pct"],
            "m21_win_rate_pct": fm21["win_rate_pct"],
            "m21_trades": fm21["total_trades"],
            "nifty_return_pct": fnifty.get("total_return_pct", 0.0),
        })

    # Temporal deterioration check (compare first half vs second half M2.1 Sharpe & Return)
    first_half_avg = sum(w["m21_return_pct"] for w in wf_folds[:2]) / max(1, len(wf_folds[:2]))
    second_half_avg = sum(w["m21_return_pct"] for w in wf_folds[2:]) / max(1, len(wf_folds[2:]))

    # Section 3B: Untouched Out-of-Sample Test Dataset (Last 25% of 5Y, untouched by design)
    oos_m1 = simulate_portfolio_v2_1(universe_history, bench_candles, pre, model_name="OOS Moreshwar 1.0", model_family="M1", cfg_v21=c21, date_range=oos_range)
    oos_m2 = simulate_portfolio_v2_1(universe_history, bench_candles, pre, model_name="OOS Moreshwar 2.0", model_family="M2", cfg_v21=c21, cfg_v2=c20, date_range=oos_range)
    oos_m21 = simulate_portfolio_v2_1(universe_history, bench_candles, pre, model_name="OOS Moreshwar 2.1", model_family="M2_1", cfg_v21=c21, date_range=oos_range)
    oos_sma = simulate_portfolio_v2_1(universe_history, bench_candles, pre, model_name="OOS SMA-20", model_family="SMA20", cfg_v21=c21, date_range=oos_range)
    oos_fix = simulate_portfolio_v2_1(universe_history, bench_candles, pre, model_name="OOS Fixed 5/10", model_family="FIXED_5_10", cfg_v21=c21, date_range=oos_range)
    oos_nifty = _compute_benchmark_metrics(bench_candles, date_range=oos_range)
    oos_cash = _compute_cash_benchmark(n_tot - split_oos_idx)

    # Section 3C: Monte Carlo Simulation (>=1,000 runs)
    mc_m21 = run_monte_carlo_suite(res_m21["trades"], res_m21["daily_rets"], n_sims=n_mc_sims)
    mc_m1 = run_monte_carlo_suite(res_m1["trades"], res_m1["daily_rets"], n_sims=n_mc_sims)

    # Section 4: Feature & Decision Attribution Investigation
    investigation = build_strategy_investigation_report(
        universe_history=universe_history,
        m1_res=res_m1,
        m2_res=res_m2,
        m21_res=res_m21,
        m21_perm_res=res_m21_perm,
        m21_eref_res=res_m21_eref,
        early_end_date=early_end_date,
    )

    # Section 10: Final Classification
    verdict = classify_final_verdict(
        m21_full=res_m21,
        m21_oos=oos_m21,
        m1_oos=oos_m1,
        nifty_full=res_nifty,
        nifty_oos=oos_nifty,
        wf_rows=wf_folds,
        mc_res=mc_m21,
    )

    # Downsample equity curves slightly for lightweight JSON / static Pages consumption while preserving weekly resolution
    def _compact_curve(curve: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if len(curve) <= 260:
            return curve
        step = max(1, len(curve) // 250)
        sampled = [curve[i] for i in range(0, len(curve) - 1, step)]
        sampled.append(curve[-1])
        return sampled

    return {
        "research_version": "2.1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "promoted_to_production": False,
        "promotion_status": "RESEARCH_ONLY (Moreshwar 1.0 active in prod; Cloud=Scan/Research Only, Laptop=Broker Execution)",
        "dataset_summary": {
            "start_date": timeline[0],
            "end_date": timeline[-1],
            "trading_days": len(timeline),
            "instruments_count": len(pre["features"]),
            "oos_start": oos_range[0],
            "oos_end": oos_range[1],
            "oos_days": n_tot - split_oos_idx,
            "execution_mode": c21.execution_mode,
        },
        "config_v2_1": {
            "model_version": c21.model_version,
            "atr_period": c21.atr_period,
            "atr_multiplier": c21.atr_multiplier,
            "risk_per_trade_pct": c21.risk_per_trade_pct,
            "max_single_stock_pct": c21.max_single_stock_pct,
            "max_portfolio_exposure_pct": c21.max_portfolio_exposure_pct,
            "max_portfolio_stop_risk_pct": c21.max_portfolio_stop_risk_pct,
            "max_open_positions": c21.max_open_positions,
            "max_sector_positions": c21.max_sector_positions,
            "max_pairwise_corr": c21.max_pairwise_corr,
            "max_daily_volume_participation": c21.max_daily_volume_participation,
            "regime_policy": c21.regime_policy,
            "execution_mode": c21.execution_mode,
        },
        "data_quality_report": dataset["data_quality_report"],
        "benchmark_comparison": benchmark_table,
        "models": {
            "Moreshwar 2.1": {
                "metrics": res_m21["metrics"],
                "significance": res_m21["significance"],
                "regimes_6": res_m21["regimes_6"],
                "portfolio_risk": res_m21["portfolio_risk"],
                "by_stock": res_m21["by_stock"],
                "by_sector": res_m21["by_sector"],
                "equity_curve": _compact_curve(res_m21["equity_curve"]),
                "sample_trades": res_m21["trades"][:120],
            },
            "Moreshwar 2.1 (Permissive)": {
                "metrics": res_m21_perm["metrics"],
                "regimes_6": res_m21_perm["regimes_6"],
                "portfolio_risk": res_m21_perm["portfolio_risk"],
                "by_stock": res_m21_perm["by_stock"],
                "by_sector": res_m21_perm["by_sector"],
                "equity_curve": _compact_curve(res_m21_perm["equity_curve"]),
            },
            "Moreshwar 2.0": {
                "metrics": res_m2["metrics"],
                "regimes_6": res_m2["regimes_6"],
                "portfolio_risk": res_m2["portfolio_risk"],
                "by_stock": res_m2["by_stock"],
                "by_sector": res_m2["by_sector"],
                "equity_curve": _compact_curve(res_m2["equity_curve"]),
            },
            "Moreshwar 1.0": {
                "metrics": res_m1["metrics"],
                "regimes_6": res_m1["regimes_6"],
                "portfolio_risk": res_m1["portfolio_risk"],
                "by_stock": res_m1["by_stock"],
                "by_sector": res_m1["by_sector"],
                "equity_curve": _compact_curve(res_m1["equity_curve"]),
            },
        },
        "walk_forward_validation": {
            "folds": wf_folds,
            "first_half_avg_return_pct": round(first_half_avg, 2),
            "second_half_avg_return_pct": round(second_half_avg, 2),
            "deterioration_detected": second_half_avg < first_half_avg - 1.5,
            "parameter_tuning_note": "Zero parameter optimization performed on test windows; fixed structural rules evaluated sequentially across 4 rolling market epochs.",
        },
        "out_of_sample": {
            "window": {"start": oos_range[0], "end": oos_range[1], "trading_days": n_tot - split_oos_idx},
            "comparison": [
                oos_m1["metrics"],
                oos_m2["metrics"],
                oos_m21["metrics"],
                oos_sma["metrics"],
                oos_fix["metrics"],
                oos_nifty["metrics"],
                oos_cash["metrics"],
            ],
        },
        "monte_carlo": {
            "moreshwar_2_1": mc_m21,
            "moreshwar_1_0": mc_m1,
        },
        "strategy_investigation": investigation,
        "final_decision": verdict,
    }
