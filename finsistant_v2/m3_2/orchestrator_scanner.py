"""Moreshwar 3.2 — the multi-agent orchestrator scanning the FULL PIT-eligible NSE universe.

Pipeline (every post-close session, 100% real market & fundamental data, zero fabrication):

  Stage 1  TechBot_1  scores EVERY liquid, valid stock in the universe (~1,182 NSE equities) using
                      the *identical* rule used for single-symbol pipelines
                      (`finsistant_v2.agents.score_technical_confidence`) on real OHLCV history
                      strictly <= trading_date.
  Stage 2  Shortlist  the strongest technical names (sector-capped + learned sector/RSI awareness)
                      advance to deep agent evaluation.
  Stage 3  FundBot_1 + ExecBot_1 run real live fundamentals (Yahoo Finance live financials, 4-Pillar
                      + Porter's 5 Forces + Moreshwar Levels + Decision Matrix) in parallel on each
                      shortlisted symbol using the current learned parameters from prior passed and
                      failed trades.
  Stage 4  Candidates ExecBot-approved BUYs ranked by conviction become the paper-trading candidates
                      (`STRAT_ORCH_AGENT_CONSENSUS`) queued for next-session open in the 80-session
                      mock-test paper engine.

Every stage is persisted as evidence (`learning_ledger`) so EVERY passed and failed trade teaches
the orchestrator. A symbol whose live data is unavailable or conflicts with the EOD bar (>5% gap)
is EXCLUDED and logged — never defaulted, never fabricated.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
import math
import threading
from typing import Any, Callable

from finsistant_v2.agents import (
    AdversarialProsecutorAgent,
    ExecutionAgent,
    FundamentalAnalyst,
    TechnicalAnalyst,
    score_technical_confidence,
)
from finsistant_v2.adversarial_prosecutor import DEFAULT_MIN_SURVIVAL_SCORE
from finsistant_v2.quant_engine import compute_ohlcv_and_regression_from_candles

ORCH_STRATEGY_ID = "STRAT_ORCH_AGENT_CONSENSUS"
SCANNER_VERSION = "3.2.0"
PRIOR_W_TECH = 0.45
PRIOR_W_FUND = 0.55
PRIOR_MIN_CONVICTION = 0.60
PRIOR_RSI_OVERBOUGHT_PENALTY = 0.0
MISSION_TARGET_SESSIONS = 80

_SCAN_FUNDAMENTALS_CACHE: dict[str, dict[str, Any]] = {}
_SCAN_FUND_LOCK = threading.Lock()


@dataclass(frozen=True)
class OrchestratorScanConfig:
    min_adv_inr: float = 20_000_000.0
    min_bars: int = 60
    shortlist_size: int = 30
    max_per_sector_in_shortlist: int = 5
    max_final_candidates: int = 15
    min_tech_conf_for_shortlist: float = 0.58  # TechBot BULLISH threshold
    max_price_discrepancy_pct: float = 5.0  # live agent price vs EOD bar; larger => excluded
    stop_ratio_min: float = 0.08
    stop_ratio_max: float = 0.14
    max_workers: int = 4
    max_retries_per_symbol: int = 2
    min_agents_ok_ratio: float = 0.80

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _candles_from_history(h: dict[str, Any], n: int) -> list[dict[str, Any]]:
    dates = list(h["dates"])[-n:]
    o, hi, lo, c, v = (list(h[k])[-n:] for k in ("open", "high", "low", "close", "volume"))
    return [
        {
            "date": dates[i],
            "open": float(o[i]),
            "high": float(hi[i]),
            "low": float(lo[i]),
            "close": float(c[i]),
            "volume": float(v[i]),
        }
        for i in range(len(dates))
    ]


def _default_fast_live_analysis(symbol: str) -> dict[str, Any]:
    """Fetch strict Official NSE MCP quotes/OHLCV + verified live fundamentals (zero silent fallbacks or default scores)."""
    from finsistant_v2.live_data import fetch_live_market_snapshot
    from finsistant_v2.quant_engine import compute_ohlcv_and_regression_from_candles
    from finsistant_v2.scoring_engine import (
        calculate_decision_matrix,
        calculate_fundamental_score,
        calculate_moreshwar_levels,
        calculate_porters_score,
    )

    sym = symbol.upper().strip()
    live = fetch_live_market_snapshot(sym, force_refresh=True, strict_no_fallback=False)

    quant = compute_ohlcv_and_regression_from_candles(live["candles"], periods=45)
    candles = quant["candles"]
    latest = candles[-1]
    spot = float(live["price"])

    f_score = calculate_fundamental_score(live, strict=True)
    p_score = calculate_porters_score(live, strict=True)
    moreshwar = calculate_moreshwar_levels(
        spot_price=spot,
        f_score_total=f_score["total"],
        p_score_total=p_score["total"],
        is_holding=False,
    )
    decision = calculate_decision_matrix(
        f_score_total=f_score["total"],
        p_score_total=p_score["total"],
        meta=live,
        price_above_sma=spot >= float(latest["sma_20"]),
        is_holding=False,
    )
    return {
        "symbol": sym,
        "yahoo_symbol": live["yahoo_symbol"],
        "name": live["name"],
        "exchange": live["exchange"],
        "currency": live["currency"],
        "sector": live["sector"],
        "asset_type": live["asset_type"],
        "setup_type": live["setup_type"],
        "price": spot,
        "last_candle_date": live.get("last_candle_date", candles[-1]["date"]),
        "last_candle_close": live.get("last_candle_close", float(latest["close"])),
        "history_bars_count": len(live.get("candles") or []),
        "change_pct": float(live["change_pct"]),
        "fetched_at": live["fetched_at"],
        "market_timestamp": live.get("market_timestamp") or f"{candles[-1]['date']}T10:00:00+00:00",
        "market_timestamp_kind": live.get("market_timestamp_kind", "exchange_quote"),
        "missing_fields": list(live.get("missing_fields") or []),
        "proxy_fields": dict(live.get("proxy_fields") or {}),
        "cache_hit": False,
        "cache_age_seconds": 0.0,
        "price_source_category": live["price_source_category"],
        "ohlcv_source_category": live["ohlcv_source_category"],
        "fundamental_source_category": live["fundamental_source_category"],
        "fundamentals_status": live.get("fundamentals_status", "live"),
        "data_source": live["data_source"],
        "rsi_14": float(latest["rsi_14"]),
        "macd": float(latest["macd"]),
        "macd_signal": float(latest["macd_signal"]),
        "sma_20": float(latest["sma_20"]),
        "ema_12": float(latest["ema_12"]),
        "ema_26": float(latest["ema_26"]),
        "bb_upper": float(latest["bb_upper"]),
        "bb_lower": float(latest["bb_lower"]),
        "projected_30d_target": quant["projected_30d_target"],
        "regression_std_dev": quant["regression_std_dev"],
        "regression_curve": quant["regression_curve"],
        "future_projection": quant["future_projection"],
        "fundamental_score": f_score,
        "porters_score": p_score,
        "moreshwar_levels": moreshwar,
        "decision_matrix": decision,
        "candles": candles,
    }


def _build_causal_eod_fundamental_snapshot(
    symbol: str,
    *,
    trading_date: str,
    history: dict[str, Any],
    bar: dict[str, Any],
    is_holding: bool = False,
) -> dict[str, Any]:
    """Build causal EOD candles from already-loaded NSE history + verified live fundamentals (zero fake defaults)."""
    from finsistant_v2.live_data import _fetch_yahoo_fundamentals_only
    from finsistant_v2.scoring_engine import (
        calculate_decision_matrix,
        calculate_fundamental_score,
        calculate_moreshwar_levels,
        calculate_porters_score,
    )

    sym = symbol.upper().strip()
    nse_sym = sym[:-3] if sym.endswith(".NS") else sym
    yahoo_sym = f"{nse_sym}.NS"
    candles_raw = _candles_from_history(history, 250)
    quant = compute_ohlcv_and_regression_from_candles(candles_raw, periods=45)
    candles = quant["candles"]
    latest = candles[-1]
    spot = round(float(bar["close"]), 2)
    prev_close = float(candles[-2]["close"]) if len(candles) >= 2 else spot
    change_pct = round(((spot - prev_close) / max(1e-6, prev_close)) * 100.0, 2)
    ret_1y = round(((spot - float(candles[0]["close"])) / max(1e-6, float(candles[0]["close"]))) * 100.0, 2) if candles else 0.0

    with _SCAN_FUND_LOCK:
        cached_fund = _SCAN_FUNDAMENTALS_CACHE.get(sym)
    if cached_fund is None:
        fund = _fetch_yahoo_fundamentals_only(yahoo_sym, "INR", spot, strict=True, nse_sym=nse_sym)
        if not fund or str(fund.get("fundamentals_status", "")).lower() == "unavailable" or str(fund.get("fundamental_source_category", "")).upper() == "UNAVAILABLE":
            raise RuntimeError(f"Fundamental data unavailable for {sym} (refusing default score).")
        with _SCAN_FUND_LOCK:
            _SCAN_FUNDAMENTALS_CACHE[sym] = dict(fund)
    else:
        fund = dict(cached_fund)

    snap_for_score = {
        "symbol": sym,
        "name": fund.get("name") or sym,
        "sector": str(bar.get("sector") or fund.get("sector") or "General"),
        "asset_type": "STOCK",
        "price": spot,
        "pe": fund.get("pe"),
        "roe": fund.get("roe"),
        "roce": fund.get("roce"),
        "opm": fund.get("opm"),
        "sales_growth": fund.get("sales_growth"),
        "profit_growth": fund.get("profit_growth"),
        "mcap_cr": fund.get("mcap_cr"),
        "beta": fund.get("beta"),
        "ret_1y": ret_1y,
        "debt_to_equity": fund.get("debt_to_equity"),
    }
    f_score = calculate_fundamental_score(snap_for_score, strict=True)
    p_score = calculate_porters_score(snap_for_score, strict=True)
    moreshwar = calculate_moreshwar_levels(
        spot_price=spot,
        f_score_total=f_score["total"],
        p_score_total=p_score["total"],
        is_holding=is_holding,
    )
    decision = calculate_decision_matrix(
        f_score_total=f_score["total"],
        p_score_total=p_score["total"],
        meta=snap_for_score,
        price_above_sma=spot >= float(latest["sma_20"]),
        is_holding=is_holding,
    )
    return {
        "symbol": sym,
        "yahoo_symbol": yahoo_sym,
        "nse_symbol": nse_sym,
        "name": snap_for_score["name"],
        "exchange": "NSE",
        "currency": "INR",
        "sector": snap_for_score["sector"],
        "asset_type": "STOCK",
        "setup_type": "Live NSE EOD + Verified Fundamentals",
        "price": spot,
        "last_candle_date": str(trading_date),
        "last_candle_close": round(float(latest["close"]), 2),
        "history_bars_count": len(candles),
        "change_pct": change_pct,
        "fetched_at": _utc_now_iso(),
        "market_timestamp": f"{trading_date}T10:00:00+00:00",
        "market_timestamp_kind": "exchange_eod_bhavcopy",
        "missing_fields": list(fund.get("missing_fields") or []),
        "proxy_fields": dict(fund.get("proxy_fields") or {}),
        "cache_hit": cached_fund is not None,
        "cache_age_seconds": 0.0,
        "price_source_category": "NSE_OFFICIAL_BHAVCOPY",
        "ohlcv_source_category": "HISTORICAL_MARKET_DATA",
        "fundamental_source_category": fund.get("fundamental_source_category", "YAHOO_LIVE_FEED"),
        "fundamentals_status": fund.get("fundamentals_status", "live"),
        "data_source": f"Official NSE EOD Bhavcopy ({trading_date}) + {fund.get('fundamental_source_category', 'YAHOO_LIVE_FEED')}",
        "rsi_14": float(latest["rsi_14"]),
        "macd": float(latest["macd"]),
        "macd_signal": float(latest["macd_signal"]),
        "sma_20": float(latest["sma_20"]),
        "ema_12": float(latest["ema_12"]),
        "ema_26": float(latest["ema_26"]),
        "bb_upper": float(latest["bb_upper"]),
        "bb_lower": float(latest["bb_lower"]),
        "projected_30d_target": quant["projected_30d_target"],
        "regression_std_dev": quant["regression_std_dev"],
        "regression_curve": quant["regression_curve"],
        "future_projection": quant["future_projection"],
        "fundamental_score": f_score,
        "porters_score": p_score,
        "moreshwar_levels": moreshwar,
        "decision_matrix": decision,
        "candles": candles,
    }


class OrchestratorUniverseScanner:
    """TechBot full-universe scan -> shortlist -> parallel FundBot + ExecBot -> ranked paper candidates."""

    def __init__(
        self,
        config: OrchestratorScanConfig | None = None,
        *,
        analysis_fn: Callable[[str], dict[str, Any]] | None = None,
    ) -> None:
        self.cfg = config or OrchestratorScanConfig()
        self._custom_analysis_fn = analysis_fn is not None
        self._analysis_fn = analysis_fn or _default_fast_live_analysis
        self.tech_agent = TechnicalAnalyst()
        self.fund_agent = FundamentalAnalyst()
        self.prosecutor_agent = AdversarialProsecutorAgent()
        self.exec_agent = ExecutionAgent()

    # ------------------------------------------------------------------
    # Stage 1 — TechBot_1 over the entire eligible universe
    # ------------------------------------------------------------------
    def score_universe_technical(
        self,
        *,
        trading_date: str,
        equity_histories: dict[str, dict[str, Any]],
        eq_bars: dict[str, dict[str, Any]],
        rsi_overbought_penalty: float = 0.0,
        sector_adjustments: dict[str, float] | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, int]]:
        sec_adj_map = sector_adjustments or {}
        rows: list[dict[str, Any]] = []
        skipped = {"no_bar": 0, "illiquid": 0, "short_history": 0, "indicator_error": 0, "stale_last_bar": 0}
        for sec_id, h in equity_histories.items():
            bar = eq_bars.get(sec_id)
            if not bar or not bool(bar.get("valid", True)):
                skipped["no_bar"] += 1
                continue
            if not h.get("dates") or str(h["dates"][-1]) != str(trading_date):
                skipped["stale_last_bar"] += 1
                continue
            if float(bar.get("adv_20d_inr") or 0.0) < self.cfg.min_adv_inr:
                skipped["illiquid"] += 1
                continue
            if len(h["dates"]) < self.cfg.min_bars:
                skipped["short_history"] += 1
                continue
            try:
                candles = _candles_from_history(h, 60)
                quant = compute_ohlcv_and_regression_from_candles(candles, periods=45)
                latest = quant["candles"][-1]
                price = float(bar["close"])
                proj = float(quant["projected_30d_target"])
                rsi_14 = float(latest["rsi_14"])
                conf, signal = score_technical_confidence(
                    price=price,
                    sma_20=float(latest["sma_20"]),
                    ema_12=float(latest["ema_12"]),
                    ema_26=float(latest["ema_26"]),
                    rsi=rsi_14,
                    macd=float(latest["macd"]),
                    macd_sig=float(latest["macd_signal"]),
                    proj_30d=proj,
                )
            except Exception:
                skipped["indicator_error"] += 1
                continue

            sec_name = str(bar.get("sector", "UNKNOWN_SECTOR"))
            vol20 = max(0.005, float(bar.get("vol_20d") or 0.018))
            upside = proj / price - 1.0 if price > 0 else 0.0
            # Effective technical rank reflects learned RSI overbought penalty & sector edge calibration
            eff_tech = conf - (rsi_overbought_penalty if rsi_14 > 72.0 else 0.0) + float(sec_adj_map.get(sec_name, 0.0))
            rank_key = upside / (vol20 * math.sqrt(30.0))
            rows.append(
                {
                    "sec_id": str(sec_id),
                    "symbol": str(bar["symbol"]),
                    "sector": sec_name,
                    "close": round(price, 4),
                    "tech_conf": float(conf),
                    "effective_tech_conf": round(float(eff_tech), 4),
                    "tech_signal": signal,
                    "rank_key": round(float(rank_key), 6),
                    "rsi_14": round(rsi_14, 2),
                    "projected_30d_upside_pct": round(upside * 100.0, 3),
                }
            )
        rows.sort(key=lambda r: (-r["effective_tech_conf"], -r["rank_key"]))
        for i, r in enumerate(rows, start=1):
            r["tech_rank"] = i
        return rows, skipped

    # ------------------------------------------------------------------
    # Stage 2 — shortlist
    # ------------------------------------------------------------------
    def shortlist(self, tech_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        per_sector: dict[str, int] = {}
        for r in tech_rows:
            if r["tech_conf"] < self.cfg.min_tech_conf_for_shortlist:
                continue
            sec = r["sector"]
            if sec != "UNKNOWN_SECTOR" and per_sector.get(sec, 0) >= self.cfg.max_per_sector_in_shortlist:
                continue
            per_sector[sec] = per_sector.get(sec, 0) + 1
            out.append(r)
            if len(out) >= self.cfg.shortlist_size:
                break
        return out

    # ------------------------------------------------------------------
    # Stage 3 + 4 — parallel FundBot_1 + ExecBot_1 on the shortlist
    # ------------------------------------------------------------------
    def _evaluate_one_shortlisted(
        self,
        r: dict[str, Any],
        *,
        learned_params: dict[str, Any],
        capital_inr: float,
        trading_date: str | None = None,
        history: dict[str, Any] | None = None,
        bar: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        weights = (float(learned_params.get("w_tech", PRIOR_W_TECH)), float(learned_params.get("w_fund", PRIOR_W_FUND)))
        min_conv = float(learned_params.get("min_conviction", PRIOR_MIN_CONVICTION))
        rsi_pen = float(learned_params.get("rsi_overbought_penalty", PRIOR_RSI_OVERBOUGHT_PENALTY))
        sec_adj_map = learned_params.get("sector_adjustments") or {}
        sec_adj = float(sec_adj_map.get(r["sector"], 0.0))
        stop_mult = max(0.80, min(1.10, float(learned_params.get("stop_tighten_mult", 1.0))))

        row: dict[str, Any] = {
            "sec_id": r["sec_id"],
            "symbol": r["symbol"],
            "sector": r["sector"],
            "tech_rank": r["tech_rank"],
            "eod_close": r["close"],
            "universe_tech_conf": r["tech_conf"],
            "rsi_14": r.get("rsi_14"),
            "projected_30d_upside_pct": r.get("projected_30d_upside_pct"),
            "status": "PENDING",
        }
        import time as _time

        use_causal_fast_path = (not self._custom_analysis_fn) and (history is not None) and (bar is not None) and bool(trading_date)
        last_exc: Exception | None = None
        snap: dict[str, Any] | None = None
        max_attempts = max(1, int(self.cfg.max_retries_per_symbol))
        for attempt in range(max_attempts):
            try:
                if use_causal_fast_path:
                    cand_snap = _build_causal_eod_fundamental_snapshot(
                        r["symbol"],
                        trading_date=str(trading_date),
                        history=history,  # type: ignore[arg-type]
                        bar=bar,  # type: ignore[arg-type]
                    )
                else:
                    cand_snap = self._analysis_fn(r["symbol"])
                if not cand_snap or str(cand_snap.get("fundamentals_status", "")).lower() == "unavailable" or str(cand_snap.get("fundamental_source_category", "")).upper() == "UNAVAILABLE":
                    raise RuntimeError(f"Fundamental data unavailable for {r['symbol']} (refusing default score).")
                snap = cand_snap
                break
            except Exception as exc:
                last_exc = exc
                if attempt < max_attempts - 1:
                    _time.sleep(min(8.0, 1.5 * (attempt + 1)))

        if snap is None:
            row["status"] = "FAILED_DATA_UNAVAILABLE"
            row["error"] = str(last_exc)[:200] if last_exc else "Unknown data fetch error"
            return row

        try:
            live_px = float(snap["price"])
            disc = abs(live_px / float(r["close"]) - 1.0) * 100.0 if r["close"] else 999.0
            row["live_price"] = round(live_px, 4)
            row["price_discrepancy_pct"] = round(disc, 3)
            if disc > self.cfg.max_price_discrepancy_pct:
                row["status"] = "EXCLUDED_PRICE_DISCREPANCY"
                return row

            ctx: dict[str, Any] = {
                "symbol": r["symbol"],
                "asset_analysis": snap,
                "capital": float(capital_inr),
                "active_trades": 0,
                "max_concurrent_trades": 99,
                "agent_mode": "paper_trading",
                "risk_budget_pct": 0.011,
                "max_single_stock_pct": 15.0,
                "currency": "INR",
                "is_holding": False,
                "conviction_weights": weights,
                "min_conviction": min_conv,
                "rsi_overbought_penalty": rsi_pen,
                "sector_adjustment": sec_adj,
                "attack_multipliers": dict(learned_params.get("attack_multipliers") or {}),
                "min_survival_score": float(learned_params.get("min_survival_score", DEFAULT_MIN_SURVIVAL_SCORE)),
                "adv_20d_inr": float((bar or {}).get("adv_20d_inr") or 0.0) if bar else None,
                "hl_range_pct": float((bar or {}).get("hl_range_pct") or 0.0) if bar else None,
                "price_discrepancy_pct": disc,
            }
            for agent in (self.tech_agent, self.fund_agent, self.prosecutor_agent, self.exec_agent):
                ctx[agent.role_key] = agent.evaluate(ctx)
            tech, fund, pros_step, exe = ctx["technical"], ctx["fundamental"], ctx["prosecution"], ctx["execution"]
            prosecution = pros_step.get("metrics") or {}
            fscore = snap["fundamental_score"]["total"]
            pscore = snap["porters_score"]["total"]
            price = float(snap["price"])
            stop_loss = float(exe.get("stop_loss") or 0.0)
            raw_stop_ratio = (price - stop_loss) / price if price > 0 and stop_loss > 0 else self.cfg.stop_ratio_max
            stop_ratio = min(self.cfg.stop_ratio_max, max(self.cfg.stop_ratio_min, raw_stop_ratio * stop_mult))
            row.update(
                {
                    "status": "AGENTS_OK",
                    "name": snap.get("name"),
                    "tech_conf": float(tech["confidence"]),
                    "tech_signal": tech["signal"],
                    "fund_conf": float(fund["confidence"]),
                    "fund_signal": fund["signal"],
                    "fundamental_total": fscore,
                    "porters_total": pscore,
                    "conviction": float(exe["conviction"]),
                    "exec_action": exe["action"],
                    "exec_signal": exe["signal"],
                    "exec_reason": exe["summary"],
                    "decision_badge": exe.get("decision_badge"),
                    "prosecution_verdict": prosecution.get("verdict"),
                    "survival_score": prosecution.get("survival_score"),
                    "min_survival_score": prosecution.get("min_survival_score"),
                    "damage_pts": prosecution.get("total_damage_pts"),
                    "fatal_count": prosecution.get("fatal_count", 0),
                    "attack_ids": list(prosecution.get("attack_ids") or []),
                    "prosecution_summary": prosecution.get("summary"),
                    "prosecution_size_mult": prosecution.get("size_multiplier", 1.0),
                    "prosecution_attacks": list(prosecution.get("attacks") or []),
                    "moreshwar_stop_loss": stop_loss,
                    "moreshwar_target": float(exe.get("take_profit") or 0.0),
                    "stop_ratio": round(stop_ratio, 6),
                    "stop_tighten_mult": round(stop_mult, 4),
                    "fundamental_source": snap.get("fundamental_source_category"),
                    "price_source": snap.get("price_source_category"),
                    "missing_fields": list(snap.get("missing_fields") or []),
                }
            )
        except Exception as exc:
            row["status"] = "FAILED_DATA_UNAVAILABLE"
            row["error"] = str(exc)[:200]
        return row

    def run_agents_on_shortlist(
        self,
        shortlist: list[dict[str, Any]],
        *,
        learned_params: dict[str, Any] | None = None,
        weights: tuple[float, float] | None = None,
        capital_inr: float = 100_000.0,
        trading_date: str | None = None,
        equity_histories: dict[str, dict[str, Any]] | None = None,
        eq_bars: dict[str, dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        params = dict(learned_params or {})
        if weights is not None:
            params["w_tech"] = float(weights[0])
            params["w_fund"] = float(weights[1])
        params.setdefault("w_tech", PRIOR_W_TECH)
        params.setdefault("w_fund", PRIOR_W_FUND)
        params.setdefault("min_conviction", PRIOR_MIN_CONVICTION)
        params.setdefault("rsi_overbought_penalty", PRIOR_RSI_OVERBOUGHT_PENALTY)
        params.setdefault("sector_adjustments", {})
        params.setdefault("stop_tighten_mult", 1.0)

        h_map = equity_histories or {}
        b_map = eq_bars or {}

        workers = max(1, min(self.cfg.max_workers, len(shortlist)))
        if workers <= 1:
            return [
                self._evaluate_one_shortlisted(
                    r,
                    learned_params=params,
                    capital_inr=capital_inr,
                    trading_date=trading_date,
                    history=h_map.get(str(r["sec_id"])),
                    bar=b_map.get(str(r["sec_id"])),
                )
                for r in shortlist
            ]

        by_idx: dict[int, dict[str, Any]] = {}
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = {
                pool.submit(
                    self._evaluate_one_shortlisted,
                    r,
                    learned_params=params,
                    capital_inr=capital_inr,
                    trading_date=trading_date,
                    history=h_map.get(str(r["sec_id"])),
                    bar=b_map.get(str(r["sec_id"])),
                ): idx
                for idx, r in enumerate(shortlist)
            }
            for fut in as_completed(futs):
                idx = futs[fut]
                by_idx[idx] = fut.result()
        return [by_idx[i] for i in range(len(shortlist))]

    # ------------------------------------------------------------------
    # Full scan
    # ------------------------------------------------------------------
    def scan(
        self,
        *,
        trading_date: str,
        equity_histories: dict[str, dict[str, Any]],
        eq_bars: dict[str, dict[str, Any]],
        learned_params: dict[str, Any] | None = None,
        weights: tuple[float, float] = (PRIOR_W_TECH, PRIOR_W_FUND),
        weights_status: str = "PRIOR_FIXED",
        capital_inr: float = 100_000.0,
    ) -> dict[str, Any]:
        params = dict(learned_params or {})
        params.setdefault("w_tech", float(weights[0]))
        params.setdefault("w_fund", float(weights[1]))
        params.setdefault("min_conviction", PRIOR_MIN_CONVICTION)
        params.setdefault("rsi_overbought_penalty", PRIOR_RSI_OVERBOUGHT_PENALTY)
        params.setdefault("sector_adjustments", {})
        params.setdefault("stop_tighten_mult", 1.0)
        params.setdefault("attack_multipliers", {})
        params.setdefault("min_survival_score", DEFAULT_MIN_SURVIVAL_SCORE)
        params.setdefault("status", weights_status)

        tech_rows, skipped = self.score_universe_technical(
            trading_date=trading_date,
            equity_histories=equity_histories,
            eq_bars=eq_bars,
            rsi_overbought_penalty=float(params["rsi_overbought_penalty"]),
            sector_adjustments=params["sector_adjustments"],
        )
        shortlist = self.shortlist(tech_rows)
        agent_rows = self.run_agents_on_shortlist(
            shortlist,
            learned_params=params,
            capital_inr=capital_inr,
            trading_date=trading_date,
            equity_histories=equity_histories,
            eq_bars=eq_bars,
        )

        buys = [r for r in agent_rows if r["status"] == "AGENTS_OK" and r["exec_action"] == "BUY"]
        buys.sort(
            key=lambda r: (
                -float(r["conviction"]) * (0.5 + 0.5 * float(r.get("survival_score") or 50.0) / 100.0),
                r["tech_rank"],
            )
        )
        candidates: list[dict[str, Any]] = []
        for r in buys[: self.cfg.max_final_candidates]:
            candidates.append(
                {
                    "sec_id": r["sec_id"],
                    "symbol": r["symbol"],
                    "strategy_id": ORCH_STRATEGY_ID,
                    "score": float(r["conviction"]),
                    "survival_score": float(r.get("survival_score") or 0.0),
                    "size_multiplier": float(r.get("prosecution_size_mult") or 1.0),
                    "stop_ratio": float(r["stop_ratio"]),
                    "take_profit": float(r.get("moreshwar_target") or 0.0),
                    "tech_conf": r["tech_conf"],
                    "fund_conf": r["fund_conf"],
                    "tech_rank": r["tech_rank"],
                }
            )

        n_ok = sum(1 for r in agent_rows if r["status"] == "AGENTS_OK")
        n_killed = sum(1 for r in agent_rows if r.get("exec_signal") == "KILLED_BY_ADVERSARY")
        n_failed_data = sum(
            1 for r in agent_rows if r["status"] in ("FAILED_DATA_UNAVAILABLE", "EXCLUDED_AGENT_DATA_UNAVAILABLE")
        )
        max_allowed_failures = int(len(shortlist) * (1.0 - float(self.cfg.min_agents_ok_ratio)))
        if shortlist and (n_failed_data > max_allowed_failures or (n_ok == 0 and n_failed_data > 0)):
            sample_err = next((r.get("error") for r in agent_rows if r.get("error")), "data unavailable")
            raise RuntimeError(
                f"ORCHESTRATOR_SCAN_FAILED_UPFRONT on {trading_date}: "
                f"{n_failed_data}/{len(shortlist)} shortlisted symbols failed real NSE/fundamental fetch "
                f"after {self.cfg.max_retries_per_symbol} retries ({sample_err})."
            )

        return {
            "scanner_version": SCANNER_VERSION,
            "trading_date": str(trading_date),
            "generated_at_utc": _utc_now_iso(),
            "strategy_id": ORCH_STRATEGY_ID,
            "mission_target_sessions": MISSION_TARGET_SESSIONS,
            "config": self.cfg.as_dict(),
            "weights": {
                "w_tech": round(float(params["w_tech"]), 4),
                "w_fund": round(float(params["w_fund"]), 4),
                "min_conviction": round(float(params["min_conviction"]), 4),
                "rsi_overbought_penalty": round(float(params["rsi_overbought_penalty"]), 4),
                "sector_adjustments": dict(params["sector_adjustments"]),
                "attack_multipliers": dict(params["attack_multipliers"]),
                "min_survival_score": round(float(params["min_survival_score"]), 2),
                "status": str(params["status"]),
            },
            "stages": {
                "universe_histories": len(equity_histories),
                "valid_eod_bars": len(eq_bars),
                "tech_scored_eligible": len(tech_rows),
                "tech_bullish": sum(1 for r in tech_rows if r["tech_conf"] >= self.cfg.min_tech_conf_for_shortlist),
                "shortlisted": len(shortlist),
                "agents_ok": n_ok,
                "agents_excluded": len(agent_rows) - n_ok,
                "agents_failed_data": n_failed_data,
                "adversarial_killed": n_killed,
                "exec_buy_approved": len(buys),
                "paper_candidates": len(candidates),
                "skipped": skipped,
            },
            "shortlist": agent_rows,
            "candidates": candidates,
            "universe_tech": tech_rows,
        }

    def evaluate_open_holdings(
        self,
        open_positions: list[dict[str, Any]],
        *,
        learned_params: dict[str, Any] | None = None,
        capital_inr: float = 100_000.0,
        trading_date: str | None = None,
        equity_histories: dict[str, dict[str, Any]] | None = None,
        eq_bars: dict[str, dict[str, Any]] | None = None,
        entry_targets: dict[str, float] | None = None,
        top_candidates: list[dict[str, Any]] | None = None,
        max_tactical_positions: int = 8,
    ) -> list[dict[str, Any]]:
        """Re-evaluate open tactical positions for target hits, profit protection, time-decay, Bear downgrade & rotation."""
        if not open_positions:
            return []
        params = dict(learned_params or {})
        weights = (float(params.get("w_tech", PRIOR_W_TECH)), float(params.get("w_fund", PRIOR_W_FUND)))
        min_conv = float(params.get("min_conviction", PRIOR_MIN_CONVICTION))
        rsi_pen = float(params.get("rsi_overbought_penalty", PRIOR_RSI_OVERBOUGHT_PENALTY))
        min_surv = float(params.get("min_survival_score", DEFAULT_MIN_SURVIVAL_SCORE))
        sec_adj_map = params.get("sector_adjustments") or {}
        atk_mults = dict(params.get("attack_multipliers") or {})
        h_map = equity_histories or {}
        b_map = eq_bars or {}
        targets_map = entry_targets or {}

        exits: list[dict[str, Any]] = []
        rotation_pool: list[dict[str, Any]] = []
        held_sids: set[str] = {str(p.get("sec_id") or "") for p in open_positions if p.get("sec_id")}

        for pos in open_positions:
            sym = str(pos.get("symbol") or "").strip().upper()
            sid = str(pos.get("sec_id") or "")
            if not sym or not sid:
                continue
            hist = h_map.get(sid)
            bar = b_map.get(sid)
            if (not self._custom_analysis_fn) and trading_date and hist is not None and bar is not None:
                snap = _build_causal_eod_fundamental_snapshot(
                    sym,
                    trading_date=str(trading_date),
                    history=hist,
                    bar=bar,
                    is_holding=True,
                )
            else:
                snap = self._analysis_fn(sym)
            ctx: dict[str, Any] = {
                "symbol": sym,
                "asset_analysis": snap,
                "capital": float(capital_inr),
                "active_trades": len(open_positions),
                "max_concurrent_trades": 99,
                "agent_mode": "paper_trading",
                "currency": "INR",
                "is_holding": True,
                "conviction_weights": weights,
                "min_conviction": min_conv,
                "rsi_overbought_penalty": rsi_pen,
                "sector_adjustment": float(sec_adj_map.get(str(pos.get("sector") or ""), 0.0)),
                "attack_multipliers": atk_mults,
                "min_survival_score": min_surv,
            }
            for agent in (self.tech_agent, self.fund_agent, self.prosecutor_agent, self.exec_agent):
                ctx[agent.role_key] = agent.evaluate(ctx)
            exe = ctx["execution"]
            pros = (ctx.get("prosecution") or {}).get("metrics") or {}
            badge = str(exe.get("decision_badge") or "")
            conv_now = float(exe.get("conviction") or 0.0)
            tech_now = float((ctx.get("technical") or {}).get("confidence") or 0.5)
            surv_now = float(pros.get("survival_score") or 50.0)

            entry_px = float(pos.get("avg_entry_price") or snap.get("price") or 0.0)
            cur_px = float((bar or {}).get("close") or pos.get("last_price") or snap.get("price") or 0.0)
            bars_held = int(pos.get("bars_held") or 0)
            unreal_pct = (((cur_px / entry_px) - 1.0) * 100.0) if (entry_px > 0 and cur_px > 0) else 0.0
            mfe_pct = max(float(pos.get("mfe_pct") or 0.0), unreal_pct)
            orig_target = float(targets_map.get(sid) or 0.0)

            exit_reason: str | None = None
            if exe.get("action") == "SELL" or badge in ("EXIT", "REDUCE"):
                exit_reason = f"ORCHESTRATOR_{badge or 'SELL'}"
            elif orig_target > entry_px > 0 and cur_px >= orig_target:
                exit_reason = "TARGET_HIT_PROFIT_TAKE"
            elif unreal_pct >= 12.0:
                exit_reason = "TARGET_HIT_12PCT"
            elif mfe_pct >= 5.0 and unreal_pct <= (mfe_pct * 0.45) and unreal_pct < (mfe_pct - 2.5):
                exit_reason = "TRAILING_PROFIT_PROTECT"
            elif (bars_held >= 7 and unreal_pct < 1.0 and (tech_now < 0.65 or conv_now < min_conv)) or (
                bars_held >= 12 and unreal_pct <= 2.0
            ):
                exit_reason = "TIME_DECAY_STAGNANT"
            elif bars_held >= 2 and surv_now < (min_surv - 6.0) and unreal_pct < 2.0:
                exit_reason = "BEAR_PROSECUTION_DOWNGRADE"
            elif conv_now < max(0.48, min_conv - 0.08):
                exit_reason = "CONVICTION_FADE_EXIT"

            if exit_reason is not None:
                exits.append(
                    {
                        "sec_id": sid,
                        "symbol": sym,
                        "action": "SELL",
                        "decision_badge": badge or "EXIT",
                        "conviction": conv_now,
                        "survival_score": surv_now,
                        "unrealized_pct": round(unreal_pct, 2),
                        "bars_held": bars_held,
                        "reason": exit_reason,
                        "summary": f"{exit_reason} (ret={unreal_pct:+.2f}%, MFE=+{mfe_pct:.2f}%, held={bars_held}d, surv={surv_now:.1f})",
                    }
                )
            elif bars_held >= 5 and unreal_pct < 3.0:
                rotation_pool.append(
                    {
                        "sec_id": sid,
                        "symbol": sym,
                        "conviction": conv_now,
                        "survival_score": surv_now,
                        "unrealized_pct": unreal_pct,
                        "bars_held": bars_held,
                        "badge": badge,
                    }
                )

        # Proactive Edge Rotation: if portfolio slots are full after exits and a fresh candidate has a clear edge
        remaining_open = len(open_positions) - len(exits)
        if remaining_open >= max(1, int(max_tactical_positions)) and rotation_pool and top_candidates:
            fresh = [
                c
                for c in top_candidates
                if str(c.get("sec_id") or "") not in held_sids
            ]
            fresh.sort(key=lambda c: -float(c.get("score") or 0.0))
            rotation_pool.sort(key=lambda r: (r["conviction"], r["unrealized_pct"]))
            if fresh and rotation_pool:
                best_cand = fresh[0]
                weakest = rotation_pool[0]
                if float(best_cand.get("score") or 0.0) >= float(weakest["conviction"]) + 0.08:
                    exits.append(
                        {
                            "sec_id": weakest["sec_id"],
                            "symbol": weakest["symbol"],
                            "action": "SELL",
                            "decision_badge": "ROTATE",
                            "conviction": weakest["conviction"],
                            "survival_score": weakest["survival_score"],
                            "unrealized_pct": round(weakest["unrealized_pct"], 2),
                            "bars_held": weakest["bars_held"],
                            "reason": "PROACTIVE_EDGE_ROTATION",
                            "summary": (
                                f"PROACTIVE_EDGE_ROTATION: {best_cand['symbol']} "
                                f"(conv={float(best_cand.get('score') or 0)*100:.1f}%) replaces stagnant {weakest['symbol']} "
                                f"(conv={weakest['conviction']*100:.1f}%, ret={weakest['unrealized_pct']:+.2f}%)"
                            ),
                        }
                    )
        return exits

