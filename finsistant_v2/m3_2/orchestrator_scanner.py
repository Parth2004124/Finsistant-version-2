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
from typing import Any, Callable

from finsistant_v2.agents import (
    ExecutionAgent,
    FundamentalAnalyst,
    TechnicalAnalyst,
    score_technical_confidence,
)
from finsistant_v2.quant_engine import compute_ohlcv_and_regression_from_candles

ORCH_STRATEGY_ID = "STRAT_ORCH_AGENT_CONSENSUS"
SCANNER_VERSION = "3.2.0"
PRIOR_W_TECH = 0.45
PRIOR_W_FUND = 0.55
PRIOR_MIN_CONVICTION = 0.60
PRIOR_RSI_OVERBOUGHT_PENALTY = 0.0
MISSION_TARGET_SESSIONS = 80


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
    max_workers: int = 6

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
    """Fetch real live Yahoo fundamentals + OHLCV analysis without hanging on NSE MCP timeouts."""
    from finsistant_v2 import live_data
    from finsistant_v2.quant_engine import compute_ohlcv_and_regression_from_candles
    from finsistant_v2.scoring_engine import (
        calculate_decision_matrix,
        calculate_fundamental_score,
        calculate_moreshwar_levels,
        calculate_porters_score,
    )

    sym = symbol.upper().strip()
    yahoo_sym, exchange_hint = live_data.resolve_yahoo_symbol(sym)
    live = live_data._fetch_via_yfinance(sym, yahoo_sym, exchange_hint)
    if not live:
        raise RuntimeError(f"Real Yahoo Finance data unavailable for {sym} ({yahoo_sym})")
    live.setdefault("price_source_category", "YAHOO_LIVE_FEED")
    live.setdefault("ohlcv_source_category", "HISTORICAL_MARKET_DATA")
    live.setdefault("fundamental_source_category", "YAHOO_LIVE_FEED")

    quant = compute_ohlcv_and_regression_from_candles(live["candles"], periods=45)
    candles = quant["candles"]
    latest = candles[-1]
    spot = float(live["price"])

    f_score = calculate_fundamental_score(live)
    p_score = calculate_porters_score(live)
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
        "market_timestamp_kind": live.get("market_timestamp_kind", "end_of_day_bar"),
        "missing_fields": list(live.get("missing_fields") or []),
        "proxy_fields": dict(live.get("proxy_fields") or {}),
        "cache_hit": False,
        "cache_age_seconds": 0.0,
        "price_source_category": live["price_source_category"],
        "ohlcv_source_category": live["ohlcv_source_category"],
        "fundamental_source_category": live["fundamental_source_category"],
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


class OrchestratorUniverseScanner:
    """TechBot full-universe scan -> shortlist -> parallel FundBot + ExecBot -> ranked paper candidates."""

    def __init__(
        self,
        config: OrchestratorScanConfig | None = None,
        *,
        analysis_fn: Callable[[str], dict[str, Any]] | None = None,
    ) -> None:
        self.cfg = config or OrchestratorScanConfig()
        self._analysis_fn = analysis_fn or _default_fast_live_analysis
        self.tech_agent = TechnicalAnalyst()
        self.fund_agent = FundamentalAnalyst()
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
    ) -> dict[str, Any]:
        weights = (float(learned_params.get("w_tech", PRIOR_W_TECH)), float(learned_params.get("w_fund", PRIOR_W_FUND)))
        min_conv = float(learned_params.get("min_conviction", PRIOR_MIN_CONVICTION))
        rsi_pen = float(learned_params.get("rsi_overbought_penalty", PRIOR_RSI_OVERBOUGHT_PENALTY))
        sec_adj_map = learned_params.get("sector_adjustments") or {}
        sec_adj = float(sec_adj_map.get(r["sector"], 0.0))

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
        try:
            snap = self._analysis_fn(r["symbol"])
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
                "currency": "INR",
                "is_holding": False,
                "conviction_weights": weights,
                "min_conviction": min_conv,
                "rsi_overbought_penalty": rsi_pen,
                "sector_adjustment": sec_adj,
            }
            for agent in (self.tech_agent, self.fund_agent, self.exec_agent):
                ctx[agent.role_key] = agent.evaluate(ctx)
            tech, fund, exe = ctx["technical"], ctx["fundamental"], ctx["execution"]
            fscore = snap["fundamental_score"]["total"]
            pscore = snap["porters_score"]["total"]
            price = float(snap["price"])
            stop_loss = float(exe.get("stop_loss") or 0.0)
            raw_stop_ratio = (price - stop_loss) / price if price > 0 and stop_loss > 0 else self.cfg.stop_ratio_max
            stop_ratio = min(self.cfg.stop_ratio_max, max(self.cfg.stop_ratio_min, raw_stop_ratio))
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
                    "moreshwar_stop_loss": stop_loss,
                    "moreshwar_target": float(exe.get("take_profit") or 0.0),
                    "stop_ratio": round(stop_ratio, 6),
                    "fundamental_source": snap.get("fundamental_source_category"),
                    "price_source": snap.get("price_source_category"),
                    "missing_fields": list(snap.get("missing_fields") or []),
                }
            )
        except Exception as exc:
            row["status"] = "EXCLUDED_AGENT_DATA_UNAVAILABLE"
            row["error"] = str(exc)[:200]
        return row

    def run_agents_on_shortlist(
        self,
        shortlist: list[dict[str, Any]],
        *,
        learned_params: dict[str, Any] | None = None,
        weights: tuple[float, float] | None = None,
        capital_inr: float = 100_000.0,
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

        workers = max(1, min(self.cfg.max_workers, len(shortlist)))
        if workers <= 1:
            return [self._evaluate_one_shortlisted(r, learned_params=params, capital_inr=capital_inr) for r in shortlist]

        by_idx: dict[int, dict[str, Any]] = {}
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = {
                pool.submit(self._evaluate_one_shortlisted, r, learned_params=params, capital_inr=capital_inr): idx
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
        params.setdefault("status", weights_status)

        tech_rows, skipped = self.score_universe_technical(
            trading_date=trading_date,
            equity_histories=equity_histories,
            eq_bars=eq_bars,
            rsi_overbought_penalty=float(params["rsi_overbought_penalty"]),
            sector_adjustments=params["sector_adjustments"],
        )
        shortlist = self.shortlist(tech_rows)
        agent_rows = self.run_agents_on_shortlist(shortlist, learned_params=params, capital_inr=capital_inr)

        buys = [r for r in agent_rows if r["status"] == "AGENTS_OK" and r["exec_action"] == "BUY"]
        buys.sort(key=lambda r: (-r["conviction"], r["tech_rank"]))
        candidates: list[dict[str, Any]] = []
        for r in buys[: self.cfg.max_final_candidates]:
            candidates.append(
                {
                    "sec_id": r["sec_id"],
                    "symbol": r["symbol"],
                    "strategy_id": ORCH_STRATEGY_ID,
                    "score": float(r["conviction"]),
                    "stop_ratio": float(r["stop_ratio"]),
                    "ev_mode": "LEARNING_PHASE",
                    "tech_conf": r["tech_conf"],
                    "fund_conf": r["fund_conf"],
                    "tech_rank": r["tech_rank"],
                }
            )

        n_ok = sum(1 for r in agent_rows if r["status"] == "AGENTS_OK")
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
                "exec_buy_approved": len(buys),
                "paper_candidates": len(candidates),
                "skipped": skipped,
            },
            "shortlist": agent_rows,
            "candidates": candidates,
            "universe_tech": tech_rows,
        }
