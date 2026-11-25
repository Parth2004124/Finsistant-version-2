"""Multi-Agent Trading Orchestrator for Finsistant Version 2.

Binds:
- TechBot_1 (TechnicalAnalyst): Evaluates RSI-14, MACD, 20D SMA, Bollinger, and Degree-2 Regression Curve
- FundBot_1 (FundamentalAnalyst): Evaluates StockSight V2 4-Pillar Score (100) + Porter's 5 Forces (100)
- ExecBot_1 (ExecutionAgent): Evaluates Moreshwar Levels, StockSight Decision Matrix, Auto-Risk Sizing, and Kite Execution Queue
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import platform
import sys
import time
from typing import Any, Sequence

from .config import FinsistantSettings, load_settings
from .quant_engine import get_asset_full_analysis


@dataclass
class AgentMessage:
    timestamp: str
    agent_name: str
    agent_role: str
    stage: str
    signal: str
    confidence: float
    summary: str
    metrics: dict[str, Any] = field(default_factory=dict)

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PipelineResult:
    run_id: str
    symbol: str
    timeframe: str
    environment: str
    agent_mode: str
    started_at: str
    duration_ms: float
    final_action: str
    decision_badge: str
    conviction_score: float
    position_size: float
    quantity: int
    entry_price: float
    stop_loss: float
    take_profit: float
    messages: list[AgentMessage]
    order_receipt: dict[str, Any]

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


class BaseTradingAgent:
    role_key: str = "base"

    def __init__(self, name: str) -> None:
        self.name = name

    def diagnostics(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "role": self.__class__.__name__,
            "role_key": self.role_key,
            "status": "READY",
        }


def score_technical_confidence(
    *,
    price: float,
    sma_20: float,
    ema_12: float,
    ema_26: float,
    rsi: float,
    macd: float,
    macd_sig: float,
    proj_30d: float,
) -> tuple[float, str]:
    """TechBot_1 scoring rule as a pure function (single source of truth).

    Used by `TechnicalAnalyst.evaluate` for single-symbol pipelines AND by the Moreshwar 3.2
    full-universe scanner, so the orchestrator scores 1,000+ stocks with the identical rule.
    """
    score = 0.50
    if price >= sma_20:
        score += 0.14
    else:
        score -= 0.12
    if ema_12 >= ema_26:
        score += 0.12
    else:
        score -= 0.10
    if macd >= macd_sig:
        score += 0.08
    else:
        score -= 0.08
    if proj_30d > price:
        score += 0.10
    else:
        score -= 0.10
    if 46.0 <= rsi <= 70.0:
        score += 0.07
    conf = round(max(0.18, min(0.96, score)), 3)
    signal = "BULLISH" if conf >= 0.58 else ("BEARISH" if conf <= 0.44 else "NEUTRAL")
    return conf, signal


class TechnicalAnalyst(BaseTradingAgent):
    """TechBot_1 — Evaluates OHLCV indicators + Finsistant Polynomial Regression projection."""

    role_key = "technical"

    def __init__(self, name: str = "TechBot_1") -> None:
        super().__init__(name=name)

    def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
        symbol = context.get("symbol", "RELIANCE")
        is_holding = bool(context.get("is_holding", False))
        snap = context.get("asset_analysis") or get_asset_full_analysis(symbol, is_holding=is_holding)
        context["asset_analysis"] = snap

        price = snap["price"]
        sma_20 = snap["sma_20"]
        ema_12 = snap["ema_12"]
        ema_26 = snap["ema_26"]
        rsi = snap["rsi_14"]
        macd = snap["macd"]
        macd_sig = snap["macd_signal"]
        proj_30d = snap["projected_30d_target"]
        reg_curve = snap["regression_curve"]

        conf, signal = score_technical_confidence(
            price=price, sma_20=sma_20, ema_12=ema_12, ema_26=ema_26,
            rsi=rsi, macd=macd, macd_sig=macd_sig, proj_30d=proj_30d,
        )
        tech_score_100 = int(round(conf * 100))

        return {
            "stage": "TechSight & Quant Regression",
            "signal": signal,
            "confidence": conf,
            "technical_score": tech_score_100,
            "summary": (
                f"Spot {price:,.2f} vs 20D SMA {sma_20:,.2f} ({reg_curve}, 30D Reg Target {proj_30d:,.2f}). "
                f"RSI(14)={rsi}, MACD={macd:+.2f} vs Sig={macd_sig:+.2f}."
            ),
            "metrics": {
                "price": price,
                "sma_20": sma_20,
                "rsi_14": rsi,
                "macd": macd,
                "projected_30d_target": proj_30d,
                "regression_curve": reg_curve,
                "technical_score": tech_score_100,
            },
        }


class FundamentalAnalyst(BaseTradingAgent):
    """FundBot_1 — Evaluates StockSight V2 4-Pillar Score + Porter's 5 Forces Moat Score."""

    role_key = "fundamental"

    def __init__(self, name: str = "FundBot_1") -> None:
        super().__init__(name=name)

    def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
        symbol = context.get("symbol", "RELIANCE")
        snap = context.get("asset_analysis") or get_asset_full_analysis(symbol)
        f_score = snap["fundamental_score"]
        p_score = snap["porters_score"]

        composite = (f_score["total"] + p_score["total"]) / 200.0
        conf = round(max(0.15, min(0.97, composite)), 3)
        signal = "BULLISH" if f_score["total"] >= 65 else ("BEARISH" if f_score["total"] <= 40 else "NEUTRAL")

        return {
            "stage": "StockSight V2 4-Pillar & Porter Moat",
            "signal": signal,
            "confidence": conf,
            "summary": (
                f"Fundamental {f_score['total']}/100 (Bus {f_score['business']}/40, Moat {f_score['moat']}/20, "
                f"Mgmt {f_score['management']}/20, Risk {f_score['risk']}/20) · "
                f"Porter's 5 Forces {p_score['total']}/100 [{f_score['industry']}]. {f_score['nlp']}"
            ),
            "metrics": {
                "fundamental_total": f_score["total"],
                "porters_total": p_score["total"],
                "pillars": f_score,
                "porters": p_score,
            },
        }


class ExecutionAgent(BaseTradingAgent):
    """ExecBot_1 — Computes Moreshwar Levels, Decision Matrix, Auto-Risk Qty, and Kite Order Receipt."""

    role_key = "execution"

    def __init__(self, name: str = "ExecBot_1") -> None:
        super().__init__(name=name)

    def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
        symbol = context.get("symbol", "RELIANCE")
        snap = context.get("asset_analysis") or get_asset_full_analysis(symbol)
        tech = context.get("technical", {})
        fund = context.get("fundamental", {})

        moreshwar = snap["moreshwar_levels"]
        decision = snap["decision_matrix"]
        price = float(snap["price"])
        stop_loss = float(moreshwar["stop_loss"])
        target = float(moreshwar["target"])
        entry = float(moreshwar["entry_price"])

        capital = float(context.get("capital", 250000.0))
        active_trades = int(context.get("active_trades", 2))
        max_trades = int(context.get("max_concurrent_trades", 5))
        agent_mode = str(context.get("agent_mode", "paper_trading"))
        currency = str(context.get("currency", "INR"))

        tech_conf = float(tech.get("confidence", 0.5))
        fund_conf = float(fund.get("confidence", 0.5))
        # Default weights 0.45/0.55; Moreshwar 3.2's learning loop may supply bounded, audit-logged weights & guards.
        w_tech, w_fund = context.get("conviction_weights") or (0.45, 0.55)
        raw_conv = float(w_tech) * tech_conf + float(w_fund) * fund_conf
        rsi_val = float(snap.get("rsi_14") or 50.0)
        rsi_pen = float(context.get("rsi_overbought_penalty") or 0.0) if rsi_val > 72.0 else 0.0
        sec_adj = float(context.get("sector_adjustment") or 0.0)
        conviction = round(max(0.05, min(0.98, raw_conv - rsi_pen + sec_adj)), 3)
        min_conviction = float(context.get("min_conviction") or 0.60)

        # Finsistant Auto-Risk Position Sizing: risk budget divided by (Entry - StopLoss), capped by single-stock cap
        risk_budget_pct = float(context.get("risk_budget_pct") or (0.011 if agent_mode == "paper_trading" else 0.015))
        max_single_pct = float(context.get("max_single_stock_pct") or (15.0 if agent_mode == "paper_trading" else 22.0))
        risk_budget = capital * risk_budget_pct
        per_share_risk = max(1.0, abs(price - stop_loss))
        raw_qty = max(1, int(risk_budget // per_share_risk))
        max_affordable_qty = max(1, int((capital * (max_single_pct / 100.0)) // max(1.0, price)))
        qty = min(raw_qty, max_affordable_qty)
        notional = round(qty * price, 2)

        matrix_action = decision["action"]
        price_above_sma = price >= float(snap.get("sma_20") or price)
        rr_ok = float(moreshwar.get("rr_ratio") or 0.0) >= 1.5
        sip_aligned = matrix_action == "SIP ONLY" and conviction >= max(0.72, min_conviction + 0.08) and price_above_sma

        if active_trades >= max_trades and matrix_action in ("BUY NOW", "ADD", "SIP ONLY"):
            action = "HOLD"
            signal = "REJECTED"
            qty = 0
            notional = 0.0
            reason = f"Guardrail triggered: Open trades ({active_trades}) reached MAX_CONCURRENT_TRADES ({max_trades})."
        elif (matrix_action in ("BUY NOW", "ADD") or sip_aligned) and rr_ok and conviction >= min_conviction:
            action = "BUY"
            signal = "APPROVED"
            reason = (
                f"Decision Matrix: {matrix_action} ({conviction*100:.1f}% conviction >= {min_conviction*100:.0f}% bar). "
                f"Auto-Risk sized {qty} shares ({currency} {notional:,.2f}) | "
                f"MOVIQ Target {target:,.2f} / SL {stop_loss:,.2f} (R:R 1:{moreshwar['rr_ratio']})."
            )
        elif matrix_action in ("EXIT", "REDUCE", "AVOID") or conviction <= 0.44:
            action = "SELL"
            signal = "APPROVED"
            reason = (
                f"Decision Matrix: {matrix_action} — Risk Warning triggered ({conviction*100:.1f}% score). "
                f"Recommended exit/hedge to preserve capital."
            )
        else:
            action = "HOLD"
            signal = "FILTERED" if matrix_action in ("BUY NOW", "ADD", "SIP ONLY") else "NEUTRAL"
            qty = 0
            notional = 0.0
            reason = (
                f"Quality Gate Hold ({matrix_action}, Conv {conviction*100:.1f}%, R:R 1:{moreshwar['rr_ratio']}): "
                f"{decision.get('rationale', '')}"
            )

        receipt = {
            "order_id": f"fnv2-ord-{int(time.time() * 1000) % 1000000:06d}",
            "symbol": symbol,
            "side": action,
            "decision_badge": matrix_action,
            "mode": agent_mode,
            "broker_bridge": "Zerodha Kite Connect",
            "status": "QUEUED" if action in ("BUY", "SELL") else "STANDBY",
            "quantity": qty,
            "entry_price": entry,
            "fill_price": price,
            "notional": notional,
            "stop_loss": stop_loss,
            "take_profit": target,
            "rr_ratio": f"1 : {moreshwar['rr_ratio']}",
            "active_trades_slots": f"{active_trades + (1 if action in ('BUY', 'SELL') else 0)} / {max_trades}",
        }

        return {
            "stage": "MOVIQ Levels & Kite Execution Bridge",
            "signal": signal,
            "action": action,
            "decision_badge": matrix_action,
            "conviction": conviction,
            "confidence": conviction,
            "quantity": qty,
            "position_size": notional,
            "entry_price": entry,
            "stop_loss": stop_loss,
            "take_profit": target,
            "summary": reason,
            "metrics": receipt,
            "order_receipt": receipt,
        }


class AgentRunner:
    """MOVIQ Multi-Agent Orchestrator."""

    def __init__(
        self,
        agents: Sequence[Any],
        settings: FinsistantSettings | None = None,
    ) -> None:
        self.agents = list(agents)
        self.settings = settings or load_settings()
        self.history: list[PipelineResult] = []

    def health_check(self) -> dict[str, Any]:
        py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        py_ok = (sys.version_info.major, sys.version_info.minor) >= (3, 10)
        agent_statuses = [
            a.diagnostics() if hasattr(a, "diagnostics") else {"name": getattr(a, "name", "Agent"), "status": "READY"}
            for a in self.agents
        ]
        all_ready = all(a.get("status") == "READY" for a in agent_statuses) and py_ok

        return {
            "pipeline_status": "HEALTHY" if all_ready else "DEGRADED",
            "platform": "MOVIQ — Quantitative Intelligence Engine",
            "merged_modules": [
                "StockSight Version 2 (4-Pillar + Porter's 5 Forces + MOVIQ Levels)",
                "Finsistant Execution Terminal (TechSight Queue + Polynomial Regression + Stocky AI)",
                "Multi-Agent Orchestrator (TechBot_1 + FundBot_1 + ExecBot_1)",
            ],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "environment": self.settings.finsistant_environment,
            "agent_mode": self.settings.agent_mode,
            "default_currency": self.settings.default_currency,
            "max_concurrent_trades": self.settings.max_concurrent_trades,
            "system": {
                "python_version": py_ver,
                "python_compatible": py_ok,
                "platform": platform.platform(),
            },
            "agents_bound": len(self.agents),
            "agents": agent_statuses,
        }

    def run_pipeline(
        self,
        symbol: str = "RELIANCE",
        timeframe: str = "1D",
        capital: float = 250000.0,
        active_trades: int = 2,
        is_holding: bool = False,
        asset_analysis: dict[str, Any] | None = None,
        learned_params: dict[str, Any] | None = None,
    ) -> PipelineResult:
        t0 = time.perf_counter()
        symbol = symbol.upper().strip()
        started_at = datetime.now(timezone.utc).isoformat()
        messages: list[AgentMessage] = []

        context: dict[str, Any] = {
            "symbol": symbol,
            "timeframe": timeframe,
            "capital": capital,
            "active_trades": active_trades,
            "is_holding": is_holding,
            "max_concurrent_trades": self.settings.max_concurrent_trades,
            "agent_mode": self.settings.agent_mode,
            "currency": self.settings.default_currency,
        }
        if asset_analysis is not None:
            context["asset_analysis"] = asset_analysis
        if learned_params:
            wt = float(learned_params.get("w_tech", 0.45))
            wf = float(learned_params.get("w_fund", 0.55))
            context["conviction_weights"] = (wt, wf)
            if "min_conviction" in learned_params:
                context["min_conviction"] = float(learned_params["min_conviction"])
            if "rsi_overbought_penalty" in learned_params:
                context["rsi_overbought_penalty"] = float(learned_params["rsi_overbought_penalty"])
            sec_map = learned_params.get("sector_adjustments") or {}
            sec_name = str((asset_analysis or {}).get("sector") or "")
            if sec_name and sec_name in sec_map:
                context["sector_adjustment"] = float(sec_map[sec_name])

        for agent in self.agents:
            step = agent.evaluate(context)
            context[agent.role_key] = step
            messages.append(
                AgentMessage(
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    agent_name=agent.name,
                    agent_role=agent.__class__.__name__,
                    stage=step.get("stage", "analysis"),
                    signal=step.get("signal", "NEUTRAL"),
                    confidence=round(float(step.get("confidence", 0.5)), 3),
                    summary=step.get("summary", ""),
                    metrics=step.get("metrics", {}),
                )
            )

        exec_data = context.get("execution", {})
        duration_ms = round((time.perf_counter() - t0) * 1000, 2)

        res = PipelineResult(
            run_id=f"fnv2-run-{int(time.time() * 1000) % 1000000:06d}",
            symbol=symbol,
            timeframe=timeframe,
            environment=self.settings.finsistant_environment,
            agent_mode=self.settings.agent_mode,
            started_at=started_at,
            duration_ms=duration_ms,
            final_action=exec_data.get("action", "HOLD"),
            decision_badge=exec_data.get("decision_badge", "HOLD"),
            conviction_score=round(float(exec_data.get("conviction", 0.5)), 3),
            position_size=round(float(exec_data.get("position_size", 0.0)), 2),
            quantity=int(exec_data.get("quantity", 0)),
            entry_price=round(float(exec_data.get("entry_price", 0.0)), 2),
            stop_loss=round(float(exec_data.get("stop_loss", 0.0)), 2),
            take_profit=round(float(exec_data.get("take_profit", 0.0)), 2),
            messages=messages,
            order_receipt=exec_data.get("order_receipt", {}),
        )
        self.history.insert(0, res)
        self.history = self.history[:25]
        return res
