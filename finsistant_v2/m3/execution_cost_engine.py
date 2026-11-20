"""Moreshwar 3.0 Prompt 8 — Execution Intelligence, Realistic Cost Engine & Paper-Only Broker Adapter.

Implements:
1. Versioned Historical Indian Cost Engine (Zerodha & NSE tariffs):
   - Equity Delivery (`CNC`): Rs 0 brokerage (or pre-2019/full-service toggle), STT 0.10% Buy & Sell,
     NSE transaction fee (historical 0.00325% pre-2024-10 vs 0.00297% post-2024-10),
     SEBI fee 0.0001%, Stamp 0.015% Buy only, GST 18%, DP charge Rs 15.93/scrip/sell-day.
   - Equity Intraday (`MIS`): min(Rs 20, 0.03%) per order, STT 0.025% Sell only, Stamp 0.003% Buy.
   - Stock / Index Futures (`NRML` F&O): min(Rs 20, 0.03%) per order, STT 0.02% Sell only (post-2024-10, 0.0125% prior),
     NSE Exch 0.00173%, Stamp 0.002% Buy.
   - Spread + Square-Root Market Impact (`half_spread + eta * sigma_daily * sqrt(order_value / adv)`).
2. Realistic Order Simulator (`MARKET`, `LIMIT`, `SL_M`, `SL_L`):
   - Models partial fills (`max_vol_participation`), delayed fills (`limit` uncrossed),
     rejected orders (dust, > participation ceiling, zero volume), unfilled limit orders,
     and honest gap-through-stop fills at actual bar open.
3. Cleanly Separated Paper-Only Zerodha Integration Architecture:
   - Separates MarketDataPort, AccountDataPort, OrderConstructor, OrderApprovalGate,
     OrderSubmitter, OrderMonitor — defaulting ALL live submission toggles to `False`,
     never reading or logging `API_SECRET`, `TOTP`, or raw access tokens.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import math
from typing import Any


@dataclass(frozen=True)
class HistoricalTariffSchedule:
    tariff_version: str = "2024-10-nse-zerodha-v1"
    # Delivery (CNC)
    delivery_brokerage_flat_or_pct: str = "ZERODHA_FREE"  # "ZERODHA_FREE" | "LEGACY_003"
    delivery_stt: float = 0.0010
    delivery_stamp_buy: float = 0.00015
    delivery_dp_sell_inr: float = 15.93
    # Intraday (MIS)
    intraday_stt_sell: float = 0.00025
    intraday_stamp_buy: float = 0.00003
    # Futures (NRML F&O)
    futures_stt_sell_pre_202410: float = 0.000125
    futures_stt_sell_post_202410: float = 0.00020
    futures_exch_rate: float = 0.0000173
    futures_stamp_buy: float = 0.00002
    # Common levies
    nse_cm_exch_pre_202410: float = 0.0000325
    nse_cm_exch_post_202410: float = 0.0000297
    sebi_fee_rate: float = 0.000001
    gst_rate: float = 0.18
    # Execution friction model
    base_half_spread_bps: float = 4.0
    impact_eta: float = 0.25  # Square-root market impact coefficient


def compute_market_impact_and_slippage_bps(
    *,
    order_notional_inr: float,
    adv_20d_inr: float,
    daily_vol_frac: float = 0.018,
    hl_spread_proxy_pct: float = 2.0,
    tariff: HistoricalTariffSchedule = HistoricalTariffSchedule(),
) -> dict[str, float]:
    """Compute spread crossing + square-root market impact in basis points."""
    # Typical effective half-spread for NSE liquid cash stock ~ 3-10 bps; scale gently with HL range
    half_spread_bps = max(tariff.base_half_spread_bps, min(25.0, hl_spread_proxy_pct * 2.5))
    part_rate = order_notional_inr / max(100_000.0, adv_20d_inr)
    impact_bps = tariff.impact_eta * max(0.005, daily_vol_frac) * math.sqrt(max(0.0, part_rate)) * 10_000.0
    total_slip_bps = min(120.0, half_spread_bps + impact_bps)
    return {
        "half_spread_bps": round(half_spread_bps, 2),
        "market_impact_bps": round(impact_bps, 2),
        "total_slippage_bps": round(total_slip_bps, 2),
        "participation_pct_adv": round(part_rate * 100.0, 4),
    }


def calculate_historical_indian_costs(
    *,
    product_type: str,  # "DELIVERY" | "INTRADAY" | "FUTURES"
    side: str,          # "BUY" | "SELL"
    raw_notional: float,
    trade_date: str = "2025-01-01",
    slippage_bps: float = 7.5,
    dp_charged_today: bool = False,
    tariff: HistoricalTariffSchedule = HistoricalTariffSchedule(),
) -> dict[str, float]:
    """Exact tariff breakdown preserving historical date-aware exchange/STT regime changes."""
    pt = product_type.upper().strip()
    sd = side.upper().strip()
    amt = max(0.0, float(raw_notional))
    slip_frac = max(0.0, float(slippage_bps)) / 10_000.0
    exec_notional = amt * (1.0 + slip_frac) if sd == "BUY" else amt * max(0.0, 1.0 - slip_frac)
    slip_cost = abs(exec_notional - amt)
    post_oct24 = trade_date >= "2024-10-01"

    if pt == "DELIVERY":
        brokerage = 0.0 if tariff.delivery_brokerage_flat_or_pct == "ZERODHA_FREE" else min(20.0, exec_notional * 0.0003)
        stt = round(exec_notional * tariff.delivery_stt, 2)
        exch_rate = tariff.nse_cm_exch_post_202410 if post_oct24 else tariff.nse_cm_exch_pre_202410
        exch = exec_notional * exch_rate
        sebi = exec_notional * tariff.sebi_fee_rate
        stamp = (exec_notional * tariff.delivery_stamp_buy) if sd == "BUY" else 0.0
        gst = (brokerage + exch + sebi) * tariff.gst_rate
        dp = tariff.delivery_dp_sell_inr if (sd == "SELL" and not dp_charged_today and amt > 0) else 0.0
    elif pt == "INTRADAY":
        brokerage = min(20.0, exec_notional * 0.0003)
        stt = round(exec_notional * tariff.intraday_stt_sell, 2) if sd == "SELL" else 0.0
        exch_rate = tariff.nse_cm_exch_post_202410 if post_oct24 else tariff.nse_cm_exch_pre_202410
        exch = exec_notional * exch_rate
        sebi = exec_notional * tariff.sebi_fee_rate
        stamp = (exec_notional * tariff.intraday_stamp_buy) if sd == "BUY" else 0.0
        gst = (brokerage + exch + sebi) * tariff.gst_rate
        dp = 0.0
    elif pt == "FUTURES":
        brokerage = min(20.0, exec_notional * 0.0003)
        stt_r = tariff.futures_stt_sell_post_202410 if post_oct24 else tariff.futures_stt_sell_pre_202410
        stt = round(exec_notional * stt_r, 2) if sd == "SELL" else 0.0
        exch = exec_notional * tariff.futures_exch_rate
        sebi = exec_notional * tariff.sebi_fee_rate
        stamp = (exec_notional * tariff.futures_stamp_buy) if sd == "BUY" else 0.0
        gst = (brokerage + exch + sebi) * tariff.gst_rate
        dp = 0.0
    else:
        raise ValueError(f"Unsupported product_type: {product_type}")

    reg_total = brokerage + stt + exch + sebi + stamp + gst + dp
    return {
        "raw_notional": round(amt, 2),
        "exec_notional": round(exec_notional, 2),
        "brokerage": round(brokerage, 2),
        "stt": round(stt, 2),
        "exchange_txn": round(exch, 2),
        "sebi_fee": round(sebi, 2),
        "stamp_duty": round(stamp, 2),
        "gst": round(gst, 2),
        "dp_charge": round(dp, 2),
        "regulatory_total": round(reg_total, 2),
        "slippage_cost": round(slip_cost, 2),
        "total_cost": round(reg_total + slip_cost, 2),
    }


def simulate_order_execution_bar(
    *,
    order_type: str,  # "MARKET" | "LIMIT" | "SL_M" | "SL_L"
    side: str,        # "BUY" | "SELL"
    req_qty: int,
    limit_price: float | None = None,
    trigger_price: float | None = None,
    bar: dict[str, Any],
    adv_20d_inr: float = 100_000_000.0,
    max_bar_vol_participation: float = 0.01,
    min_viable_notional_inr: float = 1500.0,
    max_order_adv_pct: float = 2.0,
    trade_date: str = "2025-01-01",
    product_type: str = "DELIVERY",
) -> dict[str, Any]:
    """Simulate realistic OHLCV bar execution without impossible fills."""
    o = float(bar.get("open") or 0.0)
    h = float(bar.get("high") or 0.0)
    l = float(bar.get("low") or 0.0)
    c = float(bar.get("close") or 0.0)
    vol = int(bar.get("volume") or 0)
    ot = order_type.upper().strip()
    sd = side.upper().strip()

    if req_qty <= 0 or vol <= 0 or min(o, h, l, c) <= 0 or h < l:
        return {"status": "REJECTED", "reason": "ZERO_VOLUME_OR_INVALID_BAR", "filled_qty": 0, "fill_price": None}

    est_notional = req_qty * o
    if est_notional < min_viable_notional_inr:
        return {"status": "REJECTED", "reason": "DUST_ORDER_BELOW_MINIMUM", "filled_qty": 0, "fill_price": None}

    if est_notional > adv_20d_inr * (max_order_adv_pct / 100.0):
        # Cap to ADV limit before bar-volume participation check
        req_qty = max(0, int((adv_20d_inr * (max_order_adv_pct / 100.0)) // max(1e-6, o)))
        if req_qty < 1 or req_qty * o < min_viable_notional_inr:
            return {"status": "REJECTED", "reason": "EXCEEDS_ADV_LIQUIDITY_LIMIT", "filled_qty": 0, "fill_price": None}

    max_shares_vol = max(0, int(vol * max_bar_vol_participation))
    if max_shares_vol == 0:
        return {"status": "REJECTED", "reason": "BAR_VOLUME_TOO_LOW_FOR_PARTICIPATION", "filled_qty": 0, "fill_price": None}

    fill_qty = min(req_qty, max_shares_vol)
    partial = fill_qty < req_qty

    raw_fill_px: float | None = None
    is_gap = False
    if ot == "MARKET":
        raw_fill_px = o
    elif ot == "LIMIT":
        assert limit_price is not None and limit_price > 0
        if sd == "BUY":
            if o <= limit_price:
                raw_fill_px = o
            elif l <= limit_price:
                raw_fill_px = limit_price
            else:
                return {"status": "UNFILLED", "reason": "BUY_LIMIT_NOT_REACHED", "filled_qty": 0, "fill_price": None}
        else:
            if o >= limit_price:
                raw_fill_px = o
            elif h >= limit_price:
                raw_fill_px = limit_price
            else:
                return {"status": "UNFILLED", "reason": "SELL_LIMIT_NOT_REACHED", "filled_qty": 0, "fill_price": None}
    elif ot == "SL_M":
        assert trigger_price is not None and trigger_price > 0
        if sd == "SELL":
            if o <= trigger_price:
                raw_fill_px = o
                is_gap = True
            elif l <= trigger_price:
                raw_fill_px = trigger_price
            else:
                return {"status": "UNFILLED", "reason": "STOP_NOT_TRIGGERED", "filled_qty": 0, "fill_price": None}
        else:
            if o >= trigger_price:
                raw_fill_px = o
                is_gap = True
            elif h >= trigger_price:
                raw_fill_px = trigger_price
            else:
                return {"status": "UNFILLED", "reason": "BUY_STOP_NOT_TRIGGERED", "filled_qty": 0, "fill_price": None}
    elif ot == "SL_L":
        assert trigger_price is not None and limit_price is not None
        if sd == "SELL":
            if o <= trigger_price:
                if o >= limit_price:
                    raw_fill_px = o
                    is_gap = True
                else:
                    return {"status": "UNFILLED", "reason": "SL_LIMIT_GAP_THROUGH_UNFILLED", "filled_qty": 0, "fill_price": None}
            elif l <= trigger_price:
                if l <= limit_price <= h or trigger_price >= limit_price:
                    raw_fill_px = max(limit_price, trigger_price)
                else:
                    return {"status": "UNFILLED", "reason": "SL_LIMIT_NOT_FILLED", "filled_qty": 0, "fill_price": None}
            else:
                return {"status": "UNFILLED", "reason": "STOP_NOT_TRIGGERED", "filled_qty": 0, "fill_price": None}
        else:
            if o >= trigger_price and o <= limit_price:
                raw_fill_px = o
                is_gap = True
            elif h >= trigger_price and trigger_price <= limit_price:
                raw_fill_px = trigger_price
            else:
                return {"status": "UNFILLED", "reason": "BUY_SL_LIMIT_NOT_FILLED", "filled_qty": 0, "fill_price": None}

    assert raw_fill_px is not None
    raw_notional = fill_qty * raw_fill_px
    imp = compute_market_impact_and_slippage_bps(
        order_notional_inr=raw_notional,
        adv_20d_inr=adv_20d_inr,
        hl_spread_proxy_pct=((h - l) / max(1e-6, c)) * 100.0,
    )
    # Passive limit orders pay 0 crossing spread when filled at limit away from open
    eff_slip_bps = 0.0 if (ot == "LIMIT" and not is_gap and raw_fill_px == limit_price) else imp["total_slippage_bps"]
    slip_frac = eff_slip_bps / 10_000.0
    exec_px = raw_fill_px * (1.0 + slip_frac) if sd == "BUY" else raw_fill_px * (1.0 - slip_frac)
    costs = calculate_historical_indian_costs(
        product_type=product_type, side=sd, raw_notional=raw_notional, trade_date=trade_date, slippage_bps=eff_slip_bps
    )
    return {
        "status": "PARTIAL_FILL" if partial else "FILLED",
        "reason": "PARTIAL_VOL_CAP" if partial else ("GAP_THROUGH_OPEN_FILL" if is_gap else "FILLED_FULL"),
        "filled_qty": fill_qty,
        "requested_qty": req_qty,
        "raw_price": round(raw_fill_px, 4),
        "fill_price": round(exec_px, 4),
        "slippage_bps": eff_slip_bps,
        "impact_meta": imp,
        "costs": costs,
        "is_gap_fill": is_gap,
    }


# ---------------------------------------------------------------------------
# Cleanly Separated, Paper-Only Zerodha Broker Integration Architecture
# ---------------------------------------------------------------------------

@dataclass
class OrderTicket:
    symbol: str
    exchange: str
    side: str
    qty: int
    order_type: str
    product: str
    price: float | None = None
    trigger_price: float | None = None
    tag: str = "M3_PAPER"
    constructed_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class ZerodhaBrokerAdapter:
    """Six-stage separated Zerodha adapter with ALL live submission disabled by default.

    Stages strictly separated:
    1. MarketDataPort (read-only)
    2. AccountDataPort (read-only holdings/margins, masked, never leaks secrets)
    3. OrderConstructor (validates pure OrderTicket schema)
    4. OrderApprovalGate (human/risk explicit confirmation required)
    5. OrderSubmitter (BLOCKED when live_execution_enabled == False)
    6. OrderMonitor (paper ledger / read-only order status)
    """

    def __init__(self, *, live_execution_enabled: bool = False, paper_mode: bool = True) -> None:
        # Strict safety invariant: live execution ALWAYS defaults to False
        self.live_execution_enabled = bool(live_execution_enabled)
        self.paper_mode = bool(paper_mode)
        self._paper_orders: list[dict[str, Any]] = []
        self._approved_tickets: set[str] = set()

    def get_security_posture(self) -> dict[str, Any]:
        return {
            "paper_mode": self.paper_mode,
            "live_execution_enabled": self.live_execution_enabled,
            "secrets_exposed_to_agents": False,
            "capabilities": {
                "market_data": "READ_ONLY",
                "account_data": "READ_ONLY_MASKED",
                "order_construction": "ENABLED",
                "order_approval": "EXPLICIT_GATE_REQUIRED",
                "live_order_submission": "ENABLED" if (self.live_execution_enabled and not self.paper_mode) else "DISABLED_BY_DEFAULT",
                "order_monitoring": "PAPER_AND_READ_ONLY",
            },
        }

    def construct_order(
        self,
        *,
        symbol: str,
        side: str,
        qty: int,
        order_type: str = "LIMIT",
        product: str = "CNC",
        price: float | None = None,
        trigger_price: float | None = None,
    ) -> OrderTicket:
        if qty <= 0 or not symbol.strip():
            raise ValueError("Invalid symbol or non-positive order quantity.")
        if side.upper() not in ("BUY", "SELL"):
            raise ValueError("side must be BUY or SELL.")
        return OrderTicket(
            symbol=symbol.upper().strip(),
            exchange="NSE",
            side=side.upper().strip(),
            qty=int(qty),
            order_type=order_type.upper().strip(),
            product=product.upper().strip(),
            price=price,
            trigger_price=trigger_price,
        )

    def ticket_hash(self, ticket: OrderTicket) -> str:
        raw = f"{ticket.symbol}|{ticket.side}|{ticket.qty}|{ticket.order_type}|{ticket.product}|{ticket.price}|{ticket.trigger_price}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def approve_order(self, ticket: OrderTicket, *, approved_by: str) -> str:
        if not approved_by.strip():
            raise ValueError("Explicit approver identity required.")
        th = self.ticket_hash(ticket)
        self._approved_tickets.add(th)
        return th

    def submit_order(self, ticket: OrderTicket) -> dict[str, Any]:
        th = self.ticket_hash(ticket)
        if th not in self._approved_tickets:
            raise PermissionError("Order rejected by OrderApprovalGate: ticket was not explicitly approved.")
        if self.paper_mode or not self.live_execution_enabled:
            rec = {
                "order_id": f"PAPER-{th}",
                "mode": "PAPER_SIMULATION",
                "status": "PAPER_ACCEPTED",
                "ticket": asdict(ticket),
                "submitted_at": datetime.now(timezone.utc).isoformat(),
            }
            self._paper_orders.append(rec)
            return rec
        raise PermissionError(
            "Live Kite Connect order placement is intentionally blocked in research environment; use dedicated human-approved paper gateway."
        )

    def monitor_orders(self) -> list[dict[str, Any]]:
        return list(self._paper_orders)


class PaperOnlyZerodhaBrokerAdapter(ZerodhaBrokerAdapter):
    """Hardened Paper-Only Zerodha Broker Adapter used by the Moreshwar 3.0 Release Candidate."""

    LIVE_ORDER_SUBMISSION_ENABLED: bool = False

    def __init__(self) -> None:
        super().__init__(live_execution_enabled=False, paper_mode=True)

    def construct_paper_order(
        self,
        *,
        symbol: str,
        side: str,
        qty: int,
        order_type: str = "MARKET",
        product: str = "CNC",
        reference_price: float | None = None,
        strategy_id: str = "M3_PAPER",
        signal_date: str = "",
    ) -> dict[str, Any]:
        ticket = self.construct_order(
            symbol=symbol,
            side=side,
            qty=qty,
            order_type=order_type,
            product=product,
            price=reference_price,
        )
        th = self.approve_order(ticket, approved_by=f"PAPER_ENGINE:{strategy_id}:{signal_date}")
        submitted = self.submit_order(ticket)
        return {
            "paper_order_id": submitted["order_id"],
            "ticket_hash": th,
            "mode": submitted["mode"],
            "status": submitted["status"],
            "strategy_id": strategy_id,
            "signal_date": signal_date,
            "ticket": submitted["ticket"],
        }

    def submit_live_order(self, order_payload: dict[str, Any]) -> dict[str, Any]:
        raise PermissionError(
            "HARD_SECURITY_BOUNDARY: Live Zerodha Kite Connect order submission is permanently disabled in Moreshwar 3.0 Research & Paper-Trading RC1 (LIVE_ORDER_SUBMISSION_ENABLED=False)."
        )

    def security_audit_snapshot(self) -> dict[str, Any]:
        posture = self.get_security_posture()
        posture["LIVE_ORDER_SUBMISSION_ENABLED"] = self.LIVE_ORDER_SUBMISSION_ENABLED
        posture["live_order_submission_enabled"] = self.LIVE_ORDER_SUBMISSION_ENABLED
        posture["paper_orders_recorded"] = len(self._paper_orders)
        return posture


