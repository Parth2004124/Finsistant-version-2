"""Stocky AI Assistant merged from StockSight V2 (stocky.js) and Finsistant (stocky_engine.py)."""

from __future__ import annotations

import re
from typing import Any

from .quant_engine import ASSET_UNIVERSE, get_asset_full_analysis


class StockyAssistant:
    """Mathematical & Multi-Agent AI Assistant for Finsistant Version 2."""

    def process_query(
        self,
        message: str,
        holdings: list[dict[str, Any]],
        queue: list[dict[str, Any]],
    ) -> dict[str, Any]:
        msg = message.strip()
        msg_lower = msg.lower()

        # 1. COMPARE INTENT: "Compare HDFCBANK vs ICICIBANK"
        cmp_match = re.search(r"(?:compare|vs)\s+([a-zA-Z0-9_-]+)\s+(?:and|with|vs)\s+([a-zA-Z0-9_-]+)", msg_lower)
        if cmp_match:
            s1 = cmp_match.group(1).upper()
            s2 = cmp_match.group(2).upper()
            a1 = get_asset_full_analysis(s1)
            a2 = get_asset_full_analysis(s2)
            f1, p1, m1, d1 = a1["fundamental_score"], a1["porters_score"], a1["moreshwar_levels"], a1["decision_matrix"]
            f2, p2, m2, d2 = a2["fundamental_score"], a2["porters_score"], a2["moreshwar_levels"], a2["decision_matrix"]
            winner = s1 if (f1["total"] + p1["total"]) >= (f2["total"] + p2["total"]) else s2
            reply = (
                f"📊 **Stocky Head-to-Head Comparison: {s1} vs {s2}**\n\n"
                f"• **{s1}** (LTP: ₹{a1['price']:,.2f}) — Fundamental: **{f1['total']}/100** "
                f"(Bus {f1['business']}/40, Moat {f1['moat']}/20, Mgmt {f1['management']}/20, Risk {f1['risk']}/20) | "
                f"Porter's 5 Forces: **{p1['total']}/100** | Decision: **{d1['action']}** | Target: ₹{m1['target']:,.2f}\n"
                f"• **{s2}** (LTP: ₹{a2['price']:,.2f}) — Fundamental: **{f2['total']}/100** "
                f"(Bus {f2['business']}/40, Moat {f2['moat']}/20, Mgmt {f2['management']}/20, Risk {f2['risk']}/20) | "
                f"Porter's 5 Forces: **{p2['total']}/100** | Decision: **{d2['action']}** | Target: ₹{m2['target']:,.2f}\n\n"
                f"🏆 **Mathematical Verdict:** **{winner}** leads on risk-adjusted composite moat & growth quality."
            )
            return {"reply": reply, "action_symbol": winner}

        # 2. ALLOCATION / INVEST INTENT: "Invest 5 Lakh", "Allocate 250000"
        inv_match = re.search(r"(?:invest|allocate)\s+([\d,.]+)\s*(k|l|lakh|cr|crore)?", msg_lower)
        if inv_match:
            raw_num = float(inv_match.group(1).replace(",", ""))
            unit = (inv_match.group(2) or "").lower()
            if unit.startswith("k"):
                raw_num *= 1_000
            elif unit.startswith("l"):
                raw_num *= 100_000
            elif unit.startswith("c"):
                raw_num *= 10_000_000

            candidates = []
            for sym in ASSET_UNIVERSE:
                snap = get_asset_full_analysis(sym)
                f_tot = snap["fundamental_score"]["total"]
                p_tot = snap["porters_score"]["total"]
                comp = round((f_tot + p_tot) / 2.0, 1)
                if f_tot >= 60:
                    candidates.append({"symbol": sym, "price": snap["price"], "score": comp, "action": snap["decision_matrix"]["action"]})

            candidates.sort(key=lambda x: x["score"], reverse=True)
            top = candidates[:4]
            total_score = sum(c["score"] for c in top) or 1.0
            lines = [f"💰 **Stocky Score-Weighted Allocation Plan for ₹{raw_num:,.2f}:**\n"]
            used = 0.0
            for c in top:
                w = c["score"] / total_score
                budget = raw_num * w
                qty = int(budget // c["price"])
                cost = round(qty * c["price"], 2)
                used += cost
                lines.append(
                    f"• **{c['symbol']}** (Score: {c['score']}/100 · {c['action']}): "
                    f"Buy **{qty} shares** @ ₹{c['price']:,.2f} = **₹{cost:,.2f}** ({w*100:.1f}%)"
                )
            lines.append(f"\n**Unallocated Cash Reserve:** ₹{max(0.0, raw_num - used):,.2f}")
            return {"reply": "\n".join(lines)}

        # 3. BUY EXECUTION INTENT: "Buy 10 TCS"
        buy_match = re.search(r"buy\s+(\d+)\s+([a-zA-Z0-9_-]+)", msg_lower)
        if buy_match:
            qty = int(buy_match.group(1))
            sym = buy_match.group(2).upper()
            snap = get_asset_full_analysis(sym)
            m = snap["moreshwar_levels"]
            return {
                "reply": (
                    f"✅ **Execution Bridge Ready:** Queued **BUY {qty} shares of {sym}** @ ₹{snap['price']:,.2f} "
                    f"(Total: ₹{qty * snap['price']:,.2f}).\n"
                    f"• MOVIQ Target: **₹{m['target']:,.2f}** | Stop-Loss: **₹{m['stop_loss']:,.2f}** (R:R 1:{m['rr_ratio']})."
                ),
                "queued_trade": {
                    "symbol": sym,
                    "qty": qty,
                    "entry": snap["price"],
                    "target": m["target"],
                    "stop": m["stop_loss"],
                },
            }

        # 4. PORTFOLIO / EFFICIENCY / HOLDINGS CHECK
        if any(k in msg_lower for k in ["portfolio", "holding", "efficiency", "risk", "trap"]):
            total_inv = sum(h["qty"] * h["avg_price"] for h in holdings if h["qty"] > 0)
            total_cur = sum(h["qty"] * h["ltp"] for h in holdings if h["qty"] > 0)
            pnl = total_cur - total_inv
            traps = [h["symbol"] for h in holdings if h.get("fundamental_score", 70) <= 45 and h["qty"] > 0]
            winners = [h["symbol"] for h in holdings if h.get("fundamental_score", 70) >= 72 and h["qty"] > 0]
            trap_txt = f"⚠️ **Capital Traps Detected:** {', '.join(traps)} (Fundamental Score ≤ 45 — consider exiting)." if traps else "✅ **Zero Capital Traps:** All core holdings pass fundamental safety floors."
            return {
                "reply": (
                    f"📈 **Stocky Portfolio Efficiency & Risk Audit:**\n\n"
                    f"• **Invested Capital:** ₹{total_inv:,.2f} | **Current Value:** ₹{total_cur:,.2f} | **Net P&L:** ₹{pnl:+,.2f}\n"
                    f"• **High-Conviction Compounders:** {', '.join(winners) if winners else 'None'}\n"
                    f"• {trap_txt}"
                )
            }

        # 5. ANALYZE / EXPLAIN / TARGET SPECIFIC SYMBOL
        for sym in list(ASSET_UNIVERSE.keys()) + ["ONGC", "ITC", "SBIN", "WIPRO", "BAJFINANCE"]:
            if sym.lower() in msg_lower:
                snap = get_asset_full_analysis(sym)
                f = snap["fundamental_score"]
                p = snap["porters_score"]
                m = snap["moreshwar_levels"]
                d = snap["decision_matrix"]
                return {
                    "reply": (
                        f"🔍 **Stocky Deep-Dive Analysis: {sym} ({snap['name']})**\n\n"
                        f"• **LTP:** ₹{snap['price']:,.2f} ({snap['change_pct']:+.2f}%) | **Decision Matrix:** **{d['action']}** ({d['conviction']} Conviction)\n"
                        f"• **Total Fundamental Score:** **{f['total']} / 100** "
                        f"(Business: {f['business']}/40 · Moat: {f['moat']}/20 · Management: {f['management']}/20 · Risk: {f['risk']}/20)\n"
                        f"• **Porter's 5 Forces Moat:** **{p['total']} / 100** "
                        f"(Entrants: {p['entrants']} · Suppliers: {p['suppliers']} · Buyers: {p['buyers']} · Substitutes: {p['substitutes']} · Rivalry: {p['rivalry']})\n"
                        f"• **MOVIQ Dynamic Levels:** Entry **₹{m['entry_price']:,.2f}** | Target **₹{m['target']:,.2f}** | Stop-Loss **₹{m['stop_loss']:,.2f}**\n"
                        f"• **30D Quant Regression Target:** **₹{snap['projected_30d_target']:,.2f}** ({snap['regression_curve']})\n\n"
                        f"💡 **Verdict:** {f['nlp']} {d['rationale']}"
                    ),
                    "action_symbol": sym,
                }

        return {
            "reply": (
                "🤖 **Stocky AI Assistant Ready.** Try asking:\n"
                "• `Analyze RELIANCE` or `Explain TATAMOTORS`\n"
                "• `Compare HDFCBANK vs ICICIBANK`\n"
                "• `Invest 5 Lakh` (generates a score-weighted allocation table)\n"
                "• `Portfolio efficiency check` (audits your holdings for capital traps)\n"
                "• `Buy 10 TCS` (queues an order into the Zerodha Kite execution bridge)"
            )
        }
