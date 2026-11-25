"""Tests for Adversarial Bear Prosecutor (no-mercy stress testing) and Per-Rupee Learning Loop."""

from __future__ import annotations

from pathlib import Path
import sqlite3
import tempfile
import unittest

from finsistant_v2.adversarial_prosecutor import prosecute_trade_candidate
from finsistant_v2.agents import (
    AdversarialProsecutorAgent,
    AgentRunner,
    ExecutionAgent,
    FundamentalAnalyst,
    TechnicalAnalyst,
)


def _make_candles(n: int = 45, start_px: float = 1000.0, step: float = 4.0, *, shock_last: float = 0.0) -> list[dict]:
    out = []
    for i in range(n):
        c = round(start_px + i * step, 2)
        if i == n - 1 and shock_last != 0.0:
            c = round(c * (1.0 + shock_last), 2)
        out.append(
            {
                "date": f"2026-10-{(i % 28) + 1:02d}",
                "open": round(c * 0.995, 2),
                "high": round(c * 1.012, 2),
                "low": round(c * 0.988, 2),
                "close": c,
                "volume": 500_000,
            }
        )
    return out


def _make_snapshot(
    *,
    symbol: str = "RELIANCE",
    price: float = 1350.0,
    pe: float = 24.0,
    roe: float = 21.0,
    roce: float = 23.0,
    opm: float = 22.0,
    sales_g: float = 19.0,
    profit_g: float = 24.0,
    dte: float = 0.25,
    mcap_cr: float = 180000.0,
    rsi: float = 59.0,
    sma_20: float = 1300.0,
    macd: float = 12.0,
    macd_signal: float = 7.0,
    stop_loss: float = 1260.0,
    target: float = 1560.0,
    rr_ratio: float = 2.33,
    missing_fields: list[str] | None = None,
    ret_1y: float = 22.0,
    candles: list[dict] | None = None,
) -> dict:
    return {
        "symbol": symbol,
        "name": f"{symbol} Ltd",
        "sector": "Energy & Petrochem",
        "price": price,
        "sma_20": sma_20,
        "ema_12": price * 0.99,
        "ema_26": price * 0.97,
        "rsi_14": rsi,
        "macd": macd,
        "macd_signal": macd_signal,
        "bb_upper": price * 1.05,
        "projected_30d_target": target,
        "regression_curve": "Upward Curve",
        "ret_1y": ret_1y,
        "missing_fields": list(missing_fields or []),
        "fundamental_score": {
            "total": 84,
            "business": 34,
            "moat": 17,
            "management": 16,
            "risk": 17,
            "industry": "POWER",
            "nlp": "Strong cash flow and ROCE.",
            "ratios": {
                "pe": pe,
                "roe": roe,
                "roce": roce,
                "opm": opm,
                "sales_growth": sales_g,
                "profit_growth": profit_g,
                "debt_to_equity": dte,
                "mcap_cr": mcap_cr,
            },
        },
        "porters_score": {"total": 82},
        "moreshwar_levels": {
            "entry_price": round(price * 0.995, 2),
            "stop_loss": stop_loss,
            "target": target,
            "rr_ratio": rr_ratio,
        },
        "decision_matrix": {"action": "BUY NOW", "rationale": "Strong fundamentals and favorable trend."},
        "candles": candles if candles is not None else _make_candles(45, start_px=1200.0, step=3.3),
    }


class TestAdversarialProsecutor(unittest.TestCase):
    def test_heavy_leverage_and_overbought_kills_buy_decision(self) -> None:
        # Even when decision_matrix tries to say BUY NOW, BearProsecutor must show zero mercy and KILL it.
        bad_reliance = _make_snapshot(
            symbol="RELIANCE",
            pe=62.0,
            roe=11.5,
            roce=8.2,
            opm=6.0,
            sales_g=-4.0,
            profit_g=-9.0,
            dte=2.45,
            rsi=81.0,
            sma_20=1180.0,
            stop_loss=1340.0,  # tiny stop inside daily ATR noise
            target=1395.0,
            rr_ratio=1.22,
        )
        runner = AgentRunner(
            agents=[TechnicalAnalyst(), FundamentalAnalyst(), AdversarialProsecutorAgent(), ExecutionAgent()]
        )
        res = runner.run_pipeline(symbol="RELIANCE", asset_analysis=bad_reliance, capital=100_000.0, active_trades=0)
        self.assertEqual(res.final_action, "HOLD")
        self.assertEqual(res.quantity, 0)
        self.assertEqual(res.order_receipt["status"], "KILLED")
        pros = res.order_receipt["prosecution"]
        self.assertEqual(pros["verdict"], "KILLED_BY_ADVERSARY")
        self.assertGreaterEqual(pros["fatal_count"], 1)
        self.assertGreaterEqual(pros["attack_count"], 5)
        self.assertIn("LEVERAGE_SOLVENCY_BOMB", pros["attack_ids"])

    def test_data_opacity_fatal_veto_without_mercy(self) -> None:
        opaque_snap = _make_snapshot(
            symbol="RELIANCE",
            missing_fields=["roe", "roce", "opm", "sales_growth", "pe"],
        )
        report = prosecute_trade_candidate(
            symbol="RELIANCE",
            snap=opaque_snap,
            tech_conf=0.80,
            fund_conf=0.78,
            rr_ratio=2.2,
            stop_loss=1260.0,
            target=1560.0,
            price=1350.0,
        )
        self.assertFalse(report["outlasted"])
        self.assertEqual(report["verdict"], "KILLED_BY_ADVERSARY")
        self.assertIn("DATA_OPACITY_PROVENANCE", report["attack_ids"])
        self.assertEqual(report["size_multiplier"], 0.0)

    def test_fortress_candidate_outlasts_adversarial_stress_test(self) -> None:
        fortress_snap = _make_snapshot(symbol="RELIANCE")
        runner = AgentRunner(
            agents=[TechnicalAnalyst(), FundamentalAnalyst(), AdversarialProsecutorAgent(), ExecutionAgent()]
        )
        res = runner.run_pipeline(symbol="RELIANCE", asset_analysis=fortress_snap, capital=100_000.0, active_trades=0)
        self.assertEqual(res.final_action, "BUY")
        self.assertGreater(res.quantity, 0)
        pros = res.order_receipt["prosecution"]
        self.assertTrue(pros["outlasted"])
        self.assertEqual(pros["verdict"], "OUTLASTED")
        self.assertGreaterEqual(pros["survival_score"], pros["min_survival_score"])

    def test_rupee_learning_escalates_attack_damage_after_real_losses(self) -> None:
        # Learned attack_multipliers from previous lost rupees must deal strictly more damage to future trades.
        stressed_snap = _make_snapshot(
            symbol="RELIANCE",
            pe=34.0,
            profit_g=11.0,
            rsi=76.0,
            sma_20=1230.0,
            stop_loss=1290.0,
            target=1490.0,
            rr_ratio=1.60,
        )
        base_pros = prosecute_trade_candidate(
            symbol="RELIANCE",
            snap=stressed_snap,
            tech_conf=0.78,
            fund_conf=0.72,
            rr_ratio=1.60,
            stop_loss=1290.0,
            target=1490.0,
            price=1350.0,
            attack_multipliers={},
        )
        hardened_pros = prosecute_trade_candidate(
            symbol="RELIANCE",
            snap=stressed_snap,
            tech_conf=0.78,
            fund_conf=0.72,
            rr_ratio=1.60,
            stop_loss=1290.0,
            target=1490.0,
            price=1350.0,
            attack_multipliers={"TECHNICAL_BLOWOFF_TRAP": 1.85, "VALUATION_EXTREME_TRAP": 1.60},
        )
        self.assertGreater(hardened_pros["total_damage_pts"], base_pros["total_damage_pts"])
        self.assertLess(hardened_pros["survival_score"], base_pros["survival_score"])


class TestPerRupeeLearningLedger(unittest.TestCase):
    def test_every_rupee_invested_updates_roic_and_escalates_attack_multipliers(self) -> None:
        from finsistant_v2.m3_2.learning_ledger import OrchestratorLearningLedger

        with tempfile.TemporaryDirectory() as tmp:
            tdir = Path(tmp)
            ldb = tdir / "learn.db"
            pdb = tdir / "paper.db"
            ledger = OrchestratorLearningLedger(ldb)

            scan = {
                "trading_date": "2026-11-24",
                "universe_tech": [
                    {"sec_id": "S_REL", "symbol": "RELIANCE", "tech_conf": 0.88, "rank_key": 1.3, "close": 1350.0, "rsi_14": 77.0, "sector": "Energy"},
                    {"sec_id": "S_TCS", "symbol": "TCS", "tech_conf": 0.68, "rank_key": 1.1, "close": 3900.0, "rsi_14": 58.0, "sector": "IT"},
                ],
                "shortlist": [
                    {
                        "sec_id": "S_REL", "symbol": "RELIANCE", "tech_rank": 1, "status": "AGENTS_OK",
                        "tech_conf": 0.88, "fund_conf": 0.62, "fundamental_total": 68, "porters_total": 66,
                        "conviction": 0.74, "exec_action": "BUY", "eod_close": 1350.0, "rsi_14": 77.0, "sector": "Energy",
                        "prosecution_verdict": "OUTLASTED", "survival_score": 56.0,
                        "attack_ids": ["TECHNICAL_BLOWOFF_TRAP", "VALUATION_EXTREME_TRAP"],
                    },
                    {
                        "sec_id": "S_TCS", "symbol": "TCS", "tech_rank": 2, "status": "AGENTS_OK",
                        "tech_conf": 0.68, "fund_conf": 0.86, "fundamental_total": 85, "porters_total": 84,
                        "conviction": 0.78, "exec_action": "BUY", "eod_close": 3900.0, "rsi_14": 58.0, "sector": "IT",
                        "prosecution_verdict": "OUTLASTED", "survival_score": 82.0,
                        "attack_ids": [],
                    },
                ],
            }
            ledger.record_scan(scan, {"RELIANCE", "TCS"})

            pconn = sqlite3.connect(str(pdb))
            pconn.execute(
                """CREATE TABLE positions (
                    position_id TEXT PRIMARY KEY, sec_id TEXT, symbol TEXT, sector TEXT,
                    status TEXT, entry_date TEXT, exit_date TEXT, cost_basis_inr REAL,
                    bars_held INTEGER, mae_pct REAL, mfe_pct REAL, dividends_received_inr REAL,
                    realized_pnl_inr REAL, exit_reason TEXT
                )"""
            )
            # Invest ₹20,000 in RELIANCE (loses ₹2,400 = -12%) while flagged with TECHNICAL_BLOWOFF_TRAP,
            # and ₹2,000 in TCS (wins ₹120 = +6%). Total invested = ₹22,000, net = -₹2,280.
            pconn.executemany(
                "INSERT INTO positions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                [
                    ("P_REL", "S_REL", "RELIANCE", "Energy", "CLOSED", "2026-11-25", "2026-12-02", 20000.0, 5, -13.0, 1.5, 0.0, -2400.0, "STOP_LOSS"),
                    ("P_TCS", "S_TCS", "TCS", "IT", "CLOSED", "2026-11-25", "2026-12-03", 2000.0, 6, -0.5, 7.0, 0.0, 120.0, "TRAILING_STOP"),
                ],
            )
            pconn.commit()
            pconn.close()

            synced = ledger.sync_closed_trades(pdb)
            self.assertEqual(synced, 2)
            upd = ledger.update_weights("2026-12-03")
            ct = upd["learning"]["closed_trades"]

            self.assertAlmostEqual(ct["total_rupees_invested_inr"], 22000.0, places=2)
            self.assertAlmostEqual(ct["total_realized_pnl_inr"], -2280.0, places=2)
            self.assertLess(ct["rupee_roic_pct"], 0.0)
            self.assertGreater(ct["total_rupees_at_risk_inr"], 2000.0)
            # Both attacks present when ₹2,400 was lost must escalate > 1.0x
            self.assertGreater(upd["attack_multipliers"].get("TECHNICAL_BLOWOFF_TRAP", 1.0), 1.05)
            self.assertGreater(upd["attack_multipliers"].get("VALUATION_EXTREME_TRAP", 1.0), 1.05)
            # Marginal survival (56.0 < 62.0) lost ₹2,400 -> min_survival_score must ratchet upward
            self.assertGreater(upd["min_survival_score"], 52.0)


if __name__ == "__main__":
    unittest.main()
