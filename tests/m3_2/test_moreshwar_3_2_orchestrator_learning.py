"""Unit tests for Moreshwar 3.2 — Full-Universe Orchestrator Scanner & Per-Trade Learning Ledger."""

from __future__ import annotations

from pathlib import Path
import sqlite3
import tempfile
import unittest

import numpy as np

from finsistant_v2.m3_1.prospective_paper_trading_engine import PaperOnlyExecutionGuard, PaperTradingSafetyError
from finsistant_v2.m3_2.learning_ledger import OrchestratorLearningLedger
from finsistant_v2.m3_2.orchestrator_scanner import (
    MISSION_TARGET_SESSIONS,
    ORCH_STRATEGY_ID,
    OrchestratorScanConfig,
    OrchestratorUniverseScanner,
)


def _synth_history(start_px: float, drift: float, n: int = 80, end_date: str = "2026-11-20") -> tuple[list[str], list[float]]:
    dates = [f"2026-{(i // 28) + 8:02d}-{(i % 28) + 1:02d}" for i in range(n - 1)] + [end_date]
    closes = [round(start_px * ((1.0 + drift) ** i), 2) for i in range(n)]
    return dates, closes


class TestMoreshwar32OrchestratorAndLearning(unittest.TestCase):
    def test_paper_only_guard_admits_orch_and_blocks_family_a_and_live(self) -> None:
        g_m31 = PaperOnlyExecutionGuard()
        with self.assertRaises(PaperTradingSafetyError):
            g_m31.assert_paper_only(strategy_ids=[ORCH_STRATEGY_ID])

        g_m32 = PaperOnlyExecutionGuard(extra_admitted_strategy_ids=(ORCH_STRATEGY_ID,))
        g_m32.assert_paper_only(strategy_ids=[ORCH_STRATEGY_ID])
        with self.assertRaises(PaperTradingSafetyError):
            g_m32.assert_paper_only(strategy_ids=["STRAT_A_CS_MOMENTUM"])
        with self.assertRaises(PaperTradingSafetyError):
            g_m32.assert_paper_only(live_trading_enabled=True)

    def test_scanner_excludes_price_discrepancy_and_ranks_approved_buys(self) -> None:
        dates, c_up = _synth_history(100.0, 0.004, n=70, end_date="2026-11-20")
        _, c_flat = _synth_history(200.0, 0.003, n=70, end_date="2026-11-20")
        histories = {
            "SEC_GOOD": {
                "symbol": "GOODCO", "sector": "IT", "industry": "Software",
                "dates": dates, "open": c_up, "high": [x * 1.01 for x in c_up],
                "low": [x * 0.99 for x in c_up], "close": c_up, "volume": [1_000_000.0] * 70,
            },
            "SEC_DISC": {
                "symbol": "DISCCO", "sector": "IT", "industry": "Software",
                "dates": dates, "open": c_flat, "high": [x * 1.01 for x in c_flat],
                "low": [x * 0.99 for x in c_flat], "close": c_flat, "volume": [1_000_000.0] * 70,
            },
        }
        eq_bars = {
            "SEC_GOOD": {"sec_id": "SEC_GOOD", "symbol": "GOODCO", "sector": "IT", "close": c_up[-1], "adv_20d_inr": 80_000_000.0, "vol_20d": 0.014, "valid": True},
            "SEC_DISC": {"sec_id": "SEC_DISC", "symbol": "DISCCO", "sector": "IT", "close": c_flat[-1], "adv_20d_inr": 80_000_000.0, "vol_20d": 0.014, "valid": True},
        }

        def fake_live(sym: str) -> dict:
            px = c_up[-1] if sym == "GOODCO" else c_flat[-1] * 1.25  # 25% discrepancy -> MUST exclude
            return {
                "symbol": sym, "name": sym, "price": px, "sma_20": px * 0.95, "ema_12": px * 0.99,
                "ema_26": px * 0.96, "rsi_14": 60.0, "macd": 2.0, "macd_signal": 1.0,
                "projected_30d_target": px * 1.08, "regression_curve": "Upward Curve",
                "fundamental_score": {"total": 82, "business": 34, "moat": 16, "management": 16, "risk": 16, "industry": "IT", "nlp": "Strong"},
                "porters_score": {"total": 78},
                "moreshwar_levels": {"entry_price": px * 0.98, "stop_loss": px * 0.91, "target": px * 1.18, "rr_ratio": 2.0},
                "decision_matrix": {"action": "BUY NOW", "rationale": "High quality"},
                "fundamental_source_category": "YAHOO_LIVE_FEED", "price_source_category": "YAHOO_LIVE_FEED",
                "missing_fields": [],
            }

        scanner = OrchestratorUniverseScanner(OrchestratorScanConfig(shortlist_size=5, max_workers=1), analysis_fn=fake_live)
        res = scanner.scan(trading_date="2026-11-20", equity_histories=histories, eq_bars=eq_bars)
        self.assertEqual(res["stages"]["agents_ok"], 1)
        self.assertEqual(res["stages"]["agents_excluded"], 1)
        self.assertEqual([c["symbol"] for c in res["candidates"]], ["GOODCO"])
        self.assertEqual(res["candidates"][0]["strategy_id"], ORCH_STRATEGY_ID)

    def test_learning_ledger_learns_from_passed_and_failed_trades(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tdir = Path(tmp)
            ldb = tdir / "learn.db"
            pdb = tdir / "paper.db"
            ledger = OrchestratorLearningLedger(ldb)

            # Record a scan with 4 stocks: 2 high-Fund wins (normal RSI), 2 overbought low-Fund losses
            scan = {
                "trading_date": "2026-11-20",
                "universe_tech": [
                    {"sec_id": "S1", "symbol": "WIN1", "tech_conf": 0.72, "rank_key": 1.2, "close": 100.0, "rsi_14": 58.0, "sector": "Banking"},
                    {"sec_id": "S2", "symbol": "WIN2", "tech_conf": 0.70, "rank_key": 1.1, "close": 100.0, "rsi_14": 60.0, "sector": "Banking"},
                    {"sec_id": "S3", "symbol": "LOSS1", "tech_conf": 0.92, "rank_key": 1.5, "close": 100.0, "rsi_14": 82.0, "sector": "Metals"},
                    {"sec_id": "S4", "symbol": "LOSS2", "tech_conf": 0.90, "rank_key": 1.4, "close": 100.0, "rsi_14": 79.0, "sector": "Metals"},
                ],
                "shortlist": [
                    {"sec_id": "S1", "symbol": "WIN1", "tech_rank": 3, "status": "AGENTS_OK", "tech_conf": 0.72, "fund_conf": 0.90, "fundamental_total": 88, "porters_total": 85, "conviction": 0.82, "exec_action": "BUY", "eod_close": 100.0, "rsi_14": 58.0, "sector": "Banking"},
                    {"sec_id": "S2", "symbol": "WIN2", "tech_rank": 4, "status": "AGENTS_OK", "tech_conf": 0.70, "fund_conf": 0.88, "fundamental_total": 86, "porters_total": 82, "conviction": 0.80, "exec_action": "BUY", "eod_close": 100.0, "rsi_14": 60.0, "sector": "Banking"},
                    {"sec_id": "S3", "symbol": "LOSS1", "tech_rank": 1, "status": "AGENTS_OK", "tech_conf": 0.92, "fund_conf": 0.45, "fundamental_total": 48, "porters_total": 42, "conviction": 0.65, "exec_action": "BUY", "eod_close": 100.0, "rsi_14": 82.0, "sector": "Metals"},
                    {"sec_id": "S4", "symbol": "LOSS2", "tech_rank": 2, "status": "AGENTS_OK", "tech_conf": 0.90, "fund_conf": 0.48, "fundamental_total": 50, "porters_total": 45, "conviction": 0.66, "exec_action": "BUY", "eod_close": 100.0, "rsi_14": 79.0, "sector": "Metals"},
                ],
            }
            ledger.record_scan(scan, {"WIN1", "WIN2", "LOSS1", "LOSS2"})

            # Populate closed trades in mock paper DB
            pconn = sqlite3.connect(str(pdb))
            pconn.execute(
                """CREATE TABLE positions (
                    position_id TEXT PRIMARY KEY, sec_id TEXT, symbol TEXT, sector TEXT,
                    status TEXT, entry_date TEXT, exit_date TEXT, cost_basis_inr REAL,
                    bars_held INTEGER, mae_pct REAL, mfe_pct REAL, dividends_received_inr REAL,
                    realized_pnl_inr REAL, exit_reason TEXT
                )"""
            )
            pconn.executemany(
                "INSERT INTO positions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                [
                    ("P1", "S1", "WIN1", "Banking", "CLOSED", "2026-11-23", "2026-12-04", 10000.0, 9, -1.2, 9.5, 0.0, 820.0, "TRAILING_STOP"),
                    ("P2", "S2", "WIN2", "Banking", "CLOSED", "2026-11-23", "2026-12-05", 10000.0, 10, -0.8, 8.2, 0.0, 690.0, "TRAILING_STOP"),
                    ("P3", "S3", "LOSS1", "Metals", "CLOSED", "2026-11-23", "2026-11-27", 10000.0, 4, -9.2, 0.4, 0.0, -910.0, "STOP_LOSS"),
                    ("P4", "S4", "LOSS2", "Metals", "CLOSED", "2026-11-23", "2026-11-28", 10000.0, 5, -8.7, 0.6, 0.0, -840.0, "STOP_LOSS"),
                ],
            )
            pconn.commit()
            pconn.close()

            synced = ledger.sync_closed_trades(pdb)
            self.assertEqual(synced, 4)

            upd = ledger.update_weights("2026-12-05")
            # TechBot over-scored the two failing overbought trades while FundBot correctly predicted the wins!
            self.assertLess(upd["w_tech"], 0.45)
            self.assertGreater(upd["w_fund"], 0.55)
            # Low-conviction trades failed (0% WR) while high-conviction trades passed (100% WR) -> min_conviction rises!
            self.assertGreater(upd["min_conviction"], 0.60)
            # Overbought RSI (>72) failed badly -> RSI overbought penalty activates!
            self.assertGreater(upd["rsi_overbought_penalty"], 0.0)
            # Sector bias favors Banking over Metals!
            self.assertGreater(upd["sector_adjustments"].get("Banking", 0.0), 0.0)
            self.assertLess(upd["sector_adjustments"].get("Metals", 0.0), 0.0)

            mission = ledger.build_mission_status({"risk_metrics": {"sample_sessions": 12, "current_nav_inr": 105400.0, "initial_capital_inr": 100000.0, "cumulative_return_pct": 5.4, "max_drawdown_pct": 2.1}})
            self.assertEqual(mission["target_sessions"], MISSION_TARGET_SESSIONS)
            self.assertEqual(mission["passed_trades"], 2)
            self.assertEqual(mission["failed_trades"], 2)
            self.assertFalse(mission["real_money_ready"])
            self.assertFalse(mission["live_trading_enabled"])

    def test_scoring_rejects_unavailable_fundamentals_no_fake_defaults(self) -> None:
        from finsistant_v2.scoring_engine import calculate_fundamental_score, calculate_porters_score

        bad_meta = {"fundamentals_status": "unavailable", "fundamental_source_category": "UNAVAILABLE"}
        with self.assertRaises(ValueError):
            calculate_fundamental_score(bad_meta)
        with self.assertRaises(ValueError):
            calculate_porters_score(bad_meta)

        partial_meta = {"name": "TEST", "pe": 0.0, "missing_fields": ["pe", "roe", "opm", "sales_growth", "profit_growth", "mcap_cr"]}
        with self.assertRaises(ValueError):
            calculate_fundamental_score(partial_meta, strict=True)

    def test_scanner_retries_then_fails_upfront_on_data_failure(self) -> None:
        dates, c_up = _synth_history(100.0, 0.004, n=70, end_date="2026-11-20")
        histories = {
            "SEC_1": {
                "symbol": "FAILCO", "sector": "IT", "industry": "Software",
                "dates": dates, "open": c_up, "high": [x * 1.01 for x in c_up],
                "low": [x * 0.99 for x in c_up], "close": c_up, "volume": [1_000_000.0] * 70,
            },
        }
        eq_bars = {
            "SEC_1": {"sec_id": "SEC_1", "symbol": "FAILCO", "sector": "IT", "close": c_up[-1], "adv_20d_inr": 80_000_000.0, "vol_20d": 0.014, "valid": True},
        }
        calls = {"n": 0}

        def failing_live(_sym: str) -> dict:
            calls["n"] += 1
            raise RuntimeError("NSE MCP unavailable")

        scanner = OrchestratorUniverseScanner(
            OrchestratorScanConfig(shortlist_size=5, max_workers=1, max_retries_per_symbol=2),
            analysis_fn=failing_live,
        )
        with self.assertRaises(RuntimeError, msg="ORCHESTRATOR_SCAN_FAILED_UPFRONT"):
            scanner.scan(trading_date="2026-11-20", equity_histories=histories, eq_bars=eq_bars)
        self.assertEqual(calls["n"], 2)

    def test_m32_overrides_blocked_for_m31_and_active_0_90_10_for_m32(self) -> None:
        from finsistant_v2.m3_1.prospective_paper_trading_engine import ProspectivePaperTradingEngine
        from finsistant_v2.m3_2.orchestrated_session import M32_ALPHA_REGIME_POLICY, M32_EXEC_CFG_OVERRIDE, OrchestratedSessionRunner

        with tempfile.TemporaryDirectory() as tmp:
            tdir = Path(tmp)
            with self.assertRaises(PaperTradingSafetyError):
                ProspectivePaperTradingEngine(
                    db_path=tdir / "m31_bad.db",
                    regime_policy_override=M32_ALPHA_REGIME_POLICY,
                )
            runner = OrchestratedSessionRunner(
                db_path=tdir / "m32.db",
                learning_db_path=tdir / "learn.db",
                reports_dir=tdir / "rep",
                scan_dir=tdir / "scan",
            )
            bull = [r for r in runner.orch.engine._effective_regime_policy.regime_rules if r[0] == "BULL_TREND"][0]
            self.assertEqual((bull[1], bull[2], bull[3]), (0.0, 90.0, 10.0))
            self.assertEqual(M32_EXEC_CFG_OVERRIDE["sweep_unused_tactical_to_core"], False)


if __name__ == "__main__":
    unittest.main()
