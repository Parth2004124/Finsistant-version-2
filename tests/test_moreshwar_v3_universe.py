"""Automated Validation Tests for Moreshwar 3.0 Prompt 1 (PIT NSE Research Universe)."""

from __future__ import annotations

import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from finsistant_v2.m3 import DEFAULT_M3_DATA_DIR, DEFAULT_M3_DB_PATH, DEFAULT_M3_REPORTS_DIR, PITUniverseStore
from finsistant_v2.m3.collectors import parse_corporate_action_subject, parse_nse_date
from finsistant_v2.m3.liquidity_engine import recompute_corporate_action_adjustments
from finsistant_v2.m3.quality_engine import validate_raw_bar_fields
from finsistant_v2.m3.schema import connect_m3_db, init_m3_schema


class TestMoreshwarV3PITUniverse(unittest.TestCase):
    def test_corporate_action_parser_and_backward_adjustment_math(self) -> None:
        s1 = parse_corporate_action_subject("Face Value Split (Sub-Division) - From Rs 10/- Per Share To Rs 2/- Per Share")
        self.assertEqual(s1["action_type"], "SPLIT")
        self.assertAlmostEqual(s1["adj_factor"], 5.0, places=4)

        b1 = parse_corporate_action_subject("Bonus 1:1")
        self.assertEqual(b1["action_type"], "BONUS")
        self.assertAlmostEqual(b1["adj_factor"], 2.0, places=4)

        b2 = parse_corporate_action_subject("Bonus 3:2")
        self.assertAlmostEqual(b2["adj_factor"], 2.5, places=4)

        r1 = parse_corporate_action_subject("Rights 5:41 @ Premium Rs 109/-")
        self.assertEqual(r1["action_type"], "RIGHTS")
        self.assertEqual(r1["rights_premium_inr"], 109.0)

        d1 = parse_corporate_action_subject("Interim Dividend - Rs 12.50 Per Share")
        self.assertEqual(d1["action_type"], "DIVIDEND")
        self.assertEqual(d1["cash_dividend_inr"], 12.5)

        # Verify cumulative split + bonus adjustment preserves unadjusted close & scales adj_close accurately
        with tempfile.TemporaryDirectory() as td:
            dbp = Path(td) / "test.db"
            conn = connect_m3_db(dbp)
            init_m3_schema(conn)
            conn.execute(
                "INSERT INTO security_master (sec_id, isin, current_symbol, company_name, status, instrument_type, source_primary, retrieved_at) "
                "VALUES ('INE_TEST', 'INE_TEST', 'TESTCO', 'Test Co', 'ACTIVE', 'EQUITY', 'TEST', '2026-01-01')"
            )
            bars = [
                ("2022-01-03", "TESTCO", "EQ", "INE_TEST", 990, 1000, 1010, 990, 1000, 1000, 1000, 1000, 1e6, 10, 500, 50.0, 1.0, 1000, 1010, 990, 1000, None, None, "UNAVAILABLE", None, "", 1),
                ("2022-06-15", "TESTCO", "EQ", "INE_TEST", 200, 200, 205, 198, 202, 202, 202, 5000, 1e6, 50, 2500, 50.0, 1.0, 200, 205, 198, 202, None, None, "UNAVAILABLE", None, "", 1),
                ("2023-03-01", "TESTCO", "EQ", "INE_TEST", 101, 101, 104, 100, 102, 102, 102, 10000, 1e6, 80, 5000, 50.0, 1.0, 101, 104, 100, 102, None, None, "UNAVAILABLE", None, "", 1),
            ]
            conn.executemany(
                "INSERT INTO daily_bars VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                bars,
            )
            conn.executemany(
                "INSERT INTO corporate_actions (ca_id, sec_id, symbol, ex_date, action_type, adj_factor, subject, source_provider) VALUES (?,?,?,?,?,?,?,?)",
                [
                    ("ca1", "INE_TEST", "TESTCO", "2022-06-01", "SPLIT", 5.0, "Split 10->2", "TEST"),
                    ("ca2", "INE_TEST", "TESTCO", "2023-01-01", "BONUS", 2.0, "Bonus 1:1", "TEST"),
                ],
            )
            conn.commit()
            recompute_corporate_action_adjustments(conn)

            rows = conn.execute("SELECT trade_date, close, adj_factor, adj_close FROM daily_bars ORDER BY trade_date").fetchall()
            # Unadjusted close untouched
            self.assertEqual([float(r["close"]) for r in rows], [1000.0, 202.0, 102.0])
            # Cumulative factors: 10.0x before split+bonus, 2.0x between split & bonus, 1.0x after bonus
            self.assertAlmostEqual(rows[0]["adj_factor"], 10.0)
            self.assertAlmostEqual(rows[0]["adj_close"], 100.0)
            self.assertAlmostEqual(rows[1]["adj_factor"], 2.0)
            self.assertAlmostEqual(rows[1]["adj_close"], 101.0)
            self.assertAlmostEqual(rows[2]["adj_factor"], 1.0)
            self.assertAlmostEqual(rows[2]["adj_close"], 102.0)
            conn.close()

    def test_dq_engine_non_silent_validation(self) -> None:
        ok, fl1 = validate_raw_bar_fields(expected_date="2024-01-02", bar_date="2024-01-02", open_px=100.0, high_px=105.0, low_px=99.0, close_px=104.0, volume=10_000)
        self.assertTrue(ok)
        self.assertEqual(fl1, [])

        bad, fl2 = validate_raw_bar_fields(expected_date="2024-01-02", bar_date="2024-01-02", open_px=100.0, high_px=95.0, low_px=99.0, close_px=97.0, volume=10_000)
        self.assertFalse(bad)
        self.assertIn("INVALID_OHLC_INVARIANT", fl2)

        zv, fl3 = validate_raw_bar_fields(expected_date="2024-01-02", bar_date="2024-01-03", open_px=50.0, high_px=50.0, low_px=50.0, close_px=50.0, volume=0)
        self.assertTrue(zv)
        self.assertIn("DATE_MISMATCH", fl3)
        self.assertIn("ZERO_VOLUME_BAR", fl3)
        self.assertIn("STALE_FLAT_BAR", fl3)

    def test_real_pit_database_integrity_delisted_retention_and_symbol_continuity(self) -> None:
        self.assertTrue(DEFAULT_M3_DB_PATH.exists(), f"Missing M3 DB at {DEFAULT_M3_DB_PATH}")
        conn = connect_m3_db(DEFAULT_M3_DB_PATH)
        ver = conn.execute("SELECT row_counts_json FROM dataset_versions ORDER BY created_at DESC LIMIT 1").fetchone()
        self.assertIsNotNone(ver)
        rc = json.loads(ver["row_counts_json"])
        cal_cnt = conn.execute("SELECT COUNT(*) FROM trading_calendar").fetchone()[0]
        sec_cnt = conn.execute("SELECT COUNT(*) FROM security_master").fetchone()[0]
        ca_cnt = conn.execute("SELECT COUNT(*) FROM corporate_actions").fetchone()[0]
        self.assertGreaterEqual(cal_cnt, 1400)
        self.assertGreaterEqual(sec_cnt, 3000)
        self.assertGreaterEqual(int(rc.get("daily_bars_valid", 0)), 3_000_000)
        self.assertGreaterEqual(ca_cnt, 10_000)
        # Verify ETF separation in security_master
        etf_cnt = conn.execute("SELECT COUNT(*) FROM security_master WHERE instrument_type = 'ETF'").fetchone()[0]
        self.assertGreater(etf_cnt, 100)

        store = PITUniverseStore(DEFAULT_M3_DB_PATH)

        # 1. Delisted/merged security retention BEFORE merger/delisting, excluded AFTER
        u_2022 = store.get_pit_universe("2022-03-15", min_adv_20d_inr=50_000_000.0, exclude_etfs=True)
        syms_2022 = {x["symbol"] for x in u_2022}
        self.assertIn("HDFC", syms_2022)
        self.assertIn("MINDTREE", syms_2022)
        self.assertNotIn("LIQUIDBEES", syms_2022)
        self.assertNotIn("NIFTYBEES", syms_2022)
        # JIOFIN listed in Aug 2023 -> MUST NOT appear in 2022 PIT universe
        self.assertNotIn("JIOFIN", syms_2022)

        u_2025 = store.get_pit_universe("2025-03-17", min_adv_20d_inr=50_000_000.0, exclude_etfs=True)
        syms_2025 = {x["symbol"] for x in u_2025}
        self.assertNotIn("HDFC", syms_2025)
        self.assertNotIn("MINDTREE", syms_2025)
        self.assertIn("JIOFIN", syms_2025)

        # 2. Symbol change continuity: CADILAHC -> ZYDUSLIFE under same canonical sec_id
        r_pre = store.resolve_symbol_at_date("CADILAHC", "2021-06-15")
        r_post = store.resolve_symbol_at_date("ZYDUSLIFE", "2024-06-15")
        self.assertIsNotNone(r_pre)
        self.assertIsNotNone(r_post)
        self.assertEqual(r_pre["sec_id"], r_post["sec_id"])

        hist_stitched = store.get_security_history("ZYDUSLIFE", start_date="2021-01-01", end_date="2024-06-15", adjusted=True)
        self.assertGreater(len(hist_stitched), 700)
        self.assertIn("CADILAHC", {b["symbol"] for b in hist_stitched})
        self.assertIn("ZYDUSLIFE", {b["symbol"] for b in hist_stitched})
        self.assertLessEqual(hist_stitched[-1]["trade_date"], "2024-06-15")

        # 3. Real split adjustment check: IRCTC 5:1 split (exDate 2021-10-28) has no artificial -80% adj drop
        irctc = store.get_security_history("IRCTC", start_date="2021-10-20", end_date="2021-11-05", adjusted=True)
        for i in range(1, len(irctc)):
            ret = irctc[i]["eff_close"] / irctc[i - 1]["eff_close"] - 1.0
            self.assertGreater(ret, -0.35, f"Unadjusted split cliff detected in IRCTC at {irctc[i]['trade_date']}: {ret}")

        # 4. Pre-2024 official MCAP is NULL (not fabricated), Post-mid-2024 populated
        mcap_2021 = conn.execute("SELECT COUNT(*) FROM daily_bars WHERE trade_date='2021-03-15' AND mcap_inr IS NOT NULL").fetchone()[0]
        mcap_2025 = conn.execute("SELECT COUNT(*) FROM daily_bars WHERE trade_date='2025-03-17' AND mcap_inr IS NOT NULL").fetchone()[0]
        self.assertEqual(mcap_2021, 0)
        self.assertGreater(mcap_2025, 1500)
        conn.close()

    def test_m3_deliverables_and_frozen_m2_protection(self) -> None:
        for p in (
            DEFAULT_M3_DATA_DIR / "dataset_version_registry.json",
            DEFAULT_M3_DATA_DIR / "security_master.json.gz",
            DEFAULT_M3_DATA_DIR / "corporate_actions.json.gz",
            DEFAULT_M3_REPORTS_DIR / "DATASET_DOCUMENTATION.md",
            DEFAULT_M3_REPORTS_DIR / "DATA_COVERAGE_REPORT.md",
            DEFAULT_M3_REPORTS_DIR / "data_coverage_report.json",
            DEFAULT_M3_REPORTS_DIR / "data_quality_report.json",
            DEFAULT_M3_REPORTS_DIR / "known_limitations.json",
        ):
            self.assertTrue(p.exists(), f"Missing Prompt 1 deliverable: {p}")

        # Verify frozen M2.1 & M2.2 artifacts still intact
        root = DEFAULT_M3_DATA_DIR.parent.parent
        m21 = json.loads((root / "data" / "moreshwar_v2_1_experiment_report.json").read_text(encoding="utf-8"))
        m22 = json.loads((root / "reports" / "moreshwar_2_2" / "integrated_final_report.json").read_text(encoding="utf-8"))
        self.assertFalse(m21["promoted_to_production"])
        self.assertFalse(m22["promoted_to_production"])


if __name__ == "__main__":
    unittest.main()
