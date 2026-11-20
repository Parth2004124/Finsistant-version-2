"""Automated Validation Tests for Moreshwar 3.0 Prompt 2 (Multi-Factor Alpha Discovery)."""

from __future__ import annotations

import json
import unittest

import numpy as np

from finsistant_v2.m3.factor_research import (
    _benjamini_hochberg,
    _compute_forward_returns_next_open,
    _newey_west_t,
    _rankdata_avg_1d,
)
from finsistant_v2.m3.factors import FACTOR_SPECS, REJECTED_NON_PIT_FACTORS, compute_all_m3_factors
from finsistant_v2.m3.schema import DEFAULT_M3_REPORTS_DIR


def _make_synthetic_panel(t_len: int = 280, n_len: int = 12) -> dict:
    rng = np.random.default_rng(7)
    dates = np.array([f"2023-01-{(i % 28) + 1:02d}" for i in range(t_len)], dtype=object)
    # Create clean sequential business-like unique dates
    from datetime import date, timedelta
    d0 = date(2023, 1, 2)
    ds = []
    cur = d0
    while len(ds) < t_len:
        if cur.weekday() < 5:
            ds.append(cur.isoformat())
        cur += timedelta(days=1)
    dates = np.array(ds, dtype=object)

    rets = rng.normal(0.0006, 0.015, size=(t_len, n_len))
    adj_c = 100.0 * np.cumprod(1.0 + rets, axis=0)
    adj_o = adj_c * (1.0 + rng.normal(0.0, 0.002, size=(t_len, n_len)))
    adj_h = np.maximum(adj_o, adj_c) * 1.01
    adj_l = np.minimum(adj_o, adj_c) * 0.99
    n50 = 18000.0 * np.cumprod(1.0 + rng.normal(0.0004, 0.009, size=t_len))
    sectors = np.array(["IT", "IT", "IT", "IT", "Banking", "Banking", "Banking", "Banking", "Auto", "Auto", "Auto", "Auto"], dtype=object)
    return {
        "dates": dates,
        "sec_ids": np.array([f"S{i}" for i in range(n_len)], dtype=object),
        "current_symbols": np.array([f"SYM{i}" for i in range(n_len)], dtype=object),
        "sectors": sectors,
        "adj_open": adj_o.astype(np.float32),
        "adj_high": adj_h.astype(np.float32),
        "adj_low": adj_l.astype(np.float32),
        "adj_close": adj_c.astype(np.float32),
        "close_px": adj_c.astype(np.float32),
        "volume": np.full((t_len, n_len), 500_000.0),
        "turnover_inr": adj_c * 500_000.0,
        "delivery_pct": rng.uniform(35.0, 75.0, size=(t_len, n_len)).astype(np.float32),
        "mcap_inr": adj_c * 1e7,
        "adv_20d_inr": np.full((t_len, n_len), 1e8),
        "adv_60d_inr": np.full((t_len, n_len), 1e8),
        "div_inr_ex": np.zeros((t_len, n_len), dtype=np.float32),
        "is_pit_elig": np.ones((t_len, n_len), dtype=bool),
        "is_valid_eq": np.ones((t_len, n_len), dtype=bool),
        "nifty50_close": n50,
    }


class TestMoreshwarV3Factors(unittest.TestCase):
    def test_zero_lookahead_across_all_31_factors(self) -> None:
        pnl1 = _make_synthetic_panel(280, 12)
        pnl2 = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in pnl1.items()}
        t_cut = 260
        # Corrupt ALL future prices/volumes/sectors after t_cut
        for k in ("adj_open", "adj_high", "adj_low", "adj_close", "close_px", "volume", "turnover_inr", "delivery_pct", "mcap_inr"):
            pnl2[k][t_cut + 1 :] = pnl2[k][t_cut + 1 :] * 5.0 + 999.0
        pnl2["nifty50_close"][t_cut + 1 :] = 100.0

        f1 = compute_all_m3_factors(pnl1)
        f2 = compute_all_m3_factors(pnl2)
        self.assertEqual(len(f1), len(FACTOR_SPECS))
        for spec in FACTOR_SPECS:
            fid = spec.factor_id
            a = f1[fid][: t_cut + 1]
            b = f2[fid][: t_cut + 1]
            np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-6, equal_nan=True, err_msg=f"Look-ahead leak in {fid}")

    def test_forward_returns_align_strictly_t_plus_1_open_to_t_plus_h_close(self) -> None:
        pnl = _make_synthetic_panel(30, 2)
        pnl["adj_open"][1, 0] = 100.0
        pnl["adj_close"][5, 0] = 112.0
        fwd = _compute_forward_returns_next_open(pnl, (5,))
        # At signal t=0, entry is t+1=1 open (100.0), exit is t+5=5 close (112.0) -> +0.12
        self.assertAlmostEqual(float(fwd[5][0, 0]), 0.12, places=4)
        # Tail window where t+5 >= 30 must be NaN
        self.assertTrue(np.isnan(fwd[5][-1, 0]))

    def test_statistical_helpers_fdr_nw_and_ranks(self) -> None:
        rk = _rankdata_avg_1d(np.array([10.0, 20.0, 20.0, 30.0]))
        np.testing.assert_allclose(rk, [1.0, 2.5, 2.5, 4.0])

        q = _benjamini_hochberg([0.001, 0.02, 0.50, 0.90])
        self.assertLessEqual(q[0], q[1])
        self.assertLessEqual(q[1], q[2])
        self.assertLessEqual(q[2], q[3])
        self.assertAlmostEqual(q[0], 0.004)

        t_hac, p_hac = _newey_west_t(np.array([0.05, 0.06, 0.04, 0.05, 0.07, 0.05, 0.06, 0.04]), lag=1)
        self.assertGreater(t_hac, 5.0)
        self.assertLess(p_hac, 0.001)

    def test_factor_research_deliverables_if_generated(self) -> None:
        rep_dir = DEFAULT_M3_REPORTS_DIR / "factor_research"
        rep_json = rep_dir / "factor_research_report.json"
        if not rep_json.exists():
            self.skipTest("Factor research report not yet run")
        data = json.loads(rep_json.read_text(encoding="utf-8"))
        self.assertEqual(data["total_factors_tested"], len(FACTOR_SPECS))
        self.assertGreaterEqual(data["validated_count"] + data["rejected_count"], len(FACTOR_SPECS))
        # Ensure non-PIT fundamental factors explicitly disclosed as rejected
        rej_ids = {r["factor_id"] for r in data["rejected_factors"]}
        for rnp in REJECTED_NON_PIT_FACTORS:
            self.assertIn(rnp["factor_id"], rej_ids)


if __name__ == "__main__":
    unittest.main()
