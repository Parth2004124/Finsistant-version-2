"""Corporate-Action Adjustment, Causal Liquidity Eligibility & PIT Query Engine for Moreshwar 3.0."""

from __future__ import annotations

from collections import defaultdict
import math
from pathlib import Path
import sqlite3
from typing import Any

from .schema import DEFAULT_M3_DB_PATH, connect_m3_db


def recompute_corporate_action_adjustments(conn: sqlite3.Connection) -> dict[str, int]:
    """Apply backward split/bonus adjustment factors strictly chronologically per sec_id.

    Raw `open/high/low/close` remain unadjusted NSE exchange values; `adj_factor` and
    `adj_open/high/low/close` normalize pre-exDate history cleanly.
    """
    ca_by_sec: dict[str, list[tuple[str, float]]] = defaultdict(list)
    seen_ev: set[tuple[str, str, str, float]] = set()
    for row in conn.execute(
        """
        SELECT sec_id, ex_date, action_type, adj_factor
        FROM corporate_actions
        WHERE action_type IN ('SPLIT', 'BONUS') AND adj_factor > 1.0001
        ORDER BY ex_date ASC
        """
    ):
        key = (row["sec_id"], row["ex_date"], row["action_type"], round(float(row["adj_factor"]), 4))
        if key in seen_ev:
            continue
        seen_ev.add(key)
        ca_by_sec[row["sec_id"]].append((row["ex_date"], float(row["adj_factor"])))

    # Reset base unadjusted = adjusted first for securities with no splits/bonuses
    conn.execute(
        "UPDATE daily_bars SET adj_factor = 1.0, adj_open = open, adj_high = high, adj_low = low, adj_close = close"
    )
    adjusted_secs = 0
    adjusted_rows = 0
    updates: list[tuple[float, float, float, float, float, str, str, str]] = []
    for sec_id, evs in ca_by_sec.items():
        adjusted_secs += 1
        evs_sorted = sorted(evs, key=lambda x: x[0])
        for row in conn.execute(
            "SELECT trade_date, symbol, series, open, high, low, close FROM daily_bars WHERE sec_id = ?",
            (sec_id,),
        ):
            dt = row["trade_date"]
            fac = 1.0
            for ex_dt, f in evs_sorted:
                if dt < ex_dt:
                    fac *= f
            if fac > 1.0001:
                adjusted_rows += 1
                updates.append((
                    round(fac, 8),
                    round(float(row["open"]) / fac, 4),
                    round(float(row["high"]) / fac, 4),
                    round(float(row["low"]) / fac, 4),
                    round(float(row["close"]) / fac, 4),
                    dt, row["symbol"], row["series"],
                ))
    if updates:
        conn.executemany(
            """
            UPDATE daily_bars
            SET adj_factor = ?, adj_open = ?, adj_high = ?, adj_low = ?, adj_close = ?
            WHERE trade_date = ? AND symbol = ? AND series = ?
            """,
            updates,
        )
    conn.commit()
    return {"adjusted_securities": adjusted_secs, "adjusted_bars": adjusted_rows}


def compute_and_store_pit_liquidity(conn: sqlite3.Connection) -> dict[str, int]:
    """Compute strictly causal (`<= trade_date`) rolling liquidity & eligibility metrics."""
    conn.execute("DELETE FROM daily_liquidity")

    # Map global ordered session dates -> index for exact trading-frequency denominator
    cal_dates = [r["trade_date"] for r in conn.execute("SELECT trade_date FROM trading_calendar ORDER BY trade_date")]
    cal_idx = {d: i for i, d in enumerate(cal_dates)}

    batch: list[tuple[Any, ...]] = []
    cur_key: str | None = None
    hist: list[tuple[int, float, int, float, float, float]] = []  # (cal_i, turnover, vol, adj_c, hl_pct)

    def _flush() -> None:
        nonlocal batch
        if not batch:
            return
        conn.executemany(
            """
            INSERT INTO daily_liquidity (
                trade_date, sec_id, symbol, series, close,
                adv_20d_inr, adv_60d_inr, avg_vol_20d, turnover_ratio,
                trading_freq_60d, realized_vol_20d_ann, hl_spread_proxy_pct,
                mcap_inr_pit, pit_tier, is_pit_eligible
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            batch,
        )
        batch.clear()

    total_rows = 0
    rows = conn.execute(
        """
        SELECT trade_date, sec_id, symbol, series, high, low, close, adj_close, volume, turnover_inr, mcap_inr, is_valid_bar
        FROM daily_bars
        WHERE series IN ('EQ', 'BE', 'BZ', 'SM', 'ST', 'SZ') AND is_valid_bar = 1
        ORDER BY sec_id, trade_date
        """
    )
    for r in rows:
        sec_id = r["sec_id"]
        if sec_id != cur_key:
            cur_key = sec_id
            hist = []

        dt = r["trade_date"]
        c_i = cal_idx.get(dt, len(hist))
        t_inr = float(r["turnover_inr"] or 0.0)
        vol = int(r["volume"] or 0)
        adj_c = float(r["adj_close"])
        raw_c = float(r["close"])
        hl_pct = ((float(r["high"]) - float(r["low"])) / raw_c * 100.0) if raw_c > 0 else 0.0
        hist.append((c_i, t_inr, vol, adj_c, hl_pct, raw_c))

        w20 = hist[-20:]
        w60 = hist[-60:]
        adv20 = sum(x[1] for x in w20) / len(w20)
        adv60 = sum(x[1] for x in w60) / len(w60)
        av20 = sum(x[2] for x in w20) / len(w20)

        # Count sessions with >0 volume in trailing 60 calendar trading sessions ending at c_i
        lo_cal = max(0, c_i - 59)
        active_in_60 = sum(1 for x in w60 if x[0] >= lo_cal and x[2] > 0)
        denom_60 = min(60, c_i + 1)
        freq60 = min(1.0, active_in_60 / max(1, denom_60))

        # 20D realized annualized volatility (strictly trailing)
        rets: list[float] = []
        for j in range(1, len(w20)):
            p0, p1 = w20[j - 1][3], w20[j][3]
            if p0 > 0:
                rets.append((p1 / p0) - 1.0)
        if len(rets) >= 5:
            m_r = sum(rets) / len(rets)
            var_r = sum((x - m_r) ** 2 for x in rets) / max(1, len(rets) - 1)
            rvol = math.sqrt(var_r) * math.sqrt(252.0) * 100.0
        else:
            rvol = 0.0

        hl_med = sorted(x[4] for x in w20)[len(w20) // 2]
        mcap_v = float(r["mcap_inr"]) if r["mcap_inr"] is not None else None
        t_ratio = (adv20 / mcap_v) if (mcap_v and mcap_v > 0) else None
        srs = r["series"]

        if srs in ("EQ", "BE") and adv20 >= 500_000_000.0 and freq60 >= 0.90 and len(hist) >= 20:
            tier, elig = "TIER_A_LARGE_CORE", 1
        elif srs in ("EQ", "BE") and adv20 >= 50_000_000.0 and freq60 >= 0.80 and len(hist) >= 20:
            tier, elig = "TIER_B_BROAD_LIQUID", 1
        elif adv20 > 0.0 and len(hist) >= 5:
            tier, elig = "TIER_C_SMALL_ILLIQUID", 0
        else:
            tier, elig = "EXCLUDED_SUSPENDED_OR_WARMUP", 0

        batch.append((
            dt, sec_id, r["symbol"], srs, raw_c,
            round(adv20, 2), round(adv60, 2), round(av20, 2),
            round(t_ratio, 6) if t_ratio is not None else None,
            round(freq60, 4), round(rvol, 2), round(hl_med, 3),
            mcap_v, tier, elig,
        ))
        total_rows += 1
        if len(batch) >= 25_000:
            _flush()
    _flush()
    conn.commit()
    return {"liquidity_rows": total_rows}


class PITUniverseStore:
    """Strictly causal Point-in-Time Query API for Moreshwar 3.0 NSE Research."""

    def __init__(self, db_path: Path | str = DEFAULT_M3_DB_PATH) -> None:
        self.db_path = Path(db_path)

    def _conn(self) -> sqlite3.Connection:
        return connect_m3_db(self.db_path)

    def resolve_symbol_at_date(self, symbol: str, as_of_date: str) -> dict[str, Any] | None:
        """Resolve which canonical security held `symbol` on `as_of_date`, preserving delisted & symbol changes."""
        sym = symbol.upper().strip()
        with self._conn() as conn:
            row = conn.execute(
                """
                SELECT sh.sec_id, sh.symbol, sh.effective_from, sh.effective_to, sh.event_type,
                       sm.isin, sm.current_symbol, sm.company_name, sm.listing_date,
                       sm.delisting_date, sm.status, sm.sector, sm.industry
                FROM symbol_history sh
                JOIN security_master sm ON sm.sec_id = sh.sec_id
                WHERE sh.symbol = ? AND sh.effective_from <= ? AND sh.effective_to >= ?
                ORDER BY sh.effective_from DESC LIMIT 1
                """,
                (sym, as_of_date, as_of_date),
            ).fetchone()
            return dict(row) if row else None

    def get_pit_universe(
        self,
        as_of_date: str,
        *,
        min_adv_20d_inr: float = 50_000_000.0,
        min_trading_freq_60d: float = 0.80,
        allowed_series: tuple[str, ...] = ("EQ", "BE"),
        min_price_inr: float = 10.0,
        min_mcap_inr: float | None = None,
        exclude_etfs: bool = True,
    ) -> list[dict[str, Any]]:
        """Return strictly point-in-time eligible equities on `as_of_date`.

        Zero survivorship bias: retains delisted/merged stocks active on `as_of_date`
        and excludes stocks not yet listed on `as_of_date`. Uses only trailing data `<= as_of_date`.
        """
        srs_placeholders = ",".join("?" for _ in allowed_series)
        sql = f"""
            SELECT dl.trade_date, dl.sec_id, dl.symbol, dl.series, dl.close,
                   dl.adv_20d_inr, dl.adv_60d_inr, dl.avg_vol_20d, dl.turnover_ratio,
                   dl.trading_freq_60d, dl.realized_vol_20d_ann, dl.hl_spread_proxy_pct,
                   dl.mcap_inr_pit, dl.pit_tier,
                   sm.isin, sm.current_symbol, sm.company_name, sm.listing_date,
                   sm.delisting_date, sm.status, sm.instrument_type, sm.sector, sm.industry
            FROM daily_liquidity dl
            JOIN security_master sm ON sm.sec_id = dl.sec_id
            WHERE dl.trade_date = ?
              AND dl.series IN ({srs_placeholders})
              AND dl.adv_20d_inr >= ?
              AND dl.trading_freq_60d >= ?
              AND dl.close >= ?
              AND (sm.listing_date IS NULL OR sm.listing_date <= ?)
              AND (sm.delisting_date IS NULL OR sm.delisting_date > ?)
        """
        params: list[Any] = [
            as_of_date, *allowed_series, min_adv_20d_inr, min_trading_freq_60d,
            min_price_inr, as_of_date, as_of_date,
        ]
        if exclude_etfs:
            sql += " AND sm.instrument_type = 'EQUITY'"
        if min_mcap_inr is not None:
            sql += " AND dl.mcap_inr_pit IS NOT NULL AND dl.mcap_inr_pit >= ?"
            params.append(min_mcap_inr)
        sql += " ORDER BY dl.adv_20d_inr DESC, dl.sec_id ASC"
        with self._conn() as conn:
            return [dict(r) for r in conn.execute(sql, params).fetchall()]

    def get_security_history(
        self,
        sec_id_or_symbol: str,
        *,
        start_date: str | None = None,
        end_date: str | None = None,
        adjusted: bool = True,
    ) -> list[dict[str, Any]]:
        """Retrieve chronological history keyed by canonical `sec_id` (stitching across symbol changes)."""
        key = sec_id_or_symbol.upper().strip()
        with self._conn() as conn:
            sec_id = key
            chk = conn.execute("SELECT sec_id FROM security_master WHERE sec_id = ?", (key,)).fetchone()
            if not chk:
                sh = conn.execute(
                    "SELECT sec_id FROM symbol_history WHERE symbol = ? ORDER BY effective_to DESC LIMIT 1",
                    (key,),
                ).fetchone()
                if not sh:
                    return []
                sec_id = sh["sec_id"]

            sql = """
                SELECT trade_date, symbol, series, sec_id, prev_close,
                       open, high, low, close, vwap, volume, turnover_inr, no_of_trades,
                       delivery_qty, delivery_pct, adj_factor,
                       adj_open, adj_high, adj_low, adj_close,
                       mcap_inr, mcap_source, dq_flags, is_valid_bar
                FROM daily_bars
                WHERE sec_id = ? AND is_valid_bar = 1
            """
            params: list[Any] = [sec_id]
            if start_date:
                sql += " AND trade_date >= ?"
                params.append(start_date)
            if end_date:
                sql += " AND trade_date <= ?"
                params.append(end_date)
            sql += " ORDER BY trade_date ASC"
            rows = []
            for r in conn.execute(sql, params).fetchall():
                d = dict(r)
                if adjusted:
                    d["eff_open"] = d["adj_open"]
                    d["eff_high"] = d["adj_high"]
                    d["eff_low"] = d["adj_low"]
                    d["eff_close"] = d["adj_close"]
                else:
                    d["eff_open"] = d["open"]
                    d["eff_high"] = d["high"]
                    d["eff_low"] = d["low"]
                    d["eff_close"] = d["close"]
                rows.append(d)
            return rows

    def get_corporate_actions(
        self,
        *,
        sec_id: str | None = None,
        symbol: str | None = None,
        up_to_date: str | None = None,
    ) -> list[dict[str, Any]]:
        with self._conn() as conn:
            sql = "SELECT * FROM corporate_actions WHERE 1=1"
            params: list[Any] = []
            if sec_id:
                sql += " AND sec_id = ?"
                params.append(sec_id)
            if symbol:
                sql += " AND symbol = ?"
                params.append(symbol.upper().strip())
            if up_to_date:
                sql += " AND ex_date <= ?"
                params.append(up_to_date)
            sql += " ORDER BY ex_date ASC"
            return [dict(r) for r in conn.execute(sql, params).fetchall()]
