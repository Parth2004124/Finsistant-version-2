"""SQLite schema and version registry for Moreshwar 3.0 PIT NSE Research Universe."""

from __future__ import annotations

from pathlib import Path
import sqlite3

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DEFAULT_M3_DATA_DIR = ROOT_DIR / "data" / "m3_universe"
DEFAULT_M3_RAW_CACHE_DIR = DEFAULT_M3_DATA_DIR / "raw_cache"
DEFAULT_M3_DB_PATH = DEFAULT_M3_DATA_DIR / "nse_pit_universe.db"
DEFAULT_M3_REPORTS_DIR = ROOT_DIR / "reports" / "moreshwar_3_0"


SCHEMA_SQL = """
PRAGMA journal_mode = DELETE;
PRAGMA synchronous = NORMAL;
PRAGMA foreign_keys = OFF;

CREATE TABLE IF NOT EXISTS dataset_versions (
    version_id TEXT PRIMARY KEY,
    snapshot_tag TEXT NOT NULL,
    created_at TEXT NOT NULL,
    date_start TEXT NOT NULL,
    date_end TEXT NOT NULL,
    parent_version_id TEXT,
    is_incremental INTEGER NOT NULL DEFAULT 0,
    source_manifest_json TEXT NOT NULL,
    row_counts_json TEXT NOT NULL,
    checksum_sha256 TEXT NOT NULL,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS trading_calendar (
    trade_date TEXT PRIMARY KEY,
    nifty50_open REAL,
    nifty50_high REAL,
    nifty50_low REAL,
    nifty50_close REAL,
    nifty500_close REAL,
    midcap150_close REAL,
    smallcap250_close REAL,
    eq_symbols_count INTEGER NOT NULL DEFAULT 0,
    total_bhav_rows INTEGER NOT NULL DEFAULT 0,
    has_official_mcap INTEGER NOT NULL DEFAULT 0,
    bhav_source_url TEXT,
    pr_source_url TEXT,
    retrieved_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS security_master (
    sec_id TEXT PRIMARY KEY,
    isin TEXT,
    current_symbol TEXT NOT NULL,
    company_name TEXT NOT NULL,
    series TEXT,
    listing_date TEXT,
    delisting_date TEXT,
    delisting_type TEXT,
    status TEXT NOT NULL,
    instrument_type TEXT NOT NULL DEFAULT 'EQUITY',
    face_value REAL,
    paid_up_value REAL,
    market_lot INTEGER,
    sector TEXT,
    industry TEXT,
    source_primary TEXT NOT NULL,
    first_seen_date TEXT,
    last_seen_date TEXT,
    total_sessions_observed INTEGER NOT NULL DEFAULT 0,
    retrieved_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS symbol_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sec_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    effective_from TEXT NOT NULL,
    effective_to TEXT NOT NULL,
    event_type TEXT NOT NULL,
    old_symbol TEXT,
    new_symbol TEXT,
    source_provider TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS name_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    old_name TEXT NOT NULL,
    new_name TEXT NOT NULL,
    effective_date TEXT NOT NULL,
    source_provider TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS daily_bars (
    trade_date TEXT NOT NULL,
    symbol TEXT NOT NULL,
    series TEXT NOT NULL,
    sec_id TEXT NOT NULL,
    prev_close REAL,
    open REAL NOT NULL,
    high REAL NOT NULL,
    low REAL NOT NULL,
    close REAL NOT NULL,
    last_price REAL,
    vwap REAL,
    volume INTEGER NOT NULL,
    turnover_inr REAL NOT NULL,
    no_of_trades INTEGER NOT NULL DEFAULT 0,
    delivery_qty INTEGER,
    delivery_pct REAL,
    adj_factor REAL NOT NULL DEFAULT 1.0,
    adj_open REAL NOT NULL,
    adj_high REAL NOT NULL,
    adj_low REAL NOT NULL,
    adj_close REAL NOT NULL,
    mcap_inr REAL,
    issue_size INTEGER,
    mcap_source TEXT NOT NULL DEFAULT 'UNAVAILABLE',
    exchange_category TEXT,
    dq_flags TEXT NOT NULL DEFAULT '',
    is_valid_bar INTEGER NOT NULL DEFAULT 1,
    PRIMARY KEY (trade_date, symbol, series)
);

CREATE TABLE IF NOT EXISTS corporate_actions (
    ca_id TEXT PRIMARY KEY,
    sec_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    isin TEXT,
    ex_date TEXT NOT NULL,
    record_date TEXT,
    series TEXT,
    action_type TEXT NOT NULL,
    ratio_num REAL,
    ratio_den REAL,
    adj_factor REAL NOT NULL DEFAULT 1.0,
    cash_dividend_inr REAL DEFAULT 0.0,
    rights_premium_inr REAL,
    face_value TEXT,
    subject TEXT NOT NULL,
    source_provider TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS index_snapshots (
    index_name TEXT NOT NULL,
    symbol TEXT NOT NULL,
    isin TEXT,
    company_name TEXT,
    industry TEXT,
    series TEXT,
    snapshot_as_of_date TEXT NOT NULL,
    is_static_snapshot INTEGER NOT NULL DEFAULT 1,
    source_url TEXT NOT NULL,
    PRIMARY KEY (index_name, symbol, snapshot_as_of_date)
);

CREATE TABLE IF NOT EXISTS daily_index_bars (
    trade_date TEXT NOT NULL,
    index_name TEXT NOT NULL,
    prev_close REAL,
    open REAL,
    high REAL,
    low REAL,
    close REAL NOT NULL,
    turnover_inr REAL,
    source_provider TEXT NOT NULL,
    PRIMARY KEY (trade_date, index_name)
);

CREATE TABLE IF NOT EXISTS daily_liquidity (
    trade_date TEXT NOT NULL,
    sec_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    series TEXT NOT NULL,
    close REAL NOT NULL,
    adv_20d_inr REAL NOT NULL,
    adv_60d_inr REAL NOT NULL,
    avg_vol_20d REAL NOT NULL,
    turnover_ratio REAL,
    trading_freq_60d REAL NOT NULL,
    realized_vol_20d_ann REAL NOT NULL,
    hl_spread_proxy_pct REAL NOT NULL,
    mcap_inr_pit REAL,
    pit_tier TEXT NOT NULL,
    is_pit_eligible INTEGER NOT NULL,
    PRIMARY KEY (trade_date, symbol, series)
);

CREATE TABLE IF NOT EXISTS data_quality_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    version_id TEXT NOT NULL,
    trade_date TEXT,
    symbol TEXT,
    series TEXT,
    check_code TEXT NOT NULL,
    severity TEXT NOT NULL,
    detail TEXT NOT NULL,
    action_taken TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS source_discrepancies (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    sec_id TEXT,
    check_type TEXT NOT NULL,
    primary_source TEXT NOT NULL,
    secondary_source TEXT NOT NULL,
    discrepancy_count INTEGER NOT NULL DEFAULT 0,
    max_abs_diff_pct REAL,
    detail TEXT NOT NULL,
    checked_at TEXT NOT NULL
);
"""

INDEXES_SQL = """
CREATE INDEX IF NOT EXISTS idx_sec_symbol ON security_master(current_symbol);
CREATE INDEX IF NOT EXISTS idx_sec_isin ON security_master(isin);
CREATE INDEX IF NOT EXISTS idx_sec_status ON security_master(status);
CREATE INDEX IF NOT EXISTS idx_sec_type ON security_master(instrument_type);
CREATE INDEX IF NOT EXISTS idx_symhist_sec ON symbol_history(sec_id, effective_from, effective_to);
CREATE INDEX IF NOT EXISTS idx_symhist_sym ON symbol_history(symbol, effective_from, effective_to);
CREATE INDEX IF NOT EXISTS idx_namehist_sym ON name_history(symbol);
CREATE INDEX IF NOT EXISTS idx_bars_sec_date ON daily_bars(sec_id, trade_date);
CREATE INDEX IF NOT EXISTS idx_bars_sym_date ON daily_bars(symbol, trade_date);
CREATE INDEX IF NOT EXISTS idx_bars_date_valid ON daily_bars(trade_date, is_valid_bar, series);
CREATE INDEX IF NOT EXISTS idx_ca_sec_ex ON corporate_actions(sec_id, ex_date);
CREATE INDEX IF NOT EXISTS idx_ca_sym_ex ON corporate_actions(symbol, ex_date);
CREATE INDEX IF NOT EXISTS idx_ca_type ON corporate_actions(action_type, ex_date);
CREATE INDEX IF NOT EXISTS idx_liq_date_elig ON daily_liquidity(trade_date, is_pit_eligible, adv_20d_inr);
CREATE INDEX IF NOT EXISTS idx_liq_sec_date ON daily_liquidity(sec_id, trade_date);
CREATE INDEX IF NOT EXISTS idx_dq_code ON data_quality_events(check_code, severity);
CREATE INDEX IF NOT EXISTS idx_dq_date ON data_quality_events(trade_date);
"""


def connect_m3_db(db_path: Path | str = DEFAULT_M3_DB_PATH) -> sqlite3.Connection:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=30.0)
    conn.row_factory = sqlite3.Row
    return conn


def create_m3_indexes(conn: sqlite3.Connection) -> None:
    conn.executescript(INDEXES_SQL)
    conn.commit()


def init_m3_schema(conn: sqlite3.Connection, *, create_indexes: bool = True) -> None:
    conn.executescript(SCHEMA_SQL)
    if create_indexes:
        conn.executescript(INDEXES_SQL)
    conn.commit()
