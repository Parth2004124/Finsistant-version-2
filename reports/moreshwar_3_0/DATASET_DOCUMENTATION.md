# MORESHWAR 3.0 — DATASET ARCHITECTURE & DOCUMENTATION

## 1. Frozen Moreshwar 2.1 & 2.2 Protection
- All Moreshwar 2.1 and 2.2 engines, `data/moreshwar_v2_*`, `data/expanded_history_5y_cache.json.gz`, and `reports/moreshwar_2_2/` are strictly frozen and unmodified.
- Moreshwar 3.0 lives exclusively in `finsistant_v2/m3/`, `data/m3_universe/`, and `reports/moreshwar_3_0/`.

## 2. Authoritative Data Sources
1. **Security Master (`EQUITY_L.csv`, `delisted.csv`, `symbolchange.csv`, `namechange.csv`):** Official NSE India equity registers establishing listing dates, delisting dates/types, ISIN identifiers, face value, paid-up value, and SCD Type-2 symbol timelines.
2. **Daily Exchange Market Data (`sec_bhavdata_full_DDMMYYYY.csv`):** Official NSE daily cash-market equity records with unadjusted O/H/L/C/PrevClose/VWAP, Traded Volume, Turnover (INR), No. of Trades, Delivery Volume (`DELIV_QTY`), and Delivery % (`DELIV_PER`).
3. **Official Exchange PR Archives (`PRddmmyy.zip`):** Daily index OHLC (`Pd` / `Pr` CSVs) and official NSE daily Market Capitalisation & Issue Size (`MCAPddmmyyyy.csv`, available from UDiFF migration onward).
4. **Corporate Actions (`corporates-corporateActions` API):** Official NSE Splits, Bonuses, Dividends, Rights Issues, and Demerger/Amalgamation notices keyed by `sec_id` + `ex_date`.
5. **Secondary Verification (`Yahoo Finance Chart API`):** Cross-checks daily returns, splits, and delisted 404 probes on a stratified verification sample (`source_discrepancies`).

## 3. Modular Schema (`finsistant_v2/m3/schema.py`)
- `dataset_versions`: Immutable snapshot ledger with parent lineage & SHA-256 verification.
- `trading_calendar`: Actual NSE market sessions with index OHLC and official MCAP availability flag.
- `security_master`: Canonical `sec_id`-keyed master covering active, delisted, suspended, merged, and historical equities.
- `symbol_history` & `name_history`: Slowly Changing Dimension Type-2 ticker intervals so historical lookups never confuse renamed companies.
- `daily_bars`: Point-in-time unadjusted & split/bonus-adjusted OHLCV, delivery volume/%, turnover, official MCAP (`NULL` when unavailable before mid-2024), and `dq_flags`.
- `corporate_actions`: Parsed Splits, Bonuses, Dividends, Rights, and Restructuring records.
- `index_snapshots`: Static constituent lists explicitly tagged `is_static_snapshot = 1` (never substituted for PIT history).
- `daily_liquidity`: Strictly causal (`<= trade_date`) rolling 20D/60D ADV, avg volume, turnover ratio, 60D trading frequency, 20D realized volatility, HL range spread proxy, and `pit_tier`.
- `data_quality_events` & `source_discrepancies`: Full audit trail of every warning, excluded invalid bar, gap, or cross-provider variance.

## 4. Usage & Incremental Updates
```python
from finsistant_v2.m3 import PITUniverseStore, build_m3_pit_universe

# Query point-in-time liquid universe on any date (zero survivorship / zero look-ahead):
store = PITUniverseStore()
elig = store.get_pit_universe('2023-03-15', min_adv_20d_inr=50_000_000, min_trading_freq_60d=0.80)
hist = store.get_security_history('ZYDUSLIFE', end_date='2023-03-15', adjusted=True)  # stitches CADILAHC -> ZYDUSLIFE!
```

## 5. Known Limitations
- **LIM-01 (Historical Index Membership):** NSE provides current constituent snapshots (ind_nifty50list.csv, ind_nifty500list.csv, ind_niftytotalmarket_list.csv) as of retrieval date, not free point-in-time constituent change ledgers back to 2021. Static index CSVs are explicitly isolated in `index_snapshots (is_static_snapshot = 1)` and NEVER used for PIT eligibility. PIT selection instead uses actual daily NSE trading appearance, listing_date <= t, delisting_date > t, and strictly trailing 20D/60D ADV & trading frequency.
- **LIM-02 (Official Daily Market Capitalisation Pre-July 2024):** NSE began embedding `MCAP{ddmmyyyy}.csv` inside daily `PR.zip` with UDiFF migration in mid-2024. Sessions prior to that lack official NSE daily market-cap files inside `PR.zip`. Pre-UDiFF sessions store `mcap_inr = NULL` (`mcap_source = UNAVAILABLE_PRE_2024_UDIFF`) rather than fabricating backfilled market cap from today's share count.
- **LIM-03 (Pre-2021 Delisted Companies):** The dataset spans `2021-01-01` to `2026-11-19`. Companies delisted prior to 2021-01-01 (e.g., DHFL suspended/delisted earlier) remain in `security_master` (`delisted.csv`) but have 0 active trading sessions inside the 2021–2026 Bhavcopy window. Companies delisted/merged during 2021–2026 (e.g., HDFC, MINDTREE, INFRATEL, etc.) are fully retained with their real daily prices up to their exact final session.
- **LIM-04 (Rights Issues & Demerger Price Adjustments):** Splits and Bonuses compute exact deterministic adjustment factors (`old_fv/new_fv`, `(A+B)/B`). Rights issues (`RIGHTS`) and Demergers (`DEMERGER_RESTRUCTURE`, e.g., TATAMOTORS -> TMPV/TMCV or JIOFIN spin-off) depend on ex-date fair-value ratios not fixed strictly by ratio alone. Rights & Demergers are cataloged in `corporate_actions` (`adj_factor = 1.0`) and automatically exempt post-event bars from false `UNEXPLAINED_JUMP_SHOCK` alerts while remaining flagged for strategy-level corporate-event handling.
- **LIM-05 (Bid-Ask Spread Proxy):** EOD Exchange Bhavcopy does not contain intraday Level-1 best bid/ask quotes. Liquidity engine provides trailing 20D median `(High - Low) / Close * 100` (`hl_spread_proxy_pct`) alongside exact NSE `DELIV_QTY`, `DELIV_PER`, `NO_OF_TRADES`, and `TURNOVER_INR`.
- **LIM-06 (Point-in-Time Fundamentals Not In Bhavcopy):** Quarterly balance sheet / P&L / shareholding filings are separate XBRL corporate filings outside daily Bhavcopy archives. Moreshwar 3.0 Prompt 1 establishes the price, volume, delivery, market cap, corporate action, and PIT universe backbone; fundamental PIT ingestion should be added separately with filing-date lags if used.
