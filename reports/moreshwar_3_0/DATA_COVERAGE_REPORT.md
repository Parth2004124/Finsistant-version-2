# MORESHWAR 3.0 — POINT-IN-TIME NSE UNIVERSE DATA COVERAGE REPORT

- **Dataset Version ID:** `m3_20261119T191710Z`
- **Window:** `2021-01-01` → `2026-11-19` (**1454** NSE trading sessions)
- **Generated At (UTC):** `2026-11-19T19:17:10.968991+00:00`
- **Checksum SHA-256:** `46acf0f923549319ce3835a7b025322c87f6d80a4c8ac208f63791892920cb2d`

## 1. Universe Scale vs Moreshwar 2.1 / 2.2

| Metric | Moreshwar 2.1 / 2.2 (Frozen) | Moreshwar 3.0 PIT Universe |
| :--- | :---: | :---: |
| Total Securities in Security Master | `40` (`36` full-history INR) | **`4416`** |
| Validated Daily Equity Bars | ~`43,000` | **`3,498,739`** (`3,498,739` total) |
| Trading Sessions | `1,176` | **`1454`** |
| Corporate Actions Ledger | None (provider adj only) | **`13,074`** events |
| Delivery Qty & % Coverage | Unavailable | **`90.14%`** (`3,153,666` bars) |
| Official NSE Daily Market Cap | Unavailable | **`1,768,309`** bars (from `2024-02-01`) |

## 2. Security Master Status & Series Coverage

| Lifecycle Status | Count | Series | Daily Bars |
| :--- | ---: | :--- | ---: |
| `ACTIVE` | 2629 | `EQ` | 2,809,452 |
| `INACTIVE_OR_DELISTED_MERGED` | 1097 | `BE` | 302,326 |
| `ACTIVE_ETF` | 361 | `SM` | 275,618 |
| `DELISTED` | 328 | `ST` | 67,810 |
| `SUSPENDED_OR_INACTIVE` | 1 | `BZ` | 42,727 |
| `—` |  | `SZ` | 806 |

## 3. Corporate Actions Database Breakdown

| Action Type | Count |
| :--- | ---: |
| `DIVIDEND` | 8,817 |
| `OTHER` | 3,389 |
| `SPLIT` | 290 |
| `BONUS` | 287 |
| `RIGHTS` | 220 |
| `DEMERGER_RESTRUCTURE` | 71 |

## 4. Point-in-Time Universe Sample Audit Across Regimes (2021–2026)

| As-Of Date | Active NSE Equities | Broad Liquid PIT (ADV ≥ ₹5 Cr) | Core Liquid PIT (ADV ≥ ₹50 Cr) | Delisted/Merged Retained at Date | Examples Retained |
| :---: | ---: | ---: | ---: | ---: | :--- |
| `2021-03-15` | 1,852 | **619** | **212** | 32 | `HDFC, PEL, MINDTREE, IDFC, TATAMTRDVR, MEGH` |
| `2022-03-15` | 1,957 | **673** | **202** | 31 | `HDFC, MINDTREE, PEL, DHANI, TV18BRDCST, TATAMTRDVR` |
| `2023-03-15` | 2,114 | **613** | **193** | 16 | `HDFC, PEL, IDFC, TATAMTRDVR, GSPL, TCNSBRANDS` |
| `2024-03-15` | 2,393 | **1,004** | **363** | 20 | `TATAMTRDVR, PEL, JPASSOCIAT, IDFC, ISEC, GSPL` |
| `2025-03-17` | 2,750 | **913** | **334** | 9 | `PEL, GSPL, ISEC, SETFGOLD, JBCHEPHARM, DHANI` |
| `2026-11-18` | 3,440 | **1,181** | **429** | 0 | `—` |

## 5. Data-Quality Audit Summary

| Check Code | Count | Policy |
| :--- | ---: | :--- |
| `UNEXPLAINED_JUMP_SHOCK` | 2,134 | Non-silent flag / exclusion logged |

Cross-provider NSE vs Yahoo Finance sample verification checked **`57`** active, split-adjusted, symbol-changed, and delisted tickers (details in `data_quality_report.json`).

## 6. Known Limitations

- **LIM-01 (Historical Index Membership):** NSE provides current constituent snapshots (ind_nifty50list.csv, ind_nifty500list.csv, ind_niftytotalmarket_list.csv) as of retrieval date, not free point-in-time constituent change ledgers back to 2021. *Mitigation:* Static index CSVs are explicitly isolated in `index_snapshots (is_static_snapshot = 1)` and NEVER used for PIT eligibility. PIT selection instead uses actual daily NSE trading appearance, listing_date <= t, delisting_date > t, and strictly trailing 20D/60D ADV & trading frequency.
- **LIM-02 (Official Daily Market Capitalisation Pre-July 2024):** NSE began embedding `MCAP{ddmmyyyy}.csv` inside daily `PR.zip` with UDiFF migration in mid-2024. Sessions prior to that lack official NSE daily market-cap files inside `PR.zip`. *Mitigation:* Pre-UDiFF sessions store `mcap_inr = NULL` (`mcap_source = UNAVAILABLE_PRE_2024_UDIFF`) rather than fabricating backfilled market cap from today's share count.
- **LIM-03 (Pre-2021 Delisted Companies):** The dataset spans `2021-01-01` to `2026-11-19`. Companies delisted prior to 2021-01-01 (e.g., DHFL suspended/delisted earlier) remain in `security_master` (`delisted.csv`) but have 0 active trading sessions inside the 2021–2026 Bhavcopy window. *Mitigation:* Companies delisted/merged during 2021–2026 (e.g., HDFC, MINDTREE, INFRATEL, etc.) are fully retained with their real daily prices up to their exact final session.
- **LIM-04 (Rights Issues & Demerger Price Adjustments):** Splits and Bonuses compute exact deterministic adjustment factors (`old_fv/new_fv`, `(A+B)/B`). Rights issues (`RIGHTS`) and Demergers (`DEMERGER_RESTRUCTURE`, e.g., TATAMOTORS -> TMPV/TMCV or JIOFIN spin-off) depend on ex-date fair-value ratios not fixed strictly by ratio alone. *Mitigation:* Rights & Demergers are cataloged in `corporate_actions` (`adj_factor = 1.0`) and automatically exempt post-event bars from false `UNEXPLAINED_JUMP_SHOCK` alerts while remaining flagged for strategy-level corporate-event handling.
- **LIM-05 (Bid-Ask Spread Proxy):** EOD Exchange Bhavcopy does not contain intraday Level-1 best bid/ask quotes. *Mitigation:* Liquidity engine provides trailing 20D median `(High - Low) / Close * 100` (`hl_spread_proxy_pct`) alongside exact NSE `DELIV_QTY`, `DELIV_PER`, `NO_OF_TRADES`, and `TURNOVER_INR`.
- **LIM-06 (Point-in-Time Fundamentals Not In Bhavcopy):** Quarterly balance sheet / P&L / shareholding filings are separate XBRL corporate filings outside daily Bhavcopy archives. *Mitigation:* Moreshwar 3.0 Prompt 1 establishes the price, volume, delivery, market cap, corporate action, and PIT universe backbone; fundamental PIT ingestion should be added separately with filing-date lags if used.
