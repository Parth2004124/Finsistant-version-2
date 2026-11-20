# MORESHWAR 3.0 (PROMPT 2) — MULTI-FACTOR ALPHA DISCOVERY & VALIDATION REPORT

- **Dataset:** `2021-01-01 -> 2026-11-19 (1454 sessions)` (NSE Active + Delisted/Merged PIT Equities, ETFs excluded)
- **Chronological Split:** IS `2021-07-01 -> 2025-03-31` | OOS `2025-04-01 -> 2026-11-19`
- **Cost Model:** Zerodha Equity Delivery (22.0 bps one-way incl STT/DP/7.5bps slippage)
- **Multiple Testing Scope:** `31` factors × `5` horizons = `155` hypotheses (Benjamini-Hochberg FDR + Bonferroni + Block Bootstrap)
- **Validated Independent Factors:** **`4`** | **Rejected / Redundant / Non-PIT:** **`31`**

## 1. Validated Independent Factors Entering Prompt 3

| Rank | Factor ID | Family | Best H | IS Rank IC | IS IC_IR | NW t-stat | BH q (155) | OOS Rank IC | Mono | Net Q5 Excess/yr | Max |rho| |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | `DIST_252D_HIGH` | Market_Structure | 30D | +0.0766 | +0.938 | +6.08 | 0.0000 | **+0.0818** | +0.90 | **+5.08%** | 0.00 |
| 2 | `DIV_YIELD_TTM_PIT` | Fundamental | 30D | +0.0539 | +0.680 | +3.14 | 0.0097 | **+0.0205** | +0.90 | **+6.98%** | 0.04 |
| 3 | `MOM_DETERIORATION_60D` | Momentum | 30D | +0.0485 | +0.571 | +3.10 | 0.0106 | **+0.0479** | +0.90 | **+1.18%** | 0.69 |
| 4 | `AMIHUD_LIQUIDITY_20D` | Volume_Liquidity | 10D | +0.0484 | +0.414 | +4.01 | 0.0006 | **+0.0584** | +0.50 | **-1.15%** | 0.22 |

## 2. Full 31-Factor Multi-Horizon Predictive & Cost-Adjusted Table

| Factor ID | Family | H | IS Rank IC | OOS Rank IC | Net L-S/yr | Q5-Univ Net/yr | TO | Bull IC | Non-Bull IC | Boot P05 IC | Gate |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `DIST_252D_HIGH` | Market_Structure | 30D | +0.0766 | +0.0818 | +15.60% | +5.08% | 0.50 | +0.0813 | +0.0668 | +0.0501 | PASS |
| `DIV_YIELD_TTM_PIT` | Fundamental | 30D | +0.0539 | +0.0205 | +11.36% | +6.98% | 0.25 | +0.0344 | +0.0749 | +0.0163 | PASS |
| `MOM_DETERIORATION_60D` | Momentum | 30D | +0.0485 | +0.0479 | +8.01% | +1.18% | 0.68 | +0.0434 | +0.0698 | +0.0218 | PASS |
| `AMIHUD_LIQUIDITY_20D` | Volume_Liquidity | 10D | +0.0484 | +0.0584 | +1.57% | -1.15% | 0.26 | +0.0414 | +0.0987 | +0.0344 | PASS |
| `VOL_ADJ_MOM_60D` | Volatility | 60D | +0.0441 | +0.0670 | +10.96% | +4.04% | 0.82 | +0.0531 | +0.0359 | +0.0210 | FAIL |
| `MOM_120_SKIP10` | Momentum | 60D | +0.0419 | +0.0575 | +12.91% | +4.12% | 0.65 | +0.0519 | +0.0132 | +0.0165 | FAIL |
| `MOM_60D` | Momentum | 60D | +0.0405 | +0.0495 | +11.97% | +4.36% | 0.82 | +0.0467 | +0.0213 | +0.0082 | FAIL |
| `DELIVERY_SURGE_5V20` | Volume_Liquidity | 10D | +0.0215 | +0.0086 | -12.83% | -7.47% | 0.85 | +0.0171 | +0.0198 | +0.0105 | FAIL |
| `DELIVERY_PCT_20D` | Volume_Liquidity | 60D | +0.0362 | +0.0633 | -1.10% | -2.13% | 0.53 | +0.0428 | +0.0508 | +0.0132 | FAIL |
| `IDIO_VOL_20D` | Volatility | 20D | +0.0500 | +0.0778 | -6.94% | -5.57% | 0.62 | +0.0377 | +0.1434 | +0.0303 | FAIL |
| `MOM_120D` | Momentum | 60D | +0.0339 | +0.0770 | +12.77% | +4.79% | 0.62 | +0.0503 | +0.0214 | +0.0123 | FAIL |
| `SECTOR_BREADTH_SMA50` | Sector | 60D | +0.0328 | +0.0193 | +7.61% | +9.39% | 0.71 | +0.0196 | +0.0851 | -0.0002 | FAIL |
| `LOW_VOL_20D` | Volatility | 20D | +0.0489 | +0.0501 | -6.38% | -5.44% | 0.61 | +0.0321 | +0.1435 | +0.0250 | FAIL |
| `TURNOVER_ACCEL_5V60` | Volume_Liquidity | 30D | +0.0216 | +0.0123 | +2.16% | +1.46% | 0.83 | +0.0114 | +0.0510 | +0.0036 | FAIL |
| `SIZE_LOG_ADV_PIT` | Fundamental | 10D | +0.0348 | +0.0407 | -2.27% | -2.78% | 0.23 | +0.0294 | +0.0705 | +0.0226 | FAIL |
| `REL_VOL_5V20` | Volume_Liquidity | 30D | +0.0170 | +0.0142 | -0.86% | -0.76% | 0.85 | +0.0108 | +0.0397 | +0.0039 | FAIL |
| `STOCK_VS_SECTOR_20D` | Sector | 30D | +0.0162 | +0.0163 | +6.37% | +2.62% | 0.80 | +0.0165 | +0.0151 | +0.0004 | FAIL |
| `MOM_20D` | Momentum | 30D | +0.0187 | +0.0128 | +4.77% | +2.12% | 0.82 | +0.0130 | +0.0338 | -0.0004 | FAIL |
| `BREAKOUT_QUALITY_20D` | Market_Structure | 30D | +0.0150 | +0.0263 | +1.88% | -0.53% | 0.82 | +0.0181 | +0.0200 | +0.0058 | FAIL |
| `EXTENSION_SMA50_ATR` | Market_Structure | 10D | +0.0053 | +0.0016 | -17.61% | -9.89% | 0.47 | +0.0056 | -0.0024 | -0.0071 | FAIL |
| `GAP_BEHAVIOR_10D` | Market_Structure | 60D | -0.0123 | +0.0239 | +0.16% | -0.59% | 0.79 | -0.0135 | +0.0675 | -0.0289 | FAIL |
| `REL_STRENGTH_N50_60D` | Momentum | 60D | +0.0405 | -0.0133 | +11.97% | +4.36% | 0.82 | +0.0438 | -0.0376 | -0.0017 | FAIL |
| `MOM_ACCEL_20V60` | Momentum | 30D | +0.0161 | -0.0016 | +2.46% | +1.51% | 0.90 | +0.0020 | +0.0490 | -0.0040 | FAIL |
| `VOL_CONTRACTION_10V60` | Volatility | 60D | +0.0046 | -0.0540 | -1.00% | -0.88% | 0.85 | -0.0101 | -0.0244 | -0.0326 | FAIL |
| `SECTOR_RS_60D` | Sector | 60D | +0.0072 | -0.0392 | +12.27% | +10.66% | 0.65 | -0.0044 | +0.0481 | -0.0239 | FAIL |
| `TREND_PERSISTENCE_60D` | Momentum | 10D | +0.0043 | -0.0009 | -12.16% | -7.33% | 0.49 | +0.0044 | -0.0052 | -0.0044 | FAIL |
| `VOL_EXPANSION_5V20` | Volatility | 30D | +0.0034 | -0.0027 | -3.32% | -2.37% | 0.84 | -0.0016 | +0.0153 | -0.0111 | FAIL |
| `PULLBACK_DEPTH_SMA50` | Market_Structure | 10D | +0.0013 | -0.0083 | -17.54% | -15.43% | 0.81 | -0.0016 | -0.0016 | -0.0124 | FAIL |
| `SECTOR_MOM_20D` | Sector | 20D | -0.0012 | -0.0051 | +5.43% | +7.05% | 0.67 | -0.0076 | +0.0269 | -0.0184 | FAIL |
| `OBV_FLOW_20D` | Volume_Liquidity | 30D | -0.0071 | -0.0205 | +0.45% | -0.14% | 0.81 | -0.0092 | -0.0195 | -0.0260 | FAIL |
| `SIZE_LOG_MCAP_PIT` | Fundamental | 10D | -0.0628 | -0.0611 | -22.02% | -14.64% | 0.16 | -0.0446 | -0.1345 | -0.0945 | FAIL |

## 3. Rejected & Redundant Factors (Explicitly Documented)

| Factor ID | Family | Status | Exact Reason |
| :--- | :--- | :--- | :--- |
| `VOL_ADJ_MOM_60D` | Volatility | `REJECTED` | failed BH FDR (q=0.1172) |
| `MOM_120_SKIP10` | Momentum | `REJECTED` | failed BH FDR (q=0.1664) |
| `MOM_60D` | Momentum | `REJECTED` | failed BH FDR (q=0.2102) |
| `DELIVERY_SURGE_5V20` | Volume_Liquidity | `REJECTED` | cost/turnover eroded net alpha |
| `DELIVERY_PCT_20D` | Volume_Liquidity | `REJECTED` | failed BH FDR (q=0.2629); non-monotonic quintiles (+0.40); cost/turnover eroded net alpha |
| `IDIO_VOL_20D` | Volatility | `REJECTED` | non-monotonic quintiles (-0.30); cost/turnover eroded net alpha |
| `MOM_120D` | Momentum | `REJECTED` | failed BH FDR (q=0.2910) |
| `SECTOR_BREADTH_SMA50` | Sector | `REJECTED` | failed BH FDR (q=0.2272); non-monotonic quintiles (+0.10) |
| `LOW_VOL_20D` | Volatility | `REJECTED` | non-monotonic quintiles (+0.10); cost/turnover eroded net alpha |
| `TURNOVER_ACCEL_5V60` | Volume_Liquidity | `REJECTED` | failed BH FDR (q=0.1568) |
| `SIZE_LOG_ADV_PIT` | Fundamental | `REJECTED` | non-monotonic quintiles (+0.10); cost/turnover eroded net alpha |
| `REL_VOL_5V20` | Volume_Liquidity | `REJECTED` | failed BH FDR (q=0.1392); cost/turnover eroded net alpha |
| `STOCK_VS_SECTOR_20D` | Sector | `REJECTED` | failed BH FDR (q=0.1825) |
| `MOM_20D` | Momentum | `REJECTED` | failed BH FDR (q=0.2149) |
| `BREAKOUT_QUALITY_20D` | Market_Structure | `REJECTED` | failed BH FDR (q=0.2099) |
| `EXTENSION_SMA50_ATR` | Market_Structure | `REJECTED` | weak IS Rank IC (+0.0053); failed BH FDR (q=0.7108); non-monotonic quintiles (-0.90); cost/turnover eroded net alpha |
| `GAP_BEHAVIOR_10D` | Market_Structure | `REJECTED` | weak IS Rank IC (-0.0123); failed BH FDR (q=0.6945) |
| `REL_STRENGTH_N50_60D` | Momentum | `REJECTED` | failed BH FDR (q=0.2102); OOS Rank IC reversed/zero (-0.0133) |
| `MOM_ACCEL_20V60` | Momentum | `REJECTED` | failed BH FDR (q=0.2910); OOS Rank IC reversed/zero (-0.0016) |
| `VOL_CONTRACTION_10V60` | Volatility | `REJECTED` | weak IS Rank IC (+0.0046); failed BH FDR (q=0.7999); OOS Rank IC reversed/zero (-0.0540); non-monotonic quintiles (+0.30); cost/turnover eroded net alpha |
| `SECTOR_RS_60D` | Sector | `REJECTED` | weak IS Rank IC (+0.0072); failed BH FDR (q=0.8310); OOS Rank IC reversed/zero (-0.0392); non-monotonic quintiles (+0.10) |
| `TREND_PERSISTENCE_60D` | Momentum | `REJECTED` | weak IS Rank IC (+0.0043); failed BH FDR (q=0.5904); OOS Rank IC reversed/zero (-0.0009); non-monotonic quintiles (-0.70); cost/turnover eroded net alpha |
| `VOL_EXPANSION_5V20` | Volatility | `REJECTED` | weak IS Rank IC (+0.0034); failed BH FDR (q=0.8351); OOS Rank IC reversed/zero (-0.0027); cost/turnover eroded net alpha |
| `PULLBACK_DEPTH_SMA50` | Market_Structure | `REJECTED` | weak IS Rank IC (+0.0013); failed BH FDR (q=0.9321); OOS Rank IC reversed/zero (-0.0083); non-monotonic quintiles (+0.30); cost/turnover eroded net alpha |
| `SECTOR_MOM_20D` | Sector | `REJECTED` | weak IS Rank IC (-0.0012); failed BH FDR (q=0.9646); OOS Rank IC reversed/zero (-0.0051); non-monotonic quintiles (+0.10) |
| `OBV_FLOW_20D` | Volume_Liquidity | `REJECTED` | weak IS Rank IC (-0.0071); failed BH FDR (q=0.6861); OOS Rank IC reversed/zero (-0.0205) |
| `SIZE_LOG_MCAP_PIT` | Fundamental | `REJECTED` | weak IS Rank IC (-0.0628); failed BH FDR (q=0.1398); OOS Rank IC reversed/zero (-0.0611); non-monotonic quintiles (-0.90); cost/turnover eroded net alpha |
| `FUND_EARNINGS_SURPRISE_PIT` | Fundamental | `REJECTED_UNAVAILABLE_PIT` | NSE Bhavcopy & PR archives do not carry historical multi-year quarterly EPS filing timestamps. Backfilling static 2026 Yahoo/screener financials across 2021-2025 introduces severe look-ahead bias and is strictly forbidden. |
| `FUND_ROE_PROFITABILITY_PIT` | Fundamental | `REJECTED_UNAVAILABLE_PIT` | Point-in-time historical Balance Sheet / Net Worth filing snapshots across 2021-2025 are unavailable without historical XBRL filings. |
| `FUND_SALES_PROFIT_GROWTH_PIT` | Fundamental | `REJECTED_UNAVAILABLE_PIT` | Point-in-time YoY quarterly sales/PAT announcements with original publication & restatement dates are absent in NSE EOD archives. |
| `FUND_DEBT_CASHFLOW_QUALITY_PIT` | Fundamental | `REJECTED_UNAVAILABLE_PIT` | Requires historical annual/half-yearly cash-flow and debt schedules indexed strictly by filing date. |
