# MORESHWAR 3.1 (PROMPT 2/5) — DIVIDEND & CORPORATE ACTION ACCOUNTING VALIDATION

## 1. Accounting Invariants Enforced

1. **Zero Double-Counting on Total-Return ETFs (`NIFTYBEES`):** `NIFTYBEES` (`INF204KB14I2`) is a real NSE accumulating ETF whose traded market price (`INR 168.92 -> INR 290.61`, `+72.04%`) already embeds reinvested Nifty 50 constituent dividends net of TER. Cash dividend crediting is strictly **disabled** on `NIFTYBEES`.
2. **Exact Ex-Date Cash Dividend Crediting on Individual Equities (`ti > entry_ti`):** Because `builder.py` adjusts `adj_open/high/low/close` strictly for `SPLIT` and `BONUS` (not `DIVIDEND`), held tactical positions credit `qty * (div_inr_ex[ti, col] / adj_factor)` directly into `settled_cash` on `ex_date`.
3. **Ex-Dividend Stop-Loss Protection (`EX_DIV_STOP_PROTECTIONS`):** On `ex_date`, `pos['sl_px']` and `pos['max_px']` are adjusted downward by `div_per_adj_share` BEFORE evaluating intraday stop triggers, preventing artificial ex-dividend stop-outs.
4. **Split-Adjusted Causal `DIV_YIELD_TTM_PIT`:** Computed via `compute_split_adjusted_div_yield_ttm` (`div_per_adj_share` summed over trailing 252 sessions divided by `adj_close`), eliminating all `1,704` split-inflation stock-days while keeping strictly `<= ti` ex-dates.

## 2. Quantified Dividend Impact (Experiment A vs Experiment B & Experiment I)

| Portfolio / Experiment | Price-Only Return | Cash Dividends Credited (`INR`) | Total Return With Dividends | Ex-Div Stop Protections | Ledger Reconciliation Diff (`INR`) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Exp A: Original M3.0 (Dividends Omitted)** | `+7.43%` | `INR 0.00` | `+7.43%` | `0` | `INR 0.00` |
| **Exp B: Original M3.0 + Verified Dividends** | `+7.43%` | **`INR 6,078.93`** (`59` events) | **`+13.51%`** (`+6.08 pp`) | Verified | `INR 0.00` |
| **Exp I: Full Intelligent M3.1 Architecture** | `+78.32%` | **`INR 3,089.83`** (`14` events) | **`+81.41%`** | **`14`** | **`INR +0.00`** |

## 3. Sample Verified Ex-Date Cash Dividend & Stop-Adjustment Audit Trail (`Exp I`)

| Ex-Date | Symbol | Qty | Split-Adj Div / Share (`INR`) | Cash Credited (`INR`) | Old Stop (`INR`) | Adjusted Stop (`INR`) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| `2022-02-16` | `POWERGRID` | `69` | `4.12` | **`INR 284.63`** | `147.48` | `143.35` |
| `2022-05-26` | `ITC` | `52` | `6.25` | **`INR 325.00`** | `247.51` | `241.26` |
| `2022-07-14` | `M&M` | `6` | `11.55` | **`INR 69.30`** | `1044.11` | `1032.56` |
| `2022-11-15` | `COALINDIA` | `39` | `15.00` | **`INR 585.00`** | `237.28` | `222.28` |
| `2023-02-03` | `NTPC` | `60` | `4.25` | **`INR 255.00`** | `154.95` | `150.70` |
| `2023-08-25` | `OIL` | `72` | `3.67` | **`INR 264.00`** | `184.17` | `180.50` |
| `2023-10-20` | `HCLTECH` | `10` | `12.00` | **`INR 120.00`** | `1172.47` | `1160.47` |
| `2024-01-19` | `HCLTECH` | `10` | `12.00` | **`INR 120.00`** | `1510.58` | `1498.58` |
| `2024-05-22` | `SBIN` | `22` | `13.70` | **`INR 301.40`** | `753.39` | `739.69` |
| `2024-07-19` | `TECHM` | `10` | `28.00` | **`INR 280.00`** | `1401.39` | `1373.39` |
| `2024-10-31` | `TECHM` | `10` | `15.00` | **`INR 150.00`** | `1617.54` | `1602.54` |
| `2025-02-10` | `ASTERDM` | `17` | `4.00` | **`INR 68.00`** | `432.12` | `428.12` |
| `2026-07-23` | `PIDILITIND` | `5` | `11.50` | **`INR 57.50`** | `1476.45` | `1464.95` |
| `2026-11-06` | `PETRONET` | `30` | `7.00` | **`INR 210.00`** | `287.25` | `280.25` |
