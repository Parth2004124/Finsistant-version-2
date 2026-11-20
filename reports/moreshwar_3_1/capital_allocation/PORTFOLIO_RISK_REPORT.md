# MORESHWAR 3.1 (PROMPT 2/5) — PORTFOLIO-LEVEL RISK MANAGEMENT & 3-TIER RISK REPORT

## 1. Distinction Between Individual Strategy, Combined Portfolio & Market-Wide Systemic Risk

| Risk Layer | Measured Metric (`Exp I Full Window`) | Value | Governance Control |
| :--- | :--- | :---: | :--- |
| **1. Individual Strategy Risk** | Active Sleeves (`Family E` + `Family A`) | `47` trades, PF `1.78`, Win Rate `42.6%` | Per-sleeve budget cap, initial ATR stop-loss, wide `5.0x ATR` trailing stop, `10d` post-stop cooldown |
| **2. Combined Portfolio Risk** | Annualized Vol / Max Drawdown / 95% Daily CVaR | Vol `10.53%`, Max DD **`13.37%`**, CVaR95 `-1.49%` | Pairwise 60D correlation gate (`rho <= 0.72`), Euler diversification ratio `1.296`, 4-tier sector/industry/unknown caps |
| **3. Market-Wide Systemic Risk** | Portfolio Beta vs Nifty 50 / Systemic Variance Share ($R^2$) | Beta **`0.728`**, Systemic $R^2$ **`90.0%`** | Causal 5-state regime modulation expanding `LIQUIDBEES` reserve in `WEAK_BEAR` / `STRESS_BEAR` + drawdown governor |

## 2. Individual Tactical Sleeve Attribution (`Exp I`)

| Strategy Sleeve | Trades | Win Rate % | Net PnL incl. Dividends (`INR`) | Cash Dividends Received (`INR`) | Tactical Friction Costs (`INR`) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `STRAT_E_DEFENSIVE_YIELD_LOWVOL` | `42` | `45.2%` | **`INR +14,251.57`** | `INR 3,020.53` | `INR 2,223.26` |
| `STRAT_A_CS_MOMENTUM` | `5` | `20.0%` | **`INR -634.63`** | `INR 69.30` | `INR 211.47` |

## 3. 4-Tier Sector Classification Audit & Blocked-Trade Rejection Diagnostics

- **Unclassified Securities Reduced:** `2,085` (`73.62%` of universe, `34.91%` of PIT-eligible stock-days) -> **`895` (`31.60%` of universe, `14.51%` of PIT-eligible stock-days)**.
- **Tier Breakdown:** `Tier 1 (Validated NSE Index) = 747`, `Tier 2 (Reference Mapping) = 32`, `Tier 3 (Company-Name Taxonomy) = 1,158`, `Tier 4 (Explicit UNKNOWN_SECTOR) = 895`.
- **Sector Bottleneck Impact (Exp I Ablation):** Restoring M3.0's legacy `Unclassified` bucket causes `1` sector-cap rejections vs `1` sector-cap + `0` industry-cap + `0` unknown-cap rejections under the 4-tier hierarchy.

| Pre-Trade Rejection / Protection Code | Count (`Exp I Full Window`) |
| :--- | :---: |
| `SECTOR_CAP_BLOCKED` | `1` |
| `INDUSTRY_CAP_BLOCKED` | `0` |
| `UNKNOWN_SECTOR_CAP_BLOCKED` | `0` |
| `CORRELATION_CAP_BLOCKED` | `1` |
| `MAX_POSITIONS_BLOCKED` | `0` |
| `COST_ALPHA_HURDLE_BLOCKED` | `0` |
| `ROTATION_UPGRADE_HURDLE_SKIPPED` | `0` |
| `MIN_LOT_SIZE_BLOCKED` | `6` |
| `LIQUIDITY_PARTIAL_FILLS` | `0` |
| `EX_DIV_STOP_PROTECTIONS` | `14` |
