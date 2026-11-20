# MORESHWAR 3.1 (PROMPT 4/5) — STATISTICAL ROBUSTNESS, BOOTSTRAP & MULTIPLE-TESTING AUDIT

## 1. Circular Block Bootstrap (20-Session Blocks, `B = 1,000` Resamples)

- **Frozen Winner (`G_SIMPLE_TREND_BASELINE`) vs Executable `NIFTYBEES`:**
  - CAGR 95% CI: **`[+6.44%, +27.71%]`**
  - Sharpe 95% CI: **`[+0.03, +1.78]`**
  - Excess CAGR vs `NIFTYBEES` 95% CI: **`[-0.95 pp, +12.00 pp]`** | `P(Excess CAGR > 0)` = **`0.950`**
- **Best Adaptive Policy (`G_SIMPLE_TREND_BASELINE`) vs Static Prompt 3 Primary (`A_STATIC_PROMPT3_PRIMARY`):**
  - Excess CAGR (Adaptive minus Static) 95% CI: **`[-1.33 pp, +6.77 pp]`** | `P(Adaptive > Static)` = **`0.915`**

## 2. Outlier & Concentration Audit

| Portfolio | Full Return | Excl. Top 1 Trade | Excl. Top 3 Trades | Excl. Top 5 Trades | Largest Winning Security | Excl. Largest Security |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `G_SIMPLE_TREND_BASELINE` | `+120.03%` | `+106.55%` | `+95.62%` | `+87.89%` | `HINDZINC` (`INR +13,744`) | `+106.28%` |
| `G_SIMPLE_TREND_BASELINE` | `+120.03%` | `+106.55%` | `+95.62%` | `+87.89%` | `HINDZINC` (`INR +13,744`) | `+106.28%` |

- **Top-5 Tactical Sectors (`EXP_E` Static Primary):** `Oil Gas & Consumable Fuels` (`INR +12,016`), `Fast Moving Consumer Goods` (`INR +6,355`), `Financial Services` (`INR +4,821`), `Power` (`INR +4,790`), `Services` (`INR +2,843`)

## 3. Regime Threshold Perturbation Sensitivity (`C_RULE_ADAPTIVE_FULL`)

| Perturbation | Full Return | Full Max DD | Full Sharpe | Total Cost | Outer (`2024-07->2026-11`) Ret | Outer DD | Outer Sharpe |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `SIMPLE_TREND_SMA_150` | `+100.29%` | `11.43%` | `+0.73` | `INR 4,317` | `+12.21%` | `10.62%` | `-0.14` |
| `SIMPLE_TREND_SMA_180` | `+126.24%` | `13.48%` | `+0.92` | `INR 4,772` | `+13.16%` | `10.96%` | `-0.10` |
| `SIMPLE_TREND_SMA_200` | `+120.03%` | `10.46%` | `+0.87` | `INR 3,962` | `+15.74%` | `9.73%` | `+0.01` |
| `SIMPLE_TREND_SMA_220` | `+92.03%` | `12.03%` | `+0.63` | `INR 3,787` | `+13.82%` | `11.69%` | `-0.07` |
| `SIMPLE_TREND_SMA_250` | `+87.13%` | `11.85%` | `+0.58` | `INR 3,934` | `+12.56%` | `11.99%` | `-0.12` |
| `RULE_6STATE_BASE (3d,18.5%,-10.5%)` | `+74.90%` | `13.12%` | `+0.47` | `INR 5,125` | `+3.73%` | `11.68%` | `-0.52` |
| `RULE_CONFIRM_1D (1d,18.5%,-10.5%)` | `+64.16%` | `15.29%` | `+0.33` | `INR 4,929` | `+10.88%` | `10.13%` | `-0.20` |
| `RULE_CONFIRM_5D (5d,18.5%,-10.5%)` | `+71.38%` | `13.72%` | `+0.43` | `INR 4,609` | `+2.00%` | `12.91%` | `-0.60` |
| `RULE_STRESS_TIGHT (3d,16.5%,-9.0%)` | `+75.85%` | `12.32%` | `+0.49` | `INR 5,489` | `+2.85%` | `12.17%` | `-0.59` |
| `RULE_STRESS_LOOSE (3d,20.5%,-12.5%)` | `+73.22%` | `12.15%` | `+0.45` | `INR 5,106` | `+2.22%` | `13.28%` | `-0.59` |

## 4. Cumulative Multiple-Testing & Deflated Sharpe Ratio Audit

- **Cumulative Trials Accounted For:** `204` (Prompt 3) + `161` (Prompt 4) = **`365` raw runs**.
- **Correlated-Trial Adjustment:** Because parameter sweeps and nested fold evaluations share overlapping daily returns, effective independent trials are approximated as `N_eff = 164` (`45%` independence haircut).
- **Observed Sharpe:** `+0.870` | **Expected Null Max Sharpe (`N_eff`):** `+0.486` | **Deflated Sharpe Probability:** **`0.7737`**.
- **Strict Interpretation Caveat:** A Deflated Sharpe Probability below `0.90` confirms that while historical risk-adjusted performance is economically encouraging, **historical backtests alone CANNOT prove future out-of-sample profitability**. Prospective paper trading on truly unseen future sessions is mandatory.
