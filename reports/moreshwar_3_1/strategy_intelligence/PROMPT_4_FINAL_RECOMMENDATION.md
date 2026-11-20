# MORESHWAR 3.1 (PROMPT 4/5) — FINAL RESEARCH OUTCOME & DECISION REPORT

> [!IMPORTANT]
> **PROMPT 4/5 FINAL CLASSIFICATION: `CONDITIONAL GO`**
> - **Frozen Prospective Paper-Trading Configuration:** **`G_SIMPLE_TREND_BASELINE`** (`G. Simple Trend Baseline (NIFTYBEES >= SMA200: 60/30/10 else 45/10/45)`)
> - **Full Period (`2021-07 -> 2026-11-19`):** Net **`+120.03%`** (CAGR **`+16.10%`**, Max DD **`10.46%`**, Sharpe **`+0.87`**, Calmar **`1.54`**, Cost **`INR 3,962.18`**, Ledger Diff **`INR +0.01`**) vs Executable `NIFTYBEES` **`+68.60%`** (`16.02%` DD, Sharpe `+0.33`).
> - **Live Trading / Broker Execution:** **STRICTLY DISABLED** (`live_trading_enabled: false`).

## Direct Answers to All 10 Mandatory Section 14 Questions

1. **Does regime detection improve the Prompt 3 primary configuration?** **Yes modestly** on drawdown/risk-adjusted metrics.
2. **Does Family E have a measurable conditional edge?** **Yes.** Across `2021-07 -> 2026-11`, Family E generated `INR +29,055` net tactical profit (incl. dividends), outperforming `NIFTYBEES` on a full-period risk-adjusted basis (block-bootstrap `P(Excess CAGR > 0) = 0.950`), though the edge varies meaningfully by regime.
3. **Which market regimes are favourable or unfavourable for Family E?** Most favourable relative to `NIFTYBEES`: **`SIDEWAYS_RANGE`** (`+13.35 pp` relative return, PF `1.26`). Most challenging relative to `NIFTYBEES`: **`HIGH_VOL_STRESS`** (`-4.52 pp` relative return).
4. **Does adaptive allocation improve net benchmark-relative performance?** Best adaptive variant (`G_SIMPLE_TREND_BASELINE`) delivered `+47.99 pp` full vs Static Primary's `+22.80 pp`, and `+8.99 pp` outer vs Static Primary's `+3.01 pp`.
5. **Does the adaptive approach reduce drawdown without sacrificing too much return?** Defensive policies (`C_RULE_ADAPTIVE_FULL`, `F_DEFENSIVE_STRESS_HALT`) reduced full drawdown to `13.12%` / `10.15%` (vs Static `13.82%`), at the trade-off of giving up upside during sharp post-stress rebounds.
6. **Did any new strategy genuinely qualify for inclusion?** **0 of 5 candidates passed all 6 admission gates.** None beat the Family E blend after costs, outlier checks, and outer-fold validation. Family E is retained as the sole tactical strategy, and Family A remains suspended.
7. **Is the improvement consistent across outer folds?** Across the 5 chronological outer folds, Static Primary beat `NIFTYBEES` in `3/5` folds and Inner-Selected Adaptive beat `NIFTYBEES` in `4/5` folds — proving performance is cycle-dependent rather than uniform in every 12-month slice.
8. **What are the largest remaining failure modes?** (a) Narrow mega-cap rallies (`BULL_NARROW`, e.g., 2026 sub-tail) where defensive yield/low-vol lags Nifty 50; (b) `daily_index_bars` ending on `2025-10-10` (India VIX / GS 10Yr unavailable post-Oct-2025); (c) absence of PIT earnings/financial statements; (d) cumulative multiple-testing risk across 250+ historical experiments (`DSR Prob = 0.774`); (e) lack of intraday tick/order-book data.
9. **Which configuration should be frozen for prospective paper trading?** **`G_SIMPLE_TREND_BASELINE`** (recorded verbatim in `FROZEN_PROSPECTIVE_PAPER_TRADING_CONFIG.json`, SHA prefix `47a30f4bbd004c5f`), retaining simplicity over unverified regime complexity.
10. **What evidence is still required before any live deployment can be considered?** Minimum `60–120` trading days of unmodified prospective paper-trading observations logged against live NSE fills/dividends and real `NIFTYBEES` / `45/55` benchmarks, verified `0.00 INR` accounting reconciliation in paper execution, and passing Prompt 5/5 release-readiness & operational resilience certification.
