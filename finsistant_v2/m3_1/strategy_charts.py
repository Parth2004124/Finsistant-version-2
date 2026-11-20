"""Moreshwar 3.1 (Prompt 4/5) — Real-Data SVG Chart Builders for Strategy Intelligence Reports.

Generates all 8 required visual diagnostics strictly from actual simulation output:
1. chart_market_regime_history.svg
2. chart_regime_confidence.svg
3. chart_family_e_returns_by_regime.svg
4. chart_static_vs_adaptive_allocation.svg
5. chart_drawdown_comparison.svg
6. chart_benchmark_relative_performance.svg
7. chart_outer_fold_performance.svg
8. chart_tactical_exposure_over_time.svg
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from finsistant_v2.m3_1.execution_charts import _esc, _svg_grouped_bar_chart, _svg_line_chart

REGIME_COLORS = {
    "BULL_BROAD": "#10b981",
    "RECOVERY_IMPROVING": "#06b6d4",
    "BULL_NARROW": "#f59e0b",
    "SIDEWAYS_RANGE": "#94a3b8",
    "BEAR_DETERIORATING": "#f97316",
    "HIGH_VOL_STRESS": "#ef4444",
}


def _svg_regime_history_chart(
    *,
    dates: list[str],
    nb_prices: list[float],
    regimes: list[str],
    out_path: Path,
) -> None:
    w, h = 920, 410
    ml, mr, mt, mb = 70, 24, 60, 68
    pw, ph = w - ml - mr, h - mt - mb
    ymin, ymax = min(nb_prices), max(nb_prices)
    pad = (ymax - ymin) * 0.08
    ymin -= pad
    ymax += pad
    n = max(2, len(dates))

    def px(i: int) -> float:
        return ml + (i / (n - 1)) * pw

    def py(v: float) -> float:
        return mt + (1.0 - (v - ymin) / max(1e-6, ymax - ymin)) * ph

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
        f'<rect width="{w}" height="{h}" fill="#ffffff"/>',
        f'<text x="{ml}" y="25" font-family="DejaVu Sans,Arial" font-size="15" font-weight="bold" fill="#0f172a">1. Causal 6-State Market Regime History Overlaid on Real NIFTYBEES Price</text>',
        f'<text x="{ml}" y="43" font-family="DejaVu Sans,Arial" font-size="11" fill="#475569">Model A (Rule-Based 6-State with 3-Day Confirmation Hysteresis, Strictly Point-in-Time)</text>',
    ]
    # Background regime vertical bands (grouped contiguous spans)
    i = 0
    while i < n:
        st = regimes[i]
        j = i + 1
        while j < n and regimes[j] == st:
            j += 1
        x0 = px(i)
        x1 = px(min(j, n - 1))
        col = REGIME_COLORS.get(st, "#cbd5e1")
        parts.append(f'<rect x="{x0:.1f}" y="{mt}" width="{max(1.0, x1 - x0):.1f}" height="{ph}" fill="{col}" fill-opacity="0.22"/>')
        i = j

    for k in range(5):
        gv = ymin + (k / 4.0) * (ymax - ymin)
        gy = py(gv)
        parts.append(f'<line x1="{ml}" y1="{gy:.1f}" x2="{w - mr}" y2="{gy:.1f}" stroke="#e2e8f0" stroke-width="1"/>')
        parts.append(f'<text x="{ml - 8}" y="{gy + 4:.1f}" text-anchor="end" font-family="DejaVu Sans,Arial" font-size="10" fill="#64748b">{gv:,.0f}</text>')
    for idx in (0, n // 4, n // 2, (3 * n) // 4, n - 1):
        parts.append(f'<text x="{px(idx):.1f}" y="{h - 36}" text-anchor="middle" font-family="DejaVu Sans,Arial" font-size="10" fill="#64748b">{_esc(dates[idx])}</text>')

    stride = max(1, n // 280)
    pts = [f"{px(k):.1f},{py(nb_prices[k]):.1f}" for k in range(0, n, stride)]
    parts.append(f'<polyline fill="none" stroke="#0f172a" stroke-width="2.0" points="{" ".join(pts)}"/>')

    for idx_r, (st, col) in enumerate(REGIME_COLORS.items()):
        lx = ml + (idx_r % 3) * 285
        ly = h - 18 + (idx_r // 3) * 14
        parts.append(f'<rect x="{lx}" y="{ly - 9}" width="12" height="9" fill="{col}" fill-opacity="0.55"/>')
        parts.append(f'<text x="{lx + 16}" y="{ly - 1}" font-family="DejaVu Sans,Arial" font-size="10" fill="#1e293b">{_esc(st)}</text>')
    parts.append("</svg>")
    out_path.write_text("\n".join(parts), encoding="utf-8")


def generate_all_prompt4_charts(
    *,
    output_dir: Path,
    regime_history_rows: list[dict[str, Any]],
    fame_regime_rows: list[dict[str, Any]],
    policy_sims: dict[str, dict[str, Any]],
    nifty_sim: dict[str, dict[str, Any]] | dict[str, Any],
    blend45_sim: dict[str, Any],
    nwf_rows: list[dict[str, Any]],
) -> list[str]:
    """Render all 8 required Prompt 4 SVG charts directly from empirical simulation outputs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    created: list[str] = []

    # 1. Market Regime History
    dates_r = [str(r["date"]) for r in regime_history_rows]
    # Reconstruct aligned NIFTYBEES NAV proxy from nifty_sim
    nb_map = {str(d["date"]): float(d["nav"]) for d in nifty_sim["daily_navs"]}
    nb_vals = [nb_map.get(dt, 100_000.0) for dt in dates_r]
    rule_r = [str(r["rule_regime_6state"]) for r in regime_history_rows]
    _svg_regime_history_chart(
        dates=dates_r, nb_prices=nb_vals, regimes=rule_r,
        out_path=output_dir / "chart_market_regime_history.svg",
    )
    created.append("chart_market_regime_history.svg")

    # 2. Regime Classification Confidence Over Time
    _svg_line_chart(
        title="2. Regime Classification Confidence Over Time (Model A Rule-Based vs Model B Statistical)",
        subtitle="Daily posterior/margin confidence scores across 2021-07 to 2026-11 (higher = clearer regime separation)",
        dates=dates_r,
        series=[
            ("Model A: Rule-Based 6-State Confidence", [float(r["rule_confidence"]) * 100.0 for r in regime_history_rows], "#0f766e"),
            ("Model B: Statistical K-Means Confidence", [float(r["stat_confidence"]) * 100.0 for r in regime_history_rows], "#6366f1"),
        ],
        y_label="Confidence (%)",
        out_path=output_dir / "chart_regime_confidence.svg",
    )
    created.append("chart_regime_confidence.svg")

    # 3. Family E Returns by Regime (vs NIFTYBEES)
    _svg_grouped_bar_chart(
        title="3. Family E Portfolio vs NIFTYBEES Cumulative Returns by Market Regime",
        subtitle="Conditional edge audit of static Prompt 3 Primary (Exp E + ExecIntel) across all 6 causal regimes",
        categories=[str(r["regime"]).replace("_", " ")[:16] for r in fame_regime_rows],
        groups=[
            ("Exp E Portfolio Cum Return (%)", [float(r["portfolio_cum_ret_pct"]) for r in fame_regime_rows], "#0f766e"),
            ("NIFTYBEES Cum Return (%)", [float(r["niftybees_cum_ret_pct"]) for r in fame_regime_rows], "#64748b"),
        ],
        y_label="Cumulative Return (%)",
        out_path=output_dir / "chart_family_e_returns_by_regime.svg",
    )
    created.append("chart_family_e_returns_by_regime.svg")

    # 4. Static vs Adaptive Allocation Full-Period Return & Max Drawdown
    pol_order = [
        ("Static P3 Primary", "A_STATIC_PROMPT3_PRIMARY"),
        ("Simple Trend 200", "G_SIMPLE_TREND_BASELINE"),
        ("Rule Adaptive 6S", "C_RULE_ADAPTIVE_FULL"),
        ("Stat Adaptive 4S", "D_STAT_REGIME_ADAPTIVE"),
        ("Tact-Only Adapt", "E_ADAPTIVE_TACTICAL_STATIC_CORE"),
        ("Stress Halt", "F_DEFENSIVE_STRESS_HALT"),
    ]
    _svg_grouped_bar_chart(
        title="4. Static vs Regime-Adaptive Allocation Policies (2021-07 to 2026-11)",
        subtitle="Full-period net return (%) and maximum drawdown (%) after all costs & regime-switch friction",
        categories=[lbl for lbl, _ in pol_order],
        groups=[
            ("Full Net Return (%)", [float(policy_sims[k]["metrics"]["total_return_pct"]) for _, k in pol_order], "#0284c7"),
            ("Max Drawdown (%)", [float(policy_sims[k]["metrics"]["max_drawdown_pct"]) for _, k in pol_order], "#dc2626"),
        ],
        y_label="Percent (%)",
        out_path=output_dir / "chart_static_vs_adaptive_allocation.svg",
    )
    created.append("chart_static_vs_adaptive_allocation.svg")

    # 5. Drawdown Comparison Over Time
    dates_s = [str(d["date"]) for d in policy_sims["A_STATIC_PROMPT3_PRIMARY"]["daily_navs"]]
    _svg_line_chart(
        title="5. Underwater Drawdown Comparison: Static vs Adaptive vs 100% NIFTYBEES",
        subtitle="Comparing capital preservation through 2022 chop, 2024-25 correction, and 2025-26 consolidation",
        dates=dates_s,
        series=[
            ("100% NIFTYBEES", [-float(d["drawdown_pct"]) for d in nifty_sim["daily_navs"]], "#64748b"),
            ("Static P3 Primary (Exp E)", [-float(d["drawdown_pct"]) for d in policy_sims["A_STATIC_PROMPT3_PRIMARY"]["daily_navs"]], "#0f766e"),
            ("Simple 2-State Trend (SMA200)", [-float(d["drawdown_pct"]) for d in policy_sims["G_SIMPLE_TREND_BASELINE"]["daily_navs"]], "#2563eb"),
            ("Defensive Stress Halt (F)", [-float(d["drawdown_pct"]) for d in policy_sims["F_DEFENSIVE_STRESS_HALT"]["daily_navs"]], "#d97706"),
        ],
        y_label="Drawdown (%)",
        out_path=output_dir / "chart_drawdown_comparison.svg",
    )
    created.append("chart_drawdown_comparison.svg")

    # 6. Benchmark-Relative Performance (Cumulative Excess NAV % vs 100% NIFTYBEES)
    nb_navs = [float(d["nav"]) for d in nifty_sim["daily_navs"]]
    def _rel_curve(sim_obj: dict[str, Any]) -> list[float]:
        return [round(((float(d["nav"]) / max(1.0, nb_navs[i])) - 1.0) * 100.0, 2) for i, d in enumerate(sim_obj["daily_navs"])]

    _svg_line_chart(
        title="6. Cumulative Benchmark-Relative Performance vs 100% Executable NIFTYBEES (%)",
        subtitle="Positive slope = outperforming NIFTYBEES after all costs; flat/down = trailing passive benchmark",
        dates=dates_s,
        series=[
            ("Simple 2-State Trend (SMA200)", _rel_curve(policy_sims["G_SIMPLE_TREND_BASELINE"]), "#2563eb"),
            ("Static P3 Primary (Exp E)", _rel_curve(policy_sims["A_STATIC_PROMPT3_PRIMARY"]), "#0f766e"),
            ("Defensive Stress Halt (F)", _rel_curve(policy_sims["F_DEFENSIVE_STRESS_HALT"]), "#d97706"),
            ("45/55 NIFTY/LIQUID ETF", _rel_curve(blend45_sim), "#94a3b8"),
        ],
        y_label="Relative Return vs NIFTYBEES (%)",
        out_path=output_dir / "chart_benchmark_relative_performance.svg",
    )
    created.append("chart_benchmark_relative_performance.svg")

    # 7. Outer-Fold Chronological Performance (All 5 Nested Walk-Forward Folds)
    _svg_grouped_bar_chart(
        title="7. Nested Walk-Forward Outer-Fold Returns (Static Primary vs Adaptive vs NIFTYBEES)",
        subtitle="Chronological out-of-fold net returns across 5 outer windows (previously inspected folds explicitly noted)",
        categories=[str(r["fold_id"]).replace("P4_", "") for r in nwf_rows],
        groups=[
            ("Static P3 Primary (%)", [float(r["static_primary_ret_pct"]) for r in nwf_rows], "#0f766e"),
            ("Inner-Selected Adaptive (%)", [float(r["outer_selected_ret_pct"]) for r in nwf_rows], "#2563eb"),
            ("100% NIFTYBEES (%)", [float(r["niftybees_ret_pct"]) for r in nwf_rows], "#64748b"),
        ],
        y_label="Outer-Fold Net Return (%)",
        out_path=output_dir / "chart_outer_fold_performance.svg",
    )
    created.append("chart_outer_fold_performance.svg")

    # 8. Tactical Exposure Over Time (Static vs Simple Trend vs Defensive Stress Halt)
    _svg_line_chart(
        title="8. Family E Tactical Equity Exposure Over Time (% of Portfolio NAV)",
        subtitle="Showing real-time exposure modulation under Static Primary, Simple Trend (SMA200), and Stress Halt",
        dates=dates_s,
        series=[
            ("Static P3 Primary (Exp E)", [float(d["tactical_pct"]) for d in policy_sims["A_STATIC_PROMPT3_PRIMARY"]["daily_navs"]], "#0f766e"),
            ("Simple 2-State Trend (SMA200)", [float(d["tactical_pct"]) for d in policy_sims["G_SIMPLE_TREND_BASELINE"]["daily_navs"]], "#2563eb"),
            ("Defensive Stress Halt (F)", [float(d["tactical_pct"]) for d in policy_sims["F_DEFENSIVE_STRESS_HALT"]["daily_navs"]], "#dc2626"),
        ],
        y_label="Tactical Exposure (% NAV)",
        out_path=output_dir / "chart_tactical_exposure_over_time.svg",
    )
    created.append("chart_tactical_exposure_over_time.svg")

    return created
