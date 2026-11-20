"""Moreshwar 3.1 (Prompt 3/5) — Pure-Python SVG Chart Builders for Execution Intelligence Reports.

Generates all 8 required visual diagnostics:
1. chart_gross_vs_net_returns.svg
2. chart_transaction_costs_over_time.svg
3. chart_trade_profitability_distribution.svg
4. chart_holding_period_distribution.svg
5. chart_stop_loss_effectiveness.svg
6. chart_portfolio_turnover.svg
7. chart_drawdown_comparison.svg
8. chart_alpha_by_strategy_family.svg
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _esc(s: str) -> str:
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _svg_line_chart(
    *,
    title: str,
    subtitle: str,
    dates: list[str],
    series: list[tuple[str, list[float], str]],
    y_label: str,
    out_path: Path,
    invert_y: bool = False,
) -> None:
    w, h = 920, 400
    ml, mr, mt, mb = 74, 24, 62, 58
    pw, ph = w - ml - mr, h - mt - mb
    all_vals = [v for _, ys, _ in series for v in ys]
    ymin = min(all_vals) if all_vals else 0.0
    ymax = max(all_vals) if all_vals else 1.0
    if abs(ymax - ymin) < 1e-6:
        ymax = ymin + 1.0
    pad = (ymax - ymin) * 0.08
    ymin -= pad
    ymax += pad
    n = max(2, len(dates))

    def px(i: int) -> float:
        return ml + (i / (n - 1)) * pw

    def py(v: float) -> float:
        frac = (v - ymin) / (ymax - ymin)
        return mt + frac * ph if invert_y else mt + (1.0 - frac) * ph

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
        f'<rect width="{w}" height="{h}" fill="#ffffff"/>',
        f'<text x="{ml}" y="26" font-family="DejaVu Sans,Arial" font-size="15" font-weight="bold" fill="#0f172a">{_esc(title)}</text>',
        f'<text x="{ml}" y="44" font-family="DejaVu Sans,Arial" font-size="11" fill="#475569">{_esc(subtitle)}</text>',
    ]
    for k in range(5):
        gv = ymin + (k / 4.0) * (ymax - ymin)
        gy = py(gv)
        parts.append(f'<line x1="{ml}" y1="{gy:.1f}" x2="{w - mr}" y2="{gy:.1f}" stroke="#e2e8f0" stroke-width="1"/>')
        parts.append(f'<text x="{ml - 8}" y="{gy + 4:.1f}" text-anchor="end" font-family="DejaVu Sans,Arial" font-size="10" fill="#64748b">{gv:,.1f}</text>')
    for idx in (0, n // 4, n // 2, (3 * n) // 4, n - 1):
        lx = px(idx)
        parts.append(f'<text x="{lx:.1f}" y="{h - 24}" text-anchor="middle" font-family="DejaVu Sans,Arial" font-size="10" fill="#64748b">{_esc(dates[min(idx, len(dates)-1)])}</text>')
    parts.append(f'<text x="16" y="{mt + ph / 2:.1f}" transform="rotate(-90 16 {mt + ph / 2:.1f})" text-anchor="middle" font-family="DejaVu Sans,Arial" font-size="11" fill="#334155">{_esc(y_label)}</text>')

    stride = max(1, n // 260)
    for idx_s, (label, ys, color) in enumerate(series):
        pts = [f"{px(i):.1f},{py(ys[i]):.1f}" for i in range(0, min(len(ys), n), stride)]
        if len(ys) > 1 and (len(ys) - 1) % stride != 0:
            pts.append(f"{px(len(ys)-1):.1f},{py(ys[-1]):.1f}")
        parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="2.1" points="{" ".join(pts)}"/>')
        lx = ml + (idx_s % 3) * 285
        ly = h - 7 - (idx_s // 3) * 14
        parts.append(f'<rect x="{lx}" y="{ly - 8}" width="12" height="4" fill="{color}"/>')
        parts.append(f'<text x="{lx + 16}" y="{ly - 4}" font-family="DejaVu Sans,Arial" font-size="10" fill="#1e293b">{_esc(label)}</text>')
    parts.append("</svg>")
    out_path.write_text("\n".join(parts), encoding="utf-8")


def _svg_grouped_bar_chart(
    *,
    title: str,
    subtitle: str,
    categories: list[str],
    groups: list[tuple[str, list[float], str]],
    y_label: str,
    out_path: Path,
) -> None:
    w, h = 920, 400
    ml, mr, mt, mb = 74, 24, 62, 82
    pw, ph = w - ml - mr, h - mt - mb
    all_v = [0.0] + [v for _, vs, _ in groups for v in vs]
    ymin = min(all_v)
    ymax = max(all_v)
    if abs(ymax - ymin) < 1e-6:
        ymax = ymin + 1.0
    pad = (ymax - ymin) * 0.10
    ymin = min(0.0, ymin - pad) if ymin < 0 else 0.0
    ymax += pad

    def py(v: float) -> float:
        return mt + (1.0 - (v - ymin) / (ymax - ymin)) * ph

    y0 = py(0.0)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
        f'<rect width="{w}" height="{h}" fill="#ffffff"/>',
        f'<text x="{ml}" y="26" font-family="DejaVu Sans,Arial" font-size="15" font-weight="bold" fill="#0f172a">{_esc(title)}</text>',
        f'<text x="{ml}" y="44" font-family="DejaVu Sans,Arial" font-size="11" fill="#475569">{_esc(subtitle)}</text>',
    ]
    for k in range(5):
        gv = ymin + (k / 4.0) * (ymax - ymin)
        gy = py(gv)
        parts.append(f'<line x1="{ml}" y1="{gy:.1f}" x2="{w - mr}" y2="{gy:.1f}" stroke="#e2e8f0" stroke-width="1"/>')
        parts.append(f'<text x="{ml - 8}" y="{gy + 4:.1f}" text-anchor="end" font-family="DejaVu Sans,Arial" font-size="10" fill="#64748b">{gv:,.1f}</text>')
    parts.append(f'<line x1="{ml}" y1="{y0:.1f}" x2="{w - mr}" y2="{y0:.1f}" stroke="#64748b" stroke-width="1.2"/>')
    parts.append(f'<text x="16" y="{mt + ph / 2:.1f}" transform="rotate(-90 16 {mt + ph / 2:.1f})" text-anchor="middle" font-family="DejaVu Sans,Arial" font-size="11" fill="#334155">{_esc(y_label)}</text>')

    nc = max(1, len(categories))
    ng = max(1, len(groups))
    slot_w = pw / nc
    bar_w = min(38.0, (slot_w * 0.76) / ng)

    for ci, cat in enumerate(categories):
        cx = ml + ci * slot_w + slot_w * 0.5
        start_x = cx - (ng * bar_w) / 2.0
        for gi, (_, vs, color) in enumerate(groups):
            v = vs[ci] if ci < len(vs) else 0.0
            bx = start_x + gi * bar_w
            yv = py(v)
            by = min(y0, yv)
            bh = max(1.0, abs(y0 - yv))
            parts.append(f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bar_w - 2:.1f}" height="{bh:.1f}" fill="{color}" rx="2"/>')
            val_y = (by - 4.0) if v >= 0 else (by + bh + 10.0)
            parts.append(f'<text x="{bx + (bar_w - 2) / 2:.1f}" y="{val_y:.1f}" text-anchor="middle" font-family="DejaVu Sans,Arial" font-size="9" fill="#0f172a">{v:.1f}</text>')
        parts.append(f'<text x="{cx:.1f}" y="{h - 46}" text-anchor="middle" font-family="DejaVu Sans,Arial" font-size="10" fill="#334155">{_esc(cat)}</text>')

    for gi, (glabel, _, color) in enumerate(groups):
        lx = ml + gi * 220
        parts.append(f'<rect x="{lx}" y="{h - 22}" width="12" height="10" fill="{color}" rx="2"/>')
        parts.append(f'<text x="{lx + 16}" y="{h - 13}" font-family="DejaVu Sans,Arial" font-size="10" fill="#1e293b">{_esc(glabel)}</text>')
    parts.append("</svg>")
    out_path.write_text("\n".join(parts), encoding="utf-8")


def _svg_histogram(
    *,
    title: str,
    subtitle: str,
    values: list[float],
    bin_edges: list[float],
    bin_labels: list[str],
    x_label: str,
    y_label: str,
    bar_color: str,
    out_path: Path,
) -> None:
    counts = [0 for _ in bin_labels]
    for v in values:
        placed = False
        for i in range(len(bin_labels)):
            lo = bin_edges[i]
            hi = bin_edges[i + 1]
            if (i == len(bin_labels) - 1 and lo <= v <= hi) or (lo <= v < hi):
                counts[i] += 1
                placed = True
                break
        if not placed and values:
            if v < bin_edges[0]:
                counts[0] += 1
            else:
                counts[-1] += 1
    _svg_grouped_bar_chart(
        title=title,
        subtitle=subtitle,
        categories=bin_labels,
        groups=[(x_label, [float(c) for c in counts], bar_color)],
        y_label=y_label,
        out_path=out_path,
    )


def generate_all_prompt3_charts(
    *,
    output_dir: Path,
    comparison_sims: dict[str, dict[str, Any]],
    stop_rows_E: list[dict[str, Any]],
    best_sim: dict[str, Any],
    base_I_sim: dict[str, Any],
    base_E_sim: dict[str, Any],
    nifty_sim: dict[str, Any],
) -> list[str]:
    """Render all 8 required SVG charts into `output_dir` and return generated file names."""
    output_dir.mkdir(parents=True, exist_ok=True)
    generated: list[str] = []

    # 1. Gross vs Net Returns across Key Architectures
    keys_for_gvn = [
        ("Exp I Base", base_I_sim),
        ("Exp I + ExecIntel", comparison_sims["EXP_I_COMBINED_EXEC_INTEL"]),
        ("Exp E Base", base_E_sim),
        ("Exp E + ExecIntel", comparison_sims["EXP_E_COMBINED_EXEC_INTEL"]),
        ("Challenger J (FamE)", comparison_sims["EXP_J_CHALLENGER_RB_65_25_10_FAME"]),
        ("Nifty BeES", nifty_sim),
    ]
    _svg_grouped_bar_chart(
        title="1. Gross vs Net Full-Period Returns (2021-07 to 2026-11)",
        subtitle="Quantifying exact execution friction drag across Prompt 2 baselines, Prompt 3 Execution Intelligence & Nifty BeES",
        categories=[k for k, _ in keys_for_gvn],
        groups=[
            ("Gross Return (%)", [float(s["metrics"]["gross_return_pct"]) for _, s in keys_for_gvn], "#94a3b8"),
            ("Net Return After All Costs (%)", [float(s["metrics"]["total_return_pct"]) for _, s in keys_for_gvn], "#0f766e"),
        ],
        y_label="Return (%)",
        out_path=output_dir / "chart_gross_vs_net_returns.svg",
    )
    generated.append("chart_gross_vs_net_returns.svg")

    # 2. Cumulative Transaction Costs Over Time
    dates = [str(d["date"]) for d in best_sim["daily_navs"]]
    _svg_line_chart(
        title="2. Cumulative Transaction Costs Over Time (INR)",
        subtitle="All statutory taxes (STT, Stamp, NSE, SEBI, GST, CDSL DP) + Spread & Market Impact across all 3 pillars",
        dates=dates,
        series=[
            ("Exp E Base (Prompt 2)", [float(d["cum_cost_inr"]) for d in base_E_sim["daily_navs"]], "#dc2626"),
            ("Exp I Base (Prompt 2)", [float(d["cum_cost_inr"]) for d in base_I_sim["daily_navs"]], "#ea580c"),
            ("Exp I + ExecIntel", [float(d["cum_cost_inr"]) for d in comparison_sims["EXP_I_COMBINED_EXEC_INTEL"]["daily_navs"]], "#2563eb"),
            ("Challenger J (FamE + ExecIntel)", [float(d["cum_cost_inr"]) for d in comparison_sims["EXP_J_CHALLENGER_RB_65_25_10_FAME"]["daily_navs"]], "#0f766e"),
            ("100% Nifty BeES", [float(d["cum_cost_inr"]) for d in nifty_sim["daily_navs"]], "#64748b"),
        ],
        y_label="Cumulative Costs (INR)",
        out_path=output_dir / "chart_transaction_costs_over_time.svg",
    )
    generated.append("chart_transaction_costs_over_time.svg")

    # 3. Trade Profitability Distribution (Best Combined Engine)
    trade_rets = [float(t["net_return_pct"]) for t in best_sim["trades"]]
    _svg_histogram(
        title=f"3. Trade Net Return Distribution — {best_sim['metrics']['config_id']}",
        subtitle=f"Total Closed Tactical Trades: {len(trade_rets)} | Win Rate: {best_sim['metrics']['win_rate_pct']}% | Profit Factor: {best_sim['metrics']['profit_factor']}",
        values=trade_rets,
        bin_edges=[-100.0, -10.0, -5.0, 0.0, 5.0, 15.0, 30.0, 500.0],
        bin_labels=["<-10%", "-10..-5%", "-5..0%", "0..+5%", "+5..+15%", "+15..+30%", ">+30%"],
        x_label="Closed Trades (Count)",
        y_label="Number of Trades",
        bar_color="#0284c7",
        out_path=output_dir / "chart_trade_profitability_distribution.svg",
    )
    generated.append("chart_trade_profitability_distribution.svg")

    # 4. Holding Period Distribution
    hold_days = [float(t["holding_days"]) for t in best_sim["trades"]]
    _svg_histogram(
        title=f"4. Tactical Holding-Period Distribution — {best_sim['metrics']['config_id']}",
        subtitle=f"Mean Hold: {best_sim['metrics']['avg_holding_days']} sessions (Winner retention & hysteresis prevent premature churn)",
        values=hold_days,
        bin_edges=[0.0, 10.0, 20.0, 40.0, 60.0, 90.0, 125.0],
        bin_labels=["1-10d", "11-20d", "21-40d", "41-60d", "61-90d", "91-120d"],
        x_label="Holding Period Bucket",
        y_label="Number of Trades",
        bar_color="#7c3aed",
        out_path=output_dir / "chart_holding_period_distribution.svg",
    )
    generated.append("chart_holding_period_distribution.svg")

    # 5. Stop-Loss Effectiveness (All 6 Stop Models A..F on Exp E Challenger)
    _svg_grouped_bar_chart(
        title="5. Stop-Loss & Exit Intelligence Comparison (Models A through F on Exp E)",
        subtitle="Full-Period Net Return (%) and Max Drawdown (%) across protective stop models (hard stops never removed)",
        categories=[str(r["stop_model"]).replace("_", " ")[:18] for r in stop_rows_E],
        groups=[
            ("Full Net Return (%)", [float(r["full_return_pct"]) for r in stop_rows_E], "#0f766e"),
            ("Max Drawdown (%)", [float(r["full_max_dd_pct"]) for r in stop_rows_E], "#dc2626"),
        ],
        y_label="Percent (%)",
        out_path=output_dir / "chart_stop_loss_effectiveness.svg",
    )
    generated.append("chart_stop_loss_effectiveness.svg")

    # 6. Portfolio Turnover Comparison (Total & Tactical Turnover)
    _svg_grouped_bar_chart(
        title="6. Portfolio Turnover Reduction (Prompt 2 Baselines vs Prompt 3 Execution Intelligence)",
        subtitle="Total portfolio turnover (x initial capital) & tactical equity turnover across architectures",
        categories=[k for k, _ in keys_for_gvn],
        groups=[
            ("Total Turnover (x)", [float(s["metrics"]["total_turnover_x"]) for _, s in keys_for_gvn], "#f59e0b"),
            ("Tactical Turnover (x)", [float(s["metrics"]["tactical_turnover_x"]) for _, s in keys_for_gvn], "#2563eb"),
        ],
        y_label="Turnover Multiple (x)",
        out_path=output_dir / "chart_portfolio_turnover.svg",
    )
    generated.append("chart_portfolio_turnover.svg")

    # 7. Drawdown Comparison Over Time
    _svg_line_chart(
        title="7. Underwater Drawdown Curves (2021-07 to 2026-11)",
        subtitle="Comparing drawdown resilience across Prompt 2 baselines, Prompt 3 Execution Intelligence & 100% Nifty BeES",
        dates=dates,
        series=[
            ("100% Nifty BeES", [-float(d["drawdown_pct"]) for d in nifty_sim["daily_navs"]], "#64748b"),
            ("Exp I Base (Prompt 2)", [-float(d["drawdown_pct"]) for d in base_I_sim["daily_navs"]], "#ea580c"),
            ("Exp E + ExecIntel", [-float(d["drawdown_pct"]) for d in comparison_sims["EXP_E_COMBINED_EXEC_INTEL"]["daily_navs"]], "#2563eb"),
            ("Challenger J (FamE + ExecIntel)", [-float(d["drawdown_pct"]) for d in comparison_sims["EXP_J_CHALLENGER_RB_65_25_10_FAME"]["daily_navs"]], "#0f766e"),
        ],
        y_label="Drawdown (%)",
        out_path=output_dir / "chart_drawdown_comparison.svg",
    )
    generated.append("chart_drawdown_comparison.svg")

    # 8. Alpha Contribution by Strategy Family (Family E vs Family A across Architectures)
    fam_archs = [
        ("Exp I Base", base_I_sim),
        ("Exp I + ExecIntel", comparison_sims["EXP_I_COMBINED_EXEC_INTEL"]),
        ("Exp G Base (E+A)", comparison_sims["EXP_G_BASE_PROMPT2"]),
        ("Exp G + ExecIntel", comparison_sims["EXP_G_COMBINED_EXEC_INTEL"]),
        ("Exp E + ExecIntel", comparison_sims["EXP_E_COMBINED_EXEC_INTEL"]),
        ("Challenger J (FamE)", comparison_sims["EXP_J_CHALLENGER_RB_65_25_10_FAME"]),
    ]
    fam_e_pnl = [
        float(s["sleeve_attribution"].get("STRAT_E_DEFENSIVE_YIELD_LOWVOL", {}).get("total_pnl_with_div_inr", 0.0))
        for _, s in fam_archs
    ]
    fam_a_pnl = [
        float(s["sleeve_attribution"].get("STRAT_A_CS_MOMENTUM", {}).get("total_pnl_with_div_inr", 0.0))
        for _, s in fam_archs
    ]
    _svg_grouped_bar_chart(
        title="8. Net Tactical PnL (incl. Dividends) by Strategy Family: Family E vs Family A (INR)",
        subtitle="Empirical attribution showing Family E consistent value creation vs Family A drag across configurations",
        categories=[k for k, _ in fam_archs],
        groups=[
            ("Family E Defensive Compounder (INR)", fam_e_pnl, "#0f766e"),
            ("Family A Momentum (INR)", fam_a_pnl, "#dc2626"),
        ],
        y_label="Net Tactical PnL + Dividends (INR)",
        out_path=output_dir / "chart_alpha_by_strategy_family.svg",
    )
    generated.append("chart_alpha_by_strategy_family.svg")

    return generated
