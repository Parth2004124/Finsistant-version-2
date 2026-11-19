#!/usr/bin/env python3
"""Run independent Moreshwar 2.0 quantitative experiments on real 1-year OHLCV data.

Executes:
- Phase 5: Models A, B, C, D, E (clean causal), E-Ref (static-fund warning), F, SMA-20, Nifty 50 B&H
- Phase 6: Parameter sensitivity across Train / Validation + strict Out-of-Sample + 3-fold Walk-Forward
- Phase 7: Full 15-metric table + regime / stock / sector / rejection breakdowns

Writes machine-readable JSON to `data/moreshwar_v2_experiment_report.json` and prints a concise summary.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from finsistant_v2 import ASSET_UNIVERSE, fetch_benchmark_candles, fetch_live_market_snapshot, get_asset_full_analysis
from finsistant_v2.backtest_engine_v2 import run_moreshwar_v2_research_suite
from finsistant_v2.moreshwar_v2 import MoreshwarV2Config


def main() -> None:
    print("Fetching real 1Y OHLCV history across watchlist + Nifty 50 benchmark...")
    bench = fetch_benchmark_candles("^NSEI", force_refresh=False)
    universe_history = {}
    for sym in ASSET_UNIVERSE.keys():
        snap = get_asset_full_analysis(sym, is_holding=False, force_refresh=False)
        raw = fetch_live_market_snapshot(sym, force_refresh=False)
        universe_history[sym] = {
            **raw,
            "fundamental_score": snap["fundamental_score"],
            "porters_score": snap["porters_score"],
        }
        print(f"  loaded {sym}: {len(raw.get('candles', []))} bars | {raw.get('data_source')}")

    cfg = MoreshwarV2Config()
    report = run_moreshwar_v2_research_suite(
        universe_history=universe_history,
        benchmark_data=bench,
        cfg=cfg,
        run_sensitivity=True,
    )

    out_path = ROOT_DIR / "data" / "moreshwar_v2_experiment_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Strip full per-trade lists when saving compact report unless needed (keep summary + per-model breakdown)
    compact = dict(report)
    compact["model_details"] = {
        k: {key: val for key, val in v.items() if key != "trades" and key != "equity_curve"}
        for k, v in report["model_details"].items()
    }
    out_path.write_text(json.dumps(compact, indent=2), encoding="utf-8")

    print("\n=== PHASE 5 / 7: INDEPENDENT MODEL COMPARISON (1Y REAL DATA, 0.35% COST) ===")
    hdr = f"{'Model':48} | {'Tot%':>6} | {'Ann%':>6} | {'Win%':>6} | {'PF':>5} | {'Shrp':>5} | {'MaxDD':>6} | {'Trd':>4} | {'SL%':>5} | {'TP%':>5} | {'Util%':>5} | {'Rej':>4}"
    print(hdr)
    print("-" * len(hdr))
    for m in report["models_comparison"]:
        pf_s = f"{m['profit_factor']:.2f}" if m.get("profit_factor") is not None else "  N/A"
        print(
            f"{m['model'][:48]:48} | {m['total_return_pct']:+6.2f} | {m['annualised_return_pct']:+6.2f} | "
            f"{m['win_rate_pct']:6.1f} | {pf_s:>5} | {m['sharpe_ratio']:+5.2f} | "
            f"{m['max_drawdown_pct']:6.2f} | {m['total_trades']:4d} | {m['stop_loss_freq_pct']:5.1f} | "
            f"{m['target_hit_freq_pct']:5.1f} | {m['capital_utilisation_pct']:5.1f} | {m['rejected_trades']:4d}"
        )

    print("\n=== PHASE 6: OUT-OF-SAMPLE TEST SPLIT ===")
    oos = report.get("out_of_sample") or {}
    if oos:
        print("Splits:", json.dumps(oos["splits"]))
        for k in ("moreshwar_1_oos", "moreshwar_2_oos", "nifty50_oos"):
            m = oos[k]
            print(f"  {m['model']:44} -> Tot {m['total_return_pct']:+6.2f}% | MaxDD {m['max_drawdown_pct']:5.2f}% | Win {m['win_rate_pct']:5.1f}% | Trades {m['total_trades']}")

    print("\n=== PHASE 6: PARAMETER SENSITIVITY (TRAIN vs VALIDATION TOP/MEDIAN) ===")
    for row in report.get("sensitivity_sweep", []):
        print(
            f"  ATR={row['atr_multiplier']:.1f} Policy={row['regime_policy']:22} | "
            f"Train {row['train_return_pct']:+6.2f}% (DD {row['train_max_dd_pct']:5.2f}%, n={row['train_trades']:2d}) | "
            f"Val {row['val_return_pct']:+6.2f}% (DD {row['val_max_dd_pct']:5.2f}%, n={row['val_trades']:2d})"
        )

    print("\n=== PHASE 6: ROLLING WALK-FORWARD FOLDS ===")
    for wf in report.get("rolling_walk_forward", []):
        print(
            f"  {wf['fold']} ({wf['start']}..{wf['end']}): "
            f"M1 {wf['m1_return_pct']:+6.2f}% (DD {wf['m1_max_dd_pct']:5.2f}%) vs "
            f"M2 Clean {wf['m2_return_pct']:+6.2f}% (DD {wf['m2_max_dd_pct']:5.2f}%, Win {wf['m2_win_rate_pct']:5.1f}%, n={wf['m2_trades']})"
        )

    print(f"\nSaved full experiment JSON to {out_path}")


if __name__ == "__main__":
    main()
