#!/usr/bin/env python3
"""Generate live_scan.json using real Yahoo Finance market data, Zerodha Kite,
and the 3-Agent Orchestrator (TechBot_1, FundBot_1, ExecBot_1). Zero simulation.

Supports GitHub Actions workflow_dispatch inputs via env vars:
- ADD_SYMBOL: Validate live ticker on Yahoo Finance and add to watchlist_config.json
- REMOVE_SYMBOL: Remove ticker from watchlist_config.json extra_symbols
- FORCE_KITE_REFRESH: 'true' to force a fresh TOTP Kite login
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from finsistant_v2 import get_asset_full_analysis
import server


def _update_watchlist_config(add_sym: str, remove_sym: str) -> None:
    cfg_path = ROOT_DIR / "watchlist_config.json"
    if cfg_path.exists():
        try:
            data = json.loads(cfg_path.read_text(encoding="utf-8"))
        except Exception:
            data = {}
    else:
        data = {}

    extras = [str(s).upper().strip() for s in (data.get("extra_symbols") or []) if str(s).strip()]

    if remove_sym:
        extras = [s for s in extras if s != remove_sym]

    if add_sym and add_sym not in extras:
        # Validate against real Yahoo Finance data before saving (zero simulation)
        get_asset_full_analysis(add_sym, is_holding=False, force_refresh=True)
        extras.append(add_sym)

    cfg_path.write_text(json.dumps({"extra_symbols": extras}, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    add_sym = os.getenv("ADD_SYMBOL", "").upper().strip()
    remove_sym = os.getenv("REMOVE_SYMBOL", "").upper().strip()
    force_kite = os.getenv("FORCE_KITE_REFRESH", "").lower().strip() == "true"

    if add_sym or remove_sym:
        _update_watchlist_config(add_sym, remove_sym)

    server._initialize_platform_state(force_refresh=True, force_token_refresh=force_kite)
    default_sym = next(iter(server.PORTFOLIO_STATE.keys()), "NTPC")
    active_sym = add_sym if add_sym and add_sym in server.PORTFOLIO_STATE else default_sym
    payload = server._build_full_state_payload(active_sym)
    out_path = ROOT_DIR / "live_scan.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        f"Wrote {out_path} | assets={len(payload['assets'])} | "
        f"buys={len(payload['queue']['buys'])} | sells={len(payload['queue']['sells'])} | "
        f"holdings={len(payload['portfolio']['holdings'])} | kite={payload['kite_status']} | "
        f"generated_at={payload['generated_at']}"
    )


if __name__ == "__main__":
    main()
