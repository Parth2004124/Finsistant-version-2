#!/usr/bin/env python3
"""Generate live_scan.json using real Yahoo Finance market data, Zerodha Kite (if configured),
and the 3-Agent Orchestrator (TechBot_1, FundBot_1, ExecBot_1). Zero simulation."""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import server


def main() -> None:
    server._initialize_platform_state(force_refresh=True)
    payload = server._build_full_state_payload("RELIANCE")
    out_path = ROOT_DIR / "live_scan.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        f"Wrote {out_path} | assets={len(payload['assets'])} | "
        f"buys={len(payload['queue']['buys'])} | sells={len(payload['queue']['sells'])} | "
        f"holdings={len(payload['portfolio']['holdings'])} | generated_at={payload['generated_at']}"
    )


if __name__ == "__main__":
    main()
