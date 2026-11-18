"""Configuration and environment management for Finsistant Version 2."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from pathlib import Path
from typing import Any, Literal

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def _parse_env_file(env_path: Path) -> None:
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.split("#", 1)[0].strip().strip("'").strip('"')
        if key:
            os.environ[key] = val


@dataclass
class FinsistantSettings:
    """Validated runtime settings for Finsistant Version 2."""

    finsistant_api_key: str = "fnv2_live_98f2a4d7c1e3b506"
    finsistant_environment: Literal["sandbox", "staging", "production"] = "production"
    agent_mode: Literal["backtest", "paper_trading", "live"] = "paper_trading"
    default_currency: str = "INR"
    max_concurrent_trades: int = 5
    execution_password: str = "parth_exec_2026"
    kite_api_key: str = "kite_live_71e9b0a2"
    gemini_api_key: str = "gem_configured"

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


def load_settings(env_path: str | Path | None = None) -> FinsistantSettings:
    target_path = Path(env_path) if env_path else Path(".env")
    if target_path.exists():
        _parse_env_file(target_path)

    env_val = os.getenv("FINSISTANT_ENVIRONMENT", "production").lower()
    if env_val not in ("sandbox", "staging", "production"):
        env_val = "production"

    mode_val = os.getenv("AGENT_MODE", "paper_trading").lower()
    if mode_val not in ("backtest", "paper_trading", "live"):
        mode_val = "paper_trading"

    try:
        max_trades = max(1, min(50, int(os.getenv("MAX_CONCURRENT_TRADES", "5"))))
    except ValueError:
        max_trades = 5

    return FinsistantSettings(
        finsistant_api_key=os.getenv("FINSISTANT_API_KEY", "fnv2_live_98f2a4d7c1e3b506"),
        finsistant_environment=env_val,  # type: ignore[arg-type]
        agent_mode=mode_val,  # type: ignore[arg-type]
        default_currency=os.getenv("DEFAULT_CURRENCY", "INR"),
        max_concurrent_trades=max_trades,
        execution_password=os.getenv("EXECUTION_PASSWORD", "parth_exec_2026"),
        kite_api_key=os.getenv("KITE_API_KEY", "kite_live_71e9b0a2"),
        gemini_api_key=os.getenv("GEMINI_API_KEY", "gem_configured"),
    )
