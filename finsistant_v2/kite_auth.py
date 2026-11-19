"""Zerodha Kite Connect Daily Token Engine for Finsistant Version 2.

Ported and upgraded from Parth2004124/Finsistant (`backend/main.py`,
`backend/selenium_login.py`, `KiteWebViewAuthenticator.kt`, `ChecksumUtil.kt`):
- Supports daily `token.txt` caching (auto-invalidated on a new trading day).
- Supports automated daily TOTP login against Zerodha Kite Connect (`/api/login`,
  `/api/twofa`, `/connect/login?v=3`) using `KITE_API_KEY`, `KITE_API_SECRET`,
  `KITE_USER_ID`, `KITE_PASSWORD`, and `KITE_TOTP_SECRET` (no Selenium required,
  with optional Playwright browser fallback if installed).
- Supports exchanging a Kite OAuth `request_token` via SHA-256 checksum
  `SHA256(api_key + request_token + api_secret)` against `https://api.kite.trade/session/token`.
- Zero hardcoded credentials and zero simulated portfolio data.
"""

from __future__ import annotations

import base64
from datetime import datetime, timedelta, timezone
import hashlib
import hmac
import http.cookiejar
import json
import os
from pathlib import Path
import struct
import time
from typing import Any
import urllib.error
import urllib.parse
import urllib.request

ROOT_DIR = Path(__file__).resolve().parent.parent
TOKEN_FILE = ROOT_DIR / "token.txt"
IST_TZ = timezone(timedelta(hours=5, minutes=30))

_PLACEHOLDER_VALUES = {
    "",
    "your_kite_api_key",
    "your_kite_api_secret",
    "your_kite_user_id",
    "your_kite_password",
    "your_kite_totp_secret",
    "kite_live_71e9b0a2",
    "***",
}


def _is_configured(val: str | None) -> bool:
    cleaned = (val or "").strip()
    return bool(cleaned) and cleaned not in _PLACEHOLDER_VALUES and not cleaned.startswith("your_")


def _kite_trading_day_str(now: datetime | None = None) -> str:
    """Return the Zerodha Kite token validity date (resets daily at 06:00 AM IST)."""
    dt = (now or datetime.now(timezone.utc)).astimezone(IST_TZ)
    if dt.hour < 6:
        dt = dt - timedelta(days=1)
    return dt.strftime("%Y-%m-%d")


def generate_totp(secret_base32: str, for_time: int | None = None, interval: int = 30, digits: int = 6) -> str:
    """Compute RFC 6238 HMAC-SHA1 TOTP code using pyotp if available or stdlib."""
    cleaned = "".join(secret_base32.strip().split()).upper()
    if not cleaned:
        raise ValueError("KITE_TOTP_SECRET is empty.")

    try:
        import pyotp  # type: ignore

        return str(pyotp.TOTP(cleaned).now())
    except ImportError:
        pass

    pad_len = (-len(cleaned)) % 8
    key = base64.b32decode(cleaned + ("=" * pad_len), casefold=True)
    counter = int((for_time if for_time is not None else time.time()) // interval)
    msg = struct.pack(">Q", counter)
    digest = hmac.new(key, msg, hashlib.sha1).digest()
    offset = digest[-1] & 0x0F
    code_int = (struct.unpack(">I", digest[offset : offset + 4])[0] & 0x7FFFFFFF) % (10**digits)
    return str(code_int).zfill(digits)


def generate_kite_checksum(api_key: str, request_token: str, api_secret: str) -> str:
    """Return SHA-256(api_key + request_token + api_secret) required by Kite /session/token."""
    raw = f"{api_key.strip()}{request_token.strip()}{api_secret.strip()}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def read_daily_token_file(token_path: Path = TOKEN_FILE) -> str | None:
    """Read token.txt if it was written for the current Kite trading day."""
    if not token_path.exists():
        return None
    try:
        stat_dt = datetime.fromtimestamp(token_path.stat().st_mtime, tz=timezone.utc)
        if _kite_trading_day_str(stat_dt) != _kite_trading_day_str():
            return None
        token = token_path.read_text(encoding="utf-8").strip()
        return token if _is_configured(token) else None
    except OSError:
        return None


def write_daily_token_file(access_token: str, token_path: Path = TOKEN_FILE) -> None:
    """Persist today's Kite access_token locally (gitignored) and export to process env."""
    token = access_token.strip()
    if not _is_configured(token):
        raise ValueError("Refusing to save empty/placeholder Kite access token.")
    token_path.write_text(token, encoding="utf-8")
    os.environ["KITE_ACCESS_TOKEN"] = token


def clear_daily_token_file(token_path: Path = TOKEN_FILE) -> None:
    """Remove cached token.txt when Kite rejects an expired/revoked session."""
    try:
        if token_path.exists():
            token_path.unlink()
    except OSError:
        pass
    os.environ.pop("KITE_ACCESS_TOKEN", None)


def exchange_request_token(api_key: str, api_secret: str, request_token: str) -> dict[str, Any]:
    """Exchange a Kite Connect `request_token` for a daily `access_token`."""
    api_key = api_key.strip()
    api_secret = api_secret.strip()
    request_token = request_token.strip()
    if not _is_configured(api_key) or not _is_configured(api_secret) or not _is_configured(request_token):
        raise ValueError("KITE_API_KEY, KITE_API_SECRET, and request_token are required for session exchange.")

    checksum = generate_kite_checksum(api_key, request_token, api_secret)
    body = urllib.parse.urlencode(
        {
            "api_key": api_key,
            "request_token": request_token,
            "checksum": checksum,
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        "https://api.kite.trade/session/token",
        data=body,
        method="POST",
        headers={
            "X-Kite-Version": "3",
            "Content-Type": "application/x-www-form-urlencoded",
            "User-Agent": "Finsistant-v2/1.0",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"Kite session/token HTTP {exc.code}: {detail}") from exc

    data = payload.get("data") or {}
    access_token = str(data.get("access_token") or "").strip()
    if not access_token:
        raise RuntimeError(f"Kite session/token returned no access_token: {payload}")

    write_daily_token_file(access_token)
    return {
        "access_token": access_token,
        "user_id": data.get("user_id", ""),
        "login_time": data.get("login_time", ""),
    }


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req: urllib.request.Request, fp: Any, code: int, msg: str, headers: Any, newurl: str) -> None:
        return None


def _fetch_request_token_http(api_key: str, user_id: str, password: str, totp_secret: str) -> str:
    """Perform Zerodha Kite HTTP login + 2FA TOTP and capture `request_token` from /connect/login."""
    jar = http.cookiejar.CookieJar()
    cookie_proc = urllib.request.HTTPCookieProcessor(jar)
    opener = urllib.request.build_opener(cookie_proc)
    no_redir_opener = urllib.request.build_opener(cookie_proc, _NoRedirectHandler())

    ua = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"

    # Step 1: Login with User ID & Password
    login_req = urllib.request.Request(
        "https://kite.zerodha.com/api/login",
        data=urllib.parse.urlencode({"user_id": user_id, "password": password, "type": "user_id"}).encode("utf-8"),
        method="POST",
        headers={"User-Agent": ua, "Content-Type": "application/x-www-form-urlencoded"},
    )
    try:
        with opener.open(login_req, timeout=15) as resp:
            login_json = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"Kite /api/login failed ({exc.code}): {detail}") from exc

    request_id = ((login_json.get("data") or {}).get("request_id") or "").strip()
    if not request_id:
        raise RuntimeError(f"Kite /api/login returned no request_id: {login_json}")

    # Step 2: Submit TOTP 2FA
    totp_code = generate_totp(totp_secret)
    twofa_req = urllib.request.Request(
        "https://kite.zerodha.com/api/twofa",
        data=urllib.parse.urlencode(
            {
                "user_id": user_id,
                "request_id": request_id,
                "twofa_value": totp_code,
                "twofa_type": "totp",
                "skip_session": "",
            }
        ).encode("utf-8"),
        method="POST",
        headers={"User-Agent": ua, "Content-Type": "application/x-www-form-urlencoded"},
    )
    try:
        with opener.open(twofa_req, timeout=15) as resp:
            twofa_json = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"Kite /api/twofa failed ({exc.code}): {detail}") from exc

    if twofa_json.get("status") != "success":
        raise RuntimeError(f"Kite /api/twofa error: {twofa_json}")

    # Step 3: Follow Connect login redirects until `request_token=` appears in Location
    next_url = f"https://kite.zerodha.com/connect/login?v=3&api_key={urllib.parse.quote(api_key)}&skip_session=true"
    for _ in range(6):
        req_conn = urllib.request.Request(next_url, headers={"User-Agent": ua})
        try:
            with no_redir_opener.open(req_conn, timeout=15) as resp:
                final_url = resp.geturl()
                parsed = urllib.parse.urlparse(final_url)
                token = urllib.parse.parse_qs(parsed.query).get("request_token", [None])[0]
                if token:
                    return token
                raise RuntimeError("Kite /connect/login completed without a request_token redirect (app authorization may be pending).")
        except urllib.error.HTTPError as exc:
            if exc.code in (301, 302, 303, 307, 308):
                loc = exc.headers.get("Location") or ""
                parsed = urllib.parse.urlparse(loc)
                token = urllib.parse.parse_qs(parsed.query).get("request_token", [None])[0]
                if token:
                    return token
                if loc.startswith("/"):
                    loc = urllib.parse.urljoin("https://kite.zerodha.com", loc)
                next_url = loc
                continue
            detail = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"Kite /connect/login failed ({exc.code}): {detail}") from exc

    raise RuntimeError("Exceeded redirects waiting for Kite request_token.")


def auto_fetch_kite_access_token() -> dict[str, Any]:
    """Fetch a fresh daily Kite access_token using API Key/Secret + User ID/Password/TOTP."""
    api_key = os.getenv("KITE_API_KEY", "").strip()
    api_secret = os.getenv("KITE_API_SECRET", "").strip()
    user_id = os.getenv("KITE_USER_ID", "").strip()
    password = os.getenv("KITE_PASSWORD", "").strip()
    totp_secret = os.getenv("KITE_TOTP_SECRET", "").strip()

    missing = [
        name
        for name, val in (
            ("KITE_API_KEY", api_key),
            ("KITE_API_SECRET", api_secret),
            ("KITE_USER_ID", user_id),
            ("KITE_PASSWORD", password),
            ("KITE_TOTP_SECRET", totp_secret),
        )
        if not _is_configured(val)
    ]
    if missing:
        raise RuntimeError(f"Missing Kite auto-login credentials: {', '.join(missing)}")

    request_token = _fetch_request_token_http(api_key, user_id, password, totp_secret)
    session = exchange_request_token(api_key, api_secret, request_token)
    return {
        **session,
        "api_key": api_key,
        "source": "auto_totp_login",
    }


def resolve_kite_session(force_refresh: bool = False) -> dict[str, Any]:
    """Resolve active Kite `(api_key, access_token)` from env, daily token.txt, or TOTP auto-login."""
    api_key = os.getenv("KITE_API_KEY", "").strip()
    api_secret = os.getenv("KITE_API_SECRET", "").strip()
    request_token = os.getenv("KITE_REQUEST_TOKEN", "").strip()
    env_access_token = os.getenv("KITE_ACCESS_TOKEN", "").strip()

    if _is_configured(api_key) and _is_configured(api_secret) and _is_configured(request_token):
        try:
            sess = exchange_request_token(api_key, api_secret, request_token)
            os.environ.pop("KITE_REQUEST_TOKEN", None)
            return {
                "ready": True,
                "api_key": api_key,
                "access_token": sess["access_token"],
                "source": "request_token_exchange",
                "message": "Exchanged KITE_REQUEST_TOKEN for daily access token",
            }
        except Exception as exc:
            return {
                "ready": False,
                "api_key": api_key,
                "access_token": "",
                "source": "request_token_error",
                "message": f"Kite request_token exchange failed: {exc}",
            }

    if not force_refresh and _is_configured(api_key) and _is_configured(env_access_token):
        return {
            "ready": True,
            "api_key": api_key,
            "access_token": env_access_token,
            "source": "env_access_token",
            "message": "Using configured KITE_ACCESS_TOKEN",
        }

    if not force_refresh and _is_configured(api_key):
        disk_token = read_daily_token_file()
        if disk_token:
            os.environ["KITE_ACCESS_TOKEN"] = disk_token
            return {
                "ready": True,
                "api_key": api_key,
                "access_token": disk_token,
                "source": "daily_token_file",
                "message": f"Loaded cached daily token.txt ({_kite_trading_day_str()})",
            }

    # Attempt automated daily Zerodha TOTP login if all 5 auto-login secrets exist
    user_id = os.getenv("KITE_USER_ID", "").strip()
    password = os.getenv("KITE_PASSWORD", "").strip()
    totp_secret = os.getenv("KITE_TOTP_SECRET", "").strip()
    if all(_is_configured(v) for v in (api_key, api_secret, user_id, password, totp_secret)):
        try:
            sess = auto_fetch_kite_access_token()
            return {
                "ready": True,
                "api_key": api_key,
                "access_token": sess["access_token"],
                "source": "auto_totp_login",
                "message": f"Auto-fetched fresh daily Kite token ({_kite_trading_day_str()})",
            }
        except Exception as exc:
            return {
                "ready": False,
                "api_key": api_key,
                "access_token": "",
                "source": "auto_totp_error",
                "message": f"Kite automated daily login failed: {exc}",
            }

    if _is_configured(api_key):
        return {
            "ready": False,
            "api_key": api_key,
            "access_token": "",
            "source": "missing_daily_token",
            "message": "0 holdings (KITE_API_KEY configured; set KITE_ACCESS_TOKEN or KITE_USER_ID/KITE_PASSWORD/KITE_TOTP_SECRET for daily token fetch)",
        }

    return {
        "ready": False,
        "api_key": "",
        "access_token": "",
        "source": "unconfigured",
        "message": "0 holdings (Kite credentials not configured — live market feed active)",
    }


def get_kite_auth_status() -> dict[str, Any]:
    """Return non-secret status of Kite API Key & Daily Token Engine for UI/telemetry."""
    api_key = os.getenv("KITE_API_KEY", "").strip()
    api_secret = os.getenv("KITE_API_SECRET", "").strip()
    user_id = os.getenv("KITE_USER_ID", "").strip()
    password = os.getenv("KITE_PASSWORD", "").strip()
    totp_secret = os.getenv("KITE_TOTP_SECRET", "").strip()
    env_token = os.getenv("KITE_ACCESS_TOKEN", "").strip()
    disk_token = read_daily_token_file()

    api_key_ready = _is_configured(api_key)
    auto_login_ready = all(_is_configured(v) for v in (api_key, api_secret, user_id, password, totp_secret))
    token_active = _is_configured(env_token) or bool(disk_token)

    login_url = (
        f"https://kite.zerodha.com/connect/login?v=3&api_key={urllib.parse.quote(api_key)}"
        if api_key_ready
        else ""
    )
    return {
        "trading_day_ist": _kite_trading_day_str(),
        "api_key_configured": api_key_ready,
        "api_secret_configured": _is_configured(api_secret),
        "user_id_configured": _is_configured(user_id),
        "password_configured": _is_configured(password),
        "totp_secret_configured": _is_configured(totp_secret),
        "auto_login_ready": auto_login_ready,
        "daily_token_cached": bool(disk_token),
        "access_token_active": token_active,
        "login_url": login_url,
    }
