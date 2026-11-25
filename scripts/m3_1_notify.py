#!/usr/bin/env python3
"""Moreshwar 3.1 — Smart Non-Spammy Notification Dispatcher (ntfy + Optional SMTP Email).

Sends paper-trading check-in reminders and critical alerts WITHOUT spamming on idempotent retries:
- ALWAYS sends (high priority): workflow failure, SAFE_MODE activation, critical risk alerts, or --force.
- Sends on milestone unseen sessions: Day 1, 2, 5, 10, 20, 30, 45, 60 (Interim Gate), 90, 120 (Full Gate).
- Sends weekly Friday digest after completed Friday NSE sessions.
- Suppresses notifications on idempotent duplicate runs, weekends, holidays, and WAITING_FOR_DATA.

Supported channels (configured strictly via GitHub Actions Secrets — never hardcoded):
1. ntfy (recommended, zero email/password needed):
   - `M31_NTFY_URL` (e.g. `https://ntfy.sh/<your-private-secret-topic>`) OR
   - `M31_NTFY_TOPIC` (auto-prefixed with `https://ntfy.sh/`)
2. Optional SMTP Email (stdlib smtplib):
   - `M31_SMTP_HOST`, `M31_SMTP_PORT` (default 587), `M31_SMTP_USER`, `M31_SMTP_PASS`, `M31_EMAIL_TO`
"""

from __future__ import annotations

import argparse
from datetime import date
from email.message import EmailMessage
import json
import os
from pathlib import Path
import smtplib
import ssl
import sys
from typing import Any
import urllib.request

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REPORTS_DIR = REPO_ROOT / "reports" / "moreshwar_3_1" / "prospective_paper_trading"
MILESTONE_SESSIONS = frozenset({1, 2, 5, 10, 20, 30, 45, 60, 90, 120})


def resolve_ntfy_url(ntfy_url: str = "", ntfy_topic: str = "") -> str:
    """Resolve full ntfy topic URL from either full URL or topic slug."""
    u = str(ntfy_url or "").strip()
    if u:
        return u
    t = str(ntfy_topic or "").strip().lstrip("/")
    if t:
        return f"https://ntfy.sh/{t}"
    return ""


def load_notification_context(reports_dir: Path = DEFAULT_REPORTS_DIR) -> dict[str, Any]:
    daily_p = reports_dir / "paper_trading_daily_report.json"
    prog_p = reports_dir / "prospective_validation_progress.json"
    alerts_p = reports_dir / "risk_alerts.json"
    daily = json.loads(daily_p.read_text(encoding="utf-8")) if daily_p.exists() else {}
    prog = json.loads(prog_p.read_text(encoding="utf-8")) if prog_p.exists() else {}
    alerts = json.loads(alerts_p.read_text(encoding="utf-8")) if alerts_p.exists() else {}
    return {"daily": daily, "progress": prog, "alerts": alerts}


def decide_and_format_notification(
    ctx: dict[str, Any],
    *,
    is_failure: bool = False,
    force_send: bool = False,
    session_newly_completed: bool = True,
    run_url: str = "",
) -> dict[str, Any]:
    """Determine whether to notify and build clean title/body/priority without leaking any secrets."""
    daily = ctx.get("daily") or {}
    prog = ctx.get("progress") or {}
    alerts = ctx.get("alerts") or {}

    trading_date = str(daily.get("trading_date") or prog.get("prospective_start_date") or "N/A")
    engine_status = str(daily.get("engine_status") or "UNKNOWN")
    nav_info = daily.get("portfolio_nav") or {}
    cur_nav = float(nav_info.get("current_nav_inr", 100_000.0))
    cum_ret = float(nav_info.get("cumulative_return_pct", 0.0))
    bench_nb = float(nav_info.get("benchmark_niftybees_nav", 100_000.0))
    init_cap = float(nav_info.get("initial_capital_inr", 100_000.0))
    nb_ret = ((bench_nb / max(1e-6, init_cap)) - 1.0) * 100.0
    rel_pp = cum_ret - nb_ret

    recon = daily.get("reconciliation_result") or {}
    recon_diff = float(recon.get("reconciliation_diff_inr", 0.0))
    recon_ok = bool(recon.get("passed", True))
    regime = (daily.get("regime_determination") or {}).get("confirmed_regime", "BULL_TREND")
    completed = int(prog.get("completed_unseen_sessions", 0))
    min_req = int(prog.get("minimum_required_sessions", 60))
    tgt_req = int(prog.get("target_sessions", 120))
    crit_cnt = int((alerts.get("alert_counts") or {}).get("CRITICAL", 0))
    health = str(alerts.get("overall_health") or "HEALTHY")

    # Determine trigger reason & priority
    should_send = False
    priority = "default"
    tags = "chart_with_upwards_trend"
    reason = "NO_TRIGGER"

    if is_failure or engine_status == "SAFE_MODE" or crit_cnt > 0 or (not recon_ok):
        should_send = True
        priority = "high"
        tags = "rotating_light,warning"
        reason = "CRITICAL_FAILURE_OR_SAFE_MODE"
        title = f"ALERT: MOVIQ 3.1 Paper Trading ({engine_status})"
    elif force_send:
        should_send = True
        priority = "default"
        tags = "white_check_mark,chart_with_upwards_trend"
        reason = "MANUAL_TEST_OR_FORCE"
        title = f"MOVIQ 3.1 Paper Check-In ({completed}/{min_req} sessions)"
    elif session_newly_completed and completed > 0:
        is_friday = False
        try:
            is_friday = date.fromisoformat(trading_date).weekday() == 4
        except Exception:
            pass
        if completed in MILESTONE_SESSIONS:
            should_send = True
            priority = "high" if completed in (60, 120) else "default"
            tags = "tada,chart_with_upwards_trend" if completed in (60, 120) else "bell,chart_with_upwards_trend"
            reason = f"MILESTONE_SESSION_{completed}"
            gate_lbl = (
                "INTERIM 60-SESSION REVIEW READY"
                if completed == 60
                else ("FULL 120-SESSION VALIDATION COMPLETE" if completed == 120 else f"Session {completed} Milestone")
            )
            title = f"MOVIQ 3.1 Paper: {gate_lbl}"
        elif is_friday:
            should_send = True
            priority = "default"
            tags = "calendar,chart_with_upwards_trend"
            reason = "WEEKLY_FRIDAY_DIGEST"
            title = f"MOVIQ 3.1 Weekly Paper Digest ({completed}/{min_req})"
        else:
            title = ""
    else:
        title = ""

    lines = [
        f"Date: {trading_date} | Mode: PAPER TRADING ONLY (Live Disabled)",
        f"Progress: {completed}/{min_req} min ({completed}/{tgt_req} target unseen sessions)",
        f"Paper NAV: INR {cur_nav:,.2f} ({cum_ret:+.2f}%) | vs NIFTYBEES: {rel_pp:+.2f} pp",
        f"Regime: {regime} | Engine: {engine_status} | Health: {health}",
        f"Reconciliation: {'PASS' if recon_ok else 'FAIL'} (diff INR {recon_diff:.4f})",
    ]
    if run_url:
        lines.append(f"GitHub Run: {run_url}")

    return {
        "should_send": should_send,
        "reason": reason,
        "priority": priority,
        "tags": tags,
        "title": title,
        "body": "\n".join(lines),
    }


def send_ntfy(
    *,
    url: str,
    title: str,
    body: str,
    priority: str = "default",
    tags: str = "chart_with_upwards_trend",
    click_url: str = "",
) -> dict[str, Any]:
    headers = {
        "Title": title,
        "Priority": priority,
        "Tags": tags,
        "Content-Type": "text/plain; charset=utf-8",
    }
    if click_url:
        headers["Click"] = click_url
    req = urllib.request.Request(url, data=body.encode("utf-8"), headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=15) as resp:
        return {"channel": "ntfy", "sent": True, "http_status": int(resp.status)}


def send_smtp_email(
    *,
    host: str,
    port: int,
    user: str,
    password: str,
    to_addr: str,
    subject: str,
    body: str,
) -> dict[str, Any]:
    msg = EmailMessage()
    msg["From"] = user
    msg["To"] = to_addr
    msg["Subject"] = subject
    msg.set_content(body)
    ctx = ssl.create_default_context()
    with smtplib.SMTP(host, port, timeout=20) as server:
        server.starttls(context=ctx)
        server.login(user, password)
        server.send_message(msg)
    return {"channel": "smtp_email", "sent": True}


def main() -> int:
    parser = argparse.ArgumentParser(description="Moreshwar 3.1 Smart Notification Dispatcher (ntfy + optional SMTP)")
    parser.add_argument("--reports-dir", default=str(DEFAULT_REPORTS_DIR))
    parser.add_argument("--failure", action="store_true", help="Send immediate high-priority failure alert")
    parser.add_argument("--force", action="store_true", help="Force send check-in notification (e.g., test ping)")
    parser.add_argument("--newly-completed", default="true", help="Whether a new NSE session completed in this run")
    parser.add_argument("--run-url", default=os.getenv("RUN_URL", ""))
    args = parser.parse_args()

    ctx = load_notification_context(Path(args.reports_dir))
    newly_done = str(args.newly_completed).lower().strip() in ("true", "1", "yes")
    decision = decide_and_format_notification(
        ctx,
        is_failure=bool(args.failure),
        force_send=bool(args.force),
        session_newly_completed=newly_done,
        run_url=str(args.run_url or ""),
    )

    if not decision["should_send"]:
        print(json.dumps({"sent": False, "reason": decision["reason"]}, indent=2))
        return 0

    results: list[dict[str, Any]] = []
    ntfy_url = resolve_ntfy_url(os.getenv("M31_NTFY_URL", ""), os.getenv("M31_NTFY_TOPIC", ""))
    if ntfy_url:
        try:
            results.append(
                send_ntfy(
                    url=ntfy_url,
                    title=decision["title"],
                    body=decision["body"],
                    priority=decision["priority"],
                    tags=decision["tags"],
                    click_url=str(args.run_url or ""),
                )
            )
        except Exception as exc:
            results.append({"channel": "ntfy", "sent": False, "error": type(exc).__name__})

    smtp_host = os.getenv("M31_SMTP_HOST", "").strip()
    smtp_user = os.getenv("M31_SMTP_USER", "").strip()
    smtp_pass = os.getenv("M31_SMTP_PASS", "").strip()
    email_to = os.getenv("M31_EMAIL_TO", "").strip()
    if smtp_host and smtp_user and smtp_pass and email_to:
        try:
            smtp_port = int(os.getenv("M31_SMTP_PORT", "587") or "587")
            results.append(
                send_smtp_email(
                    host=smtp_host,
                    port=smtp_port,
                    user=smtp_user,
                    password=smtp_pass,
                    to_addr=email_to,
                    subject=decision["title"],
                    body=decision["body"],
                )
            )
        except Exception as exc:
            results.append({"channel": "smtp_email", "sent": False, "error": type(exc).__name__})

    if not results:
        print(json.dumps({
            "sent": False,
            "reason": "NO_NOTIFICATION_CHANNEL_CONFIGURED",
            "would_have_triggered": decision["reason"],
            "hint": "Add GitHub Secret M31_NTFY_TOPIC (recommended) or M31_SMTP_* secrets to receive alerts.",
        }, indent=2))
        return 0

    print(json.dumps({"sent": any(r.get("sent") for r in results), "reason": decision["reason"], "channels": results}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
