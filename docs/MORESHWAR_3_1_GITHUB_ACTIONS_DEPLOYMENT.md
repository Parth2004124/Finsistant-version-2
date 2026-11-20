# Moreshwar 3.1 — GitHub Actions Unattended Paper-Trading Deployment & Recovery Guide

> **MORESHWAR 3.1 — PAPER TRADING ONLY.**  
> **LIVE TRADING: DISABLED. · BROKER ORDER PLACEMENT: DISABLED. · AUTOMATIC CAPITAL DEPLOYMENT: DISABLED.**

---

## 1. Architecture Overview

Moreshwar 3.1 (`v3.1.0-rc1`) runs unattended on eligible NSE trading days via `.github/workflows/moreshwar-3-1-paper-trading.yml` on the repository's default branch (`main`), with **zero dependency on your PC staying online** and **zero brokerage credentials**.

- **Frozen Configuration:** `reports/moreshwar_3_1/strategy_intelligence/FROZEN_PROSPECTIVE_PAPER_TRADING_CONFIG.json`
  - Canonical SHA256: `47a30f4bbd004c5f238cf97c549b4d41a10232c4cdf2a4966efecf75bf2d255a`
  - Raw File SHA256: `658d17894618de9dc974b8748e1b2d25ba2e8020d5fc4dc99f9855b1a835f068`
- **Schedule (`TZ=Asia/Kolkata`, Mon–Fri after NSE 15:30 IST close):**
  - `17:00 IST` (`30 11 * * 1-5` UTC) — primary post-close daily EOD run
  - `18:00 IST` (`30 12 * * 1-5` UTC) — idempotent retry if NSE EOD feed was delayed
  - `19:00 IST` (`30 13 * * 1-5` UTC) — final idempotent retry
- **Single-Flight Concurrency:** `concurrency.group = moreshwar-3-1-prospective-paper-trading` with `cancel-in-progress: false` (an overlapping trigger NEVER cancels an active portfolio-processing run).
- **Market Data Feed:** `finsistant_v2/m3_1/live_nse_eod_feed.py` fetches completed post-close daily OHLCV + trailing 12M dividends for `NIFTYBEES.NS`, `LIQUIDBEES.NS`, the frozen 120-stock liquid NSE universe (`data/m3_1_paper_trading/nse_prospective_universe.json`), and any open tactical holdings. If `NIFTYBEES.NS` does not yet have a completed bar for the trading date, the scheduler transitions cleanly to `WAITING_FOR_DATA` and **never fabricates prices**.

---

## 2. Persistent Storage Configuration (`config/m3_1_persistent_storage.json`)

The ephemeral GitHub Actions runner filesystem, GitHub Actions cache (`actions/cache`), and workflow artifacts are **strictly prohibited** as the authoritative portfolio ledger.

Two durable persistence modes are supported by `finsistant_v2/m3_1/persistent_storage_adapter.py`:

### Option A — External PostgreSQL (Recommended for Institutional Durability)
1. Provision any PostgreSQL instance (e.g., Supabase, Neon, or Render PostgreSQL).
2. In GitHub Repository **Settings → Secrets and variables → Actions**, add:
   - `M31_DATABASE_URL` = `postgresql://<user>:<password>@<host>:5432/<dbname>?sslmode=require`
3. No code changes are required. The adapter automatically switches to `EXTERNAL_POSTGRESQL`, acquires PostgreSQL session advisory lock `pg_try_advisory_lock(31031031)`, pulls the SHA256-verified durable ledger from `m31_durable_ledger_store`, runs the session atomically, verifies `<= INR 0.01` reconciliation, and commits the new checksummed ledger version back to PostgreSQL.

### Option B — Durable Git-Backed Ledger (Active Default Until `M31_DATABASE_URL` Is Added)
- When `M31_DATABASE_URL` is unset, `prospective_paper_ledger.db` (SQLite WAL checkpointed with `PRAGMA wal_checkpoint(TRUNCATE)`) and all 12 daily reports are persisted durably in Git on `main` (tracked via `!data/m3_1_paper_trading/prospective_paper_ledger.db` in `.gitignore`).
- Database-level atomic lease locking (`distributed_execution_locks`) and idempotent `session_checkpoints` prevent duplicate orders, dividends, settlements, or double-charged transaction costs across retries.
- GitHub Artifacts (`moreshwar-3-1-daily-reports-<run_id>`, 90-day retention) store diagnostic/report bundles only, never the authoritative ledger.

### Optional Failure Webhook Secret
- On any workflow failure, GitHub Actions automatically opens or comments on a sanitized GitHub Issue (`[Moreshwar 3.1 Paper Trading] Daily Workflow Failure / Safe-Mode Alert`).
- Optionally add `M31_ALERT_WEBHOOK_URL` (Slack / Discord / Telegram webhook) in GitHub Secrets for instant mobile push alerts.

---

## 3. Manual Workflow Dispatch Controls

In GitHub → **Actions → Moreshwar 3.1 — Daily Prospective Paper Trading (Paper-Only) → Run workflow**:

| `mode` | Purpose |
| :--- | :--- |
| `run-session` | Run one scheduled NSE daily check (`trading_date: auto` or `YYYY-MM-DD`). Safe to re-run (strictly idempotent). |
| `verify` | Run all 12 pre-launch frozen-config, 7-layer safety, and ledger checks + print live status. |
| `reconcile` | Verify four-ledger reconciliation (`<= INR 0.01`) and cryptographic append-only audit chain. |
| `recover` | Execute validated SAFE_MODE recovery after inspecting a crash or missed session. |

---

## 4. Failure & Crash Recovery Instructions

If a runner terminates mid-execution or reconciliation fails:

1. **Automatic Protection:**
   - The failed workflow does **not** push corrupted remote state (`close_without_push()`).
   - Any unreleased database lock or incomplete `session_checkpoints` row (`STARTED` / `DATA_VALIDATED` / `EXECUTED`) triggers **`SAFE_MODE`** on the next run, halting new paper entries while preserving all existing holdings and last valid state.
   - A sanitized GitHub Issue alert is opened with the failing run link (zero secrets exposed).
2. **Inspect Diagnostics:**
   - Download the `moreshwar-3-1-daily-reports-<run_id>` artifact and inspect `risk_alerts.json`, `data_quality_report.json`, and `paper_trading_daily_report.json`.
   - Confirm the frozen config SHA256 prefix remains `47a30f4bbd004c5f`.
3. **Execute Validated Recovery:**
   - Trigger **Run workflow** with `mode = recover` and a descriptive `recovery_note` (or locally run `python3 scripts/moreshwar_3_1_paper_ops.py recover --note "..."`).
   - Recovery clears the stale crash lock, marks incomplete checkpoints `FAILED`, re-verifies the frozen config hash from disk, validates the append-only audit chain, and verifies `<= INR 0.01` four-ledger reconciliation before resuming.
4. **Missed Trading Sessions:**
   - **Never fabricate** an unobserved trading session's open/close price.
   - If NSE data for a past session is permanently unavailable, trigger `mode = run-session`, `trading_date = YYYY-MM-DD`, `mark_missed_if_unavailable = true` to record an explicit `MISSED` audit event cleanly.
