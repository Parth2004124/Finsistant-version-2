# MORESHWAR 3.1 — GITHUB ACTIONS DEPLOYMENT AUDIT REPORT

> [!IMPORTANT]
> **DEPLOYMENT ADAPTATION STATUS:** **VERIFIED & READY FOR UNATTENDED DAILY EXECUTION ON `main`**  
> **PAPER TRADING ONLY — LIVE TRADING: DISABLED · BROKER ORDER PLACEMENT: DISABLED · AUTOMATIC CAPITAL DEPLOYMENT: DISABLED**

---

## 1. Requirements Compliance Matrix (All 20 Mandatory Controls)

| # | Requirement | Implementation & Evidence | Status |
| :---: | :--- | :--- | :---: |
| 1 | Dedicated GitHub Actions workflow | `.github/workflows/moreshwar-3-1-paper-trading.yml` | **PASS** |
| 2 | NSE post-close schedule (`Asia/Kolkata`) | `TZ: Asia/Kolkata`; Mon–Fri `17:00`, `18:00`, `19:00 IST` (`30 11/12/13 * * 1-5` UTC) | **PASS** |
| 3 | Manual `workflow_dispatch` | Supports `run-session`, `verify`, `reconcile`, `recover` + `trading_date` + `mark_missed_if_unavailable` | **PASS** |
| 4 | Holidays, weekends, missing & delayed data | `is_eligible_nse_trading_day()` skips weekends/NSE holidays; missing EOD returns `None` -> `WAITING_FOR_DATA` | **PASS** |
| 5 | Scheduled workflow on default branch | Deployed directly to repository default branch **`main`** (`Parth2004124/Finsistant-version-2`) | **PASS** |
| 6 | Persistent external DB / durable storage | `PersistentStorageAdapter` (`EXTERNAL_POSTGRESQL` via `M31_DATABASE_URL` + durable Git-backed SQLite fallback) | **PASS** |
| 7 | No GitHub Actions cache as permanent storage | Zero `actions/cache` usage; asserted by static YAML test | **PASS** |
| 8 | Artifacts for daily reports/diagnostics only | `actions/upload-artifact` uploads `reports/moreshwar_3_1/prospective_paper_trading/` (90d retention, non-authoritative) | **PASS** |
| 9 | DB transactions, idempotency keys & execution locks | SQLite `BEGIN IMMEDIATE` + `session_checkpoints` + `distributed_execution_locks` + PostgreSQL `pg_try_advisory_lock(31031031)` | **PASS** |
| 10 | Concurrent/retried runs cannot duplicate ledger rows | Single-flight `cancel-in-progress: false` + DB lease lock + unique order/cost/dividend constraints | **PASS** |
| 11 | Pending `NEXT_SESSION_OPEN` fills before new signals | Enforced in `ProspectivePaperTradingEngine.process_trading_session` (Step 1 fills precede Step 5 EOD signals) | **PASS** |
| 12 | Never fabricate missed prices or unobserved fills | Missing EOD data yields `WAITING_FOR_DATA` or explicit `MISSED` checkpoint; zero synthetic prices | **PASS** |
| 13 | Failure preserves last valid state & activates safe recovery | Failed runs call `close_without_push()`; stale locks & partial checkpoints trigger `SAFE_MODE` | **PASS** |
| 14 | Append-only audit trail | SQLite `BEFORE UPDATE`/`BEFORE DELETE` triggers + SHA256 chain (`decision_audit_log` + `daily_audit_trail.jsonl`) | **PASS** |
| 15 | Publish daily snapshots, risk & validation progress | Exports all 12 reports + GitHub Step Summary + 90-day diagnostic artifact | **PASS** |
| 16 | Failure notifications | `if: failure()` sanitized GitHub Issue creator/commenter + optional `M31_ALERT_WEBHOOK_URL` | **PASS** |
| 17 | Zero credential/token leakage in logs | `redact_sensitive_text()` scrubs DB URLs/passwords/tokens; zero broker secrets in workflow | **PASS** |
| 18 | Third-party GitHub Actions pinned to immutable SHAs | `actions/checkout@11d5960a...`, `actions/setup-python@a26af69b...`, `actions/upload-artifact@ea165f8d...` | **PASS** |
| 19 | Least-privilege workflow permissions | Top-level `permissions: {}`; job-level strictly `contents: write`, `issues: write` | **PASS** |
| 20 | Hard live-trading / broker / capital-deployment bans | All 3 env flags `"false"`, zero broker SDKs (`kiteconnect` excluded), zero Kite secrets, 7-layer guard active | **PASS** |

---

## 2. Verified Immutable Action Pinning

| Action | Tag | Verified 40-Character Commit SHA |
| :--- | :---: | :--- |
| `actions/checkout` | `v4` | `11d5960a326750d5838078e36cf38b85af677262` |
| `actions/setup-python` | `v5` | `a26af69be951a213d495a4c3e4e4022e16d87065` |
| `actions/upload-artifact` | `v4` | `ea165f8d65b6e75b540449e92b4886f43607fa02` |

---

## 3. Validation Test Coverage (`tests/m3_1/test_github_actions_deployment.py`)

All 10 required operational resilience tests passed alongside the full repository regression suite:

1. **Successful daily execution** (offline causal Family E pipeline + DB locking + push/pull) — **PASS**
2. **Duplicate workflow invocation** (idempotent skip, zero duplicates) — **PASS**
3. **Runner termination** (active conflict + stale crash lock + partial checkpoint -> `SAFE_MODE`, last valid `INR 100,000.00` state preserved) — **PASS**
4. **Database reconnection** (fresh adapter & engine reload preserves NAV, holdings, integrity & audit chain) — **PASS**
5. **Retry after partial execution** (`STALE_CRASH_LOCK_DETECTED` fails closed until recovery) — **PASS**
6. **Missed trading session** (explicit `MISSED` record, zero fabricated fills) — **PASS**
7. **Missing market data** (`WAITING_FOR_DATA`, zero fabricated fills) — **PASS**
8. **Recovery after workflow failure** (validated stale-lock clear + `recover_from_safe_mode()` resumes cleanly) — **PASS**
9. **Next-session-open execution timing** (Day 1 close signal fills strictly at Day 2 open before Day 2 signals) — **PASS**
10. **Persistent reconciliation (`<= INR 0.01`), credential redaction & static workflow YAML security audit** — **PASS**
