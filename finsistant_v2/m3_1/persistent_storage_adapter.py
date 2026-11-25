"""Moreshwar 3.1 — Persistent External Storage, Reconnection & Database-Level Locking Adapter.

Adapts Moreshwar 3.1 prospective paper trading for unattended GitHub Actions execution without
relying on ephemeral runner filesystems, GitHub Actions cache, or workflow artifacts as the
authoritative portfolio ledger.

Supported durable backends:
1. `EXTERNAL_POSTGRESQL` (preferred when `M31_DATABASE_URL` secret is set):
   - Acquires PostgreSQL session advisory lock (`pg_try_advisory_lock(31031031)`).
   - Pulls SHA256-verified durable SQLite ledger image from `m31_durable_ledger_store`.
   - After atomic local SQLite execution & `<= INR 0.01` reconciliation, commits the updated
     checksummed ledger version back into PostgreSQL inside an atomic transaction.
   - Retries transient connections cleanly and redacts all credentials from logs.
2. `DURABLE_GIT_LEDGER` (durable fallback when `M31_DATABASE_URL` is unset):
   - Uses atomic SQLite WAL + database-level lease locks (`distributed_execution_locks`)
     persisted durably in Git (never Actions cache or artifacts).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import time
from typing import Any

from finsistant_v2.m3_1.prospective_paper_trading_engine import (
    DEFAULT_PAPER_DB_PATH,
    PaperOnlyExecutionGuard,
    REPO_ROOT,
)

DEFAULT_STORAGE_CONFIG_PATH = REPO_ROOT / "config" / "m3_1_persistent_storage.json"
DEFAULT_LOCK_NAME = "m3_1_daily_paper_session"
POSTGRES_ADVISORY_LOCK_KEY = 31031031

_SECRET_URL_RE = re.compile(r"([a-zA-Z][a-zA-Z0-9+.-]*://)([^/\s:@]+):([^/\s@]+)@")


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_now_iso() -> str:
    return _utc_now().isoformat()


def redact_sensitive_text(text: str) -> str:
    """Redact credentials in database URLs or secret strings so nothing sensitive ever appears in logs."""
    out = _SECRET_URL_RE.sub(r"\1***:***@", str(text))
    for env_key in ("M31_DATABASE_URL", "M32_DATABASE_URL", "M31_ALERT_WEBHOOK_URL", "GITHUB_TOKEN"):
        val = os.getenv(env_key, "").strip()
        if val and len(val) >= 6:
            out = out.replace(val, f"***REDACTED_{env_key}***")
    return out


def compute_file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


class DatabaseLockError(RuntimeError):
    """Raised when an execution lock is already held by another active run or a stale crash lock requires recovery."""


class DatabaseExecutionLock:
    """SQLite database-level atomic lease lock (`distributed_execution_locks`) surviving runner crashes."""

    def __init__(self, db_path: Path = DEFAULT_PAPER_DB_PATH, *, lock_name: str = DEFAULT_LOCK_NAME, lease_ttl_seconds: int = 1800) -> None:
        self.db_path = Path(db_path)
        self.lock_name = str(lock_name)
        self.lease_ttl_seconds = int(lease_ttl_seconds)
        self._ensure_table()

    def _connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path), timeout=30.0, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=FULL;")
        return conn

    def _ensure_table(self) -> None:
        conn = self._connect()
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS distributed_execution_locks (
                    lock_name TEXT PRIMARY KEY,
                    owner_run_id TEXT NOT NULL,
                    trading_date TEXT NOT NULL,
                    acquired_at_utc TEXT NOT NULL,
                    expires_at_utc TEXT NOT NULL,
                    status TEXT NOT NULL
                );
                """
            )
        finally:
            conn.close()

    def acquire(self, *, owner_run_id: str, trading_date: str) -> dict[str, Any]:
        """Atomically acquire DB-level execution lock; fail closed on active conflict OR stale crashed lock."""
        now = _utc_now()
        now_iso = now.isoformat()
        exp_iso = (now + timedelta(seconds=self.lease_ttl_seconds)).isoformat()
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT * FROM distributed_execution_locks WHERE lock_name = ?",
                (self.lock_name,),
            ).fetchone()
            if row is not None and str(row["status"]) == "HELD":
                existing_owner = str(row["owner_run_id"])
                existing_exp = datetime.fromisoformat(str(row["expires_at_utc"]))
                if existing_owner != owner_run_id:
                    if now < existing_exp:
                        conn.execute("ROLLBACK")
                        raise DatabaseLockError(
                            f"ACTIVE_EXECUTION_LOCK_CONFLICT: lock '{self.lock_name}' held by '{existing_owner}' "
                            f"for {row['trading_date']} until {row['expires_at_utc']}."
                        )
                    # Lease expired while status remained HELD -> indicates prior runner crash!
                    conn.execute("ROLLBACK")
                    raise DatabaseLockError(
                        f"STALE_CRASH_LOCK_DETECTED: prior run '{existing_owner}' on {row['trading_date']} "
                        f"terminated without releasing lock '{self.lock_name}'. Safe recovery required."
                    )
            conn.execute(
                """
                INSERT INTO distributed_execution_locks(
                    lock_name, owner_run_id, trading_date, acquired_at_utc, expires_at_utc, status
                ) VALUES (?, ?, ?, ?, ?, 'HELD')
                ON CONFLICT(lock_name) DO UPDATE SET
                    owner_run_id = excluded.owner_run_id,
                    trading_date = excluded.trading_date,
                    acquired_at_utc = excluded.acquired_at_utc,
                    expires_at_utc = excluded.expires_at_utc,
                    status = 'HELD'
                """,
                (self.lock_name, str(owner_run_id), str(trading_date), now_iso, exp_iso),
            )
            conn.execute("COMMIT")
            return {
                "lock_name": self.lock_name,
                "owner_run_id": str(owner_run_id),
                "trading_date": str(trading_date),
                "acquired_at_utc": now_iso,
                "expires_at_utc": exp_iso,
                "status": "HELD",
            }
        except Exception:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            raise
        finally:
            conn.close()

    def release(self, *, owner_run_id: str) -> None:
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                "UPDATE distributed_execution_locks SET status = 'RELEASED' WHERE lock_name = ? AND owner_run_id = ?",
                (self.lock_name, str(owner_run_id)),
            )
            conn.execute("COMMIT")
        except Exception:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            raise
        finally:
            conn.close()

    def force_clear_stale_lock_for_recovery(self, *, operator_note: str) -> dict[str, Any]:
        """Clear a stale crashed lock ONLY during explicit validated recovery."""
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            prev = conn.execute(
                "SELECT * FROM distributed_execution_locks WHERE lock_name = ?",
                (self.lock_name,),
            ).fetchone()
            conn.execute(
                "UPDATE distributed_execution_locks SET status = 'RECOVERED' WHERE lock_name = ?",
                (self.lock_name,),
            )
            conn.execute("COMMIT")
            return {
                "cleared": True,
                "operator_note": operator_note,
                "previous_lock": dict(prev) if prev is not None else None,
            }
        except Exception:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            raise
        finally:
            conn.close()

    def inspect_lock(self) -> dict[str, Any] | None:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT * FROM distributed_execution_locks WHERE lock_name = ?",
                (self.lock_name,),
            ).fetchone()
            return dict(row) if row is not None else None
        finally:
            conn.close()


class PersistentStorageAdapter:
    """Manages durable ledger pull/push, reconnection, checksum validation, and backend selection."""

    def __init__(
        self,
        *,
        db_path: Path = DEFAULT_PAPER_DB_PATH,
        storage_config_path: Path = DEFAULT_STORAGE_CONFIG_PATH,
        database_url: str | None = None,
        lock_name: str | None = None,
        advisory_lock_key: int = POSTGRES_ADVISORY_LOCK_KEY,
        ledger_id: str = "primary",
    ) -> None:
        PaperOnlyExecutionGuard().assert_paper_only()
        self.db_path = Path(db_path)
        self.storage_config_path = Path(storage_config_path)
        self.config = (
            json.loads(self.storage_config_path.read_text(encoding="utf-8"))
            if self.storage_config_path.exists()
            else {}
        )
        env_var = str(
            self.config.get("authoritative_storage_policy", {}).get("external_connection_env_var", "M31_DATABASE_URL")
        )
        raw_url = database_url if database_url is not None else os.getenv(env_var, "").strip()
        self._database_url = raw_url.strip()
        self.backend_mode = "EXTERNAL_POSTGRESQL" if self._database_url.startswith(("postgres://", "postgresql://")) else "DURABLE_GIT_LEDGER"
        self.advisory_lock_key = int(advisory_lock_key)
        self.ledger_id = str(ledger_id or "primary")
        self._pg_conn: Any = None
        self._pg_version: int = 0
        resolved_lock = lock_name or str(self.config.get("database_locking", {}).get("lock_name", DEFAULT_LOCK_NAME))
        self.db_lock = DatabaseExecutionLock(
            db_path=self.db_path,
            lock_name=resolved_lock,
            lease_ttl_seconds=int(self.config.get("database_locking", {}).get("lease_ttl_seconds", 1800)),
        )

    def _connect_postgres_with_retry(self) -> Any:
        """Connect to external PostgreSQL with retry & backoff (supports psycopg v3 or psycopg2)."""
        max_retries = int(self.config.get("reconnection_and_integrity", {}).get("max_connect_retries", 3))
        backoff = float(self.config.get("reconnection_and_integrity", {}).get("retry_backoff_seconds", 2.0))
        driver = None
        try:
            import psycopg  # type: ignore
            driver = ("psycopg", psycopg)
        except ImportError:
            try:
                import psycopg2  # type: ignore
                driver = ("psycopg2", psycopg2)
            except ImportError as exc:
                raise RuntimeError(
                    "EXTERNAL_POSTGRESQL configured via M31_DATABASE_URL, but neither 'psycopg' nor 'psycopg2' is installed."
                ) from exc

        last_err: Exception | None = None
        for attempt in range(1, max_retries + 1):
            try:
                conn = driver[1].connect(self._database_url)
                conn.autocommit = False
                return conn
            except Exception as exc:
                last_err = exc
                if attempt < max_retries:
                    time.sleep(backoff * attempt)
        raise RuntimeError(f"PostgreSQL connection failed after {max_retries} attempts: {redact_sensitive_text(str(last_err))}")

    def pull_durable_ledger(self, *, owner_run_id: str) -> dict[str, Any]:
        """Pull authoritative durable ledger before executing a session & verify SQLite integrity."""
        if self.backend_mode == "EXTERNAL_POSTGRESQL":
            conn = self._connect_postgres_with_retry()
            self._pg_conn = conn
            cur = conn.cursor()
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS m31_durable_ledger_store (
                    ledger_id TEXT PRIMARY KEY,
                    version_num INTEGER NOT NULL,
                    sha256_hex TEXT NOT NULL,
                    updated_at_utc TEXT NOT NULL,
                    owner_run_id TEXT NOT NULL,
                    sqlite_bytes BYTEA NOT NULL
                );
                """
            )
            cur.execute("SELECT pg_try_advisory_lock(%s)", (self.advisory_lock_key,))
            got_lock = bool(cur.fetchone()[0])
            if not got_lock:
                conn.rollback()
                conn.close()
                self._pg_conn = None
                raise DatabaseLockError(
                    f"POSTGRES_ADVISORY_LOCK_CONFLICT: another workflow holds lock {self.advisory_lock_key} ({self.ledger_id})."
                )
            conn.commit()

            cur.execute(
                "SELECT version_num, sha256_hex, sqlite_bytes FROM m31_durable_ledger_store WHERE ledger_id = %s",
                (self.ledger_id,),
            )
            row = cur.fetchone()
            if row is not None:
                ver, expected_sha, blob = int(row[0]), str(row[1]), bytes(row[2])
                actual_sha = hashlib.sha256(blob).hexdigest()
                if actual_sha != expected_sha:
                    raise RuntimeError(f"POSTGRES_LEDGER_CHECKSUM_MISMATCH: expected {expected_sha[:16]}, got {actual_sha[:16]}")
                self.db_path.parent.mkdir(parents=True, exist_ok=True)
                self.db_path.write_bytes(blob)
                self._pg_version = ver
            else:
                self._pg_version = 0

        # Verify SQLite PRAGMA integrity_check on local materialized ledger
        integrity_ok = self.verify_sqlite_integrity()
        if not integrity_ok:
            raise RuntimeError(f"SQLITE_INTEGRITY_CHECK_FAILED on {self.db_path}")

        return {
            "backend_mode": self.backend_mode,
            "owner_run_id": owner_run_id,
            "db_path": str(self.db_path),
            "sqlite_integrity_ok": integrity_ok,
            "ledger_sha256": compute_file_sha256(self.db_path) if self.db_path.exists() else "",
            "remote_version": self._pg_version,
            "credentials_redacted": True,
        }

    def push_durable_ledger(self, *, owner_run_id: str) -> dict[str, Any]:
        """Flush WAL checkpoint and atomically commit durable ledger to external PostgreSQL (if configured)."""
        # Force SQLite WAL checkpoint so the single `.db` file contains 100% of committed pages
        conn_sq = sqlite3.connect(str(self.db_path), timeout=30.0)
        try:
            conn_sq.execute("PRAGMA wal_checkpoint(TRUNCATE);")
        finally:
            conn_sq.close()

        sha_now = compute_file_sha256(self.db_path)
        if self.backend_mode == "EXTERNAL_POSTGRESQL":
            conn = self._pg_conn or self._connect_postgres_with_retry()
            blob = self.db_path.read_bytes()
            next_ver = self._pg_version + 1
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO m31_durable_ledger_store(
                    ledger_id, version_num, sha256_hex, updated_at_utc, owner_run_id, sqlite_bytes
                ) VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT(ledger_id) DO UPDATE SET
                    version_num = excluded.version_num,
                    sha256_hex = excluded.sha256_hex,
                    updated_at_utc = excluded.updated_at_utc,
                    owner_run_id = excluded.owner_run_id,
                    sqlite_bytes = excluded.sqlite_bytes
                """,
                (self.ledger_id, next_ver, sha_now, _utc_now_iso(), owner_run_id, blob),
            )
            cur.execute("SELECT pg_advisory_unlock(%s)", (self.advisory_lock_key,))
            conn.commit()
            conn.close()
            self._pg_conn = None
            self._pg_version = next_ver

        return {
            "backend_mode": self.backend_mode,
            "owner_run_id": owner_run_id,
            "ledger_sha256": sha_now,
            "remote_version": self._pg_version,
            "pushed_at_utc": _utc_now_iso(),
        }

    def close_without_push(self) -> None:
        """Preserve last valid remote state on workflow failure and release advisory lock cleanly."""
        if self._pg_conn is not None:
            try:
                cur = self._pg_conn.cursor()
                cur.execute("SELECT pg_advisory_unlock(%s)", (self.advisory_lock_key,))
                self._pg_conn.commit()
            except Exception:
                pass
            try:
                self._pg_conn.close()
            except Exception:
                pass
            self._pg_conn = None

    def verify_sqlite_integrity(self) -> bool:
        if not self.db_path.exists():
            return True
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        try:
            row = conn.execute("PRAGMA integrity_check;").fetchone()
            return row is not None and str(row[0]).lower() == "ok"
        finally:
            conn.close()

    def describe_storage(self) -> dict[str, Any]:
        return {
            "backend_mode": self.backend_mode,
            "ephemeral_runner_fs_authoritative": False,
            "github_actions_cache_used": False,
            "github_artifacts_authoritative": False,
            "external_postgres_configured": self.backend_mode == "EXTERNAL_POSTGRESQL",
            "local_db_path": str(self.db_path),
            "db_lock_state": self.db_lock.inspect_lock(),
        }
