"""Transactional SQLite state store."""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ..exceptions import DuplicateExecutionError, InvalidStateTransitionError, StateStoreError
from ..results.normalize import json_safe
from .artifacts import utc_now

STATE_SCHEMA_VERSION = 2

VALID_TRANSITIONS: dict[str, set[str]] = {
    "CREATED": {"VALIDATING", "RECOVERY_REQUIRED"},
    "VALIDATING": {"VALIDATION_FAILED", "SIMULATING", "READY_FOR_HARDWARE"},
    "VALIDATION_FAILED": set(),
    "SIMULATING": {"SIMULATION_FAILED", "READY_FOR_HARDWARE"},
    "SIMULATION_FAILED": {"READY_FOR_HARDWARE"},
    "READY_FOR_HARDWARE": {"SUBMITTING"},
    "SUBMITTING": {"SUBMITTED", "PARTIALLY_SUBMITTED", "FAILED", "RECOVERY_REQUIRED"},
    "PARTIALLY_SUBMITTED": {"SUBMITTED", "FAILED", "RECOVERY_REQUIRED"},
    "SUBMITTED": {
        "QUEUED",
        "RUNNING",
        "COMPLETED",
        "PARTIALLY_COMPLETED",
        "FAILED",
        "CANCELLED",
        "RECOVERY_REQUIRED",
    },
    "QUEUED": {
        "RUNNING",
        "COMPLETED",
        "PARTIALLY_COMPLETED",
        "FAILED",
        "CANCELLED",
        "RECOVERY_REQUIRED",
    },
    "RUNNING": {
        "QUEUED",
        "COMPLETED",
        "PARTIALLY_COMPLETED",
        "FAILED",
        "CANCELLED",
        "RECOVERY_REQUIRED",
    },
    "RECOVERY_REQUIRED": {
        "SUBMITTED",
        "QUEUED",
        "RUNNING",
        "COMPLETED",
        "PARTIALLY_COMPLETED",
        "FAILED",
        "CANCELLED",
    },
    "PARTIALLY_COMPLETED": set(),
    "COMPLETED": set(),
    "FAILED": set(),
    "CANCELLED": set(),
}


def _json(value: Any) -> str:
    return json.dumps(json_safe(value), sort_keys=True, separators=(",", ":"))


def _loads(value: str | None, default: Any) -> Any:
    return json.loads(value) if value else default


class StateStore:
    """Small per-operation connection store safe for concurrent readers."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.database_path = self.root / "state.sqlite3"
        self._init_lock = threading.Lock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection

    def _initialize(self) -> None:
        with self._init_lock:
            try:
                with self._connect() as connection:
                    connection.execute("PRAGMA journal_mode = WAL")
                    connection.executescript(
                        """
                        CREATE TABLE IF NOT EXISTS schema_info (
                            version INTEGER NOT NULL
                        );
                        CREATE TABLE IF NOT EXISTS executions (
                            id TEXT PRIMARY KEY,
                            schema_version INTEGER NOT NULL,
                            primitive TEXT NOT NULL,
                            status TEXT NOT NULL,
                            created_at TEXT NOT NULL,
                            updated_at TEXT NOT NULL,
                            backend_name TEXT,
                            config_json TEXT NOT NULL,
                            execution_fingerprint TEXT NOT NULL,
                            parent_execution_id TEXT,
                            result_path TEXT,
                            simulation_path TEXT,
                            failure_json TEXT,
                            metadata_json TEXT NOT NULL,
                            qiskit_version TEXT,
                            runtime_version TEXT,
                            FOREIGN KEY(parent_execution_id) REFERENCES executions(id)
                        );
                        CREATE INDEX IF NOT EXISTS idx_execution_fingerprint
                            ON executions(execution_fingerprint);
                        CREATE TABLE IF NOT EXISTS circuits (
                            execution_id TEXT NOT NULL,
                            circuit_index INTEGER NOT NULL,
                            fingerprint TEXT NOT NULL,
                            name TEXT NOT NULL,
                            qpy_path TEXT NOT NULL,
                            metadata_json TEXT NOT NULL,
                            transpiled_json TEXT,
                            PRIMARY KEY(execution_id, circuit_index),
                            FOREIGN KEY(execution_id) REFERENCES executions(id) ON DELETE CASCADE
                        );
                        CREATE TABLE IF NOT EXISTS jobs (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            execution_id TEXT NOT NULL,
                            child_index INTEGER NOT NULL,
                            attempt INTEGER NOT NULL DEFAULT 1,
                            ibm_job_id TEXT,
                            status TEXT NOT NULL,
                            raw_status TEXT,
                            backend_name TEXT,
                            circuit_indices_json TEXT NOT NULL,
                            submitted_at TEXT,
                            updated_at TEXT NOT NULL,
                            result_path TEXT,
                            failure_json TEXT,
                            retry_json TEXT NOT NULL DEFAULT '[]',
                            batch_id TEXT,
                            UNIQUE(execution_id, child_index, attempt),
                            FOREIGN KEY(execution_id) REFERENCES executions(id) ON DELETE CASCADE
                        );
                        CREATE TABLE IF NOT EXISTS events (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            execution_id TEXT NOT NULL,
                            child_index INTEGER,
                            created_at TEXT NOT NULL,
                            event_type TEXT NOT NULL,
                            from_status TEXT,
                            to_status TEXT,
                            details_json TEXT NOT NULL,
                            FOREIGN KEY(execution_id) REFERENCES executions(id) ON DELETE CASCADE
                        );
                        """
                    )
                    row = connection.execute("SELECT version FROM schema_info").fetchone()
                    if row is None:
                        connection.execute(
                            "INSERT INTO schema_info(version) VALUES (?)", (STATE_SCHEMA_VERSION,)
                        )
                    elif int(row["version"]) == 1:
                        self._migrate_v1_to_v2(connection)
                    elif int(row["version"]) > STATE_SCHEMA_VERSION:
                        raise StateStoreError(
                            f"State schema {row['version']} is newer than supported "
                            f"{STATE_SCHEMA_VERSION}"
                        )
            except sqlite3.Error as exc:
                raise StateStoreError(f"Could not initialize {self.database_path}: {exc}") from exc

    @staticmethod
    def _migrate_v1_to_v2(connection: sqlite3.Connection) -> None:
        """Allow a provider job to be referenced by an explicit resubmission lineage."""

        connection.executescript(
            """
            ALTER TABLE jobs RENAME TO jobs_v1;
            CREATE TABLE jobs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                execution_id TEXT NOT NULL,
                child_index INTEGER NOT NULL,
                attempt INTEGER NOT NULL DEFAULT 1,
                ibm_job_id TEXT,
                status TEXT NOT NULL,
                raw_status TEXT,
                backend_name TEXT,
                circuit_indices_json TEXT NOT NULL,
                submitted_at TEXT,
                updated_at TEXT NOT NULL,
                result_path TEXT,
                failure_json TEXT,
                retry_json TEXT NOT NULL DEFAULT '[]',
                batch_id TEXT,
                UNIQUE(execution_id, child_index, attempt),
                FOREIGN KEY(execution_id) REFERENCES executions(id) ON DELETE CASCADE
            );
            INSERT INTO jobs(
                id, execution_id, child_index, attempt, ibm_job_id, status,
                raw_status, backend_name, circuit_indices_json, submitted_at,
                updated_at, result_path, failure_json, retry_json, batch_id
            )
            SELECT
                id, execution_id, child_index, attempt, ibm_job_id, status,
                raw_status, backend_name, circuit_indices_json, submitted_at,
                updated_at, result_path, failure_json, retry_json, batch_id
            FROM jobs_v1;
            DROP TABLE jobs_v1;
            UPDATE schema_info SET version = 2;
            """
        )

    def execution_dir(self, execution_id: str) -> Path:
        return self.root / "executions" / execution_id

    def create_execution(
        self,
        *,
        primitive: str,
        config: Mapping[str, Any],
        fingerprint: str,
        metadata: Mapping[str, Any],
        qiskit_version: str,
        runtime_version: str,
        parent_execution_id: str | None = None,
        execution_id: str | None = None,
        reject_duplicate: bool = False,
    ) -> str:
        execution_id = execution_id or str(uuid.uuid4())
        timestamp = utc_now()
        try:
            with self._connect() as connection:
                # Serialize the duplicate check and reservation across processes.
                connection.execute("BEGIN IMMEDIATE")
                if reject_duplicate:
                    existing = connection.execute(
                        "SELECT id FROM executions WHERE execution_fingerprint = ? LIMIT 1",
                        (fingerprint,),
                    ).fetchone()
                    if existing is not None:
                        raise DuplicateExecutionError(str(existing["id"]))
                if parent_execution_id is not None:
                    replacements = connection.execute(
                        "SELECT id, status, config_json FROM executions "
                        "WHERE parent_execution_id = ? ORDER BY created_at DESC",
                        (parent_execution_id,),
                    ).fetchall()
                    for replacement in replacements:
                        preflight_terminal = replacement["status"] == "VALIDATION_FAILED" or (
                            replacement["status"] == "SIMULATION_FAILED"
                            and _loads(replacement["config_json"], {}).get(
                                "stop_on_simulation_failure", True
                            )
                        )
                        has_jobs = connection.execute(
                            "SELECT 1 FROM jobs WHERE execution_id = ? LIMIT 1",
                            (replacement["id"],),
                        ).fetchone()
                        if not preflight_terminal or has_jobs is not None:
                            raise DuplicateExecutionError(str(replacement["id"]))
                    parent_jobs = connection.execute(
                        "SELECT status, ibm_job_id FROM jobs WHERE execution_id = ?",
                        (parent_execution_id,),
                    ).fetchall()
                    if not parent_jobs or any(
                        (
                            job["ibm_job_id"]
                            and job["status"] not in {"COMPLETED", "FAILED", "CANCELLED"}
                        )
                        or (
                            not job["ibm_job_id"]
                            and job["status"]
                            not in {"PLANNED", "NOT_SUBMITTED", "SUPERSEDED_NOT_SUBMITTED"}
                        )
                        for job in parent_jobs
                    ):
                        raise StateStoreError(
                            "Original child submission changed or remains ambiguous; "
                            "refusing replacement"
                        )
                    # Fence an original submitter paused before primitive.run.
                    # The parent recheck and this fence share the replacement
                    # reservation transaction with claim_planned_job's writer lock.
                    connection.execute(
                        "UPDATE jobs SET status = 'SUPERSEDED_NOT_SUBMITTED', updated_at = ? "
                        "WHERE execution_id = ? AND status = 'PLANNED' AND ibm_job_id IS NULL",
                        (utc_now(), parent_execution_id),
                    )
                self.execution_dir(execution_id).mkdir(parents=True, exist_ok=False)
                connection.execute(
                    """
                    INSERT INTO executions(
                        id, schema_version, primitive, status, created_at, updated_at,
                        config_json, execution_fingerprint, parent_execution_id,
                        metadata_json, qiskit_version, runtime_version
                    ) VALUES (?, ?, ?, 'CREATED', ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        execution_id,
                        STATE_SCHEMA_VERSION,
                        primitive,
                        timestamp,
                        timestamp,
                        _json(config),
                        fingerprint,
                        parent_execution_id,
                        _json(metadata),
                        qiskit_version,
                        runtime_version,
                    ),
                )
                self._event(connection, execution_id, "created", None, "CREATED", metadata)
        except sqlite3.Error as exc:
            raise StateStoreError(f"Could not create execution {execution_id}: {exc}") from exc
        return execution_id

    def _event(
        self,
        connection: sqlite3.Connection,
        execution_id: str,
        event_type: str,
        from_status: str | None,
        to_status: str | None,
        details: Mapping[str, Any] | None = None,
        child_index: int | None = None,
    ) -> None:
        connection.execute(
            """
            INSERT INTO events(
                execution_id, child_index, created_at, event_type,
                from_status, to_status, details_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                execution_id,
                child_index,
                utc_now(),
                event_type,
                from_status,
                to_status,
                _json(details or {}),
            ),
        )

    def add_event(
        self,
        execution_id: str,
        event_type: str,
        details: Mapping[str, Any],
        *,
        child_index: int | None = None,
    ) -> None:
        with self._connect() as connection:
            self._event(connection, execution_id, event_type, None, None, details, child_index)

    def transition(
        self,
        execution_id: str,
        new_status: str,
        *,
        details: Mapping[str, Any] | None = None,
        force: bool = False,
        only_from: set[str] | None = None,
    ) -> None:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT status FROM executions WHERE id = ?", (execution_id,)
            ).fetchone()
            if row is None:
                raise StateStoreError(f"Unknown execution: {execution_id}")
            current = str(row["status"])
            if only_from is not None and current not in only_from:
                return
            if current == new_status:
                return
            if not force and new_status not in VALID_TRANSITIONS.get(current, set()):
                raise InvalidStateTransitionError(
                    f"Invalid execution transition {current} -> {new_status}"
                )
            connection.execute(
                "UPDATE executions SET status = ?, updated_at = ? WHERE id = ?",
                (new_status, utc_now(), execution_id),
            )
            self._event(connection, execution_id, "transition", current, new_status, details)

    def update_execution(self, execution_id: str, **fields: Any) -> None:
        allowed = {
            "backend_name",
            "result_path",
            "simulation_path",
            "failure_json",
            "metadata_json",
        }
        unknown = set(fields) - allowed
        if unknown:
            raise StateStoreError(f"Unsupported execution fields: {sorted(unknown)}")
        converted: dict[str, Any] = {}
        for key, value in fields.items():
            converted[key] = _json(value) if key.endswith("_json") and value is not None else value
        converted["updated_at"] = utc_now()
        assignments = ", ".join(f"{name} = ?" for name in converted)
        with self._connect() as connection:
            connection.execute(
                f"UPDATE executions SET {assignments} WHERE id = ?",
                (*converted.values(), execution_id),
            )

    def add_circuits(
        self,
        execution_id: str,
        records: Sequence[Mapping[str, Any]],
        qpy_path: Path,
    ) -> None:
        with self._connect() as connection:
            for record in records:
                connection.execute(
                    """
                    INSERT INTO circuits(
                        execution_id, circuit_index, fingerprint, name, qpy_path, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        execution_id,
                        int(record["circuit_index"]),
                        str(record["fingerprint"]),
                        str(record["name"]),
                        str(qpy_path),
                        _json(record),
                    ),
                )

    def update_transpiled(
        self, execution_id: str, circuit_index: int, metadata: Mapping[str, Any]
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE circuits SET transpiled_json = ?
                WHERE execution_id = ? AND circuit_index = ?
                """,
                (_json(metadata), execution_id, circuit_index),
            )

    def plan_jobs(
        self,
        execution_id: str,
        chunks: Sequence[Sequence[int]],
        *,
        child_index_offset: int = 0,
        backend_name: str | None = None,
    ) -> None:
        """Atomically record every intended child before any submission side effect.

        Existing rows are never replaced: entering submission twice is unsafe even
        when an earlier caller failed before returning a provider identifier.
        """
        indices = [index for chunk in chunks for index in chunk]
        if not chunks or any(not chunk for chunk in chunks) or len(set(indices)) != len(indices):
            raise StateStoreError("Submission plan must contain unique, nonempty circuit groups")
        try:
            with self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                expected = [
                    int(row["circuit_index"])
                    for row in connection.execute(
                        "SELECT circuit_index FROM circuits WHERE execution_id = ? "
                        "ORDER BY circuit_index",
                        (execution_id,),
                    )
                ]
                existing = [
                    index
                    for row in connection.execute(
                        "SELECT circuit_indices_json FROM jobs WHERE execution_id = ?",
                        (execution_id,),
                    )
                    for index in _loads(row["circuit_indices_json"], [])
                ]
                if expected and sorted([*existing, *indices]) != expected:
                    raise StateStoreError(
                        "Submission plan must cover every original circuit exactly once"
                    )
                for offset, chunk in enumerate(chunks):
                    connection.execute(
                        """INSERT INTO jobs(
                            execution_id, child_index, status, backend_name,
                            circuit_indices_json, updated_at
                        ) VALUES (?, ?, 'PLANNED', ?, ?, ?)""",
                        (
                            execution_id,
                            child_index_offset + offset,
                            backend_name,
                            _json(list(chunk)),
                            utc_now(),
                        ),
                    )
                self._event(
                    connection,
                    execution_id,
                    "submission_planned",
                    None,
                    None,
                    {"chunks": list(chunks), "child_index_offset": child_index_offset},
                )
        except sqlite3.Error as exc:
            raise StateStoreError("Could not reserve the full submission plan") from exc

    def claim_planned_job(self, execution_id: str, child_index: int) -> None:
        """Durably cross the ambiguity boundary exactly once, before primitive.run."""
        with self._connect() as connection:
            updated = connection.execute(
                """UPDATE jobs SET status = 'SUBMITTING', updated_at = ?
                   WHERE execution_id = ? AND child_index = ? AND attempt = 1
                     AND status = 'PLANNED' AND ibm_job_id IS NULL""",
                (utc_now(), execution_id, child_index),
            )
            if updated.rowcount != 1:
                raise StateStoreError("Child submission was already claimed; refusing to retry")
            self._event(
                connection,
                execution_id,
                "submission_started",
                "PLANNED",
                "SUBMITTING",
                child_index=child_index,
            )

    def upsert_job(
        self,
        execution_id: str,
        child_index: int,
        *,
        circuit_indices: Sequence[int],
        status: str,
        attempt: int = 1,
        ibm_job_id: str | None = None,
        raw_status: str | None = None,
        backend_name: str | None = None,
        submitted_at: str | None = None,
        result_path: str | None = None,
        failure: Mapping[str, Any] | None = None,
        retry_history: Sequence[Mapping[str, Any]] = (),
        batch_id: str | None = None,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO jobs(
                    execution_id, child_index, attempt, ibm_job_id, status, raw_status,
                    backend_name, circuit_indices_json, submitted_at, updated_at,
                    result_path, failure_json, retry_json, batch_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(execution_id, child_index, attempt) DO UPDATE SET
                    ibm_job_id=COALESCE(excluded.ibm_job_id, jobs.ibm_job_id),
                    status=excluded.status,
                    raw_status=excluded.raw_status,
                    backend_name=excluded.backend_name,
                    submitted_at=COALESCE(excluded.submitted_at, jobs.submitted_at),
                    updated_at=excluded.updated_at,
                    result_path=COALESCE(excluded.result_path, jobs.result_path),
                    failure_json=excluded.failure_json,
                    retry_json=excluded.retry_json,
                    batch_id=COALESCE(excluded.batch_id, jobs.batch_id)
                """,
                (
                    execution_id,
                    child_index,
                    attempt,
                    ibm_job_id,
                    status,
                    raw_status,
                    backend_name,
                    _json(list(circuit_indices)),
                    submitted_at,
                    utc_now(),
                    result_path,
                    _json(failure) if failure else None,
                    _json(list(retry_history)),
                    batch_id,
                ),
            )
            self._event(
                connection,
                execution_id,
                "job_update",
                None,
                status,
                {"ibm_job_id": ibm_job_id, "raw_status": raw_status},
                child_index,
            )

    def get_execution(self, execution_id: str) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM executions WHERE id = ?", (execution_id,)
            ).fetchone()
        if row is None:
            raise StateStoreError(f"Unknown execution: {execution_id}")
        result = dict(row)
        for key, default in (
            ("config_json", {}),
            ("failure_json", None),
            ("metadata_json", {}),
        ):
            result[key.removesuffix("_json")] = _loads(result.pop(key), default)
        return result

    def get_circuits(self, execution_id: str) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM circuits WHERE execution_id = ? ORDER BY circuit_index",
                (execution_id,),
            ).fetchall()
        records: list[dict[str, Any]] = []
        for row in rows:
            record = dict(row)
            record["metadata"] = _loads(record.pop("metadata_json"), {})
            record["transpiled"] = _loads(record.pop("transpiled_json"), None)
            records.append(record)
        return records

    def get_jobs(self, execution_id: str) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM jobs WHERE execution_id = ?
                ORDER BY child_index, attempt
                """,
                (execution_id,),
            ).fetchall()
        jobs: list[dict[str, Any]] = []
        for row in rows:
            job = dict(row)
            job["circuit_indices"] = _loads(job.pop("circuit_indices_json"), [])
            job["failure"] = _loads(job.pop("failure_json"), None)
            job["retry_history"] = _loads(job.pop("retry_json"), [])
            jobs.append(job)
        return jobs

    def find_by_fingerprint(self, fingerprint: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT id FROM executions
                WHERE execution_fingerprint = ?
                ORDER BY created_at DESC LIMIT 1
                """,
                (fingerprint,),
            ).fetchone()
        return self.get_execution(str(row["id"])) if row is not None else None

    def find_by_ibm_job_id(self, ibm_job_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT execution_id FROM jobs WHERE ibm_job_id = ?
                ORDER BY id DESC LIMIT 1
                """,
                (ibm_job_id,),
            ).fetchone()
        return self.get_execution(str(row["execution_id"])) if row is not None else None

    def events(self, execution_id: str) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM events WHERE execution_id = ? ORDER BY id", (execution_id,)
            ).fetchall()
        events: list[dict[str, Any]] = []
        for row in rows:
            event = dict(row)
            event["details"] = _loads(event.pop("details_json"), {})
            events.append(event)
        return events
