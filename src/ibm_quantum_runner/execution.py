"""Persistent execution handle exposed by the public API."""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING, Any

from .results import ExecutionResult

if TYPE_CHECKING:
    from .runner import QuantumRunner


class ExecutionStatus(StrEnum):
    CREATED = "CREATED"
    VALIDATING = "VALIDATING"
    VALIDATION_FAILED = "VALIDATION_FAILED"
    SIMULATING = "SIMULATING"
    SIMULATION_FAILED = "SIMULATION_FAILED"
    READY_FOR_HARDWARE = "READY_FOR_HARDWARE"
    SUBMITTING = "SUBMITTING"
    PARTIALLY_SUBMITTED = "PARTIALLY_SUBMITTED"
    SUBMITTED = "SUBMITTED"
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    PARTIALLY_COMPLETED = "PARTIALLY_COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    RECOVERY_REQUIRED = "RECOVERY_REQUIRED"

    @property
    def is_terminal(self) -> bool:
        return self in {
            self.VALIDATION_FAILED,
            self.SIMULATION_FAILED,
            self.COMPLETED,
            self.PARTIALLY_COMPLETED,
            self.FAILED,
            self.CANCELLED,
        }


class Execution:
    """Lightweight handle backed by durable SQLite state."""

    def __init__(self, runner: QuantumRunner, execution_id: str) -> None:
        self._runner = runner
        self.id = execution_id

    @property
    def metadata(self) -> dict[str, Any]:
        return self._runner._store.get_execution(self.id)

    @property
    def ibm_job_ids(self) -> tuple[str, ...]:
        return tuple(
            str(job["ibm_job_id"])
            for job in self._runner._store.get_jobs(self.id)
            if job["ibm_job_id"]
        )

    @property
    def ibm_job_id(self) -> str | None:
        identifiers = self.ibm_job_ids
        return identifiers[0] if len(identifiers) == 1 else None

    def status(self, *, refresh: bool = True) -> ExecutionStatus:
        if refresh:
            self._runner._refresh_execution(self.id)
        return ExecutionStatus(self.metadata["status"])

    def wait(
        self,
        *,
        timeout: float | None = None,
        poll_interval: float | None = None,
    ) -> Execution:
        self._runner._wait(self.id, timeout=timeout, poll_interval=poll_interval)
        return self

    def result(
        self,
        *,
        wait: bool = True,
        allow_partial: bool = False,
        timeout: float | None = None,
    ) -> ExecutionResult:
        if wait and not ExecutionStatus(self.metadata["status"]).is_terminal:
            self.wait(timeout=timeout)
        return self._runner._result(self.id, allow_partial=allow_partial)

    def cancel(self) -> ExecutionStatus:
        return self._runner._cancel(self.id)

    def events(self) -> list[dict[str, Any]]:
        return self._runner._store.events(self.id)


__all__ = ["Execution", "ExecutionStatus"]
