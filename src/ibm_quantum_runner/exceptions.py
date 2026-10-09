"""Package-specific exceptions."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .validation import ValidationReport


class QuantumRunnerError(Exception):
    """Base class for all errors raised by this package."""


class ConfigurationError(QuantumRunnerError):
    """Configuration is missing or invalid."""


class CircuitValidationError(QuantumRunnerError):
    """Circuit validation failed before provider submission."""

    def __init__(
        self,
        message: str,
        *,
        report: ValidationReport,
        execution_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.report = report
        self.execution_id = execution_id


class SimulationError(QuantumRunnerError):
    """Local simulation or its user-supplied validation failed."""


class BackendSelectionError(QuantumRunnerError):
    """No suitable IBM backend could be selected."""


class HardwareSubmissionError(QuantumRunnerError):
    """A hardware request could not be submitted safely."""


class DuplicateExecutionError(HardwareSubmissionError):
    """An equivalent hardware execution already exists."""

    def __init__(self, existing_execution_id: str) -> None:
        super().__init__(
            f"Equivalent execution {existing_execution_id!r} already exists; "
            "resume it or set allow_duplicate_submission=True explicitly"
        )
        self.existing_execution_id = existing_execution_id


class ExecutionRecoveryError(QuantumRunnerError):
    """A persisted execution or provider job could not be recovered."""


class ResultRetrievalError(QuantumRunnerError):
    """A terminal result could not be retrieved or normalized."""


class StateStoreError(QuantumRunnerError):
    """Persistent state could not be read or written."""


class InvalidStateTransitionError(StateStoreError):
    """An impossible execution lifecycle transition was attempted."""


class ExecutionTimeoutError(QuantumRunnerError):
    """Waiting stopped before the provider job reached a terminal state."""


class HardwareResubmissionRequiredError(HardwareSubmissionError):
    """Explicit confirmation is required before resubmitting hardware work."""
