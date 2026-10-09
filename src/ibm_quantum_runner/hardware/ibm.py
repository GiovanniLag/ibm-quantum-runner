"""IBM Quantum Runtime adapter and bounded idempotent retries."""

from __future__ import annotations

import json
import logging
import random
import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, TypeVar
from urllib.parse import quote, quote_plus

from qiskit import QuantumCircuit
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from ..config import RetryConfig, RunConfig
from ..exceptions import BackendSelectionError, ConfigurationError

LOGGER = logging.getLogger(__name__)
T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class BackendSelection:
    backend: Any
    name: str
    reason: str
    candidates: tuple[Mapping[str, Any], ...] = field(default_factory=tuple)


def backend_name(backend: Any) -> str:
    if isinstance(backend, str):
        return backend
    value = getattr(backend, "name", None)
    resolved = value() if callable(value) else value
    return str(resolved) if resolved is not None else "unknown"


def raw_status_name(status: Any) -> str:
    """Return a stable provider status string across enum and string APIs."""

    value = getattr(status, "name", None)
    if value:
        return str(value).upper()
    value = getattr(status, "value", status)
    return str(value).split(".")[-1].upper()


def map_job_status(status: Any) -> str:
    raw = raw_status_name(status)
    return {
        "INITIALIZING": "SUBMITTED",
        "VALIDATING": "SUBMITTED",
        "QUEUED": "QUEUED",
        "RUNNING": "RUNNING",
        "DONE": "COMPLETED",
        "COMPLETED": "COMPLETED",
        "ERROR": "FAILED",
        "FAILED": "FAILED",
        "CANCELLED": "CANCELLED",
        "CANCELED": "CANCELLED",
    }.get(raw, "RECOVERY_REQUIRED")


def sanitize_provider_data(value: Any, sanitizer: Callable[[object], str]) -> Any:
    """Redact JSON-safe provider metadata recursively, including mapping keys.

    Callers should first normalize provider-specific objects to JSON-safe data.
    Numeric and boolean values keep their types so result data remains usable.
    """

    if isinstance(value, str):
        return sanitizer(value)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, Mapping):
        return {
            sanitizer(key): sanitize_provider_data(item, sanitizer) for key, item in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return [sanitize_provider_data(item, sanitizer) for item in value]
    return sanitizer(value)


def is_transient_provider_error(exc: BaseException) -> bool:
    """Conservatively classify retryable read/poll/result failures."""

    status_code = getattr(getattr(exc, "response", None), "status_code", None)
    if status_code == 429 or (isinstance(status_code, int) and 500 <= status_code < 600):
        return True
    if isinstance(exc, (ConnectionError, TimeoutError)):
        return True
    name = type(exc).__name__.lower()
    text = str(exc).lower()
    markers = (
        "connection",
        "timeout",
        "temporar",
        "serviceunavailable",
        "ratelimit",
        "too many requests",
    )
    return any(marker in name or marker in text for marker in markers)


def retry_idempotent(
    operation: Callable[[], T],
    policy: RetryConfig,
    *,
    description: str,
    on_retry: Callable[[Mapping[str, Any]], None] | None = None,
) -> T:
    """Retry only a caller-declared idempotent operation."""

    delay = policy.initial_delay
    for attempt in range(1, policy.max_attempts + 1):
        try:
            return operation()
        except Exception as exc:
            if attempt >= policy.max_attempts or not is_transient_provider_error(exc):
                raise
            jitter = random.uniform(1 - policy.jitter, 1 + policy.jitter)
            sleep_for = min(delay, policy.max_delay) * jitter
            details = {
                "operation": description,
                "attempt": attempt,
                "delay_seconds": sleep_for,
                "error_type": type(exc).__name__,
                "error": "transient provider operation failed",
            }
            if on_retry is not None:
                on_retry(details)
            LOGGER.warning(
                "Transient IBM operation failure; retrying",
                extra={"retry": details},
            )
            time.sleep(sleep_for)
            delay = min(delay * policy.multiplier, policy.max_delay)
    raise AssertionError("unreachable")


class IBMProvider:
    """Thin, injectable wrapper around Qiskit Runtime service APIs."""

    def __init__(
        self,
        *,
        token: str | None,
        channel: str,
        instance: str | None,
        default_backend: str | None,
        service: Any | None = None,
    ) -> None:
        self._token = token
        self.channel = channel
        self.instance = instance
        self.default_backend = default_backend
        self._service = service

    def sanitize_message(self, value: object) -> str:
        """Redact credentials even in escaped or case-normalized provider text.

        Status normalization uppercases strings, while provider exceptions may
        include URLs, JSON, or Python representations of request parameters.
        Never fall back to an object's repr if its string conversion fails.
        """

        try:
            message = str(value)
        except Exception:
            return "Provider message could not be displayed safely"
        if self._token:
            variants = {
                self._token,
                quote(self._token, safe=""),
                quote_plus(self._token, safe=""),
                json.dumps(self._token)[1:-1],
                repr(self._token)[1:-1],
            }
            pattern = "|".join(re.escape(item) for item in sorted(variants, key=len, reverse=True))
            message = re.sub(pattern, "[REDACTED]", message, flags=re.IGNORECASE)
        return message

    @property
    def service(self) -> Any:
        if self._service is None:
            if not self._token:
                raise ConfigurationError("QISKIT_IBM_TOKEN is required for IBM hardware operations")
            try:
                from qiskit_ibm_runtime import QiskitRuntimeService

                arguments: dict[str, Any] = {
                    "channel": self.channel,
                    "token": self._token,
                }
                if self.instance:
                    arguments["instance"] = self.instance
                self._service = QiskitRuntimeService(**arguments)
            except Exception as exc:
                raise ConfigurationError(
                    f"Could not initialize IBM Quantum service: {self.sanitize_message(exc)}"
                ) from None
        return self._service

    def get_backend(self, name: str) -> Any:
        try:
            return self.service.backend(name)
        except Exception as exc:
            raise BackendSelectionError(
                self.sanitize_message(
                    f"Could not retrieve IBM backend {name!r}: {self.sanitize_message(exc)}"
                )
            ) from None

    def _backend_status(self, backend: Any) -> tuple[bool, int | None, str]:
        try:
            status = backend.status()
            pending = getattr(status, "pending_jobs", None)
            return (
                bool(getattr(status, "operational", True)),
                int(pending) if pending is not None else None,
                self.sanitize_message(getattr(status, "status_msg", "")),
            )
        except Exception as exc:
            raise BackendSelectionError(
                f"Could not read IBM backend status: {self.sanitize_message(exc)}"
            ) from None

    @staticmethod
    def _num_qubits(backend: Any) -> int:
        return int(getattr(backend, "num_qubits", 0) or 0)

    def select_backend(
        self, circuits: Sequence[QuantumCircuit], config: RunConfig
    ) -> BackendSelection:
        # Backend attributes can themselves make provider calls. Keep failures
        # from name, capability, and discovery reads inside the redaction boundary.
        try:
            return self._select_backend(circuits, config)
        except Exception as exc:
            message = self.sanitize_message(exc)
            if not isinstance(exc, BackendSelectionError):
                message = f"Could not select IBM backend: {message}"
            raise BackendSelectionError(message) from None

    def _select_backend(
        self, circuits: Sequence[QuantumCircuit], config: RunConfig
    ) -> BackendSelection:
        requested = config.backend or self.default_backend
        required_qubits = max(circuit.num_qubits for circuit in circuits)
        if requested:
            backend = self.get_backend(requested)
            operational, pending, message = self._backend_status(backend)
            if not operational:
                raise BackendSelectionError(
                    f"IBM backend {requested!r} is not operational: "
                    f"{message or 'no reason provided'}"
                )
            if self._num_qubits(backend) < required_qubits:
                raise BackendSelectionError(
                    f"IBM backend {requested!r} has {self._num_qubits(backend)} qubits; "
                    f"the workload requires {required_qubits}"
                )
            return BackendSelection(
                backend,
                self.sanitize_message(backend_name(backend)),
                "explicit run configuration" if config.backend else "configured default backend",
                (
                    {
                        "name": self.sanitize_message(backend_name(backend)),
                        "pending_jobs": pending,
                        "num_qubits": self._num_qubits(backend),
                    },
                ),
            )

        if not config.auto_select_backend:
            raise BackendSelectionError(
                "No IBM backend was configured; set RunConfig.backend, "
                "QISKIT_IBM_BACKEND, or auto_select_backend=True"
            )

        try:
            candidates = list(
                self.service.backends(
                    min_num_qubits=required_qubits,
                    operational=True,
                    simulator=False,
                )
            )
        except TypeError:
            candidates = list(self.service.backends())
        except Exception as exc:
            raise BackendSelectionError(
                f"Could not discover IBM backends: {self.sanitize_message(exc)}"
            ) from None

        ranked: list[tuple[int, str, Any, dict[str, Any]]] = []
        for candidate in candidates:
            name = self.sanitize_message(backend_name(candidate))
            if bool(getattr(candidate, "simulator", False)):
                continue
            if self._num_qubits(candidate) < required_qubits:
                continue
            try:
                operational, pending, message = self._backend_status(candidate)
            except BackendSelectionError:
                continue
            if not operational:
                continue
            record = {
                "name": name,
                "pending_jobs": pending,
                "num_qubits": self._num_qubits(candidate),
                "status_message": message,
            }
            ranked.append((pending if pending is not None else 2**31, name, candidate, record))
        if not ranked:
            raise BackendSelectionError(
                f"No operational non-simulator IBM backend has {required_qubits} qubits"
            )
        ranked.sort(key=lambda item: (item[0], item[1]))
        compatible: list[tuple[int, str, Any, dict[str, Any]]] = []
        for item in ranked:
            try:
                self.transpile(circuits, item[2], config)
                item[3]["instruction_compatible"] = True
                compatible.append(item)
            except BackendSelectionError as exc:
                item[3]["instruction_compatible"] = False
                item[3]["incompatibility"] = self.sanitize_message(exc)
        if not compatible:
            raise BackendSelectionError(
                "No discovered IBM backend could transpile every circuit in the workload"
            )
        selected = compatible[0]
        return BackendSelection(
            selected[2],
            selected[1],
            "automatic selection: operational, sufficient qubits, instruction compatible, "
            "then lowest queue",
            tuple(item[3] for item in ranked),
        )

    def transpile(
        self, circuits: Sequence[QuantumCircuit], backend: Any, config: RunConfig
    ) -> list[QuantumCircuit]:
        try:
            manager = generate_preset_pass_manager(
                backend=backend,
                optimization_level=config.optimization_level,
                seed_transpiler=config.seed_transpiler,
            )
            result = manager.run(list(circuits))
            return [result] if isinstance(result, QuantumCircuit) else list(result)
        except Exception as exc:
            raise BackendSelectionError(
                f"IBM backend transpilation failed: {self.sanitize_message(exc)}"
            ) from None

    def retrieve_job(
        self,
        job_id: str,
        retry: RetryConfig,
        *,
        on_retry: Callable[[Mapping[str, Any]], None] | None = None,
    ) -> Any:
        try:
            return retry_idempotent(
                lambda: self.service.job(job_id),
                retry,
                description="retrieve job",
                on_retry=on_retry,
            )
        except Exception as exc:
            raise RuntimeError(
                self.sanitize_message(
                    f"Could not retrieve IBM job {job_id!r}: {self.sanitize_message(exc)}"
                )
            ) from None


__all__ = [
    "BackendSelection",
    "IBMProvider",
    "backend_name",
    "is_transient_provider_error",
    "map_job_status",
    "raw_status_name",
    "retry_idempotent",
    "sanitize_provider_data",
]
