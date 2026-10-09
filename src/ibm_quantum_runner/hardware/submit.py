"""Runtime primitive PUB preparation, chunking, and submission."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from contextlib import nullcontext, suppress
from dataclasses import dataclass
from typing import Any

import numpy as np
from qiskit import QuantumCircuit

from ..config import EstimateRequest, RunConfig
from ..exceptions import HardwareSubmissionError
from ..persistence import StateStore
from ..persistence.artifacts import normalize_observables, utc_now
from ..validation.core import SAMPLER_MAX_EXECUTIONS_PER_JOB
from .ibm import backend_name


@dataclass(frozen=True, slots=True)
class SubmittedJob:
    child_index: int
    ibm_job_id: str
    circuit_indices: tuple[int, ...]
    raw_job: Any
    batch_id: str | None = None


def partition_indices(count: int, primitive: str, config: RunConfig) -> list[tuple[int, ...]]:
    size = config.max_pubs_per_job
    if primitive == "sampler":
        size = min(size, SAMPLER_MAX_EXECUTIONS_PER_JOB // config.shots)
    if size <= 0:
        raise HardwareSubmissionError("Shot count exceeds IBM's per-job execution limit")
    return [tuple(range(start, min(start + size, count))) for start in range(0, count, size)]


def _primitive_options(execution_id: str, config: RunConfig) -> dict[str, Any]:
    options: dict[str, Any] = {
        "environment": {"job_tags": ["ibm-quantum-runner", f"execution:{execution_id}"]}
    }
    if config.max_execution_time is not None:
        options["max_execution_time"] = config.max_execution_time
    return options


def _job_id(job: Any) -> str:
    value = getattr(job, "job_id", None)
    identifier = value() if callable(value) else value
    if not identifier:
        raise HardwareSubmissionError("IBM Runtime returned a job without a job ID")
    return str(identifier)


def _batch_id(batch: Any) -> str | None:
    value = getattr(batch, "session_id", None)
    identifier = value() if callable(value) else value
    return str(identifier) if identifier else None


def _estimator_pub(request: EstimateRequest, transpiled: QuantumCircuit) -> tuple[Any, ...]:
    observable = normalize_observables(request.observables)
    layout = getattr(transpiled, "layout", None)
    if layout is not None:
        if isinstance(observable, np.ndarray):
            mapped = np.empty_like(observable)
            for index, item in np.ndenumerate(observable):
                mapped[index] = item.apply_layout(layout)
            observable = mapped
        else:
            observable = observable.apply_layout(layout)
    if request.parameter_values is None:
        return (transpiled, observable)
    return (transpiled, observable, request.parameter_values)


def submit_workload(
    *,
    execution_id: str,
    primitive: str,
    originals: Sequence[QuantumCircuit] | Sequence[EstimateRequest],
    transpiled: Sequence[QuantumCircuit],
    backend: Any,
    config: RunConfig,
    store: StateStore,
    selected_indices: Sequence[int] | None = None,
    child_index_offset: int = 0,
    sanitize_message: Callable[[object], str] | None = None,
) -> tuple[SubmittedJob, ...]:
    """Persist the complete plan, then submit each child at most once.

    PLANNED is known unsubmitted. SUBMITTING without an ID is ambiguous after
    interruption and must never be automatically replayed, including on resume.
    """

    # Direct adapter callers without a credential-aware sanitizer get only generic
    # errors, never raw provider text. The runner supplies its configured sanitizer.
    def safe(value: object) -> str:
        return sanitize_message(value) if sanitize_message else "provider operation failed"

    selected = (
        list(selected_indices) if selected_indices is not None else list(range(len(transpiled)))
    )
    if (
        not selected
        or len(set(selected)) != len(selected)
        or any(index < 0 or index >= len(transpiled) for index in selected)
    ):
        raise HardwareSubmissionError("Submission indices must be unique and within the workload")
    local_chunks = partition_indices(len(selected), primitive, config)
    chunks = [tuple(selected[index] for index in chunk) for chunk in local_chunks]
    name = safe(backend_name(backend)) if sanitize_message else backend_name(backend)
    store.plan_jobs(execution_id, chunks, child_index_offset=child_index_offset, backend_name=name)
    submitted: list[SubmittedJob] = []
    try:
        from qiskit_ibm_runtime import Batch, EstimatorV2, SamplerV2

        batch_context: Any = Batch(backend=backend) if len(chunks) > 1 else nullcontext(backend)
        with batch_context as mode:
            batch_identifier = _batch_id(mode) if len(chunks) > 1 else None
            if batch_identifier and sanitize_message:
                batch_identifier = safe(batch_identifier)
            for local_child_index, indices in enumerate(chunks):
                child_index = child_index_offset + local_child_index
                identifier: str | None = None
                boundary_crossed = False
                claim_succeeded = False
                try:
                    options = _primitive_options(execution_id, config)
                    if primitive == "sampler":
                        primitive_runner: Any = SamplerV2(mode=mode, options=options)
                        pubs = [transpiled[index] for index in indices]
                        run_options: dict[str, Any] = {"shots": config.shots}
                    else:
                        primitive_runner = EstimatorV2(mode=mode, options=options)
                        pubs = [
                            _estimator_pub(originals[index], transpiled[index]) for index in indices
                        ]
                        run_options = {"precision": config.estimator_precision()}
                    boundary_crossed = True
                    store.claim_planned_job(execution_id, child_index)
                    claim_succeeded = True
                    job = primitive_runner.run(pubs, **run_options)
                    identifier = _job_id(job)
                    # A provider identifier must stay byte-for-byte usable. If it
                    # unexpectedly contains a credential, refuse to persist it.
                    if sanitize_message and safe(identifier) != identifier:
                        identifier = None
                        raise HardwareSubmissionError("Provider returned an unsafe job identifier")
                    store.upsert_job(
                        execution_id,
                        child_index,
                        circuit_indices=indices,
                        status="SUBMITTED",
                        ibm_job_id=identifier,
                        raw_status="INITIALIZING",
                        backend_name=name,
                        submitted_at=utc_now(),
                        batch_id=batch_identifier,
                    )
                    submitted.append(
                        SubmittedJob(child_index, identifier, indices, job, batch_identifier)
                    )
                except Exception as exc:
                    message = safe(exc)
                    # A second persistence attempt is read/write recovery only;
                    # primitive.run is NEVER retried. Preserve any known job ID.
                    if not boundary_crossed or claim_succeeded:
                        with suppress(Exception):
                            store.upsert_job(
                                execution_id,
                                child_index,
                                circuit_indices=indices,
                                status="RECOVERY_REQUIRED" if boundary_crossed else "NOT_SUBMITTED",
                                ibm_job_id=identifier,
                                backend_name=name,
                                failure={"type": type(exc).__name__, "message": message},
                                batch_id=batch_identifier,
                            )
                    phase = "was ambiguous" if boundary_crossed else "failed before submission"
                    raise HardwareSubmissionError(
                        f"IBM child job {child_index} {phase}: {message}"
                    ) from None
    except HardwareSubmissionError:
        raise
    except Exception as exc:
        raise HardwareSubmissionError(
            f"IBM Runtime batch setup or finalization failed: {safe(exc)}"
        ) from None
    return tuple(submitted)


__all__ = ["SubmittedJob", "partition_indices", "submit_workload"]
