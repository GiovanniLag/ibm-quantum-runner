"""High-level orchestration for validation, simulation, hardware, and recovery."""

from __future__ import annotations

import importlib.metadata
import logging
import os
import time
from collections.abc import Mapping, Sequence
from functools import partial
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import qiskit
from qiskit import QuantumCircuit

from .config import EstimateRequest, RunConfig, run_config_from_dict
from .exceptions import (
    BackendSelectionError,
    CircuitValidationError,
    ConfigurationError,
    ExecutionRecoveryError,
    ExecutionTimeoutError,
    HardwareResubmissionRequiredError,
    HardwareSubmissionError,
    ResultRetrievalError,
    SimulationError,
    StateStoreError,
)
from .execution import Execution, ExecutionStatus
from .hardware import (
    IBMProvider,
    backend_name,
    map_job_status,
    partition_indices,
    raw_status_name,
    retry_idempotent,
    submit_workload,
)
from .hardware.ibm import sanitize_provider_data
from .logging import execution_logger, set_package_log_level
from .persistence import StateStore
from .persistence.artifacts import (
    ARTIFACT_SCHEMA_VERSION,
    atomic_write_json,
    circuit_summary,
    load_circuits,
    load_estimate_inputs,
    load_json,
    normalize_observables,
    save_circuits,
    save_estimate_inputs,
    serialize_observable,
    stable_hash,
    utc_now,
)
from .results import (
    CircuitResult,
    EstimatorCircuitResult,
    ExecutionResult,
    HardwareResult,
    SamplerCircuitResult,
    circuit_result_from_dict,
    normalize_estimator_result,
    normalize_sampler_result,
)
from .results.normalize import json_safe
from .simulation import (
    SimulationResult,
    simulate_estimator,
    simulate_sampler,
    simulation_result_from_dict,
)
from .validation import (
    ValidationReport,
    layout_metadata,
    validate_estimator_inputs,
    validate_sampler_inputs,
    validate_transpiled_circuits,
)

LOGGER = logging.getLogger(__name__)
TERMINAL_JOB_STATES = {"COMPLETED", "FAILED", "CANCELLED"}


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


class QuantumRunner:
    """Validate, simulate, submit, monitor, and recover Qiskit workloads."""

    def __init__(
        self,
        *,
        token: str | None = None,
        channel: str = "ibm_quantum_platform",
        instance: str | None = None,
        backend: str | None = None,
        state_dir: str | Path = ".quantum-runs",
        log_level: str | int | None = None,
        provider: IBMProvider | None = None,
    ) -> None:
        if not channel:
            raise ConfigurationError("IBM channel cannot be empty")
        self._store = StateStore(state_dir)
        self._provider = provider or IBMProvider(
            token=token,
            channel=channel,
            instance=instance,
            default_backend=backend,
        )
        self._redact_configured_token = IBMProvider(
            token=token, channel=channel, instance=None, default_backend=None
        ).sanitize_message
        self._raw_results: dict[str, list[Any]] = {}
        if log_level is not None:
            set_package_log_level(log_level)

    def _safe_provider_message(self, value: object) -> str:
        sanitizer = getattr(self._provider, "sanitize_message", None)
        message = str(sanitizer(value)) if callable(sanitizer) else str(value)
        return self._redact_configured_token(message)

    def _safe_provider_data(self, value: Any) -> Any:
        return sanitize_provider_data(json_safe(value), self._safe_provider_message)

    @classmethod
    def from_env(
        cls,
        *,
        token: str | None = None,
        channel: str | None = None,
        instance: str | None = None,
        backend: str | None = None,
        state_dir: str | Path | None = None,
        log_level: str | int | None = None,
        dotenv_path: str | Path | None = None,
        provider: IBMProvider | None = None,
    ) -> QuantumRunner:
        """Construct with explicit arguments taking precedence over environment values."""

        if dotenv_path is not None:
            try:
                from dotenv import load_dotenv
            except ImportError as exc:
                raise ConfigurationError("python-dotenv is required to load a .env file") from exc
            load_dotenv(Path(dotenv_path), override=False)
        return cls(
            token=token if token is not None else os.getenv("QISKIT_IBM_TOKEN"),
            channel=channel
            if channel is not None
            else os.getenv("QISKIT_IBM_CHANNEL", "ibm_quantum_platform"),
            instance=instance if instance is not None else os.getenv("QISKIT_IBM_INSTANCE"),
            backend=backend if backend is not None else os.getenv("QISKIT_IBM_BACKEND"),
            state_dir=state_dir
            if state_dir is not None
            else os.getenv("QUANTUM_RUNNER_STATE_DIR", ".quantum-runs"),
            log_level=log_level if log_level is not None else os.getenv("QUANTUM_RUNNER_LOG_LEVEL"),
            provider=provider,
        )

    @staticmethod
    def _circuits(value: QuantumCircuit | Sequence[QuantumCircuit]) -> list[QuantumCircuit]:
        if isinstance(value, QuantumCircuit):
            return [value]
        return list(value)

    @staticmethod
    def _requests(value: EstimateRequest | Sequence[EstimateRequest]) -> list[EstimateRequest]:
        if isinstance(value, EstimateRequest):
            return [value]
        return list(value)

    def validate(
        self,
        circuits: QuantumCircuit | Sequence[QuantumCircuit],
        *,
        config: RunConfig | None = None,
        include_hardware: bool = True,
    ) -> ValidationReport:
        config = config or RunConfig()
        values = self._circuits(circuits)
        report = validate_sampler_inputs(values, config)
        if report.is_valid and include_hardware:
            self._add_hardware_validation(report, values, config)
        return report

    def validate_estimate(
        self,
        requests: EstimateRequest | Sequence[EstimateRequest],
        *,
        config: RunConfig | None = None,
        include_hardware: bool = True,
    ) -> ValidationReport:
        config = config or RunConfig()
        values = self._requests(requests)
        report = validate_estimator_inputs(values, config)
        if report.is_valid and include_hardware:
            self._hardware_validation(
                report,
                [request.circuit for request in values],
                config,
                "estimator",
            )
        return report

    def _add_hardware_validation(
        self,
        report: ValidationReport,
        circuits: Sequence[QuantumCircuit],
        config: RunConfig,
    ) -> tuple[Any | None, list[QuantumCircuit]]:
        try:
            selection = self._provider.select_backend(circuits, config)
            transpiled = self._provider.transpile(circuits, selection.backend, config)
            chunks = partition_indices(len(transpiled), "sampler", config)
            for indices in chunks:
                provider_report = validate_transpiled_circuits(
                    [transpiled[index] for index in indices],
                    primitive="sampler",
                    config=config,
                    backend_num_qubits=int(selection.backend.num_qubits),
                )
                report.issues.extend(provider_report.issues)
            report.backend_name = self._safe_provider_message(selection.name)
            report.backend_selection = self._safe_provider_data(
                {
                    "reason": selection.reason,
                    "candidates": list(selection.candidates),
                }
            )
            report.transpiled_metadata = [layout_metadata(item) for item in transpiled]
            return selection.backend, transpiled
        except (ConfigurationError, BackendSelectionError, HardwareSubmissionError) as exc:
            report.add(
                "error",
                "hardware-preflight-failed",
                self._safe_provider_message(exc),
                details={"error_type": type(exc).__name__},
            )
            return None, []

    def _hardware_validation(
        self,
        report: ValidationReport,
        circuits: Sequence[QuantumCircuit],
        config: RunConfig,
        primitive: Literal["sampler", "estimator"],
    ) -> tuple[Any | None, list[QuantumCircuit]]:
        try:
            selection = self._provider.select_backend(circuits, config)
            transpiled = self._provider.transpile(circuits, selection.backend, config)
            for indices in partition_indices(len(transpiled), primitive, config):
                provider_report = validate_transpiled_circuits(
                    [transpiled[index] for index in indices],
                    primitive=primitive,
                    config=config,
                    backend_num_qubits=int(selection.backend.num_qubits),
                )
                report.issues.extend(provider_report.issues)
            report.backend_name = self._safe_provider_message(selection.name)
            report.backend_selection = self._safe_provider_data(
                {
                    "reason": selection.reason,
                    "candidates": list(selection.candidates),
                }
            )
            report.transpiled_metadata = [layout_metadata(item) for item in transpiled]
            return selection.backend, transpiled
        except (ConfigurationError, BackendSelectionError, HardwareSubmissionError) as exc:
            report.add(
                "error",
                "hardware-preflight-failed",
                self._safe_provider_message(exc),
                details={"error_type": type(exc).__name__},
            )
            return None, []

    def simulate(
        self,
        circuits: QuantumCircuit | Sequence[QuantumCircuit],
        *,
        config: RunConfig | None = None,
    ) -> SimulationResult:
        config = config or RunConfig()
        values = self._circuits(circuits)
        report = validate_sampler_inputs(values, config)
        if not report.is_valid:
            raise CircuitValidationError("Local circuit validation failed", report=report)
        try:
            return simulate_sampler(values, config)
        except CircuitValidationError:
            raise
        except Exception as exc:
            raise SimulationError(f"Local Sampler simulation failed: {exc}") from exc

    def simulate_estimate(
        self,
        requests: EstimateRequest | Sequence[EstimateRequest],
        *,
        config: RunConfig | None = None,
    ) -> SimulationResult:
        config = config or RunConfig()
        values = self._requests(requests)
        report = validate_estimator_inputs(values, config)
        if not report.is_valid:
            raise CircuitValidationError("Local Estimator validation failed", report=report)
        try:
            return simulate_estimator(values, config)
        except CircuitValidationError:
            raise
        except Exception as exc:
            raise SimulationError(f"Local Estimator simulation failed: {exc}") from exc

    def submit(
        self,
        circuits: QuantumCircuit | Sequence[QuantumCircuit],
        *,
        config: RunConfig | None = None,
    ) -> Execution:
        values = self._circuits(circuits)
        return self._submit_workload("sampler", values, config or RunConfig())

    def submit_estimate(
        self,
        requests: EstimateRequest | Sequence[EstimateRequest],
        *,
        config: RunConfig | None = None,
    ) -> Execution:
        values = self._requests(requests)
        return self._submit_workload("estimator", values, config or RunConfig())

    def run(
        self,
        circuits: QuantumCircuit | Sequence[QuantumCircuit],
        *,
        config: RunConfig | None = None,
    ) -> Execution:
        execution = self.submit(circuits, config=config)
        execution.wait()
        return execution

    def estimate(
        self,
        requests: EstimateRequest | Sequence[EstimateRequest],
        *,
        config: RunConfig | None = None,
    ) -> Execution:
        execution = self.submit_estimate(requests, config=config)
        execution.wait()
        return execution

    def _submit_workload(
        self,
        primitive: Literal["sampler", "estimator"],
        workload: Sequence[QuantumCircuit] | Sequence[EstimateRequest],
        config: RunConfig,
        *,
        parent_execution_id: str | None = None,
        bypass_duplicate_check: bool = False,
        submission_indices: Sequence[int] | None = None,
        copied_jobs: Sequence[Mapping[str, Any]] = (),
    ) -> Execution:
        circuits = (
            list(workload)
            if primitive == "sampler"
            else [request.circuit for request in workload if isinstance(request, EstimateRequest)]
        )
        static_report = (
            validate_sampler_inputs(circuits, config)
            if primitive == "sampler"
            else validate_estimator_inputs(workload, config)
        )
        valid_circuits = (
            [item for item in circuits if isinstance(item, QuantumCircuit)]
            if static_report.is_valid
            else []
        )
        summaries = [circuit_summary(item, index) for index, item in enumerate(valid_circuits)]
        estimator_inputs: list[dict[str, Any]] = []
        if primitive == "estimator" and static_report.is_valid:
            estimator_inputs = [
                {
                    "observables": serialize_observable(normalize_observables(request.observables)),
                    "parameter_values": json_safe(request.parameter_values),
                }
                for request in workload
                if isinstance(request, EstimateRequest)
            ]
        fingerprint = stable_hash(
            {
                "primitive": primitive,
                "circuits": [record["fingerprint"] for record in summaries],
                "estimator_inputs": estimator_inputs,
                "shots": config.shots,
                "precision": config.precision,
                "backend": config.backend or self._provider.default_backend,
                "optimization_level": config.optimization_level,
                "seed_transpiler": config.seed_transpiler,
            }
        )
        execution_id = self._store.create_execution(
            primitive=primitive,
            config=config.persisted_dict(),
            fingerprint=fingerprint,
            metadata={
                "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
                "circuit_count": len(circuits),
            },
            qiskit_version=qiskit.__version__,
            runtime_version=_package_version("qiskit-ibm-runtime"),
            parent_execution_id=parent_execution_id,
            reject_duplicate=(
                static_report.is_valid
                and not config.allow_duplicate_submission
                and not bypass_duplicate_check
            ),
        )
        logger = execution_logger(LOGGER, execution_id)
        directory = self._store.execution_dir(execution_id)
        if valid_circuits:
            qpy_path = directory / "circuits.qpy"
            save_circuits(qpy_path, valid_circuits)
            self._store.add_circuits(execution_id, summaries, qpy_path)
            if primitive == "estimator":
                save_estimate_inputs(directory, cast(Sequence[EstimateRequest], workload))
        self._store.transition(execution_id, "VALIDATING")
        atomic_write_json(directory / "validation.json", static_report.to_dict())
        if not static_report.is_valid:
            self._store.transition(
                execution_id,
                "VALIDATION_FAILED",
                details={"errors": [item.to_dict() for item in static_report.errors]},
            )
            raise CircuitValidationError(
                "Circuit validation failed",
                report=static_report,
                execution_id=execution_id,
            )

        backend, transpiled = self._hardware_validation(
            static_report, valid_circuits, config, primitive
        )
        atomic_write_json(directory / "validation.json", static_report.to_dict())
        if not static_report.is_valid or backend is None:
            self._store.transition(
                execution_id,
                "VALIDATION_FAILED",
                details={"errors": [item.to_dict() for item in static_report.errors]},
            )
            raise CircuitValidationError(
                "Hardware preflight validation failed",
                report=static_report,
                execution_id=execution_id,
            )
        self._store.update_execution(
            execution_id, backend_name=self._safe_provider_message(backend_name(backend))
        )
        transpiled_path = directory / "transpiled.qpy"
        save_circuits(transpiled_path, transpiled)
        for index, metadata in enumerate(static_report.transpiled_metadata):
            self._store.update_transpiled(execution_id, index, metadata)

        if config.simulate_first:
            self._store.transition(execution_id, "SIMULATING")
            try:
                simulation = (
                    simulate_sampler(valid_circuits, config)
                    if primitive == "sampler"
                    else simulate_estimator(cast(Sequence[EstimateRequest], workload), config)
                )
                simulation_path = directory / "simulation.json"
                atomic_write_json(simulation_path, simulation.to_dict())
                self._store.update_execution(execution_id, simulation_path=str(simulation_path))
                validator_check = simulation.validator_check
                validator_failed = validator_check is not None and not validator_check.passed
                if validator_failed:
                    assert validator_check is not None
                    self._store.transition(
                        execution_id,
                        "SIMULATION_FAILED",
                        details=validator_check.to_dict(),
                    )
                    if config.stop_on_simulation_failure:
                        raise SimulationError(
                            "User simulation validation failed; hardware was not submitted"
                        )
                    self._store.transition(execution_id, "READY_FOR_HARDWARE", force=True)
                else:
                    self._store.transition(execution_id, "READY_FOR_HARDWARE")
            except SimulationError:
                raise
            except Exception as exc:
                self._store.transition(
                    execution_id,
                    "SIMULATION_FAILED",
                    details={"type": type(exc).__name__, "message": str(exc)},
                )
                if config.stop_on_simulation_failure:
                    raise SimulationError(
                        f"Local simulation failed; hardware was not submitted: {exc}"
                    ) from exc
                self._store.transition(execution_id, "READY_FOR_HARDWARE", force=True)
        else:
            self._store.transition(execution_id, "READY_FOR_HARDWARE")

        self._store.transition(execution_id, "SUBMITTING")
        for child_index, copied in enumerate(copied_jobs):
            source_path = Path(str(copied["result_path"]))
            copied_path = directory / f"job-{child_index:04d}-result.json"
            atomic_write_json(copied_path, load_json(source_path))
            self._store.upsert_job(
                execution_id,
                child_index,
                circuit_indices=copied["circuit_indices"],
                status="COMPLETED",
                ibm_job_id=copied.get("ibm_job_id"),
                raw_status=copied.get("raw_status") or "DONE",
                backend_name=self._safe_provider_message(
                    copied.get("backend_name") or backend_name(backend)
                ),
                submitted_at=copied.get("submitted_at"),
                result_path=str(copied_path),
                batch_id=copied.get("batch_id"),
            )
        logger.info(
            "Submitting IBM Runtime workload",
            extra={"backend": self._safe_provider_message(backend_name(backend))},
        )
        try:
            submitted = submit_workload(
                execution_id=execution_id,
                primitive=primitive,
                originals=workload,
                transpiled=transpiled,
                backend=backend,
                config=config,
                store=self._store,
                selected_indices=submission_indices,
                child_index_offset=len(copied_jobs),
                sanitize_message=self._safe_provider_message,
            )
        except HardwareSubmissionError as exc:
            safe_message = self._safe_provider_message(exc)
            failure = {"type": type(exc).__name__, "message": safe_message}
            self._store.update_execution(execution_id, failure_json=failure)
            self._store.transition(
                execution_id,
                "RECOVERY_REQUIRED",
                details=failure,
                force=True,
                only_from={
                    "SUBMITTING",
                    "PARTIALLY_SUBMITTED",
                    "SUBMITTED",
                    "QUEUED",
                    "RUNNING",
                    "RECOVERY_REQUIRED",
                },
            )
            raise HardwareSubmissionError(safe_message) from None
        self._raw_results[execution_id] = []
        self._store.transition(
            execution_id,
            "SUBMITTED",
            only_from={"SUBMITTING", "PARTIALLY_SUBMITTED", "RECOVERY_REQUIRED"},
            details={
                "ibm_job_ids": [
                    *[str(job["ibm_job_id"]) for job in copied_jobs if job.get("ibm_job_id")],
                    *[job.ibm_job_id for job in submitted],
                ]
            },
        )
        return Execution(self, execution_id)

    def resume(self, execution_id: str) -> Execution:
        try:
            self._store.get_execution(execution_id)
        except StateStoreError as exc:
            raise ExecutionRecoveryError(str(exc)) from exc
        return Execution(self, execution_id)

    def recover_ibm_job(self, ibm_job_id: str) -> Execution:
        try:
            return self._recover_ibm_job(ibm_job_id)
        except Exception as exc:
            raise ExecutionRecoveryError(self._safe_provider_message(exc)) from None

    def _recover_ibm_job(self, ibm_job_id: str) -> Execution:
        if self._safe_provider_message(ibm_job_id) != ibm_job_id:
            raise ExecutionRecoveryError("Unsafe provider job identifier")
        existing = self._store.find_by_ibm_job_id(ibm_job_id)
        if existing is not None:
            return Execution(self, str(existing["id"]))
        try:
            job = self._provider.retrieve_job(ibm_job_id, RunConfig().retry)
        except Exception as exc:
            raise ExecutionRecoveryError(
                f"Could not retrieve IBM job {ibm_job_id!r}: {self._safe_provider_message(exc)}"
            ) from None
        inputs = getattr(job, "inputs", {}) or {}
        pubs = inputs.get("pubs", []) if isinstance(inputs, Mapping) else []
        circuits: list[QuantumCircuit] = []
        for pub in pubs:
            circuit = getattr(pub, "circuit", None)
            if circuit is None and isinstance(pub, Sequence) and pub:
                circuit = pub[0]
            if isinstance(circuit, QuantumCircuit):
                circuits.append(circuit)
        if not circuits:
            raise ExecutionRecoveryError(
                "IBM returned the job, but its submitted circuits are unavailable; "
                "recover it from the original local execution ID instead"
            )
        program_id = str(getattr(job, "program_id", "sampler")).lower()
        primitive: Literal["sampler", "estimator"] = (
            "estimator" if "estimator" in program_id else "sampler"
        )
        if primitive == "estimator":
            raise ExecutionRecoveryError(
                "Direct recovery of an untracked Estimator job requires observable metadata; "
                "resume from its local execution ID"
            )
        config = RunConfig(simulate_first=False, allow_duplicate_submission=True)
        records = [circuit_summary(item, index) for index, item in enumerate(circuits)]
        fingerprint = stable_hash(
            {"primitive": primitive, "recovered_job": ibm_job_id, "circuits": records}
        )
        execution_id = self._store.create_execution(
            primitive=primitive,
            config=config.persisted_dict(),
            fingerprint=fingerprint,
            metadata={"recovered_from_ibm_job_id": ibm_job_id},
            qiskit_version=qiskit.__version__,
            runtime_version=_package_version("qiskit-ibm-runtime"),
        )
        directory = self._store.execution_dir(execution_id)
        path = directory / "circuits.qpy"
        save_circuits(path, circuits)
        self._store.add_circuits(execution_id, records, path)
        raw = self._safe_provider_message(raw_status_name(job.status()))
        self._store.upsert_job(
            execution_id,
            0,
            circuit_indices=tuple(range(len(circuits))),
            status=map_job_status(raw),
            ibm_job_id=ibm_job_id,
            raw_status=raw,
            backend_name=self._safe_provider_message(
                backend_name(
                    job.backend() if callable(getattr(job, "backend", None)) else job.backend
                )
            ),
            submitted_at=utc_now(),
        )
        recovered_backend = (
            job.backend() if callable(getattr(job, "backend", None)) else job.backend
        )
        self._store.update_execution(
            execution_id, backend_name=self._safe_provider_message(backend_name(recovered_backend))
        )
        self._store.transition(execution_id, "RECOVERY_REQUIRED")
        self._raw_results[execution_id] = [job]
        self._refresh_execution(execution_id)
        return Execution(self, execution_id)

    def resubmit(
        self,
        execution_id: str,
        *,
        confirm_hardware_resubmission: bool = False,
        config: RunConfig | None = None,
    ) -> Execution:
        if not confirm_hardware_resubmission:
            raise HardwareResubmissionRequiredError(
                "Set confirm_hardware_resubmission=True to create new IBM hardware jobs"
            )
        execution = self._store.get_execution(execution_id)
        if execution["status"] not in {
            "FAILED",
            "PARTIALLY_COMPLETED",
            "RECOVERY_REQUIRED",
            "CANCELLED",
        }:
            raise ExecutionRecoveryError(
                f"Execution {execution_id} is {execution['status']}; resubmission is not applicable"
            )
        directory = self._store.execution_dir(execution_id)
        circuits = load_circuits(directory / "circuits.qpy")
        restored = config or run_config_from_dict(execution["config"])
        if execution["config"].get("simulation_validator_name") and (
            restored.simulation_validator is None or not restored.simulate_first
        ):
            raise ExecutionRecoveryError(
                "The original execution used a simulation validator; supply a fresh "
                "RunConfig with that validator and simulate_first=True before resubmitting"
            )
        # Reconcile only known IDs; this is strictly read-only provider recovery.
        self._refresh_execution(execution_id)
        jobs = self._store.get_jobs(execution_id)
        coverage_error = self._coverage_error(execution_id, jobs)
        if coverage_error:
            raise ExecutionRecoveryError(coverage_error)
        unsafe = [
            job
            for job in jobs
            if (job["ibm_job_id"] and job["status"] not in TERMINAL_JOB_STATES)
            or (
                not job["ibm_job_id"]
                and job["status"] not in {"PLANNED", "NOT_SUBMITTED", "SUPERSEDED_NOT_SUBMITTED"}
            )
        ]
        if unsafe:
            raise ExecutionRecoveryError(
                "A child may already be running or its submission is ambiguous; "
                "reconcile the original provider job before resubmitting. No new job was sent"
            )
        copied_jobs = [job for job in jobs if job["status"] == "COMPLETED" and job["result_path"]]
        completed_indices = {int(index) for job in copied_jobs for index in job["circuit_indices"]}
        submission_indices = [
            index for index in range(len(circuits)) if index not in completed_indices
        ]
        if not submission_indices:
            raise ExecutionRecoveryError(
                f"Execution {execution_id} has no failed or incomplete circuits to resubmit"
            )
        if execution["primitive"] == "sampler":
            return self._submit_workload(
                "sampler",
                circuits,
                restored,
                parent_execution_id=execution_id,
                bypass_duplicate_check=True,
                submission_indices=submission_indices,
                copied_jobs=copied_jobs,
            )
        requests = load_estimate_inputs(directory, circuits)
        return self._submit_workload(
            "estimator",
            requests,
            restored,
            parent_execution_id=execution_id,
            bypass_duplicate_check=True,
            submission_indices=submission_indices,
            copied_jobs=copied_jobs,
        )

    def _refresh_execution(self, execution_id: str) -> None:
        execution = self._store.get_execution(execution_id)
        status = ExecutionStatus(execution["status"])
        if status.is_terminal:
            if status in {ExecutionStatus.COMPLETED, ExecutionStatus.PARTIALLY_COMPLETED}:
                jobs = self._store.get_jobs(execution_id)
                if self._coverage_error(execution_id, jobs):
                    self._set_aggregate_status(execution_id)
            return
        jobs = self._store.get_jobs(execution_id)
        if not jobs:
            if status in {
                ExecutionStatus.SUBMITTING,
                ExecutionStatus.PARTIALLY_SUBMITTED,
                ExecutionStatus.SUBMITTED,
                ExecutionStatus.RECOVERY_REQUIRED,
            }:
                self._set_aggregate_status(execution_id)
            return
        config = run_config_from_dict(execution["config"])
        circuit_records = self._store.get_circuits(execution_id)
        raw_results: list[Any] = []
        recovery_error: Exception | None = None
        for record in jobs:
            if not record["ibm_job_id"]:
                continue
            if record["status"] in {"FAILED", "CANCELLED"}:
                continue
            artifact_valid = self._child_result_valid(
                record, circuit_records, str(execution["primitive"])
            )
            if record["status"] == "COMPLETED" and artifact_valid:
                continue
            retry_history = list(record["retry_history"])

            on_retry = partial(self._record_retry, execution_id, record, retry_history)

            try:
                job = self._provider.retrieve_job(
                    str(record["ibm_job_id"]), config.retry, on_retry=on_retry
                )
                provider_status = retry_idempotent(
                    job.status,
                    config.retry,
                    description="poll job status",
                    on_retry=on_retry,
                )
                raw = self._safe_provider_message(raw_status_name(provider_status))
                normalized = map_job_status(provider_status)
                result_path = record["result_path"] if artifact_valid else None
                if normalized == "COMPLETED" and not result_path:
                    raw_result = retry_idempotent(
                        job.result,
                        config.retry,
                        description="retrieve job result",
                        on_retry=on_retry,
                    )
                    raw_results.append(raw_result)
                    indices = [int(index) for index in record["circuit_indices"]]
                    selected_records = [circuit_records[index] for index in indices]
                    normalized_results: tuple[CircuitResult, ...]
                    if execution["primitive"] == "sampler":
                        normalized_results = normalize_sampler_result(raw_result, selected_records)
                    else:
                        normalized_results = normalize_estimator_result(
                            raw_result,
                            selected_records,
                            precision=config.estimator_precision(),
                        )
                    result_file = (
                        self._store.execution_dir(execution_id)
                        / f"job-{int(record['child_index']):04d}-result.json"
                    )
                    atomic_write_json(
                        result_file,
                        {
                            "schema_version": ARTIFACT_SCHEMA_VERSION,
                            "ibm_job_id": record["ibm_job_id"],
                            "circuit_results": self._safe_provider_data(
                                [item.to_dict() for item in normalized_results]
                            ),
                            "provider_metadata": self._safe_provider_data(
                                getattr(raw_result, "metadata", {})
                            ),
                        },
                    )
                    result_path = str(result_file)
                failure = None
                if normalized == "FAILED":
                    error_message = getattr(job, "error_message", None)
                    failure = {
                        "type": "IBMJobFailure",
                        "message": self._safe_provider_message(
                            error_message() if callable(error_message) else error_message
                        ),
                    }
                self._store.upsert_job(
                    execution_id,
                    int(record["child_index"]),
                    circuit_indices=record["circuit_indices"],
                    status=normalized,
                    ibm_job_id=str(record["ibm_job_id"]),
                    raw_status=raw,
                    backend_name=record["backend_name"],
                    submitted_at=record["submitted_at"],
                    result_path=result_path,
                    failure=failure,
                    retry_history=retry_history,
                    batch_id=record["batch_id"],
                )
            except Exception as exc:
                recovery_error = exc
                safe_message = self._safe_provider_message(exc)
                self._store.upsert_job(
                    execution_id,
                    int(record["child_index"]),
                    circuit_indices=record["circuit_indices"],
                    status="RECOVERY_REQUIRED",
                    ibm_job_id=str(record["ibm_job_id"]),
                    raw_status=record["raw_status"],
                    backend_name=record["backend_name"],
                    submitted_at=record["submitted_at"],
                    failure={"type": type(exc).__name__, "message": safe_message},
                    retry_history=retry_history,
                    batch_id=record["batch_id"],
                )
        if raw_results:
            self._raw_results.setdefault(execution_id, []).extend(raw_results)
        self._set_aggregate_status(execution_id, recovery_error)

    def _record_retry(
        self,
        execution_id: str,
        record: Mapping[str, Any],
        retry_history: list[dict[str, Any]],
        details: Mapping[str, Any],
    ) -> None:
        retry_history.append(dict(details))
        self._store.upsert_job(
            execution_id,
            int(record["child_index"]),
            circuit_indices=record["circuit_indices"],
            status=str(record["status"]),
            ibm_job_id=str(record["ibm_job_id"]),
            raw_status=record["raw_status"],
            backend_name=record["backend_name"],
            submitted_at=record["submitted_at"],
            retry_history=retry_history,
            batch_id=record["batch_id"],
        )

    @staticmethod
    def _result_payload_valid(item: Mapping[str, Any], primitive: str) -> bool:
        if primitive == "estimator":
            if item.get("expectation_values") is None:
                return False
            try:
                values = np.asarray(item["expectation_values"], dtype=np.float64)
                if not values.size or not np.isfinite(values).all():
                    return False
                stds = item.get("standard_deviations")
                return stds is None or bool(
                    np.asarray(stds).shape == values.shape
                    and np.isfinite(np.asarray(stds, dtype=np.float64)).all()
                )
            except (TypeError, ValueError):
                return False
        shots = item.get("shots")
        counts = item.get("counts")
        registers = item.get("register_counts")
        if (
            isinstance(shots, bool)
            or not isinstance(shots, int)
            or shots < 0
            or not isinstance(counts, Mapping)
            or not isinstance(registers, Mapping)
        ):
            return False
        for group in [counts, *registers.values()]:
            if not isinstance(group, Mapping) or any(
                not isinstance(key, str)
                or isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
                for key, value in group.items()
            ):
                return False
        return True

    @staticmethod
    def _child_result_valid(
        job: Mapping[str, Any], circuits: Sequence[Mapping[str, Any]], primitive: str
    ) -> bool:
        if not job["result_path"]:
            return False
        try:
            payload = load_json(Path(str(job["result_path"])))
            if payload.get("schema_version") != ARTIFACT_SCHEMA_VERSION or (
                job["ibm_job_id"] and payload.get("ibm_job_id") != job["ibm_job_id"]
            ):
                return False
            results = payload["circuit_results"]
            if not isinstance(results, list) or not all(
                isinstance(item, Mapping) for item in results
            ):
                return False
            actual = [int(item["circuit_index"]) for item in results]
            return sorted(actual) == sorted(job["circuit_indices"]) and all(
                item.get("status", "completed") == "completed"
                and item.get("kind") == primitive
                and item.get("circuit_id") == circuits[int(item["circuit_index"])]["fingerprint"]
                and QuantumRunner._result_payload_valid(item, primitive)
                for item in results
            )
        except (OSError, ValueError, TypeError, KeyError, IndexError):
            return False

    def _coverage_error(self, execution_id: str, jobs: Sequence[Mapping[str, Any]]) -> str | None:
        """Require exact planned and successful-artifact coverage, including legacy runs."""
        execution = self._store.get_execution(execution_id)
        circuits = self._store.get_circuits(execution_id)
        expected_count = int(execution["metadata"].get("circuit_count", len(circuits)))
        expected = list(range(expected_count))
        planned = [int(index) for job in jobs for index in job["circuit_indices"]]
        if (
            not expected
            or sorted(int(record["circuit_index"]) for record in circuits) != expected
            or sorted(planned) != expected
            or len({job["child_index"] for job in jobs}) != len(jobs)
        ):
            return "Child jobs do not cover the complete original workload exactly once"
        for job in jobs:
            if job["status"] == "COMPLETED" and not self._child_result_valid(
                job, circuits, str(execution["primitive"])
            ):
                return (
                    "A completed child result artifact is missing, invalid, or has inexact coverage"
                )
        return None

    def _set_aggregate_status(
        self, execution_id: str, recovery_error: Exception | None = None
    ) -> ExecutionStatus:
        jobs = self._store.get_jobs(execution_id)
        states = [str(job["status"]) for job in jobs]
        coverage_error = self._coverage_error(execution_id, jobs)
        if coverage_error or any(
            state not in {"SUBMITTED", "QUEUED", "RUNNING", "COMPLETED", "FAILED", "CANCELLED"}
            for state in states
        ):
            aggregate = "RECOVERY_REQUIRED"
        elif any(state == "RUNNING" for state in states):
            aggregate = "RUNNING"
        elif any(state == "QUEUED" for state in states):
            aggregate = "QUEUED"
        elif any(state == "SUBMITTED" for state in states):
            aggregate = "SUBMITTED"
        elif states and all(state == "COMPLETED" for state in states):
            aggregate = "COMPLETED"
        elif states and all(state == "CANCELLED" for state in states):
            aggregate = "CANCELLED"
        elif any(state == "COMPLETED" for state in states):
            aggregate = "PARTIALLY_COMPLETED"
        else:
            aggregate = "FAILED"
        details: dict[str, Any] = {"child_statuses": states}
        if coverage_error:
            details["coverage_error"] = coverage_error
        if recovery_error is not None:
            details["recovery_error"] = {
                "type": type(recovery_error).__name__,
                "message": self._safe_provider_message(recovery_error),
            }
        self._store.transition(execution_id, aggregate, details=details, force=True)
        return ExecutionStatus(aggregate)

    def _wait(
        self,
        execution_id: str,
        *,
        timeout: float | None,
        poll_interval: float | None,
    ) -> None:
        execution = self._store.get_execution(execution_id)
        config = run_config_from_dict(execution["config"])
        effective_timeout = timeout if timeout is not None else config.wait_timeout
        interval = poll_interval if poll_interval is not None else config.poll_interval
        if interval <= 0:
            raise ValueError("poll_interval must be positive")
        started = time.monotonic()
        while True:
            self._refresh_execution(execution_id)
            status = ExecutionStatus(self._store.get_execution(execution_id)["status"])
            if status.is_terminal:
                return
            if status == ExecutionStatus.RECOVERY_REQUIRED:
                raise ExecutionRecoveryError(
                    f"Execution {execution_id} requires recovery; no hardware was resubmitted"
                )
            if effective_timeout is not None and time.monotonic() - started >= effective_timeout:
                raise ExecutionTimeoutError(
                    f"Execution {execution_id} did not finish within {effective_timeout} seconds"
                )
            time.sleep(interval)

    def _result(self, execution_id: str, *, allow_partial: bool) -> ExecutionResult:
        jobs = self._store.get_jobs(execution_id)
        coverage_error = self._coverage_error(execution_id, jobs)
        if coverage_error:
            self._set_aggregate_status(execution_id)
            raise ResultRetrievalError(coverage_error)
        execution = self._store.get_execution(execution_id)
        status = ExecutionStatus(execution["status"])
        if status == ExecutionStatus.PARTIALLY_COMPLETED and not allow_partial:
            raise ResultRetrievalError(
                "Execution completed only partially; pass allow_partial=True "
                "to inspect durable results"
            )
        if status not in {ExecutionStatus.COMPLETED, ExecutionStatus.PARTIALLY_COMPLETED}:
            raise ResultRetrievalError(
                f"Execution {execution_id} has no successful final result (status {status})"
            )
        jobs = self._store.get_jobs(execution_id)
        circuit_results: list[CircuitResult] = []
        for job in jobs:
            if not job["result_path"]:
                continue
            payload = load_json(Path(job["result_path"]))
            circuit_results.extend(
                circuit_result_from_dict(item) for item in payload["circuit_results"]
            )
        if status == ExecutionStatus.PARTIALLY_COMPLETED:
            present = {item.circuit_index for item in circuit_results}
            config = run_config_from_dict(execution["config"])
            for record in self._store.get_circuits(execution_id):
                circuit_index = int(record["circuit_index"])
                if circuit_index in present:
                    continue
                child = next(
                    (job for job in jobs if circuit_index in job["circuit_indices"]),
                    None,
                )
                child_status = str(child["status"]).lower() if child else "failed"
                failure = child.get("failure") if child else None
                error = (
                    str(failure.get("message"))
                    if isinstance(failure, Mapping) and failure.get("message")
                    else "No successful result was available for this circuit"
                )
                if execution["primitive"] == "sampler":
                    circuit_results.append(
                        SamplerCircuitResult(
                            circuit_index=circuit_index,
                            circuit_id=str(record["fingerprint"]),
                            circuit_name=str(record["name"]),
                            counts={},
                            register_counts={},
                            shots=config.shots,
                            status=child_status,
                            error=error,
                        )
                    )
                else:
                    circuit_results.append(
                        EstimatorCircuitResult(
                            circuit_index=circuit_index,
                            circuit_id=str(record["fingerprint"]),
                            circuit_name=str(record["name"]),
                            expectation_values=None,
                            standard_deviations=None,
                            precision=config.estimator_precision(),
                            status=child_status,
                            error=error,
                        )
                    )
        circuit_results.sort(key=lambda item: item.circuit_index)
        simulation = None
        if execution["simulation_path"]:
            simulation = simulation_result_from_dict(load_json(Path(execution["simulation_path"])))
        identifiers = tuple(str(job["ibm_job_id"]) for job in jobs if job["ibm_job_id"])
        hardware = HardwareResult(
            backend_name=execution["backend_name"],
            ibm_job_ids=identifiers,
            circuit_results=tuple(circuit_results),
            timing={
                "created_at": execution["created_at"],
                "updated_at": execution["updated_at"],
                "submitted_at": [job["submitted_at"] for job in jobs],
            },
            metadata={"child_jobs": len(jobs)},
        )
        result = ExecutionResult(
            execution_id=execution_id,
            primitive=execution["primitive"],
            final_status=status.value,
            backend_name=execution["backend_name"],
            ibm_job_ids=identifiers,
            hardware=hardware,
            simulation=simulation,
            submission_metadata=execution["metadata"],
            raw_provider_result=self._raw_results.get(execution_id),
        )
        result_path = self._store.execution_dir(execution_id) / "result.json"
        atomic_write_json(result_path, result.to_dict())
        self._store.update_execution(execution_id, result_path=str(result_path))
        return result

    def _cancel(self, execution_id: str) -> ExecutionStatus:
        execution = self._store.get_execution(execution_id)
        config = run_config_from_dict(execution["config"])
        for record in self._store.get_jobs(execution_id):
            if not record["ibm_job_id"] or record["status"] in TERMINAL_JOB_STATES:
                continue
            try:
                job = self._provider.retrieve_job(str(record["ibm_job_id"]), config.retry)
                job.cancel()
            except Exception as exc:
                raise ExecutionRecoveryError(
                    f"Could not cancel IBM job {record['ibm_job_id']}: "
                    f"{self._safe_provider_message(exc)}"
                ) from None
        self._refresh_execution(execution_id)
        return ExecutionStatus(self._store.get_execution(execution_id)["status"])


__all__ = ["QuantumRunner"]
