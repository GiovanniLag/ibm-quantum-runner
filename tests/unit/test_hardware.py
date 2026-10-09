from __future__ import annotations

import pytest
from qiskit import QuantumCircuit

from ibm_quantum_runner import BackendSelectionError, RetryConfig, RunConfig
from ibm_quantum_runner.hardware import (
    IBMProvider,
    map_job_status,
    partition_indices,
    retry_idempotent,
    submit_workload,
)
from ibm_quantum_runner.persistence import StateStore


def test_sampler_partition_respects_execution_limit() -> None:
    config = RunConfig(shots=4_000_000, max_pubs_per_job=100)
    assert partition_indices(5, "sampler", config) == [(0, 1), (2, 3), (4,)]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("INITIALIZING", "SUBMITTED"),
        ("QUEUED", "QUEUED"),
        ("RUNNING", "RUNNING"),
        ("DONE", "COMPLETED"),
        ("ERROR", "FAILED"),
        ("CANCELLED", "CANCELLED"),
    ],
)
def test_job_status_mapping(raw: str, expected: str) -> None:
    assert map_job_status(raw) == expected


def test_retry_is_bounded_for_transient_reads(monkeypatch) -> None:
    attempts = 0
    history = []

    def operation() -> str:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise ConnectionError("temporary")
        return "ok"

    monkeypatch.setattr("ibm_quantum_runner.hardware.ibm.time.sleep", lambda _: None)
    value = retry_idempotent(
        operation,
        RetryConfig(max_attempts=3, initial_delay=0, max_delay=0, jitter=0),
        description="test read",
        on_retry=history.append,
    )
    assert value == "ok"
    assert attempts == 3
    assert len(history) == 2


def test_permanent_error_is_not_retried() -> None:
    attempts = 0

    def operation() -> None:
        nonlocal attempts
        attempts += 1
        raise ValueError("invalid credentials")

    with pytest.raises(ValueError, match="credentials"):
        retry_idempotent(operation, RetryConfig(), description="test")
    assert attempts == 1


def test_provider_errors_redact_token() -> None:
    secret = "credential-that-must-not-escape"

    class Service:
        def backend(self, name: str) -> None:
            raise RuntimeError(f"request failed with {secret}")

    provider = IBMProvider(
        token=secret,
        channel="ibm_quantum_platform",
        instance=None,
        default_backend=None,
        service=Service(),
    )
    with pytest.raises(BackendSelectionError) as captured:
        provider.get_backend("backend")
    assert secret not in str(captured.value)
    assert "[REDACTED]" in str(captured.value)
    assert captured.value.__cause__ is None


def test_submission_persists_job_id_before_any_status_read(tmp_path, monkeypatch) -> None:
    class Job:
        def job_id(self) -> str:
            return "durable-job-id"

        def status(self) -> str:
            raise AssertionError("submission must not poll before persistence")

    class Sampler:
        def __init__(self, **kwargs: object) -> None:
            pass

        def run(self, pubs: object, *, shots: int) -> Job:
            return Job()

    monkeypatch.setattr("qiskit_ibm_runtime.SamplerV2", Sampler)
    store = StateStore(tmp_path)
    execution_id = store.create_execution(
        primitive="sampler",
        config={},
        fingerprint="fingerprint",
        metadata={},
        qiskit_version="test",
        runtime_version="test",
    )
    circuit = QuantumCircuit(1, 1)
    circuit.measure(0, 0)

    submitted = submit_workload(
        execution_id=execution_id,
        primitive="sampler",
        originals=[circuit],
        transpiled=[circuit],
        backend=type("Backend", (), {"name": "fake"})(),
        config=RunConfig(simulate_first=False),
        store=store,
    )

    assert submitted[0].ibm_job_id == "durable-job-id"
    assert store.get_jobs(execution_id)[0]["ibm_job_id"] == "durable-job-id"
