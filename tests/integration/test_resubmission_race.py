from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest
from qiskit import QuantumCircuit

from ibm_quantum_runner import ExecutionStatus, HardwareSubmissionError, QuantumRunner, RunConfig
from tests.fixtures import FakeJob, FakeProvider


def test_resubmission_fences_original_unsubmitted_child(tmp_path, monkeypatch) -> None:
    """A refreshed handle must not let two submit loops send the same circuit."""
    original_planned = Event()
    release_original = Event()
    submitted: list[int] = []

    class PausedSampler:
        instances = 0

        def __init__(self, **kwargs: object) -> None:
            type(self).instances += 1
            self.instance = type(self).instances
            if self.instance == 1:
                # The complete plan is durable, but primitive.run has not started.
                original_planned.set()
                assert release_original.wait(timeout=10)

        def run(self, pubs: object, **kwargs: object) -> FakeJob:
            submitted.append(self.instance)
            return FakeJob(f"fake-race-{self.instance}", ["QUEUED"])

    monkeypatch.setattr("qiskit_ibm_runtime.SamplerV2", PausedSampler)
    runner = QuantumRunner(state_dir=tmp_path, provider=FakeProvider())
    circuit = QuantumCircuit(1, 1)
    circuit.measure(0, 0)
    config = RunConfig(simulate_first=False)

    with ThreadPoolExecutor(max_workers=1) as pool:
        original_future = pool.submit(runner.submit, circuit, config=config)
        try:
            assert original_planned.wait(timeout=10)
            with runner._store._connect() as connection:
                original_id = connection.execute("SELECT id FROM executions").fetchone()["id"]
            original_handle = runner.resume(original_id)
            assert original_handle.status() == ExecutionStatus.RECOVERY_REQUIRED

            replacement = runner.resubmit(original_id, confirm_hardware_resubmission=True)
            assert replacement.ibm_job_ids == ("fake-race-2",)
        finally:
            release_original.set()
        with pytest.raises(HardwareSubmissionError):
            original_future.result(timeout=10)
        original = runner.resume(original_id)

    assert submitted == [2]
    assert original.ibm_job_ids == ()
    assert replacement.metadata["parent_execution_id"] == original.id


@pytest.mark.parametrize("failure_stage", ["hardware_validation", "simulation"])
def test_failed_preflight_replacement_does_not_strand_lineage(
    tmp_path, monkeypatch, failure_stage
) -> None:
    from ibm_quantum_runner import BackendSelectionError, CircuitValidationError, SimulationError
    from tests.fixtures import fake_submitter

    jobs = {"fake-job-0": FakeJob("fake-job-0", ["ERROR"])}
    provider = FakeProvider(jobs)
    runner = QuantumRunner(state_dir=tmp_path, provider=provider)
    monkeypatch.setattr("ibm_quantum_runner.runner.submit_workload", fake_submitter(jobs))
    circuit = QuantumCircuit(1, 1)
    circuit.measure(0, 0)
    config = RunConfig(simulate_first=False, shots=4)
    original = runner.submit(circuit, config=config)
    original.wait()
    assert original.status(refresh=False) == ExecutionStatus.FAILED

    if failure_stage == "hardware_validation":

        def unavailable_backend(*args, **kwargs):
            raise BackendSelectionError("Fake backend temporarily unavailable")

        with monkeypatch.context() as temporary_patch:
            temporary_patch.setattr(provider, "select_backend", unavailable_backend)
            with pytest.raises(CircuitValidationError):
                runner.resubmit(original.id, confirm_hardware_resubmission=True, config=config)
    else:
        rejecting_config = RunConfig(
            simulate_first=True,
            shots=4,
            simulation_validator=lambda result: False,
            stop_on_simulation_failure=True,
        )
        with pytest.raises(SimulationError):
            runner.resubmit(
                original.id, confirm_hardware_resubmission=True, config=rejecting_config
            )

    with runner._store._connect() as connection:
        failed_replacement = connection.execute(
            "SELECT id, status FROM executions WHERE parent_execution_id = ?", (original.id,)
        ).fetchone()
    assert failed_replacement["status"] in {"VALIDATION_FAILED", "SIMULATION_FAILED"}
    assert runner._store.get_jobs(failed_replacement["id"]) == []

    # A fresh, explicitly confirmed request can recover once the preflight issue is fixed.
    replacement = runner.resubmit(original.id, confirm_hardware_resubmission=True, config=config)
    assert replacement.metadata["parent_execution_id"] == original.id
    assert replacement.id != failed_replacement["id"]
    assert replacement.ibm_job_ids == ("fake-job-0",)


@pytest.mark.parametrize(
    "corruption", ["missing", "invalid_json", "null_root", "list_root", "invalid_item"]
)
def test_completed_artifact_is_rebuilt_from_known_job_without_submission(
    tmp_path, monkeypatch, corruption
) -> None:
    from pathlib import Path

    from tests.integration.test_submission_safety import (
        OfflineProvider,
        RuntimeBoundary,
        circuit,
        config,
    )

    provider = OfflineProvider()
    boundary = RuntimeBoundary(monkeypatch, provider)
    runner = QuantumRunner(state_dir=tmp_path, provider=provider)
    execution = runner.submit(circuit("recover-artifact"), config=config())
    execution.wait()
    assert execution.status(refresh=False) == ExecutionStatus.COMPLETED
    artifact = Path(runner._store.get_jobs(execution.id)[0]["result_path"])
    if corruption == "missing":
        artifact.unlink()
    elif corruption == "null_root":
        artifact.write_text("null", encoding="utf-8")
    elif corruption == "list_root":
        artifact.write_text("[]", encoding="utf-8")
    elif corruption == "invalid_item":
        import json

        payload = json.loads(artifact.read_text())
        payload["circuit_results"] = [None]
        artifact.write_text(json.dumps(payload), encoding="utf-8")
    else:
        artifact.write_text('{"circuit_results":', encoding="utf-8")
    prior_reads = provider.retrieve_calls
    prior_submissions = boundary.run_calls

    resumed = QuantumRunner(state_dir=tmp_path, provider=provider).resume(execution.id)
    assert resumed.status() == ExecutionStatus.RECOVERY_REQUIRED
    assert resumed.status() == ExecutionStatus.COMPLETED
    assert resumed.result().counts == {"0": 5}
    assert artifact.exists()
    assert provider.retrieve_calls > prior_reads
    assert boundary.run_calls == prior_submissions == 1


@pytest.mark.parametrize("finalization_fails", [False, True])
def test_batch_finalization_never_regresses_concurrently_completed_execution(
    tmp_path, monkeypatch, finalization_fails
) -> None:
    from tests.integration.test_submission_safety import (
        OfflineProvider,
        RuntimeBoundary,
        circuit,
        config,
        only_execution_id,
    )

    provider = OfflineProvider()
    boundary = RuntimeBoundary(monkeypatch, provider)
    finalizing = Event()
    release_finalization = Event()

    def pause_finalization(stage: str, child: int | None) -> None:
        if stage == "batch_exit":
            finalizing.set()
            assert release_finalization.wait(timeout=10)
            if finalization_fails:
                raise RuntimeError("Fake batch cleanup failed after jobs completed")

    boundary.hook = pause_finalization
    runner = QuantumRunner(state_dir=tmp_path, provider=provider)
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(runner.submit, [circuit("first"), circuit("second")], config=config())
        try:
            assert finalizing.wait(timeout=10)
            execution_id = only_execution_id(runner._store)
            observer = QuantumRunner(state_dir=tmp_path, provider=provider).resume(execution_id)
            assert observer.status() == ExecutionStatus.COMPLETED
        finally:
            release_finalization.set()
        if finalization_fails:
            with pytest.raises(HardwareSubmissionError):
                pending.result(timeout=10)
        else:
            submitted = pending.result(timeout=10)
            assert submitted.id == execution_id

    assert observer.status(refresh=False) == ExecutionStatus.COMPLETED
    assert observer.result().final_status == "COMPLETED"
    assert boundary.run_calls == 2


@pytest.mark.parametrize("primitive", ["sampler", "estimator"])
def test_missing_required_result_payload_is_retrieved_again(
    tmp_path, monkeypatch, primitive
) -> None:
    import json
    from pathlib import Path

    from ibm_quantum_runner import EstimateRequest
    from tests.integration.test_submission_safety import (
        OfflineProvider,
        RuntimeBoundary,
        circuit,
        config,
    )

    provider = OfflineProvider()
    boundary = RuntimeBoundary(monkeypatch, provider)
    runner = QuantumRunner(state_dir=tmp_path, provider=provider)
    if primitive == "sampler":
        execution = runner.submit(circuit("missing-counts"), config=config())
        payload_key = "counts"
    else:
        execution = runner.submit_estimate(EstimateRequest(QuantumCircuit(1), "Z"), config=config())
        payload_key = "expectation_values"
    execution.wait()
    artifact = Path(runner._store.get_jobs(execution.id)[0]["result_path"])
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    del payload["circuit_results"][0][payload_key]
    artifact.write_text(json.dumps(payload), encoding="utf-8")
    prior_reads = provider.retrieve_calls

    assert execution.status() == ExecutionStatus.RECOVERY_REQUIRED
    assert execution.status() == ExecutionStatus.COMPLETED
    restored = execution.result()
    if primitive == "sampler":
        assert restored.counts == {"0": 5}
    else:
        assert restored.expectation_values == 1.0
    assert provider.retrieve_calls > prior_reads
    assert boundary.run_calls == 1
