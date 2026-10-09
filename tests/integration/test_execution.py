from __future__ import annotations

from pathlib import Path

import pytest
from qiskit import QuantumCircuit

from ibm_quantum_runner import (
    ExecutionStatus,
    HardwareResubmissionRequiredError,
    QuantumRunner,
    ResultRetrievalError,
    RunConfig,
)
from tests.fixtures import (
    FakeJob,
    FakePrimitiveResult,
    FakeProvider,
    FakeSamplerPub,
    fake_submitter,
)


def measured_circuit(name: str = "measured") -> QuantumCircuit:
    circuit = QuantumCircuit(1, 1, name=name)
    circuit.measure(0, 0)
    return circuit


def test_submit_resume_and_result_never_resubmits(tmp_path, monkeypatch) -> None:
    raw = FakePrimitiveResult([FakeSamplerPub({"0": 7, "1": 3})])
    job = FakeJob("fake-job-0", ["QUEUED", "DONE"], raw)
    jobs = {"fake-job-0": job}
    monkeypatch.setattr("ibm_quantum_runner.runner.submit_workload", fake_submitter(jobs))
    config = RunConfig(simulate_first=False, shots=10, poll_interval=0.001)
    first_runner = QuantumRunner(state_dir=tmp_path, provider=FakeProvider(jobs))
    execution = first_runner.submit(measured_circuit(), config=config)
    assert execution.ibm_job_id == "fake-job-0"
    assert execution.status(refresh=False) == ExecutionStatus.SUBMITTED

    second_provider = FakeProvider(jobs)
    resumed = QuantumRunner(state_dir=tmp_path, provider=second_provider).resume(execution.id)
    resumed.wait(poll_interval=0.001)
    result = resumed.result()

    assert result.counts == {"0": 7, "1": 3}
    assert second_provider.retrieve_calls >= 1
    assert len(resumed.ibm_job_ids) == 1
    assert Path(resumed.metadata["result_path"]).exists()


def test_partial_child_result_is_preserved(tmp_path, monkeypatch) -> None:
    jobs = {
        "fake-job-0": FakeJob(
            "fake-job-0",
            ["DONE"],
            FakePrimitiveResult([FakeSamplerPub({"0": 5})]),
        ),
        "fake-job-1": FakeJob("fake-job-1", ["ERROR"]),
    }
    monkeypatch.setattr("ibm_quantum_runner.runner.submit_workload", fake_submitter(jobs))
    runner = QuantumRunner(state_dir=tmp_path, provider=FakeProvider(jobs))
    execution = runner.submit(
        [measured_circuit("a"), measured_circuit("b")],
        config=RunConfig(
            simulate_first=False,
            shots=5,
            max_pubs_per_job=1,
            poll_interval=0.001,
        ),
    )
    execution.wait()
    assert execution.status(refresh=False) == ExecutionStatus.PARTIALLY_COMPLETED
    with pytest.raises(ResultRetrievalError, match="partially"):
        execution.result()
    result = execution.result(allow_partial=True)
    assert len(result.circuit_results) == 2
    assert result.circuit_results[0].circuit_index == 0
    assert result.circuit_results[0].status == "completed"
    assert result.circuit_results[1].status == "failed"
    assert result.circuit_results[1].error == "fake failure"


def test_explicit_resubmit_copies_success_and_submits_only_failed(tmp_path, monkeypatch) -> None:
    jobs = {
        "fake-job-0": FakeJob(
            "fake-job-0",
            ["DONE"],
            FakePrimitiveResult([FakeSamplerPub({"0": 5})]),
        ),
        "fake-job-1": FakeJob("fake-job-1", ["ERROR"]),
    }
    monkeypatch.setattr("ibm_quantum_runner.runner.submit_workload", fake_submitter(jobs))
    runner = QuantumRunner(state_dir=tmp_path, provider=FakeProvider(jobs))
    config = RunConfig(
        simulate_first=False,
        shots=5,
        max_pubs_per_job=1,
        poll_interval=0.001,
    )
    original = runner.submit([measured_circuit("a"), measured_circuit("b")], config=config)
    original.wait()
    jobs["fake-job-1"] = FakeJob(
        "fake-job-1",
        ["DONE"],
        FakePrimitiveResult([FakeSamplerPub({"1": 5})]),
    )

    replacement = runner.resubmit(
        original.id,
        confirm_hardware_resubmission=True,
        config=config,
    )
    replacement.wait()
    result = replacement.result()

    assert result.final_status == "COMPLETED"
    assert [item.circuit_index for item in result.circuit_results] == [0, 1]
    assert replacement.metadata["parent_execution_id"] == original.id


def test_resubmission_requires_explicit_confirmation(tmp_path) -> None:
    runner = QuantumRunner(state_dir=tmp_path, provider=FakeProvider())
    with pytest.raises(HardwareResubmissionRequiredError):
        runner.resubmit("anything")


def test_cancel_maps_provider_terminal_state(tmp_path, monkeypatch) -> None:
    job = FakeJob("fake-job-0", ["QUEUED"])
    jobs = {"fake-job-0": job}
    monkeypatch.setattr("ibm_quantum_runner.runner.submit_workload", fake_submitter(jobs))
    runner = QuantumRunner(state_dir=tmp_path, provider=FakeProvider(jobs))
    execution = runner.submit(measured_circuit(), config=RunConfig(simulate_first=False, shots=1))

    assert execution.cancel() == ExecutionStatus.CANCELLED
    assert job.cancel_calls == 1


def test_recover_untracked_sampler_job_from_ibm_id(tmp_path) -> None:
    circuit = measured_circuit("recovered")
    job = FakeJob(
        "external-job",
        ["DONE"],
        FakePrimitiveResult([FakeSamplerPub({"0": 9, "1": 1})]),
        inputs={"pubs": [(circuit,)]},
    )
    runner = QuantumRunner(state_dir=tmp_path, provider=FakeProvider({"external-job": job}))

    execution = runner.recover_ibm_job("external-job")
    result = execution.result()

    assert execution.ibm_job_id == "external-job"
    assert result.counts == {"0": 9, "1": 1}
    assert execution.metadata["metadata"]["recovered_from_ibm_job_id"] == "external-job"


def test_secret_token_is_absent_from_state(tmp_path, monkeypatch) -> None:
    secret = "super-secret-token-not-to-persist"
    job = FakeJob(
        "fake-job-0",
        ["QUEUED"],
        FakePrimitiveResult([FakeSamplerPub({"0": 1})]),
    )
    jobs = {"fake-job-0": job}
    monkeypatch.setattr("ibm_quantum_runner.runner.submit_workload", fake_submitter(jobs))
    runner = QuantumRunner(
        token=secret,
        state_dir=tmp_path,
        provider=FakeProvider(jobs),
    )
    runner.submit(measured_circuit(), config=RunConfig(simulate_first=False, shots=1))
    for path in tmp_path.rglob("*"):
        if path.is_file():
            assert secret.encode() not in path.read_bytes()
