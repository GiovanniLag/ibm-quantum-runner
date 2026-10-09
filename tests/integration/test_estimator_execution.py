from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from qiskit import QuantumCircuit

from ibm_quantum_runner import EstimateRequest, QuantumRunner, RunConfig
from tests.fixtures import FakeJob, FakePrimitiveResult, FakeProvider, fake_submitter


def test_estimator_submission_persistence_and_normalization(tmp_path, monkeypatch) -> None:
    circuit = QuantumCircuit(2, name="bell-estimator")
    circuit.h(0)
    circuit.cx(0, 1)
    pub = SimpleNamespace(
        data=SimpleNamespace(evs=np.asarray([0.98]), stds=np.asarray([0.01])),
        metadata={"target_precision": 0.02},
    )
    jobs = {"fake-job-0": FakeJob("fake-job-0", ["DONE"], FakePrimitiveResult([pub]))}
    monkeypatch.setattr("ibm_quantum_runner.runner.submit_workload", fake_submitter(jobs))
    runner = QuantumRunner(state_dir=tmp_path, provider=FakeProvider(jobs))

    execution = runner.submit_estimate(
        EstimateRequest(circuit, {"ZZ": 1.0}),
        config=RunConfig(simulate_first=False, precision=0.02),
    )
    execution.wait(poll_interval=0.001)
    result = execution.result()

    assert result.expectation_values == [0.98]
    assert result.circuit_results[0].standard_deviations == [0.01]
    execution_dir = tmp_path / "executions" / execution.id
    assert (execution_dir / "estimator_inputs.json").exists()
    assert (execution_dir / "circuits.qpy").exists()
