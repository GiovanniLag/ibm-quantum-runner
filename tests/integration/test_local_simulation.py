from __future__ import annotations

from qiskit import QuantumCircuit

from ibm_quantum_runner import EstimateRequest, QuantumRunner, RunConfig


def test_aer_sampler_bell_counts(tmp_path) -> None:
    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.measure_all()
    result = QuantumRunner(state_dir=tmp_path).simulate(
        circuit, config=RunConfig(shots=1_000, simulation_seed=7)
    )
    assert result.counts is not None
    assert sum(result.counts.values()) == 1_000
    assert set(result.counts) <= {"00", "11"}


def test_aer_estimator_bell_expectation(tmp_path) -> None:
    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    result = QuantumRunner(state_dir=tmp_path).simulate_estimate(
        EstimateRequest(circuit, {"ZZ": 1.0}),
        config=RunConfig(precision=0.001, simulation_seed=7),
    )
    assert abs(float(result.expectation_values) - 1.0) < 0.01
