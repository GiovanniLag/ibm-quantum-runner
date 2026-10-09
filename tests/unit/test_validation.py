from __future__ import annotations

from qiskit import QuantumCircuit

from ibm_quantum_runner import EstimateRequest, RunConfig
from ibm_quantum_runner.validation import validate_estimator_inputs, validate_sampler_inputs


def test_sampler_requires_measurement_and_classical_register() -> None:
    circuit = QuantumCircuit(1)
    report = validate_sampler_inputs([circuit], RunConfig())
    codes = {issue.code for issue in report.errors}
    assert codes == {"measurements-required", "classical-register-required"}


def test_estimator_rejects_observable_width_mismatch() -> None:
    circuit = QuantumCircuit(2)
    report = validate_estimator_inputs([EstimateRequest(circuit, "Z")], RunConfig())
    assert not report.is_valid
    assert report.errors[0].code == "observable-width-mismatch"


def test_sampler_valid_circuit_has_structured_report() -> None:
    circuit = QuantumCircuit(1, 1)
    circuit.measure(0, 0)
    report = validate_sampler_inputs([circuit], RunConfig(shots=100))
    assert report.is_valid
    assert report.to_dict()["schema_version"] == 1
