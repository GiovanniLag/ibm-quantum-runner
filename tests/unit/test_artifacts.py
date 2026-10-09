from __future__ import annotations

from qiskit import QuantumCircuit

from ibm_quantum_runner.persistence.artifacts import (
    circuit_fingerprint,
    load_circuits,
    save_circuits,
)


def test_qpy_round_trip_and_fingerprint_are_stable(tmp_path) -> None:
    circuit = QuantumCircuit(2, name="bell")
    circuit.h(0)
    circuit.cx(0, 1)
    path = tmp_path / "circuits.qpy"

    save_circuits(path, [circuit])
    restored = load_circuits(path)[0]

    assert circuit_fingerprint(restored) == circuit_fingerprint(circuit)
    assert restored.name == "bell"
