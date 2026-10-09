"""Circuit, Estimator request, and transpiled-provider validation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from qiskit import QuantumCircuit
from qiskit.primitives.containers.estimator_pub import EstimatorPub

from ..config import EstimateRequest, RunConfig
from ..persistence.artifacts import iter_observables, normalize_observables, qpy_bytes
from .models import ValidationReport

SAMPLER_MAX_EXECUTIONS_PER_JOB = 10_000_000
MAX_RZ_PER_CIRCUIT = 30_000_000
MAX_SX_PER_CIRCUIT = 20_000_000
MAX_TWO_QUBIT_GATES_PER_CIRCUIT = 5_000_000
MAX_LOW_LEVEL_INSTRUCTIONS_PER_QUBIT = 26_800_000
INSTRUCTION_WEIGHTS = {
    "rz": 1,
    "delay": 1,
    "sx": 2,
    "x": 2,
    "cx": 5,
    "cz": 5,
    "ecr": 5,
    "measure": 10,
    "reset": 17,
    "initialize": 50,
}


def _validate_circuit_type(circuit: Any, report: ValidationReport, circuit_index: int) -> bool:
    if not isinstance(circuit, QuantumCircuit):
        report.add(
            "error",
            "invalid-circuit-type",
            f"Expected QuantumCircuit, received {type(circuit).__name__}",
            circuit_index=circuit_index,
        )
        return False
    try:
        qpy_bytes(circuit)
    except Exception as exc:
        report.add(
            "error",
            "qpy-serialization-failed",
            "Circuit cannot be serialized safely using QPY",
            circuit_index=circuit_index,
            details={"error": repr(exc)},
        )
        return False
    return True


def validate_sampler_inputs(circuits: Sequence[Any], config: RunConfig) -> ValidationReport:
    report = ValidationReport()
    if not circuits:
        report.add("error", "empty-workload", "At least one circuit is required")
        return report
    for index, circuit in enumerate(circuits):
        if not _validate_circuit_type(circuit, report, index):
            continue
        assert isinstance(circuit, QuantumCircuit)
        measure_count = int(circuit.count_ops().get("measure", 0))
        if measure_count == 0:
            report.add(
                "error",
                "measurements-required",
                "Sampler circuits must contain at least one measurement",
                circuit_index=index,
            )
        if circuit.num_clbits == 0 or not circuit.cregs:
            report.add(
                "error",
                "classical-register-required",
                "Sampler circuits must expose at least one classical register",
                circuit_index=index,
            )
        if circuit.parameters:
            report.add(
                "error",
                "unbound-parameters",
                "Sampler inputs must be fully bound before submission",
                circuit_index=index,
                details={"parameters": sorted(str(item) for item in circuit.parameters)},
            )
    if len(circuits) * config.shots > SAMPLER_MAX_EXECUTIONS_PER_JOB:
        report.add(
            "info",
            "workload-requires-chunking",
            "The workload exceeds IBM's per-job Sampler execution limit and will be partitioned",
            details={
                "executions": len(circuits) * config.shots,
                "limit": SAMPLER_MAX_EXECUTIONS_PER_JOB,
            },
        )
    return report


def _validate_parameter_values(
    request: EstimateRequest, report: ValidationReport, circuit_index: int
) -> None:
    parameters = list(request.circuit.parameters)
    values = request.parameter_values
    if values is None:
        if parameters:
            report.add(
                "error",
                "missing-parameter-values",
                "Estimator circuit has unbound parameters but no parameter values",
                circuit_index=circuit_index,
                details={"parameters": [str(item) for item in parameters]},
            )
        return
    if isinstance(values, Mapping):
        names = {str(key) for key in values}
        expected = {str(parameter) for parameter in parameters}
        if names != expected:
            report.add(
                "error",
                "parameter-mapping-mismatch",
                "Estimator parameter mapping must cover every circuit parameter exactly",
                circuit_index=circuit_index,
                details={"expected": sorted(expected), "received": sorted(names)},
            )
        try:
            arrays = [np.asarray(value) for value in values.values()]
        except (TypeError, ValueError) as exc:
            report.add(
                "error",
                "invalid-parameter-values",
                "Estimator parameter values must form rectangular numeric arrays",
                circuit_index=circuit_index,
                details={"error": str(exc)},
            )
            return
    else:
        try:
            array = np.asarray(values)
        except (TypeError, ValueError) as exc:
            report.add(
                "error",
                "invalid-parameter-values",
                "Estimator parameter values must form rectangular numeric arrays",
                circuit_index=circuit_index,
                details={"error": str(exc)},
            )
            return
        arrays = [array]
        if array.ndim == 0 or array.shape[-1] != len(parameters):
            report.add(
                "error",
                "parameter-array-shape",
                "The final parameter-value dimension must match the circuit parameter count",
                circuit_index=circuit_index,
                details={"shape": array.shape, "parameter_count": len(parameters)},
            )
    for array in arrays:
        try:
            finite: bool = bool(np.isfinite(np.asarray(array, dtype=np.float64)).all())
        except (TypeError, ValueError):
            finite = False
        if not finite:
            report.add(
                "error",
                "non-finite-parameter-values",
                "Estimator parameter values must be finite real numbers",
                circuit_index=circuit_index,
            )
            break


def validate_estimator_inputs(requests: Sequence[Any], config: RunConfig) -> ValidationReport:
    report = ValidationReport()
    if not requests:
        report.add("error", "empty-workload", "At least one EstimateRequest is required")
        return report
    for index, request in enumerate(requests):
        if not isinstance(request, EstimateRequest):
            report.add(
                "error",
                "invalid-estimate-request",
                f"Expected EstimateRequest, received {type(request).__name__}",
                circuit_index=index,
            )
            continue
        circuit = request.circuit
        if not _validate_circuit_type(circuit, report, index):
            continue
        first_issue = len(report.issues)
        if circuit.count_ops().get("measure", 0):
            report.add(
                "error",
                "estimator-measurement-not-supported",
                "Estimator circuits must not contain measurement instructions",
                circuit_index=index,
            )
        try:
            observables = normalize_observables(request.observables)
            for observable_index, observable in iter_observables(observables):
                if observable.num_qubits != circuit.num_qubits:
                    report.add(
                        "error",
                        "observable-width-mismatch",
                        "Observable width must match the circuit qubit count",
                        circuit_index=index,
                        details={
                            "observable_index": list(observable_index),
                            "observable_qubits": observable.num_qubits,
                            "circuit_qubits": circuit.num_qubits,
                        },
                    )
                coefficients = np.asarray(observable.coeffs, dtype=complex)
                if not np.isfinite(coefficients).all():
                    report.add(
                        "error",
                        "non-finite-observable",
                        "Estimator observable coefficients must be finite",
                        circuit_index=index,
                        details={"observable_index": list(observable_index)},
                    )
                if not np.allclose(coefficients.imag, 0):
                    report.add(
                        "error",
                        "non-hermitian-observable",
                        "Estimator observables must have real coefficients",
                        circuit_index=index,
                        details={"observable_index": list(observable_index)},
                    )
        except Exception as exc:
            report.add(
                "error",
                "invalid-observables",
                "Observables could not be normalized to scalar or array SparsePauliOp inputs",
                circuit_index=index,
                details={"error": repr(exc)},
            )
        _validate_parameter_values(request, report, index)
        if not any(issue.severity == "error" for issue in report.issues[first_issue:]):
            try:
                EstimatorPub.coerce((circuit, observables, request.parameter_values))
            except (TypeError, ValueError) as exc:
                report.add(
                    "error",
                    "invalid-estimator-pub",
                    "Observable and parameter arrays must form a valid broadcastable Estimator PUB",
                    circuit_index=index,
                    details={"error": str(exc)},
                )
    return report


def _instruction_qubits(circuit: QuantumCircuit, instruction: Any) -> list[int]:
    return [circuit.find_bit(qubit).index for qubit in instruction.qubits]


def validate_transpiled_circuits(
    circuits: Sequence[QuantumCircuit],
    *,
    primitive: str,
    config: RunConfig,
    backend_num_qubits: int,
) -> ValidationReport:
    report = ValidationReport()
    for index, circuit in enumerate(circuits):
        if circuit.num_qubits > backend_num_qubits:
            report.add(
                "error",
                "backend-qubit-capacity",
                "Circuit exceeds backend qubit capacity",
                circuit_index=index,
                details={
                    "circuit_qubits": circuit.num_qubits,
                    "backend_qubits": backend_num_qubits,
                },
            )
        counts = {str(name): int(count) for name, count in circuit.count_ops().items()}
        if counts.get("rz", 0) > MAX_RZ_PER_CIRCUIT:
            report.add(
                "error",
                "provider-rz-limit",
                "Circuit exceeds IBM's RZ gate limit",
                circuit_index=index,
            )
        if counts.get("sx", 0) > MAX_SX_PER_CIRCUIT:
            report.add(
                "error",
                "provider-sx-limit",
                "Circuit exceeds IBM's SX gate limit",
                circuit_index=index,
            )
        two_qubit = sum(1 for instruction in circuit.data if len(instruction.qubits) == 2)
        if two_qubit > MAX_TWO_QUBIT_GATES_PER_CIRCUIT:
            report.add(
                "error",
                "provider-two-qubit-limit",
                "Circuit exceeds IBM's two-qubit gate limit",
                circuit_index=index,
            )
        per_qubit = [0] * circuit.num_qubits
        for instruction in circuit.data:
            weight = INSTRUCTION_WEIGHTS.get(instruction.operation.name, 0)
            for qubit in _instruction_qubits(circuit, instruction):
                per_qubit[qubit] += weight
        if per_qubit and max(per_qubit) > MAX_LOW_LEVEL_INSTRUCTIONS_PER_QUBIT:
            report.add(
                "error",
                "provider-control-instruction-limit",
                "Circuit exceeds IBM's low-level instruction limit for a qubit",
                circuit_index=index,
                details={"maximum": max(per_qubit)},
            )
    if primitive == "sampler" and len(circuits) * config.shots > SAMPLER_MAX_EXECUTIONS_PER_JOB:
        report.add(
            "error",
            "provider-sampler-execution-limit",
            "A single child job would exceed 10 million Sampler executions",
            details={"executions": len(circuits) * config.shots},
        )
    return report


def layout_metadata(circuit: QuantumCircuit) -> dict[str, Any]:
    layout = getattr(circuit, "layout", None)
    final_index_layout = getattr(layout, "final_index_layout", None)
    physical = []
    if callable(final_index_layout):
        try:
            physical = [int(value) for value in final_index_layout()]
        except Exception:
            physical = []
    return {
        "physical_qubits": physical,
        "logical_to_physical_qubits": {str(index): value for index, value in enumerate(physical)},
        "layout_source": "transpilation" if physical else "unavailable",
        "depth": circuit.depth(),
        "operation_counts": {str(name): int(count) for name, count in circuit.count_ops().items()},
    }
