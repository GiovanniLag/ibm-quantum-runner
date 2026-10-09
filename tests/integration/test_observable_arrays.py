"""Offline regression coverage for independent, shaped Estimator observables."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter
from qiskit.primitives.containers.estimator_pub import EstimatorPub
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer.primitives import EstimatorV2

from ibm_quantum_runner import EstimateRequest, QuantumRunner, RunConfig
from ibm_quantum_runner.exceptions import CircuitValidationError
from ibm_quantum_runner.hardware.submit import _estimator_pub as hardware_pub
from ibm_quantum_runner.persistence.artifacts import (
    atomic_write_json,
    circuit_summary,
    deserialize_observable,
    load_circuits,
    load_estimate_inputs,
    normalize_observables,
    save_circuits,
    save_estimate_inputs,
    serialize_observable,
    stable_hash,
)
from ibm_quantum_runner.results.normalize import (
    circuit_result_from_dict,
    normalize_estimator_result,
)
from ibm_quantum_runner.validation import validate_estimator_inputs
from tests.fixtures import FakeJob, FakePrimitiveResult, FakeProvider, fake_submitter


def _swept_circuit() -> QuantumCircuit:
    circuit = QuantumCircuit(2)
    circuit.ry(Parameter("theta"), 0)
    circuit.x(1)
    return circuit


@pytest.mark.parametrize(
    "value, shape",
    [
        (["IZ", "ZI"], (2,)),
        ([["IZ"], ["ZI"]], (2, 1)),
        (np.asarray([[["IZ", "ZI"]], [["ZZ", "II"]]]), (2, 1, 2)),
        ([[SparsePauliOp("IZ"), {"ZZ": 0.5, "II": 0.25}]], (1, 2)),
    ],
)
def test_normalize_preserves_independent_observable_array(value, shape) -> None:
    normalized = normalize_observables(value)
    assert isinstance(normalized, np.ndarray)
    assert normalized.shape == shape
    assert all(isinstance(item, SparsePauliOp) for item in normalized.flat)
    restored = deserialize_observable(serialize_observable(normalized))
    assert isinstance(restored, np.ndarray)
    assert restored.shape == shape
    assert all(left == right for left, right in zip(normalized.flat, restored.flat, strict=True))


@pytest.mark.parametrize("as_mapping", [False, True])
def test_explicit_sum_remains_one_scalar_expectation(tmp_path, as_mapping) -> None:
    circuit = QuantumCircuit(1)
    observable = {"Z": 0.5, "I": 0.25} if as_mapping else SparsePauliOp(["Z", "I"], [0.5, 0.25])
    normalized = normalize_observables(observable)
    assert isinstance(normalized, SparsePauliOp)
    result = QuantumRunner(state_dir=tmp_path).simulate_estimate(
        EstimateRequest(circuit, observable), config=RunConfig(precision=1e-7)
    )
    assert np.shape(result.expectation_values) == ()
    assert result.expectation_values == pytest.approx(0.75, abs=1e-5)


def test_label_list_produces_distinct_values_instead_of_sum(tmp_path) -> None:
    circuit = QuantumCircuit(2)
    circuit.x(1)
    result = QuantumRunner(state_dir=tmp_path).simulate_estimate(
        EstimateRequest(circuit, ["IZ", "ZI", "ZZ"]), config=RunConfig(precision=1e-7)
    )
    assert np.shape(result.expectation_values) == (3,)
    np.testing.assert_allclose(result.expectation_values, [1, -1, -1], atol=1e-5)


@pytest.mark.parametrize("mapping_bindings", [False, True])
def test_aer_preserves_three_dimensional_broadcast_results(tmp_path, mapping_bindings) -> None:
    circuit = _swept_circuit()
    angles = np.asarray([0, np.pi / 2, np.pi])
    parameters = {"theta": angles} if mapping_bindings else angles[:, None]
    request = EstimateRequest(circuit, [[["IZ"], ["ZI"]], [["ZZ"], ["II"]]], parameters)
    report = validate_estimator_inputs([request], RunConfig())
    assert report.is_valid, report.to_dict()
    result = QuantumRunner(state_dir=tmp_path).simulate_estimate(
        request, config=RunConfig(precision=1e-7)
    )
    expected = np.asarray([[[1, 0, -1], [-1, -1, -1]], [[-1, 0, 1], [1, 1, 1]]])
    assert np.shape(result.expectation_values) == (2, 2, 3)
    np.testing.assert_allclose(result.expectation_values, expected, atol=1e-5)
    assert np.shape(result.circuit_results[0].standard_deviations) == (2, 2, 3)


def test_real_layout_maps_every_observable_without_changing_shape_or_bindings() -> None:
    circuit = _swept_circuit()
    parameters = np.asarray([[0], [np.pi]])
    request = EstimateRequest(circuit, [["IZ"], ["ZI"], ["ZZ"]], parameters)
    compiled = transpile(
        circuit,
        basis_gates=["rz", "sx", "x", "cx"],
        coupling_map=[[0, 1], [1, 0], [1, 2], [2, 1]],
        initial_layout=[2, 0],
        optimization_level=0,
        seed_transpiler=7,
    )
    assert compiled.layout.final_index_layout() == [2, 0]
    assert compiled.num_qubits == 3
    pub = hardware_pub(request, compiled)
    assert pub[1].shape == (3, 1)
    assert [item.to_list()[0][0] for item in pub[1].flat] == ["ZII", "IIZ", "ZIZ"]
    assert pub[2] is parameters
    assert EstimatorPub.coerce(pub).shape == (3, 2)
    raw = EstimatorV2().run([pub], precision=0).result()
    np.testing.assert_allclose(raw[0].data.evs, [[1, -1], [-1, -1], [-1, 1]], atol=1e-12)
    # Result normalization and JSON persistence must retain every output axis.
    result = normalize_estimator_result(raw, [circuit_summary(circuit, 0)], precision=0)[0]
    restored = circuit_result_from_dict(json.loads(json.dumps(result.to_dict())))
    assert np.shape(restored.expectation_values) == (3, 2)
    assert np.shape(restored.standard_deviations) == (3, 2)
    np.testing.assert_allclose(restored.expectation_values, raw[0].data.evs)


def test_six_wire_ordered_z_x_features_keep_sample_and_observable_axes(tmp_path) -> None:
    circuit = QuantumCircuit(6)
    # Deliberately make circuit parameter order differ from logical qubit order.
    qubit_parameters = [Parameter(f"angle_{5 - qubit}") for qubit in range(6)]
    for qubit, parameter in enumerate(qubit_parameters):
        circuit.ry(parameter, qubit)
    qubit_angles = np.asarray(
        [[0.1, 0.3, 0.5, 0.7, 0.9, 1.1], [1.2, 1.0, 0.8, 0.6, 0.4, 0.2], [-0.2] * 6]
    )
    column_order = [qubit_parameters.index(parameter) for parameter in circuit.parameters]
    assert column_order == [5, 4, 3, 2, 1, 0]
    parameters = qubit_angles[:, column_order][:, None, :]  # (samples, 1, parameters)
    observables = []
    for basis in ("Z", "X"):
        for qubit in range(6):
            label = ["I"] * 6
            label[5 - qubit] = basis
            observables.append("".join(label))
    request = EstimateRequest(circuit, observables, parameters)
    assert validate_estimator_inputs([request], RunConfig()).is_valid
    save_circuits(tmp_path / "circuits.qpy", [circuit])
    save_estimate_inputs(tmp_path, [request])
    restored = load_estimate_inputs(tmp_path, load_circuits(tmp_path / "circuits.qpy"))[0]
    initial_layout = [5, 1, 7, 0, 6, 3]
    edges = [[left, right] for left in range(8) for right in range(8) if left != right]
    compiled = transpile(
        restored.circuit,
        basis_gates=["rz", "sx", "x", "cx"],
        coupling_map=edges,
        initial_layout=initial_layout,
        optimization_level=0,
        seed_transpiler=7,
    )
    assert compiled.layout.final_index_layout() == initial_layout
    pub = hardware_pub(restored, compiled)
    assert pub[1].shape == (12,)
    assert np.shape(pub[2]) == (3, 1, 6)
    assert EstimatorPub.coerce(pub).shape == (3, 12)
    expected = np.concatenate([np.cos(qubit_angles), np.sin(qubit_angles)], axis=1)
    raw = EstimatorV2().run([pub], precision=0).result()
    np.testing.assert_allclose(raw[0].data.evs, expected, atol=1e-12)
    normalized = normalize_estimator_result(raw, [circuit_summary(circuit, 0)], precision=0)[0]
    assert np.shape(normalized.expectation_values) == (3, 12)
    assert np.shape(normalized.standard_deviations) == (3, 12)
    np.testing.assert_allclose(normalized.expectation_values, expected, atol=1e-12)


def test_sparse_pauli_leaves_are_never_coerced_to_dense_matrices(monkeypatch) -> None:
    summed = SparsePauliOp(["IZ", "ZI"], [0.5, 0.25])
    single = SparsePauliOp("ZZ")

    def reject_matrix_coercion(*args, **kwargs):
        pytest.fail("Observable shape inference must not coerce SparsePauliOp to a matrix")

    monkeypatch.setattr(SparsePauliOp, "__array__", reject_matrix_coercion)
    normalized = normalize_observables([[summed], [single]])
    assert normalized.shape == (2, 1)
    assert normalized[0, 0] is summed
    assert normalized[1, 0] is single
    assert deserialize_observable(serialize_observable(normalized)).shape == (2, 1)


@pytest.mark.parametrize("mapping_bindings", [False, True])
def test_shaped_estimator_artifact_round_trip_with_qpy_and_parameters(tmp_path, mapping_bindings):
    circuit = _swept_circuit()
    angles = np.asarray([0, np.pi])
    parameters = {"theta": angles} if mapping_bindings else angles[:, None]
    observables = [[SparsePauliOp("IZ")], [{"ZI": 0.5, "II": 0.25}]]
    request = EstimateRequest(circuit, observables, parameters, metadata={"test": "shaped"})
    save_circuits(tmp_path / "circuits.qpy", [circuit])
    save_estimate_inputs(tmp_path, [request])
    payload = json.loads((tmp_path / "estimator_inputs.json").read_text())
    assert payload["schema_version"] == 2
    assert payload["requests"][0]["observables"]["shape"] == [2, 1]
    restored = load_estimate_inputs(tmp_path, load_circuits(tmp_path / "circuits.qpy"))[0]
    assert restored.metadata == request.metadata
    assert serialize_observable(
        normalize_observables(restored.observables)
    ) == serialize_observable(normalize_observables(observables))
    assert validate_estimator_inputs([restored], RunConfig()).is_valid
    raw = EstimatorV2().run([hardware_pub(restored, restored.circuit)], precision=0).result()
    np.testing.assert_allclose(raw[0].data.evs, [[1, -1], [-0.25, -0.25]], atol=1e-12)
    with np.load(tmp_path / "parameter_values.npz", allow_pickle=False) as archive:
        assert all(archive[key].dtype == np.float64 for key in archive.files)


def test_legacy_v1_artifact_keeps_its_original_summed_semantics(tmp_path) -> None:
    # V1 encoded one summed operator even when callers originally passed labels.
    summed = SparsePauliOp(["IZ", "ZI"])
    atomic_write_json(
        tmp_path / "estimator_inputs.json",
        {
            "schema_version": 1,
            "requests": [{"observables": serialize_observable(summed), "parameters": None}],
        },
    )
    restored = load_estimate_inputs(tmp_path, [QuantumCircuit(2)])[0]
    assert isinstance(restored.observables, SparsePauliOp)
    assert restored.observables == summed
    assert EstimatorPub.coerce(hardware_pub(restored, restored.circuit)).shape == ()


def test_fingerprint_distinguishes_sum_vector_matrix_and_order() -> None:
    inputs = [
        SparsePauliOp(["IZ", "ZI"]),
        ["IZ", "ZI"],
        [["IZ"], ["ZI"]],
        ["ZI", "IZ"],
    ]
    fingerprints = {
        stable_hash(serialize_observable(normalize_observables(value))) for value in inputs
    }
    assert len(fingerprints) == 4
    scalar_encoding = [
        {"label": "IZ", "coefficient": {"real": 1.0, "imag": 0.0}},
        {"label": "ZI", "coefficient": {"real": 1.0, "imag": 0.0}},
    ]
    assert serialize_observable(normalize_observables(inputs[0])) == scalar_encoding


@pytest.mark.parametrize(
    "observables, values, code",
    [
        (["IZ", "Z"], [[0]], "observable-width-mismatch"),
        ([["IZ"], ["IZ", "ZI"]], [[0]], "invalid-observables"),
        ([], [[0]], "invalid-observables"),
        (["IZ", "ZI"], [[0], [1], [2]], "invalid-estimator-pub"),
        (["IZ", {"ZI": 1j}], [[0]], "non-hermitian-observable"),
        (["IZ", {"ZI": np.nan}], [[0]], "non-finite-observable"),
        (["IZ"], [[0], [1, 2]], "invalid-parameter-values"),
    ],
)
def test_array_validation_reports_bad_leaf_or_broadcast_shape(observables, values, code) -> None:
    report = validate_estimator_inputs(
        [EstimateRequest(_swept_circuit(), observables, values)], RunConfig()
    )
    assert not report.is_valid
    assert code in {issue.code for issue in report.errors}


@pytest.mark.parametrize("observables", [[], [["IZ"], ["ZI", "ZZ"]]])
def test_invalid_observable_arrays_fail_public_submit_before_provider_access(
    tmp_path, monkeypatch, observables
) -> None:
    provider = FakeProvider()

    def reject_provider_access(*args, **kwargs):
        pytest.fail("Invalid observable arrays must never select a backend or submit a job")

    monkeypatch.setattr(provider, "select_backend", reject_provider_access)
    monkeypatch.setattr("ibm_quantum_runner.runner.submit_workload", reject_provider_access)
    runner = QuantumRunner(state_dir=tmp_path, provider=provider)
    with pytest.raises(CircuitValidationError) as raised:
        runner.submit_estimate(
            EstimateRequest(_swept_circuit(), observables, [[0]]),
            config=RunConfig(simulate_first=False),
        )
    assert "invalid-observables" in {issue.code for issue in raised.value.report.errors}
    assert raised.value.execution_id is not None
    directory = tmp_path / "executions" / raised.value.execution_id
    assert not (directory / "estimator_inputs.json").exists()
    assert (directory / "validation.json").exists()


@pytest.mark.parametrize(
    "payload",
    [
        {"kind": "unknown", "shape": [1], "items": []},
        {"kind": "array", "shape": [2], "items": []},
        {"kind": "array", "shape": [0], "items": []},
        {"kind": "array", "shape": [True], "items": [[]]},
        {"kind": "array", "shape": [1], "items": [{"kind": "array"}]},
    ],
)
def test_reject_corrupted_observable_array_encoding(payload) -> None:
    with pytest.raises(ValueError):
        deserialize_observable(payload)


def test_public_execution_reloads_shaped_results_and_request_artifacts(tmp_path, monkeypatch):
    evs = np.asarray([[1.0, -1.0], [-1.0, -1.0]])
    pub = SimpleNamespace(data=SimpleNamespace(evs=evs, stds=np.zeros_like(evs)), metadata={})
    jobs = {"fake-job-0": FakeJob("fake-job-0", ["DONE"], FakePrimitiveResult([pub]))}
    monkeypatch.setattr("ibm_quantum_runner.runner.submit_workload", fake_submitter(jobs))
    runner = QuantumRunner(state_dir=tmp_path, provider=FakeProvider(jobs))
    request = EstimateRequest(_swept_circuit(), [["IZ"], ["ZI"]], [[0], [np.pi]])
    execution = runner.submit_estimate(request, config=RunConfig(simulate_first=False))
    execution.wait(poll_interval=0.001)
    reloaded = QuantumRunner(state_dir=tmp_path, provider=FakeProvider(jobs)).resume(execution.id)
    np.testing.assert_array_equal(reloaded.result().expectation_values, evs)
    directory = tmp_path / "executions" / execution.id
    restored = load_estimate_inputs(directory, load_circuits(directory / "circuits.qpy"))[0]
    assert np.shape(restored.observables) == (2, 1)
    assert np.shape(restored.parameter_values) == (2, 1)
