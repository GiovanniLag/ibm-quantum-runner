"""Versioned, non-pickle artifact persistence and reproducible fingerprints."""

from __future__ import annotations

import hashlib
import io
import json
import os
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from contextlib import suppress
from datetime import UTC, datetime
from math import prod
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from qiskit import QuantumCircuit, qpy
from qiskit.quantum_info import SparsePauliOp

from ..config import EstimateRequest
from ..results.normalize import json_safe

ARTIFACT_SCHEMA_VERSION = 1
ESTIMATOR_INPUT_SCHEMA_VERSION = 2


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def stable_hash(payload: Any) -> str:
    encoded = json.dumps(
        json_safe(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(json_safe(payload), handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        with suppress(FileNotFoundError):
            os.unlink(temporary)
        raise


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("JSON artifact must contain an object")
    return payload


def qpy_bytes(circuit: QuantumCircuit) -> bytes:
    stream = io.BytesIO()
    qpy.dump(circuit, stream)
    return stream.getvalue()


def circuit_fingerprint(circuit: QuantumCircuit) -> str:
    return hashlib.sha256(qpy_bytes(circuit)).hexdigest()


def save_circuits(path: Path, circuits: Sequence[QuantumCircuit]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            qpy.dump(list(circuits), handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        with suppress(FileNotFoundError):
            os.unlink(temporary)
        raise


def load_circuits(path: Path) -> list[QuantumCircuit]:
    with path.open("rb") as handle:
        return list(qpy.load(handle))


def circuit_summary(circuit: QuantumCircuit, index: int) -> dict[str, Any]:
    return {
        "circuit_index": index,
        "fingerprint": circuit_fingerprint(circuit),
        "name": circuit.name or f"circuit-{index}",
        "num_qubits": circuit.num_qubits,
        "num_clbits": circuit.num_clbits,
        "depth": circuit.depth(),
        "operation_counts": {str(name): int(count) for name, count in circuit.count_ops().items()},
        "metadata": json_safe(circuit.metadata or {}),
    }


def _observable_shape_and_items(value: Any) -> tuple[tuple[int, ...], list[SparsePauliOp]]:
    """Find array shape without NumPy coercing SparsePauliOps into matrices."""

    if isinstance(value, SparsePauliOp):
        return (), [value]
    if isinstance(value, str):
        return (), [SparsePauliOp(value)]
    if isinstance(value, Mapping):
        return (), [
            SparsePauliOp.from_list(
                [(str(label), complex(coeff)) for label, coeff in value.items()]
            )
        ]
    if isinstance(value, np.ndarray) and value.ndim == 0:
        return _observable_shape_and_items(value.item())
    if isinstance(value, np.ndarray) or (
        isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray))
    ):
        if len(value) == 0:
            raise ValueError("Observable arrays must not be empty")
        children = [_observable_shape_and_items(item) for item in value]
        child_shape = children[0][0]
        if any(shape != child_shape for shape, _ in children):
            raise ValueError("Observable arrays must have a rectangular shape")
        return (len(children), *child_shape), [item for _, items in children for item in items]
    try:
        return (), [SparsePauliOp(value)]
    except Exception as exc:
        raise TypeError(f"Unsupported observable input: {type(value).__name__}") from exc


def normalize_observables(value: Any) -> SparsePauliOp | NDArray[np.object_]:
    """Preserve scalar sums and arbitrary rectangular arrays of observables.

    A sequence of labels is an array of independent observables, never an
    implicit sum. SparsePauliOp and coefficient mappings remain scalar sums.
    """

    shape, items = _observable_shape_and_items(value)
    if not shape:
        return items[0]
    array = np.empty(shape, dtype=object)
    for index, item in zip(np.ndindex(shape), items, strict=True):
        array[index] = item
    return array


def iter_observables(
    observables: SparsePauliOp | NDArray[np.object_],
) -> Iterator[tuple[tuple[int, ...], SparsePauliOp]]:
    """Yield each scalar observable with its position in the preserved array."""

    if isinstance(observables, np.ndarray):
        yield from np.ndenumerate(observables)
    else:
        yield (), observables


def serialize_observable(
    observable: SparsePauliOp | NDArray[np.object_],
) -> list[dict[str, Any]] | dict[str, Any]:
    if isinstance(observable, np.ndarray):
        return {
            "kind": "array",
            "shape": list(observable.shape),
            "items": [serialize_observable(item) for _, item in iter_observables(observable)],
        }
    # Retain the v1 scalar encoding and its fingerprint for existing executions.
    return [
        {"label": label, "coefficient": {"real": coefficient.real, "imag": coefficient.imag}}
        for label, coefficient in observable.to_list()
    ]


def deserialize_observable(
    payload: Sequence[Mapping[str, Any]] | Mapping[str, Any],
) -> SparsePauliOp | NDArray[np.object_]:
    if isinstance(payload, Mapping):
        if payload.get("kind") != "array":
            raise ValueError("Unsupported observable encoding")
        shape = payload["shape"]
        if (
            not isinstance(shape, list)
            or not shape
            or any(
                isinstance(size, bool) or not isinstance(size, int) or size <= 0 for size in shape
            )
        ):
            raise ValueError("Invalid observable array shape")
        items = payload["items"]
        if not isinstance(items, list) or len(items) != prod(shape):
            raise ValueError("Observable array shape does not match its item count")
        array = np.empty(tuple(shape), dtype=object)
        for index, item in zip(np.ndindex(array.shape), items, strict=True):
            if not isinstance(item, list):
                raise ValueError("Each encoded observable must be a scalar term list")
            array[index] = deserialize_observable(item)
        return array
    return SparsePauliOp.from_list(
        [
            (
                str(item["label"]),
                complex(
                    float(item["coefficient"]["real"]),
                    float(item["coefficient"]["imag"]),
                ),
            )
            for item in payload
        ]
    )


def save_estimate_inputs(directory: Path, requests: Sequence[EstimateRequest]) -> dict[str, Any]:
    """Persist observables and numeric parameter values without arbitrary pickle."""

    directory.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    records: list[dict[str, Any]] = []
    for index, request in enumerate(requests):
        observable = normalize_observables(request.observables)
        parameter_spec: dict[str, Any] | None = None
        if request.parameter_values is not None:
            if isinstance(request.parameter_values, Mapping):
                keys: list[str] = []
                array_keys: list[str] = []
                for offset, (name, values) in enumerate(request.parameter_values.items()):
                    array_key = f"request_{index}_mapping_{offset}"
                    arrays[array_key] = np.asarray(values, dtype=np.float64)
                    keys.append(str(name))
                    array_keys.append(array_key)
                parameter_spec = {"kind": "mapping", "keys": keys, "arrays": array_keys}
            else:
                array_key = f"request_{index}_values"
                arrays[array_key] = np.asarray(request.parameter_values, dtype=np.float64)
                parameter_spec = {"kind": "array", "array": array_key}
        records.append(
            {
                "observables": serialize_observable(observable),
                "parameters": parameter_spec,
                "metadata": json_safe(request.metadata),
            }
        )

    parameter_path = directory / "parameter_values.npz"
    if arrays:
        save_archive: Any = np.savez_compressed
        save_archive(parameter_path, **arrays)
    atomic_write_json(
        directory / "estimator_inputs.json",
        {"schema_version": ESTIMATOR_INPUT_SCHEMA_VERSION, "requests": records},
    )
    return {"request_count": len(records), "has_parameter_values": bool(arrays)}


def load_estimate_inputs(
    directory: Path, circuits: Sequence[QuantumCircuit]
) -> list[EstimateRequest]:
    payload = load_json(directory / "estimator_inputs.json")
    if payload.get("schema_version") not in {1, ESTIMATOR_INPUT_SCHEMA_VERSION}:
        raise ValueError("Unsupported estimator input schema")
    parameter_path = directory / "parameter_values.npz"
    arrays: dict[str, np.ndarray] = {}
    if parameter_path.exists():
        with np.load(parameter_path, allow_pickle=False) as archive:
            arrays = {name: archive[name].copy() for name in archive.files}

    requests: list[EstimateRequest] = []
    for circuit, record in zip(circuits, payload["requests"], strict=True):
        spec = record.get("parameters")
        parameters: Any = None
        if spec and spec["kind"] == "array":
            parameters = arrays[spec["array"]]
        elif spec and spec["kind"] == "mapping":
            parameters = {
                name: arrays[array_key]
                for name, array_key in zip(spec["keys"], spec["arrays"], strict=True)
            }
        requests.append(
            EstimateRequest(
                circuit=circuit,
                observables=deserialize_observable(record["observables"]),
                parameter_values=parameters,
                metadata=record.get("metadata", {}),
            )
        )
    return requests
