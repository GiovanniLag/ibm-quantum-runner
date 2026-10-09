"""Normalization helpers for local and IBM Runtime primitive results."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np

from .models import EstimatorCircuitResult, SamplerCircuitResult


def json_safe(value: Any) -> Any:
    """Convert provider metadata to a stable JSON-compatible structure."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, complex):
        return {"real": value.real, "imag": value.imag}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return json_safe(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [json_safe(item) for item in value]
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        try:
            return json_safe(to_dict())
        except Exception:
            pass
    return str(value)


def _data_items(data: Any) -> list[tuple[str, Any]]:
    items = getattr(data, "items", None)
    if callable(items):
        return [(str(name), value) for name, value in items()]
    keys = getattr(data, "keys", None)
    if callable(keys):
        return [(str(name), data[name]) for name in keys()]
    return [
        (str(name), getattr(data, name))
        for name in getattr(data, "__dict__", {})
        if not str(name).startswith("_")
    ]


def normalize_sampler_result(
    raw_result: Any,
    circuit_records: Sequence[Mapping[str, Any]],
) -> tuple[SamplerCircuitResult, ...]:
    """Normalize a PrimitiveResult returned by Sampler V2."""

    normalized: list[SamplerCircuitResult] = []
    pubs = list(raw_result)
    if len(pubs) != len(circuit_records):
        raise ValueError(
            f"Sampler returned {len(pubs)} PUB results for {len(circuit_records)} circuits"
        )
    for pub, record in zip(pubs, circuit_records, strict=True):
        register_counts: dict[str, dict[str, int]] = {}
        for name, bit_array in _data_items(pub.data):
            get_counts = getattr(bit_array, "get_counts", None)
            if callable(get_counts):
                register_counts[name] = {
                    str(bitstring): int(count) for bitstring, count in get_counts().items()
                }

        joined_counts: dict[str, int] = {}
        join_data = getattr(pub, "join_data", None)
        if callable(join_data):
            joined = join_data()
            get_counts = getattr(joined, "get_counts", None)
            if callable(get_counts):
                joined_counts = {
                    str(bitstring): int(count) for bitstring, count in get_counts().items()
                }
        if not joined_counts and len(register_counts) == 1:
            joined_counts = next(iter(register_counts.values()))

        metadata = json_safe(getattr(pub, "metadata", {}) or {})
        shots = int(metadata.get("shots") or sum(joined_counts.values()))
        normalized.append(
            SamplerCircuitResult(
                circuit_index=int(record["circuit_index"]),
                circuit_id=str(record["fingerprint"]),
                circuit_name=str(record["name"]),
                counts=joined_counts,
                register_counts=register_counts,
                shots=shots,
                metadata=metadata,
            )
        )
    return tuple(normalized)


def normalize_estimator_result(
    raw_result: Any,
    circuit_records: Sequence[Mapping[str, Any]],
    *,
    precision: float | None,
) -> tuple[EstimatorCircuitResult, ...]:
    """Normalize a PrimitiveResult returned by Estimator V2."""

    normalized: list[EstimatorCircuitResult] = []
    pubs = list(raw_result)
    if len(pubs) != len(circuit_records):
        raise ValueError(
            f"Estimator returned {len(pubs)} PUB results for {len(circuit_records)} requests"
        )
    for pub, record in zip(pubs, circuit_records, strict=True):
        data = pub.data
        evs = json_safe(np.asarray(data.evs, dtype=np.float64))
        raw_stds = getattr(data, "stds", None)
        stds = json_safe(np.asarray(raw_stds, dtype=np.float64)) if raw_stds is not None else None
        metadata = json_safe(getattr(pub, "metadata", {}) or {})
        normalized.append(
            EstimatorCircuitResult(
                circuit_index=int(record["circuit_index"]),
                circuit_id=str(record["fingerprint"]),
                circuit_name=str(record["name"]),
                expectation_values=evs,
                standard_deviations=stds,
                precision=precision,
                metadata=metadata,
            )
        )
    return tuple(normalized)


def circuit_result_from_dict(
    payload: Mapping[str, Any],
) -> SamplerCircuitResult | EstimatorCircuitResult:
    if payload["kind"] == "sampler":
        return SamplerCircuitResult(
            circuit_index=int(payload["circuit_index"]),
            circuit_id=str(payload["circuit_id"]),
            circuit_name=str(payload["circuit_name"]),
            counts={str(key): int(value) for key, value in payload.get("counts", {}).items()},
            register_counts={
                str(name): {str(key): int(value) for key, value in counts.items()}
                for name, counts in payload.get("register_counts", {}).items()
            },
            shots=int(payload.get("shots", 0)),
            metadata=payload.get("metadata", {}),
            status=str(payload.get("status", "completed")),
            error=payload.get("error"),
        )
    return EstimatorCircuitResult(
        circuit_index=int(payload["circuit_index"]),
        circuit_id=str(payload["circuit_id"]),
        circuit_name=str(payload["circuit_name"]),
        expectation_values=payload.get("expectation_values"),
        standard_deviations=payload.get("standard_deviations"),
        precision=payload.get("precision"),
        metadata=payload.get("metadata", {}),
        status=str(payload.get("status", "completed")),
        error=payload.get("error"),
    )
