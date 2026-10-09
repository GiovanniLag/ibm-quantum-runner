"""Stable library-level result models."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from ..simulation.models import SimulationResult


@dataclass(frozen=True, slots=True)
class SamplerCircuitResult:
    circuit_index: int
    circuit_id: str
    circuit_name: str
    counts: Mapping[str, int]
    register_counts: Mapping[str, Mapping[str, int]]
    shots: int
    metadata: Mapping[str, Any] = field(default_factory=dict)
    status: str = "completed"
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "sampler",
            "circuit_index": self.circuit_index,
            "circuit_id": self.circuit_id,
            "circuit_name": self.circuit_name,
            "counts": dict(self.counts),
            "register_counts": {
                name: dict(counts) for name, counts in self.register_counts.items()
            },
            "shots": self.shots,
            "metadata": dict(self.metadata),
            "status": self.status,
            "error": self.error,
        }


@dataclass(frozen=True, slots=True)
class EstimatorCircuitResult:
    circuit_index: int
    circuit_id: str
    circuit_name: str
    expectation_values: Any
    standard_deviations: Any
    precision: float | None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    status: str = "completed"
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "estimator",
            "circuit_index": self.circuit_index,
            "circuit_id": self.circuit_id,
            "circuit_name": self.circuit_name,
            "expectation_values": self.expectation_values,
            "standard_deviations": self.standard_deviations,
            "precision": self.precision,
            "metadata": dict(self.metadata),
            "status": self.status,
            "error": self.error,
        }


CircuitResult = SamplerCircuitResult | EstimatorCircuitResult


@dataclass(frozen=True, slots=True)
class HardwareResult:
    backend_name: str | None
    ibm_job_ids: tuple[str, ...]
    circuit_results: tuple[CircuitResult, ...]
    timing: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend_name": self.backend_name,
            "ibm_job_ids": list(self.ibm_job_ids),
            "circuit_results": [item.to_dict() for item in self.circuit_results],
            "timing": dict(self.timing),
            "metadata": dict(self.metadata),
        }


@dataclass(slots=True)
class ExecutionResult:
    execution_id: str
    primitive: Literal["sampler", "estimator"]
    final_status: str
    backend_name: str | None
    ibm_job_ids: tuple[str, ...]
    hardware: HardwareResult | None
    simulation: SimulationResult | None
    submission_metadata: Mapping[str, Any] = field(default_factory=dict)
    raw_provider_result: Any = field(default=None, repr=False, compare=False)

    @property
    def circuit_results(self) -> tuple[CircuitResult, ...]:
        return self.hardware.circuit_results if self.hardware is not None else ()

    @property
    def counts(self) -> Mapping[str, int] | None:
        if len(self.circuit_results) != 1:
            return None
        item = self.circuit_results[0]
        return item.counts if isinstance(item, SamplerCircuitResult) else None

    @property
    def expectation_values(self) -> Any:
        if len(self.circuit_results) != 1:
            return None
        item = self.circuit_results[0]
        return item.expectation_values if isinstance(item, EstimatorCircuitResult) else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "execution_id": self.execution_id,
            "primitive": self.primitive,
            "final_status": self.final_status,
            "backend_name": self.backend_name,
            "ibm_job_ids": list(self.ibm_job_ids),
            "hardware": self.hardware.to_dict() if self.hardware is not None else None,
            "simulation": self.simulation.to_dict() if self.simulation is not None else None,
            "submission_metadata": dict(self.submission_metadata),
        }
