"""Local simulation result models."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from ..results.models import CircuitResult, EstimatorCircuitResult, SamplerCircuitResult
from ..results.normalize import circuit_result_from_dict


@dataclass(frozen=True, slots=True)
class SimulationCheck:
    passed: bool
    message: str = ""
    details: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "message": self.message,
            "details": dict(self.details),
        }


@dataclass(frozen=True, slots=True)
class SimulationResult:
    primitive: Literal["sampler", "estimator"]
    circuit_results: tuple[CircuitResult, ...]
    validator_check: SimulationCheck | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

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
            "primitive": self.primitive,
            "circuit_results": [item.to_dict() for item in self.circuit_results],
            "validator_check": (
                self.validator_check.to_dict() if self.validator_check is not None else None
            ),
            "metadata": dict(self.metadata),
        }


def simulation_result_from_dict(payload: Mapping[str, Any]) -> SimulationResult:
    """Rehydrate the stable, JSON-persisted simulation result."""

    raw_check = payload.get("validator_check")
    check = None
    if isinstance(raw_check, Mapping):
        check = SimulationCheck(
            passed=bool(raw_check["passed"]),
            message=str(raw_check.get("message", "")),
            details=raw_check.get("details", {}),
        )
    return SimulationResult(
        primitive=payload["primitive"],
        circuit_results=tuple(
            circuit_result_from_dict(item) for item in payload.get("circuit_results", [])
        ),
        validator_check=check,
        metadata=payload.get("metadata", {}),
    )
