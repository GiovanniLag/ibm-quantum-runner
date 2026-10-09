"""Typed configuration and Estimator request models."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

PrimitiveKind: TypeAlias = Literal["sampler", "estimator"]
ObservableInput: TypeAlias = (
    SparsePauliOp | str | Mapping[str, complex | float] | Sequence["ObservableInput"] | NDArray[Any]
)
ParameterValues: TypeAlias = Mapping[str, Any] | Sequence[Sequence[float]] | np.ndarray
SimulationValidator: TypeAlias = Callable[[Any], bool | Any]


@dataclass(frozen=True, slots=True)
class RetryConfig:
    """Bounded retry policy for idempotent provider operations."""

    max_attempts: int = 3
    initial_delay: float = 1.0
    max_delay: float = 30.0
    multiplier: float = 2.0
    jitter: float = 0.2

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if self.initial_delay < 0 or self.max_delay < 0:
            raise ValueError("retry delays cannot be negative")
        if self.max_delay < self.initial_delay:
            raise ValueError("max_delay must be at least initial_delay")
        if self.multiplier < 1:
            raise ValueError("multiplier must be at least 1")
        if not 0 <= self.jitter <= 1:
            raise ValueError("jitter must be in [0, 1]")


@dataclass(slots=True)
class EstimateRequest:
    """Input for one Estimator V2 PUB, preserving observable array shape.

    Each string, coefficient mapping, or ``SparsePauliOp`` is one observable.
    Nested sequences and NumPy arrays describe independent observables and use
    Qiskit's broadcasting rules with ``parameter_values``. To estimate a sum,
    pass a coefficient mapping or a single ``SparsePauliOp`` explicitly.
    """

    circuit: QuantumCircuit
    observables: ObservableInput
    parameter_values: ParameterValues | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RunConfig:
    """Configuration shared by simulation and hardware execution."""

    shots: int = 1024
    precision: float | None = None
    simulate_first: bool = True
    stop_on_simulation_failure: bool = True
    simulation_validator: SimulationValidator | None = field(
        default=None, repr=False, compare=False
    )
    simulation_seed: int | None = 1234
    backend: str | None = None
    auto_select_backend: bool = False
    optimization_level: int = 1
    seed_transpiler: int | None = 0
    max_pubs_per_job: int = 100
    poll_interval: float = 10.0
    wait_timeout: float | None = None
    max_execution_time: int | None = None
    retry: RetryConfig = field(default_factory=RetryConfig)
    allow_duplicate_submission: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.shots, bool) or not isinstance(self.shots, int) or self.shots <= 0:
            raise ValueError("shots must be a positive integer")
        if self.precision is not None and (
            not math.isfinite(self.precision) or self.precision <= 0
        ):
            raise ValueError("precision must be finite and positive")
        if self.optimization_level not in {0, 1, 2, 3}:
            raise ValueError("optimization_level must be one of 0, 1, 2, 3")
        if self.max_pubs_per_job <= 0:
            raise ValueError("max_pubs_per_job must be positive")
        if self.poll_interval <= 0:
            raise ValueError("poll_interval must be positive")
        if self.wait_timeout is not None and self.wait_timeout <= 0:
            raise ValueError("wait_timeout must be positive when provided")
        if self.max_execution_time is not None and self.max_execution_time <= 0:
            raise ValueError("max_execution_time must be positive when provided")

    def estimator_precision(self) -> float:
        """Return explicit precision or the reference-compatible shot-derived target."""

        return self.precision if self.precision is not None else 1.0 / math.sqrt(self.shots)

    def persisted_dict(self) -> dict[str, Any]:
        """Return JSON-compatible, non-secret configuration."""

        payload = asdict(self)
        payload.pop("simulation_validator", None)
        validator = self.simulation_validator
        payload["simulation_validator_name"] = (
            f"{validator.__module__}.{validator.__qualname__}" if validator is not None else None
        )
        return payload


def run_config_from_dict(payload: Mapping[str, Any]) -> RunConfig:
    """Recreate serializable configuration when an execution is resumed."""

    values = dict(payload)
    values.pop("simulation_validator_name", None)
    retry = values.get("retry")
    if isinstance(retry, Mapping):
        values["retry"] = RetryConfig(**retry)
    values["simulation_validator"] = None
    return RunConfig(**values)
