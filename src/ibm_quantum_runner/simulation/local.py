"""Aer-backed local Sampler and Estimator simulation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from qiskit import QuantumCircuit
from qiskit_aer.primitives import EstimatorV2 as AerEstimator
from qiskit_aer.primitives import SamplerV2 as AerSampler

from ..config import EstimateRequest, RunConfig
from ..persistence.artifacts import circuit_summary, normalize_observables
from ..results.normalize import normalize_estimator_result, normalize_sampler_result
from .models import SimulationCheck, SimulationResult


def _estimator_pub(request: EstimateRequest) -> tuple[Any, ...]:
    observable = normalize_observables(request.observables)
    if request.parameter_values is None:
        return (request.circuit, observable)
    return (request.circuit, observable, request.parameter_values)


def _run_validator(result: SimulationResult, config: RunConfig) -> SimulationResult:
    validator = config.simulation_validator
    if validator is None:
        return result
    outcome = validator(result)
    if isinstance(outcome, SimulationCheck):
        check = outcome
    elif isinstance(outcome, bool):
        check = SimulationCheck(outcome, "User simulation validator returned a boolean")
    else:
        raise TypeError("simulation_validator must return bool or SimulationCheck")
    return SimulationResult(
        primitive=result.primitive,
        circuit_results=result.circuit_results,
        validator_check=check,
        metadata=result.metadata,
    )


def simulate_sampler(circuits: Sequence[QuantumCircuit], config: RunConfig) -> SimulationResult:
    sampler = AerSampler(default_shots=config.shots, seed=config.simulation_seed)
    raw = sampler.run(circuits, shots=config.shots).result()
    records = [circuit_summary(circuit, index) for index, circuit in enumerate(circuits)]
    result = SimulationResult(
        primitive="sampler",
        circuit_results=normalize_sampler_result(raw, records),
        metadata={"engine": "qiskit-aer", "ideal": True, "shots": config.shots},
    )
    return _run_validator(result, config)


def simulate_estimator(requests: Sequence[EstimateRequest], config: RunConfig) -> SimulationResult:
    precision = config.estimator_precision()
    estimator = AerEstimator(
        options={
            "default_precision": precision,
            "run_options": {"seed_simulator": config.simulation_seed},
        }
    )
    raw = estimator.run(
        [_estimator_pub(request) for request in requests], precision=precision
    ).result()
    records = [circuit_summary(request.circuit, index) for index, request in enumerate(requests)]
    result = SimulationResult(
        primitive="estimator",
        circuit_results=normalize_estimator_result(raw, records, precision=precision),
        metadata={"engine": "qiskit-aer", "ideal": True, "precision": precision},
    )
    return _run_validator(result, config)
