"""Public API for persistent IBM Quantum circuit execution."""

from .config import EstimateRequest, RetryConfig, RunConfig
from .exceptions import (
    BackendSelectionError,
    CircuitValidationError,
    ConfigurationError,
    DuplicateExecutionError,
    ExecutionRecoveryError,
    ExecutionTimeoutError,
    HardwareResubmissionRequiredError,
    HardwareSubmissionError,
    QuantumRunnerError,
    ResultRetrievalError,
    SimulationError,
    StateStoreError,
)
from .execution import Execution, ExecutionStatus
from .results import (
    EstimatorCircuitResult,
    ExecutionResult,
    HardwareResult,
    SamplerCircuitResult,
)
from .runner import QuantumRunner
from .simulation import SimulationCheck, SimulationResult
from .validation import ValidationIssue, ValidationReport

__version__ = "0.1.0"

__all__ = [
    "BackendSelectionError",
    "CircuitValidationError",
    "ConfigurationError",
    "DuplicateExecutionError",
    "EstimateRequest",
    "EstimatorCircuitResult",
    "Execution",
    "ExecutionRecoveryError",
    "ExecutionResult",
    "ExecutionStatus",
    "ExecutionTimeoutError",
    "HardwareResubmissionRequiredError",
    "HardwareResult",
    "HardwareSubmissionError",
    "QuantumRunner",
    "QuantumRunnerError",
    "ResultRetrievalError",
    "RetryConfig",
    "RunConfig",
    "SamplerCircuitResult",
    "SimulationCheck",
    "SimulationError",
    "SimulationResult",
    "StateStoreError",
    "ValidationIssue",
    "ValidationReport",
]
