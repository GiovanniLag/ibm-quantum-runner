from __future__ import annotations

import pytest

from ibm_quantum_runner import RetryConfig, RunConfig
from ibm_quantum_runner.config import run_config_from_dict


def test_run_config_round_trip_excludes_callback() -> None:
    def callback(result: object) -> bool:
        return True

    config = RunConfig(shots=500, simulation_validator=callback, retry=RetryConfig(max_attempts=2))

    payload = config.persisted_dict()
    restored = run_config_from_dict(payload)

    assert "simulation_validator" not in payload
    assert payload["simulation_validator_name"].endswith(".callback")
    assert restored.shots == 500
    assert restored.retry.max_attempts == 2
    assert restored.simulation_validator is None


@pytest.mark.parametrize("shots", [0, -1, True, 1.5])
def test_shots_must_be_positive_integer(shots: object) -> None:
    with pytest.raises(ValueError, match="shots"):
        RunConfig(shots=shots)  # type: ignore[arg-type]


def test_estimator_precision_defaults_from_shots() -> None:
    assert RunConfig(shots=10_000).estimator_precision() == pytest.approx(0.01)
    assert RunConfig(shots=10_000, precision=0.2).estimator_precision() == 0.2
