from __future__ import annotations

import pytest

from ibm_quantum_runner.exceptions import InvalidStateTransitionError
from ibm_quantum_runner.persistence import StateStore


def test_state_store_enforces_lifecycle(tmp_path) -> None:
    store = StateStore(tmp_path)
    execution_id = store.create_execution(
        primitive="sampler",
        config={},
        fingerprint="fingerprint",
        metadata={},
        qiskit_version="test",
        runtime_version="test",
    )
    store.transition(execution_id, "VALIDATING")
    with pytest.raises(InvalidStateTransitionError):
        store.transition(execution_id, "COMPLETED")


def test_state_store_uses_wal_and_persists_events(tmp_path) -> None:
    store = StateStore(tmp_path)
    execution_id = store.create_execution(
        primitive="sampler",
        config={},
        fingerprint="fingerprint",
        metadata={"safe": True},
        qiskit_version="test",
        runtime_version="test",
    )
    assert store.database_path.exists()
    assert store.events(execution_id)[0]["details"] == {"safe": True}
