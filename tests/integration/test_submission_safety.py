"""Offline regression coverage of real Runtime submission and durable recovery.

Only Runtime boundary objects are faked: the public runner, submit_workload,
SQLite transactions, artifact writes, normalization, and resume remain real.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import traceback
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from urllib.parse import quote

import pytest
import qiskit_ibm_runtime
from qiskit import QuantumCircuit

from ibm_quantum_runner import (
    DuplicateExecutionError,
    EstimateRequest,
    ExecutionRecoveryError,
    ExecutionStatus,
    HardwareSubmissionError,
    QuantumRunner,
    ResultRetrievalError,
    RunConfig,
)
from ibm_quantum_runner.hardware import IBMProvider
from ibm_quantum_runner.persistence import StateStore
from tests.fixtures import FakeJob, FakePrimitiveResult, FakeProvider, FakeSamplerPub

pytestmark = pytest.mark.integration


class SimulatedProcessDeath(BaseException):
    """Bypass ordinary exception handling, as process interruption would."""


class OfflineProvider(FakeProvider):
    """Use production credential redaction without constructing any IBM service."""

    sanitize_message = IBMProvider.sanitize_message

    def __init__(self, token: str | None = None) -> None:
        super().__init__()
        self._token = token


class RuntimeBoundary:
    def __init__(self, monkeypatch: pytest.MonkeyPatch, provider: OfflineProvider) -> None:
        self.provider = provider
        self.events: list[tuple[str, int | None]] = []
        self.hook: Callable[[str, int | None], None] = lambda _stage, _child: None
        self.run_calls = 0
        self.primitive_calls = 0
        self.submitted_pubs: list[list[Any]] = []
        self.job_statuses: dict[int, list[str]] = {}
        boundary = self

        class Batch:
            def __init__(self, *, backend: Any) -> None:
                boundary.fire("batch_init", None)

            def __enter__(self) -> Batch:
                boundary.fire("batch_enter", None)
                return self

            def session_id(self) -> str:
                boundary.fire("batch_id", None)
                return "offline-batch"

            def __exit__(self, *_args: Any) -> None:
                boundary.fire("batch_exit", None)

        class Job(FakeJob):
            def __init__(self, child: int, pubs: list[Any], primitive: str) -> None:
                self.child = child
                values = (
                    [FakeSamplerPub({"0": 5}) for _ in pubs]
                    if primitive == "sampler"
                    else [
                        SimpleNamespace(data=SimpleNamespace(evs=1.0, stds=0.0), metadata={})
                        for _ in pubs
                    ]
                )
                super().__init__(
                    f"offline-job-{child}",
                    list(boundary.job_statuses.get(child, ["DONE"])),
                    FakePrimitiveResult(values),
                )

            def job_id(self) -> str:
                boundary.fire("job_id_before", self.child)
                identifier = super().job_id()
                boundary.fire("job_id_after", self.child)
                return identifier

        class Primitive:
            primitive = "sampler"

            def __init__(self, *, mode: Any, options: dict[str, Any]) -> None:
                self.child = boundary.primitive_calls
                boundary.primitive_calls += 1
                boundary.fire("primitive_init", self.child)

            def run(self, pubs: list[Any], **_kwargs: Any) -> Job:
                boundary.run_calls += 1
                boundary.submitted_pubs.append(list(pubs))
                boundary.fire("run_before", self.child)
                job = Job(self.child, pubs, self.primitive)
                boundary.provider.jobs[job._job_id] = job
                boundary.fire("run_after", self.child)
                return job

        class Estimator(Primitive):
            primitive = "estimator"

        def forbid_live_service(*_args: Any, **_kwargs: Any) -> None:
            pytest.fail("An offline safety test attempted to construct a live IBM service")

        monkeypatch.setattr(qiskit_ibm_runtime, "Batch", Batch)
        monkeypatch.setattr(qiskit_ibm_runtime, "SamplerV2", Primitive)
        monkeypatch.setattr(qiskit_ibm_runtime, "EstimatorV2", Estimator)
        monkeypatch.setattr(qiskit_ibm_runtime, "QiskitRuntimeService", forbid_live_service)

    def fire(self, stage: str, child: int | None) -> None:
        self.events.append((stage, child))
        self.hook(stage, child)


@pytest.fixture
def runtime(monkeypatch: pytest.MonkeyPatch) -> RuntimeBoundary:
    return RuntimeBoundary(monkeypatch, OfflineProvider())


def circuit(name: str) -> QuantumCircuit:
    value = QuantumCircuit(1, 1, name=name)
    value.measure(0, 0)
    return value


def config(**changes: Any) -> RunConfig:
    return RunConfig(
        **{
            "simulate_first": False,
            "shots": 5,
            "max_pubs_per_job": 1,
            "poll_interval": 0.001,
            **changes,
        }
    )


def only_execution_id(store: StateStore) -> str:
    with sqlite3.connect(store.database_path) as connection:
        rows = connection.execute("SELECT id FROM executions").fetchall()
    assert len(rows) == 1
    return str(rows[0][0])


@pytest.mark.parametrize("primitive", ["sampler", "estimator"])
def test_full_plan_is_durable_before_first_provider_effect(tmp_path, runtime, primitive) -> None:
    runner = QuantumRunner(state_dir=tmp_path, provider=runtime.provider)
    checked_effects = []

    def inspect_plan(stage: str, child: int | None) -> None:
        # A separate connection must see the whole committed plan even at Batch().
        observer = StateStore(tmp_path)
        execution_id = only_execution_id(observer)
        jobs = observer.get_jobs(execution_id)
        assert [row["circuit_indices"] for row in jobs] == [[0], [1], [2]]
        if not checked_effects:
            assert stage == "batch_init"
            assert [row["status"] for row in jobs] == ["PLANNED"] * 3
            assert all(row["ibm_job_id"] is None for row in jobs)
        if stage == "run_before":
            assert jobs[child]["status"] == "SUBMITTING"
            assert all(row["status"] == "PLANNED" for row in jobs[child + 1 :])
        checked_effects.append(stage)

    runtime.hook = inspect_plan
    if primitive == "sampler":
        execution = runner.submit([circuit(str(i)) for i in range(3)], config=config())
    else:
        requests = [EstimateRequest(QuantumCircuit(1, name=str(i)), "Z") for i in range(3)]
        execution = runner.submit_estimate(requests, config=config())
    assert runtime.run_calls == 3
    assert len(execution.ibm_job_ids) == 3
    assert checked_effects[0] == "batch_init"
    assert execution.status(refresh=False) == ExecutionStatus.SUBMITTED


@pytest.mark.parametrize(
    ("stage", "child", "durable_ids"),
    [
        ("batch_enter", None, 0),
        ("primitive_init", 0, 0),
        ("run_before", 0, 0),
        ("run_after", 0, 0),
        ("job_id_before", 0, 0),
        ("job_id_after", 0, 0),
        ("persist_before", 0, 0),
        ("persist_after", 0, 1),
        ("primitive_init", 1, 1),
    ],
)
def test_process_death_at_submission_boundaries_never_replays(
    tmp_path, monkeypatch, runtime, stage, child, durable_ids
) -> None:
    runner = QuantumRunner(state_dir=tmp_path, provider=runtime.provider)

    def crash(current_stage: str, current_child: int | None) -> None:
        if (current_stage, current_child) == (stage, child):
            raise SimulatedProcessDeath(current_stage)

    runtime.hook = crash
    original_upsert = runner._store.upsert_job

    def interrupted_upsert(execution_id: str, child_index: int, **kwargs: Any) -> None:
        if kwargs["status"] == "SUBMITTED":
            runtime.fire("persist_before", child_index)
        original_upsert(execution_id, child_index, **kwargs)
        if kwargs["status"] == "SUBMITTED":
            runtime.fire("persist_after", child_index)

    monkeypatch.setattr(runner._store, "upsert_job", interrupted_upsert)
    with pytest.raises(SimulatedProcessDeath):
        runner.submit([circuit("a"), circuit("b")], config=config())
    execution_id = only_execution_id(runner._store)
    original_jobs = runner._store.get_jobs(execution_id)
    assert len(original_jobs) == 2
    assert sum(bool(row["ibm_job_id"]) for row in original_jobs) == durable_ids
    assert runner._store.get_execution(execution_id)["status"] != "COMPLETED"
    calls_at_crash = runtime.run_calls
    runtime.hook = lambda _stage, _child: pytest.fail("Resume attempted provider submission")

    # New runner instances model repeated process restarts; only known IDs can be read.
    for _ in range(2):
        resumed = QuantumRunner(state_dir=tmp_path, provider=runtime.provider).resume(execution_id)
        assert resumed.status() == ExecutionStatus.RECOVERY_REQUIRED
        with pytest.raises(ExecutionRecoveryError):
            resumed.wait(timeout=0.05)
        with pytest.raises(ResultRetrievalError):
            resumed.result(wait=False, allow_partial=True)
        assert runtime.run_calls == calls_at_crash
        assert len(resumed._runner._store.get_jobs(execution_id)) == 2
    assert runtime.provider.retrieve_calls == durable_ids


@pytest.mark.parametrize(
    "stage",
    [
        "batch_init",
        "batch_enter",
        "batch_id",
        "primitive_init",
        "run_before",
        "run_after",
        "job_id_before",
        "job_id_after",
        "batch_exit",
    ],
)
def test_provider_exception_token_never_reaches_files_wal_logs_or_tracebacks(
    tmp_path, monkeypatch, caplog, capsys, stage
) -> None:
    secret = "offline-fake-token+/with space='quote"
    runtime = RuntimeBoundary(monkeypatch, OfflineProvider(secret))
    runner = QuantumRunner(token=secret, state_dir=tmp_path, provider=runtime.provider)

    def provider_failure(current_stage: str, _child: int | None) -> None:
        if current_stage == stage:
            raise RuntimeError(
                f"Provider rejected token {secret}; encoded={quote(secret, safe='')}"
            )

    runtime.hook = provider_failure
    # Keep an old read snapshot open so committed pages remain in the WAL.
    with sqlite3.connect(runner._store.database_path) as reader:
        reader.execute("BEGIN")
        reader.execute("SELECT COUNT(*) FROM executions").fetchone()
        with caplog.at_level("DEBUG"), pytest.raises(HardwareSubmissionError) as caught:
            runner.submit([circuit("a"), circuit("b")], config=config())
        rendered_exception = "".join(traceback.format_exception(caught.value))
        captured = capsys.readouterr()
        visible = caplog.text + captured.out + captured.err + rendered_exception
        forbidden = [secret, secret.upper(), quote(secret, safe="")]
        for value in forbidden:
            assert value not in visible
        files = [path for path in tmp_path.rglob("*") if path.is_file()]
        assert any(path.name == "state.sqlite3-wal" for path in files)
        for path in files:
            content = path.read_bytes()
            assert all(value.encode() not in content for value in forbidden), path
    assert runtime.run_calls <= 2  # Batch exit is the only failure after both submissions.
    if stage in {"run_before", "run_after", "job_id_before", "job_id_after"}:
        assert runtime.run_calls == 1


@pytest.mark.parametrize(
    "corruption",
    [
        "missing_child",
        "duplicate_child",
        "missing_circuit",
        "missing_artifact",
        "duplicate_artifact",
    ],
)
def test_inexact_child_or_artifact_coverage_cannot_complete_or_return_result(
    tmp_path, runtime, corruption
) -> None:
    runner = QuantumRunner(state_dir=tmp_path, provider=runtime.provider)
    size = 2 if corruption == "duplicate_artifact" else 1
    execution = runner.submit([circuit("a"), circuit("b")], config=config(max_pubs_per_job=size))
    execution.wait(timeout=1)
    assert execution.status(refresh=False) == ExecutionStatus.COMPLETED
    jobs = runner._store.get_jobs(execution.id)
    if corruption.endswith("artifact"):
        path = Path(jobs[0]["result_path"])
        payload = json.loads(path.read_text())
        payload["circuit_results"] = (
            []
            if corruption == "missing_artifact"
            else [payload["circuit_results"][0], payload["circuit_results"][0]]
        )
        path.write_text(json.dumps(payload))
    else:
        with sqlite3.connect(runner._store.database_path) as connection:
            if corruption == "missing_child":
                connection.execute(
                    "DELETE FROM jobs WHERE execution_id = ? AND child_index = 1", (execution.id,)
                )
            elif corruption == "duplicate_child":
                connection.execute(
                    "UPDATE jobs SET circuit_indices_json = '[0]' "
                    "WHERE execution_id = ? AND child_index = 1",
                    (execution.id,),
                )
            else:
                connection.execute(
                    "DELETE FROM circuits WHERE execution_id = ? AND circuit_index = 1",
                    (execution.id,),
                )
    assert execution.status() == ExecutionStatus.RECOVERY_REQUIRED
    for allow_partial in (False, True):
        with pytest.raises(ResultRetrievalError):
            execution.result(wait=False, allow_partial=allow_partial)
    assert not (runner._store.execution_dir(execution.id) / "result.json").exists()
    assert runtime.run_calls == (1 if size == 2 else 2)


@pytest.mark.parametrize("ambiguous_status", ["SUBMITTING", "RECOVERY_REQUIRED", "FAILED"])
def test_missing_id_ambiguity_blocks_even_confirmed_resubmission(
    tmp_path, runtime, ambiguous_status
) -> None:
    runner = QuantumRunner(state_dir=tmp_path, provider=runtime.provider)
    execution = runner.submit(circuit("a"), config=config())
    # Mimic legacy records or an interruption after an external effect but before ID storage.
    with sqlite3.connect(runner._store.database_path) as connection:
        connection.execute(
            "UPDATE jobs SET ibm_job_id = NULL, status = ? WHERE execution_id = ?",
            (ambiguous_status, execution.id),
        )
    runner._store.transition(execution.id, "RECOVERY_REQUIRED", force=True)
    with pytest.raises(ExecutionRecoveryError, match=r"ambiguous|reconcile|running"):
        runner.resubmit(execution.id, confirm_hardware_resubmission=True)
    assert runtime.run_calls == 1
    assert only_execution_id(runner._store) == execution.id


@pytest.mark.parametrize("active_status", ["QUEUED", "RUNNING", "INITIALIZING"])
def test_known_active_job_blocks_even_confirmed_resubmission(
    tmp_path, runtime, active_status
) -> None:
    runtime.job_statuses[0] = [active_status]
    runner = QuantumRunner(state_dir=tmp_path, provider=runtime.provider)
    execution = runner.submit(circuit("a"), config=config())
    runner._store.transition(execution.id, "RECOVERY_REQUIRED", force=True)
    with pytest.raises(ExecutionRecoveryError, match=r"ambiguous|reconcile|running"):
        runner.resubmit(execution.id, confirm_hardware_resubmission=True)
    assert runtime.run_calls == 1
    assert runtime.provider.retrieve_calls >= 1
    assert only_execution_id(runner._store) == execution.id


def test_explicit_known_failure_resubmits_only_failed_circuits_via_real_adapter(
    tmp_path, runtime
) -> None:
    runtime.job_statuses[0] = ["DONE"]
    runtime.job_statuses[1] = ["ERROR"]
    runner = QuantumRunner(state_dir=tmp_path, provider=runtime.provider)
    original = runner.submit([circuit("successful"), circuit("failed")], config=config())
    original.wait(timeout=1)
    assert original.status(refresh=False) == ExecutionStatus.PARTIALLY_COMPLETED
    replacement = runner.resubmit(original.id, confirm_hardware_resubmission=True)
    replacement.wait(timeout=1)
    result = replacement.result()
    assert runtime.run_calls == 3
    assert [pub.name for pub in runtime.submitted_pubs[-1]] == ["failed"]
    assert result.final_status == "COMPLETED"
    assert [item.circuit_index for item in result.circuit_results] == [0, 1]
    assert replacement.ibm_job_ids == ("offline-job-0", "offline-job-2")
    assert replacement.metadata["parent_execution_id"] == original.id
    # One original lineage has at most one replacement reservation, even if retried.
    with pytest.raises(DuplicateExecutionError):
        runner.resubmit(original.id, confirm_hardware_resubmission=True)
    assert runtime.run_calls == 3


@pytest.mark.parametrize("reservation", ["fingerprint", "replacement_lineage"])
def test_concurrent_store_create_reserves_exactly_one_execution(tmp_path, reservation) -> None:
    stores = [StateStore(tmp_path) for _ in range(8)]
    common = {
        "primitive": "sampler",
        "config": config().persisted_dict(),
        "metadata": {},
        "qiskit_version": "offline",
        "runtime_version": "offline",
    }
    parent = (
        stores[0].create_execution(**common, fingerprint="parent")
        if reservation == "replacement_lineage"
        else None
    )
    if parent is not None:
        stores[0].upsert_job(
            parent, 0, circuit_indices=[0], status="FAILED", ibm_job_id="known-failed-job"
        )
    barrier = threading.Barrier(len(stores))

    def reserve(index: int) -> tuple[str, str]:
        barrier.wait(timeout=10)
        try:
            execution_id = stores[index].create_execution(
                **common,
                fingerprint="same-workload" if parent is None else f"replacement-{index}",
                reject_duplicate=parent is None,
                parent_execution_id=parent,
            )
            return "created", execution_id
        except DuplicateExecutionError as exc:
            return "duplicate", exc.existing_execution_id

    with ThreadPoolExecutor(max_workers=len(stores)) as executor:
        outcomes = list(executor.map(reserve, range(len(stores))))
    winners = [identifier for status, identifier in outcomes if status == "created"]
    assert len(winners) == 1
    assert all(identifier == winners[0] for _status, identifier in outcomes)
    with sqlite3.connect(stores[0].database_path) as connection:
        count = connection.execute("SELECT COUNT(*) FROM executions").fetchone()[0]
    assert count == (2 if parent else 1)
    assert len(list((tmp_path / "executions").iterdir())) == count


@pytest.mark.parametrize("replacement_config", [None, "missing_validator", "simulation_disabled"])
def test_resubmission_cannot_silently_drop_original_simulation_validator(
    tmp_path, runtime, replacement_config
) -> None:
    runtime.job_statuses[0] = ["ERROR"]

    def accept_local_simulation(_result: Any) -> bool:
        return True

    original_config = config(simulate_first=True, simulation_validator=accept_local_simulation)
    runner = QuantumRunner(state_dir=tmp_path, provider=runtime.provider)
    execution = runner.submit(circuit("a"), config=original_config)
    execution.wait(timeout=1)
    assert execution.status(refresh=False) == ExecutionStatus.FAILED
    replacement = {
        None: None,
        "missing_validator": config(simulate_first=True),
        "simulation_disabled": config(simulation_validator=accept_local_simulation),
    }[replacement_config]
    with pytest.raises(ExecutionRecoveryError, match="validator"):
        runner.resubmit(
            execution.id,
            confirm_hardware_resubmission=True,
            config=replacement,
        )
    assert runtime.run_calls == 1
    assert only_execution_id(runner._store) == execution.id


@pytest.mark.parametrize("failed_stage", ["batch_init", "primitive_init"])
def test_known_unsubmitted_plan_can_be_explicitly_resubmitted(
    tmp_path, runtime, failed_stage
) -> None:
    runner = QuantumRunner(state_dir=tmp_path, provider=runtime.provider)

    def fail_before_run(stage: str, _child: int | None) -> None:
        if stage == failed_stage:
            raise RuntimeError("Known failure before any primitive.run call")

    runtime.hook = fail_before_run
    with pytest.raises(HardwareSubmissionError):
        runner.submit([circuit("a"), circuit("b")], config=config())
    execution_id = only_execution_id(runner._store)
    jobs = runner._store.get_jobs(execution_id)
    assert [row["status"] for row in jobs] == (
        ["PLANNED", "PLANNED"] if failed_stage == "batch_init" else ["NOT_SUBMITTED", "PLANNED"]
    )
    assert runtime.run_calls == 0
    runtime.hook = lambda _stage, _child: None
    replacement = runner.resubmit(execution_id, confirm_hardware_resubmission=True)
    replacement.wait(timeout=1)
    assert runtime.run_calls == 2
    assert replacement.result().final_status == "COMPLETED"
    assert replacement.metadata["parent_execution_id"] == execution_id


@pytest.mark.parametrize("when", ["before_commit", "after_commit"])
def test_single_persistence_failure_preserves_known_id_without_resubmitting(
    tmp_path, monkeypatch, runtime, when
) -> None:
    runner = QuantumRunner(state_dir=tmp_path, provider=runtime.provider)
    original_upsert = runner._store.upsert_job
    failed = False

    def fail_once(execution_id: str, child_index: int, **kwargs: Any) -> None:
        nonlocal failed
        if kwargs["status"] != "SUBMITTED" or failed:
            original_upsert(execution_id, child_index, **kwargs)
            return
        failed = True
        if when == "after_commit":
            original_upsert(execution_id, child_index, **kwargs)
        raise sqlite3.OperationalError("Injected persistence failure")

    monkeypatch.setattr(runner._store, "upsert_job", fail_once)
    with pytest.raises(HardwareSubmissionError):
        runner.submit([circuit("a"), circuit("b")], config=config())
    execution_id = only_execution_id(runner._store)
    jobs = runner._store.get_jobs(execution_id)
    assert runtime.run_calls == 1
    assert jobs[0]["ibm_job_id"] == "offline-job-0"
    assert jobs[0]["status"] == "RECOVERY_REQUIRED"
    assert jobs[1]["status"] == "PLANNED"
    resumed = QuantumRunner(state_dir=tmp_path, provider=runtime.provider).resume(execution_id)
    assert resumed.status() == ExecutionStatus.RECOVERY_REQUIRED
    assert resumed._runner._store.get_jobs(execution_id)[0]["status"] == "COMPLETED"
    with pytest.raises(ResultRetrievalError):
        resumed.result(wait=False)
    assert runtime.run_calls == 1


def test_death_after_all_durable_ids_can_recover_complete_without_submission(
    tmp_path, runtime
) -> None:
    runner = QuantumRunner(state_dir=tmp_path, provider=runtime.provider)

    def die_at_batch_exit(stage: str, _child: int | None) -> None:
        if stage == "batch_exit":
            raise SimulatedProcessDeath("Both provider IDs are already durable")

    runtime.hook = die_at_batch_exit
    with pytest.raises(SimulatedProcessDeath):
        runner.submit([circuit("a"), circuit("b")], config=config())
    execution_id = only_execution_id(runner._store)
    assert len([row for row in runner._store.get_jobs(execution_id) if row["ibm_job_id"]]) == 2
    assert runner._store.get_execution(execution_id)["status"] == "SUBMITTING"
    runtime.hook = lambda _stage, _child: pytest.fail("Recovery attempted provider submission")
    resumed = QuantumRunner(state_dir=tmp_path, provider=runtime.provider).resume(execution_id)
    assert resumed.status() == ExecutionStatus.COMPLETED
    assert [item.circuit_index for item in resumed.result().circuit_results] == [0, 1]
    assert runtime.run_calls == 2
