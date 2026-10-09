from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, ClassVar

from ibm_quantum_runner.hardware import BackendSelection, SubmittedJob
from ibm_quantum_runner.persistence.artifacts import utc_now


class FakeBitArray:
    def __init__(self, counts: dict[str, int]) -> None:
        self._counts = counts

    def get_counts(self) -> dict[str, int]:
        return self._counts


class FakeSamplerPub:
    def __init__(self, counts: dict[str, int]) -> None:
        self.data = {"meas": FakeBitArray(counts)}
        self.metadata = {"shots": sum(counts.values())}

    def join_data(self) -> FakeBitArray:
        return self.data["meas"]


class FakePrimitiveResult(list[Any]):
    metadata: ClassVar[dict[str, str]] = {"provider": "fake"}


class FakeJob:
    def __init__(
        self,
        job_id: str,
        statuses: list[str],
        result: Any | None = None,
        error: str = "fake failure",
        inputs: dict[str, Any] | None = None,
        program_id: str = "sampler",
    ) -> None:
        self._job_id = job_id
        self._statuses = statuses
        self._last_status = statuses[-1]
        self._result = result
        self._error = error
        self.inputs = inputs or {}
        self.program_id = program_id
        self.cancel_calls = 0

    def job_id(self) -> str:
        return self._job_id

    def status(self) -> str:
        if self._statuses:
            self._last_status = self._statuses.pop(0)
        return self._last_status

    def result(self) -> Any:
        if self._result is None:
            raise RuntimeError("No fake result")
        return self._result

    def error_message(self) -> str:
        return self._error

    def cancel(self) -> None:
        self.cancel_calls += 1
        self._statuses = ["CANCELLED"]

    def backend(self) -> FakeBackend:
        return FakeBackend()


@dataclass
class FakeBackend:
    name: str = "fake_backend"
    num_qubits: int = 127

    def status(self) -> SimpleNamespace:
        return SimpleNamespace(operational=True, pending_jobs=3, status_msg="active")


class FakeProvider:
    def __init__(self, jobs: dict[str, FakeJob] | None = None) -> None:
        self.jobs = jobs or {}
        self.default_backend = "fake_backend"
        self.backend = FakeBackend()
        self.retrieve_calls = 0

    @staticmethod
    def sanitize_message(value: object) -> str:
        return str(value)

    def select_backend(self, circuits: Any, config: Any) -> BackendSelection:
        return BackendSelection(
            self.backend,
            self.backend.name,
            "fake selection",
            ({"name": self.backend.name, "pending_jobs": 3},),
        )

    def transpile(self, circuits: Any, backend: Any, config: Any) -> list[Any]:
        return list(circuits)

    def retrieve_job(self, job_id: str, retry: Any, *, on_retry: Any = None) -> FakeJob:
        self.retrieve_calls += 1
        return self.jobs[job_id]


def fake_submitter(jobs: dict[str, FakeJob]):
    def submit_workload(**kwargs: Any) -> tuple[SubmittedJob, ...]:
        execution_id = kwargs["execution_id"]
        transpiled = kwargs["transpiled"]
        store = kwargs["store"]
        config = kwargs["config"]
        primitive = kwargs["primitive"]
        from ibm_quantum_runner.hardware import partition_indices

        selected = kwargs.get("selected_indices")
        selected = list(selected) if selected is not None else list(range(len(transpiled)))
        offset = int(kwargs.get("child_index_offset", 0))
        chunks = [
            tuple(selected[index] for index in chunk)
            for chunk in partition_indices(len(selected), primitive, config)
        ]
        submitted = []
        for local_child_index, indices in enumerate(chunks):
            child_index = offset + local_child_index
            job_id = f"fake-job-{child_index}"
            job = jobs[job_id]
            store.upsert_job(
                execution_id,
                child_index,
                circuit_indices=indices,
                status="SUBMITTED",
                ibm_job_id=job_id,
                raw_status="QUEUED",
                backend_name="fake_backend",
                submitted_at=utc_now(),
            )
            submitted.append(SubmittedJob(child_index, job_id, indices, job))
        return tuple(submitted)

    return submit_workload
