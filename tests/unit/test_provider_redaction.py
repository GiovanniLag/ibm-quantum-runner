from __future__ import annotations

import json
import traceback
from types import SimpleNamespace
from urllib.parse import quote, quote_plus

import pytest
from qiskit import QuantumCircuit

from ibm_quantum_runner import BackendSelectionError, ConfigurationError, RetryConfig, RunConfig
from ibm_quantum_runner.hardware.ibm import IBMProvider, sanitize_provider_data

FAKE_TOKEN = "fake-Credential/with+quotes'and\"spaces\nONLY"


def provider(service: object | None = None, *, token: str | None = FAKE_TOKEN) -> IBMProvider:
    return IBMProvider(
        token=token,
        channel="ibm_quantum_platform",
        instance=None,
        default_backend=None,
        service=service,
    )


def assert_safe_exception(exc: BaseException) -> None:
    rendered = "".join(traceback.format_exception(exc))
    assert FAKE_TOKEN not in rendered
    assert quote(FAKE_TOKEN, safe="") not in rendered
    assert "[REDACTED]" in str(exc)
    assert exc.__cause__ is None
    assert exc.__suppress_context__


@pytest.mark.parametrize(
    "rendered_token",
    [
        FAKE_TOKEN,
        FAKE_TOKEN.upper(),
        FAKE_TOKEN.lower(),
        quote(FAKE_TOKEN, safe=""),
        quote_plus(FAKE_TOKEN, safe=""),
        json.dumps(FAKE_TOKEN)[1:-1],
        repr(FAKE_TOKEN)[1:-1],
    ],
)
def test_sanitizer_redacts_encoded_and_normalized_credentials(rendered_token: str) -> None:
    assert provider().sanitize_message(f"failure: {rendered_token}") == "failure: [REDACTED]"


def test_sanitizer_handles_regex_characters_and_multiple_occurrences() -> None:
    adapter = provider(token="fake.*+[credential]?")
    assert adapter.sanitize_message("fake.*+[credential]? twice fake.*+[credential]?") == (
        "[REDACTED] twice [REDACTED]"
    )
    assert adapter.sanitize_message("fake-other-credential") == "fake-other-credential"


def test_sanitizer_does_not_require_a_configured_token() -> None:
    assert provider(token=None).sanitize_message("ordinary failure") == "ordinary failure"


def test_sanitizer_does_not_leak_a_failing_string_conversion() -> None:
    class BrokenMessage:
        def __str__(self) -> str:
            raise RuntimeError(FAKE_TOKEN)

    assert provider().sanitize_message(BrokenMessage()) == (
        "Provider message could not be displayed safely"
    )


def test_recursive_sanitizer_cleans_keys_and_nested_values_without_changing_numbers() -> None:
    payload = {
        FAKE_TOKEN: ["ok", {"details": FAKE_TOKEN.upper()}],
        "values": (1, 0.25, True, None),
        "bytes": FAKE_TOKEN.encode(),
    }
    sanitized = sanitize_provider_data(payload, provider().sanitize_message)
    assert sanitized == {
        "[REDACTED]": ["ok", {"details": "[REDACTED]"}],
        "values": [1, 0.25, True, None],
        "bytes": "b'[REDACTED]'",
    }
    assert type(sanitized["values"][0]) is int
    assert type(sanitized["values"][1]) is float
    assert type(sanitized["values"][2]) is bool
    assert FAKE_TOKEN in payload


@pytest.mark.parametrize("error_class", [RuntimeError, ConfigurationError])
def test_service_initialization_suppresses_credential_exception_chains(
    monkeypatch, error_class: type[Exception]
) -> None:
    def failing_service(**kwargs: object) -> None:
        raise error_class(f"service rejected {FAKE_TOKEN}")

    monkeypatch.setattr("qiskit_ibm_runtime.QiskitRuntimeService", failing_service)
    with pytest.raises(ConfigurationError) as captured:
        provider().service
    assert_safe_exception(captured.value)


def test_backend_retrieval_redacts_error_and_requested_name() -> None:
    class Service:
        def backend(self, name: str) -> None:
            raise RuntimeError(f"rejected {FAKE_TOKEN}")

    with pytest.raises(BackendSelectionError) as captured:
        provider(Service()).get_backend(FAKE_TOKEN)
    assert_safe_exception(captured.value)


@pytest.mark.parametrize("failed_phase", ["construction", "run"])
def test_transpilation_suppresses_credential_exception_chains(
    monkeypatch, failed_phase: str
) -> None:
    class Manager:
        def run(self, circuits: object) -> None:
            raise RuntimeError(f"transpiler rejected {FAKE_TOKEN}")

    def pass_manager(**kwargs: object) -> Manager:
        if failed_phase == "construction":
            raise RuntimeError(f"transpiler setup rejected {FAKE_TOKEN}")
        return Manager()

    monkeypatch.setattr(
        "ibm_quantum_runner.hardware.ibm.generate_preset_pass_manager", pass_manager
    )
    with pytest.raises(BackendSelectionError) as captured:
        provider().transpile([QuantumCircuit(1)], object(), RunConfig())
    assert_safe_exception(captured.value)


def test_nonoperational_backend_redacts_status_message() -> None:
    backend = SimpleNamespace(
        name="fake_backend",
        num_qubits=1,
        status=lambda: SimpleNamespace(
            operational=False, pending_jobs=0, status_msg=f"offline {FAKE_TOKEN}"
        ),
    )
    service = SimpleNamespace(backend=lambda name: backend)
    with pytest.raises(BackendSelectionError) as captured:
        provider(service).select_backend([QuantumCircuit(1)], RunConfig(backend="fake_backend"))
    assert_safe_exception(captured.value)


@pytest.mark.parametrize("explicit", [True, False])
def test_backend_selection_redacts_provider_names_and_candidate_status(
    monkeypatch, explicit
) -> None:
    backend = SimpleNamespace(
        name=f"fake-{FAKE_TOKEN}",
        num_qubits=1,
        simulator=False,
        status=lambda: SimpleNamespace(
            operational=True, pending_jobs=0, status_msg=f"active {FAKE_TOKEN}"
        ),
    )
    service = SimpleNamespace(backend=lambda name: backend, backends=lambda **kwargs: [backend])
    adapter = provider(service)
    monkeypatch.setattr(adapter, "transpile", lambda circuits, backend, config: list(circuits))
    selected = adapter.select_backend(
        [QuantumCircuit(1)],
        RunConfig(backend="fake_backend" if explicit else None, auto_select_backend=not explicit),
    )
    assert selected.name == "fake-[REDACTED]"
    assert selected.candidates[0]["name"] == "fake-[REDACTED]"
    if not explicit:
        assert selected.candidates[0]["status_message"] == "active [REDACTED]"
    assert selected.backend is backend


def test_backend_discovery_fallback_redacts_failures() -> None:
    class Service:
        def backends(self, **kwargs: object) -> None:
            if kwargs:
                raise TypeError("legacy fake service")
            raise RuntimeError(f"fallback discovery failed {FAKE_TOKEN}")

    with pytest.raises(BackendSelectionError) as captured:
        provider(Service()).select_backend([QuantumCircuit(1)], RunConfig(auto_select_backend=True))
    assert_safe_exception(captured.value)


def test_backend_attribute_failure_is_inside_redaction_boundary() -> None:
    class Backend:
        @property
        def num_qubits(self) -> int:
            raise RuntimeError(f"capability retrieval failed {FAKE_TOKEN}")

        def status(self) -> object:
            return SimpleNamespace(operational=True, pending_jobs=0, status_msg="active")

    service = SimpleNamespace(backend=lambda name: Backend())
    with pytest.raises(BackendSelectionError) as captured:
        provider(service).select_backend([QuantumCircuit(1)], RunConfig(backend="fake_backend"))
    assert_safe_exception(captured.value)


def test_retrieval_retries_log_only_safe_details(monkeypatch, caplog) -> None:
    attempts = 0
    retry_history = []

    class Service:
        def job(self, job_id: str) -> None:
            nonlocal attempts
            attempts += 1
            raise ConnectionError(f"temporary provider failure {FAKE_TOKEN}")

    monkeypatch.setattr("ibm_quantum_runner.hardware.ibm.time.sleep", lambda _: None)
    with pytest.raises(RuntimeError) as captured:
        provider(Service()).retrieve_job(
            FAKE_TOKEN,
            RetryConfig(max_attempts=2, initial_delay=0, max_delay=0, jitter=0),
            on_retry=retry_history.append,
        )
    assert attempts == 2
    assert len(retry_history) == 1
    assert_safe_exception(captured.value)
    assert FAKE_TOKEN not in repr(retry_history)
    assert FAKE_TOKEN not in repr([record.__dict__ for record in caplog.records])
    assert all(record.exc_info is None for record in caplog.records)
