# IBM Quantum Runner

`ibm-quantum-runner` runs Qiskit circuits locally with Aer or on IBM Quantum
hardware through Runtime Sampler V2 and Estimator V2. It handles validation,
transpilation, submission, polling, and normalized results. Hardware executions
are recorded in SQLite with circuit and result artifacts so another process can
resume monitoring without submitting the work again.

Use it when your application already produces `QuantumCircuit` objects and
needs to manage their execution. Circuit construction, datasets, model inference,
and scientific post-processing stay in your application. The package is alpha.

## Install from this repository

Use [uv](https://docs.astral.sh/uv/) for dependency and environment management.
Python range is 3.11–3.14; this checkout pins Python 3.12.11 in
[`.python-version`](.python-version).

```bash
git clone https://github.com/GiovanniLag/ibm-quantum-runner.git
cd ibm-quantum-runner
uv sync --locked
```

Run the commands below from the repository root. `uv run` uses the project
environment without manual activation. To use this checkout as an editable
dependency, run this from another uv project, adjusting the path:

```bash
uv add --editable ../ibm-quantum-runner
```

The [manifest](pyproject.toml) declares Qiskit `>=2.3,<2.6`, Aer `>=0.17.2,<0.18`,
and IBM Runtime `>=0.45,<0.49`; [uv.lock](uv.lock) records resolved versions.
Recorded offline verification covers Python 3.12.11 with the locked and minimum
dependency versions. It does not establish compatibility across every Python
version or validate live IBM behavior; see the
[verification notes](docs/safety-and-limitations.md#verification).

## Quick start: a local Bell circuit

No IBM account, token, or hardware access is needed. Save this as `quickstart.py`
in the repository root:

```python
from qiskit import QuantumCircuit

from ibm_quantum_runner import QuantumRunner, RunConfig

circuit = QuantumCircuit(2)
circuit.h(0)
circuit.cx(0, 1)
circuit.measure_all()

runner = QuantumRunner(state_dir=".quantum-runs")
config = RunConfig(shots=1_000)

report = runner.validate(circuit, config=config, include_hardware=False)
if not report.is_valid:
    raise ValueError([(issue.code, issue.message) for issue in report.errors])

simulation = runner.simulate(circuit, config=config)
print(simulation.counts)
```

```bash
uv run python quickstart.py
```

The result contains `00` and `11` counts totaling 1,000, roughly half in each
state. The checked-in [simulation example](examples/simulate_only.py) is also
runnable with `uv run python examples/simulate_only.py`.

`simulate()` and `simulate_estimate()` use local, ideal Aer simulation only.
Constructing a runner creates its local state store, but standalone simulation
returns a `SimulationResult` without creating a resumable hardware execution.
Sampler circuits must be fully bound and contain measurements and a classical
register.

Use `include_hardware=False` for local-only `validate()` and `validate_estimate()`.
Their default, `True`, also authenticates with IBM, selects a backend, transpiles,
and checks backend constraints. Validation alone does not submit a job.

## Configure IBM hardware access

Hardware methods need an IBM Quantum token and access to a suitable backend.
Copy [`.env.example`](.env.example) to `.env` and fill in your own values locally,
or set the variables in the process environment. Never commit real credentials.

| Variable | Purpose and default |
| --- | --- |
| `QISKIT_IBM_TOKEN` | Required for hardware operations; no default |
| `QISKIT_IBM_CHANNEL` | Runtime channel; defaults to `ibm_quantum_platform` |
| `QISKIT_IBM_INSTANCE` | Optional instance passed to Runtime; requirements depend on your IBM account |
| `QISKIT_IBM_BACKEND` | Default backend name; no default |
| `QUANTUM_RUNNER_STATE_DIR` | State directory; defaults to `.quantum-runs` |
| `QUANTUM_RUNNER_LOG_LEVEL` | Optional package log level; handlers remain application-controlled |

Load environment configuration explicitly:

```python
from ibm_quantum_runner import QuantumRunner

runner = QuantumRunner.from_env()                   # process environment only
# Or opt in to loading a local file:
runner = QuantumRunner.from_env(dotenv_path=".env")
```

Non-`None` arguments to `from_env()` take precedence over environment values.
A requested `.env` file fills in variables without overriding the existing
process environment. `QuantumRunner(...)` uses constructor arguments and
defaults; it does not read environment variables. Neither path silently loads
Qiskit's globally saved credentials. The configured token stays in memory and
is redacted from provider messages and metadata before persistence or logging.
Keep secrets out of circuit and user-supplied metadata as well.

`RunConfig.backend` overrides the runner's default backend, including one loaded
from `QISKIT_IBM_BACKEND`. If neither is set, selection fails unless you explicitly
use `RunConfig(auto_select_backend=True)`. Automatic selection filters for
operational, non-simulator backends with enough qubits, then ranks compatible
candidates by queue size and backend name. The validation report records the
selection reason; queue length is not a hardware-quality metric.

## Submit a Sampler job and resume it later

This example submits real hardware work and can consume IBM quota. Configure
access and a backend first. Save it as `submit_bell.py` and run it with
`uv run python submit_bell.py`:

```python
from qiskit import QuantumCircuit

from ibm_quantum_runner import QuantumRunner, RunConfig

circuit = QuantumCircuit(2)
circuit.h(0)
circuit.cx(0, 1)
circuit.measure_all()

runner = QuantumRunner.from_env(dotenv_path=".env")
execution = runner.submit(circuit, config=RunConfig(shots=4_000))
print("Local execution:", execution.id)
print("IBM jobs:", execution.ibm_job_ids)
```

`submit()` validates and, by default, simulates before submitting. On success it
returns after all child job IDs are durable, without waiting for completion.
Save the local execution ID printed above. Using the same state directory, even
in a new process, resume with the checked-in script:

```bash
uv run python examples/resume.py YOUR_EXECUTION_ID
```

Replace `YOUR_EXECUTION_ID` with that ID. The script loads `.env`, waits for the
recorded jobs, and prints the normalized result. `resume()` restores the handle;
`status()`, `wait()`, and `result()` monitor its jobs.

For blocking execution, `runner.run(circuit, config=...)` submits and waits.
`execution.result()` also waits by default. Set `RunConfig(wait_timeout=...)` or
pass `timeout=` to `wait()` or `result()` to bound waiting. A timeout does not
cancel the IBM job; it can be resumed later. `execution.cancel()` requests
cancellation of known nonterminal jobs. `execution.events()` returns the durable
lifecycle history.

### Recovery and resubmission

Resume never submits a replacement. If IBM accepts work but the process stops
before its job ID is saved, the submission is ambiguous. `RECOVERY_REQUIRED`
signals that submission or monitoring needs reconciliation. Repeating `submit()`
is not a safe way to recover it.

- Retry transient read or result failures with `execution.status()`. Known job
  IDs can also rebuild missing or malformed result artifacts.
- Active jobs and unknown-ID submissions block resubmission, even with explicit
  confirmation. Manual reconciliation with IBM may be required.
- `recover_ibm_job(job_id)` finds an existing local execution, or imports an
  untracked Sampler job if IBM exposes its circuits. Untracked Estimator recovery
  requires the original local execution and observable/parameter artifacts.

After reconciliation, an eligible failed, cancelled, or partially completed
execution can be replaced explicitly. This creates new hardware usage:

```python
replacement = runner.resubmit(
    failed_execution_id,
    confirm_hardware_resubmission=True,
)
```

Set `failed_execution_id` to the original local ID. The replacement preserves
verified successful child results and submits only known-failed, cancelled, or
definitely unsubmitted work. It fences the original unsent children against
concurrent submission. Resume an existing replacement rather than creating
another one; terminal preflight-only replacement failures can be retried
explicitly from the original execution. The [lifecycle guide](docs/lifecycle.md)
explains the durable submission boundaries and recovery states.

## Estimator: expectation values and parameter sweeps

Estimator inputs pair an unmeasured circuit with observables and, for a
parameterized circuit, parameter values. This example runs locally:

```python
from qiskit import QuantumCircuit

from ibm_quantum_runner import EstimateRequest, QuantumRunner, RunConfig

circuit = QuantumCircuit(2)
circuit.x(1)

request = EstimateRequest(circuit, observables=["IZ", "ZI"])
runner = QuantumRunner()
simulation = runner.simulate_estimate(request, config=RunConfig(precision=0.01))
print(simulation.expectation_values)  # approximately [1, -1]
```

Save it as `estimate_local.py` and run `uv run python estimate_local.py`. For
hardware, configure a runner with `from_env()` and use `submit_estimate(request,
config=...)` to submit without waiting, or `estimate(request, config=...)` to
submit and wait. [submit_estimator.py](examples/submit_estimator.py) is a complete
hardware example.

Observable structure determines the output:

- A string, coefficient mapping, or `SparsePauliOp` represents one operator.
  `{"IZ": 1.0, "ZI": 1.0}` explicitly requests the sum, producing one value.
- A sequence such as `["IZ", "ZI"]` requests two independent values in that
  order. Rectangular nested sequences and NumPy arrays retain their shape.
- Qiskit's rightmost Pauli character acts on qubit 0: `"IZ"` is Z on qubit 0.
  Each operator is mapped through the transpiled layout before hardware submission.

Parameter arrays follow `list(circuit.parameters)` order. Named parameter
mappings avoid depending on an application-specific ordering. For `N` samples,
`P` parameters, and `M` independent observables, parameter values shaped
`(N, 1, P)` and observables shaped `(M,)` broadcast to results shaped `(N, M)`.
See IBM's [primitive input/output guide](https://quantum.cloud.ibm.com/docs/en/guides/primitive-input-output)
for broadcasting rules. The runner partitions requests, not parameter rows
within a request; split large sweeps in the caller when needed.

`RunConfig.precision` must be positive. If omitted, Estimator uses the target
`1 / sqrt(shots)`; `shots` is not passed as an exact Estimator shot count. Local
Estimator simulation also uses that positive precision and Aer precision
sampling. It is not an exact zero-precision reference.

## Simulation acceptance checks

Supply a callback returning `bool` or `SimulationCheck` to apply a domain-specific
check before hardware submission. For the Bell circuit above:

```python
from ibm_quantum_runner import RunConfig, SimulationCheck


def bell_check(simulation):
    counts = simulation.counts or {}
    accepted = counts.get("00", 0) + counts.get("11", 0)
    return SimulationCheck(accepted > 900, "Bell parity preflight")


config = RunConfig(
    shots=1_000,
    simulate_first=True,
    stop_on_simulation_failure=True,
    simulation_validator=bell_check,
)
```

Pass this `config` to `submit()` or `run()`. A failed check stops hardware
submission. Standalone simulation returns the check in `validator_check` for
the caller to inspect. Callback code is not serialized, only its qualified name.
Resuming submitted work does not rerun it. Resubmitting a run that used a callback
requires a fresh `RunConfig` with the callback and `simulate_first=True`.

## Results, batches, and local state

For a single circuit, read `result.counts` or `result.expectation_values`.
For multiple circuits, use `result.circuit_results`, ordered by original circuit
index. Sampler results include combined and per-register counts; Estimator results
include expectation values and standard deviations. `result.to_dict()` exposes
the normalized, JSON-compatible representation. `raw_provider_result` is an
in-process escape hatch and is not persisted; previous-process raw objects are
unavailable after a restart.

Pass a sequence of circuits to `submit()` or requests to `submit_estimate()`.
Work is partitioned by `RunConfig.max_pubs_per_job` (default 100) and the runner's
10-million-execution Sampler job limit. Multiple child jobs use a Runtime `Batch`.
Each child ID and successful result is persisted independently. Full completion
requires exact circuit coverage and verified artifacts. For a terminal partial
execution, `result(allow_partial=True)` exposes successful results alongside
failed/cancelled entries; inspect each circuit's `status` and `error` before
using its values.

Keep the entire state directory. It contains `state.sqlite3` plus per-execution
QPY circuits, JSON validation/simulation/results, and optional NPZ parameter
arrays under `executions/<execution-id>/`. Artifacts are versioned and do not use
arbitrary pickle. Estimator input schema 2 preserves observable shape; legacy schema-1
sums retain their original meaning when loaded.

The default `.quantum-runs` path is relative to the current working directory.
Use the same `state_dir=` or `QUANTUM_RUNNER_STATE_DIR` through `from_env()` after
a restart. SQLite uses transactions and WAL mode; duplicate fingerprint checks,
execution reservations, and replacement fencing are transactional.

## Errors and retries

Library execution errors derive from `QuantumRunnerError`; invalid configuration
values can raise `ValueError`. `ValidationReport` contains structured issues;
execution errors distinguish submission, recovery, timeout, and result retrieval.
See the [public exception types](src/ibm_quantum_runner/exceptions.py).

`RetryConfig` controls bounded exponential backoff for plausibly transient,
idempotent reads, polling, and result retrieval. Submission and cancellation
are never retried automatically. The package installs a `NullHandler` and does
not configure the root logger; applications choose their logging handlers.

## Development and reference

From the repository root:

```bash
uv sync --locked
uv run ruff format --check .
uv run ruff check .
uv run mypy
uv run pytest
```

Use `uv run ruff format .` to apply formatting. Unit and integration tests use
local Aer and fake provider objects. They must not contact IBM, require real
credentials, or consume hardware quota. Live hardware checks are separate manual
operations requiring explicit authorization.

- [Architecture](docs/architecture.md): orchestration, provider adapters, persistence,
  and result normalization
- [Execution lifecycle](docs/lifecycle.md): states, interruption boundaries, and recovery
- [Run and retry configuration](src/ibm_quantum_runner/config.py): complete defaults
- [Examples](examples): local simulation, hardware submission, and resume scripts
- [Changelog](CHANGELOG.md): version history

Copyright 2026 Giovanni Laganà.

Licensed under [Apache 2.0](LICENSE).
