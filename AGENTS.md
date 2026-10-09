# Agent instructions

IBM Quantum Runner (`ibm-quantum-runner`, import `ibm_quantum_runner`) is a
reusable library for local Aer simulation and persistent IBM Runtime execution.
Keep circuit construction, datasets, model logic, and scientific post-processing
in the caller. This file covers both using the library and maintaining it.

## Start here

1. Read the [README](README.md) for runnable examples and configuration. Use the
   public imports from `ibm_quantum_runner`; do not integrate through private
   runner methods or manage Runtime services/primitives in application code.
2. Establish whether the task is local simulation, new hardware work, or recovery
   of an existing execution. Default to local-only work when this is unspecified.
3. Use uv exclusively. From the repository root, run `uv sync --locked`, then
   `uv run python examples/simulate_only.py` for a credential-free smoke check.
   From another uv project, use `uv add --editable ../ibm-quantum-runner`, adjusting
   the checkout path. There is no general-purpose CLI; use the Python API.
4. Before hardware work, confirm the inputs, backend or approved automatic
   selection, shots/precision, and persistent state directory. Obtain explicit
   user authorization for the workload and its potential IBM quota/cost.
   Credentials or a request to test the repository are not submission approval.

## Choose the API and verify the inputs

| Goal | API | Input / output |
| --- | --- | --- |
| Local Sampler | `validate(..., include_hardware=False)`, `simulate(...)` | Fully bound, measured `QuantumCircuit` with a classical register (or sequence); `SimulationResult` with counts |
| Local Estimator | `validate_estimate(..., include_hardware=False)`, `simulate_estimate(...)` | `EstimateRequest` (or sequence); `SimulationResult` with expectation values |
| Submit without waiting | `submit(...)`, `submit_estimate(...)` | Same inputs; persistent `Execution` handle |
| Submit and wait | `run(...)`, `estimate(...)` | `Execution` handle after waiting; call `.result()` for `ExecutionResult` |
| Continue existing work | `resume(execution_id)` | Same state directory; `Execution` handle, no submission |

- Import `QuantumRunner`, `RunConfig`, and, for Estimator, `EstimateRequest`.
  Inspect `ValidationReport.is_valid` and its structured `errors` before proceeding.
  **Both validation methods default to `include_hardware=True`**, which contacts
  IBM for backend-aware checks. Set it to `False` for offline work.
- Estimator circuits must be unmeasured. `EstimateRequest(circuit, ["IZ", "ZI"])`
  asks for two independent values in that order; `{"IZ": 1.0, "ZI": 1.0}` or one
  `SparsePauliOp` asks for one summed operator. Never collapse an observable list
  into a sum. Rectangular nested sequences and NumPy arrays preserve their shape.
- Pauli labels use Qiskit order: the rightmost character acts on qubit 0. Supply
  observables matching the logical circuit's width; the hardware adapter applies
  the transpiled layout.
- Parameter columns follow `list(circuit.parameters)`, not assumed qubit or
  insertion order. Named mappings must cover every circuit parameter exactly.
  For `N` samples,
  `P` parameters, and `M` observables, shapes `(N, 1, P)` and `(M,)` yield `(N, M)`.
  Check the expected shape before using results; the runner splits requests,
  not parameter rows inside a request.
- `RunConfig.precision` must be positive when set; if omitted, Estimator uses
  `1 / sqrt(shots)`.
  Estimator `shots` is not an exact shot count. Local Estimator uses positive
  precision sampling, not an exact zero-precision reference.
- For one input, use `result.counts` or `result.expectation_values`; for multiple
  inputs, use `result.circuit_results` in original input order. Use `to_dict()`
  for normalized output. Estimator standard deviations are per-circuit fields.
  Local simulation does not create a resumable hardware execution.

See the README's [local example](README.md#quick-start-a-local-bell-circuit),
[Estimator examples](README.md#estimator-expectation-values-and-parameter-sweeps),
and [result contract](README.md#results-batches-and-local-state).

## Hardware, credentials, and recovery

- `QuantumRunner(...)` uses constructor arguments, not environment variables.
  Use `QuantumRunner.from_env()` explicitly, or opt into a local `.env` with
  `from_env(dotenv_path=".env")`. See [`.env.example`](.env.example) and the
  [configuration guide](README.md#configure-ibm-hardware-access). Never silently
  use globally saved Qiskit credentials. Never log, display, serialize, persist,
  or commit tokens, including in user metadata, commands, or reports.
- Hardware examples (`submit_sampler.py`, `submit_estimator.py`) submit real work;
  do not run them as smoke tests. Keep simulation preflight enabled unless the
  user explicitly requests otherwise. Backend auto-selection is opt-in via
  `RunConfig(auto_select_backend=True)`; a short queue does not establish quality.
- On submission, record `execution.id`, `execution.ibm_job_ids`, backend, and
  state directory. Keep the entire directory, including SQLite and QPY/JSON/NPZ
  artifacts. The default `.quantum-runs` is relative to the working directory;
  prefer an explicit stable path across processes. Do not delete or edit state
  to bypass duplicate or recovery safeguards.
- `resume()` only restores a handle. `status()` refreshes known jobs; use
  `status(refresh=False)` and `events()` to inspect cached state without provider
  reads. `wait()` and `result()` can block; set `timeout=` or `RunConfig.wait_timeout`.
  `wait()`, `run()`, and `estimate()` can also return after failure or cancellation;
  inspect status and call `.result()` before treating an execution as successful.
  A timeout does not cancel a job. Cancel only when authorized; submission and
  cancellation must never be retried automatically.
- Never repeat `submit()` after a failure or interruption to "see if it works."
  `RECOVERY_REQUIRED` can mean exhausted read retries or an ambiguous submission.
  Refresh known IDs with `status()`; a missing ID after crossing the submission
  boundary is not proof that IBM did not accept the work. Preserve evidence and
  report the ambiguity for reconciliation.
- Replacement work requires user authorization **and**
  `resubmit(execution_id, confirm_hardware_resubmission=True)`. The flag is a code
  safeguard, not permission. Active or unknown-ID submissions still block it.
  Resume an existing replacement instead of creating another. Never use
  `allow_duplicate_submission=True` as a recovery shortcut. A run that used a
  simulation callback needs a fresh `RunConfig` with that callback and
  `simulate_first=True` for resubmission.
- `recover_ibm_job(job_id)` can import an untracked Sampler job only if IBM exposes
  its circuits. Untracked Estimator jobs require the original local execution
  and observable/parameter artifacts. Raw provider results are not persisted.
- For terminal partial executions, use `result(allow_partial=True)` only when
  partial output is acceptable, and inspect every circuit's `status` and `error`.
  Do not report submission, a timeout, or partial completion as full success.

Read [the lifecycle guide](docs/lifecycle.md) before changing recovery behavior.

## Maintaining the repository

Module ownership under `src/ibm_quantum_runner/`:

- `config.py`: public run/retry and Estimator request types.
- `runner.py`: orchestration and recovery policy.
- `execution.py`: persistent public handle and lifecycle names.
- `validation/`: static/backend-aware structured preflight.
- `simulation/`: local Qiskit Aer execution only.
- `hardware/`: all IBM Runtime/provider types and API compatibility.
- `persistence/`: SQLite and versioned non-pickle artifacts.
- `results/`: stable normalization independent of IBM object shapes.

Preserve these boundaries and keep application/model logic outside this package.
See [architecture](docs/architecture.md) and [configuration types](src/ibm_quantum_runner/config.py).

Python 3.11 through 3.14 is declared; `.python-version` pins 3.12.11. Run these
checks from the repository root:

```bash
uv sync --locked
uv run ruff format --check .
uv run ruff check .
uv run mypy
uv run pytest
uv run python examples/simulate_only.py
```

Use `uv run ruff format .` to apply formatting. Do not add Poetry, Pipenv, Conda,
`requirements.txt`, or `setup.py`. When provider versions change, stay within
`pyproject.toml` ranges, consult official IBM/Qiskit documentation, update
`uv.lock` deliberately, and add an adapter-level regression test.

Automated tests must never contact IBM, consume hardware quota, use real
credentials, or call a live cancellation endpoint. Inject fake services,
backends, jobs, primitives, or provider adapters. Cover happy paths and
interruption boundaries: durable job IDs, bounded read retries, no submission
on resume, partial-result survival, terminal aggregation, secret redaction,
invalid transitions, replacement races, observable shape/layout, and artifact
round-trips (including legacy observable sums).

In your handoff, report commands and outcomes, what was not tested, and any
remaining blocker. For hardware tasks, include IDs and state location without
secrets. This is an alpha library: offline tests and ideal simulation do not
prove live IBM behavior, hardware accuracy, or downstream scientific equivalence.
