# Safety and compatibility limits

This alpha library uses the safeguards below to manage persistent IBM Runtime
executions. Offline tests do not prove live service behavior or the scientific
validity of a downstream application.

## Execution safeguards

1. The complete child plan is committed atomically before any Runtime batch or
   primitive submission. Each child crosses a durable, once-only claim before
   `run`. Crashes after this point without a durable job ID remain ambiguous;
   neither resume nor confirmed resubmit blindly repeats them.
2. Completion requires exact original-circuit coverage, matching circuit
   identities, valid result payloads and durable artifacts. Known provider job
   IDs can rebuild missing/malformed artifacts without sending new jobs.
3. Fingerprint checking and execution creation are atomic across SQLite writers.
   Explicit replacement reservation also rechecks and fences the original's
   unsent children, preventing a paused original from submitting them later.
   Terminal preflight-only replacements can be explicitly retried safely.
4. Resubmission of a run that used a simulation validator requires the callback
   again with simulation enabled. Active/ambiguous provider jobs block replacement.
5. Observable sequences/arrays preserve independent output axes; mappings and
   explicit `SparsePauliOp` instances preserve summed-operator semantics. Every
   scalar observable receives the transpilation layout. Estimator schema 2 stores
   shape and loads legacy schema-1 sums without changing their interpretation.
6. Configured credentials are redacted before provider failures and metadata reach
   persistence or logs. Provider exception chains are suppressed; injected tests
   cover plain, escaped, URL-encoded and case-normalized fake credentials.

## Verification

- Final locked suite: 140 passed on Python 3.12.11, Qiskit 2.5.0,
  qiskit-ibm-runtime 0.48.0 and qiskit-aer 0.17.2.
- Final minimum-version suite: 140 passed on Python 3.12.11, Qiskit 2.3.0,
  qiskit-ibm-runtime 0.45.0 and qiskit-aer 0.17.2; two upstream fractional-plugin
  deprecation warnings.
- Ruff formatting and lint passed. Strict mypy passed for 21 source files.
- The local simulation example and source/wheel build passed.

Tests exercise the real submission adapter behind fake Runtime primitives,
including process-death boundaries, persistence failures before/after commit,
concurrent reservation/resubmission, and provider-result recovery. The tests scan
SQLite/WAL files, artifacts, logs and rendered exception tracebacks for injected
credentials. No tests contact IBM Quantum or consume hardware quota.

The real-Aer observable tests include six logical wires, ordered
`[Z0, ..., Z5, X0, ..., X5]`, three samples, deliberately reversed parameter names,
QPY/JSON/NPZ round trips, and physical layout `[5, 1, 7, 0, 6, 3]` on eight wires.
The resulting `(3, 12)` values agree with analytic expectations within `1e-12`.
These are synthetic RY-circuit adapter tests; application-specific circuits and
scientific results require separate validation.
Broadcasting follows the [IBM primitive input/output contract](https://quantum.cloud.ibm.com/docs/en/guides/primitive-input-output).

## Important limitations

- Application dependencies must satisfy the ranges declared in
  [pyproject.toml](../pyproject.toml). Resolve compatibility deliberately rather
  than silently upgrading an existing experiment environment.
- Exact adapter tests call Aer with precision 0. The runner's public Estimator
  preflight uses positive configured/shot-derived precision, so it is not an
  exact zero-precision reference.
- Tests of the runner do not establish equivalence of application circuits,
  trained parameters, preprocessing, classifier outputs, or saved artifacts.
  Validate these separately using representative application fixtures.
- Runtime resilience, twirling, dynamical-decoupling controls and parameter-row
  chunking remain outside the current runner API. Preserve the application's
  scientific choices explicitly rather than inferring them from defaults.
- Unknown-ID submission ambiguity is deliberately blocking. A provider may have
  accepted work before the client lost its ID; no local-only fix can prove that
  it did not. Manual provider-side reconciliation may be needed.
- Recorded verification covers Python 3.12.11. Python 3.11, 3.13 and 3.14 and real
  IBM service behavior have not been verified by these offline checks.
