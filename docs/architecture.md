# Architecture

The public boundary is deliberately narrow: `QuantumRunner` owns orchestration,
`RunConfig` owns per-run policy, and `Execution` is a persistent handle. Users
do not need an IBM service, primitive, batch, polling loop, retry loop, or state
file in application code.

| Area | Owner | Responsibility |
|---|---|---|
| Public orchestration | `runner.py` | Validation, simulation, submission, resume, explicit resubmit, aggregation |
| Persistent handle | `execution.py` | Status, wait, result, cancel, IDs, event history |
| Configuration | `config.py` | Typed run/retry policy and Estimator requests |
| Local preflight | `validation/` | Structured static and transpiled reports |
| Local execution | `simulation/` | Aer SamplerV2 and EstimatorV2 |
| IBM boundary | `hardware/` | Credentials, discovery, transpilation, primitives, batches, retryable reads |
| Durable state | `persistence/` | SQLite lifecycle plus atomic JSON/QPY/NPZ artifacts |
| Stable output | `results/` | Provider-independent circuit and execution results |

Provider submission is a special safety boundary. The library records each job
ID in a transaction immediately after `primitive.run()` returns. It will retry
reads, but never the submission call. If the response is ambiguous or only part
of a multi-job workload was submitted, state becomes `RECOVERY_REQUIRED` and a
caller must recover or explicitly resubmit.

Circuit identity uses SHA-256 over QPY bytes. Logical execution identity adds
primitive and execution-affecting configuration. QPY stores circuits for exact
resume/resubmit; JSON stores normalized state/results; NPZ stores numeric
Estimator parameter values without pickle.

IBM-specific types stop at `hardware/` and normalization. The stable result
model retains counts, register-level counts, expectation values, deviations,
backend/job IDs, timings, metadata, and simulation output. A raw provider result
exists only as an explicitly named in-memory escape hatch.
