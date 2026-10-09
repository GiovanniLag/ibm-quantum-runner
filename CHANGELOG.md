# Changelog

All notable changes follow Keep a Changelog conventions.

## [Unreleased]

### Fixed

- Persist every planned child before provider effects; require exact circuit and
  artifact coverage before a logical execution can complete.
- Fence ambiguous and concurrent submissions, reserve duplicate fingerprints
  transactionally, and preserve known job IDs during persistence failures.
- Prevent silent loss of simulation validators during explicit resubmission.
- Preserve independent shaped Estimator observables through layout mapping,
  non-pickle artifacts and results while keeping explicit summed operators scalar.
- Redact configured credentials before provider errors or metadata reach durable
  state or logs; suppress raw exception chains at provider boundaries.
- Recover missing or malformed results from known provider jobs without replaying
  hardware work; allow explicit retries of terminal preflight-only replacements.

## [0.1.0] - 2026-07-15

### Added

- Typed SamplerV2 and EstimatorV2 workflows for ordinary Qiskit circuits.
- Structured local validation and optional Qiskit Aer simulation.
- Explicit IBM environment authentication and backend selection.
- Deterministic PUB chunking and Runtime Batch support.
- SQLite lifecycle/job persistence with atomic QPY, JSON, and NPZ artifacts.
- Resume by local execution ID and conservative recovery by IBM job ID.
- Bounded retries for idempotent provider reads and explicit-only resubmission.
- Stable normalized result models, partial child result preservation, and logs.
