# Execution lifecycle

```text
CREATED -> VALIDATING -> VALIDATION_FAILED
                    \-> SIMULATING -> SIMULATION_FAILED
                                  \-> READY_FOR_HARDWARE
                 VALIDATING ------> READY_FOR_HARDWARE
READY_FOR_HARDWARE -> SUBMITTING -> SUBMITTED -> QUEUED -> RUNNING
                              \-> PARTIALLY_SUBMITTED -> RECOVERY_REQUIRED
                              \-> FAILED
SUBMITTED / QUEUED / RUNNING -> COMPLETED
                             -> PARTIALLY_COMPLETED
                             -> FAILED
                             -> CANCELLED
                             -> RECOVERY_REQUIRED
```

States and provider raw statuses are persisted separately. IBM `INITIALIZING`
and `VALIDATING` map to `SUBMITTED`; `QUEUED`, `RUNNING`, `DONE`, `ERROR`, and
`CANCELLED` map to their corresponding library states.

`resume(execution_id)` reloads state and reattaches to every stored IBM job ID.
It cannot submit. `recover_ibm_job(job_id)` first resolves a known local job; an
untracked Sampler job can be imported only when IBM exposes its submitted
circuits. Untracked Estimator recovery requires the original local execution
because observable/parameter provenance is required for faithful normalization.

`RECOVERY_REQUIRED` means a provider read/result operation exhausted bounded
transient retries or submission was ambiguous. Calling `status()` again retries
monitoring of the existing job IDs. It never creates new usage. `resubmit()` is
the only replacement path and requires `confirm_hardware_resubmission=True`.

For a logical workload with multiple child jobs, active state is aggregated in
the order recovery-required, running, queued, submitted. All-completed becomes
`COMPLETED`; a mix of successful and terminal unsuccessful children becomes
`PARTIALLY_COMPLETED`. Completed child artifacts remain available when siblings
fail.

## Durable child submission boundaries

Child states are distinct from the public execution lifecycle:

- `PLANNED`: the complete plan is committed; this child has not crossed the run boundary.
- `NOT_SUBMITTED`: preparation failed before the run boundary.
- `SUBMITTING`: the run boundary was claimed durably; a missing job ID is ambiguous.
- `RECOVERY_REQUIRED`: submission or provider reading needs reconciliation.
- `SUPERSEDED_NOT_SUBMITTED`: an explicit replacement fenced this unsent original child.

`primitive.run` is called only after a transactional `PLANNED` claim. Every child
is planned before batch construction. If a process dies after a provider accepts
the request but before the ID becomes durable, no automated action can prove
non-submission. The library therefore does not replay that child. Explicit
resubmission also refuses active or ambiguous jobs; this is intentionally
conservative and may need manual provider-side reconciliation.

A replacement reservation rechecks the parent's current child states under the
same SQLite writer lock used to claim a child. It fences original `PLANNED`
children before releasing the lock. A failed replacement with no child jobs may
be superseded only when preflight has terminally stopped; an active replacement
still blocks another reservation.

A logical `COMPLETED` state requires exact original-index coverage, no duplicate
child coverage, and matching successful artifact indices and circuit identities.
This also protects resumed legacy state. Corrupt or missing artifacts move the
execution to `RECOVERY_REQUIRED`; a subsequent refresh can fetch the existing
known provider job's result again. This never submits new hardware work.
