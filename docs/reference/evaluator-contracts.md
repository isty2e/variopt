# Evaluator Contracts

This page describes the lifecycle and accounting contracts that apply to async
evaluation and evaluator-owned request-local kernel episodes.

## Async evaluator lifecycle

`AsyncEvaluator` submits one logical batch and later returns ordered
`CompletionGroup` slices. `EvaluationBatchSession` is the lifecycle object
returned by submission. `ResumableAsyncEvaluator` adds evaluator-owned
suspend/resume handles without changing the run method's execution model.

`Study` requires native attempt-aware evaluator capability. Synchronous study
execution calls `evaluate_attempts(...)`; async study execution uses
attempt-batch session hooks such as `open_attempt_session(...)` and, for
resumable exact-async sessions, `resume_attempt_session(...)`.

Attempts may carry request-free scalar or vector payloads such as
`ObservationPayload` and `ObjectiveVectorPayload`, or already request-aligned
record payloads. `Study` materializes successful payload attempts into feedback
records before calling the run method. To map a custom payload family to a
different record family, provide an explicit `EvaluationAttemptMaterializer` to
`Study`.

With `AsyncJoblibEvaluator`, ordinary objective exceptions become recorded
`EvaluationFailure` attempts. Candidate validation, cancellation, and backend
failures remain hard batch failures. Suspended batches remain in the same
evaluator instance's in-memory runtime state; their resume handles are not
crash-recovery artifacts.

When `infrastructure_retry_limit` is positive, `AsyncJoblibEvaluator` retries
only unfinished work after recognized backend boundary failures. User
exceptions remain user-code failures even when their class names resemble
backend exceptions.

Suspension, resumption, cancellation, and infrastructure retry are at-least-once
at the backend boundary. Joblib may not stop already-dispatched work
immediately. If the backend abort hook or fallback generator close fails,
`AsyncJoblibEvaluator` emits a `RuntimeWarning`. Treat side-effecting objectives
as non-idempotent unless an external transaction boundary protects them.

## Request-local episode eligibility

Evaluator-owned request-local dispatch is used only when:

- the execution model is `sequential` or `sync_batch`
- `count_evaluation_cost=True`
- the evaluator implements request-local episode execution
- the kernel explicitly implements the request-local capability
- every episode has a positive finite objective-call limit
- the remaining hard budget can reserve at least one evaluation per request

Custom kernels and subclasses remain coordinator-owned unless they explicitly
implement the capability. The capability is not currently a supported public
extension contract. When an eligibility condition is absent, synchronous
execution uses the coordinator-owned kernel path.

## Supported request-local placement

The table describes the development version. The `0.2.0` limitation is noted
below.

| Evaluator and execution model | Placement |
| --- | --- |
| `SequentialEvaluator`, `sequential` or `sync_batch` | Episode runs inline in the evaluator |
| `JoblibEvaluator` with threading, `sequential` or `sync_batch` | Episode may run in a Joblib worker |
| `JoblibEvaluator` with loky, `sequential` or `sync_batch` | Episode may run in a Joblib process, including refinement-bearing results |
| `MpiEvaluator`, `sync_batch` | Coordinator fallback |
| `AsyncJoblibEvaluator`, `exact_async` | Unsupported; study-level exact async requires `DirectKernel` |
| `stale_async` | Unsupported; the current study path requires `DirectKernel` |

The Joblib threading backend shares the exact `Problem` instance across
workers, so the evaluation protocol must be thread-safe. The loky backend
crosses a process serialization boundary. The problem, objective, kernel,
candidate, proposal-local context, and returned payload must be serializable.

Both `problem_transport="per_request"` and `"worker_session"` support
refinement-bearing results in the development version.

!!! warning "0.2.0 process-result limitation"

    The refinement transport fix is not yet released. In `0.2.0`, `loky` can
    fail to deserialize request-local successes carrying `CandidateRefinement`,
    including results with large integer or floating-point candidates. Use
    threading for these episodes on `0.2.0`; ordinary objective evaluation with
    `loky` is unaffected.

## Hard-budget accounting

`Study` reserves each request-local batch before sending episodes to an
evaluator. When the remaining budget is smaller than the sum of preferred
episode limits, deterministic max-min allocation divides capacity:

1. Every request receives one evaluation if the batch is dispatchable.
2. Remaining capacity raises the smallest limits together.
3. Ties are resolved in request order.
4. No request receives more than its preferred limit.

Preferred limits `(8, 8, 8)` with 15 remaining evaluations produce
`(5, 5, 5)`. Limits `(2, 8, 8)` with 10 remaining evaluations produce
`(2, 4, 4)`.

Reservation is conservative until trusted results return. If limits
`(8, 8, 8)` are reserved and completed episodes report exact costs `(3, 8, 5)`,
24 evaluations are reserved, 16 remain consumed, and 8 are refunded.

Successful and recorded-failure attempts both contribute their actual
`evaluation_count`. If the evaluator raises before returning a complete,
aligned attempt batch, the coordinator cannot know which worker-side calls
completed. The unresolved reservation remains charged and the study raises
`RunExecutionFailed`. Invalid or over-limit accounting also fails closed.

## Result and failure guarantees

Moving an eligible episode to an evaluator preserves:

- one top-level attempt slot per source proposal
- logical request order before run-method assimilation
- `CandidateRefinement` provenance for a changed candidate
- exact reported `evaluation_count`
- recorded user-code failures as `EvaluationFailure`
- inner failure summaries and terminal status in `KernelDiagnostics`
- original proposal identity

After worker execution, the coordinator independently validates request and
refinement alignment with the problem space's candidate-equality contract,
then rebinds that predicate before materialization and assimilation. Process
transport preserves prior alignment evidence but does not replace this check
or serialize the equality callable with each success.

An objective exception that is successfully recorded is data, not a backend
failure. An episode may return a successful top-level attempt with failed inner
trials summarized in diagnostics, or an `EvaluationFailure` when no trial
succeeds. Infrastructure failures, malformed output, and assimilation failures
remain hard execution failures.

## Randomness and checkpoints

Stochastic built-in episodes use a deterministic proposal-local
`RandomStateSnapshot` derived by the run method. Worker scheduling and
completion order do not select the random stream.

Durable checkpointing currently serializes CSA run-method state only. It does
not serialize reports, evaluator workers, live episode reservations, or async
resume handles. Resume creates fresh evaluator runtime state and derives later
proposal-local streams from restored run-method state.

An unresolved worker failure is not a resumable in-flight checkpoint. Recover
from the latest checkpoint-safe report and state carried by
`RunExecutionFailed`, when available.
