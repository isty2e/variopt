# Choose an Evaluator

Choose the evaluator from execution mechanics, not from optimization semantics.

## Built-In Evaluators

- `SequentialEvaluator`
  - simplest path
  - easiest to debug
  - best default when correctness matters more than throughput
- `JoblibEvaluator`
  - local batch-parallel execution
  - good when objective calls are independent and thread/process parallelism is
    enough
- `AsyncJoblibEvaluator`
  - local exact-async batch lifecycle
  - use when your run method supports `exact_async`
  - current built-in exact-async path for CSA, DE, and GA-family optimizers
- `MpiEvaluator`
  - optional MPI-backed execution
  - use only when you actually need distributed workers

## Main Rule

Evaluator choice should not silently change the meaning of the optimizer.

`variopt` treats execution models such as `sequential`, `sync_batch`,
`exact_async`, and `stale_async` as semantic contracts rather than backend
labels.

## Practical Mapping

- want the simplest one-proposal path:
  `SequentialEvaluator` with `batch_size=1`
- want local or distributed whole-batch execution:
  `JoblibEvaluator` or `MpiEvaluator`
- want built-in exact-async execution:
  `AsyncJoblibEvaluator`
- want stale-async:
  that is currently a custom-run-method path rather than a built-in evaluator
  recommendation, because the built-in population optimizers do not advertise
  `stale_async`

## Reusing a Problem in Joblib Workers

`JoblibEvaluator` sends the problem with each batch by default. If the problem
contains a large, immutable evaluation context and objective calls are much
smaller, opt into one snapshot per synchronous run:

```python
from variopt.evaluators import JoblibEvaluator

evaluator = JoblibEvaluator(
    n_jobs=4,
    backend="loky",
    problem_transport="worker_session",
)
```

This mode binds the exact `Problem` instance for one `Study.run(...)`,
`Study.optimize(...)`, or `Study.step(...)` scope. A custom kernel cannot
substitute another problem inside that scope. The default `per_request` mode
continues to accept the problem supplied with each evaluator call.

Each loky process decodes an independent problem instance and retains at most
one current generation. A worker may replace that generation when concurrent
runs share the process pool, or reconstruct it after worker replacement.
Therefore observable evaluation behavior must be derived from the request and
the immutable snapshot. Process-local memoization is suitable; mutable random
streams, counters, wall-clock-dependent state, and cross-worker coordination
are not restart-equivalent unless their observable results are request-derived.

Threading backends and effective single-worker execution use direct shared
references and do not create the snapshot transport. Checkpoints never contain
the worker session: resuming a run creates a new generation from the
coordinator-owned problem. Concurrent sessions remain isolated but frequent
generation switching can reduce cache reuse.

The snapshot is serialized with `cloudpickle` into an owner-only temporary
directory and exposed to workers as a read-only NumPy memory map. It is trusted
same-process-family data, not a portable or long-term file format. Normal and
exceptional scope closure removes the transport; abrupt coordinator termination
can leave an artifact in the operating system temporary directory for ordinary
temporary-file cleanup to reclaim.

## Related Reading

- [Concepts / Study and Execution Models](../concepts/study-and-execution-models.md)

## Exact-Async Quick Setup

Given an existing `problem` and an optimizer that advertises `exact_async`:

```python
from variopt import EXACT_ASYNC_EXECUTION_MODEL, Study
from variopt.evaluators import AsyncJoblibEvaluator

study = Study(
    problem=problem,
    run_method=optimizer,
    evaluator=AsyncJoblibEvaluator(n_jobs=4, backend="loky"),
)

result, _ = study.optimize(
    max_evaluations=60,
    batch_size=8,
    execution_model=EXACT_ASYNC_EXECUTION_MODEL,
)
```

For a fuller walkthrough, see
[Study and Execution Models](../concepts/study-and-execution-models.md#exact-async-example).

## Async Protocol Types

`AsyncEvaluator` is the public protocol for evaluators that submit one logical
batch and later return ordered `CompletionGroup` slices. `EvaluationBatchSession`
is the lifecycle object returned by that submission. `ResumableAsyncEvaluator`
adds evaluator-owned suspend/resume handles for exact-async sessions without
changing the run method's execution model.

`Study` requires native attempt-aware evaluator capability. Synchronous study
execution calls `evaluate_attempts(...)`; async study execution uses
attempt-batch session hooks such as `open_attempt_session(...)` and, for
resumable exact-async sessions, `resume_attempt_session(...)`.
Those attempts may carry request-free scalar or vector payloads such as
`ObservationPayload` and `ObjectiveVectorPayload`, or already request-aligned
record payloads. `Study` materializes successful payload attempts into feedback
records before calling the run method, so scalar and vector evaluators do not
need to build terminal records themselves. If a custom integration needs a
payload family to feed a different feedback-record family, pass an explicit
`EvaluationAttemptMaterializer` to `Study`.

`AsyncJoblibEvaluator` provides the async hooks. In that path, ordinary
objective `Exception`s become recorded `EvaluationFailure` attempts, while
candidate validation, cancellation, and backend failures remain hard batch
failures. Direct `Evaluator.evaluate(...)` remains available on the evaluator
facade, but `Study` does not adapt outcome-only batches or sessions.
Suspended async joblib batches are held in the same evaluator instance's
in-memory runtime state; their resume handles are for live same-process control
flow, not crash recovery.

When `infrastructure_retry_limit` is positive, `AsyncJoblibEvaluator` retries
only unfinished work after recognized backend boundary failures, such as
joblib/loky or `concurrent.futures` process-pool breakage. User exceptions from
the objective remain user-code failures even if their class names resemble
backend exceptions.

Suspending, resuming, cancelling, or retrying an async joblib batch is
at-least-once at the backend boundary. `variopt` preserves completed request
indices and removes evaluator-owned active state before retrying or cancelling,
but joblib may not be able to stop already-dispatched work immediately. If the
backend abort hook or fallback generator close fails, `AsyncJoblibEvaluator`
emits a `RuntimeWarning`; treat side-effecting objectives as non-idempotent
unless you provide your own external transaction boundary.
