# Evaluator-Owned Local-Search Episodes

`variopt` can execute one bounded local-search episode per proposal inside a
synchronous evaluator. This lets `JoblibEvaluator` distribute complete local
searches across workers instead of sending every inner objective call back
through the coordinator.

Use this feature when each proposal-local search is expensive enough to justify
worker scheduling. It is an execution placement rule, not a different
optimization algorithm.

## The Four Boundaries

The following concepts answer different questions:

| Concept | Question it answers |
| --- | --- |
| objective evaluation | What value or payload does one candidate produce? |
| request-local kernel episode | Which candidates should one proposal-local search evaluate before returning one top-level attempt? |
| evaluator backend | Where does evaluation work execute? |
| execution model | When and in what order does the run method assimilate completed work? |

A request-local episode may make several serial objective calls, but it still
returns exactly one top-level `EvaluationSuccess` or `EvaluationFailure` slot
for its source proposal. Its actual objective-call cost is reported through
`evaluation_count`.

```text
Coordinator                  Evaluator workers                 Coordinator

RunMethod.ask()
      |
      v
validate requests
reserve hard budget --------> bounded serial episode
                              (one per proposal)
                                      |
                                      v
                              restore request order ----------> validate attempts
                                                                settle actual cost
                                                                refund unused capacity
                                                                       |
                                                                       v
                                                              RunMethod.tell_attempts()
```

The coordinator continues to own optimizer state, hard-budget accounting, and
assimilation order. Workers own only the bounded episode they receive. Inner
objective calls remain serial so the evaluator remains the outer parallelism
owner.

## Configuration

Given a compatible `problem` and `optimizer` already constructed as in the
[Quickstart](../getting-started/quickstart.md), start with
`SequentialEvaluator` while validating a kernel:

```python
from variopt import Study
from variopt.algorithms.local_search import StructuredHillClimbKernel
from variopt.evaluators import SequentialEvaluator

study = Study(
    problem=problem,
    run_method=optimizer,
    kernel=StructuredHillClimbKernel(max_steps=8),
    evaluator=SequentialEvaluator(),
)

result, final_state = study.optimize(
    max_evaluations=200,
    batch_size=1,
)
```

Switch to synchronous `JoblibEvaluator` when a batch contains independent,
bounded episodes:

```python
from variopt import SYNC_BATCH_EXECUTION_MODEL, Study
from variopt.algorithms.local_search import StructuredHillClimbKernel
from variopt.evaluators import JoblibEvaluator

study = Study(
    problem=problem,
    run_method=optimizer,
    kernel=StructuredHillClimbKernel(max_steps=8),
    evaluator=JoblibEvaluator(
        n_jobs=4,
        backend="loky",
    ),
)

result, final_state = study.optimize(
    max_evaluations=200,
    batch_size=8,
    execution_model=SYNC_BATCH_EXECUTION_MODEL,
)
```

These configuration fragments import only supported public names. `problem`
and `optimizer` must still be compatible with the selected structured
local-search kernel. Configure this feature through `Study`; do not construct
internal episode or budget-reservation objects or call evaluator orchestration
hooks directly.

For SciPy local search, set an objective-call cap:

```python
from variopt.algorithms.local_search import ScipyMinimizeKernel

kernel = ScipyMinimizeKernel(
    method="L-BFGS-B",
    max_iterations=20,
    max_evaluations=40,
)
```

`max_iterations` limits SciPy iterations, not objective calls. A finite
`max_evaluations` value is therefore required before a SciPy episode can move
to an evaluator worker.

## Dispatch and Fallback

Evaluator-owned dispatch is used only when all of these conditions hold:

- the execution model is `sequential` or `sync_batch`
- the default objective-cost mode, `count_evaluation_cost=True`, is active
- the evaluator implements request-local episode execution
- the kernel explicitly implements the request-local capability
- every episode has a positive finite objective-call limit
- the remaining hard budget can reserve at least one evaluation per request

Built-in structured local-search kernels derive finite limits from their step
and neighborhood bounds. Stochastic structured kernels additionally require
the proposal-local random-state snapshot supplied by the run method.

If one of the dispatch conditions is not met, synchronous execution uses the
ordinary coordinator-owned kernel path. This fallback preserves kernel
behavior; it changes placement, not local-search semantics. A custom kernel or
subclass also remains coordinator-owned unless it explicitly implements the
episode capability. That capability is not currently a supported public
extension contract.

`count_evaluation_cost=False` deliberately uses the coordinator path because
there is no hard objective-cost ledger from which to reserve worker capacity.

## Supported Execution Matrix

| Evaluator and execution model | Request-local episode placement |
| --- | --- |
| `SequentialEvaluator`, `sequential` or `sync_batch` | supported; episode runs inline |
| `JoblibEvaluator`, `sequential` or `sync_batch` | supported through joblib; parallel only when worker and batch counts permit |
| `MpiEvaluator`, `sync_batch` | coordinator fallback; MPI does not own request-local episodes |
| `AsyncJoblibEvaluator`, `exact_async` | unsupported; built-in study-level exact-async execution currently requires `DirectKernel` |
| stale-async execution | unsupported; the current study path requires `DirectKernel` |

With `JoblibEvaluator`, `batch_size=1` is supported but exposes no
proposal-level parallelism. Use a batch with multiple episodes when throughput
is the goal.

The `threading` backend shares the exact `Problem` instance across workers, so
the evaluation protocol must be thread-safe. The `loky` backend crosses a
process serialization boundary. The problem, objective, kernel, candidate,
proposal-local context, and returned payload must all be serializable.

`problem_transport="worker_session"` also applies to request-local episodes. It
can avoid repeatedly serializing a large immutable problem within one
synchronous run, but it does not make mutable worker state checkpoint-safe. See
[Choose an Evaluator](choose-an-evaluator.md#reusing-a-problem-in-joblib-workers)
for its lifecycle and measurement guidance.

## Hard-Budget Accounting

`Study` reserves each batch before sending any episode to an evaluator. When
the remaining budget is smaller than the sum of preferred episode limits,
capacity is divided by deterministic max-min allocation:

1. every request receives one evaluation if the batch is dispatchable
2. remaining capacity raises the smallest limits together
3. ties are resolved in request order
4. no request receives more than its preferred limit

For example, preferred limits `(8, 8, 8)` and 15 remaining evaluations produce
limits `(5, 5, 5)`. Preferred limits `(2, 8, 8)` and 10 remaining evaluations
produce `(2, 4, 4)`.

Reservation is conservative until trusted results return. Suppose limits
`(8, 8, 8)` are reserved and the completed episodes report exact costs
`(3, 8, 5)`:

- 24 evaluations are reserved before submission
- 16 evaluations are retained as consumed
- 8 unused evaluations are returned to the run budget

Successful and recorded-failure attempts both contribute their actual
`evaluation_count`. A local-search episode may therefore consume budget through
failed inner trials even if a later trial succeeds.

If the evaluator raises before returning a complete, aligned attempt batch,
the coordinator cannot know which worker-side objective calls completed. The
whole unresolved reservation remains charged and the study raises
`RunExecutionFailed`. Invalid or over-limit accounting also fails closed rather
than guessing a refund.

## Result and Failure Guarantees

Moving an eligible episode to the evaluator preserves:

- one top-level attempt slot per source proposal
- logical request order before run-method assimilation
- `CandidateRefinement` provenance for a changed candidate
- exact reported `evaluation_count`
- recorded user-code failures as `EvaluationFailure`
- inner failure summaries and terminal status in `KernelDiagnostics`
- the original proposal identity used by the run method

Worker completion order is not observable at the assimilation boundary.
`sync_batch` is still semantically different from `sequential`, however,
because the run method commits the whole batch together. Changing `batch_size`
or execution model can therefore change an optimization trajectory even when
the evaluator restores request order.

An objective exception that is successfully recorded is data, not a backend
failure. The episode may return a successful top-level attempt with failed
inner trials summarized in diagnostics, or an `EvaluationFailure` when no trial
succeeds. Infrastructure failures, malformed worker output, and assimilation
failures remain hard execution failures.

## Randomness and Checkpoints

Stochastic built-in episodes use a deterministic proposal-local
`RandomStateSnapshot` derived by the run method. Worker scheduling and
completion order do not choose the random stream. The coordinator rebinds
candidate-equality validation after worker execution so process transport does
not weaken refinement checks.

Supported durable checkpointing currently serializes CSA run-method state only.
It does not serialize reports, evaluator workers, or live episode reservations.
Resume creates new evaluator runtime state and derives subsequent proposal-local
streams from the restored run-method state. Preserve reports separately, and
use `stop_at_checkpoint_boundary=True` when the run method requires a safe state
boundary.

An unresolved worker failure is not itself a resumable in-flight checkpoint.
Recover from the latest checkpoint-safe report and state carried by
`RunExecutionFailed`, when available.

## When Joblib Helps

Joblib helps when there are multiple eligible episodes and each episode has
enough objective work to amortize scheduling and transport. It usually hurts
when:

- `batch_size=1`
- the objective and local-search bookkeeping are very cheap
- process serialization dominates the episode
- the objective is Python-level, GIL-bound work on the `threading` backend
- a short run repeatedly pays worker or `worker_session` setup

One development measurement on Apple arm64 with Python 3.13 and Joblib 1.5.3
used four workers and five measured repeats per cell. For GIL-releasing
synthetic episode work, both `threading` and `loky` were slower with no
objective delay, reached roughly 2.5x at 2 ms per objective call, and roughly
2.7x to 3.2x at 10-50 ms. At a fixed 5 ms objective delay, one-proposal batches
were slower, while four- and eight-proposal batches were roughly 1.5x to 2.6x
faster depending on backend and worker count.

For Python-level GIL-bound work, `threading` stayed near break-even even around
10 ms, while `loky` reached roughly 2.7x in that synthetic case. A failure-heavy
workload and a stochastic workload preserved their semantic digests while
running roughly 2.1x to 2.7x faster. These are workload-specific observations,
not portable crossover thresholds.

Across the 28 measured comparison cells, timing-independent result, failure,
ordering, refinement, trace, final-state, and evaluation-accounting digests
matched the coordinator baseline.

Measure the real objective, candidate serialization cost, episode length
distribution, `batch_size`, backend, and worker count. Compare final result and
accounting digests as well as wall time; a faster run with different
assimilation semantics is not an equivalent benchmark.

## Related Reading

- [Local Optimization Methods](local-optimization-methods.md)
- [Choose an Evaluator](choose-an-evaluator.md)
- [Study and Execution Models](../concepts/study-and-execution-models.md)
- [Candidate Refinement](../concepts/candidate-refinement.md)
- [Checkpointing](../reference/checkpointing.md)
