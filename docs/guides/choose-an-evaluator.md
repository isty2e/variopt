# Choose an Evaluator

Choose an evaluator according to where and when the objective must run. The
backend controls execution mechanics; it must not change the optimizer's
meaning.

## Start with the execution model

| Need | Evaluator | Execution model |
| --- | --- | --- |
| Simplest path for debugging and validation | `SequentialEvaluator` | `sequential` |
| Local batch-parallel execution | `JoblibEvaluator` | `sync_batch` |
| Distributed whole-batch execution | `MpiEvaluator` | `sync_batch` |
| Built-in exact-async execution | `AsyncJoblibEvaluator` | `exact_async` |

Start with `SequentialEvaluator` unless evaluation cost already justifies a
parallel backend. The built-in population optimizers do not advertise
`stale_async`; that model is currently a custom-run-method path.

## Check the process boundary

When evaluation may cross a process or MPI boundary, prefer a picklable
objective function defined at an importable module level. Sequential and
threading execution can use process-local callables, and Joblib's loky backend
uses `cloudpickle`, but lambdas, closures, bound methods, and stateful callable
objects are not guaranteed to be portable across every evaluator backend.

Choose the backend from the objective workload:

- use Joblib `threading` when objective work releases the GIL and shared-memory
  execution is appropriate
- use Joblib `loky` for process isolation or Python-level CPU work that benefits
  from multiple processes
- use MPI only when evaluation must run on distributed workers

Measure the real workload before increasing worker count. Serialization, batch
shape, objective cost, and assimilation semantics all affect throughput.

## Account for local search

Synchronous `SequentialEvaluator` and `JoblibEvaluator` can execute complete,
bounded local-search episodes for individual proposals. Use this path only when
each episode is expensive enough to amortize scheduling or process transport.
MPI and the current study-level exact-async and stale-async paths do not own
these episodes.

See [Run Local Search in Evaluator Workers](request-local-episodes.md) for the
configuration steps and
[Evaluator-Owned Local Search](../concepts/evaluator-owned-local-search.md) for
the execution model behind them.

## Follow the task-specific guide

- [Reuse a Problem in Joblib Workers](reuse-a-problem-in-joblib-workers.md)
  when a large immutable problem context is serialized repeatedly
- [Run Exact-Async Evaluations](run-exact-async.md) when completed requests
  should be collected asynchronously without changing logical assimilation
- [Run Local Search in Evaluator Workers](request-local-episodes.md) when one
  proposal owns a bounded inner search episode
- [Evaluator Contracts](../reference/evaluator-contracts.md) for async lifecycle,
  request-local eligibility, budget, failure, and checkpoint guarantees
- [Study and Execution Models](../concepts/study-and-execution-models.md) for the
  distinction between evaluator backends and execution models
