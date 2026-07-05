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

`AsyncJoblibEvaluator` also exposes attempt-aware session hooks used by `Study`
to stream `EvaluationAttemptBatch` slots directly. In that path, ordinary
objective `Exception`s become recorded `EvaluationFailure` attempts, while
candidate validation, cancellation, and backend failures remain hard batch
failures.
