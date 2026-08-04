# Run Exact-Async Evaluations

Use exact-async execution when requests may finish out of order but the
optimizer must still receive results under its exact state-transition law.

This guide assumes an existing `Problem` and a built-in population optimizer
that advertises `exact_async`.

## Configure the async evaluator

The following complete example uses CSA and a local Joblib process pool:

```python
from typing_extensions import override

from variopt import EXACT_ASYNC_EXECUTION_MODEL, Objective, Problem, RealSpace, Study
from variopt.algorithms.population import CSAOptimizer
from variopt.evaluators import AsyncJoblibEvaluator


class SphereObjective(Objective[float]):
    @override
    def evaluate(self, candidate: float) -> float:
        return candidate * candidate


problem = Problem(
    space=RealSpace(-5.0, 5.0),
    objective=SphereObjective(),
)

optimizer = CSAOptimizer.from_space_defaults(
    space=problem.space,
    bank_capacity=8,
    random_state=0,
)

study = Study(
    problem=problem,
    run_method=optimizer,
    evaluator=AsyncJoblibEvaluator(n_jobs=4, backend="loky"),
)
```

## Select the execution model

Pass the exact-async model and a batch size to `Study.optimize(...)`:

```python
result, final_state = study.optimize(
    max_evaluations=60,
    batch_size=8,
    execution_model=EXACT_ASYNC_EXECUTION_MODEL,
)
```

Worker completions may arrive out of order. The study restores logical order
before handing them to the optimizer, so the optimizer sees the sequence it
would receive under `sync_batch`.

The current study-level exact-async path requires `DirectKernel`. Use
synchronous execution when a local-search kernel must own bounded inner
objective calls.

## Treat resume handles as live runtime state

`StudyExactAsyncStepSession` and `StudyExactAsyncStepResumeHandle` support
explicit polling, suspension, and resumption while the same evaluator instance
remains alive. They are not durable checkpoint artifacts and must not be used
for crash recovery.

See [Evaluator Contracts](../reference/evaluator-contracts.md) for async
lifecycle, retry, failure, and at-least-once boundary guarantees. See
[Checkpointing](../reference/checkpointing.md) for the separate durable CSA
state contract.
