# Run Local Search in Evaluator Workers

Use evaluator-owned local search to distribute complete, bounded inner searches
across synchronous evaluator workers, one episode per proposal.

Start with sequential execution. Move to Joblib only after the kernel behaves
correctly and each episode is expensive enough to amortize worker scheduling.

## Check eligibility

Before configuring worker execution, confirm that:

- the execution model is `sequential` or `sync_batch`
- default objective-cost accounting remains enabled
- the built-in kernel has a positive finite objective-call limit
- the selected evaluator supports request-local episodes
- the problem, kernel, candidates, and payloads can cross the selected backend
  boundary

Built-in structured local-search kernels derive finite limits from their step
and neighborhood bounds. `ScipyMinimizeKernel` requires an explicit
`max_evaluations` limit.

## Validate with sequential execution

Given a compatible `problem` and `optimizer` already constructed as in the
[Quickstart](../getting-started/quickstart.md), run the kernel inline first:

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

## Move complete episodes to Joblib workers

Use a batch with multiple independent episodes when throughput is the goal:

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
        backend="threading",
    ),
)

result, final_state = study.optimize(
    max_evaluations=200,
    batch_size=8,
    execution_model=SYNC_BATCH_EXECUTION_MODEL,
)
```

`batch_size=1` is supported but exposes no proposal-level parallelism.

!!! warning "Process-backend limitation in 0.2.0"

    Use the threading backend for built-in local-search episodes that can return
    `CandidateRefinement`. In 0.2.0, loky cannot deserialize a refinement-bearing
    request-local success. This limitation concerns episode result transport;
    ordinary Joblib objective evaluation remains available with loky.

For SciPy local search, cap objective calls separately from optimizer
iterations:

```python
from variopt.algorithms.local_search import ScipyMinimizeKernel

kernel = ScipyMinimizeKernel(
    method="L-BFGS-B",
    max_iterations=20,
    max_evaluations=40,
)
```

`max_iterations` limits SciPy iterations, not objective calls. The finite
`max_evaluations` value makes the episode eligible for worker dispatch.

## Measure semantic and timing equivalence

Compare the worker path with the sequential baseline under the same seed,
budget, batch semantics, and kernel limits. Compare result, failure, ordering,
refinement, trace, final-state, and evaluation-accounting digests as well as
wall time.

Joblib usually hurts when the batch contains one episode, objective work is
very cheap, process serialization dominates, or Python-level GIL-bound work is
sent to the threading backend.

See [Evaluator-Owned Local Search](../concepts/evaluator-owned-local-search.md)
for the placement rationale and measured crossover evidence. See
[Evaluator Contracts](../reference/evaluator-contracts.md) for dispatch,
budget, failure, randomness, and checkpoint guarantees.
