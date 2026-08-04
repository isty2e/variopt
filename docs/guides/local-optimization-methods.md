# Choose a Local Optimization Method

Add local optimization only after the global run method behaves correctly on
its own. Then choose a kernel from candidate structure and local objective
behavior.

## Start from the candidate space

| Candidate domain | Recommended starting point |
| --- | --- |
| Smooth continuous `RealSpace` leaves | `ScipyMinimizeKernel(method="L-BFGS-B")` |
| Rough or weakly discontinuous continuous objective | `ScipyMinimizeKernel(method="Powell")` |
| Discrete structured space | `StructuredHillClimbKernel(max_steps=...)` |
| Discrete space with expensive categorical enumeration | `StructuredStochasticNeighborhoodKernel(...)` |
| Discrete space with a justified widening sequence | `StructuredVariableNeighborhoodKernel(...)` |
| Discrete space with a bounded kick-and-refine strategy | `StructuredIteratedLocalSearchKernel(...)` |
| Discrete space with a fixed stage sequence | `StructuredScheduledLocalSearchKernel(...)` |
| Mixed real, integer, and categorical leaves | Custom kernel, explicit domain split, or no local optimization |

The built-in SciPy kernel rejects non-real leaves. Do not project integer or
categorical values into a continuous optimizer unless the relaxation and its
return to canonical candidates are explicit parts of the problem model.

## Choose between `L-BFGS-B` and `Powell`

Use `L-BFGS-B` when all optimized leaves are continuous, the objective is
reasonably smooth locally, and coordinate bounds matter. It is also the natural
built-in choice for log-scaled real leaves because the kernel optimizes in the
space coordinate system.

Finite-difference gradient estimation can consume many objective evaluations.
Set both iteration and objective-call limits when cost must remain bounded:

```python
from variopt.algorithms.local_search import ScipyMinimizeKernel

kernel = ScipyMinimizeKernel(
    method="L-BFGS-B",
    max_iterations=20,
    max_evaluations=40,
)
```

Use `Powell` when the space is still continuous but numerical gradients are
unreliable, the objective is piecewise, or `L-BFGS-B` stalls on local
non-smoothness. Powell may be more tolerant of those conditions, but it can
also require many function evaluations.

## Choose a discrete neighborhood

Use `StructuredHillClimbKernel` when every leaf is `IntegerSpace` or
`CategoricalSpace` and deterministic one-leaf first improvement is affordable.

Move to another discrete kernel only when the corresponding neighborhood law is
part of the optimization intent:

- `StructuredStochasticNeighborhoodKernel` samples a bounded subset of
  one-leaf moves when full categorical enumeration is too expensive.
- `StructuredVariableNeighborhoodKernel` widens through explicit stages and
  resets after an accepted improvement.
- `StructuredIteratedLocalSearchKernel` alternates deterministic improvement
  with bounded kicks under a separate kick policy.
- `StructuredScheduledLocalSearchKernel` executes one fixed sequence of stages
  without variable-neighborhood reset semantics.

Variable-neighborhood and iterated local search are not generic upgrades over
hill climbing. Use them only when the widening or perturbation story is
specific enough to configure and explain.

The built-in kernels do not expose analytic gradients, repair logic, or
domain-specific move families as first-class contracts. Provide a custom
`Kernel` or skip local optimization when those capabilities define the domain.

## Preserve objective-cost accounting

Keep the default `count_evaluation_cost=True` when comparing runs with and
without local search. A single top-level proposal can trigger many inner
objective calls, and the default budget charges those calls against
`max_evaluations`.

Use `count_evaluation_cost=False` only when the experiment deliberately budgets
outer attempt slots instead of objective cost. A custom kernel that evaluates
the objective must return the computed value and the true `evaluation_count` so
the study can reuse the result and account for it once.

Refinement and budget metadata are independent. A kernel can report a changed
candidate with `evaluation_count=1`, or no refinement with a larger inner cost.

## Validate sequentially before parallelizing

Start with `SequentialEvaluator`. After checking the kernel's candidates,
refinement metadata, failures, and evaluation counts, use synchronous
`JoblibEvaluator` only when a batch contains multiple bounded episodes whose
work can amortize transport.

Keep the evaluator as the outer parallel owner and each local-search episode
serial. Nested worker spawning risks oversubscription and weakens the resource
contract.

The current study-level exact-async and stale-async paths require
`DirectKernel`. MPI also keeps request-local episodes on the coordinator.

See [Run Local Search in Evaluator Workers](request-local-episodes.md) for the
configuration steps and [Evaluator Contracts](../reference/evaluator-contracts.md)
for dispatch, hard-budget, failure, and checkpoint rules.

## Recommended sequence

1. Run the global optimizer without local search.
2. Add `L-BFGS-B` for a smooth continuous domain or deterministic hill climbing
   for a discrete structured domain.
3. Compare under the default objective-cost budget.
4. Change the method only in response to observed objective or neighborhood
   behavior.
5. Add Joblib only after the sequential episode is correct and expensive enough
   to parallelize.

See [Local Search](../concepts/local-search.md) for the kernel ownership model
and [the local-search API](../reference/api/local-search.md) for constructor and
configuration types.
