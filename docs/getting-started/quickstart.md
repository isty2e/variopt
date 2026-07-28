# Quickstart

This is the smallest practical scalar optimization example.

```python
from variopt import IntegerSpace, Problem, Study
from variopt.algorithms.population import CSAOptimizer
from variopt.evaluators import SequentialEvaluator


def square(candidate: int) -> float:
    return float(candidate * candidate)


problem = Problem(
    space=IntegerSpace(-10, 10),
    objective=square,
    name="square",
)

optimizer = CSAOptimizer.from_space_defaults(
    space=problem.space,
    bank_capacity=8,
    random_state=0,
)

study = Study(
    problem=problem,
    run_method=optimizer,
    evaluator=SequentialEvaluator[int, int](),
)

result, final_state = study.optimize(max_evaluations=40)
```

## Reading The Result

`study.optimize(...)` returns a `(RunResult, state)` pair.

```python
best = result.best_observation
print(best.candidate)  # the best candidate found (e.g. 0)
print(best.value)  # raw objective value (e.g. 0.0)

print(result.evaluation_count)  # total evaluations consumed
print(len(result.observations))  # number of observed candidates
```

`best_observation` is an [`Observation`][variopt.Observation] — it carries the
candidate, the raw objective value, and an internal `score` used for
minimization ordering.

The full observation history is available as `result.observations`, ordered by
evaluation sequence.

## What This Example Uses

- [`IntegerSpace`][variopt.IntegerSpace] defines the search domain
- [`Problem`][variopt.Problem] normalizes the typed `square` callable into its
  canonical scalar objective contract
- [`CSAOptimizer.from_space_defaults(...)`][variopt.algorithms.population.CSAOptimizer.from_space_defaults]
  derives sampler, diversity metric, and perturbation schedule from the space
- [`SequentialEvaluator`][variopt.evaluators.SequentialEvaluator] evaluates
  proposals one at a time
- [`Study.optimize(...)`][variopt.Study.optimize] orchestrates the run and
  returns a scalar summary

`bank_capacity=8` is the CSA bank size, roughly the equivalent of
`population_size` in a GA: larger banks explore more candidates in parallel at
higher evaluation cost. `8` is chosen here for a small illustrative run; pick
the size based on your evaluation budget, not from this example.

An explicit [`Objective`][variopt.Objective] subclass remains useful when a
named reusable class better represents the evaluation rule. For process and
MPI evaluators, a picklable function defined at an importable module level is
the conservative callable choice. Lambdas, closures, bound methods, and
stateful callable objects may work with particular serializers but are not
guaranteed to be portable across every backend.

## Next Steps

- fuller walkthrough:
  [First Optimization Run](../tutorials/first-optimization.md)
- structured domains:
  [Structured Spaces](../tutorials/structured-spaces.md)
- evaluator choice:
  [Choose an Evaluator](../guides/choose-an-evaluator.md)
