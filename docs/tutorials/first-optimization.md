# First Optimization Run

The [Quickstart](../getting-started/quickstart.md) shows the smallest runnable
example using the house default (CSA). This tutorial uses the same
[`Study`][variopt.Study] orchestration with a different
[`RunMethod`][variopt.RunMethod] —
[`DifferentialEvolutionOptimizer`][variopt.algorithms.population.DifferentialEvolutionOptimizer]
— to illustrate the main design point: swapping one component does not force
the rest to change.

## Minimal Recipe

```python
from typing_extensions import override

from variopt import RealSpace, Objective, Problem, Study
from variopt.algorithms.population import DifferentialEvolutionOptimizer
from variopt.evaluators import SequentialEvaluator


class SphereObjective(Objective[float]):
    @override
    def evaluate(self, candidate: float) -> float:
        return candidate * candidate


problem = Problem(
    space=RealSpace(-5.0, 5.0),
    objective=SphereObjective(),
)

optimizer = DifferentialEvolutionOptimizer(
    space=problem.space,
    population_size=12,
    random_state=0,
)

study = Study(
    problem=problem,
    run_method=optimizer,
    evaluator=SequentialEvaluator[float, float](),
)

result, final_state = study.optimize(max_evaluations=60)
```

## Reading The Result

```python
best = result.best_observation
print(f"best candidate: {best.candidate:.6f}")
print(f"objective value: {best.value:.6f}")
print(f"evaluations used: {result.evaluation_count}")
```

`result` is a [`RunResult`][variopt.RunResult]:

- `best_observation` — the [`Observation`][variopt.Observation] with the
  lowest score across the run
- `observations` — the full evaluation history in execution order
- `evaluation_count` — total evaluations consumed (may exceed
  `len(observations)` when a kernel reports inner cost)

`final_state` is the optimizer's internal state at the end of the run. You
can pass it back as `initial_state=` to continue from where you left off:

```python
continued_result, _ = study.optimize(
    max_evaluations=60,
    initial_state=final_state,
)
```

## Swapping The Evaluator

The same study works with a parallel evaluator. Only the evaluator changes:

```python
from variopt.evaluators import JoblibEvaluator

parallel_study = Study(
    problem=problem,
    run_method=optimizer,
    evaluator=JoblibEvaluator[float, float](
        backend="loky",
        n_jobs=4,
    ),
)

result, _ = parallel_study.optimize(max_evaluations=60)
```

## Why This Matters

Compared to the quickstart CSA example, only the optimizer object changed.
The space, problem, evaluator, and `Study.optimize(...)` call are identical.

That is the payoff of the explicit role split: space, run method, and
evaluator each own one thing, so you can replace any one of them — a different
optimizer family, a parallel `JoblibEvaluator`, a non-trivial structured
space — without touching the other layers. The same pattern still works when
the space stops being a trivial scalar or vector. See
[Optimization Model](../concepts/optimization-model.md) for the full
rationale.
