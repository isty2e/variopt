# First Optimization Run

In this tutorial, we will define a continuous optimization problem, run
Differential Evolution, inspect the result, and continue from the final
optimizer state. By the end, you will have exercised the four objects used in
most `variopt` runs: a space, a problem, a run method, and a study.

The package must already be installed. See
[Installation](../getting-started/installation.md) if the imports below are not
available.

## Define the objective

Start with a one-dimensional sphere objective. Its minimum is at `0.0`, where
the objective value is also `0.0`.

```python
from typing_extensions import override

from variopt import Objective, Problem, RealSpace, Study
from variopt.algorithms.population import DifferentialEvolutionOptimizer
from variopt.evaluators import SequentialEvaluator


class SphereObjective(Objective[float]):
    @override
    def evaluate(self, candidate: float) -> float:
        return candidate * candidate
```

The candidate type is `float`, so define the search domain with `RealSpace` and
bind the objective to it through `Problem`:

```python
problem = Problem(
    space=RealSpace(-5.0, 5.0),
    objective=SphereObjective(),
)
```

At this point, `problem.space.validate(0.5)` succeeds, while a value outside
`[-5.0, 5.0]` would be rejected at the space boundary.

## Configure the run method

Use Differential Evolution with a small population and a fixed random seed:

```python
optimizer = DifferentialEvolutionOptimizer(
    space=problem.space,
    population_size=12,
    random_state=0,
)
```

The seed makes this tutorial reproducible. The optimizer owns search state; it
does not evaluate the objective itself.

## Assemble and run the study

Use the sequential evaluator so the first run has only one execution path to
reason about:

```python
study = Study(
    problem=problem,
    run_method=optimizer,
    evaluator=SequentialEvaluator[float, float](),
)

result, final_state = study.optimize(max_evaluations=60)
```

The run evaluates exactly 60 candidates. Inspect the best observation:

```python
best = result.best_observation
print(f"best candidate: {best.candidate:.6f}")
print(f"objective value: {best.value:.6f}")
print(f"evaluations used: {result.evaluation_count}")
```

The `variopt` 0.2.0 reference environment produced:

```text
best candidate: 0.044954
objective value: 0.002021
evaluations used: 60
```

The final digits can vary with dependency versions, but the evaluation count
must be `60`, the candidate must remain within the declared space, and squaring
it must give the reported objective value. `result.observations` contains the
complete evaluation history in execution order.

## Continue the same search

`final_state` is the optimizer memory at the end of the run. Pass it back as
`initial_state` to continue rather than initialize a new population:

```python
continued_result, _ = study.optimize(
    max_evaluations=60,
    initial_state=final_state,
)

print(f"continued objective value: {continued_result.best_observation.value:.6f}")
print(f"continued evaluations used: {continued_result.evaluation_count}")
```

The reference environment reports:

```text
continued objective value: 0.000065
continued evaluations used: 60
```

You have now completed and continued one optimization run. The same `Study`
shape applies to other optimizer and evaluator families; change those components
only after the basic path is familiar.

## Next steps

- [Structured Spaces](structured-spaces.md) applies the same workflow to named,
  typed fields.
- [Choose an Optimizer](../guides/choose-an-optimizer.md) compares the built-in
  population methods.
- [Choose an Evaluator](../guides/choose-an-evaluator.md) explains when to move
  beyond sequential execution.
- [Optimization Model](../concepts/optimization-model.md) explains why the API
  separates these responsibilities.
