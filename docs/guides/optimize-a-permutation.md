# Optimize a Permutation

Use `PermutationSpace` when each candidate is an ordering of `0..N-1`. This
guide configures a genetic algorithm for a small tour-cost problem.

## Define the tour objective

```python
from typing_extensions import override

from variopt import Objective, PermutationSpace, Problem, Study
from variopt.algorithms.population import (
    GAProfile,
    GeneticAlgorithmOptimizer,
    OrderCrossover,
    SwapMutation,
)
from variopt.evaluators import SequentialEvaluator


DISTANCES = [
    [0, 10, 15, 20],
    [10, 0, 35, 25],
    [15, 35, 0, 30],
    [20, 25, 30, 0],
]


class TourCostObjective(Objective[tuple[int, ...]]):
    @override
    def evaluate(self, candidate: tuple[int, ...]) -> float:
        total = 0.0
        for index in range(len(candidate)):
            next_index = (index + 1) % len(candidate)
            total += DISTANCES[candidate[index]][candidate[next_index]]
        return total
```

## Select permutation operators

Use crossover and mutation operators that preserve the permutation contract:

```python
space = PermutationSpace(size=4)

optimizer = GeneticAlgorithmOptimizer(
    space=space,
    population_size=20,
    crossover_operator=OrderCrossover(space=space),
    mutation_operator=SwapMutation(space=space),
    profile=GAProfile(
        crossover_probability=0.9,
        mutation_probability=0.3,
    ),
    random_state=0,
)
```

Generic numeric crossover can create duplicates or omit values and is therefore
not valid for this space. `OrderCrossover`, `SwapMutation`, and
`InversionMutation` preserve permutation candidates.

## Run the study

```python
study = Study(
    problem=Problem(space=space, objective=TourCostObjective()),
    run_method=optimizer,
    evaluator=SequentialEvaluator[tuple[int, ...], tuple[int, ...]](),
)

result, _ = study.optimize(max_evaluations=200)
print(f"best tour: {result.best_observation.candidate}")
print(f"tour cost: {result.best_observation.value}")
```

The candidate remains a `tuple[int, ...]`; no vector decoding step is needed.
See [Population Algorithms](../concepts/population-algorithms.md) for the role
of genetic operators and [the population API](../reference/api/population.md)
for the complete public operator surface.
