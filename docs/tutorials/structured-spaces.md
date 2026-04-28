# Structured Spaces

`variopt` is most useful when the search domain has explicit structure.

Built-in space families include:

- scalar spaces: `RealSpace`, `IntegerSpace`, `CategoricalSpace`
- container spaces: `TupleSpace`, `RecordSpace`, `ArraySpace`
- `PermutationSpace`

## End-To-End Example: Hyperparameter Tuning

This example optimizes a two-field record of hyperparameters. The space
preserves field semantics throughout the entire pipeline — sampling, diversity
measurement, local search, and result extraction all see named fields, not
an anonymous vector.

```python
from typing_extensions import override

from variopt import Objective, Problem, RealSpace, RecordSpace, Study
from variopt.algorithms.population import CSAOptimizer
from variopt.evaluators import SequentialEvaluator
from variopt.spaces import RecordCandidate


space = RecordSpace(
    learning_rate=RealSpace(1e-4, 1e-1, scale="log"),
    momentum=RealSpace(0.0, 0.99),
)


class MockTrainingObjective(Objective[RecordCandidate]):
    @override
    def evaluate(self, candidate: RecordCandidate) -> float:
        lr = float(candidate["learning_rate"])
        mom = float(candidate["momentum"])
        return (lr - 0.01) ** 2 + (mom - 0.9) ** 2


problem = Problem(
    space=space,
    objective=MockTrainingObjective(),
)

optimizer = CSAOptimizer.from_space_defaults(
    space=space,
    bank_capacity=10,
    random_state=42,
)

study = Study(
    problem=problem,
    run_method=optimizer,
    evaluator=SequentialEvaluator[RecordCandidate, RecordCandidate](),
)

result, _ = study.optimize(max_evaluations=80)

best = result.best_observation
print(f"learning_rate: {best.candidate['learning_rate']:.5f}")
print(f"momentum:      {best.candidate['momentum']:.4f}")
print(f"objective:     {best.value:.6f}")
```

The candidate is a `RecordCandidate` mapping with the field names declared on
the `RecordSpace`. Use `candidate.as_dict()` when you need a plain dictionary.
You get back the same structure you put in — no index-based decoding needed.

## Why Structure Matters

A `RecordSpace` is not a thin naming layer over one anonymous vector. The
package uses the declared structure for:

- **validation** — out-of-bounds or wrong-type candidates are caught at
  ingress
- **log-scale awareness** — `scale="log"` on `learning_rate` means sampling,
  normalization, and local search all operate in log coordinates
- **diversity metrics** — CSA measures inter-candidate distance per field,
  respecting each field's scale and type
- **local-search neighborhoods** — structured kernels generate moves per
  leaf field rather than perturbing a flat vector

These stay aligned automatically. You do not need to write coordinate
transforms or custom distance functions to get sensible behavior from the
optimizer.

## Permutation Example

`PermutationSpace` models domains where the candidate is an ordering of
`0..N-1`. A classic example is minimizing a tour cost over a set of cities.

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
        for i in range(len(candidate)):
            total += DISTANCES[candidate[i]][candidate[(i + 1) % len(candidate)]]
        return total


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

study = Study(
    problem=Problem(space=space, objective=TourCostObjective()),
    run_method=optimizer,
    evaluator=SequentialEvaluator[tuple[int, ...], tuple[int, ...]](),
)

result, _ = study.optimize(max_evaluations=200)
print(f"best tour: {result.best_observation.candidate}")
print(f"tour cost: {result.best_observation.value}")
```

Permutation-specific operators like `OrderCrossover`, `SwapMutation`, and
`InversionMutation` are available from `variopt.algorithms.population`.

## Next

- conceptual model:
  [Spaces and Candidates](../concepts/spaces-and-candidates.md)
- local-search details:
  [Local Search](../concepts/local-search.md)
