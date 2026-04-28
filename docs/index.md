# variopt

`variopt` is a typed optimization library for structured search spaces.

Define a search space with named fields, pick an optimizer, and run — the
package keeps your candidate structure intact from sampling through to the
result.

```python
from typing_extensions import override

from variopt import Objective, Problem, RealSpace, RecordSpace, Study
from variopt.algorithms.population import CSAOptimizer
from variopt.evaluators import SequentialEvaluator
from variopt.spaces import RecordCandidate


class SphereObjective(Objective[RecordCandidate]):
    @override
    def evaluate(self, candidate: RecordCandidate) -> float:
        x = float(candidate["x"])
        y = float(candidate["y"])
        return x * x + y * y


space = RecordSpace(
    x=RealSpace(-5.0, 5.0),
    y=RealSpace(-5.0, 5.0),
)

study = Study(
    problem=Problem(space=space, objective=SphereObjective()),
    run_method=CSAOptimizer.from_space_defaults(space=space, bank_capacity=12, random_state=0),
    evaluator=SequentialEvaluator(),
)
result, _ = study.optimize(max_evaluations=100)
print(result.best_observation.candidate.as_dict())  # {'x': ..., 'y': ...}
```

## Start Here

| Goal | Page |
| --- | --- |
| Install the package | [Installation](getting-started/installation.md) |
| One runnable example | [Quickstart](getting-started/quickstart.md) |
| End-to-end walkthrough | [First Optimization Run](tutorials/first-optimization.md) |
| Structured spaces tutorial | [Structured Spaces](tutorials/structured-spaces.md) |
| Multi-objective / non-scalar problems | [Canonical Usage Patterns](guides/canonical-usage-patterns.md) |
| Public API reference | [Reference](reference/index.md) |

## What's Included

- **Structured spaces** — `RealSpace`, `IntegerSpace`, `CategoricalSpace`,
  `RecordSpace`, `TupleSpace`, `ArraySpace`, `PermutationSpace`
- **Population optimizers** — CSA, Differential Evolution, Genetic Algorithm,
  and niching GA variants (Clearing, Restricted Tournament, Species Conserving)
- **Local-search kernels** — SciPy-backed continuous minimization, structured
  hill climb, stochastic neighborhood, variable neighborhood, iterated local
  search
- **Evaluator backends** — sequential, joblib (batch-parallel), async joblib,
  MPI
- **Study orchestration** — sync, exact-async, and stale-async execution
  models

## Documentation Map

- **[Getting Started](getting-started/introduction.md)** — install, intro,
  and quickstart
- **[Tutorials](tutorials/index.md)** — worked end-to-end examples
- **[How-To Guides](guides/index.md)** — task-oriented guidance (choosing
  optimizers, evaluators, profiles, local-search methods)
- **[Concepts](concepts/index.md)** — the model behind the API
- **[Reference](reference/index.md)** — API surface, presets, glossary,
  stability policy
