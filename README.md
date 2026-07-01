# variopt

`variopt` is a typed optimization package for structured search spaces,
canonical candidates, and explicit execution boundaries.

See [CHANGELOG.md](CHANGELOG.md) for user-visible changes,
[docs/reference/stability.md](docs/reference/stability.md) for the public-API
stability policy, and [docs](docs/index.md) for the user-facing guide.

## Quickstart

```python
from typing_extensions import override

from variopt import IntegerSpace, Objective, OptimizationDirection, Problem, Study
from variopt.algorithms.population import CSAOptimizer
from variopt.evaluators import SequentialEvaluator


class SquareObjective(Objective[int]):
    @override
    def evaluate(self, candidate: int) -> float:
        return float(candidate * candidate)


space = IntegerSpace(-10, 10)

problem = Problem(
    space=space,
    objective=SquareObjective(),
    direction=OptimizationDirection.MINIMIZE,
)

optimizer = CSAOptimizer.from_space_defaults(
    space=space,
    bank_capacity=8,
    random_state=0,
)

study = Study(
    problem=problem,
    run_method=optimizer,
    evaluator=SequentialEvaluator[int, int](),
)

result, final_state = study.optimize(max_evaluations=40)

best = result.best_observation
print(f"best candidate: {best.candidate}, value: {best.value}")
```

`Study.optimize(...)` is the scalar optimization convenience path and returns
a [`RunResult`](docs/reference/api/artifacts.md). When a problem uses a
non-scalar [`EvaluationProtocol`](docs/reference/api/variopt.md), use
`Study.run(...)` instead to get a generic
[`RunReport`](docs/reference/api/artifacts.md).

## Evaluator Backends

For batch-parallel local execution, use the joblib-backed evaluator included
in the core install:

```python
from variopt.evaluators import JoblibEvaluator

study = Study(
    problem=problem,
    run_method=optimizer,
    evaluator=JoblibEvaluator[int, int](
        backend="threading",
        n_jobs=4,
    ),
)
```

For MPI-backed batch execution, install the optional mpi extra
(`pip install "variopt[mpi]"`) and use
[`MpiEvaluator`](docs/reference/api/evaluators.md).

## Documentation

The full documentation is organized as:

- **[Getting Started](docs/getting-started/introduction.md)** — installation,
  introduction, and quickstart
- **[Tutorials](docs/tutorials/index.md)** — worked end-to-end examples
- **[How-To Guides](docs/guides/index.md)** — task-oriented guidance for
  choosing optimizers, evaluators, presets, and local-search methods
- **[Concepts](docs/concepts/index.md)** — the model behind the API:
  spaces, problems, execution models, and algorithm families
- **[Reference](docs/reference/index.md)** — API surface, presets,
  checkpointing, glossary, and stability policy

### Key Entry Points

| Goal | Start here |
| --- | --- |
| Smallest runnable example | [Quickstart](docs/getting-started/quickstart.md) |
| End-to-end walkthrough | [First Optimization Run](docs/tutorials/first-optimization.md) |
| Structured (record/tuple/array) spaces | [Structured Spaces](docs/tutorials/structured-spaces.md) |
| Pick an optimizer family | [Choose an Optimizer](docs/guides/choose-an-optimizer.md) |
| CSA preset and profile customization | [Customize an Optimizer Profile](docs/guides/customize-optimizer-profile.md) |
| Local-search kernel guidance | [Local Optimization Methods](docs/guides/local-optimization-methods.md) |
| Candidate refinement provenance | [Candidate Refinement](docs/concepts/candidate-refinement.md) |
| Non-scalar / multi-objective patterns | [Canonical Usage Patterns](docs/guides/canonical-usage-patterns.md) |
| Public API reference | [API Surface](docs/reference/api.md) |
