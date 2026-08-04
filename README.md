# variopt

[![CI](https://github.com/isty2e/variopt/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/isty2e/variopt/actions/workflows/ci.yml)
[![Docs](https://github.com/isty2e/variopt/actions/workflows/docs.yml/badge.svg?branch=main)](https://github.com/isty2e/variopt/actions/workflows/docs.yml)

`variopt` is a typed optimization package for structured search spaces,
canonical candidates, and explicit execution boundaries.

See [CHANGELOG.md](CHANGELOG.md) for user-visible changes,
[docs/reference/stability.md](docs/reference/stability.md) for the public-API
stability policy, and [docs](docs/index.md) for the user-facing guide.

## Quickstart

```python
from variopt import IntegerSpace, OptimizationDirection, Problem, Study
from variopt.algorithms.population import CSAOptimizer
from variopt.evaluators import SequentialEvaluator


def square(candidate: int) -> float:
    return float(candidate * candidate)


space = IntegerSpace(-10, 10)

problem = Problem(
    space=space,
    objective=square,
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

`Problem` accepts either a typed scalar callable or an explicit
[`Objective`](docs/reference/api/variopt.md) implementation. Prefer a
picklable, importable module-level function when the problem may cross a
process or MPI boundary; lambdas, closures, bound methods, and stateful
callable objects are not guaranteed to be portable across every evaluator
backend.

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

Synchronous `SequentialEvaluator` and `JoblibEvaluator` can also execute
eligible bounded local-search episodes per proposal. See
[Run Local Search in Evaluator Workers](docs/guides/request-local-episodes.md)
for setup and
[Evaluator Contracts](docs/reference/evaluator-contracts.md) for dispatch,
budget, failure, and reproducibility guarantees.

## Documentation

The full documentation is organized as:

- **[Getting Started](docs/getting-started/introduction.md)** — installation,
  introduction, and quickstart
- **[Tutorials](docs/tutorials/index.md)** — guided, reproducible learning paths
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
| Local-search kernel guidance | [Choose a Local Optimization Method](docs/guides/local-optimization-methods.md) |
| Parallel bounded local search | [Run Local Search in Evaluator Workers](docs/guides/request-local-episodes.md) |
| Stop and continue a CSA run | [Checkpoint and Resume CSA](docs/guides/checkpoint-and-resume-csa.md) |
| Candidate refinement provenance | [Candidate Refinement](docs/concepts/candidate-refinement.md) |
| Non-scalar / multi-objective patterns | [Advanced Usage Recipes](docs/guides/canonical-usage-patterns.md) |
| Public API reference | [API Surface](docs/reference/api.md) |
