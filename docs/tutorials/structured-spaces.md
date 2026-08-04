# Structured Spaces

In this tutorial, we will optimize two named hyperparameters without flattening
them into an anonymous vector. The candidate will remain a `RecordCandidate`
from sampling through result inspection.

Complete [First Optimization Run](first-optimization.md) first if `Problem`,
`Study`, and an evaluator are still unfamiliar.

## Declare the candidate structure

Define a learning rate on a logarithmic scale and a momentum value on a linear
scale:

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
```

The field names and leaf-space semantics are now part of the search domain.
Sampling, validation, distance measurement, and structured local search can all
use the same declaration.

## Define an objective over the record

The objective receives a mapping-like `RecordCandidate`, so access values by
name rather than by coordinate index:

```python
class MockTrainingObjective(Objective[RecordCandidate]):
    @override
    def evaluate(self, candidate: RecordCandidate) -> float:
        learning_rate = float(candidate["learning_rate"])
        momentum = float(candidate["momentum"])
        return (learning_rate - 0.01) ** 2 + (momentum - 0.9) ** 2


problem = Problem(
    space=space,
    objective=MockTrainingObjective(),
)
```

There is no decode step. The objective sees the same named structure that the
space declared.

## Derive an optimizer from the space

CSA can derive its sampler, diversity metric, and perturbation schedule from
the structured space:

```python
optimizer = CSAOptimizer.from_space_defaults(
    space=space,
    bank_capacity=10,
    random_state=42,
)
```

The logarithmic learning-rate coordinate remains logarithmic in those derived
components. It is not treated as a linearly scaled raw float.

## Run and inspect the structured result

```python
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

The `variopt` 0.2.0 reference environment produced:

```text
learning_rate: 0.00015
momentum:      0.9094
objective:     0.000186
```

The final digits can vary with dependency versions. The candidate must retain
the declared fields and satisfy both leaf spaces.

Use `best.candidate.as_dict()` when an external API needs a plain dictionary:

```python
print(best.candidate.as_dict())
```

The output retains both declared fields:

```text
{'learning_rate': 0.00014936568554617635, 'momentum': 0.9094403824985425}
```

You have now used one structural declaration for validation, optimization, and
result access without writing coordinate transforms or decode functions.

## Next steps

- [Spaces and Candidates](../concepts/spaces-and-candidates.md) explains how
  structured spaces participate across the pipeline.
- [Optimize a Permutation](../guides/optimize-a-permutation.md) applies the
  workflow to ordered combinatorial candidates.
- [Customize an Optimizer Profile](../guides/customize-optimizer-profile.md)
  shows how to override selected CSA defaults.
