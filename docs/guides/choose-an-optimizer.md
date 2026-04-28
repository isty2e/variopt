# Choose an Optimizer

Pick the optimizer family from the problem structure first, not from brand
recognition.

## Decision Table

| Search space | Scenario | Recommended starting point |
| --- | --- | --- |
| Continuous (`RealSpace` leaves) | General single-objective | `DifferentialEvolutionOptimizer` |
| Continuous, want richer policy surface | Multimodal or structured | `CSAOptimizer.from_space_defaults(...)` |
| Discrete or categorical | Single-objective | `GeneticAlgorithmOptimizer` |
| Mixed (real + integer + categorical) | Single-objective | `CSAOptimizer.from_space_defaults(...)` |
| Permutation | Combinatorial | `GeneticAlgorithmOptimizer` with permutation operators |
| Any, need diversity preservation | Multimodal with known niches | GA variant (see below) |

## Optimizer Families

### CSA

The richest built-in policy surface. Best default for structured mixed-type
spaces because `from_space_defaults(...)` derives a sampler, diversity metric,
and perturbation schedule from the declared space semantics automatically.

```python
from variopt.algorithms.population import CSAOptimizer

optimizer = CSAOptimizer.from_space_defaults(
    space=space, bank_capacity=12, random_state=0,
)
```

See [Concepts / CSA](../concepts/csa.md) and
[Customize an Optimizer Profile](customize-optimizer-profile.md).

### Differential Evolution

Good broad continuous baseline. Requires all leaves to be numeric.

```python
from variopt.algorithms.population import DifferentialEvolutionOptimizer

optimizer = DifferentialEvolutionOptimizer(
    space=space, population_size=20, random_state=0,
)
```

### Genetic Algorithm and Variants

The base `GeneticAlgorithmOptimizer` is a generational GA with tournament
selection and configurable crossover/mutation. The variants add explicit
diversity policies:

| Variant | When to use |
| --- | --- |
| `GeneticAlgorithmOptimizer` | Baseline population search |
| `ClearingGeneticAlgorithmOptimizer` | Prevent one niche from dominating — clears nearby winners each generation |
| `RestrictedTournamentGeneticAlgorithmOptimizer` | Maintain spatial diversity — replacement is restricted to nearby candidates |
| `SpeciesConservingGeneticAlgorithmOptimizer` | Preserve discovered species — each species keeps a representative |

The niching variants (`Clearing`, `RestrictedTournament`, `SpeciesConserving`)
require a [`DiversityMetric`][variopt.DiversityMetric]. For structured spaces,
the built-in `StructuredSpaceDiversityMetric` helper provides a sensible
default when you configure the variant directly.

## When To Add Local Search

Use a local-search kernel when you already have a global search method and
want bounded per-candidate refinement. Local search is composed with the
optimizer, not a replacement for it:

```python
study = Study(
    problem=problem,
    run_method=optimizer,
    evaluator=evaluator,
    kernel=some_local_search_kernel,
)
```

See [Local Optimization Methods](local-optimization-methods.md) for the
full decision table on kernel choice.

## Multi-Objective Problems

The built-in population optimizers are single-objective. For multi-objective
problems, `variopt` provides a record-first path:

1. Define an [`EvaluationProtocol`][variopt.EvaluationProtocol] that returns
   [`ObjectiveVectorRecord`][variopt.ObjectiveVectorRecord] instances.
2. Use `Study.run(...)` instead of `Study.optimize(...)` to get a
   [`RunReport`][variopt.RunReport].
3. Materialize a
   [`NondominatedRunSurface`][variopt.NondominatedRunSurface] from the
   report to extract the Pareto frontier.

See [Canonical Usage Patterns](canonical-usage-patterns.md#runreport-to-nondominatedrunsurface)
for the concrete code pattern.

## Related Reading

- [Concepts / CSA](../concepts/csa.md)
- [Concepts / Population Algorithms](../concepts/population-algorithms.md)
- [Concepts / Local Search](../concepts/local-search.md)
- [Customize an Optimizer Profile](customize-optimizer-profile.md) — change
  one or more CSA components behind `from_space_defaults`
