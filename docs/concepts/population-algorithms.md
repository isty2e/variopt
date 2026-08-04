# Population Algorithms

Population algorithms search with a collection of candidates rather than one
current point. In `variopt`, CSA, Differential Evolution, and the GA family all
implement the same run-method boundary, but they organize population memory and
variation differently.

## Differential Evolution

Differential Evolution treats population differences as search directions. It
combines numeric candidates through mutation and recombination, making it a
natural broad baseline for continuous spaces.

Its main control surface is compact: population size, mutation range,
recombination probability, and crossover count. That simplicity is useful when
the domain is continuous and no explicit niche model is required.

## Genetic algorithms

The GA family applies explicit crossover and mutation operators, then replaces
the current generation under selection and elitism rules. Operator choice can
follow candidate structure: permutation spaces use permutation-preserving
operators, while numeric and structured spaces use operators compatible with
their own canonical values.

The niching variants change how diversity survives selection:

- Clearing GA suppresses nearby lower-ranked members after a niche winner is
  selected.
- Restricted Tournament GA limits replacement competition to nearby members.
- Species Conserving GA preserves representatives of discovered species.

These variants require a diversity metric because distance is part of their
selection law rather than a reporting-only statistic.

## CSA

CSA maintains a diverse elite archive, called the bank, instead of replacing a
whole population every generation. A distance cutoff anneals over the run and
controls whether a candidate competes locally with an existing bank entry or
enters through a far-update route.

This makes diversity an explicit part of optimizer state. It also creates more
configuration surface than DE or the base GA: perturbation families, bank
updates, cutoff progression, and optional adaptation policies all have distinct
roles.

## Choosing among them

The shared `RunMethod` contract means evaluator and study orchestration do not
need to change with the algorithm family. The choice is therefore about search
memory and variation semantics:

- start with DE for a compact continuous baseline
- use GA when explicit crossover and mutation operators fit the domain
- use a niching GA when a specific diversity-preservation law is justified
- use CSA when structured-space defaults and archive-centered diversity are
  central to the search

See [Choose an Optimizer](../guides/choose-an-optimizer.md) for the practical
decision table, [CSA](csa.md) for the bank lifecycle, and
[the population API](../reference/api/population.md) for constructor details.
