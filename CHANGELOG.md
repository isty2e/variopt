# Changelog

All notable changes to `variopt` are documented here.

This project follows the [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
format. Stability guarantees for the public surface are documented in the
`Stability Policy` reference page (`docs/reference/stability.md`).

## [Unreleased]

No unreleased changes yet.

## [0.1.0] - 2026-06-15

Initial tagged release of the public `variopt` package surface.

### Added

- **Structured spaces** under `variopt.spaces`: `RealSpace`, `IntegerSpace`,
  `CategoricalSpace`, `RecordSpace`, `ArraySpace`, and the
  `StructuredSearchSpace` protocol.
- **Study orchestration** under `variopt.study`: `Study.optimize(...)` for
  scalar problems and `Study.run(...)` for generic evaluation-protocol
  problems.
- **CSA** under `variopt.algorithms.population`: `CSAOptimizer` with the
  `variopt` and `joung_2018` presets and a `from_space_defaults(...)`
  space-derived entry point. Advanced policy and schedule types are exposed
  under `variopt.algorithms.population.csa`.
- **Native population optimizers**: `DifferentialEvolutionOptimizer`,
  `GeneticAlgorithmOptimizer`, `ClearingGeneticAlgorithmOptimizer`,
  `RestrictedTournamentGeneticAlgorithmOptimizer`, and
  `SpeciesConservingGeneticAlgorithmOptimizer`.
- **Local-search kernels**: `ScipyMinimizeKernel` for continuous structured
  spaces (`L-BFGS-B`, `Powell`) and structured discrete local-search kernels
  for hill-climb, stochastic-neighborhood, scheduled, variable-neighborhood,
  and iterated local search.
- **Evaluators** under `variopt.evaluators`: `SequentialEvaluator`,
  `JoblibEvaluator`, `AsyncJoblibEvaluator`, and `MpiEvaluator` (via the
  `[mpi]` extra).
