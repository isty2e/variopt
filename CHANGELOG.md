# Changelog

All notable changes to `variopt` are documented here.

This project follows the [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
format. Stability guarantees for the public surface are documented in the
`Stability Policy` reference page (`docs/reference/stability.md`).

## [Unreleased]

### Breaking

- `Study.run(...)` and `Study.optimize(...)` now default
  `count_evaluation_cost=True`. Evaluation budgets are charged against reported
  logical evaluation cost, including inner local-search evaluations, rather than
  only the number of returned records. Code that intentionally wants outer-record
  counting must pass `count_evaluation_cost=False`.
- Study execution now raises `EvaluationBudgetExhausted` instead of silently
  assimilating a step whose reported evaluation cost exceeds the remaining hard
  budget.
- Structured spaces now enforce canonical candidate form consistently.
  Composite validation no longer coerces integer real leaves, categorical
  normalization returns the declared choice object, categorical validation
  rejects equal-but-different scalar runtime types, and non-finite categorical
  float choices or non-canonical structured scalar values are rejected at the
  relevant space boundary.

### Added

- Added `stop_at_checkpoint_boundary=True` for `Study.run(...)` and
  `Study.optimize(...)` so CSA runs can return the latest checkpoint-safe state
  when the budget ends inside an unsafe generation segment.
- Structured spaces now expose validated-candidate leaf traversal hooks used by
  built-in CSA, DE, local-search, geometry, and projection hot loops to avoid
  repeated full-candidate validation after an operation-level validation
  boundary.

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
