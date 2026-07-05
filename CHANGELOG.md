# Changelog

All notable changes to `variopt` are documented here.

This project follows the [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
format. Stability guarantees for the public surface are documented in the
`Stability Policy` reference page (`docs/reference/stability.md`).

## [Unreleased]

### Breaking

- Removed the obsolete generic request-aligned record API from the root and
  artifact facades. `EvaluationRecord`, `InteractionEvaluationRecord`, and
  `RequestAlignedEvaluationRecord` are no longer public entry points; use
  request-free protocol payloads plus `EvaluationSuccess`/terminal artifacts, or
  the concrete `Observation` and `ObjectiveVectorRecord` compatibility
  projections where those concrete views are required.
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
- `Study` orchestration now requires native attempt-aware evaluator capability.
  Custom evaluators used through `Study` must expose `evaluate_attempts(...)`
  for synchronous execution, or attempt-batch session hooks for async execution.
  Direct `Evaluator.evaluate(...)` and legacy `EvaluationOutcome` streams remain
  available on the evaluator facade, but `Study` no longer adapts outcome-only
  batches or sessions into attempt batches.
- `Study` now separates evaluator/kernel payload attempts from run-method
  feedback records. Scalar and vector protocols may return request-free
  `ObservationPayload` or `ObjectiveVectorPayload` attempts; `Study`
  materializes successful payloads into request-aligned records immediately
  before `RunMethod.tell_attempts(...)` in sync, exact-async, and stale-async
  execution. Custom materializers must preserve attempt slot count, slot order,
  success/failure variants, request identity, evaluation counts, refinements,
  diagnostics, and failure metadata while projecting successful payloads into
  request-aligned records. The public `Study[...]` generic type therefore has
  separate payload and feedback-record axes.

### Added

- Added `stop_at_checkpoint_boundary=True` for `Study.run(...)` and
  `Study.optimize(...)` so CSA runs can return the latest checkpoint-safe state
  when the budget ends inside an unsafe generation segment.
- Structured spaces now expose validated-candidate leaf traversal hooks used by
  built-in CSA, DE, local-search, geometry, and projection hot loops to avoid
  repeated full-candidate validation after an operation-level validation
  boundary.
- Added `EvaluationFailure`, `EvaluationExceptionSnapshot`, and
  `EvaluationAttemptBatch` artifacts for request-aligned recording of user-code
  evaluation failures without mixing them into successful records.
- Added `RunMethod.tell_attempts(...)` and
  `UnsupportedEvaluationFailureError` so optimizers can distinguish success-only
  attempt batches from recorded evaluation failures at the assimilation boundary.
- Added `RunExecutionFailed` so hard study execution failures carry partial
  report/state and checkpoint-safe report/state projections when available.
- Built-in sequential, joblib, async joblib, and MPI evaluators now expose
  `evaluate_attempts(...)` hooks that return `EvaluationAttemptBatch` values and
  preserve user-code evaluation failures separately from successful outcomes.
- Direct and built-in local-search kernels now use `EvaluationAttemptBatch`
  runner/result contracts so failed local-search trials remain visible instead
  of being collapsed into successful optimized outcomes.
- Outcome-aware `EvaluationAttemptBatch` now stores ordered attempt slots as its
  authoritative state and exposes request/outcome/failure index views as derived
  projections.
- Added `EvaluationAttemptMaterializer` and
  `DefaultEvaluationAttemptMaterializer` for integrations that need to describe
  the typed payload-attempt to feedback-record projection used by `Study`
  orchestration.

### Fixed

- `CSAOptimizer` now consumes recorded failed attempts from pending proposal,
  generation, and proposal-attribution lifecycle state without inserting failed
  candidates into CSA banks or adaptive score evidence. GA and DE-family
  population optimizers continue to reject failure assimilation explicitly until
  a partial-generation policy is defined for them.

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
