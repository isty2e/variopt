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
  request-free protocol payloads plus `variopt.artifacts.EvaluationSuccess`
  and terminal artifacts, or the concrete `Observation` and
  `ObjectiveVectorRecord` compatibility projections where those concrete views
  are required.
- `KernelDiagnostics` and `KernelStatus` are supported through the root facade
  and `variopt.artifacts` only. Imports such as
  `from variopt.kernel import KernelDiagnostics` now fail; use
  `from variopt import KernelDiagnostics, KernelStatus` or
  `from variopt.artifacts import KernelDiagnostics, KernelStatus` instead.
- `Study.run(...)` and `Study.optimize(...)` now default
  `count_evaluation_cost=True`. Evaluation budgets are charged against reported
  logical evaluation cost, including inner local-search evaluations, rather than
  only the number of returned attempt slots. Code that intentionally wants
  outer-attempt-slot counting must pass `count_evaluation_cost=False`.
- Study execution now raises `EvaluationBudgetExhausted` instead of silently
  assimilating a step whose reported evaluation cost exceeds the remaining hard
  budget. When `stop_at_checkpoint_boundary=True` and a checkpoint-safe snapshot
  has already been reached, that snapshot is returned instead of assimilating
  the over-budget step.
- Structured spaces now enforce canonical candidate form consistently.
  Composite validation no longer coerces integer real leaves, categorical
  normalization returns the declared choice object, categorical validation
  rejects equal-but-different scalar runtime types, and non-finite categorical
  float choices or non-canonical structured scalar values are rejected at the
  relevant space boundary.
- Structured space declarations now reject more malformed metadata at
  construction time. `RealSpace` bounds must be canonical `float` values,
  `ArraySpace.length` must be a positive canonical `int`, and structured
  candidate JSON codecs reject cyclic or excessively deep container payloads
  instead of relying on interpreter recursion limits.
- CSA checkpoint JSON codecs now fail loudly on malformed numeric payloads.
  Boolean values are no longer accepted where checkpoint fields require JSON
  integers or numbers, and non-finite bank/growth/clustering numeric values are
  rejected instead of being restored as runtime state. Durable checkpointing is
  the explicit JSON-safe `to_dict()` / `from_dict()` surface; pickle is not a
  supported persistence or compatibility boundary.
- Runtime artifact ingress now rejects malformed numeric and refinement
  provenance payloads consistently. `Observation.from_objective_value(...)`,
  `ObjectiveVectorRecord`, and async evaluator `wait(...)` boundaries reject
  booleans-as-numbers and non-finite floats, and `CandidateRefinement` requires
  `changed_leaf_paths` to be a sequence of tuple leaf paths with canonical
  `int` or `str` segments.
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
- Custom `RunMethod` implementations that need `Study` feedback must now
  consume `tell_attempts(EvaluationAttemptBatch)`. The previous outcome-stream
  assimilation hook, including `tell_outcomes(...)` implementations, is no
  longer adapted by `Study`; third-party `RunMethod` subclasses should override
  `tell_attempts(...)` directly, especially when recorded failures require
  proposal cleanup or partial-generation handling.
- Native GA, clearing GA, species-conserving GA, and restricted-tournament GA
  manual `ask(...)` / `tell(...)` loops now share
  `variopt.algorithms.population.GenerationalGAOptimizerState` instead of
  variant-local state classes such as `GAOptimizerState`,
  `ClearingGAOptimizerState`, `SpeciesGAOptimizerState`, and
  `RestrictedTournamentGAOptimizerState`. Import state artifacts from
  `variopt.algorithms.population`; deep variant `state` modules are no longer
  present. Code that inspected state internals should use
  `buffered_member_buffer` instead of `buffered_members`, and should read
  remaining queued proposals through `queued_proposals[queued_proposal_index:]`.
- `Problem(..., direction=...)` is now scalar-only constructor input. Pass
  `direction` only with `objective`, `Objective`, or
  `ObservationEvaluationProtocol` inputs; direction-free `EvaluationProtocol`
  implementations must omit it or pass `None`. Internally,
  `Problem.direction` remains a concrete `OptimizationDirection` and defaults
  unspecified scalar directions to `OptimizationDirection.MINIMIZE`.

### Added

- Added a scheduled and manually dispatchable dependency canary workflow that
  resolves the latest compatible dependencies, builds the docs and wheel, and
  smokes the installed base and MPI extras without treating `uv.lock` as release
  metadata.
- `variopt.algorithms.population` now exposes
  `GenerationalGAOptimizerState`, `GenerationalGAMemberBuffer`,
  `GenerationalGAPopulationMember`, and `GenerationalGAVariant` as supported
  type-hint/runtime state artifacts for manual generational-GA `ask(...)` /
  `tell(...)` loops. The state uses a queued-proposal cursor and immutable
  member buffer to avoid repeated split-batch tuple copies.
- Added `stop_at_checkpoint_boundary=True` for `Study.run(...)` and
  `Study.optimize(...)` so CSA runs can return the latest checkpoint-safe state
  when the budget ends or is exhausted inside an unsafe generation segment.
- Structured spaces now expose validated-candidate leaf traversal hooks used by
  built-in CSA, DE, local-search, geometry, and projection hot loops to avoid
  repeated full-candidate validation after an operation-level validation
  boundary.
- Added `EvaluationAttempt`, `EvaluationSuccess`, `EvaluationFailure`,
  `EvaluationExceptionSnapshot`, and `EvaluationAttemptBatch` artifacts for
  request-aligned recording of successful evaluations and user-code evaluation
  failures without mixing failures into successful records.
- Added request-free `ObservationPayload` and `ObjectiveVectorPayload`
  artifacts plus `materialize_success_record(...)`,
  `materialize_success_records(...)`, and
  `materialize_attempt_batch_records(...)` helpers for projecting successful
  payload attempts into request-aligned feedback records.
- Added `RunMethod.tell_attempts(...)` and
  `UnsupportedEvaluationFailureError` so optimizers can distinguish success-only
  attempt batches from recorded evaluation failures at the assimilation boundary.
- Added `RunExecutionFailed` so hard study execution failures carry partial
  report/state and checkpoint-safe report/state projections when available.
- CI now covers Python 3.10 through 3.13, smokes built wheel imports from an
  installed environment, checks joblib/loky private retry surfaces for drift,
  and verifies the optional MPI extra can be installed and imported.
- Built-in sequential, joblib, async joblib, and MPI evaluators now expose
  `evaluate_attempts(...)` hooks that return `EvaluationAttemptBatch` values and
  preserve user-code evaluation failures separately from successful attempts.
- Direct and built-in local-search kernels now use `EvaluationAttemptBatch`
  runner/result contracts. Failed inner local-search trials are charged to the
  visible top-level attempt's `evaluation_count` and summarized in
  `KernelDiagnostics`; if no local-search attempt succeeds, the top-level slot
  remains an `EvaluationFailure`.
- Attempt-aware `EvaluationAttemptBatch` now stores ordered attempt slots as its
  authoritative state and exposes request/outcome/failure index views as derived
  projections.
- Added `EvaluationAttemptMaterializer` and
  `DefaultEvaluationAttemptMaterializer` for integrations that need to describe
  the typed payload-attempt to feedback-record projection used by `Study`
  orchestration.

### Fixed

- `TupleSpace` and `RecordSpace` now compare by their declared child spaces
  instead of object identity, so composite `Problem` equality matches scalar
  and array space value-object behavior.
- CSA bank distance queries now use the structured validated-candidate geometry
  path after the optimizer validates observed candidates at the `tell(...)`
  boundary, avoiding repeated public distance validation in canonical bank hot
  loops.
- CSA now preserves custom structured diversity metric overrides and rejects
  built-in structured diversity metrics whose space does not match the optimizer
  space before using the validated-distance fast path.
- `Trace(events=...)` now copies mutable event sequences into the immutable
  tuple form used by terminal artifacts, matching the other run-artifact
  constructors.
- Built-in local-search kernels now preserve inner failed-attempt diagnostics
  even when the successful attempt itself did not emit base kernel diagnostics,
  while still omitting diagnostics objects that carry no signal.
- Checkpoint-safe `Study.run(...)` execution now stores safe snapshots as
  history cut points instead of eagerly rebuilding full success, failure, and
  trace tuples at every safe step.
- Stale-async `Study.run(...)` and `Study.optimize(...)` now reject negative
  `max_evaluations` before opening evaluator sessions, matching sync execution.
- Stale-async `RunExecutionFailed.partial_report` now includes completed groups
  that were materialized before `RunMethod.tell_attempts(...)` failed, while
  `partial_state` remains at the pre-assimilation state.
- Stale-async runs with `stop_at_checkpoint_boundary=True` no longer open
  refill sessions after reaching the requested checkpoint-safe boundary; any
  already-active sessions are cancelled before returning the safe report/state.
- Terminal artifact pickle restoration now rejects mismatched current-state
  field counts and exact-shape states that violate terminal accounting,
  best-success, or nondominated-frontier invariants.
- Hard-failure checkpoint-safe report construction now reuses the checkpoint
  cut-point projection while preserving the original execution exception if
  recovery report materialization fails.
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
