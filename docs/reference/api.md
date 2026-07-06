# API Surface

The supported public facade modules are:

- `variopt`
- `variopt.spaces`
- `variopt.sampling`
- `variopt.diversity`
- `variopt.evaluators`
- `variopt.study`
- `variopt.artifacts`
- `variopt.algorithms.population`
- `variopt.algorithms.local_search`

Deeper submodules may be importable, but they are not automatically stable
public contract.

API reference generation is tracked separately so this page stays aligned with
the supported facade surface instead of exposing every internal package by
accident.

## Generated Facade Reference

The generated API pages are limited to the supported facade modules:

- [variopt](api/variopt.md)
- [variopt.spaces](api/spaces.md)
- [variopt.sampling](api/sampling.md)
- [variopt.diversity](api/diversity.md)
- [variopt.evaluators](api/evaluators.md)
- [variopt.study](api/study.md)
- [variopt.artifacts](api/artifacts.md)
- [variopt.algorithms.population](api/population.md)
- [variopt.algorithms.population.csa](api/csa.md)
- [variopt.algorithms.local_search](api/local-search.md)

Those pages intentionally stop at the facade boundary. Importable deep
submodules are not automatically part of the stable public contract.

The root facade exposes common direct-use artifacts and contracts such as
`CandidateRefinement`, `EvaluationOutcome`, `EvaluationFailure`,
`EvaluationExceptionSnapshot`, `EvaluationAttemptBatch`, `EvaluationRequest`,
`Observation`, `ObjectiveVectorRecord`, `Proposal`, `RunReport`, `RunResult`,
and `NondominatedRunSurface`. It also exposes common execution contracts such
as `ProposalBatchQuery`, `KernelDiagnostics`, `KernelStatus`,
`ExecutionResources`, and `NestedParallelismPolicy`.

Use `variopt.artifacts` for artifact-specific construction and projection
helpers that are not part of the trimmed root facade. That facade includes
`EvaluationAttempt`, `EvaluationSuccess`, `ObservationPayload`,
`ObjectiveVectorPayload`, `EvaluationAttemptMaterializer`,
`DefaultEvaluationAttemptMaterializer`, `materialize_success_record(...)`,
`materialize_success_records(...)`, and
`materialize_attempt_batch_records(...)`. Custom payload-to-record mappings
should provide an explicit `EvaluationAttemptMaterializer` when constructing a
`Study`; materializers may change successful payload representation, but must
not drop, reorder, flip, or rewrite attempt slots or their accounting and
provenance metadata.

Run-method attempt assimilation exposes `UnsupportedEvaluationFailureError` for
optimizers that cannot safely consume recorded failures; study orchestration
exposes `RunExecutionFailed` for hard failures with partial run state. Supported
diagnostics import paths are `from variopt import KernelDiagnostics,
KernelStatus` and `from variopt.artifacts import KernelDiagnostics,
KernelStatus`; `variopt.kernel` is not a supported diagnostics facade.

The `variopt.algorithms.population` facade also exposes the generational GA
state artifacts used by manual `ask(...)` / `tell(...)` loops:
`GenerationalGAOptimizerState`, `GenerationalGAPopulationMember`, and
`GenerationalGAVariant`. These are supported type-hint/runtime state artifacts
because the GA-family optimizer methods return and accept them directly.
Lifecycle helpers under `variopt.algorithms.population.generational_ga` remain
implementation details.
