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

Execution-related public artifacts are documented through the root facade and
`variopt.artifacts`: `CandidateRefinement`, `EvaluationOutcome`,
`EvaluationFailure`, `EvaluationExceptionSnapshot`, `EvaluationSuccess`,
`ObservationPayload`, `ObjectiveVectorPayload`, `EvaluationAttemptBatch`,
`EvaluationAttemptMaterializer`, `DefaultEvaluationAttemptMaterializer`,
`RunReport`, `RunResult`, and `NondominatedRunSurface`. The root execution
facade still exposes outcome-aware execution metadata such as
`EvaluationOutcome`; the artifact facade also exposes request-owned success and
payload artifacts for integrations that need to keep request identity separate
from objective payloads, plus the materializer protocol used to project
payload attempts into feedback records. Custom payload-to-record mappings should
provide an explicit `EvaluationAttemptMaterializer` when constructing a
`Study`; materializers may change successful payload representation, but must
not drop, reorder, flip, or rewrite attempt slots or their accounting and
provenance metadata. Run-method attempt assimilation exposes
`UnsupportedEvaluationFailureError` for optimizers that cannot safely consume
recorded failures; study orchestration exposes `RunExecutionFailed` for hard
failures with partial run state. Kernel implementation contracts used by those
examples, including `ProposalBatchQuery`, `KernelDiagnostics`, `KernelStatus`,
`ExecutionResources`, and `NestedParallelismPolicy`, are also root-facade names.
Supported diagnostics import paths are `from variopt import KernelDiagnostics,
KernelStatus` and `from variopt.artifacts import KernelDiagnostics,
KernelStatus`; `variopt.kernel` is not a supported diagnostics facade.
