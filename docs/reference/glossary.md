# Glossary

Short canonical definitions of the core `variopt` vocabulary. Each term links
to the reference page for the underlying symbol. For the narrative explanation
of how these pieces fit together, see
[Optimization Model](../concepts/optimization-model.md).

## CandidateRefinement

Execution-side provenance for a candidate transformed before evaluation.
Carries the source candidate, the candidate actually evaluated, and any known
structured leaf paths changed by refinement. See
[`CandidateRefinement`][variopt.CandidateRefinement].

## EvaluationOutcome

The legacy compatibility pairing of one successful request-aligned payload with
the logical evaluation cost, optional kernel diagnostics, and optional
candidate-refinement provenance. `EvaluationOutcome` remains available for
direct evaluator compatibility APIs that expose successful outcome metadata.
`Study` execution boundaries use `EvaluationSuccess` / `EvaluationFailure`
attempt slots instead. See [`EvaluationOutcome`][variopt.EvaluationOutcome].

## EvaluationFailure

A request-aligned record of a user-code evaluation failure. It keeps the
canonical `EvaluationRequest`, a JSON- and pickle-friendly
`EvaluationExceptionSnapshot`, and the logical evaluation cost consumed by the
failed attempt. It does not contain the raw exception object and is not a fake
successful payload. See [`EvaluationFailure`][variopt.EvaluationFailure].

## EvaluationSuccess

A request-owned successful attempt artifact. It keeps the canonical
`EvaluationRequest`, a payload, the logical evaluation cost, and optional
`CandidateRefinement` provenance aligned against `request.candidate`. New
problem protocols normally return request-free payloads such as
`ObservationPayload` or `ObjectiveVectorPayload`; compatibility paths may carry
request-aligned record payloads. See
[`EvaluationSuccess`][variopt.artifacts.EvaluationSuccess].

## EvaluationAttemptBatch

The ordered aggregate whose slots are exactly `EvaluationSuccess` or
`EvaluationFailure` values. It owns request-slot order, success/failure
projections, successful payload projection, and total evaluation-count
accounting. Ordered attempt slots are the authoritative model; success and
failure index views are derived projections, not parallel storage. `Study`
transports payload attempt batches through evaluator and kernel execution, then
materializes successful payloads into feedback records at the run-method
boundary. See
[`EvaluationAttemptBatch`][variopt.artifacts.EvaluationAttemptBatch].

## DiversityMetric

The distance or dissimilarity contract used by diversity-aware search methods.
It is a search component, not part of the `SearchSpace` itself. See
[`DiversityMetric`][variopt.DiversityMetric].

## EvaluationProtocol

The per-problem rule that turns an `EvaluationRequest` into an
request-free payload. Scalar specialisations produce `ObservationPayload`
values; execution layers own request identity, success/failure attempt
recording, and compatibility projection into request-aligned payloads while
legacy record-consuming APIs remain. Specialisations include
`ScalarEvaluationProtocol`, `ObservationEvaluationProtocol`, and
`InteractionEvaluationProtocol`. See
[`EvaluationProtocol`][variopt.EvaluationProtocol].

## ObservationPayload

The request-free scalar objective payload: raw objective value, canonical
minimization score, and optional elapsed time. Request identity belongs to
`EvaluationSuccess`, not to the payload. See
[`ObservationPayload`][variopt.artifacts.ObservationPayload].

## ObjectiveVectorPayload

The request-free multi-objective payload: raw objective vector, canonical
minimization score vector, and optional elapsed time. Request identity belongs
to `EvaluationSuccess`, not to the payload. See
[`ObjectiveVectorPayload`][variopt.artifacts.ObjectiveVectorPayload].

## EvaluationRequest

The wrapped `Proposal` that an `Evaluator` receives and forwards to the
`EvaluationProtocol`. See
[`EvaluationRequest`][variopt.EvaluationRequest].

## Evaluator

The component that owns execution mechanics. The direct evaluator compatibility
API can turn a batch of `EvaluationRequest`s into successful
`EvaluationOutcome`s. `Study` orchestration requires native attempt-aware
capability instead: synchronous evaluators expose `evaluate_attempts(...)`, and
async evaluators expose attempt-batch session hooks, so recorded user-code
`EvaluationFailure`s remain separate from successful attempts. Those Study
attempts may carry request-free scalar/vector payloads; Study materializes them
before run-method feedback. Backends include `SequentialEvaluator`,
`JoblibEvaluator`, `AsyncJoblibEvaluator`, and `MpiEvaluator`. See
[`Evaluator`][variopt.Evaluator].

## ExecutionResources

Request-local resource ownership passed to kernels, including the current
parallel owner, nested-parallelism policy, worker count, and backend label. See
[`ExecutionResources`][variopt.ExecutionResources].

## Kernel

An optional bounded-episode component that turns one proposal batch into one
locally improved `EvaluationAttemptBatch`. Kernels own local search, not global
search. See [`Kernel`][variopt.Kernel].

## KernelDiagnostics

Execution-facing diagnostics for one kernel episode, including backend, method,
status, and optional message. See
[`KernelDiagnostics`][variopt.KernelDiagnostics].

## KernelStatus

The terminal status reported by one kernel episode: converged, stopped, or
failed. See [`KernelStatus`][variopt.KernelStatus].

## NestedParallelismPolicy

The request-local policy that tells kernels whether nested parallel work is
allowed below the current execution owner. See
[`NestedParallelismPolicy`][variopt.NestedParallelismPolicy].

## NondominatedRunSurface

The multi-objective terminal sibling of `RunResult`. Stores vector-valued
`EvaluationSuccess` history, exposes a stable non-dominated success frontier,
and keeps vector records as compatibility projections. `CandidateRefinement`
provenance aligns with the source successes. Recorded evaluation failures
remain separate from the frontier through `NondominatedRunSurface.failures`. See
[`NondominatedRunSurface`][variopt.NondominatedRunSurface].

## Objective

The scalar-valued function paired with a `Problem` for simple scalar
optimization. For non-scalar problems, use an `EvaluationProtocol` directly.
See [`Objective`][variopt.Objective].

## ObjectiveVectorRecord

The request-aligned record projection for multi-objective problem results.
Carries objective values and canonical minimization scores for the evaluated
candidate. Its request may preserve the source proposal when refinement changed
the evaluated candidate. See
[`ObjectiveVectorRecord`][variopt.ObjectiveVectorRecord].

## Observation

The scalar request-aligned record projection used by legacy outcome/report
boundaries and `Study.optimize`. Scalar protocols produce `ObservationPayload`
first; execution layers attach request identity when projecting to
`Observation`. Its `candidate` is the evaluated candidate; its request may
preserve the source proposal when refinement changed the evaluated candidate.
See [`Observation`][variopt.Observation].

## OptimizationDirection

The `MINIMIZE` / `MAXIMIZE` tag that turns a raw objective value into a
comparable score. See
[`OptimizationDirection`][variopt.OptimizationDirection].

## Problem

The bound pairing of one `SearchSpace` with one evaluation contract (either
an `Objective` for scalar problems or a full `EvaluationProtocol` for generic
problems). See [`Problem`][variopt.Problem].

## Proposal

A candidate offered by a `RunMethod` for evaluation, carrying an optional
`proposal_id` for traceability. See [`Proposal`][variopt.Proposal].

## ProposalEvaluationSpec

Immutable request-local metadata attached when a `Proposal` is lowered into an
`EvaluationRequest`. Use it for per-request execution meaning such as fidelity
or resume provenance, not for multi-request interaction semantics. See
[`ProposalEvaluationSpec`][variopt.artifacts.ProposalEvaluationSpec].

## ProposalBatchQuery

The kernel query object containing the problem, proposal batch, execution
resources, optional evaluation specs, and optional kernel hints for one bounded
episode. See [`ProposalBatchQuery`][variopt.ProposalBatchQuery].

## ProposalKernelHint

The marker base for immutable per-proposal hints passed from a run method to a
kernel. See [`ProposalKernelHint`][variopt.ProposalKernelHint].

## ProposalLocalSearchContext

The built-in kernel hint for per-proposal local-search enablement, local budget,
and prioritized structured leaf paths. See
[`ProposalLocalSearchContext`][variopt.ProposalLocalSearchContext].

## RunMethod

The search-state owner. Proposes candidates via `ask`, consumes successful
request-aligned records via `tell`, may opt into legacy successful-outcome
metadata through `tell_outcomes`, consumes dense materialized record-attempt
batches through `tell_attempts`, and owns the persistent search-state object.
Population optimizers
(`CSAOptimizer`, `DifferentialEvolutionOptimizer`,
`GeneticAlgorithmOptimizer`) are `RunMethod` implementations. See
[`RunMethod`][variopt.RunMethod].

## UnsupportedEvaluationFailureError

Raised by the default `RunMethod.tell_attempts` implementation when a dense
attempt batch contains recorded failures and the concrete optimizer has not
defined how failed proposal lifecycle should affect its state. `CSAOptimizer`
overrides this hook for failed proposal cleanup; GA and DE-family optimizers
currently keep the default rejection contract. See
[`UnsupportedEvaluationFailureError`][variopt.UnsupportedEvaluationFailureError].

## RunExecutionFailed

Raised when study orchestration hits a hard evaluator, backend, or assimilation
failure that cannot be represented as a recorded `EvaluationFailure`. Carries a
`partial_report` and `partial_state` for fully assimilated work, plus the latest
checkpoint-safe report and state when one was reached. See
[`RunExecutionFailed`][variopt.RunExecutionFailed].

## RunReport

The generic terminal report produced by `Study.run(...)`. Stores ordered
request-owned `EvaluationSuccess` history. For Study-produced reports,
successful payloads have already been materialized into request-aligned feedback
records, so `records` is the legacy record projection. If you construct a report
manually from arbitrary request-free payloads, read those payloads through
`successes` unless they are built-in scalar/vector payloads or already
request-aligned records. `CandidateRefinement` provenance aligns with successes
when a kernel or evaluator changed candidates before evaluation. Recorded
evaluation failures are exposed separately through `RunReport.failures`; they
are not mixed into `RunReport.successes` or `RunReport.records`. See
[`RunReport`][variopt.RunReport].

## RunResult

The scalar terminal result produced by `Study.optimize(...)`. Stores scalar
`EvaluationSuccess` history, identifies the best success by score, and exposes
`observations` as the scalar compatibility projection. `CandidateRefinement`
provenance aligns with successes when local refinement changed evaluated
candidates. Recorded evaluation failures are exposed separately through
`RunResult.failures`; they are not mixed into `RunResult.successes` or
`RunResult.observations`. See
[`RunResult`][variopt.RunResult].

## SearchSpace

The abstract container for candidate points. Owns normalization, validation,
and sampling. See [`SearchSpace`][variopt.SearchSpace].

## StructuredSearchSpace

A `SearchSpace` specialisation with a declared topology of named leaf spaces.
Enables geometry-aware sampling, diversity metrics, and local-search kernels.
See [`StructuredSearchSpace`][variopt.spaces.StructuredSearchSpace].

## Study

The orchestration layer that wires a `Problem`, `RunMethod`, optional
`Kernel`, and `Evaluator` into one `optimize` or `run` call. Does not own
search semantics. It transports evaluator/kernel payload attempts, then
materializes successful payloads into request-aligned records immediately before
run-method feedback. See [`Study`][variopt.Study].

## Trace

Append-only terminal diagnostics attached to `RunReport` and `RunResult`.
Individual `TraceEvent` entries record the executed proposal ids, evaluated
candidates, and optional diagnostic messages. See
[`Trace`][variopt.artifacts.Trace].

## VariationOperator

The candidate transformation contract used by population methods and CSA
perturbation schedules. Operators transform candidates; they do not evaluate
objectives or own search state. See
[`VariationOperator`][variopt.VariationOperator].
