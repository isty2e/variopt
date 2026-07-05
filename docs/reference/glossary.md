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

The runtime pairing of an `EvaluationRecord` with the evaluation cost it
charged against the study budget, optional kernel diagnostics, and optional
candidate-refinement provenance. Produced by an `Evaluator` or kernel path,
consumed by `Study`, exposed to outcome-aware methods through
`RunMethod.tell_outcomes`, and lowered to its contained record before the
canonical `RunMethod.tell` path. See
[`EvaluationOutcome`][variopt.EvaluationOutcome].

## EvaluationFailure

A request-aligned record of a user-code evaluation failure. It keeps the
canonical `EvaluationRequest`, a JSON- and pickle-friendly
`EvaluationExceptionSnapshot`, and the logical evaluation cost consumed by the
failed attempt. It does not contain the raw exception object and is not a fake
`EvaluationRecord`. See [`EvaluationFailure`][variopt.EvaluationFailure].

## EvaluationAttemptBatch

The dense aggregate that aligns a batch of `EvaluationRequest` slots with
successful `EvaluationOutcome`s and recorded `EvaluationFailure`s. The aggregate
owns slot indices, duplicate-index rejection, request identity alignment, and
total evaluation-count accounting. `RunMethod.tell_attempts` consumes this
aggregate at the optimizer assimilation boundary. See
[`EvaluationAttemptBatch`][variopt.EvaluationAttemptBatch].

## DiversityMetric

The distance or dissimilarity contract used by diversity-aware search methods.
It is a search component, not part of the `SearchSpace` itself. See
[`DiversityMetric`][variopt.DiversityMetric].

## EvaluationProtocol

The per-problem rule that turns an `EvaluationRequest` into an
`EvaluationRecord`. Specialisations include `ScalarEvaluationProtocol`,
`ObservationEvaluationProtocol`, and `InteractionEvaluationProtocol`. See
[`EvaluationProtocol`][variopt.EvaluationProtocol].

## EvaluationRecord

The canonical output of one evaluation. Carries the originating `Proposal`,
the canonical candidate, and protocol-specific payload (score, vector, label,
interaction outcome, …). See
[`EvaluationRecord`][variopt.EvaluationRecord].

## EvaluationRequest

The wrapped `Proposal` that an `Evaluator` receives and forwards to the
`EvaluationProtocol`. See
[`EvaluationRequest`][variopt.EvaluationRequest].

## Evaluator

The component that owns execution mechanics — how a batch of
`EvaluationRequest`s becomes successful `EvaluationOutcome`s or, through
built-in evaluator `evaluate_attempts` hooks, a dense `EvaluationAttemptBatch`
that preserves recorded user-code `EvaluationFailure`s. Backends include
`SequentialEvaluator`, `JoblibEvaluator`, `AsyncJoblibEvaluator`, and
`MpiEvaluator`. See
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

The multi-objective terminal sibling of `RunResult`. Materialised from a
`RunReport` of `ObjectiveVectorRecord`s; exposes the non-dominated candidate
set while preserving record-aligned `CandidateRefinement` provenance when the
source report carried it. Recorded evaluation failures remain separate from the
frontier through `NondominatedRunSurface.failures`. See
[`NondominatedRunSurface`][variopt.NondominatedRunSurface].

## Objective

The scalar-valued function paired with a `Problem` for simple scalar
optimization. For non-scalar problems, use an `EvaluationProtocol` directly.
See [`Objective`][variopt.Objective].

## ObjectiveVectorRecord

The record type for multi-objective problems. Carries a tuple of objective
values and their `OptimizationDirection`s. See
[`ObjectiveVectorRecord`][variopt.ObjectiveVectorRecord].

## Observation

The scalar-record sibling of `EvaluationRecord` used by
`ScalarEvaluationProtocol` and `Study.optimize`. Carries a single value and
its optimization direction. See [`Observation`][variopt.Observation].

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

The search-state owner. Proposes candidates via `ask`, consumes records via
`tell`, may opt into full outcome metadata through `tell_outcomes`, consumes
dense attempt batches through `tell_attempts`, and owns the persistent
search-state object. Population optimizers
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

The generic terminal report produced by `Study.run(...)`. Covers any
`EvaluationRecord` type and may carry record-aligned
`CandidateRefinement` provenance when a kernel or evaluator changed candidates
before evaluation. Recorded evaluation failures are exposed separately through
`RunReport.failures`; they are not mixed into `RunReport.records`. See
[`RunReport`][variopt.RunReport].

## RunResult

The scalar terminal result produced by `Study.optimize(...)`. Covers scalar
`Observation` records only and preserves observation-aligned
`CandidateRefinement` provenance when local refinement changed evaluated
candidates. Recorded evaluation failures are exposed separately through
`RunResult.failures`; they are not mixed into `RunResult.observations`. See
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
search semantics. See [`Study`][variopt.Study].

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
