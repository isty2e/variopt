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
`EvaluationRequest`s becomes a batch of `EvaluationOutcome`s. Backends include
`SequentialEvaluator`, `JoblibEvaluator`, and `MpiEvaluator`. See
[`Evaluator`][variopt.Evaluator].

## Kernel

An optional bounded-episode component that turns one proposal batch into one
locally improved report. Kernels own local search, not global search. See
[`Kernel`][variopt.Kernel].

## NondominatedRunSurface

The multi-objective terminal sibling of `RunResult`. Materialised from a
`RunReport` of `ObjectiveVectorRecord`s; exposes the non-dominated candidate
set. See [`NondominatedRunSurface`][variopt.NondominatedRunSurface].

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

## RunMethod

The search-state owner. Proposes candidates via `ask`, consumes records via
`tell`, and owns the persistent search-state object. Population optimizers
(`CSAOptimizer`, `DifferentialEvolutionOptimizer`,
`GeneticAlgorithmOptimizer`) are `RunMethod` implementations. See
[`RunMethod`][variopt.RunMethod].

## RunReport

The generic terminal report produced by `Study.run(...)`. Covers any
`EvaluationRecord` type. See [`RunReport`][variopt.RunReport].

## RunResult

The scalar terminal result produced by `Study.optimize(...)`. Covers scalar
`Observation` records only. See [`RunResult`][variopt.RunResult].

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
