# Candidate Refinement

Candidate refinement is execution-side provenance for a candidate that was
changed before evaluation.

It sits between a run method's proposal and the evaluation record:

```text
RunMethod.ask -> Proposal -> Kernel or evaluator execution
    -> EvaluationAttemptBatch(EvaluationSuccess or EvaluationFailure slots)
    -> Study -> RunMethod feedback
```

The successful attempt remains the semantic result: it owns the evaluated
request, the payload, and optional refinement metadata. New problem protocols
normally return request-free payloads; legacy records and observations are
compatibility projections from that success.

## Candidate Vocabulary

| Term | Owner | Meaning |
| --- | --- | --- |
| Proposed candidate | `RunMethod` / `Proposal` | The candidate selected by the search method before execution. |
| Source candidate | `CandidateRefinement.source_candidate` | The candidate before execution-side refinement. Usually the proposal candidate. |
| Refined candidate | `CandidateRefinement.refined_candidate` | The candidate after refinement. It must match the aligned success request candidate. |
| Evaluated candidate | `EvaluationSuccess.request.candidate` | The canonical candidate evaluated by the problem's evaluation protocol. |
| Accepted candidate | `RunMethod` state | A candidate admitted into optimizer state, such as a CSA bank entry. Not every evaluated candidate is accepted. |

The important invariant is:

```text
EvaluationSuccess.refinement.refined_candidate == EvaluationSuccess.request.candidate
```

`EvaluationSuccess`, legacy `EvaluationOutcome` compatibility records, and
terminal successes validate that invariant when refinement metadata is present.

## Ownership Boundary

Refinement is not owned by `EvaluationProtocol`.

- `Kernel` owns bounded local-search episodes and is the usual source of
  built-in refinement metadata.
- `Evaluator` owns execution mechanics. A custom evaluator may carry refinement
  metadata if it deliberately transforms a candidate before protocol evaluation.
- `EvaluationProtocol` owns only the meaning of evaluating the actual candidate
  it receives.
- `Study` transports payload attempt batches at execution boundaries,
  materializes successful payloads into request-aligned records at run-method
  feedback, records accounting, and preserves aligned successful-outcome
  refinement metadata in terminal reports.
- `RunMethod.tell(...)` remains record-based. A run method that needs
  execution-side metadata can override `RunMethod.tell_attempts(...)`.

This split keeps local search and repair-style execution behavior out of the
problem's semantic evaluation rule.

## Accounting

Refinement metadata does not count evaluations by itself. Logical evaluation cost
is carried by successful and failed attempt slots.

By default, `Study.optimize(...)` charges the reported `evaluation_count` instead
of only counting returned records. This matters when a local-search kernel
evaluates several inner candidates before returning one refined result. Set
`count_evaluation_cost=False` only when you deliberately want outer-record
counting.

Terminal surfaces preserve provenance as success-aligned metadata:

- `RunReport.refinements` aligns with `RunReport.successes`; `records` is the
  legacy payload projection. A projected record's request may preserve the
  source proposal while its `candidate` is the evaluated candidate.
- `RunResult.refinements` aligns with `RunResult.successes`; `observations` is
  the scalar compatibility projection.
- `NondominatedRunSurface.refinements` aligns with
  `NondominatedRunSurface.successes`; vector records and frontier records are
  compatibility projections.

When no refinement metadata was recorded, these fields use the compact empty
tuple. If some successes have refinement and some do not, unrefined positions are
represented by `None`.

## Local Search Behavior

Built-in local-search kernels attach `CandidateRefinement` to successful
`EvaluationSuccess` values when the episode returns a candidate with changed
canonical leaf values. Recorded `EvaluationFailure` attempts never receive
kernel refinement metadata.

For structured spaces, `changed_leaf_paths` is the authoritative set of leaf
paths reported by the refinement producer. A scalar leaf uses the root path
`()`, so a scalar change is reported as `changed_leaf_paths=((),)`.

An empty `changed_leaf_paths` sequence means the producer is explicitly reporting
no changed structured leaf paths. It is not a request for downstream inference.
If a custom component cannot provide reliable path metadata and wants a
downstream optimizer to infer paths, it should omit refinement metadata for that
outcome.

## CSA Adaptation Behavior

CSA uses refinement paths only for proposal-adaptation feedback.

- If `CandidateRefinement.changed_leaf_paths` is present, CSA treats those paths
  as authoritative local-displacement feedback.
- If refinement metadata is absent, CSA may fall back to comparing the proposed
  and evaluated candidates when the search space can expose structured leaf
  differences.
- If the explicit path set is empty, CSA records no local displacement and does
  not infer paths from the candidate values.
- If CSA proposal adaptation is disabled, refinement path metadata is ignored.

This does not change CSA bank admission, candidate scoring, evaluation budget
accounting, or checkpoint semantics.

## Current Limitations

- `CandidateRefinement` does not represent "refined candidate, but unknown path
  attribution" as a third state distinct from "no changed paths".
- Terminal result surfaces are not optimizer checkpoints and do not currently
  define `to_dict()` / `from_dict()` serialization.
- Durable local-search memory across episodes belongs to `RunMethod` state, not
  to a `Kernel`.
- Refinement metadata is provenance. It does not imply that the refined candidate
  was accepted into optimizer state.
