# Local Search

The local-search family in `variopt` is kernel-oriented.

That means these components are bounded episode procedures rather than
cross-run optimizers.

## Current Families

- structured hill climb
- structured stochastic neighborhood search
- structured variable neighborhood search
- structured iterated local search
- structured scheduled local search
- SciPy-backed numeric local minimization

## Why They Are Kernels

Local-search components do not own long-lived search memory across the whole
run. They are the bounded episode layer that a run method or higher-level
workflow can invoke explicitly.

When a kernel changes a candidate before evaluation, it should return
`EvaluationOutcome` values with `CandidateRefinement` provenance. The refined
candidate must match the aligned evaluation record candidate; acceptance into an
optimizer archive is still decided later by the run method.

For the current detailed method note, see
[local-optimization-methods.md](../guides/local-optimization-methods.md).

For the execution provenance vocabulary, see
[Candidate Refinement](candidate-refinement.md).
