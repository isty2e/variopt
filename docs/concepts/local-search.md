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

Kernel ownership and execution placement are distinct. The kernel defines the
episode, while synchronous study orchestration may place explicitly eligible
bounded episodes inside `SequentialEvaluator` or `JoblibEvaluator`. Optimizer
state and assimilation remain coordinator-owned in either case.

Kernels return `EvaluationAttemptBatch` values. Successful
`EvaluationSuccess` attempts may carry `CandidateRefinement` provenance when
the kernel changes a candidate before evaluation. A successful kernel episode
still occupies exactly one top-level attempt slot for the original proposal;
failed inner local-search trials are charged through that slot's
`evaluation_count` and summarized in `KernelDiagnostics`. If an episode produces
no successful evaluation, the top-level slot remains an `EvaluationFailure`.
Acceptance into an optimizer archive is still decided later by the run method.

For the current detailed method note, see
[local-optimization-methods.md](../guides/local-optimization-methods.md).

For execution placement, budgeting, failure, and reproducibility guarantees,
see [Evaluator-Owned Local Search](evaluator-owned-local-search.md).

For the execution provenance vocabulary, see
[Candidate Refinement](candidate-refinement.md).
