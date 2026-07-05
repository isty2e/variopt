# Local Optimization Method Guidance

## Purpose

This guide explains when local optimization is a good fit in `variopt`, which
built-in method to prefer, and how local optimization interacts with budgeting
and execution backends.

The current boundary is:

- `RunMethod` proposes candidates
- `Kernel` optionally organizes one bounded local-search episode
- `Evaluator` executes raw proposal batches only
- `Study` wires `RunMethod`, optional `Kernel`, and `Evaluator`

Local optimization therefore lives on the kernel side of the execution path,
not inside evaluator backends.

When local optimization changes a candidate, the kernel should attach
[`CandidateRefinement`](../reference/api/variopt.md) to the successful
`EvaluationSuccess` slot in the returned `EvaluationAttemptBatch`. That metadata
records the source candidate, the candidate actually evaluated, and any
structured leaf paths changed by the episode. It is provenance, not
evaluation-protocol semantics; the success payload remains the authoritative
evaluation result.

For the proposed/refined/evaluated/accepted vocabulary, see
[Candidate Refinement](../concepts/candidate-refinement.md).

## Current Built-In Support

The current reusable built-in kernels are:

- [`ScipyMinimizeKernel`](../reference/api/local-search.md)
- [`StructuredHillClimbKernel`](../reference/api/local-search.md)
- [`StructuredIteratedLocalSearchKernel`](../reference/api/local-search.md)
- [`StructuredScheduledLocalSearchKernel`](../reference/api/local-search.md)
- [`StructuredStochasticNeighborhoodKernel`](../reference/api/local-search.md)
- [`StructuredVariableNeighborhoodKernel`](../reference/api/local-search.md)

Their scopes are intentionally strict:

- the SciPy kernel only supports structured spaces whose leaves are all
  [`RealSpace`](../reference/api/spaces.md)
- the SciPy kernel optimizes in the space coordinate system, so log-scaled
  real leaves are optimized in log coordinates rather than raw value space
- the SciPy kernel currently supports only `L-BFGS-B` and `Powell`
- the structured hill climber supports only structured spaces whose leaves are
  all [`IntegerSpace`](../reference/api/spaces.md)
  or [`CategoricalSpace`](../reference/api/spaces.md)
- the structured hill climber applies deterministic one-leaf
  first-improvement moves and uses an explicit `max_steps` budget
- the stochastic structured kernel supports the same discrete structured-space
  boundary, but samples a bounded subset of one-leaf moves per local-search
  step through explicit `max_neighbors_per_step` and optional
  `max_categorical_neighbors_per_leaf` limits
- the variable-neighborhood kernel supports the same discrete structured-space
  boundary, but sequences explicit neighborhood stages and resets to the first
  stage after each accepted improvement
- the iterated local-search kernel supports the same discrete structured-space
  boundary, but alternates deterministic local improvement with explicit
  bounded kicks under a separate kick policy
- the scheduled local-search kernel supports the same discrete structured-space
  boundary, but runs one explicit sequence of local-search stages instead of
  choosing or resetting stages dynamically
- neither built-in kernel exposes analytic gradients, repair logic, or
  domain-specific move families as first-class public contracts

If your problem falls outside that scope, provide a custom `Kernel` to
[`Study`](../reference/api/study.md) or skip
local optimization entirely.

## Method Choice

### `L-BFGS-B`

Prefer `L-BFGS-B` when all of the following are roughly true:

- every optimized variable is continuous
- the objective is reasonably smooth in the local neighborhood
- local progress is not dominated by large discontinuities or hard thresholding
- coordinate bounds matter

This is the best default for smooth continuous objectives. It is also the most
natural built-in choice when your search space uses log-scaled real variables,
because the current SciPy kernel already respects coordinate-space geometry.

Practical trade-off:

- `L-BFGS-B` often reduces the number of outer proposals needed
- but it may consume many inner objective evaluations due to finite-difference
  gradient estimation

`Study.optimize(...)` budgets by actual objective-evaluation cost by default, so
SciPy inner evaluations are charged against `max_evaluations` rather than hidden
behind one outer proposal.

### `Powell`

Prefer `Powell` when the space is still continuous but the objective is less
smooth, more rugged, or derivative information is unreliable.

Typical cases:

- piecewise or weakly discontinuous objectives over real variables
- objectives with noisy directional behavior where numerical gradients are not
  trustworthy
- cases where `L-BFGS-B` stalls or oscillates due to local non-smoothness

`Powell` stays within the current continuous-only local-search boundary, but it
is usually a better practical fallback than `L-BFGS-B` for rougher objectives.

Trade-off:

- it is often more robust than `L-BFGS-B`
- but it can require many function evaluations and may be slower wall-clock wise

### Mixed, Integer, or Categorical Spaces

Do not use the built-in SciPy kernel for mixed or discrete search spaces.

The current SciPy kernel rejects any structured space that contains non-real
leaves. That is intentional. Projecting integer or categorical candidates into
a fake continuous local search would blur the canonical candidate ontology and
make the episode semantics harder to reason about.

Use one of these instead:

- [`StructuredHillClimbKernel`](../reference/api/local-search.md)
  when every structured leaf is discrete and a simple leafwise neighborhood is
  appropriate
- [`StructuredStochasticNeighborhoodKernel`](../reference/api/local-search.md)
  when the same discrete neighborhood is appropriate but categorical fanout or
  per-step neighborhood cost is too large for deterministic enumeration
- [`StructuredVariableNeighborhoodKernel`](../reference/api/local-search.md)
  when a staged escape mechanism is needed across multiple explicit discrete
  neighborhood families inside one bounded episode
- [`StructuredIteratedLocalSearchKernel`](../reference/api/local-search.md)
  when a bounded kick-and-refine loop is needed and the perturbation law can
  be explained cleanly through an explicit kick policy
- [`StructuredScheduledLocalSearchKernel`](../reference/api/local-search.md)
  when you want a fixed, explicit sequence of discrete local-search stages
  without variable-neighborhood reset semantics
- no local optimization
- a custom kernel that understands the true candidate domain
- a problem-specific continuous relaxation only if you can define it cleanly
  and map back to canonical candidates without ambiguity

As a rule of thumb, if you cannot explain the relaxation and projection in one
clear sentence, do not hide it inside a generic local-search kernel.

The current variable-neighborhood kernel should be read narrowly: it is an
explicit staged escape mechanism when you already have a concrete neighborhood
widening story, not a broad default for generic structured discrete search.

The iterated local-search kernel should also be read narrowly. It is a
kick-and-refine tool when you already have a justified perturbation story, not
a blanket replacement for the deterministic hill climber or the categorical
stochastic kernel.

Public helper types such as `StructuredKickPolicy` and
`StructuredVariableNeighborhoodStage` are configuration records for these
advanced kernels. `ScipyMinimizeMethod` is the literal method set accepted by
`ScipyMinimizeKernel`: currently `"L-BFGS-B"` and `"Powell"`.

## Budget Accounting

`variopt` distinguishes two different budgets:

- outer proposal count
- actual objective-evaluation cost

This matters because a single kernel episode can evaluate the objective many
times.

The kernel path reports that cost through successful attempt
`evaluation_count` metadata.
`Study.optimize(...)` then offers two modes:

- default: budget decreases by the sum of objective evaluations reported by the
  kernel/evaluator path
- `count_evaluation_cost=False`: budget decreases by the number of returned
  observations

Practical guidance:

- keep the default when comparing methods with and without local optimization,
  or when the objective itself is expensive
- use `count_evaluation_cost=False` only when you deliberately want an outer-step
  budget rather than an objective-cost budget

If a custom kernel already computed the objective value, it should return both
that value and the true `evaluation_count` so that `Study` can reuse the value
instead of evaluating the objective again.

Refinement metadata and budget metadata are orthogonal. A kernel can report a
refined candidate with `evaluation_count=1`, or it can report no refinement while
still charging a larger inner evaluation count.

## Evaluator Backend Interaction

Local optimization no longer lives inside evaluator backends, but kernels still
receive evaluator-owned
[`ExecutionResources`](../reference/api/variopt.md)
through [`ProposalBatchQuery`](../reference/api/variopt.md).

That means evaluator choice still affects how local search should behave.

### `SequentialEvaluator`

Use
[`SequentialEvaluator`](../reference/api/evaluators.md)
when:

- you want the simplest execution path
- your kernel already uses significant internal compute
- debugging and determinism are more important than throughput

This is the safest default while tuning or validating a new local-search kernel.

### `JoblibEvaluator`

Use [`JoblibEvaluator`](../reference/api/evaluators.md)
when objective work is request-local and worth running across multiple
workers.

The current nested-parallelism contract makes the evaluator the outer parallel
owner. In practice that means:

- `JoblibEvaluator` fans out proposals across workers
- kernels should respect `ExecutionResources` and remain serial when the
  evaluator owns parallelism
- nested worker spawning should be treated as exceptional, not as the default

Practical rule:

- outer parallelism in the evaluator
- inner local search kept serial unless you have a deliberate partitioning
  story

This avoids oversubscription and keeps execution semantics easier to reason
about.

It does not imply equivalence with fully sequential execution. If a run method
changes state only after a whole batch is evaluated, then preserving proposal
order still leaves `sync_batch` observably different from `sequential`. Treat
changes in evaluator backend, worker count, or batch size as changes in the
execution configuration, not as free throughput toggles.

## Decision Table

| Situation | Recommended local optimization choice |
| --- | --- |
| Smooth continuous objective over `RealSpace` leaves | `ScipyMinimizeKernel(method="L-BFGS-B")` |
| Continuous but rough or weakly discontinuous objective | `ScipyMinimizeKernel(method="Powell")` |
| All-discrete structured space | `StructuredHillClimbKernel(max_steps=...)` |
| All-discrete structured space that needs explicit kick-and-refine episodes | `StructuredIteratedLocalSearchKernel(max_steps=..., max_kicks=..., kick_policy=...)` |
| All-discrete structured space with large categorical fanout | `StructuredStochasticNeighborhoodKernel(max_steps=..., max_neighbors_per_step=...)` |
| All-discrete structured space that already has a justified staged neighborhood-widening story | `StructuredVariableNeighborhoodKernel(max_steps=..., stages=(...))` |
| All-discrete structured space with a fixed stage sequence | `StructuredScheduledLocalSearchKernel(stages=(...))` |
| Mixed real/integer/categorical space | no built-in generic mixed adapter yet; use a custom kernel, split the local search cleanly by domain, or skip local optimization |
| Comparing methods with and without local optimization | use the default objective-cost budget |
| Batch-parallel evaluation with joblib | keep the kernel serial and let the evaluator own parallelism |
| Early debugging or correctness validation | start with `SequentialEvaluator` |

## Recommended Starting Point

If you are unsure, start here:

1. Use no local optimization at first.
2. If the space is continuous and local improvement is clearly valuable, try
   `L-BFGS-B`.
3. If `L-BFGS-B` behaves poorly on a rough objective, switch to `Powell`.
4. Keep default objective-cost budgeting before making any fairness claims about
   efficiency.
5. Add `JoblibEvaluator` only after the kernel itself is behaving well in
   sequential execution.

This sequence keeps the execution boundary explicit and avoids mixing algorithm
effects with backend or budgeting artifacts.
