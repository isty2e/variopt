# Study and Execution Models

`Study` is the orchestration layer, not the owner of search semantics.

The current execution-model vocabulary is:

- `sequential`
- `sync_batch`
- `exact_async`
- `stale_async`

These are semantic run contracts, not just backend labels.

## Important Distinction

Evaluator backends such as sequential, joblib, and MPI answer:

- how evaluation work executes

Execution models answer:

- how the run method and study preserve state-transition law

That is why `exact_async` and `stale_async` are not mere evaluator options.

## What The Built-In Surface Supports Today

- `sequential`
  - one proposal at a time
  - simplest path for debugging and deterministic smoke checks
  - realized through [`SequentialEvaluator`][variopt.evaluators.SequentialEvaluator]
    with `batch_size=1`
- `sync_batch`
  - whole proposal batches are evaluated and assimilated together
  - realized through [`SequentialEvaluator`][variopt.evaluators.SequentialEvaluator],
    [`JoblibEvaluator`][variopt.evaluators.JoblibEvaluator], or
    [`MpiEvaluator`][variopt.evaluators.MpiEvaluator]
- `exact_async`
  - proposal completions may arrive out of order, but the run method still
    assimilates them under an exact state-transition law
  - realized through an async evaluator such as
    [`AsyncJoblibEvaluator`][variopt.evaluators.AsyncJoblibEvaluator] together
    with a run method that advertises `exact_async`
- `stale_async`
  - a separate study-level contract for rolling refill and stale assimilation
  - currently aimed at custom run methods; the built-in population optimizers
    only advertise `sequential`, `sync_batch`, and `exact_async`

So the practical default today is:

- use `SequentialEvaluator` for `sequential`
- use `JoblibEvaluator` or `MpiEvaluator` for `sync_batch`
- use `AsyncJoblibEvaluator` for `exact_async`
- treat `stale_async` as an advanced custom-run-method path rather than a
  built-in population-optimizer mode

`StudyExactAsyncStepSession` and `StudyExactAsyncStepResumeHandle` are the
study-side lifecycle objects for one exact-async step. Most users do not need
them directly; they exist for explicit polling, suspension, and resumption
around evaluators that support the corresponding async lifecycle.

## Exact-Async Example

The simplest way to use `exact_async` is to pass the execution model to
`Study.optimize(...)` or `Study.run(...)` together with an
`AsyncJoblibEvaluator`:

```python
from typing_extensions import override

from variopt import EXACT_ASYNC_EXECUTION_MODEL, Objective, Problem, RealSpace, Study
from variopt.algorithms.population import CSAOptimizer
from variopt.evaluators import AsyncJoblibEvaluator


class SphereObjective(Objective[float]):
    @override
    def evaluate(self, candidate: float) -> float:
        return candidate * candidate


problem = Problem(
    space=RealSpace(-5.0, 5.0),
    objective=SphereObjective(),
)

optimizer = CSAOptimizer.from_space_defaults(
    space=problem.space,
    bank_capacity=8,
    random_state=0,
)

study = Study(
    problem=problem,
    run_method=optimizer,
    evaluator=AsyncJoblibEvaluator(n_jobs=4, backend="loky"),
)

result, _ = study.optimize(
    max_evaluations=60,
    batch_size=8,
    execution_model=EXACT_ASYNC_EXECUTION_MODEL,
)
```

The run method still assimilates results under an exact state-transition law —
completions may arrive out of order from the evaluator, but the study
reorders them before handing them to the optimizer. The optimizer sees
the same logical sequence it would under `sync_batch`.

## Outcome Metadata

Execution boundaries transport `EvaluationAttemptBatch` values, not just raw
records. Successful attempts carry `EvaluationSuccess` metadata for kernel
diagnostics, evaluation-cost accounting, and candidate-refinement provenance.
Recorded user-code failures remain separate `EvaluationFailure` attempts.

`Study` keeps evaluator and kernel execution payload-based, then materializes
successful payload attempts into request-aligned records at the run-method
feedback boundary. In synchronous execution this happens after the kernel
returns an aligned attempt batch. In exact async execution it happens when the
step session finishes. In stale async execution it happens for each completed
group immediately before incremental feedback. Async session and resume
storage therefore keep payload attempts, not terminal feedback records.

Run methods consume the materialized record attempts through
`tell_attempts(...)`; the default implementation still delegates success-only
batches to record-based `tell(...)`. The older `tell_outcomes(...)` hook remains
for successful `EvaluationOutcome` compatibility streams and is not an
outcome-only fallback for `Study` orchestration.

Among the built-in population methods, `CSAOptimizer` consumes recorded failed
attempts by draining the failed proposal ids from pending CSA lifecycle state
without treating them as observations. GA and DE-family optimizers currently
raise `UnsupportedEvaluationFailureError` for failure-bearing attempt batches
because they require an explicit partial-generation policy before failures can
be assimilated safely.

For the refinement vocabulary, see
[Candidate Refinement](candidate-refinement.md).
