# Checkpointing

`variopt` currently exposes explicit CSA state checkpointing through
[`CSAOptimizer.state_to_dict()`](api/population.md) and
[`CSAOptimizer.state_from_dict()`](api/population.md).

## Scope

The current contract covers CSA engine state only. It is an exact
safe-boundary checkpoint: resuming with the same optimizer configuration,
seed, objective, and execution model continues exactly from the saved
boundary.

The checkpoint payload is JSON-safe and is intended to be written through JSON
or another structured serialization format. The supported durable persistence
surface is the explicit `to_dict()` / `from_dict()` checkpoint contract; Python
`pickle` round trips are runtime compatibility conveniences only and are not a
cross-version or crash-recovery checkpoint format.

## Usage

```python
import json

from variopt import IntegerSpace, Objective, Problem, Study
from variopt.algorithms.population import CSAOptimizer
from variopt.evaluators import SequentialEvaluator
from typing_extensions import override


class SquareObjective(Objective[int]):
    @override
    def evaluate(self, candidate: int) -> float:
        return float(candidate * candidate)


space = IntegerSpace(0, 20)
optimizer = CSAOptimizer.from_space_defaults(
    space=space, bank_capacity=8, random_state=0,
)
study = Study(
    problem=Problem(space=space, objective=SquareObjective()),
    run_method=optimizer,
    evaluator=SequentialEvaluator[int, int](),
)

# Run partway to a checkpoint-safe boundary and save.
result, state = study.optimize(
    max_evaluations=20,
    stop_at_checkpoint_boundary=True,
)
checkpoint = optimizer.state_to_dict(state)

with open("checkpoint.json", "w") as f:
    json.dump(checkpoint, f)

# Later: restore and continue.
with open("checkpoint.json") as f:
    loaded = json.load(f)

restored_state = optimizer.state_from_dict(loaded)
result, _ = study.optimize(max_evaluations=20, initial_state=restored_state)
```

If a reported logical evaluation cost exhausts the hard budget while the run is
inside an unsafe segment, `stop_at_checkpoint_boundary=True` returns the latest
checkpoint-safe report and state instead of assimilating the over-budget
attempts. If no safe snapshot has been reached, the budget exhaustion is still
reported as `EvaluationBudgetExhausted`.

For structured spaces the built-in recursive candidate codec handles
serialization automatically. For non-structured spaces, pass explicit
`candidate_to_dict` and `candidate_from_dict` callbacks.

## Safe Boundary Requirement

!!! warning "Safe boundary only"

    `state_to_dict()` only accepts states that are between CSA generation
    batches. Concretely, checkpointing requires:

    - no pending proposals
    - no active generation queue
    - no buffered generation observations waiting to commit
    - no reference-refresh pool in progress
    - no pending proposal attributions

    If any of those runtime domains is active, checkpointing raises
    `ValueError` instead of serializing a partial state.

## What Is Persisted

The checkpoint captures the authoritative optimizer memory needed for exact
continuation, including:

- RNG state
- bank and reference-bank contents
- growth and clustering state
- cutoff and stage progression state
- seed-selection state
- proposal adaptation statistics
- scoring state
- monotone proposal-id counter

## What Is Not Persisted

The checkpoint intentionally does not capture:

- live evaluator or worker state
- exact-async suspended sessions
- exact-async resume handles
- in-flight proposal batches
- `Study.run(...)` reports or `Study.optimize(...)` results
- trace or telemetry reducer state
- derived caches that can be recomputed from authoritative state

## Candidate Encoding

For CSA optimizers built over `StructuredSearchSpace`, the optimizer provides a
built-in recursive candidate codec. For non-structured spaces, callers must
pass explicit candidate serialization callbacks.

## Terminal Results

`RunReport`, `RunResult`, and `NondominatedRunSurface` are terminal result
objects, not optimizer checkpoints. They may carry candidate-refinement
provenance, but `variopt` does not currently define `to_dict()` / `from_dict()`
serialization for those terminal surfaces. Persisting reports, traces, or
result summaries is caller-owned for now.

## Non-Goals for v1

!!! note "Out of scope for the initial checkpointing contract"

    The current contract does **not** support:

    - mid-step checkpoint/resume
    - exact-async suspended-session checkpointing
    - exact-async resume-handle crash recovery
    - terminal report/result serialization
    - generic `Study`-level persistence across arbitrary run methods

    Those require restoring evaluator-owned lifecycle state in addition to
    the optimizer memory and are intentionally out of scope for this first
    slice.
