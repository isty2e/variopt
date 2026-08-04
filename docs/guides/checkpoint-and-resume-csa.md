# Checkpoint and Resume a CSA Run

Use a CSA checkpoint to stop at a safe generation boundary and continue later.
Exact continuation requires the same optimizer configuration, seed, objective,
and execution model.

The checkpoint contains optimizer state, not the configuration that gives the
state meaning. Record a configuration manifest beside it and compare the
manifest before restoring. See
[Record CSA Configuration Provenance](csa-configuration-provenance.md#guard-a-checkpoint-restore).

## Run to a safe boundary

```python
import json
import tempfile
from pathlib import Path

from typing_extensions import override

from variopt import IntegerSpace, Objective, Problem, Study
from variopt.algorithms.population import CSAOptimizer
from variopt.evaluators import SequentialEvaluator


class SquareObjective(Objective[int]):
    @override
    def evaluate(self, candidate: int) -> float:
        return float(candidate * candidate)


space = IntegerSpace(0, 20)
optimizer = CSAOptimizer.from_space_defaults(
    space=space,
    bank_capacity=8,
    random_state=0,
)
study = Study(
    problem=Problem(space=space, objective=SquareObjective()),
    run_method=optimizer,
    evaluator=SequentialEvaluator[int, int](),
)

result, state = study.optimize(
    max_evaluations=20,
    stop_at_checkpoint_boundary=True,
)
checkpoint = optimizer.state_to_dict(state)
```

`stop_at_checkpoint_boundary=True` returns a state between CSA generation
batches. Calling `state_to_dict()` on an active partial generation raises
`ValueError` rather than serializing incomplete optimizer memory.

If logical evaluation cost exhausts the hard budget inside an unsafe segment,
the study returns the latest checkpoint-safe report and state instead of
assimilating the over-budget attempts. If no safe snapshot exists,
`EvaluationBudgetExhausted` is still raised.

## Write the checkpoint atomically

Write to a temporary path on the same filesystem, then replace the destination:

```python
checkpoint_path = Path("checkpoint.json")
with tempfile.NamedTemporaryFile(
    "w",
    dir=checkpoint_path.parent,
    prefix=f"{checkpoint_path.name}.",
    suffix=".tmp",
    delete=False,
) as checkpoint_file:
    temporary_path = Path(checkpoint_file.name)
    json.dump(checkpoint, checkpoint_file)
    checkpoint_file.write("\n")

temporary_path.replace(checkpoint_path)
```

Do not overwrite the existing file in place. A crash during an in-place write
can leave neither the old nor the new snapshot usable.

## Restore and continue

```python
with checkpoint_path.open() as checkpoint_file:
    loaded_checkpoint = json.load(checkpoint_file)

restored_state = optimizer.state_from_dict(loaded_checkpoint)
continued_result, final_state = study.optimize(
    max_evaluations=20,
    initial_state=restored_state,
)
```

For a `StructuredSearchSpace`, the built-in recursive codec serializes
candidates automatically. For other `SearchSpace` implementations, pass
explicit `candidate_to_dict` and `candidate_from_dict` callbacks.

The durable contract is the JSON-safe `state_to_dict()` / `state_from_dict()`
representation. Python pickle round trips are runtime conveniences, not a
cross-version checkpoint format.

See [Checkpointing](../reference/checkpointing.md) for the exact persisted and
excluded state, candidate codec limits, and unsupported checkpoint modes.
