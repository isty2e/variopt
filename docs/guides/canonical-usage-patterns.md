# Canonical Usage Patterns

This guide shows the concrete caller-facing paths that became canonical as
`variopt` moved to request-first evaluation, sibling interaction semantics, and
explicit execution reports. Problem protocols should prefer request-free
payloads; execution artifacts own request identity, success/failure attempt
slots, and terminal report projections.

Use [`Study.optimize`][variopt.Study.optimize]
when the problem yields scalar
[`Observation`][variopt.Observation]
records and you want one scalar
[`RunResult`][variopt.RunResult].

Use [`Study.run`][variopt.Study.run]
when a run method consumes successful payload projections and you want one
generic
[`RunReport`][variopt.RunReport].

## Request-Free Protocol Payloads

The smallest practical custom protocol pattern is:

1. Define one request-free payload type.
2. Implement one proposal-local
   [`EvaluationProtocol`][variopt.EvaluationProtocol].
3. Let evaluation artifacts attach request identity when materializing
   [`EvaluationSuccess`][variopt.artifacts.EvaluationSuccess]
   and terminal reports.

```python
from dataclasses import dataclass

from typing_extensions import override

from variopt import (
    EvaluationProtocol,
    EvaluationRequest,
    IntegerSpace,
    Problem,
    Proposal,
    RunReport,
)
from variopt.artifacts import EvaluationSuccess


@dataclass(frozen=True, slots=True)
class LabelPayload:
    label: str


class ParityProtocol(EvaluationProtocol[int, LabelPayload]):
    @override
    def evaluate_request(self, request: EvaluationRequest[int]) -> LabelPayload:
        candidate = request.candidate
        return LabelPayload(label=f"parity:{candidate % 2}")


protocol = ParityProtocol()
problem = Problem(
    space=IntegerSpace(low=0, high=10),
    evaluation_protocol=protocol,
)
requests = (
    EvaluationRequest(proposal=Proposal(candidate=3, proposal_id="p-1")),
    EvaluationRequest(proposal=Proposal(candidate=4, proposal_id="p-2")),
)


def evaluate_success(
    request: EvaluationRequest[int],
) -> EvaluationSuccess[int, LabelPayload]:
    problem.space.validate(request.candidate)
    return EvaluationSuccess(
        request=request,
        payload=problem.evaluation_protocol.evaluate_request(request),
        evaluation_count=1,
    )


successes = tuple(
    evaluate_success(request) for request in requests
)
report = RunReport[int, LabelPayload].from_successes(successes)

assert report.evaluation_count == 2
assert [payload.label for payload in report.records] == ["parity:1", "parity:0"]
```

In normal optimization runs, [`Problem`][variopt.Problem] and
[`Study`][variopt.Study] provide the validation and orchestration shell around
the same request-free protocol contract.

## Compatibility `Study.run(...)`

`Study.run(...)` still interoperates with custom run methods that consume
successful payload projections through `RunMethod.tell(...)`. If a custom
payload is used directly with that legacy feedback path today, it must expose
the evaluated request and candidate structurally; do not subclass an
obsolete generic record base from the public facade.

```python
from dataclasses import dataclass
from collections.abc import Sequence

from typing_extensions import override

from variopt import (
    EvaluationProtocol,
    EvaluationRequest,
    IntegerSpace,
    Problem,
    Proposal,
    RunMethod,
    Study,
)
from variopt.evaluators import SequentialEvaluator


@dataclass(frozen=True, slots=True)
class LabelRecord:
    request: EvaluationRequest[int]
    candidate: int
    label: str

    @property
    def proposal(self) -> Proposal[int]:
        return self.request.proposal


class ParityProtocol(EvaluationProtocol[int, LabelRecord]):
    @override
    def evaluate_request(self, request: EvaluationRequest[int]) -> LabelRecord:
        candidate = request.candidate
        return LabelRecord(
            request=request,
            candidate=candidate,
            label=f"parity:{candidate % 2}",
        )


@dataclass(frozen=True, slots=True)
class QueueState:
    remaining_batches: tuple[tuple[Proposal[int], ...], ...]
    tell_history: tuple[tuple[LabelRecord, ...], ...] = ()


class QueueOptimizer(RunMethod[QueueState, Proposal[int], LabelRecord]):
    def __init__(self, proposal_batches: list[tuple[Proposal[int], ...]]) -> None:
        self._initial_batches = tuple(tuple(batch) for batch in proposal_batches)

    @override
    def create_initial_state(self) -> QueueState:
        return QueueState(remaining_batches=self._initial_batches)

    @override
    def is_exhausted(self, state: QueueState) -> bool:
        return len(state.remaining_batches) == 0

    @override
    def ask(
        self,
        state: QueueState,
        batch_size: int = 1,
    ) -> tuple[tuple[Proposal[int], ...], QueueState]:
        _ = batch_size
        if not state.remaining_batches:
            return (), state

        return (
            state.remaining_batches[0],
            QueueState(
                remaining_batches=state.remaining_batches[1:],
                tell_history=state.tell_history,
            ),
        )

    @override
    def tell(
        self,
        state: QueueState,
        records: Sequence[LabelRecord],
    ) -> QueueState:
        return QueueState(
            remaining_batches=state.remaining_batches,
            tell_history=state.tell_history + (tuple(records),),
        )


problem = Problem(
    space=IntegerSpace(low=0, high=10),
    evaluation_protocol=ParityProtocol(),
)
optimizer = QueueOptimizer(
    proposal_batches=[
        (Proposal(candidate=3, proposal_id="p-1"),),
        (Proposal(candidate=4, proposal_id="p-2"),),
    ],
)
study = Study(
    problem=problem,
    run_method=optimizer,
    evaluator=SequentialEvaluator[int, int, LabelRecord](),
)

report, final_state = study.run(max_evaluations=2)

assert report.evaluation_count == 2
assert [record.label for record in report.records] == ["parity:1", "parity:0"]
assert len(report.trace.events) == 2
assert len(final_state.tell_history) == 2
```

`Study.run(...)` is the terminal compatibility path for non-scalar records.
`Study.optimize(...)` stays scalar-only and will reject general record types.

## `RunReport` To `NondominatedRunSurface`

For vector-valued objective records, do not force a scalar best observation.
Materialize one
[`NondominatedRunSurface`][variopt.NondominatedRunSurface]
from the generic report instead.

```python
from variopt import (
    EvaluationRequest,
    NondominatedRunSurface,
    OptimizationDirection,
    Proposal,
    RunReport,
)
from variopt.artifacts import EvaluationSuccess, ObjectiveVectorPayload


successes = tuple(
    EvaluationSuccess(
        request=EvaluationRequest(
            proposal=Proposal(candidate=candidate, proposal_id=proposal_id),
        ),
        payload=ObjectiveVectorPayload.from_objective_values(
            objective_values=objective_values,
            directions=(
                OptimizationDirection.MINIMIZE,
                OptimizationDirection.MINIMIZE,
            ),
        ),
    )
    for candidate, proposal_id, objective_values in (
        (1, "p-1", (1.0, 3.0)),
        (2, "p-2", (2.0, 2.0)),
        (3, "p-3", (3.0, 1.0)),
        (4, "p-4", (4.0, 4.0)),
    )
)

report = RunReport[int, ObjectiveVectorPayload].from_successes(
    successes=successes,
    evaluation_count=5,
)
surface = NondominatedRunSurface[int].from_report(report)

assert [record.candidate for record in surface.nondominated_records] == [1, 2, 3]
assert surface.evaluation_count == 5
```

This keeps terminal multi-objective semantics explicit:

- `RunReport` is the generic execution report.
- `NondominatedRunSurface` is the multi-objective terminal sibling.
- `RunResult` remains the scalar optimization summary.

## Interaction-Aware `InteractionProblem`

[`InteractionProblem`][variopt.InteractionProblem]
is the sibling basis for grouped-request semantics such as self-play,
tournaments, and other request-coupled evaluation.

```python
from dataclasses import dataclass

from typing_extensions import override

from variopt import (
    EvaluationRequest,
    IntegerSpace,
    InteractionEvaluationProtocol,
    InteractionProblem,
    Proposal,
)
from variopt.artifacts import (
    InteractionEvaluationSpec,
    InteractionEvaluationUnit,
)


@dataclass(frozen=True, slots=True)
class MatchupSpec(InteractionEvaluationSpec):
    arena: str


@dataclass(frozen=True, slots=True)
class MatchupPayload:
    candidates: tuple[int, ...]
    winner: int
    arena: str


class MatchupProtocol(InteractionEvaluationProtocol[int, MatchupPayload]):
    @override
    def evaluate_interaction_unit(
        self,
        interaction_unit: InteractionEvaluationUnit[int],
    ) -> MatchupPayload:
        winner = max(interaction_unit.candidates)
        interaction_spec = interaction_unit.interaction_evaluation_spec
        arena = "default"
        if isinstance(interaction_spec, MatchupSpec):
            arena = interaction_spec.arena

        return MatchupPayload(
            candidates=interaction_unit.candidates,
            winner=winner,
            arena=arena,
        )


problem = InteractionProblem(
    space=IntegerSpace(low=0, high=10),
    interaction_evaluation_protocol=MatchupProtocol(),
    name="matchup",
)

payload = problem.interaction_evaluation_protocol.evaluate_requests(
    (
        EvaluationRequest(proposal=Proposal(candidate=2, proposal_id="left")),
        EvaluationRequest(proposal=Proposal(candidate=7, proposal_id="right")),
    ),
    interaction_evaluation_spec=MatchupSpec(arena="tournament"),
)

assert payload.winner == 7
assert payload.arena == "tournament"
assert payload.candidates == (2, 7)
```

Current boundary note:

- `InteractionProblem` is the canonical sibling problem basis today.
- `Study` orchestration is still proposal-local.
- Interaction-aware run/evaluator orchestration is follow-up work, so direct
  protocol evaluation is the honest current path.
