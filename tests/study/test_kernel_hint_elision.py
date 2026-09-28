"""Consumer and producer boundaries for unused kernel-hint elision."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from typing import Literal, TypeAlias

import pytest
from typing_extensions import override

from tests.study_support import (
    BatchQueueOptimizer,
    BatchQueueOptimizerState,
    FailingCandidateObjective,
    RecordingKernel,
    SquareObjective,
)
from variopt import CategoricalSpace, Objective, Observation, Problem, Proposal, Study
from variopt.algorithms.local_search import StructuredStochasticNeighborhoodKernel
from variopt.algorithms.population.csa import CSAOptimizer, CSAProfile
from variopt.algorithms.population.csa.engine import CSAEngineState
from variopt.algorithms.population.csa.trace.events import CSAEventTraceState
from variopt.artifacts import EvaluationAttemptBatch, ObservationPayload
from variopt.evaluators import (
    AsyncJoblibEvaluator,
    JoblibEvaluator,
    SequentialEvaluator,
)
from variopt.execution import EXACT_ASYNC_EXECUTION_MODEL, SYNC_BATCH_EXECUTION_MODEL
from variopt.kernel import DirectKernel, ProposalBatchQuery, ProposalLocalSearchContext
from variopt.randomness import RandomStateSnapshot

ScalarQuery: TypeAlias = ProposalBatchQuery[int, int, ObservationPayload]
ScalarAttempts: TypeAlias = EvaluationAttemptBatch[int, ObservationPayload]
ScalarStudy: TypeAlias = Study[
    int, int, CSAEngineState[int], ObservationPayload, Observation[int]
]


class GenericSequentialEvaluator(SequentialEvaluator[int, int]):
    """Keep optimize on the generic path without changing evaluations."""


@dataclass(frozen=True, slots=True)
class RecordingHintOptimizer(CSAOptimizer[int, int]):
    hint_states: list[CSAEngineState[int]] = field(default_factory=list, compare=False)
    hint_behavior: Literal["normal", "misaligned", "raise"] = "normal"

    @override
    def proposal_kernel_hints(
        self,
        state: CSAEngineState[int],
        proposals: Sequence[Proposal[int]],
    ) -> tuple[ProposalLocalSearchContext | None, ...] | None:
        self.hint_states.append(state)
        if self.hint_behavior == "misaligned":
            return ()
        if self.hint_behavior == "raise":
            raise ValueError("custom hint failure")
        return super(RecordingHintOptimizer, self).proposal_kernel_hints(
            state, proposals
        )


class RecordingDirectKernel(DirectKernel[ScalarQuery, ScalarAttempts]):
    def __init__(self) -> None:
        self.queries: list[ScalarQuery] = []

    @override
    def run(
        self,
        query: ScalarQuery,
        runner: Callable[[ScalarQuery], ScalarAttempts],
    ) -> ScalarAttempts:
        self.queries.append(query)
        return runner(query)


class FailingHintRunMethod(BatchQueueOptimizer):
    @override
    def proposal_kernel_hints(
        self,
        state: BatchQueueOptimizerState,
        proposals: Sequence[Proposal[int]],
    ) -> tuple[ProposalLocalSearchContext | None, ...] | None:
        raise ValueError("custom run-method hint failure")


def builtin_optimizer() -> CSAOptimizer[int, int]:
    return CSAOptimizer.from_space_defaults(
        space=CategoricalSpace(tuple(range(20))),
        bank_capacity=6,
        profile=CSAProfile(seed_count=3, cycle_limit=1000),
        random_state=7,
    )


def recording_optimizer() -> RecordingHintOptimizer:
    base = builtin_optimizer()
    return RecordingHintOptimizer(
        space=base.space,
        diversity_metric=base.diversity_metric,
        bank_capacity=base.bank_capacity,
        profile=base.profile,
        random_state=base.random_state,
    )


def sequential_study(
    optimizer: CSAOptimizer[int, int],
    *,
    objective: Objective[int] | None = None,
) -> ScalarStudy:
    return Study(
        problem=Problem(
            space=optimizer.space,
            objective=SquareObjective() if objective is None else objective,
        ),
        run_method=optimizer,
        evaluator=GenericSequentialEvaluator(),
    )


def fail_if_rng_hint_is_derived(
    snapshot: RandomStateSnapshot, *, namespace: str, keys: Sequence[str | int] = ()
) -> RandomStateSnapshot:
    raise AssertionError("unused direct-kernel RNG hint was derived")


@pytest.mark.parametrize("operation", ["step", "run", "optimize"])
def test_builtin_direct_execution_skips_unused_rng_hints(
    monkeypatch: pytest.MonkeyPatch, operation: Literal["step", "run", "optimize"]
) -> None:
    optimizer = builtin_optimizer()
    study = sequential_study(optimizer)
    monkeypatch.setattr(
        "variopt.algorithms.population.csa.optimizer.derive_random_state_snapshot",
        fail_if_rng_hint_is_derived,
    )

    if operation == "step":
        records, _ = study.step(optimizer.create_initial_state(), batch_size=4)
        assert len(records) == 4
    elif operation == "run":
        report, _ = study.run(max_evaluations=16, batch_size=4)
        assert report.evaluation_count == 16
    else:
        result, _ = study.optimize(max_evaluations=16, batch_size=4)
        assert result.evaluation_count == 16


def test_csa_subclass_keeps_eager_hint_hook() -> None:
    optimizer = recording_optimizer()
    report, _ = sequential_study(optimizer).run(max_evaluations=16, batch_size=4)

    assert report.evaluation_count == 16
    assert len(optimizer.hint_states) == 4
    assert tuple(state.proposal_index for state in optimizer.hint_states) == (
        4,
        8,
        12,
        16,
    )


def test_direct_kernel_subclass_receives_exact_csa_hints() -> None:
    optimizer = builtin_optimizer()
    kernel = RecordingDirectKernel()
    study = Study(
        problem=Problem(space=optimizer.space, objective=SquareObjective()),
        run_method=optimizer,
        evaluator=SequentialEvaluator[int, int](),
        kernel=kernel,
    )
    proposals, post_ask_state = optimizer.ask(
        optimizer.create_initial_state(), batch_size=4
    )
    expected_hints = optimizer.proposal_kernel_hints(post_ask_state, proposals)

    report, _ = study.run(max_evaluations=4, batch_size=4)

    assert report.evaluation_count == 4
    assert len(kernel.queries) == 1
    assert kernel.queries[0].proposals == proposals
    assert expected_hints is not None
    assert kernel.queries[0].proposal_kernel_hints == expected_hints


@pytest.mark.parametrize("behavior", ["misaligned", "raise"])
def test_csa_subclass_preserves_hint_validation_and_exceptions(
    behavior: Literal["misaligned", "raise"],
) -> None:
    optimizer = replace(recording_optimizer(), hint_behavior=behavior)
    study = sequential_study(optimizer)
    state = optimizer.create_initial_state()
    message = "align one-to-one" if behavior == "misaligned" else "custom hint failure"

    with pytest.raises(ValueError, match=message):
        study.step(state, batch_size=4)

    assert len(optimizer.hint_states) == 1
    assert state == optimizer.create_initial_state()


def test_unrelated_run_method_preserves_hint_exceptions() -> None:
    optimizer = FailingHintRunMethod([(Proposal(candidate=2, proposal_id="p-0"),)])
    study = Study(
        problem=Problem(
            space=CategoricalSpace[int]((1, 2, 3)), objective=SquareObjective()
        ),
        run_method=optimizer,
        evaluator=SequentialEvaluator[int, int](),
    )

    with pytest.raises(ValueError, match="custom run-method hint failure"):
        study.step(optimizer.create_initial_state())


def test_custom_kernel_does_not_query_hint_elision_capability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_if_elision_is_queried(self: CSAOptimizer[int, int]) -> bool:
        raise AssertionError("a hint consumer must not request elision")

    monkeypatch.setattr(
        CSAOptimizer, "_supports_unused_kernel_hint_elision", fail_if_elision_is_queried
    )
    optimizer = builtin_optimizer()
    kernel = RecordingKernel()
    study = Study(
        problem=Problem(space=optimizer.space, objective=SquareObjective()),
        run_method=optimizer,
        evaluator=SequentialEvaluator[int, int](),
        kernel=kernel,
    )

    study.step(optimizer.create_initial_state(), batch_size=4)

    assert len(kernel.queries) == 1
    assert kernel.queries[0].proposal_kernel_hints is not None


@pytest.mark.parametrize(
    ("backend", "transport"),
    [
        ("sequential", "per_request"),
        ("threading", "per_request"),
        ("threading", "worker_session"),
        ("loky", "per_request"),
        ("loky", "worker_session"),
        ("async", "per_request"),
    ],
)
@pytest.mark.parametrize("with_failures", [False, True])
@pytest.mark.parametrize("count_evaluation_cost", [False, True])
def test_direct_elision_matches_eager_records_trace_budget_and_state(
    backend: Literal["sequential", "threading", "loky", "async"],
    transport: Literal["per_request", "worker_session"],
    with_failures: bool,
    count_evaluation_cost: bool,
) -> None:
    optimizer = builtin_optimizer()
    eager_optimizer = recording_optimizer()
    objective = FailingCandidateObjective((3, 4, 7, 10) if with_failures else ())
    evaluator: (
        SequentialEvaluator[int, int]
        | JoblibEvaluator[int, int]
        | AsyncJoblibEvaluator[int, int]
    )
    if backend == "sequential":
        evaluator = GenericSequentialEvaluator()
    elif backend == "async":
        evaluator = AsyncJoblibEvaluator[int, int](n_jobs=2, backend="threading")
    else:
        evaluator = JoblibEvaluator[int, int](
            n_jobs=2, backend=backend, problem_transport=transport
        )
    model = (
        EXACT_ASYNC_EXECUTION_MODEL
        if backend == "async"
        else SYNC_BATCH_EXECUTION_MODEL
    )
    study = Study(
        problem=Problem(space=optimizer.space, objective=objective),
        run_method=optimizer,
        evaluator=evaluator,
    )
    eager_study = Study(
        problem=Problem(space=eager_optimizer.space, objective=objective),
        run_method=eager_optimizer,
        evaluator=evaluator,
    )
    initial_state = replace(
        optimizer.create_initial_state(), trace_state=CSAEventTraceState()
    )
    eager_initial_state = replace(
        eager_optimizer.create_initial_state(), trace_state=CSAEventTraceState()
    )

    report, state = study.run(
        max_evaluations=37,
        batch_size=4,
        initial_state=initial_state,
        execution_model=model,
        count_evaluation_cost=count_evaluation_cost,
    )
    eager_report, eager_state = eager_study.run(
        max_evaluations=37,
        batch_size=4,
        initial_state=eager_initial_state,
        execution_model=model,
        count_evaluation_cost=count_evaluation_cost,
    )

    assert report == eager_report
    assert report.evaluation_count == 37
    assert bool(report.failures) is with_failures
    assert report.trace.events
    assert state == eager_state
    assert eager_optimizer.hint_states
    assert initial_state == eager_initial_state


def test_direct_checkpoint_continuation_matches_eager_execution() -> None:
    optimizer = builtin_optimizer()
    study = sequential_study(optimizer)
    eager_kernel = RecordingDirectKernel()
    eager_study = Study(
        problem=study.problem,
        run_method=optimizer,
        evaluator=GenericSequentialEvaluator(),
        kernel=eager_kernel,
    )
    report, state = study.run(
        max_evaluations=61, batch_size=4, stop_at_checkpoint_boundary=True
    )
    eager_report, eager_state = eager_study.run(
        max_evaluations=61, batch_size=4, stop_at_checkpoint_boundary=True
    )
    snapshot = optimizer.state_to_dict(state)
    restored_state = optimizer.state_from_dict(snapshot)

    assert report == eager_report
    assert state == eager_state
    assert snapshot == optimizer.state_to_dict(eager_state)
    continuation, continued_state = study.run(
        max_evaluations=37, batch_size=4, initial_state=state
    )
    resumed, resumed_state = study.run(
        max_evaluations=37, batch_size=4, initial_state=restored_state
    )
    eager_continuation, eager_continued_state = eager_study.run(
        max_evaluations=37, batch_size=4, initial_state=eager_state
    )

    assert continuation == resumed == eager_continuation
    assert continued_state == resumed_state == eager_continued_state


def test_stochastic_kernel_keeps_hints_and_parent_rng_state() -> None:
    optimizer = builtin_optimizer()
    eager_optimizer = recording_optimizer()
    study = Study(
        problem=Problem(space=optimizer.space, objective=SquareObjective()),
        run_method=optimizer,
        evaluator=SequentialEvaluator[int, int](),
        kernel=StructuredStochasticNeighborhoodKernel[int, int](
            max_steps=2, max_neighbors_per_step=2, random_state=19
        ),
    )
    eager_study = Study(
        problem=Problem(space=eager_optimizer.space, objective=SquareObjective()),
        run_method=eager_optimizer,
        evaluator=SequentialEvaluator[int, int](),
        kernel=StructuredStochasticNeighborhoodKernel[int, int](
            max_steps=2, max_neighbors_per_step=2, random_state=19
        ),
    )

    report, state = study.run(max_evaluations=37, batch_size=4)
    eager_report, eager_state = eager_study.run(max_evaluations=37, batch_size=4)

    assert report == eager_report
    assert state == eager_state
    assert eager_optimizer.hint_states
    assert report.refinements
