"""Integration tests for synchronous request-local episode dispatch."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import ClassVar, Literal

import pytest
from typing_extensions import override

from tests.study_support import (
    BatchQueueOptimizer,
    BatchQueueOptimizerState,
    SquareObjective,
)
from variopt import (
    EvaluationAttemptBatch,
    EvaluationBudget,
    EvaluationRequest,
    IntegerSpace,
    Objective,
    Observation,
    Problem,
    Proposal,
    RealSpace,
    Study,
)
from variopt.algorithms.local_search import (
    ScipyMinimizeKernel,
    StructuredHillClimbKernel,
    StructuredIteratedLocalSearchKernel,
    StructuredScheduledLocalSearchKernel,
    StructuredStochasticNeighborhoodKernel,
    StructuredVariableNeighborhoodKernel,
    StructuredVariableNeighborhoodStage,
)
from variopt.artifacts import ObservationPayload
from variopt.evaluators import JoblibEvaluator, SequentialEvaluator
from variopt.execution import SYNC_BATCH_EXECUTION_MODEL, ExecutionResources
from variopt.kernel import (
    ProposalBatchQuery,
    ProposalKernelHint,
    ProposalLocalSearchContext,
    RequestLocalEpisode,
    RequestLocalEpisodeKernel,
    RequestLocalEvaluationRunner,
)
from variopt.methods import RunMethod
from variopt.randomness import RandomStateSnapshot
from variopt.study.execution import (
    _max_min_evaluation_limits,
    evaluate_step,
)


class CoordinatorOnlySequentialEvaluator:
    """Sequential attempt evaluator without the optional episode capability."""

    def __init__(self) -> None:
        self._delegate = SequentialEvaluator[int, int]()

    def execution_resources(self) -> ExecutionResources:
        """Return sequential evaluator-owned resources."""
        return self._delegate.execution_resources()

    def evaluate_attempts(
        self,
        problem: Problem[int, int, ObservationPayload],
        requests: Sequence[EvaluationRequest[int]],
    ) -> EvaluationAttemptBatch[int, ObservationPayload]:
        """Evaluate requests through the ordinary coordinator callback path."""
        return self._delegate.evaluate_attempts(problem, requests)


class MalformedEpisodeEvaluator(SequentialEvaluator[int, int, ObservationPayload]):
    """Episode evaluator that drops one result after worker execution."""

    @override
    def evaluate_request_local_episodes(
        self,
        problem: Problem[int, int, ObservationPayload],
        episodes: Sequence[RequestLocalEpisode[int, int, ObservationPayload]],
    ) -> EvaluationAttemptBatch[int, ObservationPayload]:
        return super().evaluate_request_local_episodes(problem, episodes[:1])


class EpisodeOnlySequentialEvaluator(SequentialEvaluator[int, int, ObservationPayload]):
    """Evaluator that rejects the coordinator attempt path."""

    @override
    def evaluate_attempts(
        self,
        problem: Problem[int, int, ObservationPayload],
        requests: Sequence[EvaluationRequest[int]],
    ) -> EvaluationAttemptBatch[int, ObservationPayload]:
        del problem, requests
        raise AssertionError("coordinator attempt path must not run")


class FloatEpisodeOnlySequentialEvaluator(
    SequentialEvaluator[float, float, ObservationPayload]
):
    """Float evaluator that rejects the coordinator attempt path."""

    @override
    def evaluate_attempts(
        self,
        problem: Problem[float, float, ObservationPayload],
        requests: Sequence[EvaluationRequest[float]],
    ) -> EvaluationAttemptBatch[float, ObservationPayload]:
        del problem, requests
        raise AssertionError("coordinator attempt path must not run")


class ObservableHillClimbKernel(StructuredHillClimbKernel[int, int]):
    """Subclass whose observable ``run`` override must retain old dispatch."""

    run_count: ClassVar[int] = 0

    @override
    def run(
        self,
        query: ProposalBatchQuery[int, int, ObservationPayload],
        runner: Callable[
            [ProposalBatchQuery[int, int, ObservationPayload]],
            EvaluationAttemptBatch[int, ObservationPayload],
        ],
    ) -> EvaluationAttemptBatch[int, ObservationPayload]:
        type(self).run_count += 1
        return super().run(query, runner)


class ExplodingEpisodeKernel(
    RequestLocalEpisodeKernel[int, int, ObservationPayload],
):
    """Explicit episode kernel used to verify fail-closed reservations."""

    @override
    def run(
        self,
        query: ProposalBatchQuery[int, int, ObservationPayload],
        runner: Callable[
            [ProposalBatchQuery[int, int, ObservationPayload]],
            EvaluationAttemptBatch[int, ObservationPayload],
        ],
    ) -> EvaluationAttemptBatch[int, ObservationPayload]:
        return runner(query)

    @override
    def preferred_request_local_evaluation_limit(
        self,
        *,
        problem: Problem[int, int, ObservationPayload],
        request: EvaluationRequest[int],
        proposal_kernel_hint: ProposalKernelHint | None,
    ) -> int:
        del problem, request, proposal_kernel_hint
        return 3

    @override
    def run_request_local_episode(
        self,
        *,
        problem: Problem[int, int, ObservationPayload],
        episode: RequestLocalEpisode[int, int, ObservationPayload],
        runner: RequestLocalEvaluationRunner[int, ObservationPayload],
    ) -> EvaluationAttemptBatch[int, ObservationPayload]:
        del problem, episode, runner
        msg = "episode kernel failed"
        raise RuntimeError(msg)


class SnapshotBatchQueueOptimizer(BatchQueueOptimizer):
    """Batch queue that assigns deterministic proposal-local RNG streams."""

    @override
    def proposal_kernel_hints(
        self,
        state: BatchQueueOptimizerState,
        proposals: Sequence[Proposal[int]],
    ) -> tuple[ProposalLocalSearchContext, ...]:
        del state
        return tuple(
            ProposalLocalSearchContext(
                local_budget=2,
                random_state_snapshot=RandomStateSnapshot.from_seed(index + 17),
            )
            for index, _ in enumerate(proposals)
        )


class DisabledLocalSearchBatchQueueOptimizer(BatchQueueOptimizer):
    """Batch queue that disables proposal-local search."""

    @override
    def proposal_kernel_hints(
        self,
        state: BatchQueueOptimizerState,
        proposals: Sequence[Proposal[int]],
    ) -> tuple[ProposalLocalSearchContext, ...]:
        del state
        return tuple(ProposalLocalSearchContext(enabled=False) for _ in proposals)


class FloatSquareObjective(Objective[float]):
    """Pickle-safe scalar objective for SciPy Study dispatch."""

    @override
    def evaluate(self, candidate: float) -> float:
        return candidate * candidate


@dataclass(frozen=True, slots=True)
class FloatOneShotOptimizerState:
    """State for a one-proposal floating-point optimizer."""

    has_proposal: bool = True


class FloatOneShotOptimizer(
    RunMethod[
        FloatOneShotOptimizerState,
        Proposal[float],
        Observation[float],
    ],
):
    """Emit one floating-point proposal for SciPy dispatch tests."""

    @override
    def create_initial_state(self) -> FloatOneShotOptimizerState:
        return FloatOneShotOptimizerState()

    @override
    def is_exhausted(self, state: FloatOneShotOptimizerState) -> bool:
        return not state.has_proposal

    @override
    def ask(
        self,
        state: FloatOneShotOptimizerState,
        batch_size: int = 1,
    ) -> tuple[tuple[Proposal[float], ...], FloatOneShotOptimizerState]:
        del batch_size
        if not state.has_proposal:
            return (), state
        return (
            (Proposal(candidate=4.0, proposal_id="p-1"),),
            FloatOneShotOptimizerState(has_proposal=False),
        )

    @override
    def tell(
        self,
        state: FloatOneShotOptimizerState,
        observations: Sequence[Observation[float]],
    ) -> FloatOneShotOptimizerState:
        del observations
        return state


def run_hill_climb_study(
    evaluator: (
        CoordinatorOnlySequentialEvaluator
        | SequentialEvaluator[int, int, ObservationPayload]
        | JoblibEvaluator[int, int, ObservationPayload]
    ),
    *,
    max_evaluations: int = 34,
) -> tuple[tuple[Observation[int], ...], tuple[int, ...], int]:
    """Run one two-proposal hill-climb batch through ``evaluator``."""
    optimizer = BatchQueueOptimizer(
        proposal_batches=[
            (
                Proposal(candidate=4, proposal_id="p-1"),
                Proposal(candidate=6, proposal_id="p-2"),
            ),
        ],
    )
    study = Study(
        problem=Problem(
            space=IntegerSpace(low=0, high=10),
            objective=SquareObjective(),
        ),
        run_method=optimizer,
        evaluator=evaluator,
        kernel=StructuredHillClimbKernel[int, int](max_steps=8),
    )

    report, final_state = study.run(
        max_evaluations=max_evaluations,
        batch_size=2,
    )
    return (
        report.records,
        tuple(success.evaluation_count for success in report.successes),
        len(final_state.tell_history),
    )


def run_structured_kernel_study(
    *,
    kernel: RequestLocalEpisodeKernel[int, int, ObservationPayload],
    evaluator: (
        CoordinatorOnlySequentialEvaluator
        | SequentialEvaluator[int, int, ObservationPayload]
    ),
    run_method: BatchQueueOptimizer,
) -> tuple[tuple[Observation[int], ...], tuple[int, ...]]:
    """Run one structured kernel through a selected synchronous boundary."""
    study = Study(
        problem=Problem(
            space=IntegerSpace(low=0, high=10),
            objective=SquareObjective(),
        ),
        run_method=run_method,
        evaluator=evaluator,
        kernel=kernel,
    )
    report, _ = study.run(
        max_evaluations=128,
        batch_size=2,
    )
    return (
        report.records,
        tuple(success.evaluation_count for success in report.successes),
    )


class RequestLocalEpisodeDispatchTests:
    """Coverage for Study-level episode capability routing."""

    def test_max_min_limits_are_fair_and_deterministic(self) -> None:
        assert _max_min_evaluation_limits(
            (2, 5, 5),
            available_evaluations=8,
        ) == (2, 3, 3)
        assert (
            _max_min_evaluation_limits(
                (2, 5, 5),
                available_evaluations=2,
            )
            is None
        )
        assert _max_min_evaluation_limits(
            (1, 4, 9),
            available_evaluations=20,
        ) == (1, 4, 9)
        assert (
            _max_min_evaluation_limits(
                (),
                available_evaluations=0,
            )
            == ()
        )
        assert _max_min_evaluation_limits(
            (8, 3, 6, 3),
            available_evaluations=15,
        ) == (5, 3, 4, 3)

    def test_episode_paths_match_coordinator_execution(self) -> None:
        coordinator_result = run_hill_climb_study(
            CoordinatorOnlySequentialEvaluator(),
        )
        sequential_episode_result = run_hill_climb_study(
            SequentialEvaluator[int, int](),
        )
        joblib_episode_result = run_hill_climb_study(
            JoblibEvaluator[int, int](
                n_jobs=2,
                backend="threading",
            ),
        )

        assert sequential_episode_result == coordinator_result
        assert joblib_episode_result == coordinator_result

    @pytest.mark.parametrize("problem_transport", ("per_request", "worker_session"))
    def test_loky_restores_candidate_equality_before_materialization(
        self,
        problem_transport: Literal["per_request", "worker_session"],
    ) -> None:
        coordinator_result = run_hill_climb_study(
            CoordinatorOnlySequentialEvaluator(),
        )
        loky_result = run_hill_climb_study(
            JoblibEvaluator[int, int](
                n_jobs=2,
                backend="loky",
                problem_transport=problem_transport,
            ),
        )

        assert loky_result == coordinator_result

    def test_budget_partition_prevents_request_order_starvation(self) -> None:
        records, evaluation_counts, tell_count = run_hill_climb_study(
            SequentialEvaluator[int, int](),
            max_evaluations=8,
        )

        assert tuple(record.candidate for record in records) == (1, 3)
        assert evaluation_counts == (4, 4)
        assert tell_count == 1

    @pytest.mark.parametrize(
        "kernel",
        [
            StructuredScheduledLocalSearchKernel[int, int](
                max_steps=3,
                pair_move_leaf_limit=1,
            ),
            StructuredVariableNeighborhoodKernel[int, int](
                max_steps=3,
                stages=(
                    StructuredVariableNeighborhoodStage.leafwise_first_improvement(),
                ),
            ),
        ],
    )
    def test_deterministic_structured_families_match_coordinator(
        self,
        kernel: RequestLocalEpisodeKernel[int, int, ObservationPayload],
    ) -> None:
        proposal_batches: list[tuple[Proposal[int], ...]] = [
            (
                Proposal(candidate=4, proposal_id="p-1"),
                Proposal(candidate=6, proposal_id="p-2"),
            ),
        ]
        coordinator_result = run_structured_kernel_study(
            kernel=kernel,
            evaluator=CoordinatorOnlySequentialEvaluator(),
            run_method=BatchQueueOptimizer(proposal_batches=proposal_batches),
        )
        episode_result = run_structured_kernel_study(
            kernel=kernel,
            evaluator=SequentialEvaluator[int, int](),
            run_method=BatchQueueOptimizer(proposal_batches=proposal_batches),
        )

        assert episode_result == coordinator_result

    @pytest.mark.parametrize(
        "kernel",
        [
            StructuredStochasticNeighborhoodKernel[int, int](
                max_steps=3,
                max_neighbors_per_step=2,
            ),
            StructuredIteratedLocalSearchKernel[int, int](
                max_steps=3,
                max_kicks=2,
            ),
            StructuredVariableNeighborhoodKernel[int, int](
                max_steps=3,
                stages=(
                    StructuredVariableNeighborhoodStage.sampled_leafwise_first_improvement(
                        max_neighbors_per_step=2,
                    ),
                ),
            ),
        ],
    )
    def test_stochastic_structured_families_use_proposal_rng(
        self,
        kernel: RequestLocalEpisodeKernel[int, int, ObservationPayload],
    ) -> None:
        proposal_batches: list[tuple[Proposal[int], ...]] = [
            (
                Proposal(candidate=4, proposal_id="p-1"),
                Proposal(candidate=6, proposal_id="p-2"),
            ),
        ]
        coordinator_result = run_structured_kernel_study(
            kernel=kernel,
            evaluator=CoordinatorOnlySequentialEvaluator(),
            run_method=SnapshotBatchQueueOptimizer(
                proposal_batches=proposal_batches,
            ),
        )
        episode_result = run_structured_kernel_study(
            kernel=kernel,
            evaluator=SequentialEvaluator[int, int](),
            run_method=SnapshotBatchQueueOptimizer(
                proposal_batches=proposal_batches,
            ),
        )

        assert episode_result == coordinator_result

    def test_inherited_episode_methods_do_not_bypass_run_override(self) -> None:
        ObservableHillClimbKernel.run_count = 0
        optimizer = BatchQueueOptimizer(
            proposal_batches=[(Proposal(candidate=4, proposal_id="p-1"),)],
        )
        study = Study(
            problem=Problem(
                space=IntegerSpace(low=0, high=10),
                objective=SquareObjective(),
            ),
            run_method=optimizer,
            evaluator=SequentialEvaluator[int, int](),
            kernel=ObservableHillClimbKernel(max_steps=2),
        )

        report, _ = study.run(max_evaluations=5)

        assert report.evaluation_count == 3
        assert ObservableHillClimbKernel.run_count == 1

    def test_episode_exception_forfeits_the_full_reservation(self) -> None:
        optimizer = BatchQueueOptimizer(
            proposal_batches=[(Proposal(candidate=4, proposal_id="p-1"),)],
        )
        study = Study(
            problem=Problem(
                space=IntegerSpace(low=0, high=10),
                objective=SquareObjective(),
            ),
            run_method=optimizer,
            evaluator=SequentialEvaluator[int, int](),
            kernel=ExplodingEpisodeKernel(),
        )
        budget = EvaluationBudget(3)

        with pytest.raises(RuntimeError, match="episode kernel failed"):
            _ = evaluate_step(
                study,
                optimizer.create_initial_state(),
                batch_size=1,
                execution_model=SYNC_BATCH_EXECUTION_MODEL,
                evaluation_budget=budget,
            )

        assert budget.remaining == 0

    def test_episode_construction_failure_forfeits_the_full_reservation(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        optimizer = BatchQueueOptimizer(
            proposal_batches=[(Proposal(candidate=4, proposal_id="p-1"),)],
        )
        study = Study(
            problem=Problem(
                space=IntegerSpace(low=0, high=10),
                objective=SquareObjective(),
            ),
            run_method=optimizer,
            evaluator=SequentialEvaluator[int, int](),
            kernel=StructuredHillClimbKernel[int, int](max_steps=2),
        )
        budget = EvaluationBudget(5)

        def raise_during_episode_validation(
            self: RequestLocalEpisode[int, int, ObservationPayload],
        ) -> None:
            del self
            raise RuntimeError("episode construction failed")

        monkeypatch.setattr(
            RequestLocalEpisode,
            "__post_init__",
            raise_during_episode_validation,
        )

        with pytest.raises(RuntimeError, match="episode construction failed"):
            _ = evaluate_step(
                study,
                optimizer.create_initial_state(),
                batch_size=1,
                execution_model=SYNC_BATCH_EXECUTION_MODEL,
                evaluation_budget=budget,
            )

        assert budget.remaining == 0

    def test_early_convergence_refunds_unused_reservation(self) -> None:
        optimizer = BatchQueueOptimizer(
            proposal_batches=[(Proposal(candidate=0, proposal_id="p-1"),)],
        )
        study = Study(
            problem=Problem(
                space=IntegerSpace(low=0, high=10),
                objective=SquareObjective(),
            ),
            run_method=optimizer,
            evaluator=SequentialEvaluator[int, int](),
            kernel=StructuredHillClimbKernel[int, int](max_steps=8),
        )
        budget = EvaluationBudget(20)

        result = evaluate_step(
            study,
            optimizer.create_initial_state(),
            batch_size=1,
            execution_model=SYNC_BATCH_EXECUTION_MODEL,
            evaluation_budget=budget,
        )

        assert result.evaluation_count == 2
        assert budget.remaining == 18

    def test_malformed_episode_batch_forfeits_reservation(self) -> None:
        optimizer = BatchQueueOptimizer(
            proposal_batches=[
                (
                    Proposal(candidate=4, proposal_id="p-1"),
                    Proposal(candidate=6, proposal_id="p-2"),
                ),
            ],
        )
        study = Study(
            problem=Problem(
                space=IntegerSpace(low=0, high=10),
                objective=SquareObjective(),
            ),
            run_method=optimizer,
            evaluator=MalformedEpisodeEvaluator(),
            kernel=StructuredHillClimbKernel[int, int](max_steps=8),
        )
        budget = EvaluationBudget(8)

        with pytest.raises(
            ValueError,
            match="attempt batch must contain exactly one slot per request",
        ):
            _ = evaluate_step(
                study,
                optimizer.create_initial_state(),
                batch_size=2,
                execution_model=SYNC_BATCH_EXECUTION_MODEL,
                evaluation_budget=budget,
            )

        assert budget.remaining == 0

    def test_disabled_stochastic_search_needs_no_rng_snapshot(self) -> None:
        optimizer = DisabledLocalSearchBatchQueueOptimizer(
            proposal_batches=[(Proposal(candidate=4, proposal_id="p-1"),)],
        )
        study = Study(
            problem=Problem(
                space=IntegerSpace(low=0, high=10),
                objective=SquareObjective(),
            ),
            run_method=optimizer,
            evaluator=EpisodeOnlySequentialEvaluator(),
            kernel=StructuredStochasticNeighborhoodKernel[int, int](
                max_steps=3,
                max_neighbors_per_step=2,
            ),
        )

        report, _ = study.run(max_evaluations=1)

        assert report.evaluation_count == 1
        assert report.records[0].candidate == 4

    def test_scipy_kernel_dispatches_when_evaluation_cap_is_explicit(self) -> None:
        study = Study(
            problem=Problem(
                space=RealSpace(low=0.0, high=10.0),
                objective=FloatSquareObjective(),
            ),
            run_method=FloatOneShotOptimizer(),
            evaluator=FloatEpisodeOnlySequentialEvaluator(),
            kernel=ScipyMinimizeKernel[float, float](
                method="Powell",
                max_evaluations=5,
            ),
        )

        report, _ = study.run(max_evaluations=5)

        assert report.evaluation_count <= 5
        assert len(report.records) == 1
