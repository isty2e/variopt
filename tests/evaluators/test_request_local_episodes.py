"""Tests for request-local episode contracts and sequential execution."""

import pickle
from _thread import LockType
from collections.abc import Callable
from dataclasses import dataclass, replace
from threading import Lock

import pytest
from typing_extensions import override

from variopt import IntegerSpace, Objective, Problem
from variopt.artifacts import (
    CandidateRefinement,
    EvaluationAttemptBatch,
    EvaluationFailure,
    EvaluationRequest,
    EvaluationSuccess,
    ObservationPayload,
    Proposal,
)
from variopt.evaluators import SequentialEvaluator
from variopt.evaluators.episodes import (
    BoundedRequestLocalEvaluationRunner,
    RequestLocalEpisodeEvaluator,
    execute_request_local_episode,
)
from variopt.execution import (
    EvaluationBudgetExhausted,
    ExecutionResources,
    NestedParallelismPolicy,
)
from variopt.kernel import (
    ProposalBatchQuery,
    ProposalKernelHint,
    ProposalLocalSearchContext,
    RequestLocalEpisode,
    RequestLocalEpisodeKernel,
    RequestLocalEvaluationRunner,
)
from variopt.randomness import RandomStateSnapshot


class ExplodingObjective(Objective[int]):
    """Objective that rejects candidate four."""

    @override
    def evaluate(self, candidate: int) -> float:
        if candidate == 4:
            msg = "boom"
            raise ValueError(msg)
        return float(candidate * candidate)


class CountingObjective(Objective[int]):
    """Objective that records calls for execution-failure assertions."""

    def __init__(self) -> None:
        self.call_count = 0

    @override
    def evaluate(self, candidate: int) -> float:
        self.call_count += 1
        return float(candidate * candidate)


class InterruptingObjective(Objective[int]):
    """Objective that raises outside the recordable Exception boundary."""

    @override
    def evaluate(self, candidate: int) -> float:
        _ = candidate
        raise KeyboardInterrupt


def identity_candidate(candidate: int) -> int:
    """Return ``candidate`` unchanged."""
    return candidate


@dataclass(slots=True)
class LockedCandidateTransform:
    """Callable carrying a genuinely non-picklable thread lock."""

    lock: LockType

    def __call__(self, candidate: int) -> int:
        return candidate


@dataclass(frozen=True, slots=True)
class ConfiguredEpisodeKernel(
    RequestLocalEpisodeKernel[int, int, ObservationPayload],
):
    """Test kernel with configurable inner calls and reported cost."""

    candidates: tuple[int, ...]
    preferred_limit: int
    candidate_transform: Callable[[int], int] = identity_candidate
    reported_evaluation_count: int | None = None
    raise_after_evaluation_count: int | None = None
    report_refinement: bool = True

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
        _ = problem, request, proposal_kernel_hint
        return self.preferred_limit

    @override
    def run_request_local_episode(
        self,
        *,
        problem: Problem[int, int, ObservationPayload],
        episode: RequestLocalEpisode[int, int, ObservationPayload],
        runner: RequestLocalEvaluationRunner[int, ObservationPayload],
    ) -> EvaluationAttemptBatch[int, ObservationPayload]:
        _ = problem
        last_attempt = EvaluationAttemptBatch[int, ObservationPayload](attempts=())
        for candidate in self.candidates:
            transformed_candidate = self.candidate_transform(candidate)
            request = EvaluationRequest(
                proposal=Proposal(
                    candidate=transformed_candidate,
                    proposal_id=episode.request.proposal_id,
                ),
                proposal_evaluation_spec=episode.request.proposal_evaluation_spec,
            )
            last_attempt = runner.evaluate(request)
            if (
                self.raise_after_evaluation_count is not None
                and runner.consumed_evaluations == self.raise_after_evaluation_count
            ):
                msg = "episode failed"
                raise RuntimeError(msg)

        if last_attempt.attempt_count == 0:
            return last_attempt

        reported_count = (
            runner.consumed_evaluations
            if self.reported_evaluation_count is None
            else self.reported_evaluation_count
        )
        final_attempt = last_attempt.attempts[0]
        if isinstance(final_attempt, EvaluationSuccess):
            refinement = None
            if (
                self.report_refinement
                and final_attempt.request.candidate != episode.request.candidate
            ):
                refinement = CandidateRefinement(
                    source_candidate=episode.request.candidate,
                    refined_candidate=final_attempt.request.candidate,
                )
            return EvaluationAttemptBatch(
                attempts=(
                    EvaluationSuccess(
                        request=final_attempt.request,
                        payload=final_attempt.payload,
                        evaluation_count=reported_count,
                        refinement=refinement,
                    ),
                ),
            )

        if isinstance(final_attempt, EvaluationFailure):
            return EvaluationAttemptBatch(
                attempts=(
                    EvaluationFailure(
                        request=final_attempt.request,
                        exception=final_attempt.exception,
                        evaluation_count=reported_count,
                    ),
                ),
            )

        msg = "unexpected test attempt variant"
        raise AssertionError(msg)


def make_problem(
    objective: Objective[int] | None = None,
) -> Problem[int, int, ObservationPayload]:
    """Build the scalar test problem."""
    return Problem(
        space=IntegerSpace(low=0, high=10),
        objective=ExplodingObjective() if objective is None else objective,
    )


def make_episode(
    *,
    request_index: int = 0,
    candidate: int = 1,
    kernel: ConfiguredEpisodeKernel | None = None,
    evaluation_limit: int = 1,
) -> RequestLocalEpisode[int, int, ObservationPayload]:
    """Build one canonical request-local test episode."""
    return RequestLocalEpisode(
        request_index=request_index,
        request=EvaluationRequest(
            proposal=Proposal(
                candidate=candidate,
                proposal_id=f"request-{request_index}",
            ),
        ),
        kernel=(
            ConfiguredEpisodeKernel(
                candidates=(candidate,),
                preferred_limit=evaluation_limit,
            )
            if kernel is None
            else kernel
        ),
        proposal_kernel_hint=ProposalLocalSearchContext(local_budget=2),
        random_state_snapshot=RandomStateSnapshot.from_seed(7),
        evaluation_limit=evaluation_limit,
        execution_resources=ExecutionResources(
            parallel_owner="evaluator",
            nested_parallelism_policy=NestedParallelismPolicy.FORBID,
            owner_worker_count=1,
            owner_backend="sequential",
        ),
    )


def test_request_local_episode_round_trips_all_process_bound_configuration() -> None:
    episode = make_episode(
        kernel=ConfiguredEpisodeKernel(
            candidates=(1, 2),
            preferred_limit=3,
            candidate_transform=identity_candidate,
        ),
        evaluation_limit=3,
    )

    restored = pickle.loads(pickle.dumps(episode))

    assert restored == episode
    assert restored.kernel.candidate_transform(4) == 4
    assert restored.random_state_snapshot == episode.random_state_snapshot
    assert restored.execution_resources == episode.execution_resources


def test_request_local_episode_rejects_nonpicklable_kernel_configuration() -> None:
    episode = make_episode(
        kernel=ConfiguredEpisodeKernel(
            candidates=(1,),
            preferred_limit=1,
            candidate_transform=LockedCandidateTransform(lock=Lock()),
        ),
    )

    with pytest.raises(TypeError, match="cannot pickle.*lock"):
        _ = pickle.dumps(episode)


@pytest.mark.parametrize(
    ("request_index", "evaluation_limit", "exception_type"),
    [
        (-1, 1, ValueError),
        (True, 1, TypeError),
        (0, 0, ValueError),
        (0, -1, ValueError),
        (0, True, TypeError),
    ],
)
def test_request_local_episode_rejects_invalid_integer_boundaries(
    request_index: int,
    evaluation_limit: int,
    exception_type: type[Exception],
) -> None:
    with pytest.raises(exception_type):
        _ = make_episode(
            request_index=request_index,
            evaluation_limit=evaluation_limit,
        )


@pytest.mark.parametrize(
    "execution_resources",
    [
        ExecutionResources(
            parallel_owner="kernel",
            nested_parallelism_policy=NestedParallelismPolicy.FORBID,
        ),
        ExecutionResources(
            parallel_owner="evaluator",
            nested_parallelism_policy=NestedParallelismPolicy.ALLOW,
        ),
    ],
)
def test_request_local_episode_rejects_nonserial_worker_ownership(
    execution_resources: ExecutionResources,
) -> None:
    with pytest.raises(ValueError):
        _ = replace(
            make_episode(),
            execution_resources=execution_resources,
        )


def test_request_local_episode_rejects_competing_hint_rng_authority() -> None:
    snapshot = RandomStateSnapshot.from_seed(3)

    with pytest.raises(ValueError, match="owned only by random_state_snapshot"):
        _ = replace(
            make_episode(),
            proposal_kernel_hint=ProposalLocalSearchContext(
                random_state_snapshot=snapshot,
            ),
            random_state_snapshot=snapshot,
        )


def test_bounded_runner_consumes_before_each_objective_call() -> None:
    problem = make_problem()
    request = EvaluationRequest(proposal=Proposal(candidate=1))
    runner = BoundedRequestLocalEvaluationRunner(
        problem=problem,
        evaluation_limit=1,
    )

    attempt = runner.evaluate(request)

    assert attempt.evaluation_count == 1
    assert runner.consumed_evaluations == 1
    assert runner.remaining_evaluations == 0
    with pytest.raises(EvaluationBudgetExhausted, match="limit exhausted"):
        _ = runner.evaluate(request)
    assert runner.consumed_evaluations == 1


def test_bounded_runner_counts_recorded_objective_failure() -> None:
    problem = make_problem()
    runner = BoundedRequestLocalEvaluationRunner(
        problem=problem,
        evaluation_limit=1,
    )

    attempt = runner.evaluate(
        EvaluationRequest(proposal=Proposal(candidate=4)),
    )

    assert attempt.failure_indices == (0,)
    assert attempt.evaluation_count == 1
    assert runner.consumed_evaluations == 1


def test_bounded_runner_is_not_picklable() -> None:
    runner = BoundedRequestLocalEvaluationRunner(
        problem=make_problem(),
        evaluation_limit=1,
    )

    with pytest.raises(TypeError, match="worker-local and cannot be pickled"):
        _ = pickle.dumps(runner)


def test_sequential_episode_matches_direct_request_attempt() -> None:
    problem = make_problem()
    episode = make_episode()
    evaluator = SequentialEvaluator[int, int]()

    direct_attempt = evaluator.evaluate_attempts(problem, (episode.request,))
    episode_attempt = evaluator.evaluate_request_local_episodes(
        problem,
        (episode,),
    )

    assert episode_attempt == direct_attempt


def test_sequential_episode_reports_many_inner_calls() -> None:
    problem = make_problem()
    episode = make_episode(
        kernel=ConfiguredEpisodeKernel(
            candidates=(1, 2, 3),
            preferred_limit=3,
        ),
        evaluation_limit=3,
    )

    attempt = SequentialEvaluator[int, int]().evaluate_request_local_episodes(
        problem,
        (episode,),
    )

    assert attempt.evaluation_count == 3
    assert attempt.single_success_or_none() is not None
    assert attempt.requests[0].candidate == 3


def test_sequential_episode_releases_capacity_by_reporting_early_convergence() -> None:
    problem = make_problem()
    episode = make_episode(
        kernel=ConfiguredEpisodeKernel(
            candidates=(2,),
            preferred_limit=5,
        ),
        evaluation_limit=5,
    )

    attempt = SequentialEvaluator[int, int]().evaluate_request_local_episodes(
        problem,
        (episode,),
    )

    assert attempt.evaluation_count == 1


def test_sequential_episode_preserves_failure_after_partial_cost() -> None:
    problem = make_problem()
    episode = make_episode(
        candidate=4,
        kernel=ConfiguredEpisodeKernel(
            candidates=(1, 4),
            preferred_limit=2,
        ),
        evaluation_limit=2,
    )

    attempt = SequentialEvaluator[int, int]().evaluate_request_local_episodes(
        problem,
        (episode,),
    )

    assert attempt.failure_indices == (0,)
    assert attempt.evaluation_count == 2
    assert attempt.failures[0].request is not episode.request
    assert attempt.failures[0].request.candidate == episode.request.candidate


def test_episode_rejects_misaligned_objective_failure() -> None:
    episode = make_episode(
        candidate=1,
        kernel=ConfiguredEpisodeKernel(
            candidates=(4,),
            preferred_limit=1,
        ),
    )

    with pytest.raises(ValueError, match="align with input request order"):
        _ = execute_request_local_episode(
            problem=make_problem(),
            episode=episode,
        )


def test_episode_kernel_exception_escapes_after_partial_consumption() -> None:
    objective = CountingObjective()
    problem = make_problem(objective)
    episode = make_episode(
        kernel=ConfiguredEpisodeKernel(
            candidates=(1, 2),
            preferred_limit=2,
            raise_after_evaluation_count=1,
        ),
        evaluation_limit=2,
    )

    with pytest.raises(RuntimeError, match="episode failed"):
        _ = execute_request_local_episode(
            problem=problem,
            episode=episode,
        )

    assert objective.call_count == 1


def test_episode_does_not_capture_base_exception() -> None:
    episode = make_episode()

    with pytest.raises(KeyboardInterrupt):
        _ = execute_request_local_episode(
            problem=make_problem(InterruptingObjective()),
            episode=episode,
        )


@pytest.mark.parametrize(
    ("candidate_count", "reported_count", "match"),
    [
        (1, 0, "positive evaluation cost"),
        (2, 1, "disagrees with worker-local consumption"),
        (2, 3, "disagrees with worker-local consumption"),
    ],
)
def test_episode_rejects_invalid_reported_cost(
    candidate_count: int,
    reported_count: int,
    match: str,
) -> None:
    episode = make_episode(
        kernel=ConfiguredEpisodeKernel(
            candidates=tuple(range(1, candidate_count + 1)),
            preferred_limit=3,
            reported_evaluation_count=reported_count,
        ),
        evaluation_limit=3,
    )

    with pytest.raises(ValueError, match=match):
        _ = execute_request_local_episode(
            problem=make_problem(),
            episode=episode,
        )


def test_episode_rejects_zero_inner_calls() -> None:
    episode = make_episode(
        kernel=ConfiguredEpisodeKernel(
            candidates=(),
            preferred_limit=1,
        ),
    )

    with pytest.raises(ValueError, match="exactly one top-level attempt"):
        _ = execute_request_local_episode(
            problem=make_problem(),
            episode=episode,
        )


def test_episode_rejects_misaligned_success_without_refinement() -> None:
    episode = make_episode(
        candidate=1,
        kernel=ConfiguredEpisodeKernel(
            candidates=(2,),
            preferred_limit=1,
            report_refinement=False,
        ),
    )

    with pytest.raises(ValueError, match="align with input request order"):
        _ = execute_request_local_episode(
            problem=make_problem(),
            episode=episode,
        )


def test_episode_cannot_exceed_its_hard_limit() -> None:
    episode = make_episode(
        kernel=ConfiguredEpisodeKernel(
            candidates=(1, 2),
            preferred_limit=2,
        ),
        evaluation_limit=1,
    )

    with pytest.raises(EvaluationBudgetExhausted, match="limit exhausted"):
        _ = execute_request_local_episode(
            problem=make_problem(),
            episode=episode,
        )


def test_sequential_episode_supports_empty_batch() -> None:
    attempts = SequentialEvaluator[int, int]().evaluate_request_local_episodes(
        make_problem(),
        (),
    )

    assert attempts.attempt_count == 0
    assert attempts.evaluation_count == 0


def test_sequential_episode_reorders_transport_input_by_request_index() -> None:
    first_episode = make_episode(request_index=0, candidate=1)
    second_episode = make_episode(request_index=1, candidate=2)

    attempts = SequentialEvaluator[int, int]().evaluate_request_local_episodes(
        make_problem(),
        (second_episode, first_episode),
    )

    assert tuple(request.candidate for request in attempts.requests) == (1, 2)


def test_request_indices_disambiguate_equal_requests_and_proposal_ids() -> None:
    first_episode = replace(
        make_episode(request_index=0, candidate=1),
        request=EvaluationRequest(
            proposal=Proposal(candidate=1, proposal_id="duplicate"),
        ),
        kernel=ConfiguredEpisodeKernel(
            candidates=(2,),
            preferred_limit=1,
        ),
    )
    second_episode = replace(
        make_episode(request_index=1, candidate=1),
        request=EvaluationRequest(
            proposal=Proposal(candidate=1, proposal_id="duplicate"),
        ),
        kernel=ConfiguredEpisodeKernel(
            candidates=(3,),
            preferred_limit=1,
        ),
    )

    attempts = SequentialEvaluator[int, int]().evaluate_request_local_episodes(
        make_problem(),
        (second_episode, first_episode),
    )

    assert tuple(request.candidate for request in attempts.requests) == (2, 3)


@pytest.mark.parametrize(
    "request_indices",
    [
        (0, 0),
        (0, 2),
        (1,),
    ],
)
def test_sequential_episode_rejects_noncontiguous_request_indices_before_work(
    request_indices: tuple[int, ...],
) -> None:
    objective = CountingObjective()
    episodes = tuple(
        make_episode(request_index=request_index) for request_index in request_indices
    )

    with pytest.raises(ValueError, match="unique and contiguous"):
        _ = SequentialEvaluator[int, int]().evaluate_request_local_episodes(
            make_problem(objective),
            episodes,
        )

    assert objective.call_count == 0


def test_sequential_evaluator_exposes_request_local_episode_capability() -> None:
    evaluator = SequentialEvaluator[int, int]()

    assert isinstance(evaluator, RequestLocalEpisodeEvaluator)
