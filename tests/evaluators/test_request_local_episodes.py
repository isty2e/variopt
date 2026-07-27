"""Tests for request-local episode contracts and sequential execution."""

import pickle
from _thread import LockType
from collections.abc import Callable
from dataclasses import dataclass, replace
from os import _exit, getpid
from pathlib import Path
from threading import Barrier, Lock, current_thread
from time import sleep
from typing import Literal

import pytest
from typing_extensions import override

import variopt.evaluators.joblib.sync as joblib_sync
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
from variopt.evaluators import JoblibEvaluator, SequentialEvaluator
from variopt.evaluators.episodes import (
    BoundedRequestLocalEvaluationRunner,
    RequestLocalEpisodeEvaluator,
    execute_request_local_episode,
    ordered_request_local_episode_attempts,
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


class ProcessIdObjective(Objective[int]):
    """Objective that identifies the process executing the request."""

    @override
    def evaluate(self, candidate: int) -> float:
        _ = candidate
        return float(getpid())


class DelayedEpisodeObjective(Objective[int]):
    """Objective that makes lower candidates finish later."""

    def __init__(self) -> None:
        self.completion_order: list[int] = []
        self.lock = Lock()

    @override
    def evaluate(self, candidate: int) -> float:
        if candidate == 1:
            sleep(0.05)
        with self.lock:
            self.completion_order.append(candidate)
        return float(candidate * candidate)


class ThreadRecordingObjective(Objective[int]):
    """Thread-safe objective that records coordinator-visible worker calls."""

    def __init__(self, barrier: Barrier | None = None) -> None:
        self.barrier = barrier
        self.calling_thread_names: list[str] = []
        self.lock = Lock()

    @override
    def evaluate(self, candidate: int) -> float:
        barrier = self.barrier
        if barrier is not None:
            _ = barrier.wait(timeout=5.0)
        with self.lock:
            self.calling_thread_names.append(current_thread().name)
        return float(candidate * candidate)


class LockedObjective(Objective[int]):
    """Objective carrying a nonserializable synchronization primitive."""

    def __init__(self) -> None:
        self.lock = Lock()

    @override
    def evaluate(self, candidate: int) -> float:
        with self.lock:
            return float(candidate * candidate)


class TerminatingObjective(Objective[int]):
    """Objective that terminates only its isolated worker process."""

    def __init__(self, marker_path: Path) -> None:
        self.marker_path = marker_path

    @override
    def evaluate(self, candidate: int) -> float:
        _ = candidate
        with self.marker_path.open("a", encoding="utf-8") as marker:
            _ = marker.write("attempt\n")
        _exit(17)


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


class JoblibRequestLocalEpisodeTests:
    """Tests for synchronous Joblib request-local episode execution."""

    @pytest.mark.parametrize("problem_transport", ("per_request", "worker_session"))
    def test_loky_executes_episode_across_process_boundary(
        self,
        problem_transport: Literal["per_request", "worker_session"],
    ) -> None:
        problem = make_problem(ProcessIdObjective())
        evaluator = JoblibEvaluator[int, int, ObservationPayload](
            backend="loky",
            n_jobs=2,
            problem_transport=problem_transport,
        )

        attempts = evaluator.evaluate_request_local_episodes(
            problem,
            (make_episode(),),
        )

        success = attempts.single_success_or_none()
        assert success is not None
        assert success.payload.value != float(getpid())

    @pytest.mark.parametrize("problem_transport", ("per_request", "worker_session"))
    def test_threading_shares_exact_problem_state(
        self,
        problem_transport: Literal["per_request", "worker_session"],
    ) -> None:
        objective = ThreadRecordingObjective()
        problem = make_problem(objective)
        evaluator = JoblibEvaluator[int, int, ObservationPayload](
            backend="threading",
            n_jobs=2,
            problem_transport=problem_transport,
        )

        attempts = evaluator.evaluate_request_local_episodes(
            problem,
            (
                make_episode(request_index=1, candidate=2),
                make_episode(request_index=0, candidate=1),
            ),
        )

        assert attempts.evaluation_count == 2
        assert len(objective.calling_thread_names) == 2
        assert all(
            thread_name != current_thread().name
            for thread_name in objective.calling_thread_names
        )

    def test_threading_executes_independent_episodes_concurrently(self) -> None:
        objective = ThreadRecordingObjective(barrier=Barrier(2))
        problem = make_problem(objective)
        evaluator = JoblibEvaluator[int, int, ObservationPayload](
            backend="threading",
            n_jobs=2,
        )

        attempts = evaluator.evaluate_request_local_episodes(
            problem,
            (
                make_episode(request_index=0, candidate=1),
                make_episode(request_index=1, candidate=2),
            ),
        )

        assert attempts.evaluation_count == 2
        assert len(set(objective.calling_thread_names)) == 2

    def test_reorders_results_by_explicit_request_index(self) -> None:
        objective = DelayedEpisodeObjective()
        problem = make_problem(objective)
        evaluator = JoblibEvaluator[int, int, ObservationPayload](
            backend="threading",
            n_jobs=2,
        )

        attempts = evaluator.evaluate_request_local_episodes(
            problem,
            (
                make_episode(request_index=1, candidate=2),
                make_episode(request_index=0, candidate=1),
            ),
        )

        assert tuple(request.candidate for request in attempts.requests) == (1, 2)
        assert tuple(success.payload.value for success in attempts.successes) == (
            1.0,
            4.0,
        )
        assert objective.completion_order == [2, 1]

    def test_request_indices_disambiguate_equal_joblib_requests(self) -> None:
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

        attempts = JoblibEvaluator[int, int, ObservationPayload](
            backend="threading",
            n_jobs=2,
        ).evaluate_request_local_episodes(
            make_problem(),
            (second_episode, first_episode),
        )

        assert tuple(request.candidate for request in attempts.requests) == (2, 3)
        assert tuple(request.proposal_id for request in attempts.requests) == (
            "duplicate",
            "duplicate",
        )

    @pytest.mark.parametrize(
        "request_indices",
        (
            (0, 0),
            (0, 2),
            (1,),
        ),
    )
    def test_rejects_invalid_indices_before_joblib_dispatch(
        self,
        request_indices: tuple[int, ...],
    ) -> None:
        objective = CountingObjective()
        episodes = tuple(
            make_episode(request_index=request_index)
            for request_index in request_indices
        )

        with pytest.raises(ValueError, match="unique and contiguous"):
            _ = JoblibEvaluator[int, int, ObservationPayload](
                backend="threading",
                n_jobs=2,
            ).evaluate_request_local_episodes(
                make_problem(objective),
                episodes,
            )

        assert objective.call_count == 0

    @pytest.mark.parametrize("problem_transport", ("per_request", "worker_session"))
    def test_single_job_executes_inline_without_serializing_problem(
        self,
        problem_transport: Literal["per_request", "worker_session"],
    ) -> None:
        attempts = JoblibEvaluator[int, int, ObservationPayload](
            backend="loky",
            n_jobs=1,
            problem_transport=problem_transport,
        ).evaluate_request_local_episodes(
            make_problem(LockedObjective()),
            (make_episode(),),
        )

        assert attempts.evaluation_count == 1

    def test_unequal_episode_limits_bound_reported_cost(self) -> None:
        episodes = (
            make_episode(
                request_index=0,
                kernel=ConfiguredEpisodeKernel(
                    candidates=(1, 2),
                    preferred_limit=2,
                ),
                evaluation_limit=2,
            ),
            make_episode(
                request_index=1,
                candidate=3,
                evaluation_limit=1,
            ),
            make_episode(
                request_index=2,
                kernel=ConfiguredEpisodeKernel(
                    candidates=(4, 5, 6),
                    preferred_limit=3,
                ),
                evaluation_limit=3,
            ),
        )

        attempts = JoblibEvaluator[int, int, ObservationPayload](
            backend="threading",
            n_jobs=3,
        ).evaluate_request_local_episodes(
            make_problem(),
            episodes,
        )

        assert attempts.evaluation_count == 6
        assert tuple(attempt.evaluation_count for attempt in attempts.attempts) == (
            2,
            1,
            3,
        )
        assert attempts.evaluation_count == sum(
            episode.evaluation_limit for episode in episodes
        )

    def test_preserves_recorded_failure_after_partial_cost(self) -> None:
        episode = make_episode(
            candidate=4,
            kernel=ConfiguredEpisodeKernel(
                candidates=(1, 4),
                preferred_limit=2,
            ),
            evaluation_limit=2,
        )

        attempts = JoblibEvaluator[int, int, ObservationPayload](
            backend="threading",
            n_jobs=2,
        ).evaluate_request_local_episodes(
            make_problem(),
            (episode,),
        )

        assert attempts.failure_indices == (0,)
        assert attempts.evaluation_count == 2

    def test_kernel_exception_escapes_after_partial_cost(self) -> None:
        objective = CountingObjective()
        episode = make_episode(
            kernel=ConfiguredEpisodeKernel(
                candidates=(1, 2),
                preferred_limit=2,
                raise_after_evaluation_count=1,
            ),
            evaluation_limit=2,
        )

        with pytest.raises(RuntimeError, match="episode failed"):
            _ = JoblibEvaluator[int, int, ObservationPayload](
                backend="threading",
                n_jobs=2,
            ).evaluate_request_local_episodes(
                make_problem(objective),
                (episode,),
            )

        assert objective.call_count == 1

    @pytest.mark.parametrize("problem_transport", ("per_request", "worker_session"))
    def test_loky_rejects_nonserializable_problem_before_execution(
        self,
        problem_transport: Literal["per_request", "worker_session"],
    ) -> None:
        expected_error = (
            pickle.PicklingError if problem_transport == "per_request" else TypeError
        )
        with pytest.raises(expected_error) as captured_exception:
            _ = JoblibEvaluator[int, int, ObservationPayload](
                backend="loky",
                n_jobs=2,
                problem_transport=problem_transport,
            ).evaluate_request_local_episodes(
                make_problem(LockedObjective()),
                (make_episode(),),
            )
        if problem_transport == "per_request":
            assert "cannot pickle" in str(captured_exception.value.__cause__)
        else:
            assert "cannot pickle" in str(captured_exception.value)

    def test_loky_propagates_worker_termination_without_retry(
        self,
        tmp_path: Path,
    ) -> None:
        marker_path = tmp_path / "worker-attempts.txt"
        with pytest.raises(Exception) as captured_exception:
            _ = JoblibEvaluator[int, int, ObservationPayload](
                backend="loky",
                n_jobs=2,
            ).evaluate_request_local_episodes(
                make_problem(TerminatingObjective(marker_path)),
                (make_episode(),),
            )

        assert type(captured_exception.value).__name__ == "TerminatedWorkerError"
        assert marker_path.read_text(encoding="utf-8") == "attempt\n"

    def test_rejects_misaligned_worker_result(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        original_task = joblib_sync.evaluate_request_local_episode_task

        def misaligned_task(
            *,
            problem: Problem[int, int, ObservationPayload],
            episode: RequestLocalEpisode[int, int, ObservationPayload],
        ) -> tuple[int, EvaluationAttemptBatch[int, ObservationPayload]]:
            _, attempt = original_task(problem=problem, episode=episode)
            return episode.request_index + 1, attempt

        monkeypatch.setattr(
            joblib_sync,
            "evaluate_request_local_episode_task",
            misaligned_task,
        )

        with pytest.raises(ValueError, match="do not align"):
            _ = JoblibEvaluator[int, int, ObservationPayload](
                backend="threading",
                n_jobs=2,
            ).evaluate_request_local_episodes(
                make_problem(),
                (make_episode(),),
            )

    def test_rejects_malformed_one_slot_attempt(self) -> None:
        with pytest.raises(ValueError, match="exactly one request"):
            _ = ordered_request_local_episode_attempts(
                ((0, EvaluationAttemptBatch[int, ObservationPayload](attempts=())),),
                request_count=1,
            )

    @pytest.mark.parametrize(
        "malformed_result",
        (
            (),
            (0,),
            (0, "not-an-attempt"),
            (True, EvaluationAttemptBatch[int, ObservationPayload](attempts=())),
        ),
    )
    def test_rejects_malformed_worker_result_envelope(
        self,
        malformed_result: object,
    ) -> None:
        with pytest.raises(TypeError):
            _ = ordered_request_local_episode_attempts(
                (malformed_result,),
                request_count=1,
            )

    def test_supports_empty_batch_and_structural_capability(self) -> None:
        evaluator = JoblibEvaluator[int, int, ObservationPayload](
            backend="threading",
            n_jobs=2,
        )

        attempts = evaluator.evaluate_request_local_episodes(make_problem(), ())

        assert attempts.attempt_count == 0
        assert isinstance(evaluator, RequestLocalEpisodeEvaluator)
