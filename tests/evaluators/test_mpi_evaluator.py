"""Tests for the optional MPI-backed evaluator."""

from collections.abc import Callable
from typing import Generic, TypeVar, final

import pytest
from typing_extensions import override

from variopt import (
    EvaluationAttemptBatch,
    EvaluationRequest,
    IntegerSpace,
    Objective,
    Observation,
    Problem,
    Proposal,
)
from variopt.evaluators import MpiEvaluator, MpiExecutorFactory
from variopt.evaluators import mpi as mpi_evaluator_module
from variopt.execution import (
    ExecutionResources,
    NestedParallelismPolicy,
)

MpiResultT = TypeVar("MpiResultT")


def _requests(
    proposals: tuple[Proposal[int], ...],
) -> tuple[EvaluationRequest[int], ...]:
    """Lower proposal fixtures into canonical evaluation requests."""
    return tuple(EvaluationRequest(proposal=proposal) for proposal in proposals)


def _successful_observation_values(
    attempts: EvaluationAttemptBatch[int],
) -> tuple[float, ...]:
    """Return scalar objective values from successful attempt slots."""
    return tuple(success.scalar_observation().value for success in attempts.successes)


class SquareObjective(Objective[int]):
    """Toy objective used to test MPI evaluator behavior."""

    @override
    def evaluate(self, candidate: int) -> float:
        return float(candidate * candidate)


class ExplodingObjective(Objective[int]):
    """Objective that raises for one candidate."""

    @override
    def evaluate(self, candidate: int) -> float:
        if candidate == 4:
            msg = "boom"
            raise ValueError(msg)
        return float(candidate * candidate)


@final
class _FakeMpiFuture(Generic[MpiResultT]):
    """Fake future that resolves the submitted callable lazily."""

    def __init__(
        self,
        thunk: Callable[
            [],
            tuple[int, MpiResultT],
        ],
    ) -> None:
        self._thunk = thunk

    def result(self) -> tuple[int, MpiResultT]:
        return self._thunk()


@final
class _FakeMpiExecutor:
    """Fake executor that runs submitted callables on future resolution."""

    def __init__(self) -> None:
        self.submission_count = 0
        self.shutdown_calls: list[bool] = []

    def submit(
        self,
        function: Callable[
            ...,
            tuple[int, MpiResultT],
        ],
        /,
        *args: object,
        **kwargs: object,
    ) -> _FakeMpiFuture[MpiResultT]:
        self.submission_count += 1

        def thunk() -> tuple[int, MpiResultT]:
            return function(*args, **kwargs)

        return _FakeMpiFuture(thunk)

    def shutdown(self, wait: bool = True) -> None:
        self.shutdown_calls.append(wait)


@final
class _MisalignedFakeMpiExecutor:
    """Fake executor that corrupts logical result indices."""

    def __init__(self) -> None:
        self.submission_count = 0
        self.shutdown_calls: list[bool] = []

    def submit(
        self,
        function: Callable[
            ...,
            tuple[int, MpiResultT],
        ],
        /,
        *args: object,
        **kwargs: object,
    ) -> _FakeMpiFuture[MpiResultT]:
        self.submission_count += 1

        def thunk() -> tuple[int, MpiResultT]:
            resolved_index, result = function(*args, **kwargs)
            return resolved_index + 1, result

        return _FakeMpiFuture(thunk)

    def shutdown(self, wait: bool = True) -> None:
        self.shutdown_calls.append(wait)


def _build_executor_factory(
    executor: _FakeMpiExecutor,
) -> MpiExecutorFactory[int, Observation[int]]:
    """Return one typed fake executor factory."""

    def factory(*, max_workers: int | None = None) -> _FakeMpiExecutor:
        _ = max_workers
        return executor

    return factory


def _build_misaligned_executor_factory(
    executor: _MisalignedFakeMpiExecutor,
) -> MpiExecutorFactory[int, Observation[int]]:
    """Return one typed misaligned fake executor factory."""

    def factory(*, max_workers: int | None = None) -> _MisalignedFakeMpiExecutor:
        _ = max_workers
        return executor

    return factory


class MpiEvaluatorTests:
    """Tests for MpiEvaluator."""

    def test_rejects_non_positive_max_workers(self) -> None:
        with pytest.raises(ValueError):
            _ = MpiEvaluator[int, int](max_workers=0)

    def test_preserves_input_proposal_order(self) -> None:
        executor = _FakeMpiExecutor()
        problem = Problem(
            space=IntegerSpace(low=0, high=10),
            objective=SquareObjective(),
        )
        evaluator = MpiEvaluator[int, int](
            max_workers=3,
            _executor_factory=_build_executor_factory(executor),
        )

        outcomes = evaluator.evaluate(
            problem,
            _requests(
                (
                Proposal(candidate=4, proposal_id="p-1"),
                Proposal(candidate=1, proposal_id="p-2"),
                )
            ),
        )

        assert tuple(outcome.observation.proposal.proposal_id for outcome in outcomes) == ("p-1", "p-2")
        assert tuple(outcome.observation.value for outcome in outcomes) == (16.0, 1.0)
        assert executor.submission_count == 2
        assert executor.shutdown_calls == [True]

    def test_evaluate_attempts_records_user_failure_and_preserves_successes(
        self,
    ) -> None:
        executor = _FakeMpiExecutor()
        problem = Problem(
            space=IntegerSpace(low=0, high=10),
            objective=ExplodingObjective(),
        )
        evaluator = MpiEvaluator[int, int](
            max_workers=3,
            _executor_factory=_build_executor_factory(executor),
        )

        attempts = evaluator.evaluate_attempts(
            problem,
            _requests(
                (
                    Proposal(candidate=1, proposal_id="p-1"),
                    Proposal(candidate=4, proposal_id="p-2"),
                    Proposal(candidate=2, proposal_id="p-3"),
                )
            ),
        )

        assert attempts.success_indices == (0, 2)
        assert attempts.failure_indices == (1,)
        assert _successful_observation_values(attempts) == (1.0, 4.0)
        assert attempts.failures[0].proposal_id == "p-2"
        assert attempts.failures[0].exception.exception_type == "builtins.ValueError"
        assert executor.submission_count == 3
        assert executor.shutdown_calls == [True]

    def test_evaluate_attempts_support_empty_batch(self) -> None:
        executor = _FakeMpiExecutor()
        problem = Problem(
            space=IntegerSpace(low=0, high=10),
            objective=ExplodingObjective(),
        )
        evaluator = MpiEvaluator[int, int](
            max_workers=3,
            _executor_factory=_build_executor_factory(executor),
        )

        attempts = evaluator.evaluate_attempts(problem, ())

        assert attempts.requests == ()
        assert attempts.successes == ()
        assert attempts.failures == ()
        assert attempts.evaluation_count == 0
        assert executor.submission_count == 0
        assert executor.shutdown_calls == [True]

    def test_evaluate_attempts_rejects_misaligned_future_result(self) -> None:
        executor = _MisalignedFakeMpiExecutor()
        problem = Problem(
            space=IntegerSpace(low=0, high=10),
            objective=SquareObjective(),
        )
        evaluator = MpiEvaluator[int, int](
            max_workers=3,
            _executor_factory=_build_misaligned_executor_factory(executor),
        )

        with pytest.raises(ValueError, match="misaligned proposal attempt"):
            _ = evaluator.evaluate_attempts(
                problem,
                _requests((Proposal(candidate=1, proposal_id="p-1"),)),
            )

        assert executor.submission_count == 1
        assert executor.shutdown_calls == [True]

    def test_execution_resources_are_mpi_owned(self) -> None:
        evaluator = MpiEvaluator[int, int](max_workers=4)

        assert evaluator.execution_resources() == ExecutionResources(
                parallel_owner="evaluator",
                nested_parallelism_policy=NestedParallelismPolicy.FORBID,
                owner_worker_count=4,
                owner_backend="mpi",
            )

    def test_raises_helpful_error_when_mpi4py_is_unavailable(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        problem = Problem(
            space=IntegerSpace(low=0, high=10),
            objective=SquareObjective(),
        )
        evaluator = MpiEvaluator[int, int]()

        def fail_import(_name: str) -> object:
            raise ImportError("No module named 'mpi4py'")

        monkeypatch.setattr(mpi_evaluator_module, "import_module", fail_import)

        with pytest.raises(ImportError, match="optional mpi extra"):
            _ = evaluator.evaluate(
                problem,
                _requests((Proposal(candidate=1, proposal_id="p-1"),)),
            )
