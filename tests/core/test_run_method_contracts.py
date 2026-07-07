"""Run-method assimilation contract regression tests."""

import pickle

import pytest
from typing_extensions import TypeVar, override

from tests.study_support import (
    BatchQueueOptimizer,
    BatchQueueOptimizerState,
    OutcomeAwareBatchQueueOptimizer,
)
from variopt import (
    CandidateRefinement,
    EvaluationAttemptBatch,
    EvaluationFailure,
    EvaluationRequest,
    Observation,
    OptimizationDirection,
    Proposal,
    UnsupportedEvaluationFailureError,
)
from variopt.artifacts import EvaluationSuccess

OutcomeCandidateT = TypeVar("OutcomeCandidateT")


def _request(candidate: int, proposal_id: str) -> EvaluationRequest[int]:
    return EvaluationRequest(
        proposal=Proposal(candidate=candidate, proposal_id=proposal_id),
    )


def _success(
    request: EvaluationRequest[int],
    *,
    candidate: int | None = None,
    changed: bool = False,
) -> EvaluationSuccess[int, Observation[int]]:
    evaluated_candidate = request.candidate if candidate is None else candidate
    success_request = request
    if evaluated_candidate != request.candidate:
        success_request = EvaluationRequest(
            proposal=Proposal(
                candidate=evaluated_candidate,
                proposal_id=request.proposal_id,
            ),
            proposal_evaluation_spec=request.proposal_evaluation_spec,
        )
    refinement = None
    if changed:
        refinement = CandidateRefinement(
            source_candidate=request.candidate,
            refined_candidate=evaluated_candidate,
            changed_leaf_paths=((),),
        )
    observation = Observation.from_objective_value(
        request=success_request,
        candidate=evaluated_candidate,
        value=float(evaluated_candidate),
        direction=OptimizationDirection.MINIMIZE,
    )
    return EvaluationSuccess(
        request=success_request,
        payload=observation,
        refinement=refinement,
    )


def _failure(
    request: EvaluationRequest[int],
    *,
    evaluation_count: int = 1,
) -> EvaluationFailure[int]:
    return EvaluationFailure[int].from_exception(
        request=request,
        exception=ValueError(f"candidate failed: {request.candidate}"),
        evaluation_count=evaluation_count,
    )


class AttemptAwareBatchQueueOptimizer(OutcomeAwareBatchQueueOptimizer):
    """Test optimizer that consumes failed attempt lifecycle explicitly."""

    failure_history: tuple[tuple[str | None, ...], ...]

    def __init__(self, proposal_batches: list[tuple[Proposal[int], ...]]) -> None:
        super().__init__(proposal_batches)
        self.failure_history = ()

    @override
    def tell_attempts(
        self,
        state: BatchQueueOptimizerState,
        attempts: EvaluationAttemptBatch[OutcomeCandidateT, Observation[int]],
    ) -> BatchQueueOptimizerState:
        self.failure_history += (
            tuple(failure.proposal_id for failure in attempts.failures),
        )
        success_attempts: EvaluationAttemptBatch[
            OutcomeCandidateT,
            Observation[int],
        ] = EvaluationAttemptBatch(attempts=attempts.successes)
        return super().tell_attempts(state, success_attempts)


class RunMethodAttemptAssimilationTests:
    """Regression tests for the attempt-aware run-method hook."""

    def test_success_only_attempts_delegate_to_outcome_aware_hook(self) -> None:
        request_one = _request(4, "p-1")
        request_two = _request(2, "p-2")
        success_one = _success(request_one, candidate=3, changed=True)
        success_two = _success(request_two)
        optimizer = OutcomeAwareBatchQueueOptimizer(proposal_batches=[])
        state = optimizer.create_initial_state()
        attempts = EvaluationAttemptBatch(
            attempts=(success_one, success_two),
        )

        next_state = optimizer.tell_attempts(state, attempts)

        observed_one, observed_two = next_state.tell_history[0]
        assert observed_one.proposal.candidate == 4
        assert observed_one.candidate == 3
        assert observed_one is not success_one.payload
        assert observed_two == success_two.payload
        assert observed_two is success_two.payload
        assert optimizer.seen_changed_leaf_paths == (((),), None)

    def test_empty_success_only_attempts_keep_default_compatibility(self) -> None:
        optimizer = BatchQueueOptimizer(proposal_batches=[])
        state = optimizer.create_initial_state()
        attempts: EvaluationAttemptBatch[int, Observation[int]] = (
            EvaluationAttemptBatch(attempts=())
        )

        next_state = optimizer.tell_attempts(state, attempts)

        assert next_state.tell_history == ((),)

    def test_default_attempt_assimilation_rejects_mixed_failures(self) -> None:
        request_one = _request(1, "p-1")
        request_two = _request(2, "p-2")
        request_three = _request(3, "p-3")
        attempts = EvaluationAttemptBatch(
            attempts=(
                _success(request_one),
                _failure(request_two),
                _success(request_three),
            ),
        )
        optimizer = BatchQueueOptimizer(proposal_batches=[])
        state = optimizer.create_initial_state()

        with pytest.raises(
            UnsupportedEvaluationFailureError,
            match="does not support evaluation failure assimilation",
        ) as exc_info:
            _ = optimizer.tell_attempts(state, attempts)

        error = exc_info.value
        assert error.failure_count == 1
        assert error.attempt_count == 3
        assert state.tell_history == ()

    def test_default_attempt_assimilation_rejects_all_failure_batch(self) -> None:
        request_one = _request(1, "p-1")
        request_two = _request(2, "p-2")
        attempts: EvaluationAttemptBatch[int, Observation[int]] = (
            EvaluationAttemptBatch(
                attempts=(_failure(request_one), _failure(request_two)),
            )
        )
        optimizer = BatchQueueOptimizer(proposal_batches=[])

        with pytest.raises(UnsupportedEvaluationFailureError) as exc_info:
            _ = optimizer.tell_attempts(optimizer.create_initial_state(), attempts)

        assert exc_info.value.failure_count == 2
        assert exc_info.value.attempt_count == 2

    def test_override_can_consume_middle_failure_without_score_evidence(self) -> None:
        request_one = _request(1, "p-1")
        request_two = _request(2, "p-2")
        request_three = _request(3, "p-3")
        success_one = _success(request_one)
        success_three = _success(request_three)
        attempts = EvaluationAttemptBatch(
            attempts=(success_one, _failure(request_two), success_three),
        )
        optimizer = AttemptAwareBatchQueueOptimizer(proposal_batches=[])

        next_state = optimizer.tell_attempts(
            optimizer.create_initial_state(),
            attempts,
        )

        assert optimizer.failure_history == (("p-2",),)
        assert next_state.tell_history == (
            (success_one.payload, success_three.payload),
        )

    def test_override_can_consume_zero_cost_failure(self) -> None:
        request_one = _request(1, "p-1")
        request_two = _request(2, "p-2")
        success_two = _success(request_two)
        attempts = EvaluationAttemptBatch(
            attempts=(_failure(request_one, evaluation_count=0), success_two),
        )
        optimizer = AttemptAwareBatchQueueOptimizer(proposal_batches=[])

        next_state = optimizer.tell_attempts(
            optimizer.create_initial_state(),
            attempts,
        )

        assert optimizer.failure_history == (("p-1",),)
        assert next_state.tell_history == ((success_two.payload,),)

    @pytest.mark.parametrize(
        ("failure_count", "attempt_count", "error_type", "match"),
        (
            (True, 1, TypeError, "failure_count must be int"),
            (1, False, TypeError, "attempt_count must be int"),
            (0, 1, ValueError, "failure_count must be positive"),
            (-1, 1, ValueError, "failure_count must be positive"),
            (2, 1, ValueError, "failure_count must not exceed attempt_count"),
        ),
    )
    def test_unsupported_failure_error_validates_counts(
        self,
        failure_count: int,
        attempt_count: int,
        error_type: type[Exception],
        match: str,
    ) -> None:
        with pytest.raises(error_type, match=match):
            _ = UnsupportedEvaluationFailureError(
                failure_count=failure_count,
                attempt_count=attempt_count,
            )

    def test_unsupported_failure_error_is_pickleable_with_count_args(self) -> None:
        error = UnsupportedEvaluationFailureError(
            failure_count=1,
            attempt_count=3,
        )

        serialized_error = pickle.dumps(error)
        pickle.loads(serialized_error)

        assert error.args == (1, 3)
        assert str(error) == (
            "run method does not support evaluation failure assimilation "
            "(1 failures in 3 attempts)"
        )
