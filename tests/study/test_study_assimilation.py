"""Tests for study-owned attempt assimilation artifacts."""

from typing import cast

from variopt import (
    EvaluationAttemptBatch,
    EvaluationRequest,
    IntegerSpace,
    Observation,
    OptimizationDirection,
    Proposal,
    RunExecutionFailed,
)
from variopt.artifacts import EvaluationFailure, EvaluationSuccess
from variopt.study.assimilation import StudyAssimilatedStep, StudyRunHistory


def _request(candidate: int, proposal_id: str) -> EvaluationRequest[int]:
    return EvaluationRequest(
        proposal=Proposal(candidate=candidate, proposal_id=proposal_id),
    )


def _success(
    request: EvaluationRequest[int],
) -> EvaluationSuccess[int, Observation[int]]:
    observation = Observation.from_objective_value(
        request=request,
        candidate=request.candidate,
        value=float(request.candidate * request.candidate),
        direction=OptimizationDirection.MINIMIZE,
    )
    return EvaluationSuccess(request=request, payload=observation)


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


class StudyAssimilationTests:
    """Regression tests for shared study attempt-assimilation ownership."""

    def test_all_failure_step_projects_trace_without_records(self) -> None:
        request_one = _request(1, "p-1")
        request_two = _request(2, "p-2")
        attempts: EvaluationAttemptBatch[int, Observation[int]] = (
            EvaluationAttemptBatch(
                attempts=(
                    _failure(request_one, evaluation_count=0),
                    _failure(request_two, evaluation_count=2),
                ),
            )
        )
        step = StudyAssimilatedStep[int, Observation[int]].from_attempts(
            attempts,
            evaluation_count=2,
        )
        history = StudyRunHistory[int, str, Observation[int]]()

        history.append_step(step)
        report = history.to_report(
            candidate_equal=IntegerSpace(low=0, high=10).candidates_equal,
        )

        assert step.records == ()
        assert (
            step.trace_event.message == "completed 2 attempt(s): 0 succeeded, 2 failed"
        )
        assert step.trace_event.value is None
        assert report.records == ()
        assert tuple(failure.proposal_id for failure in report.failures) == (
            "p-1",
            "p-2",
        )
        assert report.evaluation_count == 2

    def test_hard_failure_projection_can_use_unappended_budget_override(self) -> None:
        request = _request(3, "p-3")
        attempts: EvaluationAttemptBatch[int, Observation[int]] = (
            EvaluationAttemptBatch(
                attempts=(_success(request),),
            )
        )
        step = StudyAssimilatedStep[int, Observation[int]].from_attempts(
            attempts,
            evaluation_count=1,
        )
        history = StudyRunHistory[int, str, Observation[int]]()
        history.append_step(step)
        safe_snapshot = history.checkpoint_snapshot("safe")

        try:
            history.raise_run_execution_failed(
                cause=RuntimeError("forced failure"),
                state="unsafe",
                safe_snapshot=safe_snapshot,
                candidate_equal=IntegerSpace(low=0, high=10).candidates_equal,
                evaluation_count=3,
            )
        except RuntimeError as raw_exception:
            assert type(raw_exception) is RunExecutionFailed
            exception = cast(
                RunExecutionFailed[int, str, Observation[int]],
                raw_exception,
            )
        else:
            raise AssertionError("expected hard run failure")

        assert exception.partial_state == "unsafe"
        assert exception.partial_report.evaluation_count == 3
        assert tuple(
            record.proposal.proposal_id for record in exception.partial_report.records
        ) == ("p-3",)
        assert exception.checkpoint_safe_state == "safe"
        assert exception.checkpoint_safe_report is not None
        assert exception.checkpoint_safe_report.evaluation_count == 1
