"""Tests for evaluator-backed problem execution paths."""

from dataclasses import dataclass
from typing import TypeGuard, cast

import numpy as np
import pytest
from typing_extensions import override

from tests.problem_artifact_support import (
    LabelProtocol,
    LabelRecord,
    NaNObjective,
    ShiftedObservationProtocol,
    SquareObjective,
)
from variopt import (
    EvaluationProtocol,
    EvaluationRequest,
    IntegerSpace,
    Objective,
    OptimizationDirection,
    Problem,
    Proposal,
    SearchSpace,
)
from variopt.artifacts import (
    ObjectiveVectorPayload,
    ObjectiveVectorRecord,
    ObservationPayload,
)
from variopt.artifacts.records import RequestAlignedEvaluationRecord
from variopt.evaluation_pipeline import (
    CompatibilityEvaluationPayload,
    evaluate_request_attempt,
    evaluate_request_outcome,
    evaluate_request_payload,
    evaluate_request_success,
)
from variopt.evaluators import SequentialEvaluator


class ExplodingObjective(Objective[int]):
    """Objective that raises for one candidate."""

    @override
    def evaluate(self, candidate: int) -> float:
        if candidate == 4:
            msg = "boom"
            raise ValueError(msg)
        return float(candidate)


class InterruptingObjective(Objective[int]):
    """Objective that raises a non-recordable interruption."""

    @override
    def evaluate(self, candidate: int) -> float:
        _ = candidate
        raise KeyboardInterrupt


class ExitingObjective(Objective[int]):
    """Objective that raises a non-recordable system exit."""

    @override
    def evaluate(self, candidate: int) -> float:
        _ = candidate
        raise SystemExit(2)


class MisalignedLabelProtocol(EvaluationProtocol[int, LabelRecord]):
    """Protocol that returns a record for the wrong request."""

    @override
    def evaluate_request(self, request: EvaluationRequest[int]) -> LabelRecord:
        return LabelRecord(
            request=EvaluationRequest(proposal=Proposal(candidate=0)),
            candidate=request.candidate,
            label="misaligned",
        )


class MismatchedCandidateLabelProtocol(EvaluationProtocol[int, LabelRecord]):
    """Protocol that returns a record with the right request and wrong candidate."""

    @override
    def evaluate_request(self, request: EvaluationRequest[int]) -> LabelRecord:
        return LabelRecord(
            request=request,
            candidate=request.candidate + 1,
            label="mismatched-candidate",
        )


class AttributeBagProtocol(EvaluationProtocol[int, CompatibilityEvaluationPayload]):
    """Protocol that returns a record-shaped object without a canonical request."""

    @override
    def evaluate_request(
        self,
        request: EvaluationRequest[int],
    ) -> CompatibilityEvaluationPayload:
        candidate = request.candidate

        class AttributeBagPayload:
            request: EvaluationRequest[int]
            candidate: int

            def __init__(self) -> None:
                self.request = request
                object.__setattr__(self, "request", "not an evaluation request")
                self.candidate = candidate

        return AttributeBagPayload()


class VectorPayloadProtocol(EvaluationProtocol[int, ObjectiveVectorPayload]):
    """Protocol that returns a request-free vector payload."""

    @override
    def evaluate_request(self, request: EvaluationRequest[int]) -> ObjectiveVectorPayload:
        candidate = request.candidate
        return ObjectiveVectorPayload.from_objective_values(
            objective_values=(float(candidate), float(candidate + 1)),
            directions=(
                OptimizationDirection.MINIMIZE,
                OptimizationDirection.MAXIMIZE,
            ),
        )


@dataclass(frozen=True, slots=True)
class MarkerPayload:
    """Request-free payload that has no legacy record projection."""

    label: str


class MarkerPayloadProtocol(EvaluationProtocol[int, MarkerPayload]):
    """Protocol that returns an arbitrary request-free payload."""

    @override
    def evaluate_request(self, request: EvaluationRequest[int]) -> MarkerPayload:
        return MarkerPayload(label=f"marker:{request.candidate}")


@dataclass(frozen=True, slots=True)
class RequestOnlyPayload:
    """Payload that has only one legacy record-like attribute."""

    request: EvaluationRequest[int]


@dataclass(frozen=True, slots=True)
class CandidateOnlyPayload:
    """Payload that has only one legacy record-like attribute."""

    candidate: int


class RequestOnlyPayloadProtocol(EvaluationProtocol[int, RequestOnlyPayload]):
    """Protocol that returns a partial request-shaped payload."""

    @override
    def evaluate_request(self, request: EvaluationRequest[int]) -> RequestOnlyPayload:
        return RequestOnlyPayload(request=request)


class CandidateOnlyPayloadProtocol(EvaluationProtocol[int, CandidateOnlyPayload]):
    """Protocol that returns a partial candidate-shaped payload."""

    @override
    def evaluate_request(self, request: EvaluationRequest[int]) -> CandidateOnlyPayload:
        return CandidateOnlyPayload(candidate=request.candidate)


@dataclass(frozen=True, slots=True)
class RecordLikePayload:
    """Structural request-aligned payload used to attack legacy projection."""

    request: EvaluationRequest[int]
    candidate: int


class MisalignedRecordLikePayloadProtocol(
    EvaluationProtocol[int, RecordLikePayload]
):
    """Protocol that returns a record-like payload for the wrong request."""

    @override
    def evaluate_request(self, request: EvaluationRequest[int]) -> RecordLikePayload:
        return RecordLikePayload(
            request=EvaluationRequest(proposal=Proposal(candidate=0)),
            candidate=request.candidate,
        )


@dataclass(frozen=True, slots=True)
class EqualityHostileCandidate:
    """Candidate whose raw equality must not be used by the payload pipeline."""

    stable_id: int

    @override
    def __eq__(self, other: object) -> bool:
        _ = other
        raise AssertionError("raw equality is not the space contract")


class EqualityHostileSpace(
    SearchSpace[int | EqualityHostileCandidate, EqualityHostileCandidate]
):
    """Search space that validates candidates without relying on raw equality."""

    @override
    def normalize(
        self,
        raw_candidate: int | EqualityHostileCandidate,
    ) -> EqualityHostileCandidate:
        if isinstance(raw_candidate, EqualityHostileCandidate):
            self.validate(raw_candidate)
            return raw_candidate
        candidate = EqualityHostileCandidate(stable_id=raw_candidate)
        self.validate(candidate)
        return candidate

    @override
    def validate(self, candidate: EqualityHostileCandidate) -> None:
        if type(candidate) is not EqualityHostileCandidate:
            msg = "candidate must be an EqualityHostileCandidate"
            raise TypeError(msg)
        if candidate.stable_id < 0 or candidate.stable_id > 10:
            msg = "candidate stable_id is outside the declared bounds"
            raise ValueError(msg)

    @override
    def sample(self, random_state: np.random.RandomState) -> EqualityHostileCandidate:
        return EqualityHostileCandidate(stable_id=int(random_state.randint(0, 11)))

    @override
    def candidates_equal(
        self,
        left_candidate: EqualityHostileCandidate,
        right_candidate: EqualityHostileCandidate,
    ) -> bool:
        self.validate(left_candidate)
        self.validate(right_candidate)
        return left_candidate.stable_id == right_candidate.stable_id


class EqualityHostileObjective(Objective[EqualityHostileCandidate]):
    """Objective for candidates whose raw equality raises."""

    @override
    def evaluate(self, candidate: EqualityHostileCandidate) -> float:
        return float(candidate.stable_id)


def is_int_objective_vector_record(
    record: RequestAlignedEvaluationRecord,
) -> TypeGuard[ObjectiveVectorRecord[int]]:
    """Return whether ``record`` is an int-candidate objective-vector record."""
    return isinstance(record, ObjectiveVectorRecord)


class ProblemExecutionTests:
    """Coverage for evaluator-backed scalar and non-scalar problem execution."""

    def test_payload_pipeline_returns_request_free_scalar_payload(self) -> None:
        problem = Problem(
            space=IntegerSpace(low=0, high=10),
            objective=SquareObjective(),
        )
        request = EvaluationRequest(proposal=Proposal(candidate=4, proposal_id="p-1"))

        payload = evaluate_request_payload(problem=problem, request=request)
        success = evaluate_request_success(problem=problem, request=request)

        assert isinstance(payload, ObservationPayload)
        assert payload.value == 16.0
        assert not hasattr(payload, "request")
        assert success.request is request
        assert success.payload is not payload
        assert success.payload.value == 16.0

    def test_payload_pipeline_attempt_records_user_exception(self) -> None:
        problem = Problem(
            space=IntegerSpace(low=0, high=10),
            objective=ExplodingObjective(),
        )
        request = EvaluationRequest(proposal=Proposal(candidate=4, proposal_id="p-1"))

        attempts = evaluate_request_attempt(problem=problem, request=request)

        assert attempts.successes == ()
        assert attempts.failure_indices == (0,)
        assert attempts.failures[0].request is request
        assert attempts.failures[0].exception.exception_type == "builtins.ValueError"

    def test_payload_success_helper_keeps_user_exception_hard(self) -> None:
        problem = Problem(
            space=IntegerSpace(low=0, high=10),
            objective=ExplodingObjective(),
        )
        request = EvaluationRequest(proposal=Proposal(candidate=4, proposal_id="p-1"))

        with pytest.raises(ValueError, match="boom"):
            _ = evaluate_request_success(problem=problem, request=request)

    def test_legacy_outcome_keeps_user_exception_hard(self) -> None:
        problem = Problem(
            space=IntegerSpace(low=0, high=10),
            objective=ExplodingObjective(),
        )
        request = EvaluationRequest(proposal=Proposal(candidate=4, proposal_id="p-1"))

        with pytest.raises(ValueError, match="boom"):
            _ = evaluate_request_outcome(problem=problem, request=request)

    def test_legacy_attempt_records_user_exception(self) -> None:
        problem = Problem(
            space=IntegerSpace(low=0, high=10),
            objective=ExplodingObjective(),
        )
        request = EvaluationRequest(proposal=Proposal(candidate=4, proposal_id="p-1"))

        attempts = evaluate_request_attempt(problem=problem, request=request)

        assert attempts.successes == ()
        assert attempts.failure_indices == (0,)
        assert attempts.failures[0].request is request
        assert attempts.failures[0].exception.exception_type == "builtins.ValueError"

    def test_payload_pipeline_attempt_keeps_validation_failure_hard(self) -> None:
        problem = Problem(
            space=IntegerSpace(low=0, high=10),
            objective=ExplodingObjective(),
        )
        request = EvaluationRequest(proposal=Proposal(candidate=11, proposal_id="p-1"))

        with pytest.raises(ValueError):
            _ = evaluate_request_attempt(problem=problem, request=request)

    def test_payload_pipeline_attempt_does_not_record_base_exception(self) -> None:
        problem = Problem(
            space=IntegerSpace(low=0, high=10),
            objective=InterruptingObjective(),
        )
        request = EvaluationRequest(proposal=Proposal(candidate=4, proposal_id="p-1"))

        with pytest.raises(KeyboardInterrupt):
            _ = evaluate_request_attempt(problem=problem, request=request)

    def test_payload_pipeline_supports_vector_payload_success(self) -> None:
        problem: Problem[int, int, ObjectiveVectorPayload] = Problem(
            space=IntegerSpace(low=0, high=10),
            evaluation_protocol=VectorPayloadProtocol(),
        )
        request = EvaluationRequest(proposal=Proposal(candidate=4, proposal_id="p-1"))

        attempts = evaluate_request_attempt(problem=problem, request=request)

        assert attempts.failure_indices == ()
        assert attempts.success_indices == (0,)
        assert attempts.successes[0].request is request
        assert attempts.payloads[0].objective_values == (4.0, 5.0)
        assert attempts.payloads[0].objective_scores == (4.0, -5.0)

    def test_payload_pipeline_accepts_payload_without_legacy_projection(self) -> None:
        problem: Problem[int, int, MarkerPayload] = Problem(
            space=IntegerSpace(low=0, high=10),
            evaluation_protocol=MarkerPayloadProtocol(),
        )
        request = EvaluationRequest(proposal=Proposal(candidate=4, proposal_id="p-1"))

        payload = evaluate_request_payload(problem=problem, request=request)
        success = evaluate_request_success(problem=problem, request=request)

        assert payload == MarkerPayload(label="marker:4")
        assert success.request is request
        assert success.payload == MarkerPayload(label="marker:4")

    def test_legacy_outcome_projection_rejects_payload_without_record_shape(
        self,
    ) -> None:
        problem: Problem[int, int, MarkerPayload] = Problem(
            space=IntegerSpace(low=0, high=10),
            evaluation_protocol=MarkerPayloadProtocol(),
        )
        compatibility_problem = cast(
            Problem[int, int, CompatibilityEvaluationPayload],
            cast(object, problem),
        )
        request = EvaluationRequest(proposal=Proposal(candidate=4, proposal_id="p-1"))

        with pytest.raises(TypeError, match="could not be projected"):
            _ = evaluate_request_outcome(
                problem=compatibility_problem,
                request=request,
            )

    def test_canonical_attempt_accepts_payload_without_legacy_projection(
        self,
    ) -> None:
        problem: Problem[int, int, MarkerPayload] = Problem(
            space=IntegerSpace(low=0, high=10),
            evaluation_protocol=MarkerPayloadProtocol(),
        )
        request = EvaluationRequest(proposal=Proposal(candidate=4, proposal_id="p-1"))

        attempts = evaluate_request_attempt(problem=problem, request=request)

        assert attempts.success_indices == (0,)
        assert attempts.successes[0].request is request
        assert attempts.payloads == (MarkerPayload(label="marker:4"),)

    def test_legacy_outcome_projection_rejects_partial_record_shaped_payloads(
        self,
    ) -> None:
        request_only_problem: Problem[int, int, RequestOnlyPayload] = Problem(
            space=IntegerSpace(low=0, high=10),
            evaluation_protocol=RequestOnlyPayloadProtocol(),
        )
        candidate_only_problem: Problem[int, int, CandidateOnlyPayload] = Problem(
            space=IntegerSpace(low=0, high=10),
            evaluation_protocol=CandidateOnlyPayloadProtocol(),
        )
        request = EvaluationRequest(proposal=Proposal(candidate=4, proposal_id="p-1"))

        compatibility_problems = (
            cast(
                Problem[int, int, CompatibilityEvaluationPayload],
                cast(object, request_only_problem),
            ),
            cast(
                Problem[int, int, CompatibilityEvaluationPayload],
                cast(object, candidate_only_problem),
            ),
        )
        for compatibility_problem in compatibility_problems:
            with pytest.raises(TypeError, match="could not be projected"):
                _ = evaluate_request_outcome(
                    problem=compatibility_problem,
                    request=request,
                )

    def test_legacy_outcome_projection_rejects_misaligned_record_like_payload(
        self,
    ) -> None:
        problem: Problem[int, int, RecordLikePayload] = Problem(
            space=IntegerSpace(low=0, high=10),
            evaluation_protocol=MisalignedRecordLikePayloadProtocol(),
        )
        compatibility_problem = cast(
            Problem[int, int, CompatibilityEvaluationPayload],
            cast(object, problem),
        )
        request = EvaluationRequest(proposal=Proposal(candidate=4, proposal_id="p-1"))

        with pytest.raises(ValueError, match="outcome record request"):
            _ = evaluate_request_outcome(
                problem=compatibility_problem,
                request=request,
            )

    def test_legacy_outcome_projection_rejects_attribute_bag_payload(
        self,
    ) -> None:
        problem = Problem(
            space=IntegerSpace(low=0, high=10),
            evaluation_protocol=AttributeBagProtocol(),
        )
        request = EvaluationRequest(proposal=Proposal(candidate=4, proposal_id="p-1"))

        with pytest.raises(TypeError, match="could not be projected"):
            _ = evaluate_request_outcome(problem=problem, request=request)

    def test_legacy_outcome_projection_rejects_mismatched_record_candidate(
        self,
    ) -> None:
        problem = Problem(
            space=IntegerSpace(low=0, high=10),
            evaluation_protocol=MismatchedCandidateLabelProtocol(),
        )
        request = EvaluationRequest(proposal=Proposal(candidate=4, proposal_id="p-1"))

        with pytest.raises(ValueError, match="outcome record candidate"):
            _ = evaluate_request_outcome(problem=problem, request=request)

    def test_legacy_attempt_projection_rejects_mismatched_record_candidate(
        self,
    ) -> None:
        problem = Problem(
            space=IntegerSpace(low=0, high=10),
            evaluation_protocol=MismatchedCandidateLabelProtocol(),
        )
        request = EvaluationRequest(proposal=Proposal(candidate=4, proposal_id="p-1"))

        with pytest.raises(ValueError, match="success payload candidate"):
            _ = evaluate_request_attempt(problem=problem, request=request)

    def test_legacy_vector_payload_projection_is_request_aligned(self) -> None:
        problem: Problem[int, int, ObjectiveVectorPayload] = Problem(
            space=IntegerSpace(low=0, high=10),
            evaluation_protocol=VectorPayloadProtocol(),
        )
        request = EvaluationRequest(proposal=Proposal(candidate=4, proposal_id="p-1"))

        outcome = evaluate_request_outcome(problem=problem, request=request)

        record = outcome.record
        assert is_int_objective_vector_record(record)
        assert record.request is request
        assert record.candidate == 4
        assert record.objective_values == (4.0, 5.0)
        assert record.objective_scores == (4.0, -5.0)

    def test_payload_attempt_does_not_use_raw_candidate_equality(self) -> None:
        problem = Problem(
            space=EqualityHostileSpace(),
            objective=EqualityHostileObjective(),
        )
        request = EvaluationRequest(
            proposal=Proposal(candidate=EqualityHostileCandidate(stable_id=4)),
        )

        attempts = evaluate_request_attempt(problem=problem, request=request)

        assert attempts.success_indices == (0,)
        assert attempts.failures == ()
        assert attempts.payloads[0].value == 4.0

    def test_sequential_evaluator_rejects_non_finite_objective_values(self) -> None:
        problem = Problem(
            space=IntegerSpace(low=0, high=10),
            objective=NaNObjective(),
        )
        proposal = Proposal(candidate=4, proposal_id="p-1")

        with pytest.raises(ValueError):
            _ = SequentialEvaluator[int, int]().evaluate(
                problem,
                (EvaluationRequest(proposal=proposal),),
            )

    def test_sequential_evaluator_uses_evaluation_protocol_canonically(self) -> None:
        problem = Problem(
            space=IntegerSpace(low=0, high=10),
            evaluation_protocol=ShiftedObservationProtocol(),
        )
        proposal = Proposal(candidate=4, proposal_id="p-1")

        outcomes = SequentialEvaluator[int, int]().evaluate(
            problem,
            (EvaluationRequest(proposal=proposal),),
        )

        assert len(outcomes) == 1
        assert outcomes[0].observation.proposal == proposal
        assert outcomes[0].observation.candidate == 4
        assert outcomes[0].observation.value == 9.0

    def test_sequential_evaluator_supports_non_scalar_evaluation_records(self) -> None:
        problem = Problem(
            space=IntegerSpace(low=0, high=10),
            evaluation_protocol=LabelProtocol(),
        )
        proposal = Proposal(candidate=4, proposal_id="p-1")

        outcomes = SequentialEvaluator[int, int, LabelRecord]().evaluate(
            problem,
            (EvaluationRequest(proposal=proposal),),
        )

        assert len(outcomes) == 1
        record = outcomes[0].record
        assert isinstance(record, LabelRecord)
        assert record.proposal == proposal
        assert record.candidate == 4
        assert record.label == "parity:0"
        with pytest.raises(TypeError):
            _ = outcomes[0].observation

    def test_sequential_evaluator_attempts_record_user_exception(self) -> None:
        problem = Problem(
            space=IntegerSpace(low=0, high=10),
            objective=ExplodingObjective(),
        )
        request_one = EvaluationRequest(
            proposal=Proposal(candidate=1, proposal_id="p-1"),
        )
        request_two = EvaluationRequest(
            proposal=Proposal(candidate=4, proposal_id="p-2"),
        )
        request_three = EvaluationRequest(
            proposal=Proposal(candidate=2, proposal_id="p-3"),
        )

        attempts = SequentialEvaluator[int, int]().evaluate_attempts(
            problem,
            (request_one, request_two, request_three),
        )

        assert attempts.requests == (request_one, request_two, request_three)
        assert attempts.success_indices == (0, 2)
        assert attempts.failure_indices == (1,)
        assert tuple(
            success.scalar_observation().value for success in attempts.successes
        ) == (1.0, 2.0)
        failure = attempts.failures[0]
        assert failure.request is request_two
        assert failure.exception.exception_type == "builtins.ValueError"
        assert failure.exception.message == "boom"
        assert attempts.evaluation_count == 3

    def test_sequential_evaluator_attempts_preserve_first_and_last_failures(
        self,
    ) -> None:
        problem = Problem(
            space=IntegerSpace(low=0, high=10),
            objective=ExplodingObjective(),
        )
        request_one = EvaluationRequest(
            proposal=Proposal(candidate=4, proposal_id="p-1"),
        )
        request_two = EvaluationRequest(
            proposal=Proposal(candidate=1, proposal_id="p-2"),
        )
        request_three = EvaluationRequest(
            proposal=Proposal(candidate=4, proposal_id="p-3"),
        )

        attempts = SequentialEvaluator[int, int]().evaluate_attempts(
            problem,
            (request_one, request_two, request_three),
        )

        assert attempts.requests == (request_one, request_two, request_three)
        assert attempts.success_indices == (1,)
        assert attempts.failure_indices == (0, 2)
        assert tuple(
            success.scalar_observation().value for success in attempts.successes
        ) == (1.0,)
        assert tuple(failure.proposal_id for failure in attempts.failures) == (
            "p-1",
            "p-3",
        )
        assert attempts.evaluation_count == 3

    def test_sequential_evaluator_attempts_support_all_failure_batch(self) -> None:
        problem = Problem(
            space=IntegerSpace(low=0, high=10),
            objective=ExplodingObjective(),
        )
        request_one = EvaluationRequest(
            proposal=Proposal(candidate=4, proposal_id="p-1"),
        )
        request_two = EvaluationRequest(
            proposal=Proposal(candidate=4, proposal_id="p-2"),
        )

        attempts = SequentialEvaluator[int, int]().evaluate_attempts(
            problem,
            (request_one, request_two),
        )

        assert attempts.successes == ()
        assert attempts.success_indices == ()
        assert attempts.failure_indices == (0, 1)
        assert tuple(failure.proposal_id for failure in attempts.failures) == (
            "p-1",
            "p-2",
        )
        assert attempts.evaluation_count == 2

    def test_sequential_evaluator_attempts_support_empty_batch(self) -> None:
        problem = Problem(
            space=IntegerSpace(low=0, high=10),
            objective=ExplodingObjective(),
        )

        attempts = SequentialEvaluator[int, int]().evaluate_attempts(problem, ())

        assert attempts.requests == ()
        assert attempts.successes == ()
        assert attempts.failures == ()
        assert attempts.evaluation_count == 0

    def test_sequential_evaluator_attempts_record_non_finite_objective(self) -> None:
        problem = Problem(
            space=IntegerSpace(low=0, high=10),
            objective=NaNObjective(),
        )
        request = EvaluationRequest(
            proposal=Proposal(candidate=4, proposal_id="p-1"),
        )

        attempts = SequentialEvaluator[int, int]().evaluate_attempts(
            problem,
            (request,),
        )

        assert attempts.successes == ()
        assert attempts.failure_indices == (0,)
        assert attempts.failures[0].exception.exception_type == "builtins.ValueError"

    def test_sequential_evaluator_attempts_do_not_catch_invalid_candidate(self) -> None:
        problem = Problem(
            space=IntegerSpace(low=0, high=10),
            objective=ExplodingObjective(),
        )
        request = EvaluationRequest(
            proposal=Proposal(candidate=11, proposal_id="p-1"),
        )

        with pytest.raises(ValueError):
            _ = SequentialEvaluator[int, int]().evaluate_attempts(
                problem,
                (request,),
            )

    def test_sequential_evaluator_attempts_do_not_catch_keyboard_interrupt(self) -> None:
        problem = Problem(
            space=IntegerSpace(low=0, high=10),
            objective=InterruptingObjective(),
        )
        request = EvaluationRequest(
            proposal=Proposal(candidate=4, proposal_id="p-1"),
        )

        with pytest.raises(KeyboardInterrupt):
            _ = SequentialEvaluator[int, int]().evaluate_attempts(
                problem,
                (request,),
            )

    def test_sequential_evaluator_attempts_do_not_catch_system_exit(self) -> None:
        problem = Problem(
            space=IntegerSpace(low=0, high=10),
            objective=ExitingObjective(),
        )
        request = EvaluationRequest(
            proposal=Proposal(candidate=4, proposal_id="p-1"),
        )

        with pytest.raises(SystemExit):
            _ = SequentialEvaluator[int, int]().evaluate_attempts(
                problem,
                (request,),
            )

    def test_sequential_evaluator_attempts_do_not_record_alignment_errors(self) -> None:
        problem = Problem(
            space=IntegerSpace(low=0, high=10),
            evaluation_protocol=MisalignedLabelProtocol(),
        )
        request = EvaluationRequest(
            proposal=Proposal(candidate=4, proposal_id="p-1"),
        )

        with pytest.raises(ValueError, match="success payload request"):
            _ = SequentialEvaluator[int, int, LabelRecord]().evaluate_attempts(
                problem,
                (request,),
            )
