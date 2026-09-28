"""Record shape recognition never substitutes for request ownership validation."""

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import TypeAlias

import pytest

from tests.study_support import SpaceOwnedEqualityCandidate, SpaceOwnedEqualitySpace
from variopt import CandidateRefinement, EvaluationRequest, Observation, Proposal
from variopt.artifacts import (
    EvaluationSuccess,
    ObjectiveVectorPayload,
    ObjectiveVectorRecord,
    ObservationPayload,
)
from variopt.artifacts.records import is_request_aligned_record

BuiltinRecord: TypeAlias = Observation[int] | ObjectiveVectorRecord[int]


@dataclass(frozen=True, slots=True)
class StructuralRecord:
    request: EvaluationRequest[int]
    candidate: int


@dataclass(frozen=True, slots=True, kw_only=True)
class RequestOwningScalarPayload(ObservationPayload):
    request: EvaluationRequest[int]
    candidate: int


@dataclass(frozen=True, slots=True, kw_only=True)
class RequestOwningVectorPayload(ObjectiveVectorPayload):
    request: EvaluationRequest[int]
    candidate: int


class RecordSubclass(Observation[int]):
    pass


class IncompleteRecord:
    candidate: int = 1


class WrongRequestRecord:
    candidate: int = 1
    request: str = "not a request"


def scalar_record() -> Observation[int]:
    return Observation(
        proposal=Proposal(candidate=2), candidate=2, value=4.0, score=4.0
    )


def vector_record() -> ObjectiveVectorRecord[int]:
    return ObjectiveVectorRecord(
        proposal=Proposal(candidate=2),
        candidate=2,
        objective_values=(4.0, 2.0),
        objective_scores=(4.0, 2.0),
    )


@pytest.mark.parametrize("record_factory", [scalar_record, vector_record])
def test_builtin_record_shape_retains_actual_request_type_check(
    record_factory: Callable[[], BuiltinRecord],
) -> None:
    record = record_factory()
    assert is_request_aligned_record(record)

    object.__setattr__(record, "request", "not a request")
    assert not is_request_aligned_record(record)


@pytest.mark.parametrize("record_factory", [scalar_record, vector_record])
@pytest.mark.parametrize("missing_field", ["candidate", "request"])
def test_incomplete_builtin_record_is_not_classified_by_type_alone(
    record_factory: Callable[[], BuiltinRecord], missing_field: str
) -> None:
    record = record_factory()
    object.__delattr__(record, missing_field)

    assert not is_request_aligned_record(record)


@pytest.mark.parametrize(
    "payload",
    [
        ObservationPayload(value=4.0, score=4.0),
        ObjectiveVectorPayload(objective_values=(4.0,), objective_scores=(4.0,)),
        None,
        {"request": EvaluationRequest(proposal=Proposal(candidate=1)), "candidate": 1},
        IncompleteRecord(),
        WrongRequestRecord(),
    ],
)
def test_request_free_and_incomplete_payloads_are_not_records(payload: object) -> None:
    assert not is_request_aligned_record(payload)


@pytest.mark.parametrize(
    "factory",
    [
        StructuralRecord,
        lambda request, candidate: RequestOwningScalarPayload(
            request=request, candidate=candidate, value=4.0, score=4.0
        ),
        lambda request, candidate: RequestOwningVectorPayload(
            request=request,
            candidate=candidate,
            objective_values=(4.0,),
            objective_scores=(4.0,),
        ),
    ],
)
def test_structural_payload_and_request_free_subclass_keep_ownership_validation(
    factory: Callable[
        [EvaluationRequest[int], int],
        StructuralRecord | RequestOwningScalarPayload | RequestOwningVectorPayload,
    ],
) -> None:
    request = EvaluationRequest(proposal=Proposal(candidate=2))
    payload = factory(request, 2)
    assert is_request_aligned_record(payload)
    success = EvaluationSuccess(request=request, payload=payload)
    assert success.payload is payload

    with pytest.raises(ValueError, match="payload candidate must match"):
        EvaluationSuccess(request=request, payload=factory(request, 3))


def test_record_subclass_keeps_structural_fallback_and_alignment() -> None:
    record = RecordSubclass(
        proposal=Proposal(candidate=2), candidate=2, value=4.0, score=4.0
    )
    assert is_request_aligned_record(record)
    assert EvaluationSuccess(request=record.request, payload=record).payload is record

    with pytest.raises(ValueError, match="payload candidate must match"):
        EvaluationSuccess(
            request=EvaluationRequest(proposal=Proposal(candidate=3)), payload=record
        )


@pytest.mark.parametrize("record_factory", [scalar_record, vector_record])
def test_builtin_record_shape_does_not_certify_candidate_alignment(
    record_factory: Callable[[], BuiltinRecord],
) -> None:
    record = record_factory()
    wrong_request = EvaluationRequest(proposal=Proposal(candidate=3))

    assert is_request_aligned_record(record)
    with pytest.raises(ValueError, match="payload candidate must match"):
        EvaluationSuccess(request=wrong_request, payload=record)


@pytest.mark.parametrize("record_factory", [scalar_record, vector_record])
def test_record_payload_replacement_revalidates_candidate_alignment(
    record_factory: Callable[[], BuiltinRecord],
) -> None:
    record = record_factory()
    success = EvaluationSuccess(request=record.request, payload=record)
    mismatched = replace(record, candidate=3)

    with pytest.raises(ValueError, match="payload candidate must match"):
        success.with_payload(mismatched)


def test_builtin_refined_record_uses_space_owned_equality() -> None:
    space = SpaceOwnedEqualitySpace()
    source = SpaceOwnedEqualityCandidate(2)
    refined = SpaceOwnedEqualityCandidate(1)
    source_request = EvaluationRequest(proposal=Proposal(candidate=source))
    refined_request = EvaluationRequest(proposal=Proposal(candidate=refined))
    record = Observation(
        request=source_request,
        candidate=SpaceOwnedEqualityCandidate(1),
        value=1.0,
        score=1.0,
    )
    refinement = CandidateRefinement(
        source_candidate=source,
        refined_candidate=SpaceOwnedEqualityCandidate(1),
    )

    success = EvaluationSuccess(
        request=refined_request,
        payload=record,
        refinement=refinement,
        candidate_equal=space.candidates_equal,
    )

    assert success.payload is record
    with pytest.raises(ValueError, match="payload candidate must match"):
        success.with_payload(replace(record, candidate=SpaceOwnedEqualityCandidate(0)))
