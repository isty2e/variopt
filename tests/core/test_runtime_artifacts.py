"""Tests for runtime artifact values and terminal surfaces."""

import pickle
from dataclasses import fields, replace
from inspect import signature
from typing import cast

import pytest
from typing_extensions import override

import variopt.artifacts.terminal as terminal_artifacts
from tests import conformance as contract_cases
from tests.problem_artifact_support import (
    LabelRecord,
)
from variopt import (
    CandidateRefinement,
    EvaluationAttemptBatch,
    EvaluationExceptionSnapshot,
    EvaluationFailure,
    EvaluationOutcome,
    EvaluationRequest,
    NondominatedRunSurface,
    ObjectiveVectorRecord,
    Observation,
    OptimizationDirection,
    Proposal,
    RunReport,
    RunResult,
)
from variopt.artifacts import EvaluationAttemptBatch as ArtifactEvaluationAttemptBatch
from variopt.artifacts import (
    EvaluationSuccess,
    ObjectiveVectorPayload,
    ObservationPayload,
    Trace,
    TraceEvent,
    materialize_attempt_batch_records,
    materialize_success_record,
    materialize_success_records,
)
from variopt.kernel import DirectKernel


class AmbiguousEqualityCandidate:
    """Candidate whose equality cannot be reduced to a scalar truth value."""

    @override
    def __eq__(self, other: object) -> bool:
        _ = other
        raise ValueError("ambiguous candidate equality")


class SpaceOwnedEqualityCandidate:
    """Candidate whose usable identity belongs to a search-space comparator."""

    def __init__(self, stable_id: int) -> None:
        self.stable_id: int = stable_id

    @override
    def __eq__(self, other: object) -> bool:
        _ = other
        raise ValueError("raw candidate equality is not the space contract")


def space_owned_candidates_equal(
    left_candidate: SpaceOwnedEqualityCandidate,
    right_candidate: SpaceOwnedEqualityCandidate,
) -> bool:
    """Return equality under the test space's stable-id semantics."""
    return left_candidate.stable_id == right_candidate.stable_id


def fail_if_candidate_equal_is_called(
    left_candidate: SpaceOwnedEqualityCandidate,
    right_candidate: SpaceOwnedEqualityCandidate,
) -> bool:
    """Raise if a no-refinement path unnecessarily compares candidates."""
    _ = left_candidate
    _ = right_candidate
    raise AssertionError("candidate equality should not be called")


def make_observation_payload(value: float = 1.0) -> ObservationPayload:
    """Return a request-free scalar payload for attempt artifact tests."""
    return ObservationPayload.from_objective_value(
        value=value,
        direction=OptimizationDirection.MINIMIZE,
    )


def make_truthy_vector_equality_candidate() -> object:
    """Return a candidate whose equality result is truthy but not scalar."""

    def equality(_self: object, _other: object) -> list[bool]:
        return [True]

    candidate_type = type("TruthyVectorEqualityCandidate", (), {"__eq__": equality})
    return candidate_type()


def test_direct_kernel_preserves_mixed_attempt_batch() -> None:
    request_one: EvaluationRequest[int] = EvaluationRequest(
        proposal=Proposal(candidate=1)
    )
    request_two: EvaluationRequest[int] = EvaluationRequest(
        proposal=Proposal(candidate=2)
    )
    success: EvaluationSuccess[int, ObservationPayload] = EvaluationSuccess(
        request=request_one,
        payload=make_observation_payload(value=1.0),
    )
    failure = EvaluationFailure(
        request=request_two,
        exception=EvaluationExceptionSnapshot.from_exception(ValueError("boom")),
    )
    attempts: EvaluationAttemptBatch[int, ObservationPayload] = EvaluationAttemptBatch(
        attempts=(success, failure),
    )
    kernel: DirectKernel[
        str,
        EvaluationAttemptBatch[int, ObservationPayload],
    ] = DirectKernel()

    def runner(query: str) -> EvaluationAttemptBatch[int, ObservationPayload]:
        assert query == "query"
        return attempts

    assert kernel.run("query", runner) is attempts


def make_int_request(candidate: int, proposal_id: str) -> EvaluationRequest[int]:
    """Return a typed integer evaluation request."""
    return EvaluationRequest(proposal=Proposal(candidate=candidate, proposal_id=proposal_id))


def make_int_failure(
    request: EvaluationRequest[int],
    message: str = "bad",
    *,
    evaluation_count: int = 1,
) -> EvaluationFailure[int]:
    """Return a typed integer evaluation failure."""
    return EvaluationFailure[int].from_exception(
        request=request,
        exception=ValueError(message),
        evaluation_count=evaluation_count,
    )


def make_int_success(
    request: EvaluationRequest[int],
    *,
    evaluation_count: int = 1,
) -> EvaluationSuccess[int, ObservationPayload]:
    """Return a typed integer scalar evaluation success."""
    return EvaluationSuccess.from_scalar_observation(
        observation=Observation.from_objective_value(
            request=request,
            candidate=request.candidate,
            value=float(request.candidate),
            direction=OptimizationDirection.MINIMIZE,
        ),
        evaluation_count=evaluation_count,
    )


class RuntimeArtifactConformanceTests(contract_cases.ArtifactConformanceCase[int]):
    """Runtime-artifact conformance for Proposal, Observation, RunResult, and Trace."""

    @override
    def make_refined_observation(self) -> Observation[int]:
        proposal = Proposal(candidate=4, proposal_id="p-1")

        return Observation(
            proposal=proposal,
            candidate=3,
            value=9.0,
            score=9.0,
            elapsed_seconds=0.1,
        )

    @override
    def make_worse_observation(self) -> Observation[int]:
        proposal = Proposal(candidate=4, proposal_id="p-1")
        return Observation(proposal=proposal, candidate=4, value=16.0, score=16.0)

    @override
    def make_better_observation(self) -> Observation[int]:
        proposal = Proposal(candidate=2, proposal_id="p-2")
        return Observation(proposal=proposal, candidate=2, value=4.0, score=4.0)

    @override
    def make_trace_event(self) -> TraceEvent:
        return TraceEvent(kind="evaluation", message="evaluated p-1", proposal_id="p-1")


class RuntimeArtifactsTests:
    """Coverage for immutable runtime-artifact value objects."""

    def test_observation_payload_is_request_free(self) -> None:
        payload = ObservationPayload.from_objective_value(
            value=4.0,
            direction=OptimizationDirection.MAXIMIZE,
            elapsed_seconds=0.2,
        )

        assert payload.value == 4.0
        assert payload.score == -4.0
        assert payload.elapsed_seconds == 0.2
        assert not hasattr(payload, "request")
        assert not hasattr(payload, "candidate")

    def test_observation_payload_pickle_preserves_request_free_payload(self) -> None:
        payload = ObservationPayload.from_objective_value(
            value=4.0,
            direction=OptimizationDirection.MAXIMIZE,
            elapsed_seconds=0.2,
        )

        restored = cast(ObservationPayload, pickle.loads(pickle.dumps(payload)))

        assert type(restored) is ObservationPayload
        assert restored == payload

    def test_observation_payload_rejects_invalid_payload_values(self) -> None:
        with pytest.raises(ValueError, match="value must be finite"):
            _ = ObservationPayload(value=float("nan"), score=1.0)

        with pytest.raises(ValueError, match="score must be finite"):
            _ = ObservationPayload(value=1.0, score=float("inf"))

        with pytest.raises(ValueError, match="elapsed_seconds must be non-negative"):
            _ = ObservationPayload(value=1.0, score=1.0, elapsed_seconds=-0.1)

        with pytest.raises(ValueError, match="elapsed_seconds must be finite"):
            _ = ObservationPayload(value=1.0, score=1.0, elapsed_seconds=float("nan"))

    def test_objective_vector_payload_is_request_free(self) -> None:
        payload = ObjectiveVectorPayload.from_objective_values(
            objective_values=(2.0, 3.0),
            directions=(
                OptimizationDirection.MINIMIZE,
                OptimizationDirection.MAXIMIZE,
            ),
            elapsed_seconds=0.3,
        )

        assert payload.objective_values == (2.0, 3.0)
        assert payload.objective_scores == (2.0, -3.0)
        assert payload.elapsed_seconds == 0.3
        assert not hasattr(payload, "request")
        assert not hasattr(payload, "candidate")

    def test_objective_vector_payload_pickle_preserves_request_free_payload(
        self,
    ) -> None:
        payload = ObjectiveVectorPayload.from_objective_values(
            objective_values=(2.0, 3.0),
            directions=(
                OptimizationDirection.MINIMIZE,
                OptimizationDirection.MAXIMIZE,
            ),
            elapsed_seconds=0.3,
        )

        restored = cast(ObjectiveVectorPayload, pickle.loads(pickle.dumps(payload)))

        assert type(restored) is ObjectiveVectorPayload
        assert restored == payload

    def test_objective_vector_payload_rejects_misaligned_vectors(self) -> None:
        with pytest.raises(
            ValueError,
            match="objective_values and objective_scores must have the same length",
        ):
            _ = ObjectiveVectorPayload(
                objective_values=(1.0, 2.0),
                objective_scores=(1.0,),
            )

        with pytest.raises(ValueError, match="directions must align"):
            _ = ObjectiveVectorPayload.from_objective_values(
                objective_values=(1.0, 2.0),
                directions=(OptimizationDirection.MINIMIZE,),
            )

        with pytest.raises(ValueError, match="elapsed_seconds must be finite"):
            _ = ObjectiveVectorPayload(
                objective_values=(1.0,),
                objective_scores=(1.0,),
                elapsed_seconds=float("inf"),
            )

        with pytest.raises(ValueError, match="objective_values must not be empty"):
            _ = ObjectiveVectorPayload(objective_values=(), objective_scores=())

    def test_evaluation_success_owns_request_and_payload(self) -> None:
        request = make_int_request(candidate=5, proposal_id="p-5")
        payload = make_observation_payload(value=25.0)
        success: EvaluationSuccess[int, ObservationPayload] = EvaluationSuccess(
            request=request,
            payload=payload,
            evaluation_count=2,
        )

        assert success.request is request
        assert success.payload is payload
        assert success.candidate == 5
        assert success.proposal is request.proposal
        assert success.proposal_id == "p-5"
        assert success.evaluation_count == 2

    def test_evaluation_success_rejects_refinement_source_candidate_drift(
        self,
    ) -> None:
        request = make_int_request(candidate=5, proposal_id="p-5")
        payload_request = make_int_request(candidate=6, proposal_id="p-5")
        payload = Observation(
            request=payload_request,
            candidate=request.candidate,
            value=25.0,
            score=25.0,
        )
        refinement = CandidateRefinement(
            source_candidate=7,
            refined_candidate=5,
            changed_leaf_paths=((),),
        )

        with pytest.raises(ValueError, match="refinement source candidate"):
            _ = EvaluationSuccess(
                request=request,
                payload=payload,
                refinement=refinement,
            )

    def test_evaluation_success_accepts_unrefined_compatibility_payload_request(
        self,
    ) -> None:
        request_candidate = SpaceOwnedEqualityCandidate(1)
        payload_request_candidate = SpaceOwnedEqualityCandidate(1)
        request: EvaluationRequest[SpaceOwnedEqualityCandidate] = EvaluationRequest(
            proposal=Proposal(candidate=request_candidate, proposal_id="p-1"),
        )
        payload_request: EvaluationRequest[SpaceOwnedEqualityCandidate] = (
            EvaluationRequest(
                proposal=Proposal(
                    candidate=payload_request_candidate,
                    proposal_id="p-1",
                ),
            )
        )
        payload: Observation[SpaceOwnedEqualityCandidate] = Observation(
            request=payload_request,
            candidate=request.candidate,
            value=1.0,
            score=1.0,
        )

        success = EvaluationSuccess(
            request=request,
            payload=payload,
            candidate_equal=fail_if_candidate_equal_is_called,
        )

        assert success.payload is payload

    def test_evaluation_success_rejects_negative_evaluation_count(self) -> None:
        request = make_int_request(candidate=5, proposal_id="p-5")

        with pytest.raises(ValueError, match="evaluation_count must be non-negative"):
            _ = EvaluationSuccess(
                request=request,
                payload=make_observation_payload(),
                evaluation_count=-1,
            )

    def test_evaluation_success_without_refinement_never_calls_candidate_equal(
        self,
    ) -> None:
        request: EvaluationRequest[SpaceOwnedEqualityCandidate] = EvaluationRequest(
            proposal=Proposal(candidate=SpaceOwnedEqualityCandidate(1))
        )
        success: EvaluationSuccess[
            SpaceOwnedEqualityCandidate,
            ObservationPayload,
        ] = EvaluationSuccess(
            request=request,
            payload=make_observation_payload(),
            candidate_equal=fail_if_candidate_equal_is_called,
        )

        replaced = replace(success, evaluation_count=2)

        assert replaced.evaluation_count == 2

    def test_evaluation_success_refinement_uses_request_candidate(self) -> None:
        source_candidate = SpaceOwnedEqualityCandidate(1)
        request_candidate = SpaceOwnedEqualityCandidate(2)
        refined_candidate = SpaceOwnedEqualityCandidate(2)
        request: EvaluationRequest[SpaceOwnedEqualityCandidate] = EvaluationRequest(
            proposal=Proposal(candidate=request_candidate)
        )
        refinement = CandidateRefinement(
            source_candidate=source_candidate,
            refined_candidate=refined_candidate,
        )

        success: EvaluationSuccess[
            SpaceOwnedEqualityCandidate,
            ObservationPayload,
        ] = EvaluationSuccess(
            request=request,
            payload=make_observation_payload(),
            refinement=refinement,
            candidate_equal=space_owned_candidates_equal,
        )

        assert success.request is request
        assert success.refinement is refinement

    def test_evaluation_success_rejects_raw_equality_refinement_fallback(self) -> None:
        source_candidate = SpaceOwnedEqualityCandidate(1)
        request_candidate = SpaceOwnedEqualityCandidate(2)
        refined_candidate = SpaceOwnedEqualityCandidate(2)
        request: EvaluationRequest[SpaceOwnedEqualityCandidate] = EvaluationRequest(
            proposal=Proposal(candidate=request_candidate)
        )
        refinement = CandidateRefinement(
            source_candidate=source_candidate,
            refined_candidate=refined_candidate,
        )

        with pytest.raises(
            TypeError,
            match="candidate equality must produce a scalar truth value",
        ):
            _ = EvaluationSuccess(
                request=request,
                payload=make_observation_payload(),
                refinement=refinement,
            )

    def test_evaluation_success_rejects_explicit_equality_refinement_mismatch(
        self,
    ) -> None:
        source_candidate = SpaceOwnedEqualityCandidate(1)
        request_candidate = SpaceOwnedEqualityCandidate(2)
        refined_candidate = SpaceOwnedEqualityCandidate(3)
        request: EvaluationRequest[SpaceOwnedEqualityCandidate] = EvaluationRequest(
            proposal=Proposal(candidate=request_candidate)
        )

        with pytest.raises(
            ValueError,
            match="refinement refined_candidate must match",
        ):
            _ = EvaluationSuccess(
                request=request,
                payload=make_observation_payload(),
                refinement=CandidateRefinement(
                    source_candidate=source_candidate,
                    refined_candidate=refined_candidate,
                ),
                candidate_equal=space_owned_candidates_equal,
            )

    def test_evaluation_success_replace_reuses_explicit_candidate_equality(self) -> None:
        source_candidate = SpaceOwnedEqualityCandidate(1)
        request_candidate = SpaceOwnedEqualityCandidate(2)
        request: EvaluationRequest[SpaceOwnedEqualityCandidate] = EvaluationRequest(
            proposal=Proposal(candidate=request_candidate)
        )
        success: EvaluationSuccess[
            SpaceOwnedEqualityCandidate,
            ObservationPayload,
        ] = EvaluationSuccess(
            request=request,
            payload=make_observation_payload(),
            refinement=CandidateRefinement(
                source_candidate=source_candidate,
                refined_candidate=SpaceOwnedEqualityCandidate(2),
            ),
            candidate_equal=space_owned_candidates_equal,
        )

        replaced = replace(
            success,
            refinement=CandidateRefinement(
                source_candidate=source_candidate,
                refined_candidate=SpaceOwnedEqualityCandidate(2),
            ),
        )

        assert replaced.refinement is not success.refinement
        assert replaced.request is request

    def test_evaluation_success_replace_revalidates_changed_request(self) -> None:
        source_candidate = SpaceOwnedEqualityCandidate(1)
        request: EvaluationRequest[SpaceOwnedEqualityCandidate] = EvaluationRequest(
            proposal=Proposal(candidate=SpaceOwnedEqualityCandidate(2))
        )
        success: EvaluationSuccess[
            SpaceOwnedEqualityCandidate,
            ObservationPayload,
        ] = EvaluationSuccess(
            request=request,
            payload=make_observation_payload(),
            refinement=CandidateRefinement(
                source_candidate=source_candidate,
                refined_candidate=SpaceOwnedEqualityCandidate(2),
            ),
            candidate_equal=space_owned_candidates_equal,
        )
        mismatched_request: EvaluationRequest[SpaceOwnedEqualityCandidate] = (
            EvaluationRequest(proposal=Proposal(candidate=SpaceOwnedEqualityCandidate(3)))
        )
        matching_request: EvaluationRequest[SpaceOwnedEqualityCandidate] = (
            EvaluationRequest(proposal=Proposal(candidate=SpaceOwnedEqualityCandidate(2)))
        )

        with pytest.raises(ValueError, match="refinement refined_candidate must match"):
            _ = replace(success, request=mismatched_request)

        replaced = replace(success, request=matching_request)
        assert replaced.request is matching_request

    def test_evaluation_success_pickle_strips_candidate_equal_safely(self) -> None:
        source_candidate = SpaceOwnedEqualityCandidate(1)
        request_candidate = SpaceOwnedEqualityCandidate(2)
        request: EvaluationRequest[SpaceOwnedEqualityCandidate] = EvaluationRequest(
            proposal=Proposal(candidate=request_candidate)
        )
        success: EvaluationSuccess[
            SpaceOwnedEqualityCandidate,
            ObservationPayload,
        ] = EvaluationSuccess(
            request=request,
            payload=make_observation_payload(),
            refinement=CandidateRefinement(
                source_candidate=source_candidate,
                refined_candidate=SpaceOwnedEqualityCandidate(2),
            ),
            candidate_equal=space_owned_candidates_equal,
        )

        restored = cast(
            EvaluationSuccess[SpaceOwnedEqualityCandidate, ObservationPayload],
            pickle.loads(pickle.dumps(success)),
        )
        assert type(restored) is EvaluationSuccess
        recount = replace(restored, evaluation_count=3)

        assert recount.evaluation_count == 3
        assert recount.refinement is restored.refinement

        with pytest.raises(TypeError, match="candidate_equal is required"):
            _ = replace(
                restored,
                refinement=CandidateRefinement(
                    source_candidate=source_candidate,
                    refined_candidate=SpaceOwnedEqualityCandidate(2),
                ),
            )

        repaired = replace(
            restored,
            refinement=CandidateRefinement(
                source_candidate=source_candidate,
                refined_candidate=SpaceOwnedEqualityCandidate(2),
            ),
            candidate_equal=space_owned_candidates_equal,
        )
        assert repaired.refinement is not restored.refinement

    def test_evaluation_success_pickle_strips_unpicklable_candidate_equal(self) -> None:
        def local_candidates_equal(
            left_candidate: SpaceOwnedEqualityCandidate,
            right_candidate: SpaceOwnedEqualityCandidate,
        ) -> bool:
            return left_candidate.stable_id == right_candidate.stable_id

        source_candidate = SpaceOwnedEqualityCandidate(1)
        request_candidate = SpaceOwnedEqualityCandidate(2)
        request: EvaluationRequest[SpaceOwnedEqualityCandidate] = EvaluationRequest(
            proposal=Proposal(candidate=request_candidate)
        )
        success: EvaluationSuccess[
            SpaceOwnedEqualityCandidate,
            ObservationPayload,
        ] = EvaluationSuccess(
            request=request,
            payload=make_observation_payload(),
            refinement=CandidateRefinement(
                source_candidate=source_candidate,
                refined_candidate=SpaceOwnedEqualityCandidate(2),
            ),
            candidate_equal=local_candidates_equal,
        )

        restored = cast(
            EvaluationSuccess[SpaceOwnedEqualityCandidate, ObservationPayload],
            pickle.loads(pickle.dumps(success)),
        )

        assert restored.request.candidate.stable_id == 2
        assert type(restored) is EvaluationSuccess

    def test_artifact_attempt_batch_projects_mixed_attempt_slots(self) -> None:
        request_one = make_int_request(candidate=1, proposal_id="p-1")
        request_two = make_int_request(candidate=2, proposal_id="p-2")
        request_three = make_int_request(candidate=3, proposal_id="p-3")
        payload_one = make_observation_payload(value=1.0)
        payload_three = make_observation_payload(value=3.0)
        success_one: EvaluationSuccess[int, ObservationPayload] = EvaluationSuccess(
            request=request_one,
            payload=payload_one,
            evaluation_count=2,
        )
        failure = make_int_failure(request_two, evaluation_count=4)
        success_three: EvaluationSuccess[int, ObservationPayload] = EvaluationSuccess(
            request=request_three,
            payload=payload_three,
            evaluation_count=8,
        )
        batch: ArtifactEvaluationAttemptBatch[int, ObservationPayload] = (
            ArtifactEvaluationAttemptBatch(
                attempts=(success_one, failure, success_three),
            )
        )

        assert batch.attempt_count == 3
        assert batch.requests == (request_one, request_two, request_three)
        assert batch.success_indices == (0, 2)
        assert batch.failure_indices == (1,)
        assert batch.successes == (success_one, success_three)
        assert batch.failures == (failure,)
        assert batch.payloads == (payload_one, payload_three)
        assert batch.evaluation_count == 14
        assert batch.has_failures is True

    def test_artifact_attempt_batch_supports_all_failure_slots(self) -> None:
        request_one = make_int_request(candidate=1, proposal_id="p-1")
        request_two = make_int_request(candidate=2, proposal_id="p-2")
        failure_one = make_int_failure(request_one, evaluation_count=2)
        failure_two = make_int_failure(request_two, evaluation_count=3)
        batch: ArtifactEvaluationAttemptBatch[int, ObservationPayload] = (
            ArtifactEvaluationAttemptBatch(attempts=(failure_one, failure_two))
        )

        assert batch.attempt_count == 2
        assert batch.requests == (request_one, request_two)
        assert batch.success_indices == ()
        assert batch.failure_indices == (0, 1)
        assert batch.successes == ()
        assert batch.failures == (failure_one, failure_two)
        assert batch.payloads == ()
        assert batch.evaluation_count == 5
        assert batch.has_failures is True

    def test_artifact_attempt_batch_supports_empty_slots(self) -> None:
        batch: ArtifactEvaluationAttemptBatch[int, ObservationPayload] = (
            ArtifactEvaluationAttemptBatch(attempts=())
        )

        assert batch.attempt_count == 0
        assert batch.requests == ()
        assert batch.success_indices == ()
        assert batch.failure_indices == ()
        assert batch.successes == ()
        assert batch.failures == ()
        assert batch.payloads == ()
        assert batch.evaluation_count == 0
        assert batch.has_failures is False

    def test_artifact_attempt_batch_single_success_rejects_empty_batch(self) -> None:
        batch: ArtifactEvaluationAttemptBatch[int, ObservationPayload] = (
            ArtifactEvaluationAttemptBatch(attempts=())
        )

        with pytest.raises(ValueError, match="exactly one request"):
            _ = batch.single_success_or_none()

    def test_artifact_attempt_batch_pickle_preserves_ordered_slots(self) -> None:
        request_one = make_int_request(candidate=1, proposal_id="p-1")
        request_two = make_int_request(candidate=2, proposal_id="p-2")
        success: EvaluationSuccess[int, ObservationPayload] = EvaluationSuccess(
            request=request_one,
            payload=make_observation_payload(),
        )
        failure = make_int_failure(request_two)
        batch: ArtifactEvaluationAttemptBatch[int, ObservationPayload] = (
            ArtifactEvaluationAttemptBatch(attempts=(success, failure))
        )

        restored = cast(
            ArtifactEvaluationAttemptBatch[int, ObservationPayload],
            pickle.loads(pickle.dumps(batch)),
        )

        assert restored.attempts == batch.attempts
        assert restored.success_indices == (0,)
        assert restored.failure_indices == (1,)

    def test_artifact_attempt_batch_supports_vector_payload_successes(self) -> None:
        request = make_int_request(candidate=1, proposal_id="p-1")
        payload = ObjectiveVectorPayload.from_objective_values(
            objective_values=(1.0, 2.0),
            directions=(
                OptimizationDirection.MINIMIZE,
                OptimizationDirection.MAXIMIZE,
            ),
        )
        success: EvaluationSuccess[int, ObjectiveVectorPayload] = EvaluationSuccess(
            request=request,
            payload=payload,
        )
        batch: ArtifactEvaluationAttemptBatch[int, ObjectiveVectorPayload] = (
            ArtifactEvaluationAttemptBatch(attempts=(success,))
        )

        assert batch.payloads == (payload,)
        assert batch.single_success_or_none() is success

    def test_artifact_attempt_batch_single_success_view(self) -> None:
        request = make_int_request(candidate=1, proposal_id="p-1")
        success: EvaluationSuccess[int, ObservationPayload] = EvaluationSuccess(
            request=request,
            payload=make_observation_payload(),
        )
        success_batch: ArtifactEvaluationAttemptBatch[int, ObservationPayload] = (
            ArtifactEvaluationAttemptBatch(attempts=(success,))
        )
        failure_batch: ArtifactEvaluationAttemptBatch[int, ObservationPayload] = (
            ArtifactEvaluationAttemptBatch(
                attempts=(make_int_failure(request),),
            )
        )
        mixed_batch: ArtifactEvaluationAttemptBatch[int, ObservationPayload] = (
            ArtifactEvaluationAttemptBatch(
                attempts=(success, make_int_failure(request)),
            )
        )

        assert success_batch.single_success_or_none() is success
        assert failure_batch.single_success_or_none() is None
        with pytest.raises(ValueError, match="exactly one request"):
            _ = mixed_batch.single_success_or_none()

    def test_observation_is_scalar_evaluation_record(self) -> None:
        proposal = Proposal(candidate=4, proposal_id="p-1")
        observation = Observation(
            proposal=proposal,
            candidate=4,
            value=16.0,
            score=16.0,
        )

        assert observation.request.proposal is proposal
        assert observation.candidate == 4

    def test_objective_vector_record_is_vector_evaluation_record(self) -> None:
        proposal = Proposal(candidate=4, proposal_id="p-1")
        record = ObjectiveVectorRecord.from_objective_values(
            proposal=proposal,
            candidate=4,
            objective_values=(16.0, 3.0),
            directions=(
                OptimizationDirection.MINIMIZE,
                OptimizationDirection.MAXIMIZE,
            ),
        )

        assert record.request.proposal is proposal
        assert record.candidate == 4
        assert record.objective_values == (16.0, 3.0)
        assert record.objective_scores == (16.0, -3.0)

    def test_observation_separates_proposal_and_evaluated_candidate(self) -> None:
        proposal = Proposal(candidate=4, proposal_id="p-1")
        observation = Observation(
            proposal=proposal,
            candidate=3,
            value=9.0,
            score=9.0,
            elapsed_seconds=0.1,
        )

        assert observation.proposal.candidate == 4
        assert observation.candidate == 3
        assert observation.value == 9.0

    def test_candidate_refinement_normalizes_changed_leaf_paths(self) -> None:
        refinement = CandidateRefinement(
            source_candidate={"x": 1, "y": 2},
            refined_candidate={"x": 3, "y": 2},
            changed_leaf_paths=[("x",), ("nested", 0)],
        )

        assert refinement.changed_leaf_paths == (("x",), ("nested", 0))

    def test_candidate_refinement_rejects_duplicate_changed_leaf_paths(self) -> None:
        with pytest.raises(ValueError):
            _ = CandidateRefinement(
                source_candidate=(1, 2),
                refined_candidate=(3, 2),
                changed_leaf_paths=((0,), (0,)),
            )

    def test_candidate_refinement_rejects_bool_path_segments(self) -> None:
        with pytest.raises(TypeError, match="int or str"):
            _ = CandidateRefinement(
                source_candidate=(1, 2),
                refined_candidate=(3, 2),
                changed_leaf_paths=((True,),),
            )

    def test_evaluation_outcome_defaults_to_no_refinement_payload(self) -> None:
        observation = Observation(
            proposal=Proposal(candidate=4, proposal_id="p-1"),
            candidate=4,
            value=16.0,
            score=16.0,
        )

        outcome = EvaluationOutcome(observation=observation)

        assert outcome.record == observation
        assert outcome.refinement is None

    def test_evaluation_failure_snapshots_exception_without_raw_exception(self) -> None:
        request = make_int_request(4, "p-1")

        failure = EvaluationFailure[int].from_exception(
            request=request,
            exception=ValueError("bad candidate"),
            evaluation_count=2,
        )

        assert failure.request is request
        assert failure.candidate == 4
        assert failure.proposal_id == "p-1"
        assert failure.evaluation_count == 2
        assert failure.exception.exception_module == "builtins"
        assert failure.exception.exception_qualname == "ValueError"
        assert failure.exception.exception_type == "builtins.ValueError"
        assert failure.exception.message == "bad candidate"
        assert not isinstance(failure.exception, ValueError)

    def test_evaluation_failure_pickle_round_trip_preserves_snapshot(self) -> None:
        request = make_int_request(4, "p-1")
        failure = EvaluationFailure[int].from_exception(
            request=request,
            exception=RuntimeError("boom"),
        )

        restored_failure = cast(
            EvaluationFailure[int],
            pickle.loads(pickle.dumps(failure)),
        )

        assert restored_failure == failure
        assert restored_failure.exception.exception_type == "builtins.RuntimeError"

    def test_evaluation_failure_rejects_negative_evaluation_count(self) -> None:
        request = make_int_request(4, "p-1")
        snapshot = EvaluationExceptionSnapshot.from_exception(ValueError("bad"))

        with pytest.raises(ValueError, match="non-negative"):
            _ = EvaluationFailure(
                request=request,
                exception=snapshot,
                evaluation_count=-1,
            )

    def test_evaluation_attempt_batch_projects_requests_by_attempt_slot(self) -> None:
        request_one = make_int_request(1, "p-1")
        request_two = make_int_request(2, "p-2")
        request_three = make_int_request(3, "p-3")
        success = make_int_success(request_two)
        failure_one = make_int_failure(request_one)
        failure_three = make_int_failure(request_three)
        attempts: EvaluationAttemptBatch[int, ObservationPayload] = EvaluationAttemptBatch(
            attempts=(failure_one, success, failure_three),
        )

        assert tuple(success.request for success in attempts.successes) == (request_two,)
        assert tuple(failure.request for failure in attempts.failures) == (
            request_one,
            request_three,
        )

    def test_exception_snapshot_rejects_empty_exception_type(self) -> None:
        with pytest.raises(ValueError, match="exception_module"):
            _ = EvaluationExceptionSnapshot(
                exception_module="",
                exception_qualname="ValueError",
                message="bad",
            )

        with pytest.raises(ValueError, match="exception_qualname"):
            _ = EvaluationExceptionSnapshot(
                exception_module="builtins",
                exception_qualname="",
                message="bad",
            )

    def test_exception_snapshot_rejects_non_exception_base_exception(self) -> None:
        with pytest.raises(TypeError, match="Exception instance"):
            _ = EvaluationExceptionSnapshot.from_exception(KeyboardInterrupt())

    def test_exception_snapshot_preserves_local_exception_qualname(self) -> None:
        class LocalEvaluationError(Exception):
            """Test-local recordable exception."""

        snapshot = EvaluationExceptionSnapshot.from_exception(
            LocalEvaluationError("local failure"),
        )

        assert snapshot.exception_module == __name__
        assert snapshot.exception_qualname.endswith("LocalEvaluationError")
        assert snapshot.message == "local failure"

    def test_exception_snapshot_allows_empty_exception_message(self) -> None:
        snapshot = EvaluationExceptionSnapshot.from_exception(ValueError())

        assert snapshot.exception_type == "builtins.ValueError"
        assert snapshot.message == ""

    def test_evaluation_attempt_batch_accepts_empty_batch(self) -> None:
        batch: EvaluationAttemptBatch[int, ObservationPayload] = EvaluationAttemptBatch(
            attempts=(),
        )

        assert batch.attempt_count == 0
        assert batch.payloads == ()
        assert batch.evaluation_count == 0
        assert not batch.has_failures

    def test_evaluation_attempt_batch_accepts_all_successes(self) -> None:
        request_one = make_int_request(1, "p-1")
        request_two = make_int_request(2, "p-2")
        success_one = make_int_success(request_one)
        success_two = make_int_success(request_two, evaluation_count=2)

        batch: EvaluationAttemptBatch[int, ObservationPayload] = EvaluationAttemptBatch(
            attempts=(success_one, success_two),
        )

        assert batch.success_indices == (0, 1)
        assert batch.failure_indices == ()
        assert batch.successes == (success_one, success_two)
        assert batch.payloads == (success_one.payload, success_two.payload)
        assert batch.evaluation_count == 3
        assert not batch.has_failures

    def test_evaluation_attempt_batch_accepts_mixed_success_and_failure(self) -> None:
        request_one = make_int_request(1, "p-1")
        request_two = make_int_request(2, "p-2")
        success = make_int_success(request_one)
        failure = make_int_failure(request_two)

        batch: EvaluationAttemptBatch[int, ObservationPayload] = EvaluationAttemptBatch(
            attempts=(success, failure),
        )

        assert batch.attempt_count == 2
        assert batch.successes == (success,)
        assert batch.failures == (failure,)
        assert batch.evaluation_count == 2
        assert batch.has_failures

    def test_evaluation_attempt_batch_stores_authoritative_attempt_slots(self) -> None:
        request_one = make_int_request(1, "p-1")
        request_two = make_int_request(2, "p-2")
        success = make_int_success(request_one)
        failure = make_int_failure(request_two)

        batch: EvaluationAttemptBatch[int, ObservationPayload] = EvaluationAttemptBatch(
            attempts=(success, failure),
        )

        field_names = tuple(field.name for field in fields(batch))
        assert field_names == ("__orig_class__", "attempts")
        assert batch.attempts == (success, failure)
        assert batch.requests == (request_one, request_two)
        assert batch.successes == (success,)
        assert batch.failures == (failure,)
        assert batch.success_indices == (0,)
        assert batch.failure_indices == (1,)

    def test_evaluation_attempt_batch_accepts_canonical_attempt_slots(self) -> None:
        request_one = make_int_request(1, "p-1")
        request_two = make_int_request(2, "p-2")
        success = make_int_success(request_one)
        failure = make_int_failure(request_two)

        batch: EvaluationAttemptBatch[int, ObservationPayload] = EvaluationAttemptBatch(
            attempts=(success, failure),
        )
        replaced = replace(batch, attempts=(failure, success))

        assert batch.requests == (request_one, request_two)
        assert batch.success_indices == (0,)
        assert batch.failure_indices == (1,)
        assert replaced.requests == (request_two, request_one)
        assert replaced.success_indices == (1,)
        assert replaced.failure_indices == (0,)

    def test_evaluation_attempt_batch_rejects_non_attempt_slot(self) -> None:
        payload = make_observation_payload()

        with pytest.raises(
            TypeError,
            match="attempts must contain EvaluationSuccess or EvaluationFailure",
        ):
            _ = EvaluationAttemptBatch[int, ObservationPayload](
                attempts=(payload,),
            )

    def test_evaluation_attempt_batch_accepts_all_failures(self) -> None:
        request_one = make_int_request(1, "p-1")
        request_two = make_int_request(2, "p-2")
        failure_one = make_int_failure(request_one, "bad one")
        failure_two = make_int_failure(
            request_two,
            "bad two",
            evaluation_count=2,
        )

        batch: EvaluationAttemptBatch[int, ObservationPayload] = EvaluationAttemptBatch(
            attempts=(failure_one, failure_two),
        )

        assert batch.success_indices == ()
        assert batch.failure_indices == (0, 1)
        assert batch.payloads == ()
        assert batch.evaluation_count == 3

    def test_evaluation_attempt_batch_counts_zero_cost_failure(self) -> None:
        request = make_int_request(1, "p-1")
        failure = make_int_failure(request, evaluation_count=0)

        batch: EvaluationAttemptBatch[int, ObservationPayload] = EvaluationAttemptBatch(
            attempts=(failure,),
        )

        assert batch.evaluation_count == 0

    def test_evaluation_attempt_batch_merges_single_request_attempts(self) -> None:
        request_one = make_int_request(1, "p-1")
        request_two = make_int_request(2, "p-2")
        request_three = make_int_request(3, "p-3")
        success_one = make_int_success(request_one)
        success_three = make_int_success(request_three, evaluation_count=2)
        failure_two = make_int_failure(request_two, evaluation_count=3)

        batch = EvaluationAttemptBatch[
            int,
            ObservationPayload,
        ].from_single_request_attempts(
            (
                EvaluationAttemptBatch(
                    attempts=(success_one,),
                ),
                EvaluationAttemptBatch(
                    attempts=(failure_two,),
                ),
                EvaluationAttemptBatch(
                    attempts=(success_three,),
                ),
            )
        )

        assert batch.requests == (request_one, request_two, request_three)
        assert batch.success_indices == (0, 2)
        assert batch.failure_indices == (1,)
        assert batch.payloads == (success_one.payload, success_three.payload)
        assert batch.failures == (failure_two,)
        assert batch.evaluation_count == 6

    def test_evaluation_attempt_batch_merges_empty_single_attempt_sequence(
        self,
    ) -> None:
        batch = EvaluationAttemptBatch[
            int,
            ObservationPayload,
        ].from_single_request_attempts(())

        assert batch.requests == ()
        assert batch.successes == ()
        assert batch.failures == ()

    def test_materialize_success_record_projects_scalar_payload(self) -> None:
        request = make_int_request(candidate=2, proposal_id="p-2")
        refinement = CandidateRefinement(
            source_candidate=4,
            refined_candidate=2,
            changed_leaf_paths=((),),
        )
        success: EvaluationSuccess[int, ObservationPayload] = EvaluationSuccess(
            request=request,
            payload=make_observation_payload(value=4.0),
            refinement=refinement,
        )

        record = materialize_success_record(success)

        assert type(record) is Observation
        assert record.candidate == 2
        assert record.proposal.candidate == 4
        assert record.proposal.proposal_id == "p-2"
        assert record.value == 4.0

    def test_materialize_success_records_preserves_existing_record_payload(self) -> None:
        request = make_int_request(candidate=7, proposal_id="p-7")
        record = LabelRecord(
            request=request,
            candidate=request.candidate,
            label="seven",
        )
        success: EvaluationSuccess[int, LabelRecord] = EvaluationSuccess(
            request=request,
            payload=record,
        )

        assert materialize_success_records((success,)) == (record,)

    def test_materialize_success_record_rejects_attribute_bag_payload(self) -> None:
        request = make_int_request(candidate=7, proposal_id="p-7")

        class AttributeBagPayload:
            def __init__(self, candidate: int) -> None:
                self.request = "not an evaluation request"
                self.candidate = candidate

        success: EvaluationSuccess[int, object] = EvaluationSuccess(
            request=request,
            payload=AttributeBagPayload(request.candidate),
        )

        with pytest.raises(TypeError, match="cannot be materialized"):
            _ = materialize_success_record(success)

    def test_materialize_attempt_batch_records_preserves_failure_slots(self) -> None:
        request_one = make_int_request(candidate=1, proposal_id="p-1")
        request_two = make_int_request(candidate=2, proposal_id="p-2")
        success: EvaluationSuccess[int, ObservationPayload] = EvaluationSuccess(
            request=request_one,
            payload=make_observation_payload(value=1.0),
        )
        failure = make_int_failure(request_two, evaluation_count=2)
        batch: EvaluationAttemptBatch[int, ObservationPayload] = EvaluationAttemptBatch(
            attempts=(success, failure),
        )

        materialized = materialize_attempt_batch_records(batch)

        assert materialized.requests == (request_one, request_two)
        assert materialized.success_indices == (0,)
        assert materialized.failure_indices == (1,)
        assert materialized.failures == (failure,)
        assert type(materialized.successes[0].payload) is Observation

    def test_evaluation_attempt_batch_merge_rejects_multi_request_attempt(
        self,
    ) -> None:
        request_one = make_int_request(1, "p-1")
        request_two = make_int_request(2, "p-2")
        success_one = make_int_success(request_one)
        success_two = make_int_success(request_two)
        attempt: EvaluationAttemptBatch[int, ObservationPayload] = (
            EvaluationAttemptBatch(
                attempts=(success_one, success_two),
            )
        )

        with pytest.raises(ValueError, match="exactly one request"):
            _ = EvaluationAttemptBatch[
                int,
                ObservationPayload,
            ].from_single_request_attempts((attempt,))

    def test_evaluation_attempt_batch_merge_rejects_empty_attempt_element(
        self,
    ) -> None:
        attempt: EvaluationAttemptBatch[int, ObservationPayload] = (
            EvaluationAttemptBatch(attempts=())
        )

        with pytest.raises(ValueError, match="exactly one request"):
            _ = EvaluationAttemptBatch[
                int,
                ObservationPayload,
            ].from_single_request_attempts((attempt,))

    def test_evaluation_attempt_batch_concatenates_batches(self) -> None:
        request_one = make_int_request(1, "p-1")
        request_two = make_int_request(2, "p-2")
        request_three = make_int_request(3, "p-3")
        success_one = make_int_success(request_one)
        success_three = make_int_success(request_three)
        failure_two = make_int_failure(request_two)
        first_batch: EvaluationAttemptBatch[int, ObservationPayload] = (
            EvaluationAttemptBatch(
                attempts=(success_one, failure_two),
            )
        )
        second_batch: EvaluationAttemptBatch[int, ObservationPayload] = (
            EvaluationAttemptBatch(
                attempts=(success_three,),
            )
        )

        batch = EvaluationAttemptBatch[
            int,
            ObservationPayload,
        ].concatenate((first_batch, second_batch))

        assert batch.requests == (request_one, request_two, request_three)
        assert batch.success_indices == (0, 2)
        assert batch.failure_indices == (1,)
        assert batch.successes == (success_one, success_three)
        assert batch.failures == (failure_two,)

    def test_evaluation_attempt_batch_single_success_view_returns_success(
        self,
    ) -> None:
        request = make_int_request(1, "p-1")
        success = make_int_success(request)
        attempt: EvaluationAttemptBatch[int, ObservationPayload] = (
            EvaluationAttemptBatch(
                attempts=(success,),
            )
        )

        assert attempt.single_success_or_none() is success

    def test_evaluation_attempt_batch_single_success_view_returns_none_for_failure(
        self,
    ) -> None:
        request = make_int_request(1, "p-1")
        failure = make_int_failure(request)
        attempt: EvaluationAttemptBatch[int, ObservationPayload] = EvaluationAttemptBatch(
            attempts=(failure,),
        )

        assert attempt.single_success_or_none() is None

    def test_evaluation_attempt_batch_single_success_view_rejects_multi_slot_batch(
        self,
    ) -> None:
        request_one = make_int_request(1, "p-1")
        request_two = make_int_request(2, "p-2")
        failure_one = make_int_failure(request_one)
        failure_two = make_int_failure(request_two)
        attempt: EvaluationAttemptBatch[int, ObservationPayload] = EvaluationAttemptBatch(
            attempts=(failure_one, failure_two),
        )

        with pytest.raises(ValueError, match="exactly one request"):
            result = attempt.single_success_or_none()
            assert result is None

    def test_evaluation_outcome_preserves_scalar_refinement_payload(self) -> None:
        observation = Observation(
            proposal=Proposal(candidate=4, proposal_id="p-1"),
            candidate=3,
            value=9.0,
            score=9.0,
        )
        refinement = CandidateRefinement(
            source_candidate=4,
            refined_candidate=3,
            changed_leaf_paths=((),),
        )

        outcome = EvaluationOutcome(
            observation=observation,
            evaluation_count=2,
            refinement=refinement,
        )

        assert outcome.refinement == refinement
        assert outcome.evaluation_count == 2

    def test_evaluation_outcome_preserves_non_scalar_refinement_payload(self) -> None:
        request = EvaluationRequest(proposal=Proposal(candidate=4, proposal_id="p-1"))
        record = LabelRecord(
            request=request,
            candidate=3,
            label="parity:1",
        )
        refinement = CandidateRefinement(
            source_candidate=4,
            refined_candidate=3,
            changed_leaf_paths=((),),
        )

        outcome: EvaluationOutcome[int, LabelRecord] = EvaluationOutcome(
            record=record,
            refinement=refinement,
        )

        assert outcome.record == record
        assert outcome.refinement == refinement

    def test_evaluation_outcome_rejects_mismatched_refined_candidate(self) -> None:
        observation = Observation(
            proposal=Proposal(candidate=4, proposal_id="p-1"),
            candidate=3,
            value=9.0,
            score=9.0,
        )
        refinement = CandidateRefinement(
            source_candidate=4,
            refined_candidate=2,
            changed_leaf_paths=((),),
        )

        with pytest.raises(ValueError):
            _ = EvaluationOutcome(observation=observation, refinement=refinement)

    def test_evaluation_outcome_rejects_truthy_non_scalar_candidate_equality(
        self,
    ) -> None:
        record_candidate = make_truthy_vector_equality_candidate()
        refined_candidate = make_truthy_vector_equality_candidate()
        proposal: Proposal[object] = Proposal(
            candidate=record_candidate,
            proposal_id="p-1",
        )
        observation: Observation[object] = Observation(
            proposal=proposal,
            candidate=record_candidate,
            value=1.0,
            score=1.0,
        )
        refinement: CandidateRefinement[object] = CandidateRefinement(
            source_candidate=record_candidate,
            refined_candidate=refined_candidate,
            changed_leaf_paths=((),),
        )

        with pytest.raises(TypeError, match="scalar truth value"):
            _ = EvaluationOutcome(observation=observation, refinement=refinement)

    def test_evaluation_outcome_accepts_explicit_candidate_equality(self) -> None:
        record_candidate = SpaceOwnedEqualityCandidate(1)
        refined_candidate = SpaceOwnedEqualityCandidate(1)
        observation: Observation[SpaceOwnedEqualityCandidate] = Observation(
            proposal=Proposal(candidate=SpaceOwnedEqualityCandidate(2), proposal_id="p-1"),
            candidate=record_candidate,
            value=1.0,
            score=1.0,
        )
        refinement = CandidateRefinement(
            source_candidate=SpaceOwnedEqualityCandidate(2),
            refined_candidate=refined_candidate,
            changed_leaf_paths=((),),
        )

        outcome = EvaluationOutcome(
            observation=observation,
            refinement=refinement,
            candidate_equal=space_owned_candidates_equal,
        )

        assert outcome.refinement == refinement

    def test_evaluation_outcome_rejects_explicit_candidate_equality_mismatch(
        self,
    ) -> None:
        observation: Observation[SpaceOwnedEqualityCandidate] = Observation(
            proposal=Proposal(candidate=SpaceOwnedEqualityCandidate(2), proposal_id="p-1"),
            candidate=SpaceOwnedEqualityCandidate(1),
            value=1.0,
            score=1.0,
        )
        refinement = CandidateRefinement(
            source_candidate=SpaceOwnedEqualityCandidate(2),
            refined_candidate=SpaceOwnedEqualityCandidate(3),
            changed_leaf_paths=((),),
        )

        with pytest.raises(ValueError, match="record candidate"):
            _ = EvaluationOutcome(
                observation=observation,
                refinement=refinement,
                candidate_equal=space_owned_candidates_equal,
            )

    def test_evaluation_outcome_skips_candidate_equality_without_refinement(
        self,
    ) -> None:
        observation: Observation[SpaceOwnedEqualityCandidate] = Observation(
            proposal=Proposal(candidate=SpaceOwnedEqualityCandidate(2), proposal_id="p-1"),
            candidate=SpaceOwnedEqualityCandidate(1),
            value=1.0,
            score=1.0,
        )

        outcome = EvaluationOutcome(
            observation=observation,
            candidate_equal=fail_if_candidate_equal_is_called,
        )

        assert outcome.refinement is None

    def test_evaluation_outcome_replace_preserves_candidate_equality(self) -> None:
        record_candidate = SpaceOwnedEqualityCandidate(1)
        refined_candidate = SpaceOwnedEqualityCandidate(1)
        observation: Observation[SpaceOwnedEqualityCandidate] = Observation(
            proposal=Proposal(candidate=SpaceOwnedEqualityCandidate(2), proposal_id="p-1"),
            candidate=record_candidate,
            value=1.0,
            score=1.0,
        )
        refinement = CandidateRefinement(
            source_candidate=SpaceOwnedEqualityCandidate(2),
            refined_candidate=refined_candidate,
            changed_leaf_paths=((),),
        )
        outcome = EvaluationOutcome(
            observation=observation,
            refinement=refinement,
            candidate_equal=space_owned_candidates_equal,
        )

        updated_outcome = replace(outcome, evaluation_count=2)

        assert updated_outcome.evaluation_count == 2
        assert updated_outcome.refinement == refinement

    def test_evaluation_outcome_replace_revalidates_changed_refinement(
        self,
    ) -> None:
        observation: Observation[SpaceOwnedEqualityCandidate] = Observation(
            proposal=Proposal(candidate=SpaceOwnedEqualityCandidate(2), proposal_id="p-1"),
            candidate=SpaceOwnedEqualityCandidate(1),
            value=1.0,
            score=1.0,
        )
        refinement = CandidateRefinement(
            source_candidate=SpaceOwnedEqualityCandidate(2),
            refined_candidate=SpaceOwnedEqualityCandidate(1),
            changed_leaf_paths=((),),
        )
        outcome = EvaluationOutcome(
            observation=observation,
            refinement=refinement,
            candidate_equal=space_owned_candidates_equal,
        )
        mismatched_refinement = CandidateRefinement(
            source_candidate=SpaceOwnedEqualityCandidate(2),
            refined_candidate=SpaceOwnedEqualityCandidate(3),
            changed_leaf_paths=((),),
        )

        with pytest.raises(ValueError, match="record candidate"):
            _ = replace(outcome, refinement=mismatched_refinement)

    def test_evaluation_outcome_pickle_omits_candidate_equality_revalidation(
        self,
    ) -> None:
        def local_candidate_equal(
            left_candidate: SpaceOwnedEqualityCandidate,
            right_candidate: SpaceOwnedEqualityCandidate,
        ) -> bool:
            return left_candidate.stable_id == right_candidate.stable_id

        observation: Observation[SpaceOwnedEqualityCandidate] = Observation(
            proposal=Proposal(candidate=SpaceOwnedEqualityCandidate(2), proposal_id="p-1"),
            candidate=SpaceOwnedEqualityCandidate(1),
            value=1.0,
            score=1.0,
        )
        refinement = CandidateRefinement(
            source_candidate=SpaceOwnedEqualityCandidate(2),
            refined_candidate=SpaceOwnedEqualityCandidate(1),
            changed_leaf_paths=((),),
        )
        outcome = EvaluationOutcome(
            observation=observation,
            refinement=refinement,
            candidate_equal=local_candidate_equal,
        )

        restored_outcome = cast(
            EvaluationOutcome[SpaceOwnedEqualityCandidate],
            pickle.loads(pickle.dumps(outcome)),
        )
        updated_outcome = replace(restored_outcome, evaluation_count=2)

        assert restored_outcome.refinement is not None
        assert restored_outcome.refinement.refined_candidate.stable_id == 1
        assert updated_outcome.evaluation_count == 2

    def test_unpickled_evaluation_outcome_requires_candidate_equal_for_new_refinement(
        self,
    ) -> None:
        observation: Observation[SpaceOwnedEqualityCandidate] = Observation(
            proposal=Proposal(candidate=SpaceOwnedEqualityCandidate(2), proposal_id="p-1"),
            candidate=SpaceOwnedEqualityCandidate(1),
            value=1.0,
            score=1.0,
        )
        refinement = CandidateRefinement(
            source_candidate=SpaceOwnedEqualityCandidate(2),
            refined_candidate=SpaceOwnedEqualityCandidate(1),
            changed_leaf_paths=((),),
        )
        outcome = EvaluationOutcome(
            observation=observation,
            refinement=refinement,
            candidate_equal=space_owned_candidates_equal,
        )
        restored_outcome = cast(
            EvaluationOutcome[SpaceOwnedEqualityCandidate],
            pickle.loads(pickle.dumps(outcome)),
        )
        replacement_refinement = CandidateRefinement(
            source_candidate=SpaceOwnedEqualityCandidate(2),
            refined_candidate=SpaceOwnedEqualityCandidate(1),
            changed_leaf_paths=(("replacement",),),
        )

        with pytest.raises(TypeError, match="candidate_equal is required"):
            _ = replace(restored_outcome, refinement=replacement_refinement)

        updated_outcome = replace(
            restored_outcome,
            refinement=replacement_refinement,
            candidate_equal=space_owned_candidates_equal,
        )

        assert updated_outcome.refinement == replacement_refinement

    def test_observation_rejects_negative_elapsed_seconds(self) -> None:
        proposal = Proposal(candidate=4, proposal_id="p-1")

        with pytest.raises(ValueError):
            _ = Observation(
                proposal=proposal,
                candidate=4,
                value=16.0,
                score=16.0,
                elapsed_seconds=-0.1,
            )

    def test_observation_rejects_nan_value(self) -> None:
        proposal = Proposal(candidate=4, proposal_id="p-1")

        with pytest.raises(ValueError):
            _ = Observation(
                proposal=proposal,
                candidate=4,
                value=float("nan"),
                score=16.0,
            )

    def test_observation_rejects_positive_infinity_value(self) -> None:
        proposal = Proposal(candidate=4, proposal_id="p-1")

        with pytest.raises(ValueError):
            _ = Observation(
                proposal=proposal,
                candidate=4,
                value=float("inf"),
                score=16.0,
            )

    def test_observation_rejects_negative_infinity_value(self) -> None:
        proposal = Proposal(candidate=4, proposal_id="p-1")

        with pytest.raises(ValueError):
            _ = Observation(
                proposal=proposal,
                candidate=4,
                value=float("-inf"),
                score=16.0,
            )

    def test_observation_rejects_nan_score(self) -> None:
        proposal = Proposal(candidate=4, proposal_id="p-1")

        with pytest.raises(ValueError):
            _ = Observation(
                proposal=proposal,
                candidate=4,
                value=16.0,
                score=float("nan"),
            )

    def test_trace_append_returns_new_trace(self) -> None:
        initial = Trace()
        event = TraceEvent(kind="evaluation", message="evaluated p-1", proposal_id="p-1")
        updated = initial.append(event)

        assert initial.events == ()
        assert updated.events == (event,)

    def test_run_result_from_observations_uses_minimization_semantics(self) -> None:
        proposal_one = Proposal(candidate=4, proposal_id="p-1")
        proposal_two = Proposal(candidate=2, proposal_id="p-2")
        observation_one: Observation[int] = Observation(
            proposal=proposal_one,
            candidate=4,
            value=16.0,
            score=16.0,
        )
        observation_two: Observation[int] = Observation(
            proposal=proposal_two,
            candidate=2,
            value=4.0,
            score=4.0,
        )
        trace = Trace(events=(TraceEvent(kind="run", message="completed"),))

        result: RunResult[int] = RunResult[int].from_observations(
            observations=(observation_one, observation_two),
            trace=trace,
        )

        assert result.best_observation == observation_two
        assert result.observations == (observation_one, observation_two)
        assert result.evaluation_count == 2
        assert result.trace == trace
        assert result.refinements == ()

    def test_run_result_preserves_observation_aligned_refinements(self) -> None:
        proposal_one = Proposal(candidate=4, proposal_id="p-1")
        proposal_two = Proposal(candidate=2, proposal_id="p-2")
        observation_one: Observation[int] = Observation(
            proposal=proposal_one,
            candidate=3,
            value=9.0,
            score=9.0,
        )
        observation_two: Observation[int] = Observation(
            proposal=proposal_two,
            candidate=2,
            value=4.0,
            score=4.0,
        )
        refinement = CandidateRefinement(
            source_candidate=4,
            refined_candidate=3,
            changed_leaf_paths=((),),
        )

        result = RunResult[int].from_observations(
            observations=(observation_one, observation_two),
            refinements=(refinement, None),
        )

        assert result.refinements == (refinement, None)

    def test_run_result_rejects_unaligned_refinements(self) -> None:
        observation = Observation(
            proposal=Proposal(candidate=4, proposal_id="p-1"),
            candidate=4,
            value=16.0,
            score=16.0,
        )

        with pytest.raises(ValueError):
            _ = RunResult[int].from_observations(
                observations=(observation,),
                refinements=(None, None),
            )

    def test_run_result_rejects_mismatched_refinement_candidate(self) -> None:
        observation = Observation(
            proposal=Proposal(candidate=4, proposal_id="p-1"),
            candidate=3,
            value=9.0,
            score=9.0,
        )
        refinement = CandidateRefinement(
            source_candidate=4,
            refined_candidate=2,
            changed_leaf_paths=((),),
        )

        with pytest.raises(ValueError):
            _ = RunResult[int].from_observations(
                observations=(observation,),
                refinements=(refinement,),
            )

    def test_run_result_canonicalizes_all_none_refinements(self) -> None:
        observation = Observation(
            proposal=Proposal(candidate=4, proposal_id="p-1"),
            candidate=4,
            value=16.0,
            score=16.0,
        )

        result = RunResult[int].from_observations(
            observations=(observation,),
            refinements=(None,),
        )

        assert result.refinements == ()

    def test_run_result_rejects_zero_observation_refinement_metadata(self) -> None:
        with pytest.raises(ValueError):
            _ = RunResult[int].from_observations(
                observations=(),
                refinements=(None,),
            )

    def test_objective_vector_record_rejects_empty_objective_values(self) -> None:
        proposal = Proposal(candidate=4, proposal_id="p-1")

        with pytest.raises(ValueError):
            _ = ObjectiveVectorRecord.from_objective_values(
                proposal=proposal,
                candidate=4,
                objective_values=(),
                directions=(),
            )

    def test_run_report_from_records_preserves_order_and_count(self) -> None:
        proposal_one = Proposal(candidate=4, proposal_id="p-1")
        proposal_two = Proposal(candidate=2, proposal_id="p-2")
        record_one = LabelRecord(
            request=EvaluationRequest(proposal=proposal_one),
            candidate=4,
            label="parity:0",
        )
        record_two = LabelRecord(
            request=EvaluationRequest(proposal=proposal_two),
            candidate=2,
            label="parity:0",
        )

        report = RunReport[int, LabelRecord].from_records(
            records=(record_one, record_two),
            evaluation_count=3,
        )

        assert report.records == (record_one, record_two)
        assert report.evaluation_count == 3
        assert report.trace.events == ()
        assert report.refinements == ()
        assert report.failures == ()

    def test_terminal_artifacts_preserve_failure_fields(self) -> None:
        proposal_one = Proposal(candidate=4, proposal_id="p-1")
        observation = Observation(
            proposal=proposal_one,
            candidate=4,
            value=16.0,
            score=16.0,
        )
        vector_record = ObjectiveVectorRecord.from_objective_values(
            proposal=proposal_one,
            candidate=4,
            objective_values=(16.0,),
            directions=(OptimizationDirection.MINIMIZE,),
        )
        failure = make_int_failure(make_int_request(2, "p-2"), "bad candidate")

        result = RunResult[int].from_observations(
            observations=(observation,),
            failures=(failure,),
        )
        report = RunReport[int, Observation[int]].from_records(
            records=(observation,),
            failures=(failure,),
        )
        surface = NondominatedRunSurface[int].from_records(
            records=(vector_record,),
            failures=(failure,),
        )
        surface_from_report = NondominatedRunSurface[int].from_report(
            RunReport[int, ObjectiveVectorRecord[int]].from_records(
                records=(vector_record,),
                failures=(failure,),
            ),
        )

        assert result.observations == (observation,)
        assert report.records == (observation,)
        assert surface.records == (vector_record,)
        assert result.failures == (failure,)
        assert report.failures == (failure,)
        assert surface.failures == (failure,)
        assert surface_from_report.failures == (failure,)
        assert result.evaluation_count == 2
        assert report.evaluation_count == 2
        assert surface.evaluation_count == 2

    def test_terminal_artifacts_default_count_includes_failure_cost(self) -> None:
        proposal = Proposal(candidate=4, proposal_id="p-1")
        observation = Observation(
            proposal=proposal,
            candidate=4,
            value=16.0,
            score=16.0,
        )
        failure = make_int_failure(
            make_int_request(2, "p-2"),
            "bad candidate",
            evaluation_count=3,
        )

        result = RunResult[int].from_observations(
            observations=(observation,),
            failures=(failure,),
        )

        assert result.evaluation_count == 4

    def test_terminal_artifacts_support_all_failure_without_successes(self) -> None:
        failure = make_int_failure(
            make_int_request(2, "p-2"),
            "bad candidate",
            evaluation_count=3,
        )

        result = RunResult[int].from_successes(successes=(), failures=(failure,))
        report = RunReport[int, Observation[int]].from_successes(
            successes=(),
            failures=(failure,),
        )
        surface = NondominatedRunSurface[int].from_successes(
            successes=(),
            failures=(failure,),
        )

        assert result.best_success is None
        assert result.observations == ()
        assert report.records == ()
        assert surface.nondominated_records == ()
        assert result.evaluation_count == 3
        assert report.evaluation_count == 3
        assert surface.evaluation_count == 3

    def test_terminal_artifacts_reject_evaluation_count_below_failure_cost(self) -> None:
        proposal_one = Proposal(candidate=4, proposal_id="p-1")
        observation = Observation(
            proposal=proposal_one,
            candidate=4,
            value=16.0,
            score=16.0,
        )
        failure = make_int_failure(make_int_request(2, "p-2"), "bad candidate")

        with pytest.raises(ValueError, match="evaluation_count"):
            _ = RunResult[int].from_observations(
                observations=(observation,),
                failures=(failure,),
                evaluation_count=1,
            )

        with pytest.raises(ValueError, match="evaluation_count"):
            _ = RunReport[int, Observation[int]].from_records(
                records=(observation,),
                failures=(failure,),
                evaluation_count=1,
            )

    def test_terminal_failure_fields_pickle_round_trip(self) -> None:
        proposal_one = Proposal(candidate=4, proposal_id="p-1")
        observation = Observation(
            proposal=proposal_one,
            candidate=4,
            value=16.0,
            score=16.0,
        )
        failure = make_int_failure(make_int_request(2, "p-2"), "bad candidate")
        result = RunResult[int].from_observations(
            observations=(observation,),
            failures=(failure,),
        )

        restored_result = cast(
            RunResult[int],
            pickle.loads(pickle.dumps(result)),
        )

        assert restored_result == result
        assert restored_result.failures == (failure,)

    def test_run_report_preserves_record_aligned_refinements(self) -> None:
        proposal_one = Proposal(candidate=4, proposal_id="p-1")
        proposal_two = Proposal(candidate=2, proposal_id="p-2")
        record_one = LabelRecord(
            request=EvaluationRequest(proposal=proposal_one),
            candidate=3,
            label="parity:1",
        )
        record_two = LabelRecord(
            request=EvaluationRequest(proposal=proposal_two),
            candidate=2,
            label="parity:0",
        )
        refinement = CandidateRefinement(
            source_candidate=4,
            refined_candidate=3,
            changed_leaf_paths=((),),
        )

        report = RunReport[int, LabelRecord].from_records(
            records=(record_one, record_two),
            refinements=(refinement, None),
        )

        assert report.refinements == (refinement, None)

    def test_run_report_rejects_unaligned_refinements(self) -> None:
        proposal = Proposal(candidate=4, proposal_id="p-1")
        record = LabelRecord(
            request=EvaluationRequest(proposal=proposal),
            candidate=3,
            label="parity:1",
        )
        refinement = CandidateRefinement(
            source_candidate=4,
            refined_candidate=3,
            changed_leaf_paths=((),),
        )

        with pytest.raises(ValueError):
            _ = RunReport[int, LabelRecord].from_records(
                records=(record,),
                refinements=(refinement, None),
            )

    def test_run_report_canonicalizes_all_none_refinements(self) -> None:
        proposal = Proposal(candidate=4, proposal_id="p-1")
        record = LabelRecord(
            request=EvaluationRequest(proposal=proposal),
            candidate=4,
            label="parity:0",
        )

        report = RunReport[int, LabelRecord].from_records(
            records=(record,),
            refinements=(None,),
        )

        assert report.refinements == ()

    def test_run_report_skips_candidate_equality_for_all_none_refinements(
        self,
    ) -> None:
        proposal = Proposal(
            candidate=SpaceOwnedEqualityCandidate(1),
            proposal_id="p-1",
        )
        record = Observation(
            proposal=proposal,
            candidate=SpaceOwnedEqualityCandidate(1),
            value=1.0,
            score=1.0,
        )

        report = RunReport[
            SpaceOwnedEqualityCandidate,
            Observation[SpaceOwnedEqualityCandidate],
        ].from_records(
            records=(record,),
            refinements=(None,),
            candidate_equal=fail_if_candidate_equal_is_called,
        )

        assert report.refinements == ()

    def test_run_report_constructor_canonicalizes_all_none_refinements(self) -> None:
        proposal = Proposal(candidate=4, proposal_id="p-1")
        record = LabelRecord(
            request=EvaluationRequest(proposal=proposal),
            candidate=4,
            label="parity:0",
        )

        report = RunReport[int, LabelRecord].from_records(
            records=(record,),
            evaluation_count=1,
            refinements=(None,),
        )

        assert report.refinements == ()

    def test_run_report_rejects_zero_record_refinement_metadata(self) -> None:
        with pytest.raises(ValueError):
            _ = RunReport[int, LabelRecord].from_records(
                records=(),
                refinements=(None,),
            )

    def test_run_report_rejects_mismatched_refinement_candidate(self) -> None:
        proposal = Proposal(candidate=4, proposal_id="p-1")
        record = LabelRecord(
            request=EvaluationRequest(proposal=proposal),
            candidate=3,
            label="parity:1",
        )
        refinement = CandidateRefinement(
            source_candidate=4,
            refined_candidate=2,
            changed_leaf_paths=((),),
        )

        with pytest.raises(ValueError):
            _ = RunReport[int, LabelRecord].from_records(
                records=(record,),
                refinements=(refinement,),
            )

    def test_run_report_rejects_ambiguous_refinement_candidate_equality(
        self,
    ) -> None:
        candidate = AmbiguousEqualityCandidate()
        proposal = Proposal(candidate=candidate, proposal_id="p-1")
        record = Observation(
            request=EvaluationRequest(proposal=proposal),
            candidate=candidate,
            value=1.0,
            score=1.0,
        )
        refinement = CandidateRefinement(
            source_candidate=candidate,
            refined_candidate=candidate,
            changed_leaf_paths=((),),
        )

        with pytest.raises(TypeError):
            _ = RunReport[
                AmbiguousEqualityCandidate,
                Observation[AmbiguousEqualityCandidate],
            ].from_records(
                records=(record,),
                refinements=(refinement,),
            )

    def test_run_report_rejects_truthy_non_scalar_refinement_candidate_equality(
        self,
    ) -> None:
        record_candidate = make_truthy_vector_equality_candidate()
        refined_candidate = make_truthy_vector_equality_candidate()
        proposal: Proposal[object] = Proposal(
            candidate=record_candidate,
            proposal_id="p-1",
        )
        request: EvaluationRequest[object] = EvaluationRequest(
            proposal=proposal,
        )
        record: Observation[object] = Observation(
            request=request,
            candidate=record_candidate,
            value=1.0,
            score=1.0,
        )
        refinement: CandidateRefinement[object] = CandidateRefinement(
            source_candidate=record_candidate,
            refined_candidate=refined_candidate,
            changed_leaf_paths=((),),
        )

        with pytest.raises(TypeError, match="scalar truth value"):
            _ = RunReport[
                object,
                Observation[object],
            ].from_records(
                records=(record,),
                refinements=(refinement,),
            )

    def test_terminal_surfaces_accept_explicit_candidate_equality(self) -> None:
        record_candidate = SpaceOwnedEqualityCandidate(1)
        refined_candidate = SpaceOwnedEqualityCandidate(1)
        proposal = Proposal(
            candidate=SpaceOwnedEqualityCandidate(2),
            proposal_id="p-1",
        )
        observation = Observation(
            proposal=proposal,
            candidate=record_candidate,
            value=1.0,
            score=1.0,
        )
        refinement = CandidateRefinement(
            source_candidate=proposal.candidate,
            refined_candidate=refined_candidate,
            changed_leaf_paths=((),),
        )

        result = RunResult[SpaceOwnedEqualityCandidate].from_observations(
            observations=(observation,),
            refinements=(refinement,),
            candidate_equal=space_owned_candidates_equal,
        )
        report = RunReport[
            SpaceOwnedEqualityCandidate,
            Observation[SpaceOwnedEqualityCandidate],
        ].from_records(
            records=(observation,),
            refinements=(refinement,),
            candidate_equal=space_owned_candidates_equal,
        )
        surface = NondominatedRunSurface[SpaceOwnedEqualityCandidate].from_records(
            records=(
                ObjectiveVectorRecord.from_objective_values(
                    proposal=proposal,
                    candidate=record_candidate,
                    objective_values=(1.0,),
                    directions=(OptimizationDirection.MINIMIZE,),
                ),
            ),
            refinements=(refinement,),
            candidate_equal=space_owned_candidates_equal,
        )

        assert result.refinements == (refinement,)
        assert report.refinements == (refinement,)
        assert surface.refinements == (refinement,)

    def test_terminal_surface_replace_preserves_candidate_equality(self) -> None:
        def reject_candidate_equal(
            left_candidate: SpaceOwnedEqualityCandidate,
            right_candidate: SpaceOwnedEqualityCandidate,
        ) -> bool:
            _ = left_candidate
            _ = right_candidate
            return False

        observation: Observation[SpaceOwnedEqualityCandidate] = Observation(
            proposal=Proposal(candidate=SpaceOwnedEqualityCandidate(2), proposal_id="p-1"),
            candidate=SpaceOwnedEqualityCandidate(1),
            value=1.0,
            score=1.0,
        )
        vector_record = ObjectiveVectorRecord.from_objective_values(
            proposal=observation.proposal,
            candidate=observation.candidate,
            objective_values=(1.0,),
            directions=(OptimizationDirection.MINIMIZE,),
        )
        refinement = CandidateRefinement(
            source_candidate=SpaceOwnedEqualityCandidate(2),
            refined_candidate=SpaceOwnedEqualityCandidate(1),
            changed_leaf_paths=((),),
        )
        result = RunResult[SpaceOwnedEqualityCandidate].from_observations(
            observations=(observation,),
            refinements=(refinement,),
            candidate_equal=space_owned_candidates_equal,
        )
        report = RunReport[
            SpaceOwnedEqualityCandidate,
            Observation[SpaceOwnedEqualityCandidate],
        ].from_records(
            records=(observation,),
            refinements=(refinement,),
            candidate_equal=space_owned_candidates_equal,
        )
        surface = NondominatedRunSurface[SpaceOwnedEqualityCandidate].from_records(
            records=(vector_record,),
            refinements=(refinement,),
            candidate_equal=space_owned_candidates_equal,
        )
        replacement_refinement = CandidateRefinement(
            source_candidate=SpaceOwnedEqualityCandidate(2),
            refined_candidate=SpaceOwnedEqualityCandidate(1),
            changed_leaf_paths=(("replacement",),),
        )
        mismatched_refinement = CandidateRefinement(
            source_candidate=SpaceOwnedEqualityCandidate(2),
            refined_candidate=SpaceOwnedEqualityCandidate(3),
            changed_leaf_paths=(("mismatch",),),
        )

        updated_result = replace(result, refinements=(replacement_refinement,))
        updated_report = replace(report, refinements=(replacement_refinement,))
        updated_surface = replace(surface, refinements=(replacement_refinement,))

        assert updated_result.refinements == (replacement_refinement,)
        assert updated_report.refinements == (replacement_refinement,)
        assert updated_surface.refinements == (replacement_refinement,)
        with pytest.raises(ValueError, match="success request candidate"):
            _ = replace(result, refinements=(mismatched_refinement,))
        with pytest.raises(ValueError, match="success request candidate"):
            _ = replace(report, refinements=(mismatched_refinement,))
        with pytest.raises(ValueError, match="success request candidate"):
            _ = replace(surface, refinements=(mismatched_refinement,))
        with pytest.raises(ValueError, match="refinement source candidate"):
            _ = replace(report, candidate_equal=reject_candidate_equal)

    def test_run_result_projection_preserves_refinement_source_proposal(self) -> None:
        observation = Observation(
            proposal=Proposal(candidate=4, proposal_id="p-1"),
            candidate=2,
            value=4.0,
            score=4.0,
        )
        refinement = CandidateRefinement(
            source_candidate=4,
            refined_candidate=2,
            changed_leaf_paths=((),),
        )

        result = RunResult[int].from_observations(
            observations=(observation,),
            refinements=(refinement,),
        )

        assert result.successes[0].request.candidate == 2
        assert result.observations == (observation,)

    def test_unpickled_run_report_requires_candidate_equal_for_new_refinement(
        self,
    ) -> None:
        def local_candidate_equal(
            left_candidate: SpaceOwnedEqualityCandidate,
            right_candidate: SpaceOwnedEqualityCandidate,
        ) -> bool:
            return left_candidate.stable_id == right_candidate.stable_id

        observation: Observation[SpaceOwnedEqualityCandidate] = Observation(
            proposal=Proposal(candidate=SpaceOwnedEqualityCandidate(2), proposal_id="p-1"),
            candidate=SpaceOwnedEqualityCandidate(1),
            value=1.0,
            score=1.0,
        )
        vector_record = ObjectiveVectorRecord.from_objective_values(
            proposal=observation.proposal,
            candidate=observation.candidate,
            objective_values=(1.0,),
            directions=(OptimizationDirection.MINIMIZE,),
        )
        refinement = CandidateRefinement(
            source_candidate=SpaceOwnedEqualityCandidate(2),
            refined_candidate=SpaceOwnedEqualityCandidate(1),
            changed_leaf_paths=((),),
        )
        result = RunResult[SpaceOwnedEqualityCandidate].from_observations(
            observations=(observation,),
            refinements=(refinement,),
            candidate_equal=local_candidate_equal,
        )
        report = RunReport[
            SpaceOwnedEqualityCandidate,
            Observation[SpaceOwnedEqualityCandidate],
        ].from_records(
            records=(observation,),
            refinements=(refinement,),
            candidate_equal=local_candidate_equal,
        )
        surface = NondominatedRunSurface[SpaceOwnedEqualityCandidate].from_records(
            records=(vector_record,),
            refinements=(refinement,),
            candidate_equal=local_candidate_equal,
        )
        restored_result = cast(
            RunResult[SpaceOwnedEqualityCandidate],
            pickle.loads(pickle.dumps(result)),
        )
        restored_report = cast(
            RunReport[
                SpaceOwnedEqualityCandidate,
                Observation[SpaceOwnedEqualityCandidate],
            ],
            pickle.loads(pickle.dumps(report)),
        )
        restored_surface = cast(
            NondominatedRunSurface[SpaceOwnedEqualityCandidate],
            pickle.loads(pickle.dumps(surface)),
        )
        replacement_refinement = CandidateRefinement(
            source_candidate=SpaceOwnedEqualityCandidate(2),
            refined_candidate=SpaceOwnedEqualityCandidate(1),
            changed_leaf_paths=(("replacement",),),
        )

        updated_result = replace(restored_result, evaluation_count=2)
        updated_report = replace(restored_report, evaluation_count=2)
        updated_surface = replace(restored_surface, evaluation_count=2)

        assert updated_result.evaluation_count == 2
        assert updated_report.evaluation_count == 2
        assert updated_surface.evaluation_count == 2
        with pytest.raises(TypeError, match="candidate_equal is required"):
            _ = replace(restored_result, refinements=(replacement_refinement,))
        with pytest.raises(TypeError, match="candidate_equal is required"):
            _ = replace(restored_report, refinements=(replacement_refinement,))
        with pytest.raises(TypeError, match="candidate_equal is required"):
            _ = replace(restored_surface, refinements=(replacement_refinement,))

        revalidated_result = replace(
            restored_result,
            refinements=(replacement_refinement,),
            candidate_equal=space_owned_candidates_equal,
        )
        revalidated_report = replace(
            restored_report,
            refinements=(replacement_refinement,),
            candidate_equal=space_owned_candidates_equal,
        )
        revalidated_surface = replace(
            restored_surface,
            refinements=(replacement_refinement,),
            candidate_equal=space_owned_candidates_equal,
        )

        assert revalidated_result.refinements == (replacement_refinement,)
        assert revalidated_report.refinements == (replacement_refinement,)
        assert revalidated_surface.refinements == (replacement_refinement,)

    def test_nondominated_run_surface_rejects_truthy_non_scalar_refinement_equality(
        self,
    ) -> None:
        record_candidate = make_truthy_vector_equality_candidate()
        refined_candidate = make_truthy_vector_equality_candidate()
        record = ObjectiveVectorRecord.from_objective_values(
            proposal=Proposal(candidate=record_candidate, proposal_id="p-1"),
            candidate=record_candidate,
            objective_values=(1.0, 2.0),
            directions=(
                OptimizationDirection.MINIMIZE,
                OptimizationDirection.MINIMIZE,
            ),
        )
        refinement: CandidateRefinement[object] = CandidateRefinement(
            source_candidate=record_candidate,
            refined_candidate=refined_candidate,
            changed_leaf_paths=((),),
        )

        with pytest.raises(TypeError, match="scalar truth value"):
            _ = NondominatedRunSurface[object].from_records(
                records=(record,),
                refinements=(refinement,),
            )

    def test_nondominated_run_surface_from_report_preserves_frontier_order(self) -> None:
        record_one = ObjectiveVectorRecord.from_objective_values(
            proposal=Proposal(candidate=1, proposal_id="p-1"),
            candidate=1,
            objective_values=(1.0, 3.0),
            directions=(
                OptimizationDirection.MINIMIZE,
                OptimizationDirection.MINIMIZE,
            ),
        )
        record_two = ObjectiveVectorRecord.from_objective_values(
            proposal=Proposal(candidate=2, proposal_id="p-2"),
            candidate=2,
            objective_values=(2.0, 2.0),
            directions=(
                OptimizationDirection.MINIMIZE,
                OptimizationDirection.MINIMIZE,
            ),
        )
        record_three = ObjectiveVectorRecord.from_objective_values(
            proposal=Proposal(candidate=3, proposal_id="p-3"),
            candidate=3,
            objective_values=(3.0, 1.0),
            directions=(
                OptimizationDirection.MINIMIZE,
                OptimizationDirection.MINIMIZE,
            ),
        )
        dominated_record = ObjectiveVectorRecord.from_objective_values(
            proposal=Proposal(candidate=4, proposal_id="p-4"),
            candidate=4,
            objective_values=(4.0, 4.0),
            directions=(
                OptimizationDirection.MINIMIZE,
                OptimizationDirection.MINIMIZE,
            ),
        )
        trace = Trace(events=(TraceEvent(kind="run", message="completed"),))
        report = RunReport[int, ObjectiveVectorRecord[int]].from_records(
            records=(record_one, record_two, record_three, dominated_record),
            evaluation_count=5,
            trace=trace,
        )

        surface = NondominatedRunSurface[int].from_report(report)

        assert surface.nondominated_records == (record_one, record_two, record_three)
        assert surface.records == report.records
        assert surface.evaluation_count == 5
        assert surface.trace == trace
        assert surface.refinements == ()

    def test_nondominated_run_surface_from_report_preserves_refinements(
        self,
    ) -> None:
        record_one = ObjectiveVectorRecord.from_objective_values(
            proposal=Proposal(candidate=4, proposal_id="p-1"),
            candidate=3,
            objective_values=(3.0, 1.0),
            directions=(
                OptimizationDirection.MINIMIZE,
                OptimizationDirection.MINIMIZE,
            ),
        )
        record_two = ObjectiveVectorRecord.from_objective_values(
            proposal=Proposal(candidate=2, proposal_id="p-2"),
            candidate=2,
            objective_values=(1.0, 3.0),
            directions=(
                OptimizationDirection.MINIMIZE,
                OptimizationDirection.MINIMIZE,
            ),
        )
        refinement = CandidateRefinement(
            source_candidate=4,
            refined_candidate=3,
            changed_leaf_paths=((),),
        )
        report = RunReport[int, ObjectiveVectorRecord[int]].from_records(
            records=(record_one, record_two),
            refinements=(refinement, None),
        )

        surface = NondominatedRunSurface[int].from_report(report)

        assert surface.records == report.records
        assert tuple(success.request.candidate for success in surface.successes) == (3, 2)
        assert surface.successes[0].request.proposal_id == "p-1"
        assert surface.refinements == (refinement, None)

    def test_nondominated_run_surface_from_records_reuses_prevalidated_frontier(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        record = ObjectiveVectorRecord.from_objective_values(
            proposal=Proposal(candidate=1, proposal_id="p-1"),
            candidate=1,
            objective_values=(1.0, 2.0),
            directions=(
                OptimizationDirection.MINIMIZE,
                OptimizationDirection.MINIMIZE,
            ),
        )
        dominated_record = ObjectiveVectorRecord.from_objective_values(
            proposal=Proposal(candidate=2, proposal_id="p-2"),
            candidate=2,
            objective_values=(3.0, 4.0),
            directions=(
                OptimizationDirection.MINIMIZE,
                OptimizationDirection.MINIMIZE,
            ),
        )
        collection_calls: list[
            tuple[EvaluationSuccess[int, ObjectiveVectorPayload], ...]
        ] = []
        original_collect_nondominated_successes = (
            terminal_artifacts.collect_nondominated_successes
        )

        def collect_counting_frontier(
            successes: tuple[EvaluationSuccess[int, ObjectiveVectorPayload], ...],
        ) -> tuple[EvaluationSuccess[int, ObjectiveVectorPayload], ...]:
            collection_calls.append(successes)
            return original_collect_nondominated_successes(successes)

        monkeypatch.setattr(
            terminal_artifacts,
            "collect_nondominated_successes",
            collect_counting_frontier,
        )

        surface = NondominatedRunSurface[int].from_records((record, dominated_record))

        assert len(collection_calls) == 1
        assert tuple(
            success.payload.objective_scores for success in collection_calls[0]
        ) == (record.objective_scores, dominated_record.objective_scores)
        assert surface.nondominated_records == (record,)

    def test_nondominated_run_surface_rejects_public_prevalidation_cache_injection(
        self,
    ) -> None:
        constructor_parameters = signature(NondominatedRunSurface).parameters

        assert "_validated_frontier_source_records" not in constructor_parameters
        assert "_validated_frontier_records" not in constructor_parameters

    def test_nondominated_run_surface_from_records_initializes_dataclass_fields(
        self,
    ) -> None:
        record = ObjectiveVectorRecord.from_objective_values(
            proposal=Proposal(candidate=1, proposal_id="p-1"),
            candidate=1,
            objective_values=(1.0, 2.0),
            directions=(
                OptimizationDirection.MINIMIZE,
                OptimizationDirection.MINIMIZE,
            ),
        )

        surface = NondominatedRunSurface[int].from_records((record,))

        assert all(
            hasattr(surface, dataclass_field.name)
            for dataclass_field in fields(surface)
        )

    def test_nondominated_run_surface_replace_revalidates_changed_records(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        record = ObjectiveVectorRecord.from_objective_values(
            proposal=Proposal(candidate=1, proposal_id="p-1"),
            candidate=1,
            objective_values=(1.0, 2.0),
            directions=(
                OptimizationDirection.MINIMIZE,
                OptimizationDirection.MINIMIZE,
            ),
        )
        replacement_record = ObjectiveVectorRecord.from_objective_values(
            proposal=Proposal(candidate=2, proposal_id="p-2"),
            candidate=2,
            objective_values=(2.0, 1.0),
            directions=(
                OptimizationDirection.MINIMIZE,
                OptimizationDirection.MINIMIZE,
            ),
        )
        surface = NondominatedRunSurface[int].from_records((record,))
        collection_calls: list[
            tuple[EvaluationSuccess[int, ObjectiveVectorPayload], ...]
        ] = []
        original_collect_nondominated_successes = (
            terminal_artifacts.collect_nondominated_successes
        )

        def collect_counting_frontier(
            successes: tuple[EvaluationSuccess[int, ObjectiveVectorPayload], ...],
        ) -> tuple[EvaluationSuccess[int, ObjectiveVectorPayload], ...]:
            collection_calls.append(successes)
            return original_collect_nondominated_successes(successes)

        monkeypatch.setattr(
            "variopt.artifacts.terminal.collect_nondominated_successes",
            collect_counting_frontier,
        )

        replaced = replace(surface, records=(replacement_record,))

        assert len(collection_calls) == 2
        assert tuple(
            success.payload.objective_scores for success in collection_calls[0]
        ) == (replacement_record.objective_scores,)
        assert replaced.records == (replacement_record,)
        assert replaced.nondominated_records == (replacement_record,)

        with pytest.raises(ValueError, match="stable nondominated frontier"):
            _ = NondominatedRunSurface[int](
                records=(replacement_record,),
                nondominated_records=(record,),
                evaluation_count=1,
            )

    def test_nondominated_run_surface_rejects_unaligned_refinements(self) -> None:
        record = ObjectiveVectorRecord.from_objective_values(
            proposal=Proposal(candidate=4, proposal_id="p-1"),
            candidate=4,
            objective_values=(4.0, 1.0),
            directions=(
                OptimizationDirection.MINIMIZE,
                OptimizationDirection.MINIMIZE,
            ),
        )

        with pytest.raises(ValueError):
            _ = NondominatedRunSurface[int].from_records(
                records=(record,),
                refinements=(None, None),
            )

    def test_nondominated_run_surface_rejects_mismatched_refinement_candidate(
        self,
    ) -> None:
        record = ObjectiveVectorRecord.from_objective_values(
            proposal=Proposal(candidate=4, proposal_id="p-1"),
            candidate=3,
            objective_values=(3.0, 1.0),
            directions=(
                OptimizationDirection.MINIMIZE,
                OptimizationDirection.MINIMIZE,
            ),
        )
        refinement = CandidateRefinement(
            source_candidate=4,
            refined_candidate=2,
            changed_leaf_paths=((),),
        )

        with pytest.raises(ValueError):
            _ = NondominatedRunSurface[int].from_records(
                records=(record,),
                refinements=(refinement,),
            )

    def test_nondominated_run_surface_canonicalizes_all_none_refinements(self) -> None:
        record = ObjectiveVectorRecord.from_objective_values(
            proposal=Proposal(candidate=4, proposal_id="p-1"),
            candidate=4,
            objective_values=(4.0, 1.0),
            directions=(
                OptimizationDirection.MINIMIZE,
                OptimizationDirection.MINIMIZE,
            ),
        )

        surface = NondominatedRunSurface[int].from_records(
            records=(record,),
            refinements=(None,),
        )

        assert surface.refinements == ()

    def test_nondominated_run_surface_rejects_mixed_objective_dimensions(self) -> None:
        record_one = ObjectiveVectorRecord.from_objective_values(
            proposal=Proposal(candidate=1, proposal_id="p-1"),
            candidate=1,
            objective_values=(1.0, 2.0),
            directions=(
                OptimizationDirection.MINIMIZE,
                OptimizationDirection.MINIMIZE,
            ),
        )
        record_two = ObjectiveVectorRecord.from_objective_values(
            proposal=Proposal(candidate=2, proposal_id="p-2"),
            candidate=2,
            objective_values=(3.0, 4.0, 5.0),
            directions=(
                OptimizationDirection.MINIMIZE,
                OptimizationDirection.MINIMIZE,
                OptimizationDirection.MINIMIZE,
            ),
        )

        with pytest.raises(ValueError):
            _ = NondominatedRunSurface[int].from_records((record_one, record_two))

    def test_run_report_rejects_evaluation_count_below_record_count(self) -> None:
        proposal = Proposal(candidate=4, proposal_id="p-1")
        record = LabelRecord(
            request=EvaluationRequest(proposal=proposal),
            candidate=4,
            label="parity:0",
        )

        with pytest.raises(ValueError):
            _report: RunReport[int, LabelRecord] = RunReport.from_records(
                records=(record,),
                evaluation_count=0,
            )

    def test_run_result_rejects_foreign_best_observation(self) -> None:
        proposal_one = Proposal(candidate=4, proposal_id="p-1")
        proposal_two = Proposal(candidate=2, proposal_id="p-2")
        observation_one = Observation(
            proposal=proposal_one,
            candidate=4,
            value=16.0,
            score=16.0,
        )
        foreign_observation = Observation(
            proposal=proposal_two,
            candidate=2,
            value=4.0,
            score=4.0,
        )

        with pytest.raises(ValueError):
            _ = RunResult(
                best_observation=foreign_observation,
                observations=(observation_one,),
                evaluation_count=1,
            )

    def test_run_result_rejects_nonminimal_best_observation(self) -> None:
        proposal_one = Proposal(candidate=4, proposal_id="p-1")
        proposal_two = Proposal(candidate=2, proposal_id="p-2")
        observation_one = Observation(
            proposal=proposal_one,
            candidate=4,
            value=16.0,
            score=16.0,
        )
        observation_two = Observation(
            proposal=proposal_two,
            candidate=2,
            value=4.0,
            score=4.0,
        )

        with pytest.raises(ValueError):
            _ = RunResult(
                best_observation=observation_one,
                observations=(observation_one, observation_two),
                evaluation_count=2,
            )

    def test_run_result_rejects_evaluation_count_below_observation_count(self) -> None:
        proposal = Proposal(candidate=4, proposal_id="p-1")
        observation = Observation(
            proposal=proposal,
            candidate=4,
            value=16.0,
            score=16.0,
        )

        with pytest.raises(ValueError):
            _ = RunResult(
                best_observation=observation,
                observations=(observation,),
                evaluation_count=0,
            )

    def test_run_result_uses_score_not_raw_value_for_maximize(self) -> None:
        proposal_one = Proposal(candidate=4, proposal_id="p-1")
        proposal_two = Proposal(candidate=2, proposal_id="p-2")
        observation_one = Observation.from_objective_value(
            proposal=proposal_one,
            candidate=4,
            value=16.0,
            direction=OptimizationDirection.MAXIMIZE,
        )
        observation_two = Observation.from_objective_value(
            proposal=proposal_two,
            candidate=2,
            value=4.0,
            direction=OptimizationDirection.MAXIMIZE,
        )

        result = RunResult[int].from_observations(
            observations=(observation_one, observation_two),
        )

        assert result.best_observation == observation_one
        assert result.best_observation is not None
        assert result.best_observation.value == 16.0
        assert result.best_observation.score == -16.0

    def test_run_result_replace_observations_recomputes_best_success(self) -> None:
        original_observation = Observation(
            proposal=Proposal(candidate=4, proposal_id="p-1"),
            candidate=4,
            value=16.0,
            score=16.0,
        )
        replacement_observation = Observation(
            proposal=Proposal(candidate=2, proposal_id="p-2"),
            candidate=2,
            value=4.0,
            score=4.0,
        )
        result = RunResult[int].from_observations((original_observation,))

        replaced = replace(result, observations=(replacement_observation,))

        assert replaced.observations == (replacement_observation,)
        assert replaced.best_observation == replacement_observation
