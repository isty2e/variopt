"""Tests for runtime artifact values and terminal surfaces."""

import pickle
from collections.abc import Sequence
from dataclasses import dataclass, fields, is_dataclass, replace
from inspect import signature
from typing import Protocol, cast

import numpy as np
import pytest
from typing_extensions import override

import variopt.artifacts.records as record_artifacts
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
    KernelDiagnostics,
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
from variopt.artifacts.records import RequestAlignedEvaluationRecord
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


class FloatSubclass(float):
    """Runtime float subclass used to verify scalar canonicalization."""


class TerminalSurfacePickleHooks(Protocol):
    """Dynamically installed pickle hooks for terminal-surface tests."""

    def __getstate__(self) -> list[object | None]:
        """Return the installed terminal-surface pickle state."""
        ...

    def __setstate__(self, state: list[object | None]) -> None:
        """Restore one installed terminal-surface pickle state."""
        ...


class ExceptionSnapshotPickleHooks(Protocol):
    """Dynamically installed pickle hooks for exception snapshot tests."""

    def __setstate__(self, state: tuple[str, str, str]) -> None:
        """Restore one exception-snapshot pickle state."""
        ...


class EvaluationFailurePickleHooks(Protocol):
    """Dynamically installed pickle hooks for evaluation-failure tests."""

    def __setstate__(
        self,
        state: tuple[EvaluationRequest[int], EvaluationExceptionSnapshot, int],
    ) -> None:
        """Restore one evaluation-failure pickle state."""
        ...


def terminal_surface_pickle_hooks(surface: object) -> TerminalSurfacePickleHooks:
    """Return dynamically installed terminal-surface pickle hooks for tests."""
    return cast(TerminalSurfacePickleHooks, surface)


def exception_snapshot_pickle_hooks(
    snapshot: object,
) -> ExceptionSnapshotPickleHooks:
    """Return dynamically installed exception-snapshot pickle hooks for tests."""
    return cast(ExceptionSnapshotPickleHooks, snapshot)


def evaluation_failure_pickle_hooks(failure: object) -> EvaluationFailurePickleHooks:
    """Return dynamically installed evaluation-failure pickle hooks for tests."""
    return cast(EvaluationFailurePickleHooks, failure)


def terminal_surface_pickle_state_with_field(
    surface: object,
    *,
    field_name: str,
    value: object | None,
) -> list[object | None]:
    """Return a terminal-surface pickle state with one named field replaced."""
    if not is_dataclass(surface):
        msg = "surface must be a dataclass instance"
        raise TypeError(msg)

    pickle_state = terminal_surface_pickle_hooks(surface).__getstate__()
    field_names = tuple(dataclass_field.name for dataclass_field in fields(surface))
    pickle_state[field_names.index(field_name)] = value
    return pickle_state


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


@dataclass(frozen=True, slots=True)
class SpaceOwnedRecord:
    """Picklable request-aligned payload for space-owned equality candidates."""

    request: EvaluationRequest[SpaceOwnedEqualityCandidate]
    candidate: SpaceOwnedEqualityCandidate
    label: str


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


def make_vector_record_success(
    record: ObjectiveVectorRecord[int],
    *,
    refinement: CandidateRefinement[int] | None = None,
) -> EvaluationSuccess[int, ObjectiveVectorRecord[int]]:
    """Return a typed vector-record success aligned to the evaluated candidate."""
    request = EvaluationRequest(
        proposal=Proposal(
            candidate=record.candidate,
            proposal_id=record.proposal.proposal_id,
        ),
        proposal_evaluation_spec=record.proposal_evaluation_spec,
    )
    return EvaluationSuccess(
        request=request,
        payload=record,
        refinement=refinement,
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

    def test_observation_payload_normalizes_numeric_fields(self) -> None:
        payload = ObservationPayload(
            value=cast(float, cast(object, np.int64(4))),
            score=cast(float, cast(object, np.float64(-4.0))),
            elapsed_seconds=cast(float, cast(object, np.float64(0.25))),
        )

        assert payload.value == 4.0
        assert type(payload.value) is float
        assert payload.score == -4.0
        assert type(payload.score) is float
        assert payload.elapsed_seconds == 0.25
        assert type(payload.elapsed_seconds) is float

    def test_observation_payload_rejects_bool_numeric_fields(self) -> None:
        with pytest.raises(TypeError, match="value must be a real number"):
            _ = ObservationPayload(value=cast(float, True), score=1.0)

        with pytest.raises(TypeError, match="value must be a real number"):
            _ = ObservationPayload(
                value=cast(float, cast(object, np.bool_(True))),
                score=1.0,
            )

        with pytest.raises(TypeError, match="score must be a real number"):
            _ = ObservationPayload(value=1.0, score=cast(float, False))

        with pytest.raises(TypeError, match="elapsed_seconds must be a real number"):
            _ = ObservationPayload(
                value=1.0,
                score=1.0,
                elapsed_seconds=cast(float, cast(object, np.bool_(False))),
            )

    def test_kernel_diagnostics_tracks_inner_failed_attempt_accounting(self) -> None:
        diagnostics = KernelDiagnostics(
            backend="local-search",
            failed_attempt_count=2,
            failed_evaluation_count=3,
        )

        updated = diagnostics.with_failed_attempts(
            failed_attempt_count=5,
            failed_evaluation_count=8,
        )

        assert updated.backend == "local-search"
        assert updated.failed_attempt_count == 5
        assert updated.failed_evaluation_count == 8
        assert diagnostics.failed_attempt_count == 2
        assert diagnostics.failed_evaluation_count == 3

    def test_kernel_diagnostics_rejects_invalid_failed_attempt_accounting(
        self,
    ) -> None:
        with pytest.raises(TypeError, match="failed_attempt_count must be int"):
            _ = KernelDiagnostics(
                backend="local-search",
                failed_attempt_count=True,
            )

        with pytest.raises(ValueError, match="failed_attempt_count must be non-negative"):
            _ = KernelDiagnostics(
                backend="local-search",
                failed_attempt_count=-1,
            )

        with pytest.raises(TypeError, match="failed_evaluation_count must be int"):
            _ = KernelDiagnostics(
                backend="local-search",
                failed_evaluation_count=False,
            )

        with pytest.raises(
            ValueError,
            match="failed_evaluation_count must be non-negative",
        ):
            _ = KernelDiagnostics(
                backend="local-search",
                failed_evaluation_count=-1,
            )

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

    def test_objective_vector_payload_constructor_normalizes_vectors_once(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        calls: list[str] = []
        original_normalize_objective_vector = (
            record_artifacts.normalize_objective_vector
        )

        def counting_normalize_objective_vector(
            *,
            values: Sequence[float],
            field_name: str,
        ) -> tuple[float, ...]:
            calls.append(field_name)
            return original_normalize_objective_vector(
                values=values,
                field_name=field_name,
            )

        monkeypatch.setattr(
            record_artifacts,
            "normalize_objective_vector",
            counting_normalize_objective_vector,
        )

        payload = ObjectiveVectorPayload(
            objective_values=(1.0, 2.0),
            objective_scores=(3.0, 4.0),
        )

        assert payload.objective_values == (1.0, 2.0)
        assert payload.objective_scores == (3.0, 4.0)
        assert calls == ["objective_values", "objective_scores"]

    def test_objective_vector_payload_normalizes_numeric_fields(self) -> None:
        payload = ObjectiveVectorPayload(
            objective_values=(
                cast(float, cast(object, np.int64(2))),
                cast(float, cast(object, np.float64(3.5))),
            ),
            objective_scores=(
                cast(float, cast(object, np.float64(2.0))),
                cast(float, cast(object, np.int64(-3))),
            ),
            elapsed_seconds=cast(float, cast(object, np.float64(0.3))),
        )

        assert payload.objective_values == (2.0, 3.5)
        assert all(type(value) is float for value in payload.objective_values)
        assert payload.objective_scores == (2.0, -3.0)
        assert all(type(score) is float for score in payload.objective_scores)
        assert payload.elapsed_seconds == 0.3
        assert type(payload.elapsed_seconds) is float

    def test_objective_vector_payload_rejects_bool_and_non_finite_numbers(
        self,
    ) -> None:
        with pytest.raises(TypeError, match="objective_values must be a real number"):
            _ = ObjectiveVectorPayload(
                objective_values=(cast(float, True),),
                objective_scores=(1.0,),
            )

        with pytest.raises(TypeError, match="objective_scores must be a real number"):
            _ = ObjectiveVectorPayload(
                objective_values=(1.0,),
                objective_scores=(cast(float, cast(object, np.bool_(True))),),
            )

        with pytest.raises(ValueError, match="objective_values must contain only finite"):
            _ = ObjectiveVectorPayload.from_objective_values(
                objective_values=(float("inf"),),
                directions=(OptimizationDirection.MINIMIZE,),
            )

        with pytest.raises(TypeError, match="elapsed_seconds must be a real number"):
            _ = ObjectiveVectorPayload(
                objective_values=(1.0,),
                objective_scores=(1.0,),
                elapsed_seconds=cast(float, False),
            )

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

    def test_evaluation_success_with_payload_preserves_candidate_equality(
        self,
    ) -> None:
        source_candidate = SpaceOwnedEqualityCandidate(1)
        request: EvaluationRequest[SpaceOwnedEqualityCandidate] = EvaluationRequest(
            proposal=Proposal(candidate=SpaceOwnedEqualityCandidate(2))
        )
        success: EvaluationSuccess[
            SpaceOwnedEqualityCandidate,
            ObservationPayload,
        ] = EvaluationSuccess(
            request=request,
            payload=make_observation_payload(value=1.0),
            refinement=CandidateRefinement(
                source_candidate=source_candidate,
                refined_candidate=SpaceOwnedEqualityCandidate(2),
            ),
            candidate_equal=space_owned_candidates_equal,
        )

        projected = success.with_payload(make_observation_payload(value=3.0))
        revalidated = replace(
            projected,
            refinement=CandidateRefinement(
                source_candidate=source_candidate,
                refined_candidate=SpaceOwnedEqualityCandidate(2),
            ),
        )

        assert projected.payload.value == 3.0
        assert revalidated.refinement is not projected.refinement

    def test_evaluation_success_with_kernel_diagnostics_preserves_candidate_equality(
        self,
    ) -> None:
        source_candidate = SpaceOwnedEqualityCandidate(1)
        request: EvaluationRequest[SpaceOwnedEqualityCandidate] = EvaluationRequest(
            proposal=Proposal(candidate=SpaceOwnedEqualityCandidate(2))
        )
        success: EvaluationSuccess[
            SpaceOwnedEqualityCandidate,
            ObservationPayload,
        ] = EvaluationSuccess(
            request=request,
            payload=make_observation_payload(value=1.0),
            refinement=CandidateRefinement(
                source_candidate=source_candidate,
                refined_candidate=SpaceOwnedEqualityCandidate(2),
            ),
            candidate_equal=space_owned_candidates_equal,
        )

        projected = success.with_kernel_diagnostics(
            KernelDiagnostics(
                backend="local-search",
                failed_attempt_count=1,
                failed_evaluation_count=1,
            )
        )
        revalidated = replace(
            projected,
            refinement=CandidateRefinement(
                source_candidate=source_candidate,
                refined_candidate=SpaceOwnedEqualityCandidate(2),
            ),
        )

        assert projected.kernel_diagnostics is not None
        assert projected.kernel_diagnostics.failed_attempt_count == 1
        assert revalidated.refinement is not projected.refinement

    def test_evaluation_success_pickle_preserves_refined_record_payload_cache(
        self,
    ) -> None:
        source_candidate = SpaceOwnedEqualityCandidate(1)
        refined_candidate = SpaceOwnedEqualityCandidate(2)
        source_request: EvaluationRequest[SpaceOwnedEqualityCandidate] = (
            EvaluationRequest(
                proposal=Proposal(candidate=source_candidate, proposal_id="p-1")
            )
        )
        refined_request: EvaluationRequest[SpaceOwnedEqualityCandidate] = (
            EvaluationRequest(
                proposal=Proposal(candidate=refined_candidate, proposal_id="p-1")
            )
        )
        record = SpaceOwnedRecord(
            request=source_request,
            candidate=refined_candidate,
            label="refined",
        )
        success: EvaluationSuccess[
            SpaceOwnedEqualityCandidate,
            SpaceOwnedRecord,
        ] = EvaluationSuccess(
            request=refined_request,
            payload=record,
            refinement=CandidateRefinement(
                source_candidate=source_candidate,
                refined_candidate=SpaceOwnedEqualityCandidate(2),
            ),
            candidate_equal=space_owned_candidates_equal,
        )

        restored = cast(
            EvaluationSuccess[SpaceOwnedEqualityCandidate, SpaceOwnedRecord],
            pickle.loads(pickle.dumps(success)),
        )

        assert restored.payload.request.candidate.stable_id == 1
        assert restored.payload.candidate.stable_id == 2

    def test_refined_record_payload_materializes_with_candidate_equality(
        self,
    ) -> None:
        source_request: EvaluationRequest[SpaceOwnedEqualityCandidate] = (
            EvaluationRequest(
                proposal=Proposal(
                    candidate=SpaceOwnedEqualityCandidate(1),
                    proposal_id="p-1",
                )
            )
        )
        refined_request: EvaluationRequest[SpaceOwnedEqualityCandidate] = (
            EvaluationRequest(
                proposal=Proposal(
                    candidate=SpaceOwnedEqualityCandidate(2),
                    proposal_id="p-1",
                )
            )
        )
        payload = SpaceOwnedRecord(
            request=source_request,
            candidate=refined_request.candidate,
            label="refined",
        )
        success: EvaluationSuccess[
            SpaceOwnedEqualityCandidate,
            SpaceOwnedRecord,
        ] = EvaluationSuccess(
            request=refined_request,
            payload=payload,
            refinement=CandidateRefinement(
                source_candidate=SpaceOwnedEqualityCandidate(1),
                refined_candidate=SpaceOwnedEqualityCandidate(2),
            ),
            candidate_equal=space_owned_candidates_equal,
        )

        materialized = materialize_success_record(success)

        assert materialized is payload

    def test_evaluation_success_with_payload_after_pickle_requires_candidate_equal(
        self,
    ) -> None:
        source_candidate = SpaceOwnedEqualityCandidate(1)
        refined_candidate = SpaceOwnedEqualityCandidate(2)
        request: EvaluationRequest[SpaceOwnedEqualityCandidate] = EvaluationRequest(
            proposal=Proposal(candidate=refined_candidate, proposal_id="p-1")
        )
        success: EvaluationSuccess[
            SpaceOwnedEqualityCandidate,
            ObservationPayload,
        ] = EvaluationSuccess(
            request=request,
            payload=make_observation_payload(value=1.0),
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
        refinement = restored.refinement
        assert refinement is not None
        source_request: EvaluationRequest[SpaceOwnedEqualityCandidate] = (
            EvaluationRequest(
                proposal=Proposal(
                    candidate=refinement.source_candidate,
                    proposal_id=restored.proposal_id,
                )
            )
        )
        record = SpaceOwnedRecord(
            request=source_request,
            candidate=restored.request.candidate,
            label="refined",
        )

        with pytest.raises(TypeError, match="candidate_equal is required"):
            _ = restored.with_payload(record)

        repaired = restored.with_payload(
            record,
            candidate_equal=space_owned_candidates_equal,
        )

        assert repaired.payload is record

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

    @pytest.mark.parametrize(
        ("value", "expected_error"),
        (
            (cast(float, True), TypeError),
            (cast(float, cast(object, np.bool_(True))), TypeError),
            (float("nan"), ValueError),
            (float("inf"), ValueError),
            (float("-inf"), ValueError),
        ),
    )
    def test_observation_factory_rejects_non_canonical_objective_value(
        self,
        value: float,
        expected_error: type[Exception],
    ) -> None:
        proposal = Proposal(candidate=4, proposal_id="p-1")

        with pytest.raises(expected_error, match="value must"):
            _ = Observation.from_objective_value(
                proposal=proposal,
                candidate=4,
                value=value,
                direction=OptimizationDirection.MINIMIZE,
            )

    def test_observation_factory_normalizes_numeric_fields(self) -> None:
        proposal = Proposal(candidate=4, proposal_id="p-1")

        observation = Observation.from_objective_value(
            proposal=proposal,
            candidate=4,
            value=cast(float, cast(object, np.int64(4))),
            direction=OptimizationDirection.MAXIMIZE,
            elapsed_seconds=cast(float, cast(object, np.float64(0.25))),
        )

        assert observation.value == 4.0
        assert type(observation.value) is float
        assert observation.score == -4.0
        assert type(observation.score) is float
        assert observation.elapsed_seconds == 0.25
        assert type(observation.elapsed_seconds) is float

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

    @pytest.mark.parametrize(
        ("elapsed_seconds", "expected_error"),
        (
            (cast(float, False), TypeError),
            (cast(float, cast(object, np.bool_(False))), TypeError),
            (float("nan"), ValueError),
            (float("inf"), ValueError),
            (float("-inf"), ValueError),
            (-0.1, ValueError),
        ),
    )
    def test_objective_vector_record_rejects_non_canonical_elapsed_seconds(
        self,
        elapsed_seconds: float,
        expected_error: type[Exception],
    ) -> None:
        proposal = Proposal(candidate=4, proposal_id="p-1")

        with pytest.raises(expected_error, match="elapsed_seconds must"):
            _ = ObjectiveVectorRecord(
                proposal=proposal,
                candidate=4,
                objective_values=(16.0,),
                objective_scores=(16.0,),
                elapsed_seconds=elapsed_seconds,
            )

    def test_objective_vector_record_normalizes_elapsed_seconds(self) -> None:
        proposal = Proposal(candidate=4, proposal_id="p-1")

        record = ObjectiveVectorRecord(
            proposal=proposal,
            candidate=4,
            objective_values=(16.0,),
            objective_scores=(16.0,),
            elapsed_seconds=cast(float, cast(object, np.float64(0.5))),
        )

        assert record.elapsed_seconds == 0.5
        assert type(record.elapsed_seconds) is float

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

    @pytest.mark.parametrize(
        "changed_leaf_paths",
        (
            cast(tuple[tuple[int | str, ...], ...], cast(object, "abc")),
            cast(tuple[tuple[int | str, ...], ...], cast(object, b"abc")),
            cast(
                tuple[tuple[int | str, ...], ...],
                cast(object, bytearray(b"abc")),
            ),
            ("abc",),
            (b"abc",),
            (bytearray(b"abc"),),
            cast(tuple[tuple[int | str, ...], ...], (["x", 1],)),
        ),
    )
    def test_candidate_refinement_rejects_malformed_path_containers(
        self,
        changed_leaf_paths: tuple[tuple[int | str, ...], ...],
    ) -> None:
        with pytest.raises(TypeError, match="changed_leaf_paths"):
            _ = CandidateRefinement(
                source_candidate=(1, 2),
                refined_candidate=(3, 2),
                changed_leaf_paths=changed_leaf_paths,
            )

    def test_candidate_refinement_accepts_string_path_segments(self) -> None:
        refinement = CandidateRefinement(
            source_candidate={"x": (1, 2)},
            refined_candidate={"x": (1, 3)},
            changed_leaf_paths=(("x", 1),),
        )

        assert refinement.changed_leaf_paths == (("x", 1),)

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

    def test_exception_snapshot_setstate_rejects_invalid_state(self) -> None:
        snapshot = object.__new__(EvaluationExceptionSnapshot)
        pickle_hooks = exception_snapshot_pickle_hooks(snapshot)

        with pytest.raises(ValueError, match="exception_module"):
            pickle_hooks.__setstate__(("", "ValueError", "bad"))

    def test_evaluation_failure_setstate_rejects_invalid_state(self) -> None:
        failure = cast(EvaluationFailure[int], object.__new__(EvaluationFailure))
        snapshot = EvaluationExceptionSnapshot.from_exception(ValueError("bad"))
        pickle_hooks = evaluation_failure_pickle_hooks(failure)

        with pytest.raises(ValueError, match="evaluation_count must be non-negative"):
            pickle_hooks.__setstate__((make_int_request(4, "p-1"), snapshot, -1))

    def test_evaluation_failure_setstate_rejects_invalid_nested_exception(
        self,
    ) -> None:
        failure = cast(EvaluationFailure[int], object.__new__(EvaluationFailure))
        invalid_snapshot = object.__new__(EvaluationExceptionSnapshot)
        object.__setattr__(invalid_snapshot, "exception_module", "")
        object.__setattr__(invalid_snapshot, "exception_qualname", "ValueError")
        object.__setattr__(invalid_snapshot, "message", "bad")
        pickle_hooks = evaluation_failure_pickle_hooks(failure)

        with pytest.raises(ValueError, match="exception_module"):
            pickle_hooks.__setstate__(
                (make_int_request(4, "p-1"), invalid_snapshot, 1),
            )

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

        compared_field_names = tuple(
            field.name for field in fields(batch) if field.compare
        )
        assert compared_field_names == ("attempts",)
        assert batch.attempts == (success, failure)
        assert batch.requests == (request_one, request_two)
        assert batch.successes == (success,)
        assert batch.failures == (failure,)
        assert batch.success_indices == (0,)
        assert batch.failure_indices == (1,)

    def test_evaluation_attempt_batch_reuses_derived_tuple_views(self) -> None:
        request_one = make_int_request(1, "p-1")
        request_two = make_int_request(2, "p-2")
        success = make_int_success(request_one)
        failure = make_int_failure(request_two)
        batch: EvaluationAttemptBatch[int, ObservationPayload] = EvaluationAttemptBatch(
            attempts=(success, failure),
        )

        assert batch.requests is batch.requests
        assert batch.successes is batch.successes
        assert batch.failures is batch.failures
        assert batch.success_indices is batch.success_indices
        assert batch.failure_indices is batch.failure_indices
        assert batch.payloads is batch.payloads
        assert batch.evaluation_count == 2
        assert batch.evaluation_count == 2
        assert batch.has_failures is True
        assert batch.has_failures is True

    def test_evaluation_attempt_batch_cached_views_do_not_affect_equality(
        self,
    ) -> None:
        request_one = make_int_request(1, "p-1")
        request_two = make_int_request(2, "p-2")
        success = make_int_success(request_one)
        failure = make_int_failure(request_two)
        cached: EvaluationAttemptBatch[int, ObservationPayload] = (
            EvaluationAttemptBatch(attempts=(success, failure))
        )
        uncached: EvaluationAttemptBatch[int, ObservationPayload] = (
            EvaluationAttemptBatch(attempts=(success, failure))
        )

        _ = cached.requests
        _ = cached.successes
        _ = cached.failures
        _ = cached.success_indices
        _ = cached.failure_indices
        _ = cached.payloads
        _ = cached.evaluation_count
        _ = cached.has_failures

        assert cached == uncached

    def test_evaluation_attempt_batch_pickle_preserves_cached_views(self) -> None:
        request_one = make_int_request(1, "p-1")
        request_two = make_int_request(2, "p-2")
        success = make_int_success(request_one)
        failure = make_int_failure(request_two)
        batch: EvaluationAttemptBatch[int, ObservationPayload] = EvaluationAttemptBatch(
            attempts=(success, failure),
        )
        _ = batch.requests
        _ = batch.successes
        _ = batch.failures
        _ = batch.success_indices
        _ = batch.failure_indices
        _ = batch.payloads
        _ = batch.evaluation_count
        _ = batch.has_failures

        restored = cast(
            EvaluationAttemptBatch[int, ObservationPayload],
            pickle.loads(pickle.dumps(batch)),
        )

        assert restored.attempts == (success, failure)
        assert restored.requests == (request_one, request_two)
        assert restored.requests is restored.requests
        assert restored.successes == (success,)
        assert restored.successes is restored.successes
        assert restored.failures == (failure,)
        assert restored.failures is restored.failures
        assert restored.success_indices == (0,)
        assert restored.success_indices is restored.success_indices
        assert restored.failure_indices == (1,)
        assert restored.failure_indices is restored.failure_indices
        assert restored.payloads == (success.payload,)
        assert restored.payloads is restored.payloads
        assert restored.evaluation_count == 2
        assert restored.has_failures is True

    def test_evaluation_attempt_batch_accepts_canonical_attempt_slots(self) -> None:
        request_one = make_int_request(1, "p-1")
        request_two = make_int_request(2, "p-2")
        success = make_int_success(request_one)
        failure = make_int_failure(request_two)

        batch: EvaluationAttemptBatch[int, ObservationPayload] = EvaluationAttemptBatch(
            attempts=(success, failure),
        )
        _ = batch.requests
        _ = batch.successes
        _ = batch.failures
        _ = batch.success_indices
        _ = batch.failure_indices
        _ = batch.payloads
        _ = batch.evaluation_count
        _ = batch.has_failures

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
            batch = EvaluationAttemptBatch[int, ObservationPayload](attempts=())
            object.__setattr__(batch, "attempts", (payload,))
            batch.__post_init__()

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

    def test_evaluation_success_scalar_observation_projects_refinement_source(
        self,
    ) -> None:
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

        observation = success.scalar_observation()
        record = materialize_success_record(success)

        assert type(record) is Observation
        assert observation == record
        assert observation.proposal.candidate == 4
        assert observation.candidate == 2

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

    def test_materialize_success_record_reuses_aligned_observation_payload(
        self,
    ) -> None:
        request = make_int_request(candidate=7, proposal_id="p-7")
        record = Observation.from_objective_value(
            request=request,
            candidate=request.candidate,
            value=7.0,
            direction=OptimizationDirection.MINIMIZE,
        )
        success: EvaluationSuccess[int, Observation[int]] = EvaluationSuccess(
            request=request,
            payload=record,
        )

        assert materialize_success_record(success) is record

    def test_materialize_success_record_rejects_attribute_bag_payload(self) -> None:
        request = make_int_request(candidate=7, proposal_id="p-7")

        class AttributeBagPayload:
            request: str
            candidate: int

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

    def test_materialize_attempt_batch_records_supports_empty_batch(self) -> None:
        batch: EvaluationAttemptBatch[int, ObservationPayload] = EvaluationAttemptBatch(
            attempts=(),
        )

        materialized = materialize_attempt_batch_records(batch)

        assert materialized.requests == ()
        assert materialized.successes == ()
        assert materialized.failures == ()
        assert materialized.evaluation_count == 0

    def test_materialize_attempt_batch_records_preserves_all_failure_batch(self) -> None:
        request_one = make_int_request(candidate=1, proposal_id="p-1")
        request_two = make_int_request(candidate=2, proposal_id="p-2")
        failure_one = make_int_failure(request_one, evaluation_count=2)
        failure_two = make_int_failure(request_two, evaluation_count=3)
        batch: EvaluationAttemptBatch[int, ObservationPayload] = EvaluationAttemptBatch(
            attempts=(failure_one, failure_two),
        )

        materialized = materialize_attempt_batch_records(batch)

        assert materialized.requests == (request_one, request_two)
        assert materialized.successes == ()
        assert materialized.failures == (failure_one, failure_two)
        assert materialized.evaluation_count == 5

    def test_materialize_attempt_batch_records_projects_vector_payload(self) -> None:
        request = make_int_request(candidate=2, proposal_id="p-2")
        refinement = CandidateRefinement(
            source_candidate=4,
            refined_candidate=2,
            changed_leaf_paths=((),),
        )
        payload = ObjectiveVectorPayload.from_objective_values(
            objective_values=(3.0, 5.0),
            directions=(
                OptimizationDirection.MINIMIZE,
                OptimizationDirection.MAXIMIZE,
            ),
        )
        success: EvaluationSuccess[int, ObjectiveVectorPayload] = EvaluationSuccess(
            request=request,
            payload=payload,
            refinement=refinement,
        )
        batch: EvaluationAttemptBatch[int, ObjectiveVectorPayload] = (
            EvaluationAttemptBatch(attempts=(success,))
        )

        materialized = materialize_attempt_batch_records(batch)
        record = materialized.successes[0].payload

        assert type(record) is ObjectiveVectorRecord
        assert record.candidate == 2
        assert record.proposal.candidate == 4
        assert record.proposal.proposal_id == "p-2"
        assert record.objective_values == (3.0, 5.0)
        assert record.objective_scores == (3.0, -5.0)

    def test_materialize_attempt_batch_records_preserves_record_payload(self) -> None:
        request_one = make_int_request(candidate=5, proposal_id="p-5")
        request_two = make_int_request(candidate=6, proposal_id="p-6")
        record = LabelRecord(
            request=request_one,
            candidate=request_one.candidate,
            label="five",
        )
        success: EvaluationSuccess[int, LabelRecord] = EvaluationSuccess(
            request=request_one,
            payload=record,
        )
        failure = make_int_failure(request_two, evaluation_count=3)
        batch: EvaluationAttemptBatch[int, LabelRecord] = EvaluationAttemptBatch(
            attempts=(success, failure),
        )

        materialized = materialize_attempt_batch_records(batch)

        assert materialized.requests == (request_one, request_two)
        assert materialized.successes[0].payload is record
        assert materialized.failures == (failure,)
        assert materialized.evaluation_count == 4

    def test_materialize_attempt_batch_records_preserves_builtin_record_payload(
        self,
    ) -> None:
        request = make_int_request(candidate=5, proposal_id="p-5")
        record = Observation.from_objective_value(
            request=request,
            candidate=request.candidate,
            value=5.0,
            direction=OptimizationDirection.MINIMIZE,
        )
        success: EvaluationSuccess[int, Observation[int]] = EvaluationSuccess(
            request=request,
            payload=record,
        )
        batch: EvaluationAttemptBatch[int, Observation[int]] = EvaluationAttemptBatch(
            attempts=(success,),
        )

        materialized = materialize_attempt_batch_records(batch)

        assert materialized.successes[0].payload is record

    def test_materialize_attempt_batch_records_preserves_vector_record_payload(
        self,
    ) -> None:
        request = make_int_request(candidate=5, proposal_id="p-5")
        record = ObjectiveVectorRecord(
            request=request,
            candidate=request.candidate,
            objective_values=(1.0, 2.0),
            objective_scores=(1.0, -2.0),
        )
        success: EvaluationSuccess[int, ObjectiveVectorRecord[int]] = (
            EvaluationSuccess(
                request=request,
                payload=record,
            )
        )
        batch: EvaluationAttemptBatch[int, ObjectiveVectorRecord[int]] = (
            EvaluationAttemptBatch(
                attempts=(success,),
            )
        )

        materialized = materialize_attempt_batch_records(batch)

        assert materialized.successes[0].payload is record

    def test_materialize_attempt_batch_records_preserves_refined_record_payload(
        self,
    ) -> None:
        source_candidate = 4000
        refined_candidate = 2000
        source_request = make_int_request(candidate=source_candidate, proposal_id="p-x")
        refined_request = make_int_request(
            candidate=refined_candidate,
            proposal_id="p-x",
        )
        refinement = CandidateRefinement(
            source_candidate=source_candidate,
            refined_candidate=refined_candidate,
            changed_leaf_paths=((),),
        )
        record = LabelRecord(
            request=source_request,
            candidate=refined_candidate,
            label="refined",
        )
        success: EvaluationSuccess[int, LabelRecord] = EvaluationSuccess(
            request=refined_request,
            payload=record,
            refinement=refinement,
        )
        batch: EvaluationAttemptBatch[int, LabelRecord] = EvaluationAttemptBatch(
            attempts=(success,),
        )

        materialized = materialize_attempt_batch_records(batch)

        assert materialized.successes[0].payload is record

    def test_materialize_attempt_batch_records_rejects_mismatched_record_request(
        self,
    ) -> None:
        request = make_int_request(candidate=5, proposal_id="p-shared")
        payload_request = make_int_request(candidate=6, proposal_id="p-shared")
        record = LabelRecord(
            request=payload_request,
            candidate=request.candidate,
            label="wrong-request",
        )
        success: EvaluationSuccess[int, LabelRecord] = EvaluationSuccess(
            request=request,
            payload=record,
        )
        batch: EvaluationAttemptBatch[int, LabelRecord] = EvaluationAttemptBatch(
            attempts=(success,),
        )

        with pytest.raises(TypeError, match="cannot be materialized"):
            _ = materialize_attempt_batch_records(batch)

    def test_materialize_attempt_batch_records_rejects_refined_request_record(
        self,
    ) -> None:
        source_candidate = 4000
        refined_candidate = 2000
        refined_request = make_int_request(
            candidate=refined_candidate,
            proposal_id="p-refined",
        )
        refinement = CandidateRefinement(
            source_candidate=source_candidate,
            refined_candidate=refined_candidate,
            changed_leaf_paths=((),),
        )
        record = LabelRecord(
            request=refined_request,
            candidate=refined_candidate,
            label="refined-request",
        )
        success: EvaluationSuccess[int, LabelRecord] = EvaluationSuccess(
            request=refined_request,
            payload=record,
            refinement=refinement,
        )
        batch: EvaluationAttemptBatch[int, LabelRecord] = EvaluationAttemptBatch(
            attempts=(success,),
        )

        with pytest.raises(TypeError, match="cannot be materialized"):
            _ = materialize_attempt_batch_records(batch)

    def test_materialize_attempt_batch_records_does_not_compare_record_candidates(
        self,
    ) -> None:
        @dataclass(frozen=True, slots=True)
        class ObjectRecord:
            request: EvaluationRequest[SpaceOwnedEqualityCandidate]
            candidate: SpaceOwnedEqualityCandidate
            label: str

        candidate = SpaceOwnedEqualityCandidate(stable_id=1)
        request = EvaluationRequest(
            proposal=Proposal(candidate=candidate, proposal_id="p-object"),
        )
        record = ObjectRecord(
            request=request,
            candidate=candidate,
            label="object",
        )
        success: EvaluationSuccess[
            SpaceOwnedEqualityCandidate,
            ObjectRecord,
        ] = EvaluationSuccess(
            request=request,
            payload=record,
        )
        batch: EvaluationAttemptBatch[
            SpaceOwnedEqualityCandidate,
            ObjectRecord,
        ] = EvaluationAttemptBatch(attempts=(success,))

        materialized = materialize_attempt_batch_records(batch)

        assert materialized.successes[0].payload is record

    def test_evaluation_success_rejects_stale_pre_materialized_observation(
        self,
    ) -> None:
        stale_request = make_int_request(candidate=99, proposal_id="stale")
        request = make_int_request(candidate=2, proposal_id="p-2")
        payload = Observation.from_objective_value(
            request=stale_request,
            candidate=99,
            value=12.0,
            direction=OptimizationDirection.MINIMIZE,
        )

        with pytest.raises(ValueError, match="payload candidate"):
            _ = EvaluationSuccess(
                request=request,
                payload=payload,
            )

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

    def test_observation_normalizes_numeric_fields_to_builtin_float(self) -> None:
        proposal = Proposal(candidate=4, proposal_id="p-1")

        observation = Observation(
            proposal=proposal,
            candidate=4,
            value=np.float64(16.0),
            score=FloatSubclass(9.0),
            elapsed_seconds=cast(float, cast(object, np.int64(1))),
        )

        assert observation.value == 16.0
        assert type(observation.value) is float
        assert observation.score == 9.0
        assert type(observation.score) is float
        assert observation.elapsed_seconds == 1.0
        assert type(observation.elapsed_seconds) is float

    def test_observation_normalizes_int_fields_to_builtin_float(self) -> None:
        proposal = Proposal(candidate=4, proposal_id="p-1")

        observation = Observation(
            proposal=proposal,
            candidate=4,
            value=16,
            score=9,
            elapsed_seconds=1,
        )

        assert observation.value == 16.0
        assert type(observation.value) is float
        assert observation.score == 9.0
        assert type(observation.score) is float
        assert observation.elapsed_seconds == 1.0
        assert type(observation.elapsed_seconds) is float

    def test_observation_rejects_bool_value(self) -> None:
        proposal = Proposal(candidate=4, proposal_id="p-1")

        with pytest.raises(TypeError, match="value must be a real number"):
            _ = Observation(
                proposal=proposal,
                candidate=4,
                value=cast(float, True),
                score=16.0,
            )

    def test_observation_rejects_bool_score(self) -> None:
        proposal = Proposal(candidate=4, proposal_id="p-1")

        with pytest.raises(TypeError, match="score must be a real number"):
            _ = Observation(
                proposal=proposal,
                candidate=4,
                value=16.0,
                score=cast(float, cast(object, np.bool_(True))),
            )

    def test_observation_rejects_bool_elapsed_seconds(self) -> None:
        proposal = Proposal(candidate=4, proposal_id="p-1")

        with pytest.raises(TypeError, match="elapsed_seconds must be a real number"):
            _ = Observation(
                proposal=proposal,
                candidate=4,
                value=16.0,
                score=16.0,
                elapsed_seconds=cast(float, True),
            )

    @pytest.mark.parametrize("elapsed_seconds", (float("nan"), float("inf")))
    def test_observation_rejects_non_finite_elapsed_seconds(
        self,
        elapsed_seconds: float,
    ) -> None:
        proposal = Proposal(candidate=4, proposal_id="p-1")

        with pytest.raises(ValueError, match="elapsed_seconds must be finite"):
            _ = Observation(
                proposal=proposal,
                candidate=4,
                value=16.0,
                score=16.0,
                elapsed_seconds=elapsed_seconds,
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
        vector_report = RunReport[int, ObjectiveVectorRecord[int]].from_successes(
            successes=(make_vector_record_success(vector_record),),
            failures=(failure,),
        )
        surface_from_report = NondominatedRunSurface[int].from_report(
            vector_report,
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

    def test_terminal_surface_setstate_rejects_legacy_short_state(self) -> None:
        observation = Observation(
            proposal=Proposal(candidate=4, proposal_id="p-1"),
            candidate=4,
            value=16.0,
            score=16.0,
        )
        result = RunResult[int].from_observations((observation,))
        pickle_hooks = terminal_surface_pickle_hooks(result)
        current_state = pickle_hooks.__getstate__()

        with pytest.raises(TypeError, match="field count mismatch"):
            pickle_hooks.__setstate__(current_state[:-1])

    def test_terminal_surface_setstate_rejects_future_long_state(self) -> None:
        observation = Observation(
            proposal=Proposal(candidate=4, proposal_id="p-1"),
            candidate=4,
            value=16.0,
            score=16.0,
        )
        result = RunResult[int].from_observations((observation,))
        pickle_hooks = terminal_surface_pickle_hooks(result)
        current_state = pickle_hooks.__getstate__()

        with pytest.raises(TypeError, match="field count mismatch"):
            pickle_hooks.__setstate__(current_state + ["future-field"])

    def test_terminal_surface_setstate_rejects_invalid_nested_failure(self) -> None:
        observation = Observation(
            proposal=Proposal(candidate=4, proposal_id="p-1"),
            candidate=4,
            value=16.0,
            score=16.0,
        )
        result = RunResult[int].from_observations((observation,))
        invalid_failure = cast(EvaluationFailure[int], object.__new__(EvaluationFailure))
        object.__setattr__(invalid_failure, "request", make_int_request(2, "p-2"))
        object.__setattr__(
            invalid_failure,
            "exception",
            EvaluationExceptionSnapshot.from_exception(ValueError("bad candidate")),
        )
        object.__setattr__(invalid_failure, "evaluation_count", -1)
        pickle_state = terminal_surface_pickle_state_with_field(
            result,
            field_name="failures",
            value=(invalid_failure,),
        )
        pickle_hooks = terminal_surface_pickle_hooks(result)

        with pytest.raises(ValueError, match="evaluation_count must be non-negative"):
            pickle_hooks.__setstate__(pickle_state)

    def test_terminal_surface_setstate_rejects_invalid_nested_failure_exception(
        self,
    ) -> None:
        observation = Observation(
            proposal=Proposal(candidate=4, proposal_id="p-1"),
            candidate=4,
            value=16.0,
            score=16.0,
        )
        result = RunResult[int].from_observations((observation,))
        invalid_snapshot = object.__new__(EvaluationExceptionSnapshot)
        object.__setattr__(invalid_snapshot, "exception_module", "")
        object.__setattr__(invalid_snapshot, "exception_qualname", "ValueError")
        object.__setattr__(invalid_snapshot, "message", "bad candidate")
        invalid_failure = cast(EvaluationFailure[int], object.__new__(EvaluationFailure))
        object.__setattr__(invalid_failure, "request", make_int_request(2, "p-2"))
        object.__setattr__(invalid_failure, "exception", invalid_snapshot)
        object.__setattr__(invalid_failure, "evaluation_count", 1)
        pickle_state = terminal_surface_pickle_state_with_field(
            result,
            field_name="failures",
            value=(invalid_failure,),
        )
        pickle_hooks = terminal_surface_pickle_hooks(result)

        with pytest.raises(ValueError, match="exception_module"):
            pickle_hooks.__setstate__(pickle_state)

    def test_terminal_surface_pickle_state_matches_current_field_shape(self) -> None:
        proposal = Proposal(candidate=4, proposal_id="p-1")
        observation = Observation(
            proposal=proposal,
            candidate=4,
            value=16.0,
            score=16.0,
        )
        vector_record = ObjectiveVectorRecord.from_objective_values(
            proposal=proposal,
            candidate=4,
            objective_values=(16.0,),
            directions=(OptimizationDirection.MINIMIZE,),
        )
        surfaces = (
            RunResult[int].from_observations((observation,)),
            RunReport[int, Observation[int]].from_records((observation,)),
            NondominatedRunSurface[int].from_records((vector_record,)),
        )

        for surface in surfaces:
            pickle_hooks = terminal_surface_pickle_hooks(surface)
            assert len(pickle_hooks.__getstate__()) == len(fields(surface))

    def test_terminal_surface_setstate_rejects_negative_evaluation_count(self) -> None:
        proposal = Proposal(candidate=4, proposal_id="p-1")
        observation = Observation(
            proposal=proposal,
            candidate=4,
            value=16.0,
            score=16.0,
        )
        vector_record = ObjectiveVectorRecord.from_objective_values(
            proposal=proposal,
            candidate=4,
            objective_values=(16.0,),
            directions=(OptimizationDirection.MINIMIZE,),
        )
        surfaces = (
            RunResult[int].from_observations((observation,)),
            RunReport[int, Observation[int]].from_records((observation,)),
            NondominatedRunSurface[int].from_records((vector_record,)),
        )

        for surface in surfaces:
            pickle_hooks = terminal_surface_pickle_hooks(surface)
            invalid_state = terminal_surface_pickle_state_with_field(
                surface,
                field_name="evaluation_count",
                value=-1,
            )

            with pytest.raises(ValueError, match="evaluation_count"):
                pickle_hooks.__setstate__(invalid_state)

    def test_terminal_surface_setstate_rejects_attempt_cost_underflow(self) -> None:
        proposal = Proposal(candidate=4, proposal_id="p-1")
        observation = Observation(
            proposal=proposal,
            candidate=4,
            value=16.0,
            score=16.0,
        )
        vector_record = ObjectiveVectorRecord.from_objective_values(
            proposal=proposal,
            candidate=4,
            objective_values=(16.0,),
            directions=(OptimizationDirection.MINIMIZE,),
        )
        surfaces = (
            RunResult[int].from_observations((observation,)),
            RunReport[int, Observation[int]].from_records((observation,)),
            NondominatedRunSurface[int].from_records((vector_record,)),
        )

        for surface in surfaces:
            pickle_hooks = terminal_surface_pickle_hooks(surface)
            invalid_state = terminal_surface_pickle_state_with_field(
                surface,
                field_name="evaluation_count",
                value=0,
            )

            with pytest.raises(ValueError, match="terminal attempt cost"):
                pickle_hooks.__setstate__(invalid_state)

    def test_run_result_setstate_rejects_foreign_best_success(self) -> None:
        request_one = make_int_request(4, "p-1")
        request_two = make_int_request(2, "p-2")
        result = RunResult[int].from_successes(
            successes=(make_int_success(request_one),),
        )
        foreign_best_success = make_int_success(request_two)
        pickle_hooks = terminal_surface_pickle_hooks(result)
        invalid_state = terminal_surface_pickle_state_with_field(
            result,
            field_name="best_success",
            value=foreign_best_success,
        )

        with pytest.raises(ValueError, match="best_success must come from successes"):
            pickle_hooks.__setstate__(invalid_state)

    def test_nondominated_surface_setstate_rejects_inconsistent_frontier(
        self,
    ) -> None:
        frontier_record = ObjectiveVectorRecord.from_objective_values(
            proposal=Proposal(candidate=1, proposal_id="p-1"),
            candidate=1,
            objective_values=(1.0, 1.0),
            directions=(
                OptimizationDirection.MINIMIZE,
                OptimizationDirection.MINIMIZE,
            ),
        )
        dominated_record = ObjectiveVectorRecord.from_objective_values(
            proposal=Proposal(candidate=2, proposal_id="p-2"),
            candidate=2,
            objective_values=(2.0, 2.0),
            directions=(
                OptimizationDirection.MINIMIZE,
                OptimizationDirection.MINIMIZE,
            ),
        )
        surface = NondominatedRunSurface[int].from_records(
            (frontier_record, dominated_record)
        )
        dominated_success = make_vector_record_success(dominated_record)
        pickle_hooks = terminal_surface_pickle_hooks(surface)
        invalid_state = terminal_surface_pickle_state_with_field(
            surface,
            field_name="nondominated_successes",
            value=(dominated_success,),
        )

        with pytest.raises(ValueError, match="stable nondominated frontier"):
            pickle_hooks.__setstate__(invalid_state)

    def test_nondominated_surface_setstate_does_not_trust_prevalidated_cache(
        self,
    ) -> None:
        frontier_record = ObjectiveVectorRecord.from_objective_values(
            proposal=Proposal(candidate=1, proposal_id="p-1"),
            candidate=1,
            objective_values=(1.0, 1.0),
            directions=(
                OptimizationDirection.MINIMIZE,
                OptimizationDirection.MINIMIZE,
            ),
        )
        dominated_record = ObjectiveVectorRecord.from_objective_values(
            proposal=Proposal(candidate=2, proposal_id="p-2"),
            candidate=2,
            objective_values=(2.0, 2.0),
            directions=(
                OptimizationDirection.MINIMIZE,
                OptimizationDirection.MINIMIZE,
            ),
        )
        surface = NondominatedRunSurface[int].from_records(
            (frontier_record, dominated_record)
        )
        invalid_frontier = (surface.successes[1],)
        pickle_hooks = terminal_surface_pickle_hooks(surface)
        invalid_state = pickle_hooks.__getstate__()
        field_names = tuple(dataclass_field.name for dataclass_field in fields(surface))
        invalid_state[field_names.index("nondominated_successes")] = invalid_frontier
        invalid_state[field_names.index("_validated_frontier_source_successes")] = (
            surface.successes
        )
        invalid_state[field_names.index("_validated_frontier_successes")] = (
            invalid_frontier
        )

        with pytest.raises(ValueError, match="stable nondominated frontier"):
            pickle_hooks.__setstate__(invalid_state)

    def test_terminal_surface_setstate_rejects_invalid_success_element(self) -> None:
        proposal = Proposal(candidate=4, proposal_id="p-1")
        observation = Observation(
            proposal=proposal,
            candidate=4,
            value=16.0,
            score=16.0,
        )
        vector_record = ObjectiveVectorRecord.from_objective_values(
            proposal=proposal,
            candidate=4,
            objective_values=(16.0,),
            directions=(OptimizationDirection.MINIMIZE,),
        )
        surfaces = (
            RunResult[int].from_observations((observation,)),
            RunReport[int, Observation[int]].from_records((observation,)),
            NondominatedRunSurface[int].from_records((vector_record,)),
        )

        for surface in surfaces:
            pickle_hooks = terminal_surface_pickle_hooks(surface)
            invalid_state = terminal_surface_pickle_state_with_field(
                surface,
                field_name="successes",
                value=("not-a-success",),
            )

            with pytest.raises(TypeError, match="EvaluationSuccess"):
                pickle_hooks.__setstate__(invalid_state)

    def test_terminal_surface_setstate_rejects_invalid_failure_element(self) -> None:
        proposal = Proposal(candidate=4, proposal_id="p-1")
        observation = Observation(
            proposal=proposal,
            candidate=4,
            value=16.0,
            score=16.0,
        )
        vector_record = ObjectiveVectorRecord.from_objective_values(
            proposal=proposal,
            candidate=4,
            objective_values=(16.0,),
            directions=(OptimizationDirection.MINIMIZE,),
        )
        surfaces = (
            RunResult[int].from_observations((observation,)),
            RunReport[int, Observation[int]].from_records((observation,)),
            NondominatedRunSurface[int].from_records((vector_record,)),
        )

        for surface in surfaces:
            pickle_hooks = terminal_surface_pickle_hooks(surface)
            invalid_state = terminal_surface_pickle_state_with_field(
                surface,
                field_name="failures",
                value=("not-a-failure",),
            )

            with pytest.raises(TypeError, match="EvaluationFailure"):
                pickle_hooks.__setstate__(invalid_state)

    def test_nondominated_surface_setstate_rejects_mixed_objective_dimensions(
        self,
    ) -> None:
        record_two_dimensional = ObjectiveVectorRecord.from_objective_values(
            proposal=Proposal(candidate=1, proposal_id="p-1"),
            candidate=1,
            objective_values=(1.0, 1.0),
            directions=(
                OptimizationDirection.MINIMIZE,
                OptimizationDirection.MINIMIZE,
            ),
        )
        record_three_dimensional = ObjectiveVectorRecord.from_objective_values(
            proposal=Proposal(candidate=2, proposal_id="p-2"),
            candidate=2,
            objective_values=(2.0, 2.0, 2.0),
            directions=(
                OptimizationDirection.MINIMIZE,
                OptimizationDirection.MINIMIZE,
                OptimizationDirection.MINIMIZE,
            ),
        )
        surface = NondominatedRunSurface[int].from_records((record_two_dimensional,))
        mixed_successes = surface.successes + (
            make_vector_record_success(record_three_dimensional),
        )
        pickle_hooks = terminal_surface_pickle_hooks(surface)
        invalid_state = terminal_surface_pickle_state_with_field(
            surface,
            field_name="successes",
            value=mixed_successes,
        )
        field_names = tuple(dataclass_field.name for dataclass_field in fields(surface))
        invalid_state[field_names.index("evaluation_count")] = len(mixed_successes)

        with pytest.raises(ValueError, match="objective score vectors"):
            pickle_hooks.__setstate__(invalid_state)

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

    def test_run_report_rejects_divergent_record_without_refinement(self) -> None:
        proposal = Proposal(candidate=4, proposal_id="p-1")
        record = LabelRecord(
            request=EvaluationRequest(proposal=proposal),
            candidate=3,
            label="parity:1",
        )

        with pytest.raises(ValueError, match="refinement is required"):
            _ = RunReport[int, LabelRecord].from_records(records=(record,))

    def test_run_report_requires_candidate_equal_for_non_scalar_unrefined_records(
        self,
    ) -> None:
        source_candidate = SpaceOwnedEqualityCandidate(1)
        evaluated_candidate = SpaceOwnedEqualityCandidate(1)
        record = Observation(
            proposal=Proposal(candidate=source_candidate, proposal_id="p-1"),
            candidate=evaluated_candidate,
            value=1.0,
            score=1.0,
        )

        with pytest.raises(TypeError, match="candidate equality"):
            _ = RunReport[
                SpaceOwnedEqualityCandidate,
                Observation[SpaceOwnedEqualityCandidate],
            ].from_records(records=(record,))

    def test_run_report_accepts_semantically_equal_unrefined_records(
        self,
    ) -> None:
        source_candidate = SpaceOwnedEqualityCandidate(1)
        evaluated_candidate = SpaceOwnedEqualityCandidate(1)
        record = Observation(
            proposal=Proposal(candidate=source_candidate, proposal_id="p-1"),
            candidate=evaluated_candidate,
            value=1.0,
            score=1.0,
        )

        report = RunReport[
            SpaceOwnedEqualityCandidate,
            Observation[SpaceOwnedEqualityCandidate],
        ].from_records(
            records=(record,),
            candidate_equal=space_owned_candidates_equal,
        )

        projected_record = report.records[0]
        assert projected_record.candidate is evaluated_candidate
        assert projected_record.request.candidate is evaluated_candidate
        assert projected_record.request.proposal_id == "p-1"
        assert report.refinements == ()

    def test_run_result_rejects_divergent_observation_without_refinement(self) -> None:
        observation = Observation(
            proposal=Proposal(candidate=4, proposal_id="p-1"),
            candidate=2,
            value=4.0,
            score=4.0,
        )

        with pytest.raises(ValueError, match="refinement is required"):
            _ = RunResult[int].from_observations(observations=(observation,))

    def test_nondominated_surface_rejects_divergent_record_without_refinement(
        self,
    ) -> None:
        record = ObjectiveVectorRecord.from_objective_values(
            proposal=Proposal(candidate=4, proposal_id="p-1"),
            candidate=2,
            objective_values=(4.0,),
            directions=(OptimizationDirection.MINIMIZE,),
        )

        with pytest.raises(ValueError, match="refinement is required"):
            _ = NondominatedRunSurface[int].from_records(records=(record,))

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
        candidate = SpaceOwnedEqualityCandidate(1)
        proposal = Proposal(
            candidate=candidate,
            proposal_id="p-1",
        )
        record = Observation(
            proposal=proposal,
            candidate=candidate,
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

    def test_run_result_explicit_best_observation_uses_candidate_equality(
        self,
    ) -> None:
        worse_observation: Observation[SpaceOwnedEqualityCandidate] = Observation(
            proposal=Proposal(
                candidate=SpaceOwnedEqualityCandidate(3),
                proposal_id="p-worse",
            ),
            candidate=SpaceOwnedEqualityCandidate(3),
            value=9.0,
            score=9.0,
        )
        best_observation: Observation[SpaceOwnedEqualityCandidate] = Observation(
            proposal=Proposal(
                candidate=SpaceOwnedEqualityCandidate(2),
                proposal_id="p-best",
            ),
            candidate=SpaceOwnedEqualityCandidate(1),
            value=1.0,
            score=1.0,
        )
        refinement = CandidateRefinement(
            source_candidate=SpaceOwnedEqualityCandidate(2),
            refined_candidate=SpaceOwnedEqualityCandidate(1),
            changed_leaf_paths=((),),
        )

        result = RunResult[SpaceOwnedEqualityCandidate](
            best_observation=best_observation,
            observations=(worse_observation, best_observation),
            refinements=(None, refinement),
            evaluation_count=2,
            candidate_equal=space_owned_candidates_equal,
        )

        assert result.best_success is result.successes[1]
        best_success = result.best_success
        assert best_success is not None
        assert best_success.refinement is refinement
        assert result.best_observation is not None
        assert result.best_observation.value == 1.0

    def test_nondominated_explicit_records_use_candidate_equality(
        self,
    ) -> None:
        dominated_record = ObjectiveVectorRecord.from_objective_values(
            proposal=Proposal(
                candidate=SpaceOwnedEqualityCandidate(3),
                proposal_id="p-dominated",
            ),
            candidate=SpaceOwnedEqualityCandidate(3),
            objective_values=(5.0,),
            directions=(OptimizationDirection.MINIMIZE,),
        )
        frontier_record = ObjectiveVectorRecord.from_objective_values(
            proposal=Proposal(
                candidate=SpaceOwnedEqualityCandidate(2),
                proposal_id="p-frontier",
            ),
            candidate=SpaceOwnedEqualityCandidate(1),
            objective_values=(1.0,),
            directions=(OptimizationDirection.MINIMIZE,),
        )
        refinement = CandidateRefinement(
            source_candidate=SpaceOwnedEqualityCandidate(2),
            refined_candidate=SpaceOwnedEqualityCandidate(1),
            changed_leaf_paths=((),),
        )

        surface = NondominatedRunSurface[SpaceOwnedEqualityCandidate](
            records=(dominated_record, frontier_record),
            nondominated_records=(frontier_record,),
            refinements=(None, refinement),
            evaluation_count=2,
            candidate_equal=space_owned_candidates_equal,
        )

        assert surface.nondominated_successes == (surface.successes[1],)
        assert surface.nondominated_successes[0].refinement is refinement
        projected_frontier = surface.nondominated_records[0]
        assert projected_frontier.request.proposal_id == "p-frontier"
        assert projected_frontier.request.candidate.stable_id == 2
        assert projected_frontier.candidate.stable_id == 1
        assert projected_frontier.objective_scores == (1.0,)

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
        report = RunReport[int, ObjectiveVectorRecord[int]].from_successes(
            successes=(
                make_vector_record_success(record_one),
                make_vector_record_success(record_two),
                make_vector_record_success(record_three),
                make_vector_record_success(dominated_record),
            ),
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
        report = RunReport[int, ObjectiveVectorRecord[int]].from_successes(
            successes=(
                make_vector_record_success(record_one, refinement=refinement),
                make_vector_record_success(record_two),
            ),
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
            _ = RunReport[int, RequestAlignedEvaluationRecord[int]].from_records(
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
