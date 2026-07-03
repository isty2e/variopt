"""Tests for runtime artifact values and terminal surfaces."""

import pickle
from dataclasses import replace
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
    EvaluationOutcome,
    EvaluationRecord,
    EvaluationRequest,
    NondominatedRunSurface,
    ObjectiveVectorRecord,
    Observation,
    OptimizationDirection,
    Proposal,
    RunReport,
    RunResult,
)
from variopt.artifacts import Trace, TraceEvent


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


def make_truthy_vector_equality_candidate() -> object:
    """Return a candidate whose equality result is truthy but not scalar."""

    def equality(_self: object, _other: object) -> list[bool]:
        return [True]

    candidate_type = type("TruthyVectorEqualityCandidate", (), {"__eq__": equality})
    return candidate_type()


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

    def test_observation_is_scalar_evaluation_record(self) -> None:
        proposal = Proposal(candidate=4, proposal_id="p-1")
        observation = Observation(
            proposal=proposal,
            candidate=4,
            value=16.0,
            score=16.0,
        )

        assert isinstance(observation, EvaluationRecord)

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

        assert isinstance(record, EvaluationRecord)
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

        report = RunReport[int, LabelRecord](
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
        record = EvaluationRecord(
            request=EvaluationRequest(proposal=proposal),
            candidate=candidate,
        )
        refinement = CandidateRefinement(
            source_candidate=candidate,
            refined_candidate=candidate,
            changed_leaf_paths=((),),
        )

        with pytest.raises(TypeError):
            _ = RunReport[AmbiguousEqualityCandidate, EvaluationRecord[AmbiguousEqualityCandidate]].from_records(
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
        record: EvaluationRecord[object] = EvaluationRecord(
            request=request,
            candidate=record_candidate,
        )
        refinement: CandidateRefinement[object] = CandidateRefinement(
            source_candidate=record_candidate,
            refined_candidate=refined_candidate,
            changed_leaf_paths=((),),
        )

        with pytest.raises(TypeError, match="scalar truth value"):
            _ = RunReport[
                object,
                EvaluationRecord[object],
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
        with pytest.raises(ValueError, match="observations candidate"):
            _ = replace(result, refinements=(mismatched_refinement,))
        with pytest.raises(ValueError, match="records candidate"):
            _ = replace(report, refinements=(mismatched_refinement,))
        with pytest.raises(ValueError, match="records candidate"):
            _ = replace(surface, refinements=(mismatched_refinement,))
        with pytest.raises(ValueError, match="records candidate"):
            _ = replace(report, candidate_equal=reject_candidate_equal)

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
            _ = replace(restored_report, refinements=(replacement_refinement,))

        revalidated_report = replace(
            restored_report,
            refinements=(replacement_refinement,),
            candidate_equal=space_owned_candidates_equal,
        )

        assert revalidated_report.refinements == (replacement_refinement,)

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
        collection_calls: list[tuple[ObjectiveVectorRecord[int], ...]] = []
        original_collect_nondominated_records = (
            terminal_artifacts.collect_nondominated_records
        )

        def collect_counting_frontier(
            records: tuple[ObjectiveVectorRecord[int], ...],
        ) -> tuple[ObjectiveVectorRecord[int], ...]:
            collection_calls.append(records)
            return original_collect_nondominated_records(records)

        monkeypatch.setattr(
            terminal_artifacts,
            "collect_nondominated_records",
            collect_counting_frontier,
        )

        surface = NondominatedRunSurface[int].from_records((record, dominated_record))

        assert collection_calls == [(record, dominated_record)]
        assert surface.nondominated_records == (record,)

    def test_nondominated_run_surface_rejects_public_prevalidation_cache_injection(
        self,
    ) -> None:
        constructor_parameters = signature(NondominatedRunSurface).parameters

        assert "_validated_frontier_source_records" not in constructor_parameters
        assert "_validated_frontier_records" not in constructor_parameters

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
        collection_calls: list[tuple[ObjectiveVectorRecord[int], ...]] = []

        def collect_empty_frontier(
            records: tuple[ObjectiveVectorRecord[int], ...],
        ) -> tuple[ObjectiveVectorRecord[int], ...]:
            collection_calls.append(records)
            return ()

        monkeypatch.setattr(
            "variopt.artifacts.terminal.collect_nondominated_records",
            collect_empty_frontier,
        )

        with pytest.raises(ValueError, match="stable nondominated frontier"):
            _ = replace(surface, records=(replacement_record,))

        assert collection_calls == [(replacement_record,)]

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
            _report: RunReport[int, LabelRecord] = RunReport(
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
