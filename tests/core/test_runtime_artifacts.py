"""Tests for runtime artifact values and terminal surfaces."""

import pytest
from typing_extensions import override

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
