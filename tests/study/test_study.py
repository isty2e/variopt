"""Tests for sync and general Study facade behavior."""

from collections.abc import Callable
from typing import cast

import pytest
from typing_extensions import override

from tests.study_support import (
    BatchQueueOptimizer,
    ContextAwareBatchQueueOptimizer,
    CountingObjective,
    DecrementKernel,
    LabelBatchQueueOptimizer,
    LabelProtocol,
    LabelRecord,
    MisorderedEvaluator,
    OutcomeAwareBatchQueueOptimizer,
    RecordingExecutionResourcesKernel,
    RecordingKernel,
    RefinementKernel,
    ScoringKernel,
    ShiftedObservationProtocol,
    SquareObjective,
)
from variopt import (
    EvaluationOutcome,
    EvaluationRequest,
    IntegerSpace,
    Observation,
    OptimizationDirection,
    Problem,
    Proposal,
    RunReport,
    Study,
)
from variopt.artifacts import ProposalEvaluationSpec
from variopt.evaluators import SequentialEvaluator
from variopt.execution import (
    EXACT_ASYNC_EXECUTION_MODEL,
    SEQUENTIAL_EXECUTION_MODEL,
    NestedParallelismPolicy,
)
from variopt.kernel import (
    DirectKernel,
    Kernel,
    ProposalBatchQuery,
    ProposalLocalSearchContext,
)
from variopt.study.common import build_evaluation_requests
from variopt.study.execution import evaluate_batch_sync


class RepeatingSubqueryKernel(
    Kernel[
        ProposalBatchQuery[int, int, Observation[int]],
        tuple[EvaluationOutcome[int, Observation[int]], ...],
    ],
):
    """Kernel that intentionally reuses one trial subquery object."""

    @override
    def run(
        self,
        query: ProposalBatchQuery[int, int, Observation[int]],
        runner: Callable[
            [ProposalBatchQuery[int, int, Observation[int]]],
            tuple[EvaluationOutcome[int, Observation[int]], ...],
        ],
    ) -> tuple[EvaluationOutcome[int, Observation[int]], ...]:
        subquery = ProposalBatchQuery(
            problem=query.problem,
            proposals=(Proposal(candidate=max(0, query.proposals[0].candidate - 1)),),
            execution_resources=query.execution_resources,
        )
        first_outcome = runner(subquery)[0]
        second_outcome = runner(subquery)[0]
        refined_record = second_outcome.record
        return (
            EvaluationOutcome(
                record=Observation.from_objective_value(
                    proposal=query.proposals[0],
                    candidate=refined_record.candidate,
                    value=refined_record.value,
                    direction=query.problem.direction,
                ),
                evaluation_count=(
                    first_outcome.evaluation_count
                    + second_outcome.evaluation_count
                ),
            ),
        )


class StudyTests:
    """Coverage for sync and execution-model-agnostic Study behavior."""

    def test_study_canonicalizes_missing_kernel_to_direct_kernel(self) -> None:
        problem = Problem(
            space=IntegerSpace(low=0, high=10),
            objective=SquareObjective(),
        )
        optimizer = BatchQueueOptimizer(
            proposal_batches=[(Proposal(candidate=3, proposal_id="p-1"),)],
        )
        evaluator = SequentialEvaluator[int, int]()

        study = Study(problem=problem, run_method=optimizer, evaluator=evaluator)

        assert isinstance(study.kernel, DirectKernel)

    def test_step_runs_through_custom_kernel(self) -> None:
        problem = Problem(
            space=IntegerSpace(low=0, high=10),
            objective=SquareObjective(),
        )
        optimizer = BatchQueueOptimizer(
            proposal_batches=[(Proposal(candidate=3, proposal_id="p-1"),)],
        )
        evaluator = SequentialEvaluator[int, int]()
        kernel = RecordingKernel()
        study = Study(
            problem=problem,
            run_method=optimizer,
            evaluator=evaluator,
            kernel=kernel,
        )
        state = optimizer.create_initial_state()

        observations, _ = study.step(state, batch_size=1)

        assert len(kernel.queries) == 1
        assert kernel.queries[0].proposals[0].proposal_id == "p-1"
        assert observations[0].proposal.proposal_id == "p-1"

    def test_step_propagates_run_method_kernel_hints_into_kernel_query(
        self,
    ) -> None:
        problem = Problem(
            space=IntegerSpace(low=0, high=10),
            objective=SquareObjective(),
        )
        optimizer = ContextAwareBatchQueueOptimizer(
            proposal_batches=[
                (
                    Proposal(candidate=3, proposal_id="p-1"),
                    Proposal(candidate=1, proposal_id="p-2"),
                ),
            ],
        )
        evaluator = SequentialEvaluator[int, int]()
        kernel = RecordingKernel()
        study = Study(
            problem=problem,
            run_method=optimizer,
            evaluator=evaluator,
            kernel=kernel,
        )

        _, _ = study.step(optimizer.create_initial_state(), batch_size=2)

        assert len(kernel.queries) == 1
        contexts = kernel.queries[0].proposal_kernel_hints
        assert contexts is not None
        assert contexts is not None
        assert all(isinstance(context, ProposalLocalSearchContext) for context in contexts)
        typed_contexts = cast(tuple[ProposalLocalSearchContext, ...], contexts)
        assert tuple(context.local_budget for context in typed_contexts) == (1, 2)

    def test_step_runs_ask_evaluate_tell_once(self) -> None:
        problem = Problem(
            space=IntegerSpace(low=0, high=10),
            objective=SquareObjective(),
        )
        optimizer = BatchQueueOptimizer(
            proposal_batches=[(Proposal(candidate=3, proposal_id="p-1"),)],
        )
        evaluator = SequentialEvaluator[int, int]()
        study = Study(problem=problem, run_method=optimizer, evaluator=evaluator)
        state = optimizer.create_initial_state()

        observations, next_state = study.step(state, batch_size=1)

        assert next_state.ask_history == (1,)
        assert len(next_state.tell_history) == 1
        assert observations == next_state.tell_history[0]
        assert observations[0].proposal.candidate == 3
        assert observations[0].candidate == 3
        assert observations[0].value == 9.0
        assert observations[0].score == 9.0

    def test_direct_step_reuses_request_batch_for_validation(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        build_call_count = 0
        original_builder = build_evaluation_requests

        def counting_builder(
            proposals: tuple[Proposal[int], ...],
            *,
            proposal_evaluation_specs: (
                tuple[ProposalEvaluationSpec | None, ...] | None
            ),
        ) -> tuple[EvaluationRequest[int], ...]:
            nonlocal build_call_count
            build_call_count += 1
            return original_builder(
                proposals,
                proposal_evaluation_specs=proposal_evaluation_specs,
            )

        monkeypatch.setattr(
            "variopt.study.execution.build_evaluation_requests",
            counting_builder,
        )
        problem = Problem(
            space=IntegerSpace(low=0, high=10),
            objective=SquareObjective(),
        )
        optimizer = BatchQueueOptimizer(
            proposal_batches=[(Proposal(candidate=3, proposal_id="p-1"),)],
        )
        evaluator = SequentialEvaluator[int, int]()
        study = Study(problem=problem, run_method=optimizer, evaluator=evaluator)

        _ = study.step(optimizer.create_initial_state(), batch_size=1)

        assert build_call_count == 1

    def test_transformed_kernel_step_does_not_reuse_unrelated_request_batch(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        build_call_count = 0
        original_builder = build_evaluation_requests

        def counting_builder(
            proposals: tuple[Proposal[int], ...],
            *,
            proposal_evaluation_specs: (
                tuple[ProposalEvaluationSpec | None, ...] | None
            ),
        ) -> tuple[EvaluationRequest[int], ...]:
            nonlocal build_call_count
            build_call_count += 1
            return original_builder(
                proposals,
                proposal_evaluation_specs=proposal_evaluation_specs,
            )

        monkeypatch.setattr(
            "variopt.study.execution.build_evaluation_requests",
            counting_builder,
        )
        problem = Problem(
            space=IntegerSpace(low=0, high=10),
            objective=SquareObjective(),
        )
        optimizer = BatchQueueOptimizer(
            proposal_batches=[(Proposal(candidate=3, proposal_id="p-1"),)],
        )
        evaluator = SequentialEvaluator[int, int]()
        study = Study(
            problem=problem,
            run_method=optimizer,
            evaluator=evaluator,
            kernel=DecrementKernel(),
        )

        observations, _ = study.step(optimizer.create_initial_state(), batch_size=1)

        assert observations[0].proposal.candidate == 3
        assert observations[0].candidate == 2
        assert build_call_count == 2

    def test_repeated_kernel_subquery_is_not_retained_in_request_cache(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        build_call_count = 0
        original_builder = build_evaluation_requests

        def counting_builder(
            proposals: tuple[Proposal[int], ...],
            *,
            proposal_evaluation_specs: (
                tuple[ProposalEvaluationSpec | None, ...] | None
            ),
        ) -> tuple[EvaluationRequest[int], ...]:
            nonlocal build_call_count
            build_call_count += 1
            return original_builder(
                proposals,
                proposal_evaluation_specs=proposal_evaluation_specs,
            )

        monkeypatch.setattr(
            "variopt.study.execution.build_evaluation_requests",
            counting_builder,
        )
        problem = Problem(
            space=IntegerSpace(low=0, high=10),
            objective=SquareObjective(),
        )
        optimizer = BatchQueueOptimizer(
            proposal_batches=[(Proposal(candidate=3, proposal_id="p-1"),)],
        )
        evaluator = SequentialEvaluator[int, int]()
        study = Study(
            problem=problem,
            run_method=optimizer,
            evaluator=evaluator,
            kernel=RepeatingSubqueryKernel(),
        )

        observations, _ = study.step(optimizer.create_initial_state(), batch_size=1)

        assert observations[0].proposal.candidate == 3
        assert observations[0].candidate == 2
        assert build_call_count == 3

    def test_evaluate_batch_sync_uses_supplied_requests_without_rebuilding(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def reject_builder(
            proposals: tuple[Proposal[int], ...],
            *,
            proposal_evaluation_specs: (
                tuple[ProposalEvaluationSpec | None, ...] | None
            ),
        ) -> tuple[EvaluationRequest[int], ...]:
            _ = proposals
            _ = proposal_evaluation_specs
            raise AssertionError("requests should be supplied by the caller")

        monkeypatch.setattr(
            "variopt.study.execution.build_evaluation_requests",
            reject_builder,
        )
        problem = Problem(
            space=IntegerSpace(low=0, high=10),
            objective=SquareObjective(),
        )
        optimizer = BatchQueueOptimizer(
            proposal_batches=[(Proposal(candidate=3, proposal_id="p-1"),)],
        )
        evaluator = SequentialEvaluator[int, int]()
        study = Study(problem=problem, run_method=optimizer, evaluator=evaluator)
        proposals = (Proposal(candidate=3, proposal_id="p-1"),)
        requests = build_evaluation_requests(
            proposals,
            proposal_evaluation_specs=None,
        )
        query = ProposalBatchQuery(
            problem=problem,
            proposals=proposals,
            execution_resources=evaluator.execution_resources(),
        )

        outcomes = evaluate_batch_sync(study, query, requests=requests)

        assert outcomes[0].record.candidate == 3
        assert outcomes[0].record.value == 9.0

    def test_step_uses_problem_evaluation_protocol_basis(self) -> None:
        problem = Problem(
            space=IntegerSpace(low=0, high=10),
            evaluation_protocol=ShiftedObservationProtocol(),
        )
        optimizer = BatchQueueOptimizer(
            proposal_batches=[(Proposal(candidate=4, proposal_id="p-1"),)],
        )
        evaluator = SequentialEvaluator[int, int]()
        study = Study(problem=problem, run_method=optimizer, evaluator=evaluator)

        observations, _ = study.step(optimizer.create_initial_state(), batch_size=1)

        assert len(observations) == 1
        assert observations[0].proposal.candidate == 4
        assert observations[0].candidate == 3
        assert observations[0].value == 9.0
        assert observations[0].score == 9.0

    def test_step_supports_non_scalar_evaluation_records(self) -> None:
        problem = Problem(
            space=IntegerSpace(low=0, high=10),
            evaluation_protocol=LabelProtocol(),
        )
        optimizer = LabelBatchQueueOptimizer(
            proposal_batches=[(Proposal(candidate=4, proposal_id="p-1"),)],
        )
        evaluator = SequentialEvaluator[int, int, LabelRecord]()
        study = Study(problem=problem, run_method=optimizer, evaluator=evaluator)

        records, next_state = study.step(optimizer.create_initial_state(), batch_size=1)

        assert len(records) == 1
        assert records[0].proposal.candidate == 4
        assert records[0].candidate == 4
        assert records[0].label == "parity:0"
        assert next_state.tell_history == ((records[0],),)

    def test_optimize_rejects_non_scalar_evaluation_records(self) -> None:
        problem = Problem(
            space=IntegerSpace(low=0, high=10),
            evaluation_protocol=LabelProtocol(),
        )
        optimizer = LabelBatchQueueOptimizer(
            proposal_batches=[(Proposal(candidate=3, proposal_id="p-1"),)],
        )
        evaluator = SequentialEvaluator[int, int, LabelRecord]()
        study = Study(problem=problem, run_method=optimizer, evaluator=evaluator)

        with pytest.raises(TypeError):
            _ = study.optimize(max_evaluations=1)

    def test_run_returns_terminal_report_for_non_scalar_evaluation_records(self) -> None:
        problem = Problem(
            space=IntegerSpace(low=0, high=10),
            evaluation_protocol=LabelProtocol(),
        )
        optimizer = LabelBatchQueueOptimizer(
            proposal_batches=[
                (Proposal(candidate=3, proposal_id="p-1"),),
                (Proposal(candidate=4, proposal_id="p-2"),),
            ],
        )
        evaluator = SequentialEvaluator[int, int, LabelRecord]()
        study = Study(problem=problem, run_method=optimizer, evaluator=evaluator)

        report, final_state = study.run(max_evaluations=2)

        assert isinstance(report, RunReport)
        assert report.evaluation_count == 2
        assert len(report.records) == 2
        assert tuple(record.label for record in report.records) == ("parity:1", "parity:0")
        assert len(report.trace.events) == 2
        assert all(event.value is None for event in report.trace.events)
        assert final_state.tell_history == (tuple(report.records[:1]), tuple(report.records[1:]))

    def test_optimize_returns_terminal_run_result(self) -> None:
        problem = Problem(
            space=IntegerSpace(low=0, high=10),
            objective=SquareObjective(),
        )
        optimizer = BatchQueueOptimizer(
            proposal_batches=[
                (
                    Proposal(candidate=4, proposal_id="p-1"),
                    Proposal(candidate=2, proposal_id="p-2"),
                ),
                (Proposal(candidate=1, proposal_id="p-3"),),
            ],
        )
        evaluator = SequentialEvaluator[int, int]()
        study = Study(problem=problem, run_method=optimizer, evaluator=evaluator)

        result, _ = study.optimize(max_evaluations=3, batch_size=2)

        assert len(result.observations) == 3
        assert result.evaluation_count == 3
        assert result.best_observation is not None
        assert result.best_observation is not None
        assert result.best_observation.candidate == 1
        assert result.best_observation.value == 1.0
        assert result.best_observation.score == 1.0
        assert len(result.trace.events) == 2
        assert result.refinements == ()

    def test_run_returns_terminal_run_report_for_scalar_observations(self) -> None:
        problem = Problem(
            space=IntegerSpace(low=0, high=10),
            objective=SquareObjective(),
        )
        optimizer = BatchQueueOptimizer(
            proposal_batches=[
                (
                    Proposal(candidate=4, proposal_id="p-1"),
                    Proposal(candidate=2, proposal_id="p-2"),
                ),
                (Proposal(candidate=1, proposal_id="p-3"),),
            ],
        )
        evaluator = SequentialEvaluator[int, int]()
        study = Study(problem=problem, run_method=optimizer, evaluator=evaluator)

        report, _ = study.run(max_evaluations=3, batch_size=2)

        assert report.evaluation_count == 3
        assert len(report.records) == 3
        assert report.records[0].value == 16.0
        assert report.records[1].value == 4.0
        assert report.records[2].value == 1.0
        assert tuple(event.value for event in report.trace.events) == (4.0, 1.0)

    def test_run_preserves_record_aligned_refinement_metadata(self) -> None:
        problem = Problem(
            space=IntegerSpace(low=0, high=10),
            objective=SquareObjective(),
        )
        optimizer = BatchQueueOptimizer(
            proposal_batches=[
                (
                    Proposal(candidate=4, proposal_id="p-1"),
                    Proposal(candidate=2, proposal_id="p-2"),
                ),
                (Proposal(candidate=0, proposal_id="p-3"),),
            ],
        )
        evaluator = SequentialEvaluator[int, int]()
        study = Study(
            problem=problem,
            run_method=optimizer,
            evaluator=evaluator,
            kernel=RefinementKernel(),
        )

        report, final_state = study.run(max_evaluations=3, batch_size=2)

        assert tuple(record.proposal.proposal_id for record in report.records) == (
            "p-1",
            "p-2",
            "p-3",
        )
        assert tuple(record.candidate for record in report.records) == (3, 1, 0)
        assert len(report.refinements) == 3
        first_refinement = report.refinements[0]
        second_refinement = report.refinements[1]
        assert first_refinement is not None
        assert second_refinement is not None
        assert first_refinement.source_candidate == 4
        assert first_refinement.refined_candidate == report.records[0].candidate
        assert second_refinement.source_candidate == 2
        assert second_refinement.refined_candidate == report.records[1].candidate
        assert report.refinements[2] is None
        assert final_state.tell_history == (
            tuple(report.records[:2]),
            tuple(report.records[2:]),
        )

    def test_run_keeps_no_refinement_report_allocation_light(self) -> None:
        problem = Problem(
            space=IntegerSpace(low=0, high=10),
            objective=SquareObjective(),
        )
        optimizer = BatchQueueOptimizer(
            proposal_batches=[(Proposal(candidate=4, proposal_id="p-1"),)],
        )
        evaluator = SequentialEvaluator[int, int]()
        study = Study(problem=problem, run_method=optimizer, evaluator=evaluator)

        report, _ = study.run(max_evaluations=1)

        assert report.refinements == ()

    def test_run_backfills_unrefined_history_when_late_refinement_appears(self) -> None:
        problem = Problem(
            space=IntegerSpace(low=0, high=10),
            objective=SquareObjective(),
        )
        optimizer = BatchQueueOptimizer(
            proposal_batches=[
                (Proposal(candidate=0, proposal_id="p-1"),),
                (Proposal(candidate=3, proposal_id="p-2"),),
            ],
        )
        evaluator = SequentialEvaluator[int, int]()
        study = Study(
            problem=problem,
            run_method=optimizer,
            evaluator=evaluator,
            kernel=RefinementKernel(),
        )

        report, _ = study.run(max_evaluations=2)

        assert len(report.refinements) == 2
        assert report.refinements[0] is None
        late_refinement = report.refinements[1]
        assert late_refinement is not None
        assert late_refinement.source_candidate == 3
        assert late_refinement.refined_candidate == report.records[1].candidate

    def test_run_preserves_refinement_when_evaluation_cost_overshoots_budget(
        self,
    ) -> None:
        problem = Problem(
            space=IntegerSpace(low=0, high=10),
            objective=SquareObjective(),
        )
        optimizer = BatchQueueOptimizer(
            proposal_batches=[(Proposal(candidate=4, proposal_id="p-1"),)],
        )
        evaluator = SequentialEvaluator[int, int]()
        study = Study(
            problem=problem,
            run_method=optimizer,
            evaluator=evaluator,
            kernel=ScoringKernel(),
        )

        report, _ = study.run(
            max_evaluations=3,
            count_evaluation_cost=True,
        )

        assert report.evaluation_count == 7
        assert len(report.records) == 1
        assert len(report.refinements) == 1
        refinement = report.refinements[0]
        assert refinement is not None
        assert refinement.source_candidate == 4
        assert refinement.refined_candidate == report.records[0].candidate

    def test_study_kernel_makes_local_optimization_visible(self) -> None:
        problem = Problem(
            space=IntegerSpace(low=0, high=10),
            objective=SquareObjective(),
        )
        optimizer = BatchQueueOptimizer(
            proposal_batches=[(Proposal(candidate=4, proposal_id="p-1"),)],
        )
        evaluator = SequentialEvaluator[int, int]()
        study = Study(
            problem=problem,
            run_method=optimizer,
            evaluator=evaluator,
            kernel=DecrementKernel(),
        )

        observations, _ = study.step(optimizer.create_initial_state(), batch_size=1)

        assert len(observations) == 1
        assert observations[0].proposal.candidate == 4
        assert observations[0].candidate == 3
        assert observations[0].value == 9.0
        assert observations[0].score == 9.0

    def test_study_kernel_can_supply_precomputed_objective_value_and_cost(self) -> None:
        objective = CountingObjective()
        problem = Problem(
            space=IntegerSpace(low=0, high=10),
            objective=objective,
        )
        optimizer = BatchQueueOptimizer(
            proposal_batches=[(Proposal(candidate=4, proposal_id="p-1"),)],
        )
        evaluator = SequentialEvaluator[int, int]()
        study = Study(
            problem=problem,
            run_method=optimizer,
            evaluator=evaluator,
            kernel=ScoringKernel(),
        )

        result, final_state = study.optimize(
            max_evaluations=7,
            count_evaluation_cost=True,
        )

        assert objective.evaluation_count == 0
        assert result.evaluation_count == 7
        assert len(result.observations) == 1
        assert result.observations[0].proposal.candidate == 4
        assert result.observations[0].candidate == 2
        assert result.observations[0].value == 4.0
        assert result.observations[0].score == 4.0
        assert len(result.refinements) == 1
        refinement = result.refinements[0]
        assert refinement is not None
        assert refinement.source_candidate == 4
        assert refinement.refined_candidate == 2
        assert len(final_state.tell_history) == 1

    def test_optimize_backfills_unrefined_history_when_late_refinement_appears(
        self,
    ) -> None:
        problem = Problem(
            space=IntegerSpace(low=0, high=10),
            objective=SquareObjective(),
        )
        optimizer = BatchQueueOptimizer(
            proposal_batches=[
                (Proposal(candidate=0, proposal_id="p-1"),),
                (Proposal(candidate=3, proposal_id="p-2"),),
            ],
        )
        evaluator = SequentialEvaluator[int, int]()
        study = Study(
            problem=problem,
            run_method=optimizer,
            evaluator=evaluator,
            kernel=RefinementKernel(),
        )

        result, _ = study.optimize(max_evaluations=2)

        assert tuple(
            observation.proposal.proposal_id for observation in result.observations
        ) == ("p-1", "p-2")
        assert len(result.refinements) == 2
        assert result.refinements[0] is None
        late_refinement = result.refinements[1]
        assert late_refinement is not None
        assert late_refinement.source_candidate == 3
        assert late_refinement.refined_candidate == result.observations[1].candidate

    def test_study_assimilation_uses_outcome_aware_run_method_hook(self) -> None:
        problem = Problem(
            space=IntegerSpace(low=0, high=10),
            objective=SquareObjective(),
        )
        optimizer = OutcomeAwareBatchQueueOptimizer(
            proposal_batches=[
                (
                    Proposal(candidate=4, proposal_id="p-1"),
                    Proposal(candidate=0, proposal_id="p-2"),
                ),
            ],
        )
        evaluator = SequentialEvaluator[int, int]()
        study = Study(
            problem=problem,
            run_method=optimizer,
            evaluator=evaluator,
            kernel=RefinementKernel(),
        )

        report, _ = study.run(max_evaluations=2, batch_size=2)

        assert tuple(record.candidate for record in report.records) == (3, 0)
        assert optimizer.seen_changed_leaf_paths == (((),), None)

    def test_study_passes_evaluator_execution_resources_to_kernel(self) -> None:
        problem = Problem(
            space=IntegerSpace(low=0, high=10),
            objective=SquareObjective(),
        )
        optimizer = BatchQueueOptimizer(
            proposal_batches=[(Proposal(candidate=4, proposal_id="p-1"),)],
        )
        kernel = RecordingExecutionResourcesKernel()
        evaluator = SequentialEvaluator[int, int]()
        study = Study(
            problem=problem,
            run_method=optimizer,
            evaluator=evaluator,
            kernel=kernel,
        )

        _ = study.step(optimizer.create_initial_state(), batch_size=1)

        assert kernel.last_execution_resources is not None
        assert kernel.last_execution_resources.parallel_owner == "evaluator"
        assert kernel.last_execution_resources.nested_parallelism_policy == NestedParallelismPolicy.FORBID
        assert kernel.last_execution_resources.owner_worker_count == 1
        assert kernel.last_execution_resources.owner_backend == "sequential"

    def test_step_rejects_non_positive_batch_size(self) -> None:
        problem = Problem(
            space=IntegerSpace(low=0, high=10),
            objective=SquareObjective(),
        )
        optimizer = BatchQueueOptimizer(
            proposal_batches=[(Proposal(candidate=3, proposal_id="p-1"),)],
        )
        evaluator = SequentialEvaluator[int, int]()
        study = Study(problem=problem, run_method=optimizer, evaluator=evaluator)
        state = optimizer.create_initial_state()

        with pytest.raises(ValueError):
            _ = study.step(state, batch_size=0)

    def test_step_rejects_sequential_model_with_batch_size_greater_than_one(self) -> None:
        problem = Problem(
            space=IntegerSpace(low=0, high=10),
            objective=SquareObjective(),
        )
        optimizer = BatchQueueOptimizer(
            proposal_batches=[
                (
                    Proposal(candidate=3, proposal_id="p-1"),
                    Proposal(candidate=1, proposal_id="p-2"),
                ),
            ],
        )
        evaluator = SequentialEvaluator[int, int]()
        study = Study(problem=problem, run_method=optimizer, evaluator=evaluator)
        state = optimizer.create_initial_state()

        with pytest.raises(ValueError, match="sequential execution model requires batch_size == 1"):
            _ = study.step(
                state,
                batch_size=2,
                execution_model=SEQUENTIAL_EXECUTION_MODEL,
            )

    def test_step_rejects_unsupported_execution_model(self) -> None:
        problem = Problem(
            space=IntegerSpace(low=0, high=10),
            objective=SquareObjective(),
        )
        optimizer = BatchQueueOptimizer(
            proposal_batches=[(Proposal(candidate=3, proposal_id="p-1"),)],
        )
        evaluator = SequentialEvaluator[int, int]()
        study = Study(problem=problem, run_method=optimizer, evaluator=evaluator)
        state = optimizer.create_initial_state()

        with pytest.raises(ValueError, match="run_method does not support the requested execution model: exact_async"):
            _ = study.step(
                state,
                execution_model=EXACT_ASYNC_EXECUTION_MODEL,
            )

    def test_study_rejects_misaligned_evaluator_outcomes(self) -> None:
        problem = Problem(
            space=IntegerSpace(low=0, high=10),
            objective=SquareObjective(),
        )
        optimizer = BatchQueueOptimizer(
            proposal_batches=[
                (
                    Proposal(candidate=4, proposal_id="p-1"),
                    Proposal(candidate=2, proposal_id="p-2"),
                ),
            ],
        )
        study = Study(
            problem=problem,
            run_method=optimizer,
            evaluator=MisorderedEvaluator(),
        )
        state = optimizer.create_initial_state()

        with pytest.raises(ValueError, match="evaluator outcomes must align with input request order"):
            _ = study.step(state, batch_size=2)

    def test_optimize_allows_zero_evaluations(self) -> None:
        problem = Problem(
            space=IntegerSpace(low=0, high=10),
            objective=SquareObjective(),
        )
        optimizer = BatchQueueOptimizer(proposal_batches=[])
        evaluator = SequentialEvaluator[int, int]()
        study = Study(problem=problem, run_method=optimizer, evaluator=evaluator)

        result, final_state = study.optimize(max_evaluations=0)

        assert result.observations == ()
        assert result.best_observation is None
        assert result.trace.events == ()
        assert optimizer.is_exhausted(final_state)

    def test_step_rejects_exhausted_optimizer(self) -> None:
        problem = Problem(
            space=IntegerSpace(low=0, high=10),
            objective=SquareObjective(),
        )
        optimizer = BatchQueueOptimizer(proposal_batches=[])
        evaluator = SequentialEvaluator[int, int]()
        study = Study(problem=problem, run_method=optimizer, evaluator=evaluator)
        state = optimizer.create_initial_state()

        with pytest.raises(RuntimeError, match="exhausted"):
            _ = study.step(state, batch_size=1)

    def test_optimize_stops_when_optimizer_exhausts_before_budget(self) -> None:
        problem = Problem(
            space=IntegerSpace(low=0, high=10),
            objective=SquareObjective(),
        )
        optimizer = BatchQueueOptimizer(
            proposal_batches=[(Proposal(candidate=3, proposal_id="p-1"),)],
        )
        evaluator = SequentialEvaluator[int, int]()
        study = Study(problem=problem, run_method=optimizer, evaluator=evaluator)

        result, _ = study.optimize(max_evaluations=4, batch_size=2)

        assert len(result.observations) == 1
        assert result.evaluation_count == 1
        assert result.observations[0].candidate == 3

    def test_optimize_respects_maximize_direction(self) -> None:
        problem = Problem(
            space=IntegerSpace(low=0, high=10),
            objective=SquareObjective(),
            direction=OptimizationDirection.MAXIMIZE,
        )
        optimizer = BatchQueueOptimizer(
            proposal_batches=[
                (
                    Proposal(candidate=1, proposal_id="p-1"),
                    Proposal(candidate=4, proposal_id="p-2"),
                ),
            ],
        )
        evaluator = SequentialEvaluator[int, int]()
        study = Study(problem=problem, run_method=optimizer, evaluator=evaluator)

        result, _ = study.optimize(max_evaluations=2, batch_size=2)

        assert result.best_observation is not None
        assert result.best_observation.candidate == 4
        assert result.best_observation.value == 16.0
        assert result.best_observation.score == -16.0

    def test_optimize_can_budget_by_local_optimization_cost(self) -> None:
        problem = Problem(
            space=IntegerSpace(low=0, high=10),
            objective=SquareObjective(),
        )
        optimizer = BatchQueueOptimizer(
            proposal_batches=[
                (Proposal(candidate=4, proposal_id="p-1"),),
                (Proposal(candidate=2, proposal_id="p-2"),),
            ],
        )
        evaluator = SequentialEvaluator[int, int]()
        study = Study(
            problem=problem,
            run_method=optimizer,
            evaluator=evaluator,
            kernel=ScoringKernel(),
        )

        result, final_state = study.optimize(
            max_evaluations=7,
            count_evaluation_cost=True,
        )

        assert len(result.observations) == 1
        assert result.evaluation_count == 7
        assert len(final_state.tell_history) == 1

    def test_optimize_can_ignore_local_optimization_cost_for_budgeting(self) -> None:
        problem = Problem(
            space=IntegerSpace(low=0, high=10),
            objective=SquareObjective(),
        )
        optimizer = BatchQueueOptimizer(
            proposal_batches=[
                (Proposal(candidate=4, proposal_id="p-1"),),
                (Proposal(candidate=2, proposal_id="p-2"),),
            ],
        )
        evaluator = SequentialEvaluator[int, int]()
        study = Study(
            problem=problem,
            run_method=optimizer,
            evaluator=evaluator,
            kernel=ScoringKernel(),
        )

        result, final_state = study.optimize(
            max_evaluations=2,
            count_evaluation_cost=False,
        )

        assert len(result.observations) == 2
        assert result.evaluation_count == 2
        assert len(final_state.tell_history) == 2
