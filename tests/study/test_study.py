"""Tests for sync and general Study facade behavior."""

from collections.abc import Callable, Sequence
from dataclasses import replace
from typing import Protocol, cast, runtime_checkable

import pytest
from typing_extensions import override

from tests.study_support import (
    BatchQueueOptimizer,
    BatchQueueOptimizerState,
    ContextAwareBatchQueueOptimizer,
    CountingObjective,
    DecrementKernel,
    FailingCandidateObjective,
    FailureRecordingBatchQueueOptimizer,
    HardFailingEvaluator,
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
    SpaceOwnedEqualityCandidate,
    SpaceOwnedEqualityObjective,
    SpaceOwnedEqualityOptimizer,
    SpaceOwnedEqualityRefinementKernel,
    SpaceOwnedEqualitySpace,
    SquareObjective,
)
from variopt import (
    CandidateRefinement,
    EvaluationAttemptBatch,
    EvaluationBudgetExhausted,
    EvaluationOutcome,
    EvaluationRequest,
    IntegerSpace,
    Objective,
    Observation,
    OptimizationDirection,
    Problem,
    Proposal,
    RunReport,
    Study,
)
from variopt.artifacts import (
    EvaluationSuccess,
    ProposalEvaluationSpec,
    Trace,
    TraceEvent,
)
from variopt.evaluators import JoblibEvaluator, SequentialEvaluator
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
from variopt.study.common import build_evaluation_requests, validate_aligned_outcomes
from variopt.study.execution import evaluate_batch_sync


@runtime_checkable
class BatchQueueRunFailure(Protocol):
    """Typed shape for hard-failure assertions over batch-queue study runs."""

    partial_report: RunReport[int, Observation[int]]
    partial_state: BatchQueueOptimizerState
    checkpoint_safe_report: RunReport[int, Observation[int]] | None
    checkpoint_safe_state: BatchQueueOptimizerState | None
    cause: Exception


class UnsafeCheckpointBatchQueueOptimizer(BatchQueueOptimizer):
    """Batch-queue optimizer that never exposes a checkpoint-safe state."""

    @override
    def is_checkpoint_safe_state(self, state: BatchQueueOptimizerState) -> bool:
        _ = state
        return False


class KeyboardInterruptObjective(Objective[int]):
    """Objective that raises KeyboardInterrupt for propagation tests."""

    @override
    def evaluate(self, candidate: int) -> float:
        _ = candidate
        raise KeyboardInterrupt


class RepeatingSubqueryKernel(
    Kernel[
        ProposalBatchQuery[int, int, Observation[int]],
        EvaluationAttemptBatch[int, Observation[int]],
    ],
):
    """Kernel that intentionally reuses one trial subquery object."""

    @override
    def run(
        self,
        query: ProposalBatchQuery[int, int, Observation[int]],
        runner: Callable[
            [ProposalBatchQuery[int, int, Observation[int]]],
            EvaluationAttemptBatch[int, Observation[int]],
        ],
    ) -> EvaluationAttemptBatch[int, Observation[int]]:
        subquery = ProposalBatchQuery(
            problem=query.problem,
            proposals=(Proposal(candidate=max(0, query.proposals[0].candidate - 1)),),
            execution_resources=query.execution_resources,
        )
        first_success = runner(subquery).successes[0]
        second_success = runner(subquery).successes[0]
        refined_record = second_success.scalar_observation()
        refinement = None
        if refined_record.candidate != query.proposals[0].candidate:
            refinement = CandidateRefinement(
                source_candidate=query.proposals[0].candidate,
                refined_candidate=refined_record.candidate,
                changed_leaf_paths=((),),
            )
        proposal_evaluation_spec = None
        if query.proposal_evaluation_specs is not None:
            proposal_evaluation_spec = query.proposal_evaluation_specs[0]
        request = EvaluationRequest(
            proposal=Proposal(
                candidate=refined_record.candidate,
                proposal_id=query.proposals[0].proposal_id,
            ),
            proposal_evaluation_spec=proposal_evaluation_spec,
        )
        observation = Observation.from_objective_value(
            request=request,
            candidate=refined_record.candidate,
            value=refined_record.value,
            direction=query.problem.direction,
        )
        success = EvaluationSuccess(
            request=request,
            payload=observation,
            evaluation_count=(
                first_success.evaluation_count + second_success.evaluation_count
            ),
            refinement=refinement,
        )
        return EvaluationAttemptBatch(
            attempts=(success,),
        )


class LocalProposalEvaluationSpec(ProposalEvaluationSpec):
    """Request metadata marker for direct-scalar fast-path tests."""


class StaticProposalSpecOptimizer(BatchQueueOptimizer):
    """Batch-queue optimizer that returns a fixed proposal-spec batch."""

    _proposal_evaluation_specs: tuple[ProposalEvaluationSpec | None, ...]

    def __init__(
        self,
        *,
        proposal_batches: list[tuple[Proposal[int], ...]],
        proposal_evaluation_specs: tuple[ProposalEvaluationSpec | None, ...],
    ) -> None:
        super().__init__(proposal_batches)
        self._proposal_evaluation_specs = proposal_evaluation_specs

    @override
    def proposal_evaluation_specs(
        self,
        state: BatchQueueOptimizerState,
        proposals: Sequence[Proposal[int]],
    ) -> tuple[ProposalEvaluationSpec | None, ...] | None:
        _ = state, proposals
        return self._proposal_evaluation_specs


class StaticKernelHintOptimizer(BatchQueueOptimizer):
    """Batch-queue optimizer that returns a fixed kernel-hint batch."""

    _proposal_kernel_hints: tuple[ProposalLocalSearchContext | None, ...]

    def __init__(
        self,
        *,
        proposal_batches: list[tuple[Proposal[int], ...]],
        proposal_kernel_hints: tuple[ProposalLocalSearchContext | None, ...],
    ) -> None:
        super().__init__(proposal_batches)
        self._proposal_kernel_hints = proposal_kernel_hints

    @override
    def proposal_kernel_hints(
        self,
        state: BatchQueueOptimizerState,
        proposals: Sequence[Proposal[int]],
    ) -> tuple[ProposalLocalSearchContext | None, ...] | None:
        _ = state, proposals
        return self._proposal_kernel_hints


class RequestAwareObjective(Objective[int]):
    """Objective that relies on the canonical request evaluation protocol."""

    direct_evaluation_count: int
    request_evaluation_count: int

    def __init__(self) -> None:
        self.direct_evaluation_count = 0
        self.request_evaluation_count = 0

    @override
    def evaluate(self, candidate: int) -> float:
        self.direct_evaluation_count += 1
        return float(-(candidate * candidate))

    @override
    def evaluate_request(
        self,
        request: EvaluationRequest[int],
        *,
        direction: OptimizationDirection,
    ) -> Observation[int]:
        self.request_evaluation_count += 1
        spec_bonus = (
            100.0
            if isinstance(request.proposal_evaluation_spec, LocalProposalEvaluationSpec)
            else 0.0
        )
        return Observation.from_objective_value(
            request=request,
            candidate=request.candidate,
            value=float(request.candidate) + spec_bonus,
            direction=direction,
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
        assert all(
            isinstance(context, ProposalLocalSearchContext) for context in contexts
        )
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

    def test_validate_aligned_outcomes_uses_candidate_equal_for_request_alignment(
        self,
    ) -> None:
        expected_request = EvaluationRequest(
            proposal=Proposal(
                candidate=SpaceOwnedEqualityCandidate(3),
                proposal_id="p-1",
            ),
        )
        outcome_request = EvaluationRequest(
            proposal=Proposal(
                candidate=SpaceOwnedEqualityCandidate(3),
                proposal_id="p-1",
            ),
        )
        outcome = EvaluationOutcome(
            record=Observation(
                request=outcome_request,
                candidate=outcome_request.candidate,
                value=9.0,
                score=9.0,
            ),
        )

        validate_aligned_outcomes(
            (expected_request,),
            (outcome,),
            candidate_equal=SpaceOwnedEqualitySpace().candidates_equal,
        )

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
        assert observations[0].candidate == 4
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

    def test_run_returns_terminal_report_for_non_scalar_evaluation_records(
        self,
    ) -> None:
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
        assert tuple(record.label for record in report.records) == (
            "parity:1",
            "parity:0",
        )
        assert len(report.trace.events) == 2
        assert all(event.value is None for event in report.trace.events)
        assert final_state.tell_history == (
            tuple(report.records[:1]),
            tuple(report.records[1:]),
        )

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

    def test_optimize_uses_direct_scalar_sequential_fast_path(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def fail_evaluate_step(*args: object, **kwargs: object) -> object:
            _ = args, kwargs
            raise AssertionError("generic evaluate_step should not be called")

        monkeypatch.setattr(
            "variopt.study.execution.evaluate_step",
            fail_evaluate_step,
        )
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

        result, final_state = study.optimize(max_evaluations=3, batch_size=2)

        assert tuple(
            observation.proposal.proposal_id for observation in result.observations
        ) == (
            "p-1",
            "p-2",
            "p-3",
        )
        assert tuple(observation.value for observation in result.observations) == (
            16.0,
            4.0,
            1.0,
        )
        assert result.evaluation_count == 3
        assert result.best_observation is not None
        assert result.best_observation.proposal.proposal_id == "p-3"
        assert tuple(event.value for event in result.trace.events) == (4.0, 1.0)
        assert final_state.ask_history == (2, 1)
        assert tuple(
            tuple(observation.proposal.proposal_id for observation in batch)
            for batch in final_state.tell_history
        ) == (("p-1", "p-2"), ("p-3",))

    def test_optimize_fast_path_preserves_outcome_aware_tell_hook(self) -> None:
        problem = Problem(
            space=IntegerSpace(low=0, high=10),
            objective=SquareObjective(),
        )
        optimizer = OutcomeAwareBatchQueueOptimizer(
            proposal_batches=[
                (
                    Proposal(candidate=4, proposal_id="p-1"),
                    Proposal(candidate=2, proposal_id="p-2"),
                ),
            ],
        )
        evaluator = SequentialEvaluator[int, int]()
        study = Study(problem=problem, run_method=optimizer, evaluator=evaluator)

        result, _ = study.optimize(max_evaluations=2, batch_size=2)

        assert tuple(observation.candidate for observation in result.observations) == (
            4,
            2,
        )
        assert optimizer.seen_changed_leaf_paths == (None, None)

    def test_optimize_fast_path_carries_result_candidate_equality(self) -> None:
        problem = Problem(
            space=SpaceOwnedEqualitySpace(),
            objective=SpaceOwnedEqualityObjective(),
        )
        optimizer = SpaceOwnedEqualityOptimizer()
        evaluator = SequentialEvaluator[
            int | SpaceOwnedEqualityCandidate,
            SpaceOwnedEqualityCandidate,
        ]()
        study = Study(problem=problem, run_method=optimizer, evaluator=evaluator)

        result, _ = study.optimize(max_evaluations=1)
        refinement = CandidateRefinement(
            source_candidate=SpaceOwnedEqualityCandidate(99),
            refined_candidate=SpaceOwnedEqualityCandidate(2),
            changed_leaf_paths=((),),
        )

        updated_result = replace(result, refinements=(refinement,))

        assert result.refinements == ()
        assert updated_result.refinements == (refinement,)

    def test_optimize_keeps_request_aware_scalar_protocol_on_generic_path(self) -> None:
        problem = Problem(
            space=IntegerSpace(low=0, high=10),
            evaluation_protocol=ShiftedObservationProtocol(),
        )
        optimizer = BatchQueueOptimizer(
            proposal_batches=[(Proposal(candidate=4, proposal_id="p-1"),)],
        )
        evaluator = SequentialEvaluator[int, int]()
        study = Study(problem=problem, run_method=optimizer, evaluator=evaluator)

        result, _ = study.optimize(max_evaluations=1)

        assert problem.direct_objective is None
        assert result.observations[0].proposal.proposal_id == "p-1"
        assert result.observations[0].candidate == 4

    def test_optimize_keeps_request_overriding_objective_on_generic_path(self) -> None:
        spec = LocalProposalEvaluationSpec()
        objective = RequestAwareObjective()
        problem = Problem(
            space=IntegerSpace(low=0, high=10),
            objective=objective,
        )
        optimizer = StaticProposalSpecOptimizer(
            proposal_batches=[(Proposal(candidate=4, proposal_id="p-1"),)],
            proposal_evaluation_specs=(spec,),
        )
        evaluator = SequentialEvaluator[int, int]()
        study = Study(problem=problem, run_method=optimizer, evaluator=evaluator)

        result, _ = study.optimize(max_evaluations=1)

        assert result.observations[0].request.proposal_evaluation_spec is spec
        assert result.observations[0].value == 104.0
        assert objective.request_evaluation_count == 1
        assert objective.direct_evaluation_count == 0

    def test_optimize_fast_path_validates_candidates_before_objective(self) -> None:
        objective = CountingObjective()
        problem = Problem(
            space=IntegerSpace(low=0, high=10),
            objective=objective,
        )
        optimizer = BatchQueueOptimizer(
            proposal_batches=[(Proposal(candidate=99, proposal_id="p-1"),)],
        )
        evaluator = SequentialEvaluator[int, int]()
        study = Study(problem=problem, run_method=optimizer, evaluator=evaluator)

        try:
            _ = study.optimize(max_evaluations=1)
        except RuntimeError as raw_exception:
            assert raw_exception.__class__.__name__ == "RunExecutionFailed"
            assert isinstance(raw_exception, BatchQueueRunFailure)
            exception = raw_exception
        else:
            pytest.fail("expected invalid candidate hard failure")

        assert isinstance(exception.cause, ValueError)
        assert exception.partial_report.records == ()
        assert exception.partial_report.failures == ()
        assert exception.partial_report.evaluation_count == 0
        assert objective.evaluation_count == 0

    def test_optimize_fast_path_preserves_proposal_evaluation_specs(self) -> None:
        spec = LocalProposalEvaluationSpec()
        problem = Problem(
            space=IntegerSpace(low=0, high=10),
            objective=SquareObjective(),
        )
        optimizer = StaticProposalSpecOptimizer(
            proposal_batches=[(Proposal(candidate=4, proposal_id="p-1"),)],
            proposal_evaluation_specs=(spec,),
        )
        evaluator = SequentialEvaluator[int, int]()
        study = Study(problem=problem, run_method=optimizer, evaluator=evaluator)

        result, _ = study.optimize(max_evaluations=1)

        assert result.observations[0].request.proposal_evaluation_spec is spec

    def test_optimize_fast_path_rejects_misaligned_proposal_metadata(self) -> None:
        problem = Problem(
            space=IntegerSpace(low=0, high=10),
            objective=SquareObjective(),
        )
        spec_optimizer = StaticProposalSpecOptimizer(
            proposal_batches=[(Proposal(candidate=4, proposal_id="p-1"),)],
            proposal_evaluation_specs=(LocalProposalEvaluationSpec(), None),
        )
        hint_optimizer = StaticKernelHintOptimizer(
            proposal_batches=[(Proposal(candidate=4, proposal_id="p-1"),)],
            proposal_kernel_hints=(ProposalLocalSearchContext(), None),
        )
        evaluator = SequentialEvaluator[int, int]()

        try:
            _ = Study(
                problem=problem,
                run_method=spec_optimizer,
                evaluator=evaluator,
            ).optimize(max_evaluations=1)
        except RuntimeError as raw_exception:
            assert raw_exception.__class__.__name__ == "RunExecutionFailed"
            assert isinstance(raw_exception, BatchQueueRunFailure)
            assert isinstance(raw_exception.cause, ValueError)
            assert "proposal_evaluation_specs" in str(raw_exception.cause)
        else:
            pytest.fail("expected proposal_evaluation_specs hard failure")

        try:
            _ = Study(
                problem=problem,
                run_method=hint_optimizer,
                evaluator=evaluator,
            ).optimize(max_evaluations=1)
        except RuntimeError as raw_exception:
            assert raw_exception.__class__.__name__ == "RunExecutionFailed"
            assert isinstance(raw_exception, BatchQueueRunFailure)
            assert isinstance(raw_exception.cause, ValueError)
            assert "proposal_kernel_hints" in str(raw_exception.cause)
        else:
            pytest.fail("expected proposal_kernel_hints hard failure")

    def test_optimize_fast_path_preserves_budget_boundaries(self) -> None:
        problem = Problem(
            space=IntegerSpace(low=0, high=10),
            objective=SquareObjective(),
        )
        optimizer = BatchQueueOptimizer(
            proposal_batches=[(Proposal(candidate=4, proposal_id="p-1"),)],
        )
        evaluator = SequentialEvaluator[int, int]()
        study = Study(problem=problem, run_method=optimizer, evaluator=evaluator)

        result, final_state = study.optimize(max_evaluations=0)

        assert result.evaluation_count == 0
        assert result.observations == ()
        assert result.trace.events == ()
        assert final_state.ask_history == ()
        with pytest.raises(ValueError, match="batch_size"):
            _ = study.optimize(
                max_evaluations=1,
                batch_size=2,
                execution_model=SEQUENTIAL_EXECUTION_MODEL,
            )

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

    def test_run_buffers_trace_events_without_trace_append(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def reject_trace_append(_trace: Trace, _event: TraceEvent) -> Trace:
            raise AssertionError("run should buffer trace events before materialization")

        monkeypatch.setattr(Trace, "append", reject_trace_append)
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

    def test_run_records_sync_mixed_evaluation_failures(self) -> None:
        problem = Problem(
            space=IntegerSpace(low=0, high=10),
            objective=FailingCandidateObjective(failed_candidates=(5,)),
        )
        optimizer = FailureRecordingBatchQueueOptimizer(
            proposal_batches=[
                (
                    Proposal(candidate=2, proposal_id="p-1"),
                    Proposal(candidate=5, proposal_id="p-2"),
                    Proposal(candidate=1, proposal_id="p-3"),
                ),
            ],
        )
        evaluator = SequentialEvaluator[int, int]()
        study = Study(problem=problem, run_method=optimizer, evaluator=evaluator)

        report, final_state = study.run(max_evaluations=3, batch_size=3)

        assert tuple(record.proposal.proposal_id for record in report.records) == (
            "p-1",
            "p-3",
        )
        assert tuple(failure.proposal_id for failure in report.failures) == ("p-2",)
        assert report.failures[0].exception.exception_type == "builtins.ValueError"
        assert report.evaluation_count == 3
        assert tuple(
            failure_proposal_id
            for failure_batch in final_state.failure_history
            for failure_proposal_id in failure_batch
        ) == ("p-2",)
        assert final_state.tell_history == ((report.records[0], report.records[1]),)
        assert report.trace.events[0].message == (
            "completed 3 attempt(s): 2 succeeded, 1 failed"
        )

    def test_run_with_joblib_evaluator_records_failures(self) -> None:
        problem = Problem(
            space=IntegerSpace(low=0, high=10),
            objective=FailingCandidateObjective(failed_candidates=(5,)),
        )
        optimizer = FailureRecordingBatchQueueOptimizer(
            proposal_batches=[
                (
                    Proposal(candidate=2, proposal_id="p-1"),
                    Proposal(candidate=5, proposal_id="p-2"),
                    Proposal(candidate=1, proposal_id="p-3"),
                ),
            ],
        )
        evaluator = JoblibEvaluator[int, int](backend="threading", n_jobs=2)
        study = Study(problem=problem, run_method=optimizer, evaluator=evaluator)

        report, final_state = study.run(max_evaluations=3, batch_size=3)

        assert tuple(record.proposal.proposal_id for record in report.records) == (
            "p-1",
            "p-3",
        )
        assert tuple(failure.proposal_id for failure in report.failures) == ("p-2",)
        assert final_state.failure_history == (("p-2",),)

    def test_run_hard_failure_carries_partial_and_checkpoint_safe_report(self) -> None:
        problem = Problem(
            space=IntegerSpace(low=0, high=10),
            objective=SquareObjective(),
        )
        optimizer = BatchQueueOptimizer(
            proposal_batches=[
                (Proposal(candidate=2, proposal_id="p-1"),),
                (Proposal(candidate=3, proposal_id="p-2"),),
            ],
        )
        evaluator = HardFailingEvaluator(fail_on_call=2)
        study = Study(problem=problem, run_method=optimizer, evaluator=evaluator)

        try:
            _ = study.run(
                max_evaluations=2,
                batch_size=1,
                stop_at_checkpoint_boundary=True,
            )
        except RuntimeError as raw_exception:
            assert raw_exception.__class__.__name__ == "RunExecutionFailed"
            assert isinstance(raw_exception, BatchQueueRunFailure)
            exception = raw_exception
        else:
            pytest.fail("expected hard evaluator failure")

        assert isinstance(exception.cause, RuntimeError)
        assert tuple(
            record.proposal.proposal_id
            for record in exception.partial_report.records
        ) == ("p-1",)
        assert exception.partial_report.evaluation_count == 2
        assert exception.partial_state.tell_history == (
            (exception.partial_report.records[0],),
        )
        checkpoint_report = exception.checkpoint_safe_report
        checkpoint_state = exception.checkpoint_safe_state
        assert checkpoint_report is not None
        assert checkpoint_state is not None
        assert tuple(
            record.proposal.proposal_id for record in checkpoint_report.records
        ) == ("p-1",)
        assert checkpoint_report.evaluation_count == 1
        assert checkpoint_state.tell_history == ((checkpoint_report.records[0],),)

    def test_run_hard_failure_before_safe_state_has_no_checkpoint_projection(
        self,
    ) -> None:
        problem = Problem(
            space=IntegerSpace(low=0, high=10),
            objective=SquareObjective(),
        )
        optimizer = UnsafeCheckpointBatchQueueOptimizer(
            proposal_batches=[(Proposal(candidate=2, proposal_id="p-1"),)],
        )
        evaluator = HardFailingEvaluator(fail_on_call=1)
        study = Study(problem=problem, run_method=optimizer, evaluator=evaluator)

        try:
            _ = study.run(
                max_evaluations=1,
                batch_size=1,
                stop_at_checkpoint_boundary=True,
            )
        except RuntimeError as raw_exception:
            assert raw_exception.__class__.__name__ == "RunExecutionFailed"
            assert isinstance(raw_exception, BatchQueueRunFailure)
            exception = raw_exception
        else:
            pytest.fail("expected hard evaluator failure")

        assert exception.partial_report.records == ()
        assert exception.partial_report.evaluation_count == 1
        assert exception.checkpoint_safe_report is None
        assert exception.checkpoint_safe_state is None

    def test_run_does_not_wrap_keyboard_interrupt(self) -> None:
        problem = Problem(
            space=IntegerSpace(low=0, high=10),
            objective=KeyboardInterruptObjective(),
        )
        optimizer = BatchQueueOptimizer(
            proposal_batches=[(Proposal(candidate=2, proposal_id="p-1"),)],
        )
        evaluator = SequentialEvaluator[int, int]()
        study = Study(problem=problem, run_method=optimizer, evaluator=evaluator)

        with pytest.raises(KeyboardInterrupt):
            _ = study.run(max_evaluations=1)

    def test_optimize_direct_scalar_fast_path_records_failures(self) -> None:
        problem = Problem(
            space=IntegerSpace(low=0, high=10),
            objective=FailingCandidateObjective(failed_candidates=(5,)),
        )
        optimizer = FailureRecordingBatchQueueOptimizer(
            proposal_batches=[
                (
                    Proposal(candidate=2, proposal_id="p-1"),
                    Proposal(candidate=5, proposal_id="p-2"),
                    Proposal(candidate=1, proposal_id="p-3"),
                ),
            ],
        )
        evaluator = SequentialEvaluator[int, int]()
        study = Study(problem=problem, run_method=optimizer, evaluator=evaluator)

        result, final_state = study.optimize(max_evaluations=3, batch_size=3)

        assert tuple(
            observation.proposal.proposal_id for observation in result.observations
        ) == ("p-1", "p-3")
        assert tuple(failure.proposal_id for failure in result.failures) == ("p-2",)
        assert result.evaluation_count == 3
        assert result.best_observation is not None
        assert result.best_observation.proposal.proposal_id == "p-3"
        assert tuple(
            failure_proposal_id
            for failure_batch in final_state.failure_history
            for failure_proposal_id in failure_batch
        ) == ("p-2",)

    def test_optimize_direct_scalar_fast_path_returns_failure_only_result(self) -> None:
        problem = Problem(
            space=IntegerSpace(low=0, high=10),
            objective=FailingCandidateObjective(failed_candidates=(2, 5)),
        )
        optimizer = FailureRecordingBatchQueueOptimizer(
            proposal_batches=[
                (
                    Proposal(candidate=2, proposal_id="p-1"),
                    Proposal(candidate=5, proposal_id="p-2"),
                ),
            ],
        )
        evaluator = SequentialEvaluator[int, int]()
        study = Study(problem=problem, run_method=optimizer, evaluator=evaluator)

        result, final_state = study.optimize(max_evaluations=2, batch_size=2)

        assert result.observations == ()
        assert result.best_observation is None
        assert tuple(failure.proposal_id for failure in result.failures) == (
            "p-1",
            "p-2",
        )
        assert final_state.tell_history == ((),)
        assert final_state.failure_history == (("p-1", "p-2"),)

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

    def test_run_rejects_refinement_when_evaluation_cost_overshoots_budget(
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

        with pytest.raises(EvaluationBudgetExhausted):
            _ = study.run(
                max_evaluations=3,
                count_evaluation_cost=True,
            )

    def test_run_uses_space_candidate_equality_for_report_refinements(self) -> None:
        problem = Problem(
            space=SpaceOwnedEqualitySpace(),
            objective=SpaceOwnedEqualityObjective(),
        )
        optimizer = SpaceOwnedEqualityOptimizer()
        evaluator = SequentialEvaluator[
            int | SpaceOwnedEqualityCandidate,
            SpaceOwnedEqualityCandidate,
        ]()
        study = Study(
            problem=problem,
            run_method=optimizer,
            evaluator=evaluator,
            kernel=SpaceOwnedEqualityRefinementKernel(),
        )

        report, _ = study.run(max_evaluations=1)

        assert len(report.refinements) == 1
        refinement = report.refinements[0]
        assert refinement is not None
        assert (
            refinement.refined_candidate.stable_id
            == report.records[0].candidate.stable_id
        )

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

    def test_optimize_uses_space_candidate_equality_for_result_refinements(
        self,
    ) -> None:
        problem = Problem(
            space=SpaceOwnedEqualitySpace(),
            objective=SpaceOwnedEqualityObjective(),
        )
        optimizer = SpaceOwnedEqualityOptimizer()
        evaluator = SequentialEvaluator[
            int | SpaceOwnedEqualityCandidate,
            SpaceOwnedEqualityCandidate,
        ]()
        study = Study(
            problem=problem,
            run_method=optimizer,
            evaluator=evaluator,
            kernel=SpaceOwnedEqualityRefinementKernel(),
        )

        result, _ = study.optimize(max_evaluations=1)

        assert len(result.refinements) == 1
        refinement = result.refinements[0]
        assert refinement is not None
        assert (
            refinement.refined_candidate.stable_id
            == result.observations[0].candidate.stable_id
        )

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
        assert (
            kernel.last_execution_resources.nested_parallelism_policy
            == NestedParallelismPolicy.FORBID
        )
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

    def test_step_rejects_sequential_model_with_batch_size_greater_than_one(
        self,
    ) -> None:
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

        with pytest.raises(
            ValueError, match="sequential execution model requires batch_size == 1"
        ):
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

        with pytest.raises(
            ValueError,
            match="run_method does not support the requested execution model: exact_async",
        ):
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

        with pytest.raises(
            ValueError, match="attempt batch requests must align with input request order"
        ):
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

    def test_optimize_budgets_by_local_optimization_cost_by_default(self) -> None:
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
        assert result.evaluation_count == 14
        assert len(final_state.tell_history) == 2
