"""Tests for exact-async Study execution."""

from collections.abc import Sequence
from typing import TypeAlias

import pytest
from typing_extensions import override

from tests.study_support import (
    AttemptOutOfOrderAsyncEvaluator,
    BatchQueueOptimizerState,
    ExactAsyncCapableBatchQueueOptimizer,
    FailingCandidateObjective,
    FailureRecordingBatchQueueOptimizer,
    FailureRecordingBatchQueueOptimizerState,
    NonResumableSessionResumableAsyncEvaluator,
    OutcomeAwareBatchQueueOptimizer,
    OutOfOrderAsyncEvaluator,
    PayloadResumableOutOfOrderAsyncEvaluator,
    RecordingKernel,
    RefinementKernel,
    ResumableOutOfOrderAsyncEvaluator,
    ScoringKernel,
    SessionRecordingAsyncEvaluator,
    ShiftedObservationProtocol,
    SpaceOwnedEqualityAsyncEvaluator,
    SpaceOwnedEqualityCandidate,
    SpaceOwnedEqualityObjective,
    SpaceOwnedEqualityOptimizer,
    SpaceOwnedEqualityOptimizerState,
    SpaceOwnedEqualitySpace,
    SquareObjective,
)
from variopt import (
    EvaluationRequest,
    IntegerSpace,
    Observation,
    Problem,
    Proposal,
    Study,
)
from variopt.algorithms.population.csa import CSAOptimizer, CSAProfile
from variopt.algorithms.population.csa.generation.proposal import CSAProposalPolicy
from variopt.artifacts import (
    EvaluationAttemptBatch,
    EvaluationSuccess,
    ObservationPayload,
    ProposalEvaluationSpec,
)
from variopt.evaluators import (
    CompletionGroup,
    EvaluationBatchHandle,
    EvaluationBatchSessionState,
    SequentialEvaluator,
)
from variopt.evaluators.async_evaluator.sessions import EvaluationBatchSession
from variopt.execution import (
    EXACT_ASYNC_EXECUTION_MODEL,
    SEQUENTIAL_EXECUTION_MODEL,
    SYNC_BATCH_EXECUTION_MODEL,
    ExecutionModel,
)
from variopt.study.common import build_evaluation_requests
from variopt.study.exact_async.orchestration import evaluate_batch_exact_async
from variopt.study.exact_async.session import StudyExactAsyncStepSession

ExactAsyncScalarStudy: TypeAlias = Study[
    int,
    int,
    BatchQueueOptimizerState,
    ObservationPayload,
    Observation[int],
]
ExactAsyncFailureRecordingStudy: TypeAlias = Study[
    int,
    int,
    FailureRecordingBatchQueueOptimizerState,
    ObservationPayload,
    Observation[int],
]
ExactAsyncSpaceOwnedEqualityStudy: TypeAlias = Study[
    int | SpaceOwnedEqualityCandidate,
    SpaceOwnedEqualityCandidate,
    SpaceOwnedEqualityOptimizerState,
    ObservationPayload,
    Observation[SpaceOwnedEqualityCandidate],
]


class InvalidCompletionAttemptSession(
    EvaluationBatchSession[EvaluationAttemptBatch[int, ObservationPayload]]
):
    """Exact-async test session that emits preconfigured malformed groups."""

    _handle: EvaluationBatchHandle
    _completion_groups: tuple[
        CompletionGroup[EvaluationAttemptBatch[int, ObservationPayload]],
        ...,
    ]
    _cancel_raises: bool
    cancel_count: int

    def __init__(
        self,
        *,
        handle: EvaluationBatchHandle,
        completion_groups: tuple[
            CompletionGroup[EvaluationAttemptBatch[int, ObservationPayload]],
            ...,
        ],
        cancel_raises: bool = False,
    ) -> None:
        self._handle = handle
        self._completion_groups = completion_groups
        self._cancel_raises = cancel_raises
        self.cancel_count = 0

    @property
    @override
    def handle(self) -> EvaluationBatchHandle:
        return self._handle

    @override
    def poll(
        self,
    ) -> tuple[CompletionGroup[EvaluationAttemptBatch[int, ObservationPayload]], ...]:
        return self._completion_groups

    @override
    def cancel(self) -> None:
        self.cancel_count += 1
        if self._cancel_raises:
            msg = "forced cancel cleanup failure"
            raise RuntimeError(msg)


class DenseAttemptPathFailsSessionEvaluator(SessionRecordingAsyncEvaluator):
    """Evaluator that must be driven through exact-async attempt sessions."""

    def evaluate_attempts(
        self,
        problem: Problem[int, int, ObservationPayload],
        requests: Sequence[EvaluationRequest[int]],
    ) -> EvaluationAttemptBatch[int, ObservationPayload]:
        _ = (problem, requests)
        msg = "dense attempt path must not be used for exact async"
        raise AssertionError(msg)


def _payload_attempt(
    request: EvaluationRequest[int],
) -> EvaluationAttemptBatch[int, ObservationPayload]:
    return EvaluationAttemptBatch(
        attempts=(
            EvaluationSuccess(
                request=request,
                payload=ObservationPayload(value=1.0, score=1.0),
            ),
        ),
    )


class OutcomeAwareExactAsyncBatchQueueOptimizer(OutcomeAwareBatchQueueOptimizer):
    """Outcome-aware batch optimizer that advertises exact-async compatibility."""

    @override
    def supported_execution_models(self) -> frozenset[ExecutionModel]:
        return frozenset(
            {
                SEQUENTIAL_EXECUTION_MODEL,
                SYNC_BATCH_EXECUTION_MODEL,
                EXACT_ASYNC_EXECUTION_MODEL,
            },
        )


class StudyExactAsyncTests:
    """Coverage for exact-async Study execution and session lifecycle."""

    def test_step_rejects_exact_async_model_with_non_async_evaluator(self) -> None:
        problem = Problem(
            space=IntegerSpace(low=0, high=10),
            objective=SquareObjective(),
        )
        optimizer = ExactAsyncCapableBatchQueueOptimizer(
            proposal_batches=[(Proposal(candidate=3, proposal_id="p-1"),)],
        )
        evaluator = SequentialEvaluator[int, int]()
        study: ExactAsyncScalarStudy = Study(
            problem=problem,
            run_method=optimizer,
            evaluator=evaluator,
        )
        state = optimizer.create_initial_state()

        with pytest.raises(
            ValueError, match="ordered_async execution models require an AsyncEvaluator"
        ):
            _ = study.step(
                state,
                execution_model=EXACT_ASYNC_EXECUTION_MODEL,
            )

    def test_csa_exact_async_matches_sync_at_checkpoint_boundary(self) -> None:
        space = IntegerSpace(low=-10, high=10)
        problem = Problem(
            space=space,
            objective=SquareObjective(),
        )
        profile = CSAProfile(
            seed_count=1,
            proposal_policy=CSAProposalPolicy(enabled=True),
        )
        sync_optimizer = CSAOptimizer.from_space_defaults(
            space=space,
            bank_capacity=2,
            profile=profile,
            random_state=7,
        )
        async_optimizer = CSAOptimizer.from_space_defaults(
            space=space,
            bank_capacity=2,
            profile=profile,
            random_state=7,
        )
        sync_study = Study(
            problem=problem,
            run_method=sync_optimizer,
            evaluator=SequentialEvaluator[int, int](),
        )
        async_study = Study(
            problem=problem,
            run_method=async_optimizer,
            evaluator=OutOfOrderAsyncEvaluator(),
        )

        sync_result, sync_state = sync_study.optimize(
            max_evaluations=12,
            batch_size=2,
            execution_model=SYNC_BATCH_EXECUTION_MODEL,
            stop_at_checkpoint_boundary=True,
        )
        async_result, async_state = async_study.optimize(
            max_evaluations=12,
            batch_size=2,
            execution_model=EXACT_ASYNC_EXECUTION_MODEL,
            stop_at_checkpoint_boundary=True,
        )

        assert tuple(
            (observation.candidate, observation.score)
            for observation in async_result.observations
        ) == tuple(
            (observation.candidate, observation.score)
            for observation in sync_result.observations
        )
        assert async_optimizer.state_to_dict(
            async_state
        ) == sync_optimizer.state_to_dict(
            sync_state,
        )

    def test_step_exact_async_reorders_out_of_order_completions(self) -> None:
        problem = Problem(
            space=IntegerSpace(low=0, high=10),
            objective=SquareObjective(),
        )
        optimizer = ExactAsyncCapableBatchQueueOptimizer(
            proposal_batches=[
                (
                    Proposal(candidate=4, proposal_id="p-1"),
                    Proposal(candidate=2, proposal_id="p-2"),
                ),
            ],
        )
        evaluator = OutOfOrderAsyncEvaluator()
        study = Study(problem=problem, run_method=optimizer, evaluator=evaluator)
        state = optimizer.create_initial_state()

        observations, next_state = study.step(
            state,
            batch_size=2,
            execution_model=EXACT_ASYNC_EXECUTION_MODEL,
        )

        assert tuple(
            observation.proposal.proposal_id for observation in observations
        ) == ("p-1", "p-2")
        assert tuple(observation.value for observation in observations) == (16.0, 4.0)
        assert tuple(
            observation.proposal.proposal_id
            for observation in next_state.tell_history[0]
        ) == ("p-1", "p-2")

    def test_step_exact_async_opens_batch_session(self) -> None:
        problem = Problem(
            space=IntegerSpace(low=0, high=10),
            objective=SquareObjective(),
        )
        optimizer = ExactAsyncCapableBatchQueueOptimizer(
            proposal_batches=[
                (
                    Proposal(candidate=4, proposal_id="p-1"),
                    Proposal(candidate=2, proposal_id="p-2"),
                ),
            ],
        )
        evaluator = SessionRecordingAsyncEvaluator()
        study = Study(problem=problem, run_method=optimizer, evaluator=evaluator)

        _records, _next_state = study.step(
            optimizer.create_initial_state(),
            batch_size=2,
            execution_model=EXACT_ASYNC_EXECUTION_MODEL,
        )

        assert evaluator.opened_batch_sizes == (2,)

    def test_step_exact_async_prefers_attempt_session_over_dense_attempts(
        self,
    ) -> None:
        problem = Problem(
            space=IntegerSpace(low=0, high=10),
            objective=SquareObjective(),
        )
        optimizer = ExactAsyncCapableBatchQueueOptimizer(
            proposal_batches=[
                (
                    Proposal(candidate=4, proposal_id="p-1"),
                    Proposal(candidate=2, proposal_id="p-2"),
                ),
            ],
        )
        evaluator = DenseAttemptPathFailsSessionEvaluator()
        study = Study(problem=problem, run_method=optimizer, evaluator=evaluator)

        records, _next_state = study.step(
            optimizer.create_initial_state(),
            batch_size=2,
            execution_model=EXACT_ASYNC_EXECUTION_MODEL,
        )

        assert evaluator.opened_batch_sizes == (2,)
        assert tuple(record.value for record in records) == (16.0, 4.0)

    def test_step_exact_async_preserves_dense_attempt_fallback(self) -> None:
        problem = Problem(
            space=IntegerSpace(low=0, high=10),
            objective=SquareObjective(),
        )
        requests = build_evaluation_requests(
            proposals=(
                Proposal(candidate=5, proposal_id="p-1"),
                Proposal(candidate=3, proposal_id="p-2"),
            ),
            proposal_evaluation_specs=None,
        )

        attempts = evaluate_batch_exact_async(
            SequentialEvaluator(),
            problem=problem,
            requests=requests,
        )

        assert tuple(success.payload.value for success in attempts.successes) == (
            25.0,
            9.0,
        )

    def test_run_exact_async_returns_checkpoint_snapshot_when_kernel_cost_exhausts_budget(
        self,
    ) -> None:
        problem = Problem(
            space=IntegerSpace(low=0, high=10),
            objective=SquareObjective(),
        )
        optimizer = ExactAsyncCapableBatchQueueOptimizer(
            proposal_batches=[(Proposal(candidate=4, proposal_id="p-1"),)],
        )
        evaluator = OutOfOrderAsyncEvaluator()
        study: ExactAsyncScalarStudy = Study(
            problem=problem,
            run_method=optimizer,
            evaluator=evaluator,
            kernel=ScoringKernel(),
        )

        report, final_state = study.run(
            max_evaluations=3,
            execution_model=EXACT_ASYNC_EXECUTION_MODEL,
            stop_at_checkpoint_boundary=True,
        )

        assert report.records == ()
        assert report.evaluation_count == 0
        assert final_state == optimizer.create_initial_state()

    def test_step_exact_async_uses_space_candidate_equality_for_refinement(
        self,
    ) -> None:
        problem = Problem(
            space=SpaceOwnedEqualitySpace(),
            objective=SpaceOwnedEqualityObjective(),
        )
        optimizer = SpaceOwnedEqualityOptimizer()
        evaluator = SpaceOwnedEqualityAsyncEvaluator()
        study = Study(problem=problem, run_method=optimizer, evaluator=evaluator)

        observations, _ = study.step(
            optimizer.create_initial_state(),
            execution_model=EXACT_ASYNC_EXECUTION_MODEL,
        )

        assert len(observations) == 1
        assert observations[0].candidate.stable_id == 1

    def test_direct_step_exact_async_reuses_request_batch_for_validation(
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
        optimizer = ExactAsyncCapableBatchQueueOptimizer(
            proposal_batches=[(Proposal(candidate=3, proposal_id="p-1"),)],
        )
        evaluator = OutOfOrderAsyncEvaluator()
        study = Study(problem=problem, run_method=optimizer, evaluator=evaluator)

        _ = study.step(
            optimizer.create_initial_state(),
            execution_model=EXACT_ASYNC_EXECUTION_MODEL,
        )

        assert build_call_count == 1

    def test_step_exact_async_runs_through_custom_kernel(self) -> None:
        problem = Problem(
            space=IntegerSpace(low=0, high=10),
            objective=SquareObjective(),
        )
        optimizer = ExactAsyncCapableBatchQueueOptimizer(
            proposal_batches=[(Proposal(candidate=4, proposal_id="p-1"),)],
        )
        evaluator = OutOfOrderAsyncEvaluator()
        kernel = RecordingKernel()
        study: Study[
            int,
            int,
            BatchQueueOptimizerState,
            ObservationPayload,
            Observation[int],
        ] = Study(
            problem=problem,
            run_method=optimizer,
            evaluator=evaluator,
            kernel=kernel,
        )
        state = optimizer.create_initial_state()

        observations, _ = study.step(
            state,
            batch_size=1,
            execution_model=EXACT_ASYNC_EXECUTION_MODEL,
        )

        assert len(kernel.queries) == 1
        assert kernel.queries[0].proposals[0].proposal_id == "p-1"
        assert observations[0].proposal.proposal_id == "p-1"

    def test_run_exact_async_preserves_report_refinement_order(self) -> None:
        problem = Problem(
            space=IntegerSpace(low=0, high=10),
            objective=SquareObjective(),
        )
        optimizer = ExactAsyncCapableBatchQueueOptimizer(
            proposal_batches=[
                (
                    Proposal(candidate=4, proposal_id="p-1"),
                    Proposal(candidate=2, proposal_id="p-2"),
                ),
            ],
        )
        evaluator = OutOfOrderAsyncEvaluator()
        study: ExactAsyncScalarStudy = Study(
            problem=problem,
            run_method=optimizer,
            evaluator=evaluator,
            kernel=RefinementKernel(),
        )

        report, next_state = study.run(
            max_evaluations=2,
            batch_size=2,
            execution_model=EXACT_ASYNC_EXECUTION_MODEL,
        )

        assert tuple(
            observation.proposal.proposal_id for observation in report.records
        ) == (
            "p-1",
            "p-2",
        )
        assert tuple(observation.candidate for observation in report.records) == (3, 1)
        assert len(report.refinements) == 2
        first_refinement = report.refinements[0]
        second_refinement = report.refinements[1]
        assert first_refinement is not None
        assert second_refinement is not None
        assert first_refinement.source_candidate == 4
        assert second_refinement.source_candidate == 2
        assert tuple(
            observation.proposal.proposal_id
            for observation in next_state.tell_history[0]
        ) == ("p-1", "p-2")

    def test_run_exact_async_preserves_evaluator_refinement_order(self) -> None:
        problem = Problem(
            space=IntegerSpace(low=0, high=10),
            evaluation_protocol=ShiftedObservationProtocol(),
        )
        optimizer = ExactAsyncCapableBatchQueueOptimizer(
            proposal_batches=[
                (
                    Proposal(candidate=4, proposal_id="p-1"),
                    Proposal(candidate=2, proposal_id="p-2"),
                ),
            ],
        )
        evaluator = OutOfOrderAsyncEvaluator(attach_refinement=True)
        study: ExactAsyncScalarStudy = Study(
            problem=problem,
            run_method=optimizer,
            evaluator=evaluator,
        )

        report, next_state = study.run(
            max_evaluations=2,
            batch_size=2,
            execution_model=EXACT_ASYNC_EXECUTION_MODEL,
        )

        assert tuple(
            observation.proposal.proposal_id for observation in report.records
        ) == (
            "p-1",
            "p-2",
        )
        assert tuple(observation.candidate for observation in report.records) == (4, 2)
        assert len(report.refinements) == 2
        first_refinement = report.refinements[0]
        second_refinement = report.refinements[1]
        assert first_refinement is not None
        assert second_refinement is not None
        assert first_refinement.source_candidate == 4
        assert first_refinement.refined_candidate == 4
        assert second_refinement.source_candidate == 2
        assert second_refinement.refined_candidate == 2
        assert tuple(
            observation.proposal.proposal_id
            for observation in next_state.tell_history[0]
        ) == ("p-1", "p-2")

    def test_run_exact_async_records_out_of_order_attempt_failures(self) -> None:
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
        evaluator = AttemptOutOfOrderAsyncEvaluator()
        study: ExactAsyncFailureRecordingStudy = Study(
            problem=problem,
            run_method=optimizer,
            evaluator=evaluator,
        )

        report, final_state = study.run(
            max_evaluations=3,
            batch_size=3,
            execution_model=EXACT_ASYNC_EXECUTION_MODEL,
        )

        assert tuple(record.proposal.proposal_id for record in report.records) == (
            "p-1",
            "p-3",
        )
        assert all(type(success.payload) is Observation for success in report.successes)
        assert tuple(failure.proposal_id for failure in report.failures) == ("p-2",)
        assert report.evaluation_count == 3
        assert final_state.tell_history == ((report.records[0], report.records[1]),)
        assert tuple(
            failure_proposal_id
            for failure_batch in final_state.failure_history
            for failure_proposal_id in failure_batch
        ) == ("p-2",)

    def test_optimize_exact_async_projects_evaluator_refinements(self) -> None:
        problem = Problem(
            space=IntegerSpace(low=0, high=10),
            evaluation_protocol=ShiftedObservationProtocol(),
        )
        optimizer = ExactAsyncCapableBatchQueueOptimizer(
            proposal_batches=[
                (
                    Proposal(candidate=4, proposal_id="p-1"),
                    Proposal(candidate=2, proposal_id="p-2"),
                ),
            ],
        )
        evaluator = OutOfOrderAsyncEvaluator(attach_refinement=True)
        study = Study(problem=problem, run_method=optimizer, evaluator=evaluator)

        result, next_state = study.optimize(
            max_evaluations=2,
            batch_size=2,
            execution_model=EXACT_ASYNC_EXECUTION_MODEL,
        )

        assert tuple(
            observation.proposal.proposal_id for observation in result.observations
        ) == (
            "p-1",
            "p-2",
        )
        assert tuple(observation.candidate for observation in result.observations) == (
            4,
            2,
        )
        assert len(result.refinements) == 2
        first_refinement = result.refinements[0]
        second_refinement = result.refinements[1]
        assert first_refinement is not None
        assert second_refinement is not None
        assert first_refinement.source_candidate == 4
        assert first_refinement.refined_candidate == 4
        assert second_refinement.source_candidate == 2
        assert second_refinement.refined_candidate == 2
        assert tuple(
            observation.proposal.proposal_id
            for observation in next_state.tell_history[0]
        ) == ("p-1", "p-2")

    def test_open_exact_async_step_session_rejects_non_resumable_async_evaluator(
        self,
    ) -> None:
        problem = Problem(
            space=IntegerSpace(low=0, high=10),
            objective=SquareObjective(),
        )
        optimizer = ExactAsyncCapableBatchQueueOptimizer(
            proposal_batches=[(Proposal(candidate=4, proposal_id="p-1"),)],
        )
        evaluator = OutOfOrderAsyncEvaluator()
        study = Study(problem=problem, run_method=optimizer, evaluator=evaluator)

        with pytest.raises(
            TypeError,
            match="study-level resumable exact_async orchestration requires a ResumableAsyncEvaluator",
        ):
            _ = study.open_exact_async_step_session(
                optimizer.create_initial_state(),
                batch_size=1,
            )

    def test_open_exact_async_step_session_cancels_non_resumable_attempt_session(
        self,
    ) -> None:
        problem = Problem(
            space=IntegerSpace(low=0, high=10),
            objective=SquareObjective(),
        )
        optimizer = ExactAsyncCapableBatchQueueOptimizer(
            proposal_batches=[(Proposal(candidate=4, proposal_id="p-1"),)],
        )
        evaluator = NonResumableSessionResumableAsyncEvaluator()
        study = Study(problem=problem, run_method=optimizer, evaluator=evaluator)

        with pytest.raises(
            TypeError,
            match="resumable async evaluator returned a non-resumable batch session",
        ):
            _ = study.open_exact_async_step_session(
                optimizer.create_initial_state(),
                batch_size=1,
            )
        assert evaluator.pending_attempt_batch_ids == ()

    def test_open_exact_async_step_session_rejects_non_direct_kernel(self) -> None:
        problem = Problem(
            space=IntegerSpace(low=0, high=10),
            objective=SquareObjective(),
        )
        optimizer = ExactAsyncCapableBatchQueueOptimizer(
            proposal_batches=[(Proposal(candidate=4, proposal_id="p-1"),)],
        )
        evaluator = ResumableOutOfOrderAsyncEvaluator()
        study = Study(
            problem=problem,
            run_method=optimizer,
            evaluator=evaluator,
            kernel=RecordingKernel(),
        )

        with pytest.raises(
            ValueError,
            match="study-level resumable exact_async orchestration currently requires DirectKernel",
        ):
            _ = study.open_exact_async_step_session(
                optimizer.create_initial_state(),
                batch_size=1,
            )

    def test_suspend_and_resume_exact_async_step_session_preserves_tell_order(
        self,
    ) -> None:
        problem = Problem(
            space=IntegerSpace(low=0, high=10),
            objective=SquareObjective(),
        )
        optimizer = ExactAsyncCapableBatchQueueOptimizer(
            proposal_batches=[
                (
                    Proposal(candidate=4, proposal_id="p-1"),
                    Proposal(candidate=2, proposal_id="p-2"),
                ),
            ],
        )
        evaluator = ResumableOutOfOrderAsyncEvaluator()
        study = Study(problem=problem, run_method=optimizer, evaluator=evaluator)

        session = study.open_exact_async_step_session(
            optimizer.create_initial_state(),
            batch_size=2,
        )
        first_completion_groups = tuple(session.poll())
        resume_handle = session.suspend()

        assert session.state() == EvaluationBatchSessionState(
            request_count=2,
            completed_count=1,
            pending_count=1,
            lifecycle="suspended",
        )
        assert len(first_completion_groups) == 1
        assert first_completion_groups[0].start_index == 1

        resumed_session = study.resume_exact_async_step_session(resume_handle)
        observations, next_state = resumed_session.finish()

        assert tuple(
            observation.proposal.proposal_id for observation in observations
        ) == ("p-1", "p-2")
        assert tuple(observation.value for observation in observations) == (16.0, 4.0)
        assert tuple(
            observation.proposal.proposal_id
            for observation in next_state.tell_history[0]
        ) == ("p-1", "p-2")

    @pytest.mark.parametrize("method_name", ["poll", "wait"])
    @pytest.mark.parametrize("malformation", ["overlap", "out_of_bounds"])
    def test_exact_async_step_session_validation_failure_is_terminal(
        self,
        method_name: str,
        malformation: str,
    ) -> None:
        problem = Problem(
            space=IntegerSpace(low=0, high=10),
            objective=SquareObjective(),
        )
        optimizer = ExactAsyncCapableBatchQueueOptimizer(
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
            evaluator=SequentialEvaluator[int, int](),
        )
        proposals, post_ask_state = optimizer.ask(
            optimizer.create_initial_state(),
            batch_size=2,
        )
        requests = build_evaluation_requests(
            proposals,
            proposal_evaluation_specs=None,
        )
        first_attempt = _payload_attempt(requests[0])
        second_attempt = _payload_attempt(requests[1])

        ordered_attempts: list[
            EvaluationAttemptBatch[int, ObservationPayload] | None
        ] = []
        if malformation == "overlap":
            ordered_attempts = [first_attempt, None]
            completion_group = CompletionGroup(
                start_index=0,
                outcomes=(second_attempt,),
            )
            expected_message = "overlap"
        else:
            completion_group = CompletionGroup(
                start_index=1,
                outcomes=(first_attempt, second_attempt),
            )
            expected_message = "bounds"

        batch_session = InvalidCompletionAttemptSession(
            handle=EvaluationBatchHandle(
                batch_id=f"invalid-{method_name}-{malformation}",
                request_count=2,
            ),
            completion_groups=(completion_group,),
        )
        session = StudyExactAsyncStepSession[
            int,
            int,
            BatchQueueOptimizerState,
            ObservationPayload,
            Observation[int],
        ](
            study=study,
            requests=requests,
            post_ask_state=post_ask_state,
            batch_session=batch_session,
            candidate_equal=study.problem.space.candidates_equal,
            ordered_attempts=ordered_attempts,
        )

        with pytest.raises(ValueError, match=expected_message):
            if method_name == "poll":
                _ = session.poll()
            else:
                _ = session.wait(timeout=0.0)

        assert batch_session.cancel_count == 1
        assert session.state().lifecycle == "failed"
        with pytest.raises(RuntimeError, match="no longer active"):
            _ = session.poll()

    def test_exact_async_step_session_cancel_failure_does_not_mask_validation_failure(
        self,
    ) -> None:
        problem = Problem(
            space=IntegerSpace(low=0, high=10),
            objective=SquareObjective(),
        )
        optimizer = ExactAsyncCapableBatchQueueOptimizer(
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
            evaluator=SequentialEvaluator[int, int](),
        )
        proposals, post_ask_state = optimizer.ask(
            optimizer.create_initial_state(),
            batch_size=2,
        )
        requests = build_evaluation_requests(
            proposals,
            proposal_evaluation_specs=None,
        )
        attempt = _payload_attempt(requests[0])
        batch_session = InvalidCompletionAttemptSession(
            handle=EvaluationBatchHandle(
                batch_id="invalid-cancel-failure",
                request_count=2,
            ),
            completion_groups=(
                CompletionGroup(
                    start_index=1,
                    outcomes=(attempt, attempt),
                ),
            ),
            cancel_raises=True,
        )
        session = StudyExactAsyncStepSession[
            int,
            int,
            BatchQueueOptimizerState,
            ObservationPayload,
            Observation[int],
        ](
            study=study,
            requests=requests,
            post_ask_state=post_ask_state,
            batch_session=batch_session,
            candidate_equal=study.problem.space.candidates_equal,
        )

        with pytest.raises(ValueError, match="bounds"):
            _ = session.poll()

        assert batch_session.cancel_count == 1
        assert session.state().lifecycle == "failed"

    def test_resume_handle_stores_payload_attempts_until_finish(self) -> None:
        problem = Problem(
            space=IntegerSpace(low=0, high=10),
            objective=SquareObjective(),
        )
        optimizer = ExactAsyncCapableBatchQueueOptimizer(
            proposal_batches=[
                (
                    Proposal(candidate=4, proposal_id="p-1"),
                    Proposal(candidate=2, proposal_id="p-2"),
                ),
            ],
        )
        evaluator = PayloadResumableOutOfOrderAsyncEvaluator()
        study = Study(problem=problem, run_method=optimizer, evaluator=evaluator)

        session = study.open_exact_async_step_session(
            optimizer.create_initial_state(),
            batch_size=2,
        )
        _ = tuple(session.poll())
        resume_handle = session.suspend()
        stored_attempt = resume_handle.ordered_attempts[1]
        assert stored_attempt is not None
        stored_success = stored_attempt.single_success_or_none()

        assert stored_success is not None
        assert type(stored_success.payload) is ObservationPayload

        resumed_session = study.resume_exact_async_step_session(resume_handle)
        observations, next_state = resumed_session.finish()

        assert all(isinstance(observation, Observation) for observation in observations)
        assert tuple(
            observation.proposal.proposal_id for observation in observations
        ) == (
            "p-1",
            "p-2",
        )
        assert tuple(observation.value for observation in observations) == (16.0, 4.0)
        assert next_state.tell_history == (observations,)

    def test_suspend_and_resume_exact_async_step_session_preserves_refinement_payload(
        self,
    ) -> None:
        problem = Problem(
            space=IntegerSpace(low=0, high=10),
            evaluation_protocol=ShiftedObservationProtocol(),
        )
        optimizer = ExactAsyncCapableBatchQueueOptimizer(
            proposal_batches=[
                (
                    Proposal(candidate=4, proposal_id="p-1"),
                    Proposal(candidate=2, proposal_id="p-2"),
                ),
            ],
        )
        evaluator = ResumableOutOfOrderAsyncEvaluator(attach_refinement=True)
        study = Study(problem=problem, run_method=optimizer, evaluator=evaluator)

        session = study.open_exact_async_step_session(
            optimizer.create_initial_state(),
            batch_size=2,
        )
        first_completion_groups = tuple(session.poll())
        resume_handle = session.suspend()
        stored_attempt = resume_handle.ordered_attempts[1]
        assert stored_attempt is not None
        stored_outcome = stored_attempt.single_success_or_none()

        assert len(first_completion_groups) == 1
        assert stored_outcome is not None
        assert stored_outcome.refinement is not None
        assert stored_outcome.refinement.source_candidate == 2
        assert stored_outcome.refinement.refined_candidate == 2

        resumed_session = study.resume_exact_async_step_session(resume_handle)
        observations, next_state = resumed_session.finish()
        first_resumed_attempt = resumed_session.ordered_attempts[0]
        second_resumed_attempt = resumed_session.ordered_attempts[1]
        assert first_resumed_attempt is not None
        assert second_resumed_attempt is not None
        first_resumed_outcome = first_resumed_attempt.single_success_or_none()
        second_resumed_outcome = second_resumed_attempt.single_success_or_none()

        assert tuple(observation.candidate for observation in observations) == (4, 2)
        assert first_resumed_outcome is not None
        assert second_resumed_outcome is not None
        assert first_resumed_outcome.refinement is not None
        assert second_resumed_outcome.refinement is not None
        assert first_resumed_outcome.refinement.source_candidate == 4
        assert second_resumed_outcome.refinement.source_candidate == 2
        assert tuple(
            observation.proposal.proposal_id
            for observation in next_state.tell_history[0]
        ) == ("p-1", "p-2")

    def test_suspend_and_resume_exact_async_step_session_preserves_outcome_feedback_order(
        self,
    ) -> None:
        problem = Problem(
            space=IntegerSpace(low=0, high=10),
            evaluation_protocol=ShiftedObservationProtocol(),
        )
        optimizer = OutcomeAwareExactAsyncBatchQueueOptimizer(
            proposal_batches=[
                (
                    Proposal(candidate=4, proposal_id="p-1"),
                    Proposal(candidate=2, proposal_id="p-2"),
                ),
            ],
        )
        evaluator = ResumableOutOfOrderAsyncEvaluator(attach_refinement=True)
        study = Study(problem=problem, run_method=optimizer, evaluator=evaluator)

        session = study.open_exact_async_step_session(
            optimizer.create_initial_state(),
            batch_size=2,
        )
        _ = tuple(session.poll())
        resume_handle = session.suspend()

        resumed_session = study.resume_exact_async_step_session(resume_handle)
        observations, _ = resumed_session.finish()

        assert tuple(
            observation.proposal.proposal_id for observation in observations
        ) == (
            "p-1",
            "p-2",
        )
        assert optimizer.seen_changed_leaf_paths == (((),), ((),))
