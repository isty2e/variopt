"""Tests for Study materializer output contract validation."""

from collections.abc import Callable
from enum import Enum
from typing import TypeAlias

import pytest
from typing_extensions import override

from tests.study_support import (
    BatchQueueOptimizer,
    ExactAsyncCapableBatchQueueOptimizer,
    OutOfOrderAsyncEvaluator,
    ResumableOutOfOrderAsyncEvaluator,
    RollingStaleAsyncOptimizer,
    SquareObjective,
)
from variopt import (
    CandidateRefinement,
    EvaluationAttemptBatch,
    EvaluationRequest,
    IntegerSpace,
    Observation,
    OptimizationDirection,
    Problem,
    Proposal,
    RunExecutionFailed,
    Study,
)
from variopt.artifacts import (
    DefaultEvaluationAttemptMaterializer,
    EvaluationAttemptMaterializer,
    EvaluationFailure,
    EvaluationSuccess,
    KernelDiagnostics,
    KernelStatus,
    ObservationPayload,
)
from variopt.evaluators import SequentialEvaluator
from variopt.execution import EXACT_ASYNC_EXECUTION_MODEL, STALE_ASYNC_EXECUTION_MODEL
from variopt.study.common import validate_materialized_attempts

ObservationAttempt: TypeAlias = (
    EvaluationSuccess[int, Observation[int]] | EvaluationFailure[int]
)


class AttemptMaterializerCorruption(Enum):
    """Materializer contract violation modes for Study boundary tests."""

    DROP_FIRST_SLOT = "drop-first-slot"
    REORDER_SLOTS = "reorder-slots"
    SUCCESS_TO_FAILURE = "success-to-failure"
    FAILURE_TO_SUCCESS = "failure-to-success"
    DROP_FAILURE_METADATA = "drop-failure-metadata"
    DROP_SUCCESS_EVALUATION_COUNT = "drop-success-evaluation-count"
    DROP_SUCCESS_REFINEMENT = "drop-success-refinement"
    DROP_SUCCESS_DIAGNOSTICS = "drop-success-diagnostics"
    NON_RECORD_SUCCESS_PAYLOAD = "non-record-success-payload"


class CorruptingObservationMaterializer(
    EvaluationAttemptMaterializer[int, ObservationPayload, Observation[int]],
):
    """Observation materializer that deliberately violates one contract axis."""

    _default: DefaultEvaluationAttemptMaterializer[int]
    _corruption: AttemptMaterializerCorruption

    def __init__(self, corruption: AttemptMaterializerCorruption) -> None:
        self._default = DefaultEvaluationAttemptMaterializer()
        self._corruption = corruption

    @override
    def materialize_attempts(
        self,
        attempts: EvaluationAttemptBatch[int, ObservationPayload],
    ) -> EvaluationAttemptBatch[int, Observation[int]]:
        materialized_attempts = self._default.materialize_attempts(attempts)
        if self._corruption is AttemptMaterializerCorruption.DROP_FIRST_SLOT:
            return EvaluationAttemptBatch(
                attempts=materialized_attempts.attempts[1:],
            )

        if self._corruption is AttemptMaterializerCorruption.REORDER_SLOTS:
            return EvaluationAttemptBatch(
                attempts=tuple(reversed(materialized_attempts.attempts)),
            )

        if self._corruption is AttemptMaterializerCorruption.SUCCESS_TO_FAILURE:
            return self._replace_first_success(
                materialized_attempts,
                lambda success: EvaluationFailure[int].from_exception(
                    request=success.request,
                    exception=RuntimeError("materializer flipped success"),
                    evaluation_count=success.evaluation_count,
                ),
            )

        if self._corruption is AttemptMaterializerCorruption.FAILURE_TO_SUCCESS:
            return self._replace_first_failure(
                materialized_attempts,
                self._failure_to_success,
            )

        if self._corruption is AttemptMaterializerCorruption.DROP_FAILURE_METADATA:
            return self._replace_first_failure(
                materialized_attempts,
                lambda failure: EvaluationFailure[int].from_exception(
                    request=failure.request,
                    exception=RuntimeError("materializer lost failure metadata"),
                    evaluation_count=failure.evaluation_count + 1,
                ),
            )

        if (
            self._corruption
            is AttemptMaterializerCorruption.DROP_SUCCESS_EVALUATION_COUNT
        ):
            return self._replace_first_success(
                materialized_attempts,
                lambda success: EvaluationSuccess(
                    request=success.request,
                    payload=success.payload,
                    evaluation_count=max(0, success.evaluation_count - 1),
                    refinement=success.refinement,
                    kernel_diagnostics=success.kernel_diagnostics,
                ),
            )

        if self._corruption is AttemptMaterializerCorruption.DROP_SUCCESS_REFINEMENT:
            return self._replace_first_success(
                materialized_attempts,
                lambda success: EvaluationSuccess(
                    request=success.request,
                    payload=success.payload,
                    evaluation_count=success.evaluation_count,
                    kernel_diagnostics=success.kernel_diagnostics,
                ),
            )

        if self._corruption is AttemptMaterializerCorruption.DROP_SUCCESS_DIAGNOSTICS:
            return self._replace_first_success(
                materialized_attempts,
                lambda success: EvaluationSuccess(
                    request=success.request,
                    payload=success.payload,
                    evaluation_count=success.evaluation_count,
                    refinement=success.refinement,
                ),
            )

        if self._corruption is AttemptMaterializerCorruption.NON_RECORD_SUCCESS_PAYLOAD:
            return self._replace_first_success(
                materialized_attempts,
                self._replace_success_payload_with_request_free_payload,
            )

        msg = "unsupported materializer corruption mode"
        raise RuntimeError(msg)

    def _replace_first_success(
        self,
        attempts: EvaluationAttemptBatch[int, Observation[int]],
        replacement: Callable[
            [EvaluationSuccess[int, Observation[int]]],
            ObservationAttempt,
        ],
    ) -> EvaluationAttemptBatch[int, Observation[int]]:
        replaced_attempts: list[ObservationAttempt] = []
        replaced = False
        for attempt in attempts.attempts:
            if not replaced and type(attempt) is EvaluationSuccess:
                replaced_attempts.append(replacement(attempt))
                replaced = True
                continue
            replaced_attempts.append(attempt)

        if not replaced:
            msg = "corruption mode requires at least one successful attempt"
            raise RuntimeError(msg)

        return EvaluationAttemptBatch(attempts=tuple(replaced_attempts))

    def _replace_first_failure(
        self,
        attempts: EvaluationAttemptBatch[int, Observation[int]],
        replacement: Callable[[EvaluationFailure[int]], ObservationAttempt],
    ) -> EvaluationAttemptBatch[int, Observation[int]]:
        replaced_attempts: list[ObservationAttempt] = []
        replaced = False
        for attempt in attempts.attempts:
            if not replaced and type(attempt) is EvaluationFailure:
                replaced_attempts.append(replacement(attempt))
                replaced = True
                continue
            replaced_attempts.append(attempt)

        if not replaced:
            msg = "corruption mode requires at least one failed attempt"
            raise RuntimeError(msg)

        return EvaluationAttemptBatch(attempts=tuple(replaced_attempts))

    def _failure_to_success(
        self,
        failure: EvaluationFailure[int],
    ) -> EvaluationSuccess[int, Observation[int]]:
        observation = Observation(
            request=failure.request,
            candidate=failure.candidate,
            value=0.0,
            score=0.0,
        )
        return EvaluationSuccess(
            request=failure.request,
            payload=observation,
            evaluation_count=failure.evaluation_count,
        )

    def _replace_success_payload_with_request_free_payload(
        self,
        success: EvaluationSuccess[int, Observation[int]],
    ) -> EvaluationSuccess[int, Observation[int]]:
        corrupt_success = success.with_payload(success.payload)
        object.__setattr__(
            corrupt_success,
            "payload",
            ObservationPayload(
                value=success.payload.value,
                score=success.payload.score,
                elapsed_seconds=success.payload.elapsed_seconds,
            ),
        )
        return corrupt_success


def _metadata_rich_source_attempts() -> EvaluationAttemptBatch[int, ObservationPayload]:
    source_request = EvaluationRequest(
        proposal=Proposal(candidate=1, proposal_id="p-success-source"),
    )
    refined_request = EvaluationRequest(
        proposal=Proposal(candidate=2, proposal_id="p-success-source"),
    )
    failure_request = EvaluationRequest(
        proposal=Proposal(candidate=3, proposal_id="p-failure"),
    )
    success = EvaluationSuccess(
        request=refined_request,
        payload=ObservationPayload.from_objective_value(
            value=4.0,
            direction=OptimizationDirection.MINIMIZE,
        ),
        evaluation_count=3,
        refinement=CandidateRefinement(
            source_candidate=source_request.candidate,
            refined_candidate=refined_request.candidate,
            changed_leaf_paths=((),),
        ),
        kernel_diagnostics=KernelDiagnostics(
            backend="test",
            status=KernelStatus.CONVERGED,
        ),
    )
    failure = EvaluationFailure[int].from_exception(
        request=failure_request,
        exception=RuntimeError("source failure"),
        evaluation_count=2,
    )
    return EvaluationAttemptBatch(attempts=(success, failure))


@pytest.mark.parametrize(
    ("corruption", "exception_type", "message"),
    [
        (
            AttemptMaterializerCorruption.DROP_FIRST_SLOT,
            ValueError,
            "preserve attempt slot count",
        ),
        (
            AttemptMaterializerCorruption.REORDER_SLOTS,
            ValueError,
            "preserve success slots",
        ),
        (
            AttemptMaterializerCorruption.SUCCESS_TO_FAILURE,
            ValueError,
            "preserve success slots",
        ),
        (
            AttemptMaterializerCorruption.FAILURE_TO_SUCCESS,
            ValueError,
            "preserve failure slots",
        ),
        (
            AttemptMaterializerCorruption.DROP_FAILURE_METADATA,
            ValueError,
            "preserve exception and evaluation_count",
        ),
        (
            AttemptMaterializerCorruption.DROP_SUCCESS_EVALUATION_COUNT,
            ValueError,
            "preserve evaluation_count",
        ),
        (
            AttemptMaterializerCorruption.DROP_SUCCESS_REFINEMENT,
            ValueError,
            "preserve refinement",
        ),
        (
            AttemptMaterializerCorruption.DROP_SUCCESS_DIAGNOSTICS,
            ValueError,
            "preserve kernel_diagnostics",
        ),
        (
            AttemptMaterializerCorruption.NON_RECORD_SUCCESS_PAYLOAD,
            TypeError,
            "request-aligned record",
        ),
    ],
)
def test_validate_materialized_attempts_rejects_contract_violations(
    corruption: AttemptMaterializerCorruption,
    exception_type: type[Exception],
    message: str,
) -> None:
    source_attempts = _metadata_rich_source_attempts()
    materializer = CorruptingObservationMaterializer(corruption)
    materialized_attempts = materializer.materialize_attempts(source_attempts)

    with pytest.raises(exception_type, match=message):
        validate_materialized_attempts(source_attempts, materialized_attempts)


def test_sync_step_rejects_materializer_slot_drop() -> None:
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
    evaluator = SequentialEvaluator[int, int]()
    study = Study(
        problem=problem,
        run_method=optimizer,
        evaluator=evaluator,
        attempt_materializer=CorruptingObservationMaterializer(
            AttemptMaterializerCorruption.DROP_FIRST_SLOT,
        ),
    )

    with pytest.raises(ValueError, match="preserve attempt slot count"):
        _ = study.step(optimizer.create_initial_state(), batch_size=2)


def test_exact_async_step_rejects_materializer_variant_flip() -> None:
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
    study = Study(
        problem=problem,
        run_method=optimizer,
        evaluator=evaluator,
        attempt_materializer=CorruptingObservationMaterializer(
            AttemptMaterializerCorruption.SUCCESS_TO_FAILURE,
        ),
    )

    with pytest.raises(ValueError, match="preserve success slots"):
        _ = study.step(
            optimizer.create_initial_state(),
            batch_size=2,
            execution_model=EXACT_ASYNC_EXECUTION_MODEL,
        )


def test_exact_async_step_session_rejects_materializer_reorder() -> None:
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
    study = Study(
        problem=problem,
        run_method=optimizer,
        evaluator=evaluator,
        attempt_materializer=CorruptingObservationMaterializer(
            AttemptMaterializerCorruption.REORDER_SLOTS,
        ),
    )
    session = study.open_exact_async_step_session(
        optimizer.create_initial_state(),
        batch_size=2,
    )

    with pytest.raises(ValueError, match="preserve source request identity"):
        _ = session.finish()


def test_stale_async_run_rejects_non_record_materialized_payload() -> None:
    problem = Problem(
        space=IntegerSpace(low=0, high=10),
        objective=SquareObjective(),
    )
    optimizer = RollingStaleAsyncOptimizer(
        proposals=(
            Proposal(candidate=4, proposal_id="p-1"),
            Proposal(candidate=2, proposal_id="p-2"),
        ),
    )
    evaluator = OutOfOrderAsyncEvaluator()
    study = Study(
        problem=problem,
        run_method=optimizer,
        evaluator=evaluator,
        attempt_materializer=CorruptingObservationMaterializer(
            AttemptMaterializerCorruption.NON_RECORD_SUCCESS_PAYLOAD,
        ),
    )

    with pytest.raises(RuntimeError, match="request-aligned record") as exc_info:
        _ = study.run(
            max_evaluations=2,
            batch_size=2,
            execution_model=STALE_ASYNC_EXECUTION_MODEL,
        )

    assert type(exc_info.value) is RunExecutionFailed
