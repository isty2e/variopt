"""Alignment remains enforced at evaluator and custom-kernel boundaries."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from typing import Literal, TypeAlias

import pytest
from typing_extensions import override

from tests.study_support import (
    ExactAsyncCapableBatchQueueOptimizer,
    OutOfOrderAsyncEvaluator,
    SquareObjective,
)
from variopt import EvaluationRequest, IntegerSpace, Problem, Proposal, Study
from variopt.artifacts import (
    EvaluationAttemptBatch,
    EvaluationFailure,
    ObservationPayload,
    ProposalEvaluationSpec,
)
from variopt.evaluators import SequentialEvaluator
from variopt.execution import EXACT_ASYNC_EXECUTION_MODEL, SYNC_BATCH_EXECUTION_MODEL
from variopt.kernel import DirectKernel, ProposalBatchQuery

ScalarQuery: TypeAlias = ProposalBatchQuery[int, int, ObservationPayload]
ScalarAttempts: TypeAlias = EvaluationAttemptBatch[int, ObservationPayload]
Corruption: TypeAlias = Literal["candidate", "proposal_id", "spec", "order", "drop"]


@dataclass(frozen=True, slots=True)
class CountingEqualitySpace(IntegerSpace):
    comparisons: list[tuple[int, int]] = field(default_factory=list, compare=False)

    @override
    def candidates_equal(self, left_candidate: int, right_candidate: int) -> bool:
        self.comparisons.append((left_candidate, right_candidate))
        return super(CountingEqualitySpace, self).candidates_equal(
            left_candidate, right_candidate
        )


class PassthroughKernel(DirectKernel[ScalarQuery, ScalarAttempts]):
    pass


class RejectingEqualitySpace(IntegerSpace):
    @override
    def candidates_equal(self, left_candidate: int, right_candidate: int) -> bool:
        raise ValueError("space equality rejected candidates")


class ReorderingKernel(DirectKernel[ScalarQuery, ScalarAttempts]):
    @override
    def run(
        self, query: ScalarQuery, runner: Callable[[ScalarQuery], ScalarAttempts]
    ) -> ScalarAttempts:
        attempts = runner(query)
        return EvaluationAttemptBatch(attempts=reversed(attempts.attempts))


class CorruptingEvaluator(SequentialEvaluator[int, int]):
    def __init__(self, corruption: Corruption, *, failed: bool) -> None:
        super().__init__()
        self.corruption = corruption
        self.failed = failed

    @override
    def evaluate_attempts(
        self,
        problem: Problem[int, int, ObservationPayload],
        requests: Sequence[EvaluationRequest[int]],
    ) -> ScalarAttempts:
        changed_requests = list(requests)
        first = requests[0]
        if self.corruption == "candidate":
            changed_requests[0] = replace(
                first, proposal=replace(first.proposal, candidate=9)
            )
        elif self.corruption == "proposal_id":
            changed_requests[0] = replace(
                first, proposal=replace(first.proposal, proposal_id="wrong")
            )
        elif self.corruption == "spec":
            changed_requests[0] = replace(
                first, proposal_evaluation_spec=ProposalEvaluationSpec()
            )
        elif self.corruption == "order":
            changed_requests.reverse()
        else:
            changed_requests.pop()

        if self.failed:
            return EvaluationAttemptBatch(
                attempts=(
                    EvaluationFailure[int].from_exception(
                        request=request, exception=ValueError("objective failure")
                    )
                    for request in changed_requests
                )
            )
        return super().evaluate_attempts(problem, changed_requests)


def queue_optimizer() -> ExactAsyncCapableBatchQueueOptimizer:
    return ExactAsyncCapableBatchQueueOptimizer(
        [
            (
                Proposal(candidate=2, proposal_id="p-0"),
                Proposal(candidate=4, proposal_id="p-1"),
            )
        ]
    )


@pytest.mark.parametrize("custom_kernel", [False, True])
@pytest.mark.parametrize("exact_async", [False, True])
def test_direct_kernel_does_not_repeat_runner_alignment(
    custom_kernel: bool, exact_async: bool
) -> None:
    space = CountingEqualitySpace(low=0, high=10)
    optimizer = queue_optimizer()
    kernel = (
        PassthroughKernel()
        if custom_kernel
        else DirectKernel[ScalarQuery, ScalarAttempts]()
    )
    study = Study(
        problem=Problem(space=space, objective=SquareObjective()),
        run_method=optimizer,
        evaluator=(
            OutOfOrderAsyncEvaluator()
            if exact_async
            else SequentialEvaluator[int, int]()
        ),
        kernel=kernel,
    )

    records, _ = study.step(
        optimizer.create_initial_state(),
        batch_size=2,
        execution_model=(
            EXACT_ASYNC_EXECUTION_MODEL if exact_async else SYNC_BATCH_EXECUTION_MODEL
        ),
    )

    assert [record.value for record in records] == [4.0, 16.0]
    assert space.comparisons == [(2, 2), (4, 4)] * (2 if custom_kernel else 1)


@pytest.mark.parametrize(
    "corruption", ["candidate", "proposal_id", "spec", "order", "drop"]
)
@pytest.mark.parametrize("failed", [False, True])
def test_direct_kernel_rejects_misaligned_evaluator_attempts(
    corruption: Corruption, failed: bool
) -> None:
    optimizer = queue_optimizer()
    study = Study(
        problem=Problem(space=IntegerSpace(0, 10), objective=SquareObjective()),
        run_method=optimizer,
        evaluator=CorruptingEvaluator(corruption, failed=failed),
    )

    with pytest.raises(ValueError, match="align|exactly one slot"):
        study.step(optimizer.create_initial_state(), batch_size=2)


def test_direct_kernel_subclass_revalidates_after_runner() -> None:
    optimizer = queue_optimizer()
    study = Study(
        problem=Problem(space=IntegerSpace(0, 10), objective=SquareObjective()),
        run_method=optimizer,
        evaluator=SequentialEvaluator[int, int](),
        kernel=ReorderingKernel(),
    )

    with pytest.raises(ValueError, match="align with input request order"):
        study.step(optimizer.create_initial_state(), batch_size=2)


def test_direct_kernel_does_not_replace_space_equality_with_identity() -> None:
    optimizer = queue_optimizer()
    study = Study(
        problem=Problem(
            space=RejectingEqualitySpace(0, 10), objective=SquareObjective()
        ),
        run_method=optimizer,
        evaluator=SequentialEvaluator[int, int](),
    )

    with pytest.raises(ValueError, match="space equality rejected candidates"):
        study.step(optimizer.create_initial_state(), batch_size=2)
