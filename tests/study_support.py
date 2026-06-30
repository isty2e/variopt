"""Shared test doubles and helper types for Study execution tests."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import final

from typing_extensions import override

from variopt import (
    CandidateRefinement,
    EvaluationOutcome,
    EvaluationProtocol,
    EvaluationRecord,
    EvaluationRequest,
    Evaluator,
    Kernel,
    Objective,
    Observation,
    ObservationEvaluationProtocol,
    OptimizationDirection,
    Problem,
    Proposal,
    RunMethod,
)
from variopt.evaluators import (
    AsyncEvaluator,
    BatchExecutionFailed,
    CompletionGroup,
    EvaluationBatchHandle,
    EvaluationBatchResumeHandle,
    EvaluationBatchSession,
    ResumableAsyncEvaluator,
    ResumableBatchSession,
)
from variopt.execution import (
    EXACT_ASYNC_EXECUTION_MODEL,
    SEQUENTIAL_EXECUTION_MODEL,
    STALE_ASYNC_EXECUTION_MODEL,
    SYNC_BATCH_EXECUTION_MODEL,
    ExecutionModel,
    ExecutionResources,
)
from variopt.kernel import (
    KernelDiagnostics,
    KernelStatus,
    ProposalBatchQuery,
    ProposalLocalSearchContext,
)


class SquareObjective(Objective[int]):
    """Toy objective used to test study orchestration."""

    @override
    def evaluate(self, candidate: int) -> float:
        return float(candidate * candidate)


@final
class CountingObjective(Objective[int]):
    """Objective that records how often it has been evaluated."""

    def __init__(self) -> None:
        self.evaluation_count = 0

    @override
    def evaluate(self, candidate: int) -> float:
        self.evaluation_count += 1
        return float(candidate * candidate)


class ShiftedObservationProtocol(ObservationEvaluationProtocol[int]):
    """Non-Objective scalar protocol used to test protocol-first execution."""

    @override
    def evaluate_request(
        self,
        request: EvaluationRequest[int],
        *,
        direction: OptimizationDirection,
    ) -> Observation[int]:
        candidate = max(0, request.candidate - 1)
        return Observation.from_objective_value(
            request=request,
            candidate=candidate,
            value=float(candidate * candidate),
            direction=direction,
        )


@dataclass(frozen=True, slots=True)
class LabelRecord(EvaluationRecord[int]):
    """Simple non-scalar evaluation record for study regression tests."""

    label: str


class LabelProtocol(EvaluationProtocol[int, LabelRecord]):
    """Non-scalar protocol used to verify record-first study execution."""

    @override
    def evaluate_request(
        self,
        request: EvaluationRequest[int],
    ) -> LabelRecord:
        return LabelRecord(
            request=request,
            candidate=request.candidate,
            label=f"parity:{request.candidate % 2}",
        )


def _make_async_evaluation_outcome(
    problem: Problem[int, int],
    request: EvaluationRequest[int],
    *,
    attach_refinement: bool,
) -> EvaluationOutcome[int, Observation[int]]:
    record = problem.evaluation_protocol.evaluate_request(request)
    refinement = None
    if attach_refinement:
        refinement = CandidateRefinement(
            source_candidate=request.candidate,
            refined_candidate=record.candidate,
            changed_leaf_paths=((),),
        )
    return EvaluationOutcome(
        record=record,
        evaluation_count=1,
        refinement=refinement,
    )


class DecrementKernel(
    Kernel[
        ProposalBatchQuery[int, int, Observation[int]],
        tuple[EvaluationOutcome[int, Observation[int]], ...],
    ],
):
    """Kernel that deterministically moves candidates toward zero."""

    @override
    def run(
        self,
        query: ProposalBatchQuery[int, int, Observation[int]],
        runner: Callable[
            [ProposalBatchQuery[int, int, Observation[int]]],
            tuple[EvaluationOutcome[int, Observation[int]], ...],
        ],
    ) -> tuple[EvaluationOutcome[int, Observation[int]], ...]:
        outcomes: list[EvaluationOutcome[int, Observation[int]]] = []
        for proposal in query.proposals:
            refined_candidate = max(0, proposal.candidate - 1)
            local_outcomes = runner(
                ProposalBatchQuery(
                    problem=query.problem,
                    proposals=(Proposal(candidate=refined_candidate),),
                    execution_resources=query.execution_resources,
                )
            )
            local_outcome = local_outcomes[0]
            outcomes.append(
                EvaluationOutcome(
                    observation=Observation.from_objective_value(
                        proposal=proposal,
                        candidate=refined_candidate,
                        value=local_outcome.observation.value,
                        direction=query.problem.direction,
                    ),
                    evaluation_count=local_outcome.evaluation_count,
                )
            )
        return tuple(outcomes)


class RefinementKernel(
    Kernel[
        ProposalBatchQuery[int, int, Observation[int]],
        tuple[EvaluationOutcome[int, Observation[int]], ...],
    ],
):
    """Kernel that emits explicit refinement metadata when it changes a candidate."""

    @override
    def run(
        self,
        query: ProposalBatchQuery[int, int, Observation[int]],
        runner: Callable[
            [ProposalBatchQuery[int, int, Observation[int]]],
            tuple[EvaluationOutcome[int, Observation[int]], ...],
        ],
    ) -> tuple[EvaluationOutcome[int, Observation[int]], ...]:
        outcomes: list[EvaluationOutcome[int, Observation[int]]] = []
        for proposal in query.proposals:
            refined_candidate = max(0, proposal.candidate - 1)
            local_outcomes = runner(
                ProposalBatchQuery(
                    problem=query.problem,
                    proposals=(Proposal(candidate=refined_candidate),),
                    execution_resources=query.execution_resources,
                )
            )
            local_outcome = local_outcomes[0]
            refinement = None
            if refined_candidate != proposal.candidate:
                refinement = CandidateRefinement(
                    source_candidate=proposal.candidate,
                    refined_candidate=refined_candidate,
                    changed_leaf_paths=((),),
                )
            outcomes.append(
                EvaluationOutcome(
                    observation=Observation.from_objective_value(
                        proposal=proposal,
                        candidate=refined_candidate,
                        value=local_outcome.observation.value,
                        direction=query.problem.direction,
                    ),
                    evaluation_count=local_outcome.evaluation_count,
                    refinement=refinement,
                )
            )
        return tuple(outcomes)


class ScoringKernel(
    Kernel[
        ProposalBatchQuery[int, int, Observation[int]],
        tuple[EvaluationOutcome[int, Observation[int]], ...],
    ],
):
    """Kernel that returns precomputed candidate scores and cost."""

    @override
    def run(
        self,
        query: ProposalBatchQuery[int, int, Observation[int]],
        runner: Callable[
            [ProposalBatchQuery[int, int, Observation[int]]],
            tuple[EvaluationOutcome[int, Observation[int]], ...],
        ],
    ) -> tuple[EvaluationOutcome[int, Observation[int]], ...]:
        _ = runner
        outcomes: list[EvaluationOutcome[int, Observation[int]]] = []
        for proposal in query.proposals:
            refined_candidate = max(0, proposal.candidate - 2)
            refinement = None
            if refined_candidate != proposal.candidate:
                refinement = CandidateRefinement(
                    source_candidate=proposal.candidate,
                    refined_candidate=refined_candidate,
                    changed_leaf_paths=((),),
                )
            outcomes.append(
                EvaluationOutcome(
                    observation=Observation.from_objective_value(
                        proposal=proposal,
                        candidate=refined_candidate,
                        value=float(refined_candidate * refined_candidate),
                        direction=query.problem.direction,
                    ),
                    evaluation_count=7,
                    kernel_diagnostics=KernelDiagnostics(
                        backend="test",
                        method="scoring",
                        status=KernelStatus.CONVERGED,
                        message="ok",
                    ),
                    refinement=refinement,
                )
            )
        return tuple(outcomes)


class RecordingExecutionResourcesKernel(
    Kernel[
        ProposalBatchQuery[int, int, Observation[int]],
        tuple[EvaluationOutcome[int, Observation[int]], ...],
    ],
):
    """Kernel that records received execution resources."""

    def __init__(self) -> None:
        self.last_execution_resources: ExecutionResources | None = None

    @override
    def run(
        self,
        query: ProposalBatchQuery[int, int, Observation[int]],
        runner: Callable[
            [ProposalBatchQuery[int, int, Observation[int]]],
            tuple[EvaluationOutcome[int, Observation[int]], ...],
        ],
    ) -> tuple[EvaluationOutcome[int, Observation[int]], ...]:
        self.last_execution_resources = query.execution_resources
        return runner(query)


@dataclass(frozen=True, slots=True)
class BatchQueueOptimizerState:
    """Explicit state for the batch-queue test optimizer."""

    remaining_batches: tuple[tuple[Proposal[int], ...], ...]
    tell_history: tuple[tuple[Observation[int], ...], ...] = ()
    ask_history: tuple[int, ...] = ()


class BatchQueueOptimizer(RunMethod[BatchQueueOptimizerState, Proposal[int], Observation[int]]):
    """Toy optimizer that emits precomputed proposal batches."""

    _initial_batches: tuple[tuple[Proposal[int], ...], ...]

    def __init__(self, proposal_batches: list[tuple[Proposal[int], ...]]) -> None:
        self._initial_batches = tuple(tuple(batch) for batch in proposal_batches)

    @override
    def create_initial_state(self) -> BatchQueueOptimizerState:
        return BatchQueueOptimizerState(remaining_batches=self._initial_batches)

    @override
    def is_exhausted(self, state: BatchQueueOptimizerState) -> bool:
        return len(state.remaining_batches) == 0

    @override
    def ask(
        self,
        state: BatchQueueOptimizerState,
        batch_size: int = 1,
    ) -> tuple[tuple[Proposal[int], ...], BatchQueueOptimizerState]:
        next_state = BatchQueueOptimizerState(
            remaining_batches=state.remaining_batches,
            tell_history=state.tell_history,
            ask_history=state.ask_history + (batch_size,),
        )
        if not state.remaining_batches:
            return (), next_state

        return (
            state.remaining_batches[0],
            BatchQueueOptimizerState(
                remaining_batches=state.remaining_batches[1:],
                tell_history=state.tell_history,
                ask_history=next_state.ask_history,
            ),
        )

    @override
    def tell(
        self,
        state: BatchQueueOptimizerState,
        observations: Sequence[Observation[int]],
    ) -> BatchQueueOptimizerState:
        return BatchQueueOptimizerState(
            remaining_batches=state.remaining_batches,
            tell_history=state.tell_history + (tuple(observations),),
            ask_history=state.ask_history,
        )


class ExactAsyncCapableBatchQueueOptimizer(BatchQueueOptimizer):
    """Batch-queue optimizer that advertises exact-async compatibility."""

    @override
    def supported_execution_models(self) -> frozenset[ExecutionModel]:
        return frozenset(
            {
                SEQUENTIAL_EXECUTION_MODEL,
                SYNC_BATCH_EXECUTION_MODEL,
                EXACT_ASYNC_EXECUTION_MODEL,
            },
        )


@dataclass(frozen=True, slots=True)
class RollingStaleAsyncOptimizerState:
    """State for a run method that refills its frontier from stale async tells."""

    queued_proposals: tuple[Proposal[int], ...]
    ask_history: tuple[int, ...] = ()
    tell_history: tuple[tuple[Observation[int], ...], ...] = ()


@final
class RollingStaleAsyncOptimizer(
    RunMethod[RollingStaleAsyncOptimizerState, Proposal[int], Observation[int]]
):
    """Run method that exposes stale async rolling refill behavior."""

    _initial_proposals: tuple[Proposal[int], ...]

    def __init__(self, proposals: Sequence[Proposal[int]]) -> None:
        self._initial_proposals = tuple(proposals)

    @override
    def create_initial_state(self) -> RollingStaleAsyncOptimizerState:
        return RollingStaleAsyncOptimizerState(
            queued_proposals=self._initial_proposals,
        )

    @override
    def is_exhausted(self, state: RollingStaleAsyncOptimizerState) -> bool:
        return len(state.queued_proposals) == 0

    @override
    def ask(
        self,
        state: RollingStaleAsyncOptimizerState,
        batch_size: int = 1,
    ) -> tuple[tuple[Proposal[int], ...], RollingStaleAsyncOptimizerState]:
        proposal_batch = state.queued_proposals[:batch_size]
        return (
            proposal_batch,
            RollingStaleAsyncOptimizerState(
                queued_proposals=state.queued_proposals[len(proposal_batch) :],
                ask_history=state.ask_history + (batch_size,),
                tell_history=state.tell_history,
            ),
        )

    @override
    def tell(
        self,
        state: RollingStaleAsyncOptimizerState,
        observations: Sequence[Observation[int]],
    ) -> RollingStaleAsyncOptimizerState:
        spawned_proposals = tuple(
            Proposal(
                candidate=observation.candidate + 10,
                proposal_id=f"spawn-{observation.proposal.proposal_id}",
            )
            for observation in observations
            if observation.candidate < 10
        )
        return RollingStaleAsyncOptimizerState(
            queued_proposals=state.queued_proposals + spawned_proposals,
            ask_history=state.ask_history,
            tell_history=state.tell_history + (tuple(observations),),
        )

    @override
    def supported_execution_models(self) -> frozenset[ExecutionModel]:
        return frozenset(
            {
                SEQUENTIAL_EXECUTION_MODEL,
                SYNC_BATCH_EXECUTION_MODEL,
                EXACT_ASYNC_EXECUTION_MODEL,
                STALE_ASYNC_EXECUTION_MODEL,
            },
        )


@dataclass(frozen=True, slots=True)
class LabelBatchQueueOptimizerState:
    """Explicit state for non-scalar batch-queue optimizer tests."""

    remaining_batches: tuple[tuple[Proposal[int], ...], ...]
    tell_history: tuple[tuple[LabelRecord, ...], ...] = ()


@final
class LabelBatchQueueOptimizer(
    RunMethod[LabelBatchQueueOptimizerState, Proposal[int], LabelRecord]
):
    """Toy optimizer that records non-scalar evaluation records."""

    _initial_batches: tuple[tuple[Proposal[int], ...], ...]

    def __init__(self, proposal_batches: list[tuple[Proposal[int], ...]]) -> None:
        self._initial_batches = tuple(tuple(batch) for batch in proposal_batches)

    @override
    def create_initial_state(self) -> LabelBatchQueueOptimizerState:
        return LabelBatchQueueOptimizerState(remaining_batches=self._initial_batches)

    @override
    def is_exhausted(self, state: LabelBatchQueueOptimizerState) -> bool:
        return len(state.remaining_batches) == 0

    @override
    def ask(
        self,
        state: LabelBatchQueueOptimizerState,
        batch_size: int = 1,
    ) -> tuple[tuple[Proposal[int], ...], LabelBatchQueueOptimizerState]:
        _ = batch_size
        if not state.remaining_batches:
            return (), state

        return (
            state.remaining_batches[0],
            LabelBatchQueueOptimizerState(
                remaining_batches=state.remaining_batches[1:],
                tell_history=state.tell_history,
            ),
        )

    @override
    def tell(
        self,
        state: LabelBatchQueueOptimizerState,
        observations: Sequence[LabelRecord],
    ) -> LabelBatchQueueOptimizerState:
        return LabelBatchQueueOptimizerState(
            remaining_batches=state.remaining_batches,
            tell_history=state.tell_history + (tuple(observations),),
        )


class ContextAwareBatchQueueOptimizer(BatchQueueOptimizer):
    """Batch-queue optimizer that emits per-proposal local-search context."""

    @override
    def proposal_kernel_hints(
        self,
        state: BatchQueueOptimizerState,
        proposals: Sequence[Proposal[int]],
    ) -> tuple[ProposalLocalSearchContext | None, ...] | None:
        _ = state
        return tuple(
            ProposalLocalSearchContext(local_budget=index + 1)
            for index, _proposal in enumerate(proposals)
        )


class OutOfOrderAsyncEvaluator(
    AsyncEvaluator[
        Problem[int, int],
        EvaluationRequest[int],
        EvaluationOutcome[int, Observation[int]],
    ]
):
    """Async evaluator that completes proposals in reverse order."""

    _next_batch_id: int
    _pending_groups: dict[
        str,
        tuple[CompletionGroup[EvaluationOutcome[int, Observation[int]]], ...],
    ]
    _attach_refinement: bool

    def __init__(self, *, attach_refinement: bool = False) -> None:
        self._next_batch_id = 0
        self._pending_groups = {}
        self._attach_refinement = attach_refinement

    @override
    def submit_batch(
        self,
        problem: Problem[int, int],
        requests: Sequence[EvaluationRequest[int]],
    ) -> EvaluationBatchHandle:
        handle = EvaluationBatchHandle(
            batch_id=f"batch-{self._next_batch_id}",
            request_count=len(requests),
        )
        self._next_batch_id += 1
        self._pending_groups[handle.batch_id] = tuple(
            CompletionGroup(
                start_index=index,
                outcomes=(
                    _make_async_evaluation_outcome(
                        problem,
                        request,
                        attach_refinement=self._attach_refinement,
                    ),
                ),
            )
            for index, request in reversed(tuple(enumerate(requests)))
        )
        return handle

    @override
    def poll(
        self,
        handle: EvaluationBatchHandle,
    ) -> Sequence[CompletionGroup[EvaluationOutcome[int, Observation[int]]]]:
        pending_groups = self._pending_groups.get(handle.batch_id)
        if pending_groups is None:
            msg = f"unknown async batch handle: {handle.batch_id}"
            raise ValueError(msg)

        if len(pending_groups) == 0:
            raise BatchExecutionFailed(
                handle=handle,
                kind="infrastructure",
                cause=RuntimeError("async batch exhausted unexpectedly"),
            )

        completion_group = pending_groups[0]
        remaining_groups = pending_groups[1:]
        if len(remaining_groups) == 0:
            _ = self._pending_groups.pop(handle.batch_id, None)
        else:
            self._pending_groups[handle.batch_id] = remaining_groups
        return (completion_group,)

    @override
    def cancel(self, handle: EvaluationBatchHandle) -> None:
        _ = self._pending_groups.pop(handle.batch_id, None)


class SessionRecordingAsyncEvaluator(OutOfOrderAsyncEvaluator):
    """Async evaluator that records exact-async session openings."""

    def __init__(self, *, attach_refinement: bool = False) -> None:
        super().__init__(attach_refinement=attach_refinement)
        self.opened_batch_sizes: tuple[int, ...] = ()

    @override
    def open_session(
        self,
        problem: Problem[int, int],
        requests: Sequence[EvaluationRequest[int]],
    ) -> EvaluationBatchSession[EvaluationOutcome[int, Observation[int]]]:
        self.opened_batch_sizes += (len(requests),)
        return super().open_session(problem, requests)


@dataclass(slots=True)
class _ResumableOutOfOrderBatchSession(
    ResumableBatchSession[EvaluationOutcome[int, Observation[int]]]
):
    """Resumable in-memory exact-async session for study orchestration tests."""

    evaluator: "ResumableOutOfOrderAsyncEvaluator"
    _handle: EvaluationBatchHandle
    _completed_count: int = 0

    @property
    @override
    def handle(self) -> EvaluationBatchHandle:
        return self._handle

    @override
    def poll(
        self,
    ) -> Sequence[CompletionGroup[EvaluationOutcome[int, Observation[int]]]]:
        completion_groups = tuple(self.evaluator.poll(self.handle))
        self._completed_count += sum(
            len(completion_group.outcomes)
            for completion_group in completion_groups
        )
        return completion_groups

    @override
    def cancel(self) -> None:
        self.evaluator.cancel(self.handle)

    @override
    def suspend(self) -> EvaluationBatchResumeHandle:
        return self.evaluator.suspend_batch(
            self.handle,
            completed_count=self._completed_count,
        )


@final
class ResumableOutOfOrderAsyncEvaluator(
    ResumableAsyncEvaluator[
        Problem[int, int],
        EvaluationRequest[int],
        EvaluationOutcome[int, Observation[int]],
    ]
):
    """Resumable async evaluator that completes proposals in reverse order."""

    _next_batch_id: int
    _pending_groups: dict[
        str,
        tuple[CompletionGroup[EvaluationOutcome[int, Observation[int]]], ...],
    ]
    _suspended_groups: dict[
        str,
        tuple[CompletionGroup[EvaluationOutcome[int, Observation[int]]], ...],
    ]

    def __init__(self, *, attach_refinement: bool = False) -> None:
        self._next_batch_id = 0
        self._pending_groups = {}
        self._suspended_groups = {}
        self._attach_refinement = attach_refinement

    @override
    def open_session(
        self,
        problem: Problem[int, int],
        requests: Sequence[EvaluationRequest[int]],
    ) -> EvaluationBatchSession[EvaluationOutcome[int, Observation[int]]]:
        return _ResumableOutOfOrderBatchSession(
            evaluator=self,
            _handle=self.submit_batch(problem, requests),
        )

    @override
    def submit_batch(
        self,
        problem: Problem[int, int],
        requests: Sequence[EvaluationRequest[int]],
    ) -> EvaluationBatchHandle:
        handle = EvaluationBatchHandle(
            batch_id=f"resumable-batch-{self._next_batch_id}",
            request_count=len(requests),
        )
        self._next_batch_id += 1
        self._pending_groups[handle.batch_id] = tuple(
            CompletionGroup(
                start_index=index,
                outcomes=(
                    _make_async_evaluation_outcome(
                        problem,
                        request,
                        attach_refinement=self._attach_refinement,
                    ),
                ),
            )
            for index, request in reversed(tuple(enumerate(requests)))
        )
        return handle

    @override
    def poll(
        self,
        handle: EvaluationBatchHandle,
    ) -> Sequence[CompletionGroup[EvaluationOutcome[int, Observation[int]]]]:
        pending_groups = self._pending_groups.get(handle.batch_id)
        if pending_groups is None:
            msg = f"unknown async batch handle: {handle.batch_id}"
            raise ValueError(msg)

        if len(pending_groups) == 0:
            raise BatchExecutionFailed(
                handle=handle,
                kind="infrastructure",
                cause=RuntimeError("async batch exhausted unexpectedly"),
            )

        completion_group = pending_groups[0]
        remaining_groups = pending_groups[1:]
        if len(remaining_groups) == 0:
            _ = self._pending_groups.pop(handle.batch_id, None)
        else:
            self._pending_groups[handle.batch_id] = remaining_groups
        return (completion_group,)

    @override
    def cancel(self, handle: EvaluationBatchHandle) -> None:
        _ = self._pending_groups.pop(handle.batch_id, None)
        _ = self._suspended_groups.pop(handle.batch_id, None)

    def suspend_batch(
        self,
        handle: EvaluationBatchHandle,
        *,
        completed_count: int,
    ) -> EvaluationBatchResumeHandle:
        pending_groups = self._pending_groups.pop(handle.batch_id, None)
        if pending_groups is None:
            msg = f"unknown async batch handle: {handle.batch_id}"
            raise ValueError(msg)

        self._suspended_groups[handle.batch_id] = pending_groups
        return EvaluationBatchResumeHandle(
            batch_id=handle.batch_id,
            request_count=handle.request_count,
            completed_count=completed_count,
        )

    @override
    def resume_session(
        self,
        handle: EvaluationBatchResumeHandle,
    ) -> EvaluationBatchSession[EvaluationOutcome[int, Observation[int]]]:
        pending_groups = self._suspended_groups.pop(handle.batch_id, None)
        if pending_groups is None:
            msg = f"unknown suspended batch handle: {handle.batch_id}"
            raise ValueError(msg)

        self._pending_groups[handle.batch_id] = pending_groups
        return _ResumableOutOfOrderBatchSession(
            evaluator=self,
            _handle=EvaluationBatchHandle(
                batch_id=handle.batch_id,
                request_count=handle.request_count,
            ),
            _completed_count=handle.completed_count,
        )


class MisorderedEvaluator(
    Evaluator[
        Problem[int, int],
        EvaluationRequest[int],
        EvaluationOutcome[int, Observation[int]],
    ]
):
    """Broken evaluator used to verify Study-level order validation."""

    @override
    def evaluate(
        self,
        problem: Problem[int, int],
        requests: Sequence[EvaluationRequest[int]],
    ) -> Sequence[EvaluationOutcome[int, Observation[int]]]:
        return tuple(
            EvaluationOutcome(
                record=problem.evaluation_protocol.evaluate_request(request),
                evaluation_count=1,
            )
            for request in reversed(requests)
        )


class RecordingKernel(
    Kernel[
        ProposalBatchQuery[int, int, Observation[int]],
        tuple[EvaluationOutcome[int, Observation[int]], ...],
    ],
):
    """Pass-through kernel that records received queries."""

    def __init__(self) -> None:
        self.queries: list[ProposalBatchQuery[int, int, Observation[int]]] = []

    @override
    def run(
        self,
        query: ProposalBatchQuery[int, int, Observation[int]],
        runner: Callable[
            [ProposalBatchQuery[int, int, Observation[int]]],
            tuple[EvaluationOutcome[int, Observation[int]], ...],
        ],
    ) -> tuple[EvaluationOutcome[int, Observation[int]], ...]:
        self.queries.append(query)
        return runner(query)
