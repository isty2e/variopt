"""Shared test doubles and helper types for Study execution tests."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TypeGuard, TypeVar, final

import numpy as np
from typing_extensions import override

from variopt import (
    CandidateRefinement,
    EvaluationAttemptBatch,
    EvaluationOutcome,
    EvaluationProtocol,
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
    SearchSpace,
)
from variopt.artifacts import (
    EvaluationFailure,
    EvaluationSuccess,
    KernelDiagnostics,
    KernelStatus,
    ObservationPayload,
    ProposalEvaluationSpec,
    materialize_success_records,
)
from variopt.artifacts.records import RequestAlignedEvaluationRecord
from variopt.evaluation_pipeline import (
    evaluate_request_attempt,
    evaluate_request_outcome,
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
    NestedParallelismPolicy,
)
from variopt.kernel import (
    ProposalBatchQuery,
    ProposalLocalSearchContext,
)
from variopt.spaces import LeafPath

OutcomeCandidateT = TypeVar("OutcomeCandidateT")


class SpaceOwnedEqualityCandidate:
    """Candidate whose raw equality is deliberately not usable."""

    def __init__(self, stable_id: int) -> None:
        self.stable_id: int = stable_id

    @override
    def __eq__(self, other: object) -> bool:
        _ = other
        raise ValueError("raw candidate equality is not the space contract")


class SpaceOwnedEqualitySpace(
    SearchSpace[int | SpaceOwnedEqualityCandidate, SpaceOwnedEqualityCandidate],
):
    """Test search space that owns stable-id candidate equality."""

    @override
    def normalize(
        self,
        raw_candidate: int | SpaceOwnedEqualityCandidate,
    ) -> SpaceOwnedEqualityCandidate:
        if isinstance(raw_candidate, SpaceOwnedEqualityCandidate):
            self.validate(raw_candidate)
            return raw_candidate
        candidate = SpaceOwnedEqualityCandidate(raw_candidate)
        self.validate(candidate)
        return candidate

    @override
    def validate(self, candidate: SpaceOwnedEqualityCandidate) -> None:
        if candidate.stable_id < 0:
            msg = "candidate stable_id must be non-negative"
            raise ValueError(msg)

    @override
    def sample(
        self,
        random_state: np.random.RandomState,
    ) -> SpaceOwnedEqualityCandidate:
        _ = random_state
        return SpaceOwnedEqualityCandidate(0)

    @override
    def candidates_equal(
        self,
        left_candidate: SpaceOwnedEqualityCandidate,
        right_candidate: SpaceOwnedEqualityCandidate,
    ) -> bool:
        self.validate(left_candidate)
        self.validate(right_candidate)
        return left_candidate.stable_id == right_candidate.stable_id


class SpaceOwnedEqualityObjective(Objective[SpaceOwnedEqualityCandidate]):
    """Objective over the stable id carried by ambiguous-equality candidates."""

    @override
    def evaluate(self, candidate: SpaceOwnedEqualityCandidate) -> float:
        return float(candidate.stable_id)


@dataclass(frozen=True, slots=True)
class SpaceOwnedEqualityOptimizerState:
    """State for one-shot ambiguous-equality candidate tests."""

    has_proposal: bool = True


class SpaceOwnedEqualityOptimizer(
    RunMethod[
        SpaceOwnedEqualityOptimizerState,
        Proposal[SpaceOwnedEqualityCandidate],
        Observation[SpaceOwnedEqualityCandidate],
    ],
):
    """One-shot optimizer for space-owned equality propagation tests."""

    @override
    def create_initial_state(self) -> SpaceOwnedEqualityOptimizerState:
        return SpaceOwnedEqualityOptimizerState()

    @override
    def is_exhausted(self, state: SpaceOwnedEqualityOptimizerState) -> bool:
        return not state.has_proposal

    @override
    def ask(
        self,
        state: SpaceOwnedEqualityOptimizerState,
        batch_size: int = 1,
    ) -> tuple[
        tuple[Proposal[SpaceOwnedEqualityCandidate], ...],
        SpaceOwnedEqualityOptimizerState,
    ]:
        _ = batch_size
        if not state.has_proposal:
            return (), state
        return (
            (Proposal(candidate=SpaceOwnedEqualityCandidate(2), proposal_id="p-1"),),
            SpaceOwnedEqualityOptimizerState(has_proposal=False),
        )

    @override
    def tell(
        self,
        state: SpaceOwnedEqualityOptimizerState,
        observations: Sequence[Observation[SpaceOwnedEqualityCandidate]],
    ) -> SpaceOwnedEqualityOptimizerState:
        _ = observations
        return state

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


class SpaceOwnedEqualityRefinementKernel(
    Kernel[
        ProposalBatchQuery[
            int | SpaceOwnedEqualityCandidate,
            SpaceOwnedEqualityCandidate,
            ObservationPayload,
        ],
        EvaluationAttemptBatch[SpaceOwnedEqualityCandidate, ObservationPayload],
    ],
):
    """Kernel that emits distinct but space-equal refined candidate instances."""

    @override
    def run(
        self,
        query: ProposalBatchQuery[
            int | SpaceOwnedEqualityCandidate,
            SpaceOwnedEqualityCandidate,
            ObservationPayload,
        ],
        runner: Callable[
            [
                ProposalBatchQuery[
                    int | SpaceOwnedEqualityCandidate,
                    SpaceOwnedEqualityCandidate,
                    ObservationPayload,
                ],
            ],
            EvaluationAttemptBatch[SpaceOwnedEqualityCandidate, ObservationPayload],
        ],
    ) -> EvaluationAttemptBatch[SpaceOwnedEqualityCandidate, ObservationPayload]:
        _ = runner
        proposal = query.proposals[0]
        record_candidate = SpaceOwnedEqualityCandidate(1)
        refinement_candidate = SpaceOwnedEqualityCandidate(1)
        request = EvaluationRequest(
            proposal=Proposal(
                candidate=record_candidate,
                proposal_id=proposal.proposal_id,
            ),
        )
        payload = ObservationPayload.from_objective_value(
            value=1.0,
            direction=query.problem.direction,
        )
        success: EvaluationSuccess[
            SpaceOwnedEqualityCandidate,
            ObservationPayload,
        ] = EvaluationSuccess(
            request=request,
            payload=payload,
            refinement=CandidateRefinement(
                source_candidate=proposal.candidate,
                refined_candidate=refinement_candidate,
                changed_leaf_paths=((),),
            ),
            candidate_equal=query.problem.space.candidates_equal,
        )
        return EvaluationAttemptBatch(
            attempts=(success,),
        )


class SpaceOwnedEqualityAsyncEvaluator(
    AsyncEvaluator[
        Problem[
            int | SpaceOwnedEqualityCandidate,
            SpaceOwnedEqualityCandidate,
            ObservationPayload,
        ],
        EvaluationRequest[SpaceOwnedEqualityCandidate],
        EvaluationOutcome[
            SpaceOwnedEqualityCandidate, Observation[SpaceOwnedEqualityCandidate]
        ],
    ],
):
    """Async evaluator that returns space-equal distinct refinement candidates."""

    _next_batch_id: int
    _pending_groups: dict[
        str,
        tuple[
            CompletionGroup[
                EvaluationOutcome[
                    SpaceOwnedEqualityCandidate,
                    Observation[SpaceOwnedEqualityCandidate],
                ]
            ],
            ...,
        ],
    ]
    _pending_attempt_groups: dict[
        str,
        tuple[
            CompletionGroup[
                EvaluationAttemptBatch[
                    SpaceOwnedEqualityCandidate,
                    ObservationPayload,
                ]
            ],
            ...,
        ],
    ]

    def __init__(self) -> None:
        self._next_batch_id = 0
        self._pending_groups = {}
        self._pending_attempt_groups = {}

    @override
    def submit_batch(
        self,
        problem: Problem[
            int | SpaceOwnedEqualityCandidate,
            SpaceOwnedEqualityCandidate,
            ObservationPayload,
        ],
        requests: Sequence[EvaluationRequest[SpaceOwnedEqualityCandidate]],
    ) -> EvaluationBatchHandle:
        handle = EvaluationBatchHandle(
            batch_id=f"space-owned-equality-{self._next_batch_id}",
            request_count=len(requests),
        )
        self._next_batch_id += 1
        self._pending_groups[handle.batch_id] = tuple(
            CompletionGroup(
                start_index=index,
                outcomes=(
                    _make_space_owned_equality_outcome(
                        problem=problem,
                        request=request,
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
    ) -> Sequence[
        CompletionGroup[
            EvaluationOutcome[
                SpaceOwnedEqualityCandidate,
                Observation[SpaceOwnedEqualityCandidate],
            ]
        ]
    ]:
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

    def open_attempt_session(
        self,
        problem: Problem[
            int | SpaceOwnedEqualityCandidate,
            SpaceOwnedEqualityCandidate,
            ObservationPayload,
        ],
        requests: Sequence[EvaluationRequest[SpaceOwnedEqualityCandidate]],
    ) -> EvaluationBatchSession[
        EvaluationAttemptBatch[
            SpaceOwnedEqualityCandidate,
            ObservationPayload,
        ]
    ]:
        handle = EvaluationBatchHandle(
            batch_id=f"space-owned-equality-attempt-{self._next_batch_id}",
            request_count=len(requests),
        )
        self._next_batch_id += 1
        self._pending_attempt_groups[handle.batch_id] = tuple(
            CompletionGroup(
                start_index=index,
                outcomes=(
                    _make_space_owned_equality_attempt(
                        problem=problem,
                        request=request,
                    ),
                ),
            )
            for index, request in reversed(tuple(enumerate(requests)))
        )
        return _SpaceOwnedEqualityAttemptBatchSession(evaluator=self, _handle=handle)

    def poll_attempts(
        self,
        handle: EvaluationBatchHandle,
    ) -> Sequence[
        CompletionGroup[
            EvaluationAttemptBatch[
                SpaceOwnedEqualityCandidate,
                ObservationPayload,
            ]
        ]
    ]:
        pending_groups = self._pending_attempt_groups.get(handle.batch_id)
        if pending_groups is None:
            msg = f"unknown async attempt batch handle: {handle.batch_id}"
            raise ValueError(msg)
        if len(pending_groups) == 0:
            raise BatchExecutionFailed(
                handle=handle,
                kind="infrastructure",
                cause=RuntimeError("async attempt batch exhausted unexpectedly"),
            )

        completion_group = pending_groups[0]
        remaining_groups = pending_groups[1:]
        if len(remaining_groups) == 0:
            _ = self._pending_attempt_groups.pop(handle.batch_id, None)
        else:
            self._pending_attempt_groups[handle.batch_id] = remaining_groups
        return (completion_group,)

    def cancel_attempts(self, handle: EvaluationBatchHandle) -> None:
        _ = self._pending_attempt_groups.pop(handle.batch_id, None)


def _make_space_owned_equality_outcome(
    *,
    problem: Problem[
        int | SpaceOwnedEqualityCandidate,
        SpaceOwnedEqualityCandidate,
        ObservationPayload,
    ],
    request: EvaluationRequest[SpaceOwnedEqualityCandidate],
) -> EvaluationOutcome[
    SpaceOwnedEqualityCandidate, Observation[SpaceOwnedEqualityCandidate]
]:
    record_candidate = SpaceOwnedEqualityCandidate(1)
    refinement_candidate = SpaceOwnedEqualityCandidate(1)
    return EvaluationOutcome(
        observation=Observation.from_objective_value(
            request=request,
            candidate=record_candidate,
            value=1.0,
            direction=problem.direction,
        ),
        refinement=CandidateRefinement(
            source_candidate=request.candidate,
            refined_candidate=refinement_candidate,
            changed_leaf_paths=((),),
        ),
        candidate_equal=problem.space.candidates_equal,
    )


def _make_space_owned_equality_attempt(
    *,
    problem: Problem[
        int | SpaceOwnedEqualityCandidate,
        SpaceOwnedEqualityCandidate,
        ObservationPayload,
    ],
    request: EvaluationRequest[SpaceOwnedEqualityCandidate],
) -> EvaluationAttemptBatch[
    SpaceOwnedEqualityCandidate,
    ObservationPayload,
]:
    outcome = _make_space_owned_equality_outcome(problem=problem, request=request)
    observation = outcome.observation
    success_request = EvaluationRequest(
        proposal=Proposal(
            candidate=observation.candidate,
            proposal_id=request.proposal_id,
        ),
        proposal_evaluation_spec=request.proposal_evaluation_spec,
    )
    return EvaluationAttemptBatch(
        attempts=(
            EvaluationSuccess(
                request=success_request,
                payload=ObservationPayload.from_objective_value(
                    value=observation.value,
                    direction=problem.direction,
                    elapsed_seconds=observation.elapsed_seconds,
                ),
                evaluation_count=outcome.evaluation_count,
                refinement=outcome.refinement,
                candidate_equal=problem.space.candidates_equal,
            ),
        )
    )


@dataclass(slots=True)
class _SpaceOwnedEqualityAttemptBatchSession(
    EvaluationBatchSession[
        EvaluationAttemptBatch[
            SpaceOwnedEqualityCandidate,
            ObservationPayload,
        ]
    ]
):
    """Attempt-aware async session for space-owned equality tests."""

    evaluator: SpaceOwnedEqualityAsyncEvaluator
    _handle: EvaluationBatchHandle

    @property
    @override
    def handle(self) -> EvaluationBatchHandle:
        return self._handle

    @override
    def poll(
        self,
    ) -> Sequence[
        CompletionGroup[
            EvaluationAttemptBatch[
                SpaceOwnedEqualityCandidate,
                ObservationPayload,
            ]
        ]
    ]:
        return self.evaluator.poll_attempts(self.handle)

    @override
    def cancel(self) -> None:
        self.evaluator.cancel_attempts(self.handle)


class SquareObjective(Objective[int]):
    """Toy objective used to test study orchestration."""

    @override
    def evaluate(self, candidate: int) -> float:
        return float(candidate * candidate)


@final
class FailingCandidateObjective(Objective[int]):
    """Objective that raises for a configured set of candidates."""

    _failed_candidates: frozenset[int]

    def __init__(self, failed_candidates: Sequence[int]) -> None:
        self._failed_candidates = frozenset(failed_candidates)

    @override
    def evaluate(self, candidate: int) -> float:
        if candidate in self._failed_candidates:
            raise ValueError(f"bad candidate: {candidate}")

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
    ) -> ObservationPayload:
        candidate = max(0, request.candidate - 1)
        return ObservationPayload.from_objective_value(
            value=float(candidate * candidate),
            direction=direction,
        )


@dataclass(frozen=True, slots=True)
class LabelRecord:
    """Simple request-aligned compatibility payload for study regression tests."""

    request: EvaluationRequest[int]
    candidate: int

    label: str

    @property
    def proposal(self) -> Proposal[int]:
        """Return the proposal compatibility view."""
        return self.request.proposal

    @property
    def proposal_evaluation_spec(self) -> ProposalEvaluationSpec | None:
        """Return request-local metadata attached to the source request."""
        return self.request.proposal_evaluation_spec


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
    outcome = make_observation_outcome(problem=problem, request=request)
    record = outcome.record
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


def _make_async_evaluation_attempt(
    problem: Problem[int, int],
    request: EvaluationRequest[int],
    *,
    attach_refinement: bool,
) -> EvaluationAttemptBatch[int, ObservationPayload]:
    attempt = evaluate_request_attempt(problem=problem, request=request)
    attempt_slots: list[
        EvaluationSuccess[int, ObservationPayload] | EvaluationFailure[int]
    ] = []
    for attempt_slot in attempt.attempts:
        if isinstance(attempt_slot, EvaluationFailure):
            attempt_slots.append(attempt_slot)
            continue

        refinement = None
        if attach_refinement:
            refinement = CandidateRefinement(
                source_candidate=request.candidate,
                refined_candidate=attempt_slot.request.candidate,
                changed_leaf_paths=((),),
            )
        attempt_slots.append(
            EvaluationSuccess(
                request=attempt_slot.request,
                payload=attempt_slot.payload,
                evaluation_count=attempt_slot.evaluation_count,
                refinement=refinement,
                kernel_diagnostics=attempt_slot.kernel_diagnostics,
            )
        )

    return EvaluationAttemptBatch(
        attempts=tuple(attempt_slots),
    )


def make_observation_outcome(
    *,
    problem: Problem[int, int],
    request: EvaluationRequest[int],
) -> EvaluationOutcome[int, Observation[int]]:
    outcome = evaluate_request_outcome(problem=problem, request=request)
    return _require_observation_outcome(outcome)


def _require_observation_outcome(
    outcome: EvaluationOutcome[int, RequestAlignedEvaluationRecord],
) -> EvaluationOutcome[int, Observation[int]]:
    record = outcome.record
    if not _is_int_observation_record(record):
        msg = "test fixture expected a scalar Observation compatibility record"
        raise TypeError(msg)

    return EvaluationOutcome(
        record=record,
        evaluation_count=outcome.evaluation_count,
        refinement=outcome.refinement,
    )


def _is_int_observation_record(
    record: RequestAlignedEvaluationRecord,
) -> TypeGuard[Observation[int]]:
    return isinstance(record, Observation)


def make_observation_payload_attempt(
    *,
    problem: Problem[int, int],
    request: EvaluationRequest[int],
) -> EvaluationAttemptBatch[int, ObservationPayload]:
    attempt = evaluate_request_attempt(problem=problem, request=request)
    attempt_slots: list[
        EvaluationSuccess[int, ObservationPayload] | EvaluationFailure[int]
    ] = []
    for attempt_slot in attempt.attempts:
        if isinstance(attempt_slot, EvaluationFailure):
            attempt_slots.append(attempt_slot)
            continue

        attempt_slots.append(
            EvaluationSuccess(
                request=attempt_slot.request,
                payload=attempt_slot.payload,
                evaluation_count=attempt_slot.evaluation_count,
                refinement=attempt_slot.refinement,
                kernel_diagnostics=attempt_slot.kernel_diagnostics,
            )
        )
    return EvaluationAttemptBatch(
        attempts=tuple(attempt_slots),
    )


class DecrementKernel(
    Kernel[
        ProposalBatchQuery[int, int, ObservationPayload],
        EvaluationAttemptBatch[int, ObservationPayload],
    ],
):
    """Kernel that deterministically moves candidates toward zero."""

    @override
    def run(
        self,
        query: ProposalBatchQuery[int, int, ObservationPayload],
        runner: Callable[
            [ProposalBatchQuery[int, int, ObservationPayload]],
            EvaluationAttemptBatch[int, ObservationPayload],
        ],
    ) -> EvaluationAttemptBatch[int, ObservationPayload]:
        successes: list[EvaluationSuccess[int, ObservationPayload]] = []
        for proposal_index, proposal in enumerate(query.proposals):
            refined_candidate = max(0, proposal.candidate - 1)
            local_attempts = runner(
                ProposalBatchQuery(
                    problem=query.problem,
                    proposals=(Proposal(candidate=refined_candidate),),
                    execution_resources=query.execution_resources,
                )
            )
            local_success = local_attempts.successes[0]
            local_payload = local_success.payload
            refinement = None
            if refined_candidate != proposal.candidate:
                refinement = CandidateRefinement(
                    source_candidate=proposal.candidate,
                    refined_candidate=refined_candidate,
                    changed_leaf_paths=((),),
                )
            proposal_evaluation_spec = None
            if query.proposal_evaluation_specs is not None:
                proposal_evaluation_spec = query.proposal_evaluation_specs[
                    proposal_index
                ]
            request = EvaluationRequest(
                proposal=Proposal(
                    candidate=refined_candidate,
                    proposal_id=proposal.proposal_id,
                ),
                proposal_evaluation_spec=proposal_evaluation_spec,
            )
            payload = ObservationPayload.from_objective_value(
                value=local_payload.value,
                direction=query.problem.direction,
                elapsed_seconds=local_payload.elapsed_seconds,
            )
            successes.append(
                EvaluationSuccess(
                    request=request,
                    payload=payload,
                    evaluation_count=local_success.evaluation_count,
                    refinement=refinement,
                )
            )
        return EvaluationAttemptBatch(attempts=tuple(successes))


class RefinementKernel(
    Kernel[
        ProposalBatchQuery[int, int, ObservationPayload],
        EvaluationAttemptBatch[int, ObservationPayload],
    ],
):
    """Kernel that emits explicit refinement metadata when it changes a candidate."""

    @override
    def run(
        self,
        query: ProposalBatchQuery[int, int, ObservationPayload],
        runner: Callable[
            [ProposalBatchQuery[int, int, ObservationPayload]],
            EvaluationAttemptBatch[int, ObservationPayload],
        ],
    ) -> EvaluationAttemptBatch[int, ObservationPayload]:
        successes: list[EvaluationSuccess[int, ObservationPayload]] = []
        for proposal_index, proposal in enumerate(query.proposals):
            refined_candidate = max(0, proposal.candidate - 1)
            local_attempts = runner(
                ProposalBatchQuery(
                    problem=query.problem,
                    proposals=(Proposal(candidate=refined_candidate),),
                    execution_resources=query.execution_resources,
                )
            )
            local_success = local_attempts.successes[0]
            local_payload = local_success.payload
            refinement = None
            if refined_candidate != proposal.candidate:
                refinement = CandidateRefinement(
                    source_candidate=proposal.candidate,
                    refined_candidate=refined_candidate,
                    changed_leaf_paths=((),),
                )
            proposal_evaluation_spec = None
            if query.proposal_evaluation_specs is not None:
                proposal_evaluation_spec = query.proposal_evaluation_specs[
                    proposal_index
                ]
            request = EvaluationRequest(
                proposal=Proposal(
                    candidate=refined_candidate,
                    proposal_id=proposal.proposal_id,
                ),
                proposal_evaluation_spec=proposal_evaluation_spec,
            )
            payload = ObservationPayload.from_objective_value(
                value=local_payload.value,
                direction=query.problem.direction,
                elapsed_seconds=local_payload.elapsed_seconds,
            )
            successes.append(
                EvaluationSuccess(
                    request=request,
                    payload=payload,
                    evaluation_count=local_success.evaluation_count,
                    refinement=refinement,
                )
            )
        return EvaluationAttemptBatch(attempts=tuple(successes))


class ScoringKernel(
    Kernel[
        ProposalBatchQuery[int, int, ObservationPayload],
        EvaluationAttemptBatch[int, ObservationPayload],
    ],
):
    """Kernel that returns precomputed candidate scores and cost."""

    @override
    def run(
        self,
        query: ProposalBatchQuery[int, int, ObservationPayload],
        runner: Callable[
            [ProposalBatchQuery[int, int, ObservationPayload]],
            EvaluationAttemptBatch[int, ObservationPayload],
        ],
    ) -> EvaluationAttemptBatch[int, ObservationPayload]:
        _ = runner
        successes: list[EvaluationSuccess[int, ObservationPayload]] = []
        for proposal_index, proposal in enumerate(query.proposals):
            refined_candidate = max(0, proposal.candidate - 2)
            refinement = None
            if refined_candidate != proposal.candidate:
                refinement = CandidateRefinement(
                    source_candidate=proposal.candidate,
                    refined_candidate=refined_candidate,
                    changed_leaf_paths=((),),
                )
            proposal_evaluation_spec = None
            if query.proposal_evaluation_specs is not None:
                proposal_evaluation_spec = query.proposal_evaluation_specs[
                    proposal_index
                ]
            request = EvaluationRequest(
                proposal=Proposal(
                    candidate=refined_candidate,
                    proposal_id=proposal.proposal_id,
                ),
                proposal_evaluation_spec=proposal_evaluation_spec,
            )
            payload = ObservationPayload.from_objective_value(
                value=float(refined_candidate * refined_candidate),
                direction=query.problem.direction,
            )
            successes.append(
                EvaluationSuccess(
                    request=request,
                    payload=payload,
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
        return EvaluationAttemptBatch(attempts=tuple(successes))


class RecordingExecutionResourcesKernel(
    Kernel[
        ProposalBatchQuery[int, int, ObservationPayload],
        EvaluationAttemptBatch[int, ObservationPayload],
    ],
):
    """Kernel that records received execution resources."""

    def __init__(self) -> None:
        self.last_execution_resources: ExecutionResources | None = None

    @override
    def run(
        self,
        query: ProposalBatchQuery[int, int, ObservationPayload],
        runner: Callable[
            [ProposalBatchQuery[int, int, ObservationPayload]],
            EvaluationAttemptBatch[int, ObservationPayload],
        ],
    ) -> EvaluationAttemptBatch[int, ObservationPayload]:
        self.last_execution_resources = query.execution_resources
        return runner(query)


@dataclass(frozen=True, slots=True)
class BatchQueueOptimizerState:
    """Explicit state for the batch-queue test optimizer."""

    remaining_batches: tuple[tuple[Proposal[int], ...], ...]
    tell_history: tuple[tuple[Observation[int], ...], ...] = ()
    ask_history: tuple[int, ...] = ()


class BatchQueueOptimizer(
    RunMethod[BatchQueueOptimizerState, Proposal[int], Observation[int]]
):
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


@dataclass(frozen=True, slots=True)
class FailureRecordingBatchQueueOptimizerState:
    """State for a batch-queue optimizer that consumes failed attempts."""

    remaining_batches: tuple[tuple[Proposal[int], ...], ...]
    tell_history: tuple[tuple[Observation[int], ...], ...] = ()
    failure_history: tuple[tuple[str | None, ...], ...] = ()
    ask_history: tuple[int, ...] = ()


class FailureRecordingBatchQueueOptimizer(
    RunMethod[
        FailureRecordingBatchQueueOptimizerState,
        Proposal[int],
        Observation[int],
    ]
):
    """Batch-queue optimizer that records success and failure attempts."""

    _initial_batches: tuple[tuple[Proposal[int], ...], ...]

    def __init__(self, proposal_batches: list[tuple[Proposal[int], ...]]) -> None:
        self._initial_batches = tuple(tuple(batch) for batch in proposal_batches)

    @override
    def create_initial_state(self) -> FailureRecordingBatchQueueOptimizerState:
        return FailureRecordingBatchQueueOptimizerState(
            remaining_batches=self._initial_batches,
        )

    @override
    def is_exhausted(self, state: FailureRecordingBatchQueueOptimizerState) -> bool:
        return len(state.remaining_batches) == 0

    @override
    def ask(
        self,
        state: FailureRecordingBatchQueueOptimizerState,
        batch_size: int = 1,
    ) -> tuple[
        tuple[Proposal[int], ...],
        FailureRecordingBatchQueueOptimizerState,
    ]:
        next_state = FailureRecordingBatchQueueOptimizerState(
            remaining_batches=state.remaining_batches,
            tell_history=state.tell_history,
            failure_history=state.failure_history,
            ask_history=state.ask_history + (batch_size,),
        )
        if not state.remaining_batches:
            return (), next_state

        return (
            state.remaining_batches[0],
            FailureRecordingBatchQueueOptimizerState(
                remaining_batches=state.remaining_batches[1:],
                tell_history=state.tell_history,
                failure_history=state.failure_history,
                ask_history=next_state.ask_history,
            ),
        )

    @override
    def tell(
        self,
        state: FailureRecordingBatchQueueOptimizerState,
        observations: Sequence[Observation[int]],
    ) -> FailureRecordingBatchQueueOptimizerState:
        return FailureRecordingBatchQueueOptimizerState(
            remaining_batches=state.remaining_batches,
            tell_history=state.tell_history + (tuple(observations),),
            failure_history=state.failure_history + ((),),
            ask_history=state.ask_history,
        )

    @override
    def tell_attempts(
        self,
        state: FailureRecordingBatchQueueOptimizerState,
        attempts: EvaluationAttemptBatch[OutcomeCandidateT, Observation[int]],
    ) -> FailureRecordingBatchQueueOptimizerState:
        return FailureRecordingBatchQueueOptimizerState(
            remaining_batches=state.remaining_batches,
            tell_history=state.tell_history
            + (materialize_success_records(attempts.successes),),
            failure_history=state.failure_history
            + (tuple(failure.proposal_id for failure in attempts.failures),),
            ask_history=state.ask_history,
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


class OutcomeAwareBatchQueueOptimizer(BatchQueueOptimizer):
    """Batch-queue optimizer that records full outcome metadata."""

    seen_changed_leaf_paths: tuple[tuple[LeafPath, ...] | None, ...]

    def __init__(self, proposal_batches: list[tuple[Proposal[int], ...]]) -> None:
        super().__init__(proposal_batches)
        self.seen_changed_leaf_paths = ()

    @override
    def tell_attempts(
        self,
        state: BatchQueueOptimizerState,
        attempts: EvaluationAttemptBatch[OutcomeCandidateT, Observation[int]],
    ) -> BatchQueueOptimizerState:
        if attempts.has_failures:
            return super().tell_attempts(state, attempts)

        self.seen_changed_leaf_paths += tuple(
            None
            if success.refinement is None
            else success.refinement.changed_leaf_paths
            for success in attempts.successes
        )
        return super().tell(
            state,
            materialize_success_records(attempts.successes),
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
    _pending_attempt_groups: dict[
        str,
        tuple[CompletionGroup[EvaluationAttemptBatch[int, ObservationPayload]], ...],
    ]
    _attach_refinement: bool

    def __init__(self, *, attach_refinement: bool = False) -> None:
        self._next_batch_id = 0
        self._pending_groups = {}
        self._pending_attempt_groups = {}
        self._attach_refinement = attach_refinement

    @property
    def pending_attempt_batch_ids(self) -> tuple[str, ...]:
        """Return pending attempt-batch ids still owned by this evaluator."""
        return tuple(self._pending_attempt_groups)

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

    def open_attempt_session(
        self,
        problem: Problem[int, int],
        requests: Sequence[EvaluationRequest[int]],
    ) -> EvaluationBatchSession[EvaluationAttemptBatch[int, ObservationPayload]]:
        handle = EvaluationBatchHandle(
            batch_id=f"attempt-batch-{self._next_batch_id}",
            request_count=len(requests),
        )
        self._next_batch_id += 1
        self._pending_attempt_groups[handle.batch_id] = tuple(
            CompletionGroup(
                start_index=index,
                outcomes=(
                    _make_async_evaluation_attempt(
                        problem,
                        request,
                        attach_refinement=self._attach_refinement,
                    ),
                ),
            )
            for index, request in reversed(tuple(enumerate(requests)))
        )
        return _AttemptOutOfOrderBatchSession(evaluator=self, _handle=handle)

    def poll_attempts(
        self,
        handle: EvaluationBatchHandle,
    ) -> Sequence[CompletionGroup[EvaluationAttemptBatch[int, ObservationPayload]]]:
        pending_groups = self._pending_attempt_groups.get(handle.batch_id)
        if pending_groups is None:
            msg = f"unknown async attempt batch handle: {handle.batch_id}"
            raise ValueError(msg)

        if len(pending_groups) == 0:
            raise BatchExecutionFailed(
                handle=handle,
                kind="infrastructure",
                cause=RuntimeError("async attempt batch exhausted unexpectedly"),
            )

        completion_group = pending_groups[0]
        remaining_groups = pending_groups[1:]
        if len(remaining_groups) == 0:
            _ = self._pending_attempt_groups.pop(handle.batch_id, None)
        else:
            self._pending_attempt_groups[handle.batch_id] = remaining_groups
        return (completion_group,)

    def cancel_attempts(self, handle: EvaluationBatchHandle) -> None:
        _ = self._pending_attempt_groups.pop(handle.batch_id, None)


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

    @override
    def open_attempt_session(
        self,
        problem: Problem[int, int],
        requests: Sequence[EvaluationRequest[int]],
    ) -> EvaluationBatchSession[EvaluationAttemptBatch[int, ObservationPayload]]:
        self.opened_batch_sizes += (len(requests),)
        return super().open_attempt_session(problem, requests)


@dataclass(slots=True)
class PayloadAttemptBatchSession(
    ResumableBatchSession[EvaluationAttemptBatch[int, ObservationPayload]]
):
    """Resumable session that streams payload attempt batches."""

    evaluator: "PayloadResumableOutOfOrderAsyncEvaluator"
    _handle: EvaluationBatchHandle
    _completed_count: int = 0

    @property
    @override
    def handle(self) -> EvaluationBatchHandle:
        return self._handle

    @override
    def poll(
        self,
    ) -> Sequence[CompletionGroup[EvaluationAttemptBatch[int, ObservationPayload]]]:
        completion_groups = self.evaluator.poll_attempts(self.handle)
        self._completed_count += sum(
            len(completion_group.outcomes) for completion_group in completion_groups
        )
        return completion_groups

    @override
    def cancel(self) -> None:
        self.evaluator.cancel(self.handle)

    @override
    def suspend(self) -> EvaluationBatchResumeHandle:
        return self.evaluator.suspend_attempt_batch(
            self.handle,
            completed_count=self._completed_count,
        )


class PayloadResumableOutOfOrderAsyncEvaluator(
    ResumableAsyncEvaluator[
        Problem[int, int, ObservationPayload],
        EvaluationRequest[int],
        EvaluationAttemptBatch[int, ObservationPayload],
    ],
):
    """Resumable async evaluator that emits request-free scalar payload attempts."""

    _next_batch_id: int
    _pending_attempt_groups: dict[
        str,
        tuple[CompletionGroup[EvaluationAttemptBatch[int, ObservationPayload]], ...],
    ]
    _suspended_attempt_groups: dict[
        str,
        tuple[CompletionGroup[EvaluationAttemptBatch[int, ObservationPayload]], ...],
    ]

    def __init__(self) -> None:
        self._next_batch_id = 0
        self._pending_attempt_groups = {}
        self._suspended_attempt_groups = {}

    @override
    def execution_resources(self) -> ExecutionResources:
        return ExecutionResources(
            parallel_owner="evaluator",
            nested_parallelism_policy=NestedParallelismPolicy.FORBID,
        )

    @override
    def submit_batch(
        self,
        problem: Problem[int, int, ObservationPayload],
        requests: Sequence[EvaluationRequest[int]],
    ) -> EvaluationBatchHandle:
        handle = self._new_handle(request_count=len(requests))
        self._pending_attempt_groups[handle.batch_id] = self._completion_groups(
            problem=problem,
            requests=requests,
        )
        return handle

    @override
    def poll(
        self,
        handle: EvaluationBatchHandle,
    ) -> Sequence[CompletionGroup[EvaluationAttemptBatch[int, ObservationPayload]]]:
        return self.poll_attempts(handle)

    @override
    def cancel(self, handle: EvaluationBatchHandle) -> None:
        _ = self._pending_attempt_groups.pop(handle.batch_id, None)
        _ = self._suspended_attempt_groups.pop(handle.batch_id, None)

    @override
    def resume_session(
        self,
        handle: EvaluationBatchResumeHandle,
    ) -> PayloadAttemptBatchSession:
        return self.resume_attempt_session(handle)

    def open_attempt_session(
        self,
        problem: Problem[int, int, ObservationPayload],
        requests: Sequence[EvaluationRequest[int]],
    ) -> PayloadAttemptBatchSession:
        handle = self.submit_batch(problem, requests)
        return PayloadAttemptBatchSession(evaluator=self, _handle=handle)

    def resume_attempt_session(
        self,
        handle: EvaluationBatchResumeHandle,
    ) -> PayloadAttemptBatchSession:
        pending_groups = self._suspended_attempt_groups.pop(handle.batch_id)
        self._pending_attempt_groups[handle.batch_id] = pending_groups
        return PayloadAttemptBatchSession(
            evaluator=self,
            _handle=EvaluationBatchHandle(
                batch_id=handle.batch_id,
                request_count=handle.request_count,
            ),
            _completed_count=handle.completed_count,
        )

    def poll_attempts(
        self,
        handle: EvaluationBatchHandle,
    ) -> tuple[CompletionGroup[EvaluationAttemptBatch[int, ObservationPayload]], ...]:
        pending_groups = self._pending_attempt_groups[handle.batch_id]
        completion_group = pending_groups[0]
        remaining_groups = pending_groups[1:]
        if len(remaining_groups) == 0:
            _ = self._pending_attempt_groups.pop(handle.batch_id)
        else:
            self._pending_attempt_groups[handle.batch_id] = remaining_groups
        return (completion_group,)

    def suspend_attempt_batch(
        self,
        handle: EvaluationBatchHandle,
        *,
        completed_count: int,
    ) -> EvaluationBatchResumeHandle:
        pending_groups = self._pending_attempt_groups.pop(handle.batch_id)
        self._suspended_attempt_groups[handle.batch_id] = pending_groups
        return EvaluationBatchResumeHandle(
            batch_id=handle.batch_id,
            request_count=handle.request_count,
            completed_count=completed_count,
        )

    def _new_handle(self, *, request_count: int) -> EvaluationBatchHandle:
        handle = EvaluationBatchHandle(
            batch_id=f"payload-attempt-batch-{self._next_batch_id}",
            request_count=request_count,
        )
        self._next_batch_id += 1
        return handle

    def _completion_groups(
        self,
        *,
        problem: Problem[int, int, ObservationPayload],
        requests: Sequence[EvaluationRequest[int]],
    ) -> tuple[CompletionGroup[EvaluationAttemptBatch[int, ObservationPayload]], ...]:
        return tuple(
            CompletionGroup(
                start_index=index,
                outcomes=(self._payload_attempt(problem=problem, request=request),),
            )
            for index, request in reversed(tuple(enumerate(requests)))
        )

    def _payload_attempt(
        self,
        *,
        problem: Problem[int, int, ObservationPayload],
        request: EvaluationRequest[int],
    ) -> EvaluationAttemptBatch[int, ObservationPayload]:
        payload = ObservationPayload.from_objective_value(
            value=float(request.candidate * request.candidate),
            direction=problem.direction,
        )
        return EvaluationAttemptBatch(
            attempts=(
                EvaluationSuccess(
                    request=request,
                    payload=payload,
                    evaluation_count=1,
                ),
            ),
        )


@dataclass(slots=True)
class _AttemptOutOfOrderBatchSession(
    EvaluationBatchSession[EvaluationAttemptBatch[int, ObservationPayload]]
):
    """Attempt-aware exact-async session for failure-recording tests."""

    evaluator: "OutOfOrderAsyncEvaluator"
    _handle: EvaluationBatchHandle

    @property
    @override
    def handle(self) -> EvaluationBatchHandle:
        return self._handle

    @override
    def poll(
        self,
    ) -> Sequence[CompletionGroup[EvaluationAttemptBatch[int, ObservationPayload]]]:
        return self.evaluator.poll_attempts(self.handle)

    @override
    def cancel(self) -> None:
        self.evaluator.cancel_attempts(self.handle)


@final
class AttemptOutOfOrderAsyncEvaluator(OutOfOrderAsyncEvaluator):
    """Async evaluator that streams one-slot attempt batches in reverse order."""

    _pending_attempt_groups: dict[
        str,
        tuple[CompletionGroup[EvaluationAttemptBatch[int, ObservationPayload]], ...],
    ]

    def __init__(self) -> None:
        super().__init__()
        self._pending_attempt_groups = {}

    @override
    def open_attempt_session(
        self,
        problem: Problem[int, int],
        requests: Sequence[EvaluationRequest[int]],
    ) -> EvaluationBatchSession[EvaluationAttemptBatch[int, ObservationPayload]]:
        handle = EvaluationBatchHandle(
            batch_id=f"attempt-batch-{self._next_batch_id}",
            request_count=len(requests),
        )
        self._next_batch_id += 1
        self._pending_attempt_groups[handle.batch_id] = tuple(
            CompletionGroup(
                start_index=index,
                outcomes=(
                    _make_async_evaluation_attempt(
                        problem,
                        request,
                        attach_refinement=self._attach_refinement,
                    ),
                ),
            )
            for index, request in reversed(tuple(enumerate(requests)))
        )
        return _AttemptOutOfOrderBatchSession(evaluator=self, _handle=handle)

    @override
    def poll_attempts(
        self,
        handle: EvaluationBatchHandle,
    ) -> Sequence[CompletionGroup[EvaluationAttemptBatch[int, ObservationPayload]]]:
        pending_groups = self._pending_attempt_groups.get(handle.batch_id)
        if pending_groups is None:
            msg = f"unknown async attempt batch handle: {handle.batch_id}"
            raise ValueError(msg)

        if len(pending_groups) == 0:
            raise BatchExecutionFailed(
                handle=handle,
                kind="infrastructure",
                cause=RuntimeError("async attempt batch exhausted unexpectedly"),
            )

        completion_group = pending_groups[0]
        remaining_groups = pending_groups[1:]
        if len(remaining_groups) == 0:
            _ = self._pending_attempt_groups.pop(handle.batch_id, None)
        else:
            self._pending_attempt_groups[handle.batch_id] = remaining_groups
        return (completion_group,)

    @override
    def cancel_attempts(self, handle: EvaluationBatchHandle) -> None:
        _ = self._pending_attempt_groups.pop(handle.batch_id, None)


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
            len(completion_group.outcomes) for completion_group in completion_groups
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


@dataclass(slots=True)
class _ResumableAttemptOutOfOrderBatchSession(
    ResumableBatchSession[EvaluationAttemptBatch[int, ObservationPayload]]
):
    """Resumable attempt-aware async session for study orchestration tests."""

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
    ) -> Sequence[CompletionGroup[EvaluationAttemptBatch[int, ObservationPayload]]]:
        completion_groups = tuple(self.evaluator.poll_attempts(self.handle))
        self._completed_count += sum(
            len(completion_group.outcomes) for completion_group in completion_groups
        )
        return completion_groups

    @override
    def cancel(self) -> None:
        self.evaluator.cancel_attempts(self.handle)

    @override
    def suspend(self) -> EvaluationBatchResumeHandle:
        return self.evaluator.suspend_attempt_batch(
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
    _pending_attempt_groups: dict[
        str,
        tuple[CompletionGroup[EvaluationAttemptBatch[int, ObservationPayload]], ...],
    ]
    _suspended_attempt_groups: dict[
        str,
        tuple[CompletionGroup[EvaluationAttemptBatch[int, ObservationPayload]], ...],
    ]

    def __init__(self, *, attach_refinement: bool = False) -> None:
        self._next_batch_id = 0
        self._pending_groups = {}
        self._suspended_groups = {}
        self._pending_attempt_groups = {}
        self._suspended_attempt_groups = {}
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

    def open_attempt_session(
        self,
        problem: Problem[int, int],
        requests: Sequence[EvaluationRequest[int]],
    ) -> EvaluationBatchSession[EvaluationAttemptBatch[int, ObservationPayload]]:
        handle = EvaluationBatchHandle(
            batch_id=f"resumable-attempt-batch-{self._next_batch_id}",
            request_count=len(requests),
        )
        self._next_batch_id += 1
        self._pending_attempt_groups[handle.batch_id] = tuple(
            CompletionGroup(
                start_index=index,
                outcomes=(
                    _make_async_evaluation_attempt(
                        problem,
                        request,
                        attach_refinement=self._attach_refinement,
                    ),
                ),
            )
            for index, request in reversed(tuple(enumerate(requests)))
        )
        return _ResumableAttemptOutOfOrderBatchSession(
            evaluator=self,
            _handle=handle,
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

    def poll_attempts(
        self,
        handle: EvaluationBatchHandle,
    ) -> Sequence[CompletionGroup[EvaluationAttemptBatch[int, ObservationPayload]]]:
        pending_groups = self._pending_attempt_groups.get(handle.batch_id)
        if pending_groups is None:
            msg = f"unknown async attempt batch handle: {handle.batch_id}"
            raise ValueError(msg)

        if len(pending_groups) == 0:
            raise BatchExecutionFailed(
                handle=handle,
                kind="infrastructure",
                cause=RuntimeError("async attempt batch exhausted unexpectedly"),
            )

        completion_group = pending_groups[0]
        remaining_groups = pending_groups[1:]
        if len(remaining_groups) == 0:
            _ = self._pending_attempt_groups.pop(handle.batch_id, None)
        else:
            self._pending_attempt_groups[handle.batch_id] = remaining_groups
        return (completion_group,)

    def cancel_attempts(self, handle: EvaluationBatchHandle) -> None:
        _ = self._pending_attempt_groups.pop(handle.batch_id, None)
        _ = self._suspended_attempt_groups.pop(handle.batch_id, None)

    def suspend_attempt_batch(
        self,
        handle: EvaluationBatchHandle,
        *,
        completed_count: int,
    ) -> EvaluationBatchResumeHandle:
        pending_groups = self._pending_attempt_groups.pop(handle.batch_id, None)
        if pending_groups is None:
            msg = f"unknown async attempt batch handle: {handle.batch_id}"
            raise ValueError(msg)

        self._suspended_attempt_groups[handle.batch_id] = pending_groups
        return EvaluationBatchResumeHandle(
            batch_id=handle.batch_id,
            request_count=handle.request_count,
            completed_count=completed_count,
        )

    def resume_attempt_session(
        self,
        handle: EvaluationBatchResumeHandle,
    ) -> EvaluationBatchSession[EvaluationAttemptBatch[int, ObservationPayload]]:
        pending_groups = self._suspended_attempt_groups.pop(handle.batch_id, None)
        if pending_groups is None:
            msg = f"unknown suspended attempt batch handle: {handle.batch_id}"
            raise ValueError(msg)

        self._pending_attempt_groups[handle.batch_id] = pending_groups
        return _ResumableAttemptOutOfOrderBatchSession(
            evaluator=self,
            _handle=EvaluationBatchHandle(
                batch_id=handle.batch_id,
                request_count=handle.request_count,
            ),
            _completed_count=handle.completed_count,
        )


@final
class NonResumableSessionResumableAsyncEvaluator(
    ResumableAsyncEvaluator[
        Problem[int, int],
        EvaluationRequest[int],
        EvaluationOutcome[int, Observation[int]],
    ]
):
    """Resumable evaluator double that opens non-resumable batch sessions."""

    _delegate: OutOfOrderAsyncEvaluator

    def __init__(self) -> None:
        self._delegate = OutOfOrderAsyncEvaluator()

    @property
    def pending_attempt_batch_ids(self) -> tuple[str, ...]:
        """Return non-resumable attempt batches still owned by the delegate."""
        return self._delegate.pending_attempt_batch_ids

    @override
    def submit_batch(
        self,
        problem: Problem[int, int],
        requests: Sequence[EvaluationRequest[int]],
    ) -> EvaluationBatchHandle:
        return self._delegate.submit_batch(problem, requests)

    @override
    def poll(
        self,
        handle: EvaluationBatchHandle,
    ) -> Sequence[CompletionGroup[EvaluationOutcome[int, Observation[int]]]]:
        return self._delegate.poll(handle)

    @override
    def cancel(self, handle: EvaluationBatchHandle) -> None:
        self._delegate.cancel(handle)

    def open_attempt_session(
        self,
        problem: Problem[int, int],
        requests: Sequence[EvaluationRequest[int]],
    ) -> EvaluationBatchSession[EvaluationAttemptBatch[int, ObservationPayload]]:
        return self._delegate.open_attempt_session(problem, requests)

    def resume_attempt_session(
        self,
        handle: EvaluationBatchResumeHandle,
    ) -> EvaluationBatchSession[EvaluationAttemptBatch[int, ObservationPayload]]:
        _ = handle
        msg = "test double does not support resumed non-resumable attempt sessions"
        raise RuntimeError(msg)

    @override
    def resume_session(
        self,
        handle: EvaluationBatchResumeHandle,
    ) -> EvaluationBatchSession[EvaluationOutcome[int, Observation[int]]]:
        _ = handle
        msg = "test double does not support resumed non-resumable sessions"
        raise RuntimeError(msg)


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
            make_observation_outcome(problem=problem, request=request)
            for request in reversed(requests)
        )

    def evaluate_attempts(
        self,
        problem: Problem[int, int],
        requests: Sequence[EvaluationRequest[int]],
    ) -> EvaluationAttemptBatch[int, ObservationPayload]:
        request_attempts: tuple[EvaluationAttemptBatch[int, ObservationPayload], ...]
        request_attempts = tuple(
            make_observation_payload_attempt(problem=problem, request=request)
            for request in reversed(requests)
        )
        return EvaluationAttemptBatch[int, ObservationPayload].concatenate(
            request_attempts,
        )


@final
class HardFailingEvaluator(
    Evaluator[
        Problem[int, int],
        EvaluationRequest[int],
        EvaluationOutcome[int, Observation[int]],
    ]
):
    """Evaluator that raises an infrastructure-style failure on one call."""

    _call_count: int
    _fail_on_call: int

    def __init__(self, *, fail_on_call: int) -> None:
        self._call_count = 0
        self._fail_on_call = fail_on_call

    @override
    def evaluate(
        self,
        problem: Problem[int, int],
        requests: Sequence[EvaluationRequest[int]],
    ) -> Sequence[EvaluationOutcome[int, Observation[int]]]:
        self._call_count += 1
        if self._call_count == self._fail_on_call:
            raise RuntimeError(f"hard evaluator failure {self._call_count}")

        return tuple(
            make_observation_outcome(problem=problem, request=request)
            for request in requests
        )

    def evaluate_attempts(
        self,
        problem: Problem[int, int],
        requests: Sequence[EvaluationRequest[int]],
    ) -> EvaluationAttemptBatch[int, ObservationPayload]:
        self._call_count += 1
        if self._call_count == self._fail_on_call:
            raise RuntimeError(f"hard evaluator failure {self._call_count}")

        request_attempts: tuple[EvaluationAttemptBatch[int, ObservationPayload], ...]
        request_attempts = tuple(
            make_observation_payload_attempt(problem=problem, request=request)
            for request in requests
        )
        return EvaluationAttemptBatch[int, ObservationPayload].concatenate(
            request_attempts,
        )


class RecordingKernel(
    Kernel[
        ProposalBatchQuery[int, int, ObservationPayload],
        EvaluationAttemptBatch[int, ObservationPayload],
    ],
):
    """Pass-through kernel that records received queries."""

    def __init__(self) -> None:
        self.queries: list[ProposalBatchQuery[int, int, ObservationPayload]] = []

    @override
    def run(
        self,
        query: ProposalBatchQuery[int, int, ObservationPayload],
        runner: Callable[
            [ProposalBatchQuery[int, int, ObservationPayload]],
            EvaluationAttemptBatch[int, ObservationPayload],
        ],
    ) -> EvaluationAttemptBatch[int, ObservationPayload]:
        self.queries.append(query)
        return runner(query)
