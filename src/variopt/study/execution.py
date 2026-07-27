"""Generic study step and run orchestration."""

from contextlib import nullcontext
from dataclasses import dataclass, replace
from typing import Generic, Protocol, TypeGuard

from typing_extensions import TypeVar

from variopt.generic_runtime import FrozenGenericSlotsCompat

from ..artifacts import (
    DefaultEvaluationAttemptMaterializer,
    EvaluationAttemptBatch,
    EvaluationAttemptMaterializer,
    EvaluationFailure,
    EvaluationRequest,
    EvaluationSuccess,
    Observation,
    ObservationPayload,
    Proposal,
    RunReport,
    RunResult,
    Trace,
    materialize_success_records,
)
from ..artifacts.alignment import validate_aligned_attempts
from ..evaluators.episodes import RequestLocalEpisodeEvaluator
from ..evaluators.sequential import SequentialEvaluator
from ..execution import (
    EXACT_ASYNC_EXECUTION_MODEL,
    SEQUENTIAL_EXECUTION_MODEL,
    STALE_ASYNC_EXECUTION_MODEL,
    SYNC_BATCH_EXECUTION_MODEL,
    EvaluationBudget,
    EvaluationBudgetExhausted,
    EvaluationReservationBatch,
    ExecutionAssimilationMode,
    ExecutionModel,
)
from ..kernel import (
    DirectKernel,
    Kernel,
    ProposalBatchQuery,
    ProposalKernelHint,
    ProposalLocalSearchContext,
    RequestLocalEpisode,
    is_explicit_request_local_episode_kernel,
)
from ..methods import RunMethod
from ..objective import Objective, ScalarEvaluationProtocol
from ..problem import Problem
from ..randomness import RandomStateSnapshot
from ..spaces import CandidateEquality
from ..typevars import CandidateT, RunMethodStateT
from .assimilation import (
    StudyAssimilatedStep,
    StudyRunHistory,
    materialize_feedback_attempts,
)
from .common import (
    AttemptBatchEvaluator,
    CheckpointSafeRunSnapshot,
    StudyEvaluator,
    StudyPayloadT,
    StudyRecordT,
    build_evaluation_requests,
    open_attempt_run_scope,
    supports_attempt_batches,
)
from .exact_async.orchestration import evaluate_batch_exact_async
from .validation import validate_execution_request

BoundaryT = TypeVar("BoundaryT")


@dataclass(frozen=True, slots=True)
class StudyStepResult(
    FrozenGenericSlotsCompat,
    Generic[CandidateT, RunMethodStateT, StudyRecordT],
):
    """Canonical in-process result for one ask/evaluate/tell study step.

    Parameters
    ----------
    attempts : EvaluationAttemptBatch[CandidateT, StudyRecordT]
        Dense materialized feedback attempts assimilated by the run method.
    state : RunMethodStateT
        Run-method state after assimilating ``attempts``.
    evaluation_count : int
        Evaluation units consumed by the step after hard-budget reconciliation.
    """

    attempts: EvaluationAttemptBatch[CandidateT, StudyRecordT]
    state: RunMethodStateT
    evaluation_count: int


@dataclass(frozen=True, slots=True)
class _StudyStepFeedback(
    FrozenGenericSlotsCompat,
    Generic[CandidateT, RunMethodStateT, StudyRecordT],
):
    """Materialized step feedback before run-method assimilation."""

    attempts: EvaluationAttemptBatch[CandidateT, StudyRecordT]
    post_ask_state: RunMethodStateT
    evaluation_count: int


class StudyExecutionOwner(
    Protocol[BoundaryT, CandidateT, RunMethodStateT, StudyPayloadT, StudyRecordT]
):
    """Protocol for the subset of Study state generic execution needs.

    Notes
    -----
    The generic execution helpers operate on this protocol instead of the
    concrete :class:`variopt.study.Study` class so the same orchestration can
    be reused across sync, exact-async, and stale-async study tiers.
    """

    @property
    def problem(
        self,
    ) -> Problem[BoundaryT, CandidateT, StudyPayloadT]:
        """Return the configured problem."""
        ...

    @property
    def run_method(
        self,
    ) -> RunMethod[
        RunMethodStateT,
        Proposal[CandidateT],
        StudyRecordT,
    ]:
        """Return the configured run method."""
        ...

    @property
    def evaluator(
        self,
    ) -> StudyEvaluator[BoundaryT, CandidateT, StudyPayloadT]:
        """Return the configured evaluator."""
        ...

    @property
    def kernel(
        self,
    ) -> Kernel[
        ProposalBatchQuery[BoundaryT, CandidateT, StudyPayloadT],
        EvaluationAttemptBatch[CandidateT, StudyPayloadT],
    ]:
        """Return the configured kernel."""
        ...

    @property
    def attempt_materializer(
        self,
    ) -> EvaluationAttemptMaterializer[CandidateT, StudyPayloadT, StudyRecordT]:
        """Return the payload-to-record feedback materializer."""
        ...


class DirectScalarSequentialStudyOwner(
    Protocol[BoundaryT, CandidateT, RunMethodStateT],
):
    """Protocol for the narrow direct-scalar sequential optimize fast path."""

    @property
    def problem(self) -> Problem[BoundaryT, CandidateT, Observation[CandidateT]]:
        """Return the configured scalar problem."""
        ...

    @property
    def run_method(
        self,
    ) -> RunMethod[RunMethodStateT, Proposal[CandidateT], Observation[CandidateT]]:
        """Return the configured run method."""
        ...

    @property
    def evaluator(
        self,
    ) -> SequentialEvaluator[BoundaryT, CandidateT, Observation[CandidateT]]:
        """Return the configured sequential evaluator."""
        ...

    @property
    def kernel(
        self,
    ) -> DirectKernel[
        ProposalBatchQuery[BoundaryT, CandidateT, Observation[CandidateT]],
        EvaluationAttemptBatch[CandidateT, Observation[CandidateT]],
    ]:
        """Return the configured direct kernel."""
        ...


def _supports_direct_scalar_sequential_path(
    study: StudyExecutionOwner[
        BoundaryT,
        CandidateT,
        RunMethodStateT,
        StudyPayloadT,
        StudyRecordT,
    ],
    *,
    execution_model: ExecutionModel,
) -> TypeGuard[
    DirectScalarSequentialStudyOwner[BoundaryT, CandidateT, RunMethodStateT]
]:
    if execution_model not in {
        SEQUENTIAL_EXECUTION_MODEL,
        SYNC_BATCH_EXECUTION_MODEL,
    }:
        return False

    kernel = study.kernel
    if not isinstance(kernel, DirectKernel):
        return False
    if kernel.__class__ is not DirectKernel:
        return False

    evaluator = study.evaluator
    if not isinstance(evaluator, SequentialEvaluator):
        return False
    if evaluator.__class__ is not SequentialEvaluator:
        return False

    attempt_materializer = study.attempt_materializer
    if not isinstance(attempt_materializer, DefaultEvaluationAttemptMaterializer):
        return False
    if attempt_materializer.__class__ is not DefaultEvaluationAttemptMaterializer:
        return False

    objective = study.problem.direct_objective
    return objective is not None and _uses_default_scalar_request_evaluation(objective)


def _uses_default_scalar_request_evaluation(
    objective: Objective[CandidateT],
) -> bool:
    for cls in type(objective).__mro__:
        if "evaluate_request" in cls.__dict__:
            return cls is ScalarEvaluationProtocol

    return False


def _query_with_evaluation_budget(
    query: ProposalBatchQuery[BoundaryT, CandidateT, StudyPayloadT],
    evaluation_budget: EvaluationBudget | None,
) -> ProposalBatchQuery[BoundaryT, CandidateT, StudyPayloadT]:
    """Return ``query`` with the active budget ledger attached."""
    if evaluation_budget is None or query.evaluation_budget is not None:
        return query
    return replace(query, evaluation_budget=evaluation_budget)


def _consume_reported_evaluation_cost(
    *,
    evaluation_budget: EvaluationBudget | None,
    remaining_before: int | None,
    reported_evaluation_count: int,
) -> int:
    """Reconcile reported outcome cost against pre-consumed runner cost."""
    if evaluation_budget is None or remaining_before is None:
        return reported_evaluation_count

    consumed_by_runner = remaining_before - evaluation_budget.remaining
    if reported_evaluation_count > consumed_by_runner:
        evaluation_budget.consume(reported_evaluation_count - consumed_by_runner)
        return reported_evaluation_count
    return consumed_by_runner


def _max_min_evaluation_limits(
    preferred_limits: tuple[int, ...],
    *,
    available_evaluations: int,
) -> tuple[int, ...] | None:
    """Allocate a deterministic starvation-free batch of hard limits."""
    request_count = len(preferred_limits)
    if request_count == 0:
        return ()
    if available_evaluations < request_count:
        return None
    if available_evaluations >= sum(preferred_limits):
        return preferred_limits

    limits = [1] * request_count
    remaining = available_evaluations - request_count
    active_limits = sorted(
        (preferred_limit, request_index)
        for request_index, preferred_limit in enumerate(preferred_limits)
        if preferred_limit > 1
    )
    active_start = 0
    water_level = 1
    while remaining > 0:
        next_limit = active_limits[active_start][0]
        active_count = len(active_limits) - active_start
        evaluations_to_next_limit = (next_limit - water_level) * active_count
        if remaining < evaluations_to_next_limit:
            complete_rounds, remainder = divmod(remaining, active_count)
            final_level = water_level + complete_rounds
            active_request_indices = sorted(
                request_index for _, request_index in active_limits[active_start:]
            )
            for request_index in active_request_indices:
                limits[request_index] = final_level
            for request_index in active_request_indices[:remainder]:
                limits[request_index] += 1
            remaining = 0
            break

        remaining -= evaluations_to_next_limit
        water_level = next_limit
        while (
            active_start < len(active_limits)
            and active_limits[active_start][0] == water_level
        ):
            _, request_index = active_limits[active_start]
            limits[request_index] = water_level
            active_start += 1

        if remaining == 0:
            for _, request_index in active_limits[active_start:]:
                limits[request_index] = water_level
            break

    return tuple(limits)


def _lower_request_local_hint(
    hint: ProposalKernelHint | None,
) -> tuple[ProposalKernelHint | None, RandomStateSnapshot | None]:
    """Separate a local-search hint from its authoritative RNG snapshot."""
    if not isinstance(hint, ProposalLocalSearchContext):
        return hint, None
    return (
        replace(hint, random_state_snapshot=None),
        hint.random_state_snapshot,
    )


def _rebind_attempt_candidate_equality(
    attempts: EvaluationAttemptBatch[CandidateT, StudyPayloadT],
    *,
    candidate_equal: CandidateEquality[CandidateT],
) -> EvaluationAttemptBatch[CandidateT, StudyPayloadT]:
    """Bind coordinator-owned equality after evaluator-owned execution."""
    if len(attempts.successes) == 0:
        return attempts

    rebound_attempts = list(attempts.attempts)
    for success_index, success in zip(
        attempts.success_indices,
        attempts.successes,
        strict=True,
    ):
        rebound_attempts[success_index] = success.with_payload(
            success.payload,
            candidate_equal=candidate_equal,
        )
    return EvaluationAttemptBatch(attempts=rebound_attempts)


def _evaluate_request_local_episode_batch(
    study: StudyExecutionOwner[
        BoundaryT,
        CandidateT,
        RunMethodStateT,
        StudyPayloadT,
        StudyRecordT,
    ],
    *,
    execution_model: ExecutionModel,
    attempt_evaluator: StudyEvaluator[
        BoundaryT,
        CandidateT,
        StudyPayloadT,
    ],
    query: ProposalBatchQuery[BoundaryT, CandidateT, StudyPayloadT],
    requests: tuple[EvaluationRequest[CandidateT], ...],
    evaluation_budget: EvaluationBudget | None,
) -> tuple[EvaluationAttemptBatch[CandidateT, StudyPayloadT], int] | None:
    """Execute one eligible synchronous batch as evaluator-owned episodes."""
    if execution_model not in {
        SEQUENTIAL_EXECUTION_MODEL,
        SYNC_BATCH_EXECUTION_MODEL,
    }:
        return None
    if evaluation_budget is None:
        return None
    if not isinstance(attempt_evaluator, RequestLocalEpisodeEvaluator):
        return None

    kernel = study.kernel
    if not is_explicit_request_local_episode_kernel(kernel):
        return None

    hints = query.proposal_kernel_hints
    preferred_limits: list[int] = []
    lowered_hints: list[ProposalKernelHint | None] = []
    random_state_snapshots: list[RandomStateSnapshot | None] = []
    for request_index, request in enumerate(requests):
        hint = None if hints is None else hints[request_index]
        preferred_limit = kernel.preferred_request_local_evaluation_limit(
            problem=study.problem,
            request=request,
            proposal_kernel_hint=hint,
        )
        if preferred_limit is None:
            return None
        if type(preferred_limit) is not int:
            msg = "preferred request-local evaluation limits must be exact integers"
            raise TypeError(msg)
        if preferred_limit <= 0:
            msg = "preferred request-local evaluation limits must be positive"
            raise ValueError(msg)

        lowered_hint, random_state_snapshot = _lower_request_local_hint(hint)
        preferred_limits.append(preferred_limit)
        lowered_hints.append(lowered_hint)
        random_state_snapshots.append(random_state_snapshot)

    limits = _max_min_evaluation_limits(
        tuple(preferred_limits),
        available_evaluations=evaluation_budget.remaining,
    )
    if limits is None:
        return None

    reservation = EvaluationReservationBatch(
        budget=evaluation_budget,
        limits=limits,
    )
    try:
        episodes = tuple(
            RequestLocalEpisode(
                request_index=request_index,
                request=request,
                kernel=kernel,
                proposal_kernel_hint=lowered_hints[request_index],
                random_state_snapshot=random_state_snapshots[request_index],
                evaluation_limit=limits[request_index],
                execution_resources=query.execution_resources,
            )
            for request_index, request in enumerate(requests)
        )
        attempts = attempt_evaluator.evaluate_request_local_episodes(
            study.problem,
            episodes,
        )
        validate_aligned_attempts(
            requests,
            attempts,
            candidate_equal=study.problem.space.candidates_equal,
        )
        attempts = _rebind_attempt_candidate_equality(
            attempts,
            candidate_equal=study.problem.space.candidates_equal,
        )
        consumed_counts = tuple(
            attempt.evaluation_count for attempt in attempts.attempts
        )
        consumed_evaluations = reservation.settle(consumed_counts)
    except Exception:
        if not reservation.is_finalized:
            reservation.forfeit()
        raise

    return attempts, consumed_evaluations


def _current_remaining_budget(
    *,
    evaluation_budget: EvaluationBudget | None,
    record_budget_remaining: int,
) -> int:
    """Return the active loop budget for evaluation- or attempt-slot-count mode."""
    if evaluation_budget is None:
        return record_budget_remaining
    return evaluation_budget.remaining


def _current_evaluation_count(
    *,
    max_evaluations: int,
    evaluation_budget: EvaluationBudget | None,
    record_budget_remaining: int,
) -> int:
    """Return the current terminal accounting projection."""
    if evaluation_budget is None:
        return max_evaluations - record_budget_remaining
    return max_evaluations - evaluation_budget.remaining


def _optimize_direct_scalar_sequential(
    study: DirectScalarSequentialStudyOwner[
        BoundaryT,
        CandidateT,
        RunMethodStateT,
    ],
    max_evaluations: int,
    batch_size: int,
    *,
    execution_model: ExecutionModel,
    count_evaluation_cost: bool,
    initial_state: RunMethodStateT | None,
    stop_at_checkpoint_boundary: bool,
) -> tuple[RunResult[CandidateT], RunMethodStateT]:
    if max_evaluations < 0:
        msg = "max_evaluations must be non-negative"
        raise ValueError(msg)

    validate_execution_request(
        study,
        batch_size=batch_size,
        execution_model=execution_model,
    )

    objective = study.problem.direct_objective
    if objective is None:
        msg = "direct scalar objective fast path requires a direct Objective"
        raise RuntimeError(msg)

    run_history = StudyRunHistory[
        CandidateT, RunMethodStateT, Observation[CandidateT]
    ]()
    observations: list[Observation[CandidateT]] = []
    evaluation_budget = (
        EvaluationBudget(max_evaluations) if count_evaluation_cost else None
    )
    record_budget_remaining = max_evaluations
    state = (
        study.run_method.create_initial_state()
        if initial_state is None
        else initial_state
    )
    safe_snapshot: CheckpointSafeRunSnapshot[RunMethodStateT] | None = None
    unsafe_since_safe_snapshot = False
    if stop_at_checkpoint_boundary and study.run_method.is_checkpoint_safe_state(state):
        safe_snapshot = run_history.checkpoint_snapshot(state)

    while _current_remaining_budget(
        evaluation_budget=evaluation_budget,
        record_budget_remaining=record_budget_remaining,
    ) > 0 and not study.run_method.is_exhausted(state):
        remaining = _current_remaining_budget(
            evaluation_budget=evaluation_budget,
            record_budget_remaining=record_budget_remaining,
        )
        current_batch_size = min(batch_size, remaining)
        try:
            proposals, next_state = study.run_method.ask(
                state,
                batch_size=current_batch_size,
            )
            if len(proposals) == 0:
                msg = "run_method returned no proposals"
                raise RuntimeError(msg)

            if len(proposals) > current_batch_size:
                msg = "run_method returned more proposals than requested"
                raise ValueError(msg)

            proposal_evaluation_specs = study.run_method.proposal_evaluation_specs(
                next_state,
                proposals,
            )
            if proposal_evaluation_specs is not None and (
                len(proposal_evaluation_specs) != len(proposals)
            ):
                msg = "proposal_evaluation_specs must align one-to-one with proposals"
                raise ValueError(msg)

            batch_attempt_slots: list[
                EvaluationSuccess[CandidateT, Observation[CandidateT]]
                | EvaluationFailure[CandidateT]
            ] = []
            if evaluation_budget is not None:
                evaluation_budget.consume(len(proposals))
            for index, proposal in enumerate(proposals):
                candidate = proposal.candidate
                study.problem.space.validate(candidate)
                proposal_evaluation_spec = (
                    None
                    if proposal_evaluation_specs is None
                    else proposal_evaluation_specs[index]
                )
                request = EvaluationRequest(
                    proposal=proposal,
                    proposal_evaluation_spec=proposal_evaluation_spec,
                )
                try:
                    value = objective.evaluate(candidate)
                    observation = Observation.from_objective_value(
                        request=request,
                        candidate=candidate,
                        value=value,
                        direction=study.problem.direction,
                    )
                except Exception as exception:  # noqa: BLE001 - capture objective failures
                    batch_attempt_slots.append(
                        EvaluationFailure[CandidateT].from_exception(
                            request=request,
                            exception=exception,
                        ),
                    )
                    continue

                batch_attempt_slots.append(
                    EvaluationSuccess[CandidateT, Observation[CandidateT]](
                        request=request,
                        payload=observation,
                        evaluation_count=1,
                    )
                )

            batch_attempts: EvaluationAttemptBatch[
                CandidateT,
                Observation[CandidateT],
            ] = EvaluationAttemptBatch(
                attempts=tuple(batch_attempt_slots),
            )
        except EvaluationBudgetExhausted:
            if stop_at_checkpoint_boundary and safe_snapshot is not None:
                checkpoint_report = run_history.checkpoint_report(
                    safe_snapshot,
                    candidate_equal=study.problem.space.candidates_equal,
                )
                return (
                    materialize_scalar_run_result(
                        checkpoint_report,
                        candidate_equal=study.problem.space.candidates_equal,
                    ),
                    safe_snapshot.state,
                )
            raise
        except Exception as exception:  # noqa: BLE001 - wrap execution failures
            run_history.raise_run_execution_failed(
                cause=exception,
                evaluation_count=_current_evaluation_count(
                    max_evaluations=max_evaluations,
                    evaluation_budget=evaluation_budget,
                    record_budget_remaining=record_budget_remaining,
                ),
                state=state,
                safe_snapshot=safe_snapshot,
                candidate_equal=study.problem.space.candidates_equal,
            )
        step = StudyAssimilatedStep[
            CandidateT,
            Observation[CandidateT],
        ].from_attempts(
            batch_attempts,
            evaluation_count=batch_attempts.evaluation_count,
        )
        run_history.append_step(step)
        try:
            next_run_state = study.run_method.tell_attempts(next_state, batch_attempts)
        except Exception as exception:  # noqa: BLE001 - wrap run-method failures
            run_history.raise_run_execution_failed(
                cause=exception,
                state=next_state,
                safe_snapshot=safe_snapshot,
                candidate_equal=study.problem.space.candidates_equal,
            )
        state = next_run_state
        observations.extend(step.records)
        if evaluation_budget is None:
            record_budget_remaining -= batch_attempts.attempt_count
        if stop_at_checkpoint_boundary:
            if study.run_method.is_checkpoint_safe_state(state):
                safe_snapshot = run_history.checkpoint_snapshot(state)
                if unsafe_since_safe_snapshot:
                    break
                unsafe_since_safe_snapshot = False
            else:
                unsafe_since_safe_snapshot = True

    if stop_at_checkpoint_boundary and not study.run_method.is_checkpoint_safe_state(
        state
    ):
        if safe_snapshot is None:
            msg = (
                "run did not reach a checkpoint-safe state within the evaluation budget"
            )
            raise RuntimeError(msg)
        checkpoint_report = run_history.checkpoint_report(
            safe_snapshot,
            candidate_equal=study.problem.space.candidates_equal,
        )
        return (
            materialize_scalar_run_result(
                checkpoint_report,
                candidate_equal=study.problem.space.candidates_equal,
            ),
            safe_snapshot.state,
        )

    return (
        RunResult[CandidateT].from_observations(
            observations=tuple(observations),
            evaluation_count=(
                max_evaluations - record_budget_remaining
                if evaluation_budget is None
                else max_evaluations - evaluation_budget.remaining
            ),
            trace=Trace(events=tuple(run_history.trace_events)),
            failures=tuple(run_history.failures),
            candidate_equal=study.problem.space.candidates_equal,
        ),
        state,
    )


def evaluate_attempts_sync(
    query: ProposalBatchQuery[BoundaryT, CandidateT, StudyPayloadT],
    *,
    attempt_evaluator: AttemptBatchEvaluator[
        BoundaryT,
        CandidateT,
        StudyPayloadT,
    ],
    requests: tuple[EvaluationRequest[CandidateT], ...] | None = None,
) -> EvaluationAttemptBatch[CandidateT, StudyPayloadT]:
    """Execute one synchronous request batch into a dense attempt batch.

    Parameters
    ----------
    query : ProposalBatchQuery[BoundaryT, CandidateT, StudyPayloadT]
        Proposal batch and evaluation metadata to execute.
    attempt_evaluator : AttemptBatchEvaluator[BoundaryT, CandidateT, StudyPayloadT]
        Active synchronous evaluator for this execution scope.
    requests : tuple[EvaluationRequest[CandidateT], ...] | None, default=None
        Optional prebuilt request batch aligned with ``query``.

    Returns
    -------
    EvaluationAttemptBatch[CandidateT, StudyPayloadT]
        Dense attempt batch preserving successful and failed request slots.
    """
    resolved_requests = (
        build_evaluation_requests(
            query.proposals,
            proposal_evaluation_specs=query.proposal_evaluation_specs,
        )
        if requests is None
        else requests
    )
    if query.evaluation_budget is not None:
        query.evaluation_budget.consume(len(resolved_requests))

    return attempt_evaluator.evaluate_attempts(
        query.problem,
        resolved_requests,
    )


def _evaluate_step_feedback(
    study: StudyExecutionOwner[
        BoundaryT,
        CandidateT,
        RunMethodStateT,
        StudyPayloadT,
        StudyRecordT,
    ],
    state: RunMethodStateT,
    batch_size: int,
    *,
    execution_model: ExecutionModel,
    attempt_evaluator: StudyEvaluator[
        BoundaryT,
        CandidateT,
        StudyPayloadT,
    ],
    evaluation_budget: EvaluationBudget | None = None,
) -> _StudyStepFeedback[CandidateT, RunMethodStateT, StudyRecordT]:
    """Evaluate and materialize one step before run-method assimilation."""
    if batch_size <= 0:
        msg = "batch_size must be positive"
        raise ValueError(msg)

    if study.run_method.is_exhausted(state):
        msg = "run_method is exhausted"
        raise RuntimeError(msg)

    proposals, next_state = study.run_method.ask(state, batch_size=batch_size)
    if len(proposals) == 0:
        msg = "run_method returned no proposals"
        raise RuntimeError(msg)

    if len(proposals) > batch_size:
        msg = "run_method returned more proposals than requested"
        raise ValueError(msg)

    proposal_kernel_hints = study.run_method.proposal_kernel_hints(
        next_state,
        proposals,
    )
    proposal_evaluation_specs = study.run_method.proposal_evaluation_specs(
        next_state,
        proposals,
    )
    top_level_query = ProposalBatchQuery(
        problem=study.problem,
        proposals=proposals,
        execution_resources=attempt_evaluator.execution_resources(),
        proposal_evaluation_specs=proposal_evaluation_specs,
        proposal_kernel_hints=proposal_kernel_hints,
        evaluation_budget=evaluation_budget,
    )
    top_level_requests = build_evaluation_requests(
        top_level_query.proposals,
        proposal_evaluation_specs=top_level_query.proposal_evaluation_specs,
    )

    def requests_for_query(
        query: ProposalBatchQuery[
            BoundaryT,
            CandidateT,
            StudyPayloadT,
        ],
    ) -> tuple[EvaluationRequest[CandidateT], ...]:
        if query is top_level_query:
            return top_level_requests

        return build_evaluation_requests(
            query.proposals,
            proposal_evaluation_specs=query.proposal_evaluation_specs,
        )

    if execution_model == EXACT_ASYNC_EXECUTION_MODEL:

        def batch_executor(
            query: ProposalBatchQuery[
                BoundaryT,
                CandidateT,
                StudyPayloadT,
            ],
        ) -> EvaluationAttemptBatch[CandidateT, StudyPayloadT]:
            query = _query_with_evaluation_budget(query, evaluation_budget)
            query_requests = requests_for_query(query)
            if query.evaluation_budget is not None:
                query.evaluation_budget.consume(len(query_requests))
            attempts = evaluate_batch_exact_async(
                attempt_evaluator,
                query.problem,
                query_requests,
            )
            validate_aligned_attempts(
                query_requests,
                attempts,
                candidate_equal=study.problem.space.candidates_equal,
            )
            return attempts
    else:
        if not supports_attempt_batches(attempt_evaluator):
            msg = "sync execution models require evaluator.evaluate_attempts"
            raise TypeError(msg)

        def batch_executor(
            query: ProposalBatchQuery[
                BoundaryT,
                CandidateT,
                StudyPayloadT,
            ],
        ) -> EvaluationAttemptBatch[CandidateT, StudyPayloadT]:
            query = _query_with_evaluation_budget(query, evaluation_budget)
            query_requests = requests_for_query(query)
            attempts = evaluate_attempts_sync(
                query,
                attempt_evaluator=attempt_evaluator,
                requests=query_requests,
            )
            validate_aligned_attempts(
                query_requests,
                attempts,
                candidate_equal=study.problem.space.candidates_equal,
            )
            return attempts

    request_local_result = _evaluate_request_local_episode_batch(
        study,
        execution_model=execution_model,
        attempt_evaluator=attempt_evaluator,
        query=top_level_query,
        requests=top_level_requests,
        evaluation_budget=evaluation_budget,
    )
    if request_local_result is None:
        remaining_before = (
            None if evaluation_budget is None else evaluation_budget.remaining
        )
        kernel_attempts = study.kernel.run(top_level_query, batch_executor)
        requests = requests_for_query(top_level_query)
        validate_aligned_attempts(
            requests,
            kernel_attempts,
            candidate_equal=study.problem.space.candidates_equal,
        )
        reported_evaluation_count = kernel_attempts.evaluation_count
        step_evaluation_count = _consume_reported_evaluation_cost(
            evaluation_budget=evaluation_budget,
            remaining_before=remaining_before,
            reported_evaluation_count=reported_evaluation_count,
        )
    else:
        kernel_attempts, step_evaluation_count = request_local_result

    feedback_attempts = materialize_feedback_attempts(
        kernel_attempts,
        study.attempt_materializer,
        candidate_equal=study.problem.space.candidates_equal,
    )
    return _StudyStepFeedback(
        attempts=feedback_attempts,
        post_ask_state=next_state,
        evaluation_count=step_evaluation_count,
    )


def evaluate_step(
    study: StudyExecutionOwner[
        BoundaryT,
        CandidateT,
        RunMethodStateT,
        StudyPayloadT,
        StudyRecordT,
    ],
    state: RunMethodStateT,
    batch_size: int,
    *,
    execution_model: ExecutionModel,
    evaluation_budget: EvaluationBudget | None = None,
) -> StudyStepResult[CandidateT, RunMethodStateT, StudyRecordT]:
    """Run one ask/kernel/evaluate/tell step and return attempts and state.

    Parameters
    ----------
    study : StudyExecutionOwner[BoundaryT, CandidateT, RunMethodStateT, StudyPayloadT, StudyRecordT]
        Study-like owner exposing the problem, run method, evaluator, and
        kernel.
    state : RunMethodStateT
        Run-method state to advance.
    batch_size : int
        Maximum number of proposals to request from the run method.
    execution_model : ExecutionModel
        Execution model controlling whether evaluation is synchronous or
        exact-async.
    evaluation_budget : EvaluationBudget | None, default=None
        Optional hard evaluation-budget ledger shared across kernel subqueries.

    Returns
    -------
    StudyStepResult[CandidateT, RunMethodStateT, StudyRecordT]
        Evaluator attempts and the next run-method state.

    Raises
    ------
    ValueError
        If ``batch_size`` is invalid or the run method returns an invalid
        proposal batch.
    RuntimeError
        If the run method is exhausted or returns no proposals.
    """
    with open_attempt_run_scope(
        study.evaluator,
        study.problem,
    ) as attempt_evaluator:
        step_feedback = _evaluate_step_feedback(
            study,
            state,
            batch_size,
            execution_model=execution_model,
            attempt_evaluator=attempt_evaluator,
            evaluation_budget=evaluation_budget,
        )
    next_state = study.run_method.tell_attempts(
        step_feedback.post_ask_state,
        step_feedback.attempts,
    )
    return StudyStepResult(
        attempts=step_feedback.attempts,
        state=next_state,
        evaluation_count=step_feedback.evaluation_count,
    )


def step(
    study: StudyExecutionOwner[
        BoundaryT,
        CandidateT,
        RunMethodStateT,
        StudyPayloadT,
        StudyRecordT,
    ],
    state: RunMethodStateT,
    batch_size: int = 1,
    *,
    execution_model: ExecutionModel = SYNC_BATCH_EXECUTION_MODEL,
) -> tuple[tuple[StudyRecordT, ...], RunMethodStateT]:
    """Run one ask/evaluate/tell step and return records and next state.

    Parameters
    ----------
    study : StudyExecutionOwner[BoundaryT, CandidateT, RunMethodStateT, StudyPayloadT, StudyRecordT]
        Study-like owner exposing the problem, run method, evaluator, and
        kernel.
    state : RunMethodStateT
        Run-method state to advance.
    batch_size : int, default=1
        Maximum number of proposals to request from the run method.
    execution_model : ExecutionModel, default=SYNC_BATCH_EXECUTION_MODEL
        Execution model controlling evaluation behavior.

    Returns
    -------
    tuple[tuple[StudyRecordT, ...], RunMethodStateT]
        Step records and the next run-method state.

    Raises
    ------
    NotImplementedError
        If ``execution_model`` requests stale-async semantics.
    """
    if execution_model.assimilation_mode is ExecutionAssimilationMode.STALE_INCREMENTAL:
        msg = (
            "stale_async execution model is only supported by "
            "Study.run and Study.optimize"
        )
        raise NotImplementedError(msg)

    validate_execution_request(
        study,
        batch_size=batch_size,
        execution_model=execution_model,
    )
    step_result = evaluate_step(
        study,
        state,
        batch_size,
        execution_model=execution_model,
    )
    records = materialize_success_records(step_result.attempts.successes)
    return records, step_result.state


def run(
    study: StudyExecutionOwner[
        BoundaryT,
        CandidateT,
        RunMethodStateT,
        StudyPayloadT,
        StudyRecordT,
    ],
    max_evaluations: int,
    batch_size: int = 1,
    *,
    execution_model: ExecutionModel = SYNC_BATCH_EXECUTION_MODEL,
    count_evaluation_cost: bool = True,
    initial_state: RunMethodStateT | None = None,
    stop_at_checkpoint_boundary: bool = False,
) -> tuple[RunReport[CandidateT, StudyRecordT], RunMethodStateT]:
    """Run repeated ask/evaluate/tell steps and return one generic run report.

    Parameters
    ----------
    study : StudyExecutionOwner[BoundaryT, CandidateT, RunMethodStateT, StudyPayloadT, StudyRecordT]
        Study-like owner exposing the problem, run method, evaluator, and
        kernel.
    max_evaluations : int
        Evaluation budget to consume. With ``count_evaluation_cost=True``,
        this budget is charged against reported logical ``evaluation_count``
        values, including inner kernel work and recorded evaluation failures.
        With ``count_evaluation_cost=False``, it is charged against returned
        attempt slots instead.
    batch_size : int, default=1
        Maximum number of proposals requested per step.
    execution_model : ExecutionModel, default=SYNC_BATCH_EXECUTION_MODEL
        Execution model controlling evaluation behavior.
    count_evaluation_cost : bool, default=True
        Whether to debit the budget using reported logical evaluation cost
        instead of returned attempt-slot count.
    initial_state : RunMethodStateT | None, default=None
        Optional initial run-method state. ``None`` creates a fresh state.
    stop_at_checkpoint_boundary : bool, default=False
        Whether to return a checkpoint-safe state. If the budget ends inside an
        unsafe run-method segment, the report and state are rolled back to the
        most recent checkpoint-safe boundary reached during this call.

    Returns
    -------
    tuple[RunReport[CandidateT, StudyRecordT], RunMethodStateT]
        Run report and the final run-method state.

    Raises
    ------
    ValueError
        If ``max_evaluations`` is negative or ``execution_model`` requests
        stale-async semantics.
    """
    if max_evaluations < 0:
        msg = "max_evaluations must be non-negative"
        raise ValueError(msg)

    if execution_model == STALE_ASYNC_EXECUTION_MODEL:
        msg = "stale_async execution is handled by the stale_async study tier"
        raise ValueError(msg)

    validate_execution_request(
        study,
        batch_size=batch_size,
        execution_model=execution_model,
    )

    run_history = StudyRunHistory[CandidateT, RunMethodStateT, StudyRecordT]()
    evaluation_budget = (
        EvaluationBudget(max_evaluations) if count_evaluation_cost else None
    )
    record_budget_remaining = max_evaluations
    state = (
        study.run_method.create_initial_state()
        if initial_state is None
        else initial_state
    )
    safe_snapshot: CheckpointSafeRunSnapshot[RunMethodStateT] | None = None
    unsafe_since_safe_snapshot = False
    if stop_at_checkpoint_boundary and study.run_method.is_checkpoint_safe_state(state):
        safe_snapshot = run_history.checkpoint_snapshot(state)

    has_evaluation_work = _current_remaining_budget(
        evaluation_budget=evaluation_budget,
        record_budget_remaining=record_budget_remaining,
    ) > 0 and not study.run_method.is_exhausted(state)
    attempt_scope = (
        open_attempt_run_scope(study.evaluator, study.problem)
        if has_evaluation_work
        else nullcontext(study.evaluator)
    )
    with attempt_scope as attempt_evaluator:
        while _current_remaining_budget(
            evaluation_budget=evaluation_budget,
            record_budget_remaining=record_budget_remaining,
        ) > 0 and not study.run_method.is_exhausted(state):
            remaining = _current_remaining_budget(
                evaluation_budget=evaluation_budget,
                record_budget_remaining=record_budget_remaining,
            )
            current_batch_size = min(batch_size, remaining)
            try:
                step_feedback = _evaluate_step_feedback(
                    study,
                    state,
                    batch_size=current_batch_size,
                    execution_model=execution_model,
                    attempt_evaluator=attempt_evaluator,
                    evaluation_budget=evaluation_budget,
                )
            except EvaluationBudgetExhausted:
                if stop_at_checkpoint_boundary and safe_snapshot is not None:
                    return (
                        run_history.checkpoint_report(
                            safe_snapshot,
                            candidate_equal=study.problem.space.candidates_equal,
                        ),
                        safe_snapshot.state,
                    )
                raise
            except Exception as exception:  # noqa: BLE001 - wrap execution failures
                run_history.raise_run_execution_failed(
                    cause=exception,
                    evaluation_count=_current_evaluation_count(
                        max_evaluations=max_evaluations,
                        evaluation_budget=evaluation_budget,
                        record_budget_remaining=record_budget_remaining,
                    ),
                    state=state,
                    safe_snapshot=safe_snapshot,
                    candidate_equal=study.problem.space.candidates_equal,
                )
            step = StudyAssimilatedStep[
                CandidateT,
                StudyRecordT,
            ].from_attempts(
                step_feedback.attempts,
                evaluation_count=step_feedback.evaluation_count,
            )
            run_history.append_step(step)
            try:
                next_state = study.run_method.tell_attempts(
                    step_feedback.post_ask_state,
                    step_feedback.attempts,
                )
            except Exception as exception:  # noqa: BLE001 - wrap run-method failures
                run_history.raise_run_execution_failed(
                    cause=exception,
                    state=step_feedback.post_ask_state,
                    safe_snapshot=safe_snapshot,
                    candidate_equal=study.problem.space.candidates_equal,
                )
            state = next_state
            if evaluation_budget is None:
                remaining -= step_feedback.attempts.attempt_count
                record_budget_remaining = remaining
            if stop_at_checkpoint_boundary:
                if study.run_method.is_checkpoint_safe_state(state):
                    safe_snapshot = run_history.checkpoint_snapshot(state)
                    if unsafe_since_safe_snapshot:
                        break
                    unsafe_since_safe_snapshot = False
                else:
                    unsafe_since_safe_snapshot = True

    if stop_at_checkpoint_boundary and not study.run_method.is_checkpoint_safe_state(
        state
    ):
        if safe_snapshot is None:
            msg = (
                "run did not reach a checkpoint-safe state within the evaluation budget"
            )
            raise RuntimeError(msg)
        return (
            run_history.checkpoint_report(
                safe_snapshot,
                candidate_equal=study.problem.space.candidates_equal,
            ),
            safe_snapshot.state,
        )

    return (
        run_history.to_report(
            candidate_equal=study.problem.space.candidates_equal,
        ),
        state,
    )


def materialize_scalar_run_result(
    run_report: RunReport[CandidateT, StudyRecordT],
    *,
    candidate_equal: CandidateEquality[CandidateT] | None = None,
) -> RunResult[CandidateT]:
    """Project a generic run report into a scalar terminal result.

    Parameters
    ----------
    run_report : RunReport[CandidateT, StudyRecordT]
        Generic terminal report to project.
    candidate_equal : CandidateEquality[CandidateT] | None, optional
        Explicit candidate equality predicate used to validate refinement
        alignment. When absent, strict scalar Python equality is used.

    Returns
    -------
    RunResult[CandidateT]
        Scalar terminal result that preserves trace, accounting, and aligned
        refinement provenance from ``run_report``.

    Raises
    ------
    TypeError
        If the report does not contain scalar
        :class:`~variopt.artifacts.Observation` record successes.
    """
    successes: list[EvaluationSuccess[CandidateT, ObservationPayload]] = []
    for success in run_report.successes:
        payload = success.payload
        if isinstance(payload, Observation):
            scalar_payload = ObservationPayload(
                value=payload.value,
                score=payload.score,
                elapsed_seconds=payload.elapsed_seconds,
            )
        else:
            msg = (
                "Study.optimize currently requires scalar Observation records; "
                "use Study.run for non-scalar feedback records"
            )
            raise TypeError(msg)
        successes.append(
            EvaluationSuccess(
                request=success.request,
                payload=scalar_payload,
                evaluation_count=success.evaluation_count,
                refinement=success.refinement,
                kernel_diagnostics=success.kernel_diagnostics,
                candidate_equal=candidate_equal,
            ),
        )

    return RunResult[CandidateT].from_successes(
        successes=tuple(successes),
        evaluation_count=run_report.evaluation_count,
        trace=run_report.trace,
        failures=run_report.failures,
        candidate_equal=candidate_equal,
    )


def optimize(
    study: StudyExecutionOwner[
        BoundaryT,
        CandidateT,
        RunMethodStateT,
        StudyPayloadT,
        StudyRecordT,
    ],
    max_evaluations: int,
    batch_size: int = 1,
    *,
    execution_model: ExecutionModel = SYNC_BATCH_EXECUTION_MODEL,
    count_evaluation_cost: bool = True,
    initial_state: RunMethodStateT | None = None,
    stop_at_checkpoint_boundary: bool = False,
) -> tuple[RunResult[CandidateT], RunMethodStateT]:
    """Run repeated ask/evaluate/tell steps until the budget is exhausted.

    Parameters
    ----------
    study : StudyExecutionOwner[BoundaryT, CandidateT, RunMethodStateT, StudyPayloadT, StudyRecordT]
        Study-like owner exposing the problem, run method, evaluator, and
        kernel.
    max_evaluations : int
        Evaluation budget to consume. With ``count_evaluation_cost=True``,
        this budget is charged against reported logical ``evaluation_count``
        values, including inner kernel work and recorded evaluation failures.
        With ``count_evaluation_cost=False``, it is charged against returned
        attempt slots instead.
    batch_size : int, default=1
        Maximum number of proposals requested per step.
    execution_model : ExecutionModel, default=SYNC_BATCH_EXECUTION_MODEL
        Execution model controlling evaluation behavior.
    count_evaluation_cost : bool, default=True
        Whether to debit the budget using reported logical evaluation cost
        instead of returned attempt-slot count.
    initial_state : RunMethodStateT | None, default=None
        Optional initial run-method state. ``None`` creates a fresh state.
    stop_at_checkpoint_boundary : bool, default=False
        Whether to return only checkpoint-safe terminal state boundaries.

    Returns
    -------
    tuple[RunResult[CandidateT], RunMethodStateT]
        Scalar optimization result and the final run-method state.

    Raises
    ------
    TypeError
        If the study does not emit scalar :class:`Observation` records.
    """
    if _supports_direct_scalar_sequential_path(
        study,
        execution_model=execution_model,
    ):
        return _optimize_direct_scalar_sequential(
            study,
            max_evaluations=max_evaluations,
            batch_size=batch_size,
            execution_model=execution_model,
            count_evaluation_cost=count_evaluation_cost,
            initial_state=initial_state,
            stop_at_checkpoint_boundary=stop_at_checkpoint_boundary,
        )

    run_report, state = run(
        study,
        max_evaluations=max_evaluations,
        batch_size=batch_size,
        execution_model=execution_model,
        count_evaluation_cost=count_evaluation_cost,
        initial_state=initial_state,
        stop_at_checkpoint_boundary=stop_at_checkpoint_boundary,
    )

    return (
        materialize_scalar_run_result(
            run_report,
            candidate_equal=study.problem.space.candidates_equal,
        ),
        state,
    )
