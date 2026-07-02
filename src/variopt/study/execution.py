"""Generic study step and run orchestration."""

from dataclasses import dataclass, replace
from typing import Generic, Protocol, TypeGuard, cast

from typing_extensions import TypeVar

from variopt.generic_runtime import FrozenGenericSlotsCompat

from ..artifacts import (
    CandidateRefinement,
    EvaluationRequest,
    Observation,
    Proposal,
    RunReport,
    RunResult,
    Trace,
    TraceEvent,
)
from ..evaluators.base import Evaluator
from ..evaluators.sequential import SequentialEvaluator
from ..execution import (
    EXACT_ASYNC_EXECUTION_MODEL,
    SEQUENTIAL_EXECUTION_MODEL,
    STALE_ASYNC_EXECUTION_MODEL,
    SYNC_BATCH_EXECUTION_MODEL,
    EvaluationBudget,
    ExecutionAssimilationMode,
    ExecutionModel,
)
from ..kernel import DirectKernel, Kernel, ProposalBatchQuery
from ..methods import RunMethod
from ..objective import Objective, ScalarEvaluationProtocol
from ..outcomes import EvaluationOutcome
from ..problem import Problem
from ..spaces import CandidateEquality
from ..typevars import CandidateT, RunMethodStateT
from .common import (
    StudyEvaluationRecordT,
    build_evaluation_requests,
    trace_value_for_records,
    validate_aligned_outcomes,
)
from .exact_async.orchestration import evaluate_batch_exact_async
from .validation import require_async_evaluator, validate_execution_request

BoundaryT = TypeVar("BoundaryT")


@dataclass(frozen=True, slots=True)
class StudyStepResult(
    FrozenGenericSlotsCompat,
    Generic[CandidateT, RunMethodStateT, StudyEvaluationRecordT],
):
    """Canonical in-process result for one ask/evaluate/tell study step.

    Parameters
    ----------
    outcomes : tuple[EvaluationOutcome[CandidateT, StudyEvaluationRecordT], ...]
        Evaluator outcomes returned for the step.
    state : RunMethodStateT
        Run-method state after assimilating ``outcomes``.
    evaluation_count : int
        Evaluation units consumed by the step after hard-budget reconciliation.
    """

    outcomes: tuple[EvaluationOutcome[CandidateT, StudyEvaluationRecordT], ...]
    state: RunMethodStateT
    evaluation_count: int


@dataclass(frozen=True, slots=True)
class _CheckpointSafeRunSnapshot(
    Generic[CandidateT, RunMethodStateT, StudyEvaluationRecordT]
):
    """Last known checkpoint-safe run projection."""

    records: tuple[StudyEvaluationRecordT, ...]
    refinements: tuple[CandidateRefinement[CandidateT] | None, ...] | None
    trace: Trace
    evaluation_count: int
    state: RunMethodStateT


class StudyExecutionOwner(
    Protocol[BoundaryT, CandidateT, RunMethodStateT, StudyEvaluationRecordT]
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
    ) -> Problem[BoundaryT, CandidateT, StudyEvaluationRecordT]:
        """Return the configured problem."""
        ...

    @property
    def run_method(
        self,
    ) -> RunMethod[
        RunMethodStateT,
        Proposal[CandidateT],
        StudyEvaluationRecordT,
    ]:
        """Return the configured run method."""
        ...

    @property
    def evaluator(
        self,
    ) -> Evaluator[
        Problem[BoundaryT, CandidateT, StudyEvaluationRecordT],
        EvaluationRequest[CandidateT],
        EvaluationOutcome[CandidateT, StudyEvaluationRecordT],
    ]:
        """Return the configured evaluator."""
        ...

    @property
    def kernel(
        self,
    ) -> Kernel[
        ProposalBatchQuery[BoundaryT, CandidateT, StudyEvaluationRecordT],
        tuple[EvaluationOutcome[CandidateT, StudyEvaluationRecordT], ...],
    ]:
        """Return the configured kernel."""
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
        tuple[EvaluationOutcome[CandidateT, Observation[CandidateT]], ...],
    ]:
        """Return the configured direct kernel."""
        ...


def _supports_direct_scalar_sequential_path(
    study: StudyExecutionOwner[
        BoundaryT,
        CandidateT,
        RunMethodStateT,
        StudyEvaluationRecordT,
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

    if type(study.evaluator) is not SequentialEvaluator:
        return False

    if type(study.kernel) is not DirectKernel:
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
    query: ProposalBatchQuery[BoundaryT, CandidateT, StudyEvaluationRecordT],
    evaluation_budget: EvaluationBudget | None,
) -> ProposalBatchQuery[BoundaryT, CandidateT, StudyEvaluationRecordT]:
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


def _current_remaining_budget(
    *,
    evaluation_budget: EvaluationBudget | None,
    record_budget_remaining: int,
) -> int:
    """Return the active loop budget for evaluation- or record-count mode."""
    if evaluation_budget is None:
        return record_budget_remaining
    return evaluation_budget.remaining


def _current_step_batch_size(
    _study: StudyExecutionOwner[
        BoundaryT,
        CandidateT,
        RunMethodStateT,
        StudyEvaluationRecordT,
    ],
    *,
    batch_size: int,
    remaining: int,
    evaluation_budget: EvaluationBudget | None,
) -> int:
    """Return a batch size that preserves hard budget safety for kernels."""
    _ = evaluation_budget
    return min(batch_size, remaining)


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

    observations: list[Observation[CandidateT]] = []
    trace_events: list[TraceEvent] = []
    evaluation_budget = (
        EvaluationBudget(max_evaluations) if count_evaluation_cost else None
    )
    record_budget_remaining = max_evaluations
    state = (
        study.run_method.create_initial_state()
        if initial_state is None
        else initial_state
    )

    while _current_remaining_budget(
        evaluation_budget=evaluation_budget,
        record_budget_remaining=record_budget_remaining,
    ) > 0 and not study.run_method.is_exhausted(state):
        remaining = _current_remaining_budget(
            evaluation_budget=evaluation_budget,
            record_budget_remaining=record_budget_remaining,
        )
        current_batch_size = min(batch_size, remaining)
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

        # DirectKernel ignores hint payloads, but the generic path still
        # validates alignment when it builds ProposalBatchQuery.
        proposal_kernel_hints = study.run_method.proposal_kernel_hints(
            next_state,
            proposals,
        )
        if proposal_kernel_hints is not None and (
            len(proposal_kernel_hints) != len(proposals)
        ):
            msg = "proposal_kernel_hints must align one-to-one with proposals"
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

        batch_observations: list[Observation[CandidateT]] = []
        batch_outcomes: list[
            EvaluationOutcome[CandidateT, Observation[CandidateT]]
        ] = []
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
            if evaluation_budget is not None:
                evaluation_budget.consume()
            observation = Observation.from_objective_value(
                request=request,
                candidate=candidate,
                value=objective.evaluate(candidate),
                direction=study.problem.direction,
            )
            batch_observations.append(observation)
            batch_outcomes.append(
                EvaluationOutcome(
                    record=observation,
                    evaluation_count=1,
                ),
            )

        batch_observation_tuple = tuple(batch_observations)
        batch_outcome_tuple = tuple(batch_outcomes)
        state = study.run_method.tell_outcomes(next_state, batch_outcome_tuple)
        observations.extend(batch_observation_tuple)
        if evaluation_budget is None:
            record_budget_remaining -= len(batch_observation_tuple)

        trace_events.append(
            TraceEvent(
                kind="study.step",
                message=f"evaluated {len(batch_observation_tuple)} proposal(s)",
                value=trace_value_for_records(batch_observation_tuple),
            ),
        )

    return (
        RunResult[CandidateT].from_observations(
            observations=tuple(observations),
            evaluation_count=(
                max_evaluations - record_budget_remaining
                if evaluation_budget is None
                else max_evaluations - evaluation_budget.remaining
            ),
            trace=Trace(events=tuple(trace_events)),
            candidate_equal=study.problem.space.candidates_equal,
        ),
        state,
    )


def evaluate_batch_sync(
    study: StudyExecutionOwner[
        BoundaryT,
        CandidateT,
        RunMethodStateT,
        StudyEvaluationRecordT,
    ],
    query: ProposalBatchQuery[BoundaryT, CandidateT, StudyEvaluationRecordT],
    *,
    requests: tuple[EvaluationRequest[CandidateT], ...] | None = None,
) -> tuple[EvaluationOutcome[CandidateT, StudyEvaluationRecordT], ...]:
    """Execute one request batch through the synchronous evaluator path.

    Parameters
    ----------
    study : StudyExecutionOwner[BoundaryT, CandidateT, RunMethodStateT, StudyEvaluationRecordT]
        Study-like owner exposing the problem, evaluator, and kernel.
    query : ProposalBatchQuery[BoundaryT, CandidateT, StudyEvaluationRecordT]
        Proposal batch and evaluation metadata to execute.
    requests : tuple[EvaluationRequest[CandidateT], ...] | None, default=None
        Optional prebuilt request batch aligned with ``query``.

    Returns
    -------
    tuple[EvaluationOutcome[CandidateT, StudyEvaluationRecordT], ...]
        Outcomes returned by the synchronous evaluator in request order.
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
    return tuple(study.evaluator.evaluate(query.problem, resolved_requests))


def evaluate_step(
    study: StudyExecutionOwner[
        BoundaryT,
        CandidateT,
        RunMethodStateT,
        StudyEvaluationRecordT,
    ],
    state: RunMethodStateT,
    batch_size: int,
    *,
    execution_model: ExecutionModel,
    evaluation_budget: EvaluationBudget | None = None,
) -> StudyStepResult[CandidateT, RunMethodStateT, StudyEvaluationRecordT]:
    """Run one ask/kernel/evaluate/tell step and return outcomes and state.

    Parameters
    ----------
    study : StudyExecutionOwner[BoundaryT, CandidateT, RunMethodStateT, StudyEvaluationRecordT]
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
    StudyStepResult[CandidateT, RunMethodStateT, StudyEvaluationRecordT]
        Evaluator outcomes and the next run-method state.

    Raises
    ------
    ValueError
        If ``batch_size`` is invalid or the run method returns an invalid
        proposal batch.
    RuntimeError
        If the run method is exhausted or returns no proposals.
    """
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
        execution_resources=study.evaluator.execution_resources(),
        proposal_evaluation_specs=proposal_evaluation_specs,
        proposal_kernel_hints=proposal_kernel_hints,
        evaluation_budget=evaluation_budget,
    )
    top_level_requests: tuple[EvaluationRequest[CandidateT], ...] | None = None

    def requests_for_query(
        query: ProposalBatchQuery[
            BoundaryT,
            CandidateT,
            StudyEvaluationRecordT,
        ],
    ) -> tuple[EvaluationRequest[CandidateT], ...]:
        nonlocal top_level_requests
        if query is top_level_query and top_level_requests is not None:
            return top_level_requests

        requests = build_evaluation_requests(
            query.proposals,
            proposal_evaluation_specs=query.proposal_evaluation_specs,
        )
        if query is top_level_query:
            top_level_requests = requests
        return requests

    if execution_model == EXACT_ASYNC_EXECUTION_MODEL:

        def batch_executor(
            query: ProposalBatchQuery[
                BoundaryT,
                CandidateT,
                StudyEvaluationRecordT,
            ],
        ) -> tuple[EvaluationOutcome[CandidateT, StudyEvaluationRecordT], ...]:
            query = _query_with_evaluation_budget(query, evaluation_budget)
            query_requests = requests_for_query(query)
            if query.evaluation_budget is not None:
                query.evaluation_budget.consume(len(query_requests))
            return evaluate_batch_exact_async(
                require_async_evaluator(study),
                query.problem,
                query_requests,
            )
    else:

        def batch_executor(
            query: ProposalBatchQuery[
                BoundaryT,
                CandidateT,
                StudyEvaluationRecordT,
            ],
        ) -> tuple[EvaluationOutcome[CandidateT, StudyEvaluationRecordT], ...]:
            query = _query_with_evaluation_budget(query, evaluation_budget)
            return evaluate_batch_sync(
                study,
                query,
                requests=requests_for_query(query),
            )

    remaining_before = (
        None if evaluation_budget is None else evaluation_budget.remaining
    )
    outcomes = study.kernel.run(top_level_query, batch_executor)
    requests = requests_for_query(top_level_query)
    validate_aligned_outcomes(
        requests,
        outcomes,
        candidate_equal=study.problem.space.candidates_equal,
    )
    reported_evaluation_count = sum(outcome.evaluation_count for outcome in outcomes)
    step_evaluation_count = _consume_reported_evaluation_cost(
        evaluation_budget=evaluation_budget,
        remaining_before=remaining_before,
        reported_evaluation_count=reported_evaluation_count,
    )
    next_state = study.run_method.tell_outcomes(next_state, outcomes)
    return StudyStepResult(
        outcomes=outcomes,
        state=next_state,
        evaluation_count=step_evaluation_count,
    )


def step(
    study: StudyExecutionOwner[
        BoundaryT,
        CandidateT,
        RunMethodStateT,
        StudyEvaluationRecordT,
    ],
    state: RunMethodStateT,
    batch_size: int = 1,
    *,
    execution_model: ExecutionModel = SYNC_BATCH_EXECUTION_MODEL,
) -> tuple[tuple[StudyEvaluationRecordT, ...], RunMethodStateT]:
    """Run one ask/evaluate/tell step and return records and next state.

    Parameters
    ----------
    study : StudyExecutionOwner[BoundaryT, CandidateT, RunMethodStateT, StudyEvaluationRecordT]
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
    tuple[tuple[StudyEvaluationRecordT, ...], RunMethodStateT]
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
    return tuple(outcome.record for outcome in step_result.outcomes), step_result.state


def run(
    study: StudyExecutionOwner[
        BoundaryT,
        CandidateT,
        RunMethodStateT,
        StudyEvaluationRecordT,
    ],
    max_evaluations: int,
    batch_size: int = 1,
    *,
    execution_model: ExecutionModel = SYNC_BATCH_EXECUTION_MODEL,
    count_evaluation_cost: bool = True,
    initial_state: RunMethodStateT | None = None,
    stop_at_checkpoint_boundary: bool = False,
) -> tuple[RunReport[CandidateT, StudyEvaluationRecordT], RunMethodStateT]:
    """Run repeated ask/evaluate/tell steps and return one generic run report.

    Parameters
    ----------
    study : StudyExecutionOwner[BoundaryT, CandidateT, RunMethodStateT, StudyEvaluationRecordT]
        Study-like owner exposing the problem, run method, evaluator, and
        kernel.
    max_evaluations : int
        Evaluation budget to consume.
    batch_size : int, default=1
        Maximum number of proposals requested per step.
    execution_model : ExecutionModel, default=SYNC_BATCH_EXECUTION_MODEL
        Execution model controlling evaluation behavior.
    count_evaluation_cost : bool, default=True
        Whether to debit the budget using evaluator-reported evaluation counts
        instead of completed record count.
    initial_state : RunMethodStateT | None, default=None
        Optional initial run-method state. ``None`` creates a fresh state.
    stop_at_checkpoint_boundary : bool, default=False
        Whether to return a checkpoint-safe state. If the budget ends inside an
        unsafe run-method segment, the report and state are rolled back to the
        most recent checkpoint-safe boundary reached during this call.

    Returns
    -------
    tuple[RunReport[CandidateT, StudyEvaluationRecordT], RunMethodStateT]
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

    records: list[StudyEvaluationRecordT] = []
    refinements: list[CandidateRefinement[CandidateT] | None] | None = None
    trace = Trace()
    evaluation_budget = (
        EvaluationBudget(max_evaluations) if count_evaluation_cost else None
    )
    record_budget_remaining = max_evaluations
    state = (
        study.run_method.create_initial_state()
        if initial_state is None
        else initial_state
    )
    safe_snapshot: (
        _CheckpointSafeRunSnapshot[
            CandidateT,
            RunMethodStateT,
            StudyEvaluationRecordT,
        ]
        | None
    ) = None
    unsafe_since_safe_snapshot = False
    if stop_at_checkpoint_boundary and study.run_method.is_checkpoint_safe_state(state):
        safe_snapshot = _CheckpointSafeRunSnapshot(
            records=(),
            refinements=None,
            trace=trace,
            evaluation_count=0,
            state=state,
        )

    while _current_remaining_budget(
        evaluation_budget=evaluation_budget,
        record_budget_remaining=record_budget_remaining,
    ) > 0 and not study.run_method.is_exhausted(state):
        remaining = _current_remaining_budget(
            evaluation_budget=evaluation_budget,
            record_budget_remaining=record_budget_remaining,
        )
        current_batch_size = _current_step_batch_size(
            study,
            batch_size=batch_size,
            remaining=remaining,
            evaluation_budget=evaluation_budget,
        )
        step_result = evaluate_step(
            study,
            state,
            batch_size=current_batch_size,
            execution_model=execution_model,
            evaluation_budget=evaluation_budget,
        )
        batch_records = tuple(outcome.record for outcome in step_result.outcomes)
        batch_refinements = tuple(
            outcome.refinement for outcome in step_result.outcomes
        )
        state = step_result.state
        records_before_batch = len(records)
        records.extend(batch_records)
        if refinements is not None:
            refinements.extend(batch_refinements)
        elif any(refinement is not None for refinement in batch_refinements):
            refinement_history: list[CandidateRefinement[CandidateT] | None] = [
                None for _index in range(records_before_batch)
            ]
            refinement_history.extend(batch_refinements)
            refinements = refinement_history
        if evaluation_budget is None:
            remaining -= len(batch_records)
            record_budget_remaining = remaining
        trace = trace.append(
            TraceEvent(
                kind="study.step",
                message=f"evaluated {len(batch_records)} proposal(s)",
                value=trace_value_for_records(batch_records),
            ),
        )
        if stop_at_checkpoint_boundary:
            evaluation_count = (
                max_evaluations - record_budget_remaining
                if evaluation_budget is None
                else max_evaluations - evaluation_budget.remaining
            )
            if study.run_method.is_checkpoint_safe_state(state):
                safe_snapshot = _CheckpointSafeRunSnapshot(
                    records=tuple(records),
                    refinements=None if refinements is None else tuple(refinements),
                    trace=trace,
                    evaluation_count=evaluation_count,
                    state=state,
                )
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
            RunReport[CandidateT, StudyEvaluationRecordT].from_records(
                records=safe_snapshot.records,
                evaluation_count=safe_snapshot.evaluation_count,
                trace=safe_snapshot.trace,
                refinements=safe_snapshot.refinements,
                candidate_equal=study.problem.space.candidates_equal,
            ),
            safe_snapshot.state,
        )

    return (
        RunReport[CandidateT, StudyEvaluationRecordT].from_records(
            records=records,
            evaluation_count=(
                max_evaluations - record_budget_remaining
                if evaluation_budget is None
                else max_evaluations - evaluation_budget.remaining
            ),
            trace=trace,
            refinements=None if refinements is None else tuple(refinements),
            candidate_equal=study.problem.space.candidates_equal,
        ),
        state,
    )


def materialize_scalar_run_result(
    run_report: RunReport[CandidateT, StudyEvaluationRecordT],
    *,
    candidate_equal: CandidateEquality[CandidateT] | None = None,
) -> RunResult[CandidateT]:
    """Project a generic run report into a scalar terminal result.

    Parameters
    ----------
    run_report : RunReport[CandidateT, StudyEvaluationRecordT]
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
        If the report does not contain scalar :class:`Observation` records.
    """
    observations: list[Observation[CandidateT]] = []
    for record in run_report.records:
        if not isinstance(record, Observation):
            msg = (
                "Study.optimize currently requires scalar Observation records; "
                "use Study.run for non-scalar evaluation protocols"
            )
            raise TypeError(msg)
        observations.append(cast(Observation[CandidateT], record))

    return RunResult[CandidateT].from_observations(
        observations=tuple(observations),
        evaluation_count=run_report.evaluation_count,
        trace=run_report.trace,
        refinements=run_report.refinements,
        candidate_equal=candidate_equal,
    )


def optimize(
    study: StudyExecutionOwner[
        BoundaryT,
        CandidateT,
        RunMethodStateT,
        StudyEvaluationRecordT,
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
    study : StudyExecutionOwner[BoundaryT, CandidateT, RunMethodStateT, StudyEvaluationRecordT]
        Study-like owner exposing the problem, run method, evaluator, and
        kernel.
    max_evaluations : int
        Evaluation budget to consume.
    batch_size : int, default=1
        Maximum number of proposals requested per step.
    execution_model : ExecutionModel, default=SYNC_BATCH_EXECUTION_MODEL
        Execution model controlling evaluation behavior.
    count_evaluation_cost : bool, default=True
        Whether to debit the budget using evaluator-reported evaluation counts
        instead of completed record count.
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
    if not stop_at_checkpoint_boundary and _supports_direct_scalar_sequential_path(
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
