"""Request-local episode support for structured local-search kernels."""

from collections.abc import Callable

from typing_extensions import assert_never

from .....artifacts import (
    EvaluationAttemptBatch,
    EvaluationRequest,
    ObservationPayload,
)
from .....execution import EvaluationBudget
from .....kernel import (
    ProposalBatchQuery,
    ProposalKernelHint,
    ProposalLocalSearchContext,
    RequestLocalEpisode,
    RequestLocalEvaluationRunner,
)
from .....problem import Problem
from .....spaces import CategoricalSpace, IntegerSpace, LeafPath
from ..neighborhood import (
    BoundaryT,
    DiscreteLeafSpace,
    StructuredCandidateT,
    StructuredDiscreteNeighborhood,
    StructuredVariableNeighborhoodStage,
)
from .prepared import (
    PreparedStructuredLocalSearchRuntime,
    prepare_structured_local_search_runtime,
)


def structured_local_search_context(
    hint: ProposalKernelHint | None,
) -> ProposalLocalSearchContext | None:
    """Return a validated structured local-search context."""
    if hint is None:
        return None
    if not isinstance(hint, ProposalLocalSearchContext):
        msg = "structured local-search kernels require ProposalLocalSearchContext hints"
        raise TypeError(msg)
    return hint


def structured_episode_max_steps(
    *,
    default_max_steps: int,
    context: ProposalLocalSearchContext | None,
) -> int:
    """Return the effective algorithmic step cap for one proposal."""
    if context is None or context.local_budget is None:
        return default_max_steps
    return context.local_budget


def structured_episode_is_disabled(
    context: ProposalLocalSearchContext | None,
) -> bool:
    """Return whether a proposal explicitly disables local search."""
    return context is not None and not context.enabled


def structured_episode_has_random_state(
    context: ProposalLocalSearchContext | None,
) -> bool:
    """Return whether a proposal supplies an authoritative RNG snapshot."""
    return context is not None and context.random_state_snapshot is not None


def structured_leaf_schedule(
    *,
    problem: Problem[BoundaryT, StructuredCandidateT, ObservationPayload],
    request: EvaluationRequest[StructuredCandidateT],
    context: ProposalLocalSearchContext | None,
) -> tuple[tuple[LeafPath, DiscreteLeafSpace], ...]:
    """Build the effective leaf schedule for one request."""
    neighborhood = StructuredDiscreteNeighborhood[
        BoundaryT,
        StructuredCandidateT,
    ].from_space(problem.space)
    problem.space.validate(request.candidate)
    default_schedule = tuple(
        zip(
            neighborhood.leaf_paths,
            neighborhood.leaf_spaces,
            strict=True,
        )
    )
    if context is None or len(context.prioritized_leaf_paths) == 0:
        return default_schedule

    leaf_space_by_path = dict(default_schedule)
    prioritized_schedule: list[tuple[LeafPath, DiscreteLeafSpace]] = []
    seen_paths: set[LeafPath] = set()
    for path in context.prioritized_leaf_paths:
        leaf_space = leaf_space_by_path.get(path)
        if leaf_space is None:
            msg = (
                "proposal local-search context referenced a leaf path "
                "outside the structured neighborhood"
            )
            raise ValueError(msg)
        prioritized_schedule.append((path, leaf_space))
        seen_paths.add(path)

    prioritized_schedule.extend(
        (path, leaf_space)
        for path, leaf_space in default_schedule
        if path not in seen_paths
    )
    return tuple(prioritized_schedule)


def structured_leaf_neighbor_limits(
    leaf_schedule: tuple[tuple[LeafPath, DiscreteLeafSpace], ...],
) -> tuple[int, ...]:
    """Return candidate-independent maximum neighbor counts per leaf."""
    limits: list[int] = []
    for _, leaf_space in leaf_schedule:
        if isinstance(leaf_space, IntegerSpace):
            limits.append(min(2, leaf_space.high - leaf_space.low))
            continue
        if isinstance(leaf_space, CategoricalSpace):
            limits.append(len(leaf_space.choices) - 1)
            continue
        msg = "structured local-search leaf schedule contains a non-discrete space"
        raise TypeError(msg)
    return tuple(limits)


def structured_single_scan_limit(
    leaf_schedule: tuple[tuple[LeafPath, DiscreteLeafSpace], ...],
) -> int:
    """Return the maximum objective calls in one full single-leaf scan."""
    return sum(structured_leaf_neighbor_limits(leaf_schedule))


def structured_pair_scan_limit(
    leaf_schedule: tuple[tuple[LeafPath, DiscreteLeafSpace], ...],
    *,
    pair_move_leaf_limit: int,
) -> int:
    """Return the maximum objective calls in one bounded pair-move scan."""
    neighbor_limits = structured_leaf_neighbor_limits(
        leaf_schedule[:pair_move_leaf_limit]
    )
    return sum(
        left_limit * right_limit
        for left_index, left_limit in enumerate(neighbor_limits[:-1])
        for right_limit in neighbor_limits[left_index + 1 :]
    )


def structured_stage_evaluation_limit(
    stage: StructuredVariableNeighborhoodStage,
    leaf_schedule: tuple[tuple[LeafPath, DiscreteLeafSpace], ...],
) -> int:
    """Return a safe objective-call cap for one variable-neighborhood stage."""
    if stage.kind == "leafwise_first_improvement":
        return structured_single_scan_limit(leaf_schedule)
    if stage.kind == "sampled_leafwise_first_improvement":
        if stage.max_neighbors_per_step is None:
            msg = "sampled stage must define max_neighbors_per_step"
            raise ValueError(msg)
        return min(
            structured_single_scan_limit(leaf_schedule),
            stage.max_neighbors_per_step,
        )
    if stage.kind == "scheduled_single_then_pair":
        if stage.pair_move_leaf_limit is None:
            msg = "scheduled stage must define pair_move_leaf_limit"
            raise ValueError(msg)
        return structured_single_scan_limit(leaf_schedule) + structured_pair_scan_limit(
            leaf_schedule,
            pair_move_leaf_limit=stage.pair_move_leaf_limit,
        )
    assert_never(stage.kind)


def prepare_structured_request_local_runtime(
    *,
    problem: Problem[BoundaryT, StructuredCandidateT, ObservationPayload],
    episode: RequestLocalEpisode[
        BoundaryT,
        StructuredCandidateT,
        ObservationPayload,
    ],
    runner: RequestLocalEvaluationRunner[
        StructuredCandidateT,
        ObservationPayload,
    ],
) -> PreparedStructuredLocalSearchRuntime[BoundaryT, StructuredCandidateT]:
    """Prepare the existing structured runtime over a worker-local runner."""
    local_budget = EvaluationBudget(episode.evaluation_limit)
    hint = episode.proposal_kernel_hint
    query = ProposalBatchQuery(
        problem=problem,
        proposals=(episode.request.proposal,),
        execution_resources=episode.execution_resources,
        proposal_evaluation_specs=(
            None
            if episode.request.proposal_evaluation_spec is None
            else (episode.request.proposal_evaluation_spec,)
        ),
        proposal_kernel_hints=None if hint is None else (hint,),
        evaluation_budget=local_budget,
    )

    def evaluate_query(
        local_query: ProposalBatchQuery[
            BoundaryT,
            StructuredCandidateT,
            ObservationPayload,
        ],
    ) -> EvaluationAttemptBatch[StructuredCandidateT, ObservationPayload]:
        if len(local_query.proposals) != 1:
            msg = "request-local structured runners require one proposal"
            raise ValueError(msg)
        proposal_evaluation_spec = (
            None
            if local_query.proposal_evaluation_specs is None
            else local_query.proposal_evaluation_specs[0]
        )
        local_budget.consume(1)
        return runner.evaluate(
            EvaluationRequest(
                proposal=local_query.proposals[0],
                proposal_evaluation_spec=proposal_evaluation_spec,
            ),
        )

    runner_callable: Callable[
        [
            ProposalBatchQuery[
                BoundaryT,
                StructuredCandidateT,
                ObservationPayload,
            ]
        ],
        EvaluationAttemptBatch[StructuredCandidateT, ObservationPayload],
    ] = evaluate_query
    return prepare_structured_local_search_runtime(
        query=query,
        runner=runner_callable,
    )
