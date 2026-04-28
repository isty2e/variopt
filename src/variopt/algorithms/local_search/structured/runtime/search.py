"""Search helpers for structured local-search runtime episodes."""

import numpy as np

from .....artifacts import Observation, Proposal, ProposalEvaluationSpec
from .....kernel import KernelStatus
from .....outcomes import EvaluationOutcome
from .....randomness import random_state_choice_indices_without_replacement
from .....spaces import CategoricalSpace, LeafPath
from ..neighborhood import (
    BoundaryT,
    DiscreteLeafSpace,
    SampledStructuredNeighborhood,
    StructuredCandidateT,
    StructuredDiscreteMove,
    StructuredVariableNeighborhoodStage,
    discrete_leaf_neighbors,
)
from .artifacts import (
    StructuredLocalImprovementResult,
    StructuredVariableNeighborhoodStageAttempt,
)
from .prepared import PreparedStructuredLocalSearchRuntime
from .support import sample_neighbors_without_replacement


def sample_structured_discrete_neighborhood(
    *,
    runtime: PreparedStructuredLocalSearchRuntime[BoundaryT, StructuredCandidateT],
    current_candidate: StructuredCandidateT,
    leaf_schedule: tuple[tuple[LeafPath, DiscreteLeafSpace], ...],
    random_state: np.random.RandomState,
    max_neighbors_per_step: int,
    max_categorical_neighbors_per_leaf: int | None,
) -> SampledStructuredNeighborhood:
    """Sample a bounded single-leaf neighborhood around one incumbent.

    Parameters
    ----------
    runtime : PreparedStructuredLocalSearchRuntime[BoundaryT, StructuredCandidateT]
        Prepared runtime that provides space access and candidate evaluation.
    current_candidate : StructuredCandidateT
        Incumbent candidate around which the neighborhood is sampled.
    leaf_schedule : tuple[tuple[LeafPath, DiscreteLeafSpace], ...]
        Ordered editable discrete leaves to consider for neighbor generation.
    random_state : np.random.RandomState
        Random state used for bounded leaf and move sampling.
    max_neighbors_per_step : int
        Maximum number of total moves to keep in the sampled neighborhood.
    max_categorical_neighbors_per_leaf : int | None
        Optional per-leaf bound for categorical replacement enumeration.

    Returns
    -------
    SampledStructuredNeighborhood
        Sampled neighborhood and a flag indicating whether the sampled set
        still covers the full discrete neighborhood.
    """
    moves: list[StructuredDiscreteMove] = []
    covers_full_neighborhood = True
    for path, leaf_space in leaf_schedule:
        current_leaf_value = runtime.neighborhood.space.leaf_value_at_path(
            current_candidate,
            path,
        )
        leaf_neighbors = discrete_leaf_neighbors(leaf_space, current_leaf_value)
        if (
            max_categorical_neighbors_per_leaf is not None
            and isinstance(leaf_space, CategoricalSpace)
        ):
            bounded_leaf_neighbors = sample_neighbors_without_replacement(
                neighbors=leaf_neighbors,
                random_state=random_state,
                max_samples=max_categorical_neighbors_per_leaf,
            )
            if len(bounded_leaf_neighbors) != len(leaf_neighbors):
                covers_full_neighborhood = False
            leaf_neighbors = bounded_leaf_neighbors

        moves.extend(
            StructuredDiscreteMove(path=path, replacement=replacement)
            for replacement in leaf_neighbors
        )

    if len(moves) <= max_neighbors_per_step:
        return SampledStructuredNeighborhood(
            moves=tuple(moves),
            covers_full_neighborhood=covers_full_neighborhood,
        )

    selected_indices = random_state_choice_indices_without_replacement(
        random_state,
        population_size=len(moves),
        count=max_neighbors_per_step,
    )
    sampled_moves = tuple(moves[index] for index in sorted(selected_indices))
    return SampledStructuredNeighborhood(
        moves=sampled_moves,
        covers_full_neighborhood=False,
    )


def first_improving_single_leaf_outcome(
    *,
    runtime: PreparedStructuredLocalSearchRuntime[BoundaryT, StructuredCandidateT],
    candidate: StructuredCandidateT,
    current_score: float,
    leaf_schedule: tuple[tuple[LeafPath, DiscreteLeafSpace], ...],
    proposal_evaluation_spec: ProposalEvaluationSpec | None,
) -> tuple[EvaluationOutcome[StructuredCandidateT] | None, int]:
    """Return the first improving single-leaf move, if any.

    Parameters
    ----------
    runtime : PreparedStructuredLocalSearchRuntime[BoundaryT, StructuredCandidateT]
        Prepared runtime that evaluates candidate perturbations.
    candidate : StructuredCandidateT
        Incumbent candidate to perturb.
    current_score : float
        Score of the incumbent candidate.
    leaf_schedule : tuple[tuple[LeafPath, DiscreteLeafSpace], ...]
        Ordered editable leaves to scan for first improvement.
    proposal_evaluation_spec : ProposalEvaluationSpec | None
        Optional proposal metadata forwarded to evaluation.

    Returns
    -------
    tuple[EvaluationOutcome[StructuredCandidateT] | None, int]
        Improving outcome when found and the total number of evaluations
        consumed while scanning the neighborhood.
    """
    evaluated_neighbor_count = 0
    for path, leaf_space in leaf_schedule:
        current_leaf_value = runtime.neighborhood.space.leaf_value_at_path(
            candidate,
            path,
        )
        for replacement in discrete_leaf_neighbors(leaf_space, current_leaf_value):
            proposed_candidate = runtime.neighborhood.space.replace_leaf_values(
                candidate,
                {path: replacement},
            )
            proposed_outcome = runtime.evaluate_candidate(
                candidate=proposed_candidate,
                proposal_evaluation_spec=proposal_evaluation_spec,
            )
            evaluated_neighbor_count += proposed_outcome.evaluation_count
            if proposed_outcome.record.score < current_score:
                return proposed_outcome, evaluated_neighbor_count

    return None, evaluated_neighbor_count


def first_improving_pair_move_outcome(
    *,
    runtime: PreparedStructuredLocalSearchRuntime[BoundaryT, StructuredCandidateT],
    candidate: StructuredCandidateT,
    current_score: float,
    leaf_schedule: tuple[tuple[LeafPath, DiscreteLeafSpace], ...],
    proposal_evaluation_spec: ProposalEvaluationSpec | None,
    pair_move_leaf_limit: int,
) -> tuple[EvaluationOutcome[StructuredCandidateT] | None, int]:
    """Return the first improving two-leaf move, if any.

    Parameters
    ----------
    runtime : PreparedStructuredLocalSearchRuntime[BoundaryT, StructuredCandidateT]
        Prepared runtime that evaluates candidate perturbations.
    candidate : StructuredCandidateT
        Incumbent candidate to perturb.
    current_score : float
        Score of the incumbent candidate.
    leaf_schedule : tuple[tuple[LeafPath, DiscreteLeafSpace], ...]
        Ordered editable leaves that define the pairwise search frontier.
    proposal_evaluation_spec : ProposalEvaluationSpec | None
        Optional proposal metadata forwarded to evaluation.
    pair_move_leaf_limit : int
        Maximum prefix of ``leaf_schedule`` to consider for pair moves.

    Returns
    -------
    tuple[EvaluationOutcome[StructuredCandidateT] | None, int]
        Improving outcome when found and the total number of evaluations
        consumed while scanning pair moves.
    """
    limited_schedule = leaf_schedule[:pair_move_leaf_limit]
    if len(limited_schedule) < 2:
        return None, 0

    evaluated_neighbor_count = 0
    for left_index, (left_path, left_space) in enumerate(limited_schedule[:-1]):
        left_current_value = runtime.neighborhood.space.leaf_value_at_path(
            candidate,
            left_path,
        )
        left_neighbors = discrete_leaf_neighbors(left_space, left_current_value)
        if len(left_neighbors) == 0:
            continue

        for right_path, right_space in limited_schedule[left_index + 1 :]:
            right_current_value = runtime.neighborhood.space.leaf_value_at_path(
                candidate,
                right_path,
            )
            right_neighbors = discrete_leaf_neighbors(
                right_space,
                right_current_value,
            )
            if len(right_neighbors) == 0:
                continue

            for left_replacement in left_neighbors:
                for right_replacement in right_neighbors:
                    proposed_candidate = runtime.neighborhood.space.replace_leaf_values(
                        candidate,
                        {
                            left_path: left_replacement,
                            right_path: right_replacement,
                        },
                    )
                    proposed_outcome = runtime.evaluate_candidate(
                        candidate=proposed_candidate,
                        proposal_evaluation_spec=proposal_evaluation_spec,
                    )
                    evaluated_neighbor_count += proposed_outcome.evaluation_count
                    if proposed_outcome.record.score < current_score:
                        return proposed_outcome, evaluated_neighbor_count

    return None, evaluated_neighbor_count


def run_structured_variable_neighborhood_stage_once(
    *,
    stage: StructuredVariableNeighborhoodStage,
    runtime: PreparedStructuredLocalSearchRuntime[BoundaryT, StructuredCandidateT],
    candidate: StructuredCandidateT,
    current_score: float,
    leaf_schedule: tuple[tuple[LeafPath, DiscreteLeafSpace], ...],
    proposal_evaluation_spec: ProposalEvaluationSpec | None,
    random_state: np.random.RandomState,
) -> StructuredVariableNeighborhoodStageAttempt[StructuredCandidateT]:
    """Execute one configured variable-neighborhood stage.

    Parameters
    ----------
    stage : StructuredVariableNeighborhoodStage
        Stage configuration describing the neighborhood strategy to run.
    runtime : PreparedStructuredLocalSearchRuntime[BoundaryT, StructuredCandidateT]
        Prepared runtime that evaluates generated neighbors.
    candidate : StructuredCandidateT
        Current incumbent candidate.
    current_score : float
        Score of the current incumbent.
    leaf_schedule : tuple[tuple[LeafPath, DiscreteLeafSpace], ...]
        Ordered editable leaves exposed to the stage.
    proposal_evaluation_spec : ProposalEvaluationSpec | None
        Optional proposal metadata forwarded to evaluation.
    random_state : np.random.RandomState
        Random state used by sampled neighborhood stages.

    Returns
    -------
    StructuredVariableNeighborhoodStageAttempt[StructuredCandidateT]
        Attempt artifact describing any improvement, consumed evaluations, and
        the terminal stage status.

    Raises
    ------
    ValueError
        If the stage configuration is incomplete or unsupported.
    """
    if stage.kind == "leafwise_first_improvement":
        proposed_outcome, evaluation_count = first_improving_single_leaf_outcome(
            runtime=runtime,
            candidate=candidate,
            current_score=current_score,
            leaf_schedule=leaf_schedule,
            proposal_evaluation_spec=proposal_evaluation_spec,
        )
        return StructuredVariableNeighborhoodStageAttempt(
            improved_outcome=proposed_outcome,
            evaluation_count=evaluation_count,
            terminal_status=KernelStatus.CONVERGED,
            terminal_message="no improving move found in the full leafwise neighborhood",
        )

    if stage.kind == "sampled_leafwise_first_improvement":
        if stage.max_neighbors_per_step is None:
            msg = "sampled stage must define max_neighbors_per_step"
            raise ValueError(msg)

        sampled_neighborhood = sample_structured_discrete_neighborhood(
            runtime=runtime,
            current_candidate=candidate,
            leaf_schedule=leaf_schedule,
            random_state=random_state,
            max_neighbors_per_step=stage.max_neighbors_per_step,
            max_categorical_neighbors_per_leaf=(
                stage.max_categorical_neighbors_per_leaf
            ),
        )
        evaluation_count = 0
        for move in sampled_neighborhood.moves:
            proposed_candidate = runtime.neighborhood.space.replace_leaf_values(
                candidate,
                {move.path: move.replacement},
            )
            proposed_outcome = runtime.evaluate_candidate(
                candidate=proposed_candidate,
                proposal_evaluation_spec=proposal_evaluation_spec,
            )
            evaluation_count += proposed_outcome.evaluation_count
            if proposed_outcome.record.score < current_score:
                return StructuredVariableNeighborhoodStageAttempt(
                    improved_outcome=proposed_outcome,
                    evaluation_count=evaluation_count,
                    terminal_status=KernelStatus.STOPPED,
                    terminal_message=(
                        "sampled variable-neighborhood stage found an improving move"
                    ),
                )

        if sampled_neighborhood.covers_full_neighborhood:
            return StructuredVariableNeighborhoodStageAttempt(
                improved_outcome=None,
                evaluation_count=evaluation_count,
                terminal_status=KernelStatus.CONVERGED,
                terminal_message=(
                    "no improving move found in the full sampled leafwise neighborhood"
                ),
            )

        return StructuredVariableNeighborhoodStageAttempt(
            improved_outcome=None,
            evaluation_count=evaluation_count,
            terminal_status=KernelStatus.STOPPED,
            terminal_message="no improving move found in the sampled variable neighborhood",
        )

    if stage.kind == "scheduled_single_then_pair":
        if stage.pair_move_leaf_limit is None:
            msg = "scheduled stage must define pair_move_leaf_limit"
            raise ValueError(msg)

        proposed_outcome, evaluation_count = first_improving_single_leaf_outcome(
            runtime=runtime,
            candidate=candidate,
            current_score=current_score,
            leaf_schedule=leaf_schedule,
            proposal_evaluation_spec=proposal_evaluation_spec,
        )
        if proposed_outcome is not None:
            return StructuredVariableNeighborhoodStageAttempt(
                improved_outcome=proposed_outcome,
                evaluation_count=evaluation_count,
                terminal_status=KernelStatus.STOPPED,
                terminal_message=(
                    "scheduled variable-neighborhood stage found an improving move"
                ),
            )

        pair_outcome, pair_evaluation_count = first_improving_pair_move_outcome(
            runtime=runtime,
            candidate=candidate,
            current_score=current_score,
            leaf_schedule=leaf_schedule,
            proposal_evaluation_spec=proposal_evaluation_spec,
            pair_move_leaf_limit=stage.pair_move_leaf_limit,
        )
        return StructuredVariableNeighborhoodStageAttempt(
            improved_outcome=pair_outcome,
            evaluation_count=evaluation_count + pair_evaluation_count,
            terminal_status=KernelStatus.CONVERGED,
            terminal_message=(
                "no improving move found in the scheduled single-then-pair neighborhood"
            ),
        )

    msg = f"unsupported structured variable-neighborhood stage: {stage.kind!r}"
    raise ValueError(msg)


def run_leafwise_local_search_episode(
    *,
    runtime: PreparedStructuredLocalSearchRuntime[BoundaryT, StructuredCandidateT],
    initial_candidate: StructuredCandidateT,
    proposal: Proposal[StructuredCandidateT],
    proposal_evaluation_spec: ProposalEvaluationSpec | None,
    leaf_schedule: tuple[tuple[LeafPath, DiscreteLeafSpace], ...],
    max_steps: int,
) -> StructuredLocalImprovementResult[StructuredCandidateT]:
    """Run one deterministic first-improvement local-search episode.

    Parameters
    ----------
    runtime : PreparedStructuredLocalSearchRuntime[BoundaryT, StructuredCandidateT]
        Prepared runtime that evaluates candidate perturbations.
    initial_candidate : StructuredCandidateT
        Starting incumbent for the episode.
    proposal : Proposal[StructuredCandidateT]
        Proposal that owns the local-search episode.
    proposal_evaluation_spec : ProposalEvaluationSpec | None
        Optional proposal metadata forwarded to evaluation.
    leaf_schedule : tuple[tuple[LeafPath, DiscreteLeafSpace], ...]
        Ordered editable leaves scanned at each local-search step.
    max_steps : int
        Maximum number of successful improvement steps to execute.

    Returns
    -------
    StructuredLocalImprovementResult[StructuredCandidateT]
        Final observation and episode accounting after local improvement.
    """
    current_outcome = runtime.evaluate_candidate(
        candidate=initial_candidate,
        proposal_evaluation_spec=proposal_evaluation_spec,
    )
    current_record = current_outcome.record
    current_candidate = current_record.candidate
    current_value = current_record.value
    current_score = current_record.score
    evaluation_count = current_outcome.evaluation_count
    completed_steps = 0
    converged = False

    while completed_steps < max_steps:
        proposed_outcome, neighbor_evaluation_count = first_improving_single_leaf_outcome(
            runtime=runtime,
            candidate=current_candidate,
            current_score=current_score,
            leaf_schedule=leaf_schedule,
            proposal_evaluation_spec=proposal_evaluation_spec,
        )
        evaluation_count += neighbor_evaluation_count
        if proposed_outcome is None:
            converged = True
            break

        proposed_record = proposed_outcome.record
        current_candidate = proposed_record.candidate
        current_value = proposed_record.value
        current_score = proposed_record.score
        completed_steps += 1

    return StructuredLocalImprovementResult(
        record=Observation.from_objective_value(
            proposal=proposal,
            proposal_evaluation_spec=proposal_evaluation_spec,
            candidate=current_candidate,
            value=current_value,
            direction=runtime.query.problem.direction,
        ),
        evaluation_count=evaluation_count,
        completed_steps=completed_steps,
        converged=converged,
    )
