"""Search helpers for structured local-search runtime episodes."""

import numpy as np

from .....artifacts import Observation, Proposal, ProposalEvaluationSpec
from .....kernel import KernelStatus
from .....outcomes import EvaluationAttemptBatch
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
    StructuredImprovementScanResult,
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
    space = runtime.neighborhood.space
    space.validate(current_candidate)
    for path, leaf_space in leaf_schedule:
        current_leaf_value = space.leaf_value_at_validated_path(
            current_candidate,
            path,
        )
        leaf_neighbors = discrete_leaf_neighbors(leaf_space, current_leaf_value)
        if max_categorical_neighbors_per_leaf is not None and isinstance(
            leaf_space, CategoricalSpace
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
    reserved_count: int = 0,
) -> StructuredImprovementScanResult[StructuredCandidateT]:
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
    reserved_count : int, default=0
        Evaluation units reserved for later proposals in the same top-level
        batch.

    Returns
    -------
    StructuredImprovementScanResult[StructuredCandidateT]
        Improving outcome, accounting, failure attempts, and budget status for
        the neighborhood scan.
    """
    evaluated_neighbor_count = 0
    failed_attempts: list[EvaluationAttemptBatch[StructuredCandidateT]] = []
    space = runtime.neighborhood.space
    space.validate(candidate)
    for path, leaf_space in leaf_schedule:
        current_leaf_value = space.leaf_value_at_validated_path(
            candidate,
            path,
        )
        for replacement in discrete_leaf_neighbors(leaf_space, current_leaf_value):
            if not runtime.can_evaluate(reserved_count=reserved_count):
                return StructuredImprovementScanResult(
                    improved_outcome=None,
                    evaluation_count=evaluated_neighbor_count,
                    failed_attempts=tuple(failed_attempts),
                    budget_exhausted=True,
                )
            proposed_candidate = space.replace_leaf_values_in_validated_candidate(
                candidate,
                {path: replacement},
            )
            proposed_attempt = runtime.evaluate_candidate_attempt(
                candidate=proposed_candidate,
                proposal_evaluation_spec=proposal_evaluation_spec,
            )
            evaluated_neighbor_count += proposed_attempt.evaluation_count
            proposed_outcome = proposed_attempt.single_outcome_or_none()
            if proposed_outcome is None:
                failed_attempts.append(proposed_attempt)
                continue
            if proposed_outcome.record.score < current_score:
                return StructuredImprovementScanResult(
                    improved_outcome=proposed_outcome,
                    evaluation_count=evaluated_neighbor_count,
                    failed_attempts=tuple(failed_attempts),
                )

    return StructuredImprovementScanResult(
        improved_outcome=None,
        evaluation_count=evaluated_neighbor_count,
        failed_attempts=tuple(failed_attempts),
    )


def first_improving_pair_move_outcome(
    *,
    runtime: PreparedStructuredLocalSearchRuntime[BoundaryT, StructuredCandidateT],
    candidate: StructuredCandidateT,
    current_score: float,
    leaf_schedule: tuple[tuple[LeafPath, DiscreteLeafSpace], ...],
    proposal_evaluation_spec: ProposalEvaluationSpec | None,
    pair_move_leaf_limit: int,
    reserved_count: int = 0,
) -> StructuredImprovementScanResult[StructuredCandidateT]:
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
    reserved_count : int, default=0
        Evaluation units reserved for later proposals in the same top-level
        batch.

    Returns
    -------
    StructuredImprovementScanResult[StructuredCandidateT]
        Improving outcome, accounting, failure attempts, and budget status for
        the pair-move scan.
    """
    limited_schedule = leaf_schedule[:pair_move_leaf_limit]
    if len(limited_schedule) < 2:
        return StructuredImprovementScanResult(
            improved_outcome=None,
            evaluation_count=0,
        )

    evaluated_neighbor_count = 0
    failed_attempts: list[EvaluationAttemptBatch[StructuredCandidateT]] = []
    space = runtime.neighborhood.space
    space.validate(candidate)
    for left_index in range(len(limited_schedule) - 1):
        left_path, left_space = limited_schedule[left_index]
        left_current_value = space.leaf_value_at_validated_path(
            candidate,
            left_path,
        )
        left_neighbors = discrete_leaf_neighbors(left_space, left_current_value)
        if len(left_neighbors) == 0:
            continue

        for right_index in range(left_index + 1, len(limited_schedule)):
            right_path, right_space = limited_schedule[right_index]
            right_current_value = space.leaf_value_at_validated_path(
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
                    if not runtime.can_evaluate(reserved_count=reserved_count):
                        return StructuredImprovementScanResult(
                            improved_outcome=None,
                            evaluation_count=evaluated_neighbor_count,
                            failed_attempts=tuple(failed_attempts),
                            budget_exhausted=True,
                        )
                    proposed_candidate = space.replace_leaf_values_in_validated_candidate(
                        candidate,
                        {
                            left_path: left_replacement,
                            right_path: right_replacement,
                        },
                    )
                    proposed_attempt = runtime.evaluate_candidate_attempt(
                        candidate=proposed_candidate,
                        proposal_evaluation_spec=proposal_evaluation_spec,
                    )
                    evaluated_neighbor_count += proposed_attempt.evaluation_count
                    proposed_outcome = proposed_attempt.single_outcome_or_none()
                    if proposed_outcome is None:
                        failed_attempts.append(proposed_attempt)
                        continue
                    if proposed_outcome.record.score < current_score:
                        return StructuredImprovementScanResult(
                            improved_outcome=proposed_outcome,
                            evaluation_count=evaluated_neighbor_count,
                            failed_attempts=tuple(failed_attempts),
                        )

    return StructuredImprovementScanResult(
        improved_outcome=None,
        evaluation_count=evaluated_neighbor_count,
        failed_attempts=tuple(failed_attempts),
    )


def run_structured_variable_neighborhood_stage_once(
    *,
    stage: StructuredVariableNeighborhoodStage,
    runtime: PreparedStructuredLocalSearchRuntime[BoundaryT, StructuredCandidateT],
    candidate: StructuredCandidateT,
    current_score: float,
    leaf_schedule: tuple[tuple[LeafPath, DiscreteLeafSpace], ...],
    proposal_evaluation_spec: ProposalEvaluationSpec | None,
    random_state: np.random.RandomState,
    reserved_count: int = 0,
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
    reserved_count : int, default=0
        Evaluation units reserved for later proposals in the same top-level
        batch.

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
        scan_result = first_improving_single_leaf_outcome(
            runtime=runtime,
            candidate=candidate,
            current_score=current_score,
            leaf_schedule=leaf_schedule,
            proposal_evaluation_spec=proposal_evaluation_spec,
            reserved_count=reserved_count,
        )
        if scan_result.budget_exhausted:
            return StructuredVariableNeighborhoodStageAttempt(
                improved_outcome=None,
                evaluation_count=scan_result.evaluation_count,
                terminal_status=KernelStatus.STOPPED,
                terminal_message="evaluation budget exhausted before local convergence",
                failed_attempts=scan_result.failed_attempts,
                budget_exhausted=True,
            )
        return StructuredVariableNeighborhoodStageAttempt(
            improved_outcome=scan_result.improved_outcome,
            evaluation_count=scan_result.evaluation_count,
            terminal_status=KernelStatus.CONVERGED,
            terminal_message="no improving move found in the full leafwise neighborhood",
            failed_attempts=scan_result.failed_attempts,
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
        failed_attempts: list[EvaluationAttemptBatch[StructuredCandidateT]] = []
        for move in sampled_neighborhood.moves:
            if not runtime.can_evaluate(reserved_count=reserved_count):
                return StructuredVariableNeighborhoodStageAttempt(
                    improved_outcome=None,
                    evaluation_count=evaluation_count,
                    terminal_status=KernelStatus.STOPPED,
                    terminal_message="evaluation budget exhausted before local convergence",
                    failed_attempts=tuple(failed_attempts),
                    budget_exhausted=True,
                )
            proposed_candidate = (
                runtime.neighborhood.space.replace_leaf_values_in_validated_candidate(
                    candidate,
                    {move.path: move.replacement},
                )
            )
            proposed_attempt = runtime.evaluate_candidate_attempt(
                candidate=proposed_candidate,
                proposal_evaluation_spec=proposal_evaluation_spec,
            )
            evaluation_count += proposed_attempt.evaluation_count
            proposed_outcome = proposed_attempt.single_outcome_or_none()
            if proposed_outcome is None:
                failed_attempts.append(proposed_attempt)
                continue
            if proposed_outcome.record.score < current_score:
                return StructuredVariableNeighborhoodStageAttempt(
                    improved_outcome=proposed_outcome,
                    evaluation_count=evaluation_count,
                    terminal_status=KernelStatus.STOPPED,
                    terminal_message=(
                        "sampled variable-neighborhood stage found an improving move"
                    ),
                    failed_attempts=tuple(failed_attempts),
                )

        if sampled_neighborhood.covers_full_neighborhood:
            return StructuredVariableNeighborhoodStageAttempt(
                improved_outcome=None,
                evaluation_count=evaluation_count,
                terminal_status=KernelStatus.CONVERGED,
                terminal_message=(
                    "no improving move found in the full sampled leafwise neighborhood"
                ),
                failed_attempts=tuple(failed_attempts),
            )

        return StructuredVariableNeighborhoodStageAttempt(
            improved_outcome=None,
            evaluation_count=evaluation_count,
            terminal_status=KernelStatus.STOPPED,
            terminal_message="no improving move found in the sampled variable neighborhood",
            failed_attempts=tuple(failed_attempts),
        )

    if stage.kind == "scheduled_single_then_pair":
        if stage.pair_move_leaf_limit is None:
            msg = "scheduled stage must define pair_move_leaf_limit"
            raise ValueError(msg)

        single_scan_result = first_improving_single_leaf_outcome(
            runtime=runtime,
            candidate=candidate,
            current_score=current_score,
            leaf_schedule=leaf_schedule,
            proposal_evaluation_spec=proposal_evaluation_spec,
            reserved_count=reserved_count,
        )
        if single_scan_result.budget_exhausted:
            return StructuredVariableNeighborhoodStageAttempt(
                improved_outcome=None,
                evaluation_count=single_scan_result.evaluation_count,
                terminal_status=KernelStatus.STOPPED,
                terminal_message="evaluation budget exhausted before local convergence",
                failed_attempts=single_scan_result.failed_attempts,
                budget_exhausted=True,
            )
        if single_scan_result.improved_outcome is not None:
            return StructuredVariableNeighborhoodStageAttempt(
                improved_outcome=single_scan_result.improved_outcome,
                evaluation_count=single_scan_result.evaluation_count,
                terminal_status=KernelStatus.STOPPED,
                terminal_message=(
                    "scheduled variable-neighborhood stage found an improving move"
                ),
                failed_attempts=single_scan_result.failed_attempts,
            )

        pair_scan_result = first_improving_pair_move_outcome(
            runtime=runtime,
            candidate=candidate,
            current_score=current_score,
            leaf_schedule=leaf_schedule,
            proposal_evaluation_spec=proposal_evaluation_spec,
            pair_move_leaf_limit=stage.pair_move_leaf_limit,
            reserved_count=reserved_count,
        )
        combined_failed_attempts = (
            single_scan_result.failed_attempts + pair_scan_result.failed_attempts
        )
        evaluation_count = (
            single_scan_result.evaluation_count + pair_scan_result.evaluation_count
        )
        if pair_scan_result.budget_exhausted:
            return StructuredVariableNeighborhoodStageAttempt(
                improved_outcome=None,
                evaluation_count=evaluation_count,
                terminal_status=KernelStatus.STOPPED,
                terminal_message="evaluation budget exhausted before local convergence",
                failed_attempts=combined_failed_attempts,
                budget_exhausted=True,
            )
        return StructuredVariableNeighborhoodStageAttempt(
            improved_outcome=pair_scan_result.improved_outcome,
            evaluation_count=evaluation_count,
            terminal_status=KernelStatus.CONVERGED,
            terminal_message=(
                "no improving move found in the scheduled single-then-pair neighborhood"
            ),
            failed_attempts=combined_failed_attempts,
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
    reserved_count: int = 0,
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
    reserved_count : int, default=0
        Evaluation units reserved for later proposals in the same top-level
        batch.

    Returns
    -------
    StructuredLocalImprovementResult[StructuredCandidateT]
        Final observation and episode accounting after local improvement.
    """
    original_proposal = proposal if initial_candidate is proposal.candidate else None
    current_attempt = runtime.evaluate_candidate_attempt(
        candidate=initial_candidate,
        proposal=original_proposal,
        proposal_evaluation_spec=proposal_evaluation_spec,
    )
    failed_attempts: list[EvaluationAttemptBatch[StructuredCandidateT]] = []
    current_outcome = current_attempt.single_outcome_or_none()
    if current_outcome is None:
        failed_attempts.append(current_attempt)
        return StructuredLocalImprovementResult(
            record=None,
            evaluation_count=current_attempt.evaluation_count,
            completed_steps=0,
            converged=False,
            failed_attempts=tuple(failed_attempts),
        )

    current_record = current_outcome.record
    current_candidate = current_record.candidate
    current_value = current_record.value
    current_score = current_record.score
    evaluation_count = current_outcome.evaluation_count
    completed_steps = 0
    converged = False
    budget_exhausted = False

    while completed_steps < max_steps:
        scan_result = first_improving_single_leaf_outcome(
            runtime=runtime,
            candidate=current_candidate,
            current_score=current_score,
            leaf_schedule=leaf_schedule,
            proposal_evaluation_spec=proposal_evaluation_spec,
            reserved_count=reserved_count,
        )
        evaluation_count += scan_result.evaluation_count
        failed_attempts.extend(scan_result.failed_attempts)
        proposed_outcome = scan_result.improved_outcome
        budget_exhausted = scan_result.budget_exhausted
        if proposed_outcome is None:
            converged = not budget_exhausted
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
        failed_attempts=tuple(failed_attempts),
        budget_exhausted=budget_exhausted,
    )
