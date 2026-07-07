"""CSA bank-growth transition logic."""

from collections.abc import Sequence

from ......artifacts import Observation
from ......distance import require_valid_distance
from ......diversity import DiversityMetric
from ......typevars import CandidateT
from ...scoring.model_state import CSAScoreModelState, ScoredBank, ScoredTrial
from ..bank import Bank, BankEntry
from ..queries import BankDistanceWorkspace
from .state import CSABankGrowthState


def energy_cut_for_state(
    *,
    state: CSABankGrowthState[CandidateT],
    minimum_real_score: float,
) -> float:
    """Return the active real-score threshold used by legacy growth/removal.

    Parameters
    ----------
    state : CSABankGrowthState[CandidateT]
        Bank-growth state that carries the active energy-gap limit.
    minimum_real_score : float
        Minimum real score currently present in the bank.

    Returns
    -------
    float
        Active real-score threshold used by growth and removal logic.
    """
    return minimum_real_score + state.active_energy_gap_limit


def try_append_growth_entry(
    *,
    state: CSABankGrowthState[CandidateT],
    bank: Bank[CandidateT],
    observation: Observation[CandidateT],
    scored_bank: ScoredBank[CandidateT],
    trial: ScoredTrial[CandidateT],
    nearest_distance: float,
    active_distance_cutoff: float,
    adaptive_potential_active: bool,
) -> tuple[Bank[CandidateT], CSABankGrowthState[CandidateT], bool]:
    """Return a bank grown by one entry when legacy increase-bank conditions hold.

    Parameters
    ----------
    state : CSABankGrowthState[CandidateT]
        Current bank-growth state.
    bank : Bank[CandidateT]
        Current bank snapshot.
    observation : Observation[CandidateT]
        Observation under consideration for bank growth.
    scored_bank : ScoredBank[CandidateT]
        Current scored bank view.
    trial : ScoredTrial[CandidateT]
        Scored view of ``observation``.
    nearest_distance : float
        Distance from ``observation`` to the nearest bank entry.
    active_distance_cutoff : float
        Active CSA distance cutoff.
    adaptive_potential_active : bool
        Whether adaptive-potential logic allows bypassing some dominance checks.

    Returns
    -------
    tuple[Bank[CandidateT], CSABankGrowthState[CandidateT], bool]
        Possibly grown bank, next growth state, and a flag indicating whether
        growth happened.

    Raises
    ------
    ValueError
        If ``nearest_distance`` or ``active_distance_cutoff`` is negative.
    """
    maximum_capacity = state.policy.maximum_capacity
    if (
        not state.enabled
        or maximum_capacity is None
        or len(bank.entries) >= maximum_capacity
        or state.generation_growth_count >= state.policy.maximum_growth_per_generation
    ):
        return bank, state, False

    if nearest_distance < 0.0:
        msg = "nearest_distance must be non-negative"
        raise ValueError(msg)

    if active_distance_cutoff < 0.0:
        msg = "active_distance_cutoff must be non-negative"
        raise ValueError(msg)

    if (
        state.policy.require_distance_cutoff
        and nearest_distance < active_distance_cutoff
    ):
        return bank, state, False

    if trial.shaped_score > energy_cut_for_state(
        state=state,
        minimum_real_score=min(scored_bank.real_scores),
    ):
        return bank, state, False

    if max(scored_bank.real_scores) < trial.real_score and not adaptive_potential_active:
        return bank, state, False

    grown_bank = Bank(
        capacity=len(bank.entries) + 1,
        entries=bank.entries
        + (
            BankEntry(
                candidate=observation.candidate,
                value=observation.score,
                proposal_id=observation.proposal.proposal_id,
            ),
        ),
    )
    return grown_bank, CSABankGrowthState[CandidateT](
        policy=state.policy,
        active_energy_gap_limit=state.active_energy_gap_limit,
        generation_growth_count=state.generation_growth_count + 1,
    ), True


def should_attempt_remove_top(
    *,
    state: CSABankGrowthState[CandidateT],
    bank: Bank[CandidateT],
    scored_bank: ScoredBank[CandidateT],
    trial: ScoredTrial[CandidateT],
    nearest_distance: float,
    active_distance_cutoff: float,
    minimum_capacity: int,
    adaptive_potential_active: bool,
) -> bool:
    """Return whether one trial should enter legacy remove-top comparison.

    Parameters
    ----------
    state : CSABankGrowthState[CandidateT]
        Current bank-growth state.
    bank : Bank[CandidateT]
        Current bank snapshot.
    scored_bank : ScoredBank[CandidateT]
        Current scored bank view.
    trial : ScoredTrial[CandidateT]
        Scored view of the candidate under consideration.
    nearest_distance : float
        Distance from the candidate to the nearest bank entry.
    active_distance_cutoff : float
        Active CSA distance cutoff.
    minimum_capacity : int
        Minimum bank size below which remove-top should not shrink the bank.
    adaptive_potential_active : bool
        Whether adaptive-potential logic allows bypassing some dominance checks.

    Returns
    -------
    bool
        ``True`` when the candidate should enter the remove-top path.
    """
    if nearest_distance >= active_distance_cutoff:
        return (
            max(scored_bank.real_scores) >= trial.real_score
            or adaptive_potential_active
        )

    if not state.enabled or len(bank.entries) <= minimum_capacity:
        return False

    return trial.shaped_score <= energy_cut_for_state(
        state=state,
        minimum_real_score=min(scored_bank.real_scores),
    )


def reduce_bank_by_energy_cut(
    *,
    state: CSABankGrowthState[CandidateT],
    bank: Bank[CandidateT],
    minimum_capacity: int,
    score_model_state: CSAScoreModelState[CandidateT],
    diversity_metric: DiversityMetric[CandidateT],
    distance_cutoff: float,
    minimum_distance_cutoff: float | None,
    distance_workspace: BankDistanceWorkspace[CandidateT] | None = None,
) -> tuple[
    Bank[CandidateT],
    frozenset[int],
    CSAScoreModelState[CandidateT],
]:
    """Return a bank shrunk by legacy energy-cut reduction rules.

    Parameters
    ----------
    state : CSABankGrowthState[CandidateT]
        Current bank-growth state.
    bank : Bank[CandidateT]
        Current bank snapshot.
    minimum_capacity : int
        Minimum bank size to preserve.
    score_model_state : CSAScoreModelState[CandidateT]
        Score-model runtime state used to rescore the bank.
    diversity_metric : DiversityMetric[CandidateT]
        Diversity metric used while rescoring the bank.
    distance_cutoff : float
        Active CSA distance cutoff.
    minimum_distance_cutoff : float | None
        Optional cutoff floor used by the score model.
    distance_workspace : BankDistanceWorkspace[CandidateT] | None, default=None
        Optional operation-local distance workspace aligned to ``bank.entries``.

    Returns
    -------
    tuple[Bank[CandidateT], frozenset[int], CSAScoreModelState[CandidateT]]
        Reduced bank, removed index set, and updated score-model state.
    """
    if not state.enabled or len(bank.entries) <= minimum_capacity:
        return bank, frozenset(), score_model_state

    scored_bank, score_model_state = score_model_state.score_bank(
        entries=bank.entries,
        diversity_metric=diversity_metric,
        distance_cutoff=distance_cutoff,
        minimum_distance_cutoff=minimum_distance_cutoff,
        masked_entry_indices=frozenset(),
        distance_workspace=distance_workspace,
    )
    if state.policy.energy_gap_update_mode == "max_score_ratio":
        cut_energy = max(scored_bank.real_scores)
    else:
        cut_energy = min(scored_bank.shaped_scores) + state.active_energy_gap_limit

    sorted_indices = sorted(
        range(len(scored_bank.shaped_scores)),
        key=scored_bank.shaped_scores.__getitem__,
    )
    removable_indices = tuple(
        index
        for index in reversed(sorted_indices[minimum_capacity:])
        if scored_bank.shaped_scores[index] > cut_energy
    )
    if not removable_indices:
        return bank, frozenset(), score_model_state

    removed_index_set = frozenset(removable_indices)
    reduced_entries = tuple(
        entry
        for index, entry in enumerate(bank.entries)
        if index not in removed_index_set
    )
    return (
        Bank(
            capacity=len(reduced_entries),
            entries=reduced_entries,
        ),
        removed_index_set,
        score_model_state,
    )


def advance_growth_state(
    *,
    state: CSABankGrowthState[CandidateT],
    bank: Bank[CandidateT],
    diversity_metric: DiversityMetric[CandidateT],
    score_model_state: CSAScoreModelState[CandidateT],
    distance_cutoff: float | None,
    minimum_distance_cutoff: float | None,
) -> CSABankGrowthState[CandidateT]:
    """Return the next generation state after one committed batch.

    Parameters
    ----------
    state : CSABankGrowthState[CandidateT]
        Current bank-growth state.
    bank : Bank[CandidateT]
        Bank snapshot after the committed batch.
    diversity_metric : DiversityMetric[CandidateT]
        Diversity metric used when rescoring the bank.
    score_model_state : CSAScoreModelState[CandidateT]
        Score-model runtime state used to rescore the bank.
    distance_cutoff : float | None
        Active CSA distance cutoff, if initialized.
    minimum_distance_cutoff : float | None
        Optional cutoff floor.

    Returns
    -------
    CSABankGrowthState[CandidateT]
        Next generation bank-growth state.
    """
    if not state.enabled:
        return CSABankGrowthState[CandidateT](
            policy=state.policy,
            active_energy_gap_limit=state.active_energy_gap_limit,
            generation_growth_count=0,
        )

    next_energy_gap_limit = state.active_energy_gap_limit
    mode = state.policy.energy_gap_update_mode
    if mode == "fixed":
        next_energy_gap_limit = state.policy.initial_energy_gap_limit
    elif mode == "max_score_ratio":
        scored_bank, _ = score_model_state.score_bank(
            entries=bank.entries,
            diversity_metric=diversity_metric,
            distance_cutoff=0.0 if distance_cutoff is None else distance_cutoff,
            minimum_distance_cutoff=minimum_distance_cutoff,
            masked_entry_indices=frozenset(),
        )
        next_energy_gap_limit = (
            max(scored_bank.shaped_scores) - min(scored_bank.shaped_scores)
        ) * state.policy.energy_gap_update_factor
    elif (
        mode == "multiplicative_decay"
        and minimum_distance_cutoff is not None
        and minimum_pairwise_distance(
            entries=bank.entries,
            diversity_metric=diversity_metric,
        )
        > minimum_distance_cutoff
    ):
        next_energy_gap_limit = (
            state.active_energy_gap_limit * state.policy.energy_gap_update_factor
        )

    return CSABankGrowthState[CandidateT](
        policy=state.policy,
        active_energy_gap_limit=next_energy_gap_limit,
        generation_growth_count=0,
    )


def reset_growth_state(
    state: CSABankGrowthState[CandidateT],
) -> CSABankGrowthState[CandidateT]:
    """Return the initial state implied by one growth policy.

    Parameters
    ----------
    state : CSABankGrowthState[CandidateT]
        Current bank-growth state whose policy should be preserved.

    Returns
    -------
    CSABankGrowthState[CandidateT]
        Reset growth state derived from the policy defaults.
    """
    return CSABankGrowthState[CandidateT](
        policy=state.policy,
        active_energy_gap_limit=state.policy.initial_energy_gap_limit,
        generation_growth_count=0,
    )


def minimum_pairwise_distance(
    *,
    entries: Sequence[BankEntry[CandidateT]],
    diversity_metric: DiversityMetric[CandidateT],
) -> float:
    """Return the minimum pairwise diversity distance across bank entries.

    Parameters
    ----------
    entries : Sequence[BankEntry[CandidateT]]
        Bank entries to compare.
    diversity_metric : DiversityMetric[CandidateT]
        Diversity metric used for pairwise distance computation.

    Returns
    -------
    float
        Minimum validated pairwise distance, or ``0.0`` when fewer than two
        entries are present.
    """
    if len(entries) < 2:
        return 0.0

    minimum_distance: float | None = None
    for left_index, left_entry in enumerate(entries[:-1]):
        for right_entry in entries[left_index + 1 :]:
            distance = require_valid_distance(
                diversity_metric.distance(
                    left_entry.candidate,
                    right_entry.candidate,
                )
            )
            if minimum_distance is None or distance < minimum_distance:
                minimum_distance = distance

    if minimum_distance is None:
        return 0.0

    return minimum_distance
