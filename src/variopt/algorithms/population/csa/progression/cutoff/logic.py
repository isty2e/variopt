"""CSA cutoff runtime-state transition logic."""

from ..state import CSAProgressionState
from .policy import CSACutoffSchedule


def initialize_cutoff_state(
    *,
    state: CSAProgressionState,
    schedule: CSACutoffSchedule,
    average_distance: float | None,
    score_gap: float | None,
) -> CSAProgressionState:
    """Initialize cutoff runtime from an average pairwise distance.

    Parameters
    ----------
    state : CSAProgressionState
        Current progression state.
    schedule : CSACutoffSchedule
        Cutoff schedule used to derive the initial cutoff pair.
    average_distance : float | None
        Average pairwise bank distance used to seed the cutoff when the
        schedule needs inferred cutoffs.
    score_gap : float | None
        Optional current score-gap observation.

    Returns
    -------
    CSAProgressionState
        Progression state with initialized cutoff runtime.
    """
    if state.cutoff_is_initialized:
        return state

    distance_cutoff, minimum_distance_cutoff = schedule.resolve_initial_cutoffs(
        average_distance=average_distance,
    )
    return state.initialize_cutoff(
        distance_cutoff=distance_cutoff,
        minimum_distance_cutoff=minimum_distance_cutoff,
        previous_score_gap=score_gap,
    )


def advance_cutoff_state(
    *,
    state: CSAProgressionState,
    schedule: CSACutoffSchedule,
    score_gap: float | None,
    unused_entry_count: int,
) -> tuple[CSAProgressionState, bool]:
    """Advance cutoff runtime by one CSA iteration.

    Parameters
    ----------
    state : CSAProgressionState
        Current progression state with initialized cutoff runtime.
    schedule : CSACutoffSchedule
        Cutoff schedule controlling decay, recovery, and cycle increments.
    score_gap : float | None
        Current score-gap observation.
    unused_entry_count : int
        Number of unused bank entries used by cycle-increment logic.

    Returns
    -------
    tuple[CSAProgressionState, bool]
        Advanced progression state and whether the cutoff logic incremented the
        cycle count.
    """
    assert state.distance_cutoff is not None
    assert state.minimum_distance_cutoff is not None
    assert state.cutoff_recover_limit is not None

    next_distance_cutoff = schedule.reduce(
        distance_cutoff=state.distance_cutoff,
        minimum_distance_cutoff=state.minimum_distance_cutoff,
    )
    next_recover_limit = state.cutoff_recover_limit
    if (
        schedule.should_recover(
            previous_score_gap=state.previous_score_gap,
            current_score_gap=score_gap,
        )
        and next_recover_limit >= state.distance_cutoff
    ):
        next_recover_limit = state.distance_cutoff
        next_distance_cutoff = schedule.recover(
            distance_cutoff=state.distance_cutoff,
        )

    if next_distance_cutoff <= state.minimum_distance_cutoff:
        next_distance_cutoff = state.minimum_distance_cutoff
        next_recover_limit = state.minimum_distance_cutoff

    cycle_increment = schedule.should_increment_cycle(
        unused_entry_count=unused_entry_count,
        cutoff_at_minimum=next_distance_cutoff <= state.minimum_distance_cutoff,
    )
    next_state = state.advance_iteration(
        distance_cutoff=next_distance_cutoff,
        cycle_increment=cycle_increment,
        cutoff_recover_limit=next_recover_limit,
        previous_score_gap=score_gap,
    )
    return next_state, cycle_increment
