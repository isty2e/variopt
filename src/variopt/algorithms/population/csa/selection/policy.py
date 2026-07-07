"""Private seed and partner selection policies for CSA."""

from collections.abc import Callable

import numpy as np

from .....distance import require_valid_distance
from .....randomness import (
    random_state_choice_index,
    random_state_choice_indices_without_replacement,
)
from .....typevars import CandidateT
from ..banking.bank import BankEntry
from .state import SeedSelectionState

IndexDistance = Callable[[int, int], float]
VALID_RANDOM_SEED_MODES = frozenset({0, 1, 2, 3})
EMPTY_MASK: frozenset[int] = frozenset()


def validate_random_seed_mode(random_seed_mode: int) -> None:
    """Validate a legacy-compatible random-seed mode.

    Parameters
    ----------
    random_seed_mode : int
        Legacy-compatible seed-selection mode identifier.

    Raises
    ------
    ValueError
        Raised when the mode is outside the supported CSA range.
    """
    if random_seed_mode not in VALID_RANDOM_SEED_MODES:
        msg = "random_seed_mode must be one of 0, 1, 2, or 3"
        raise ValueError(msg)


def prepare_seed_batch(
    *,
    current_state: SeedSelectionState,
    entries: tuple[BankEntry[CandidateT], ...],
    seed_count: int,
    random_seed_mode: int,
    masked_seed_indices: frozenset[int] = EMPTY_MASK,
    distance_between_indices: IndexDistance,
    random_state: np.random.RandomState,
) -> SeedSelectionState:
    """Prepare the next active CSA seed batch.

    Parameters
    ----------
    current_state : SeedSelectionState
        Current seed-selection state.
    entries : tuple[BankEntry[CandidateT], ...]
        Bank entries from which seeds are selected.
    seed_count : int
        Requested number of active seeds.
    random_seed_mode : int
        Legacy-compatible seed-selection mode identifier.
    masked_seed_indices : frozenset[int], default=EMPTY_MASK
        Entry indices that must not be chosen as active seeds.
    distance_between_indices : IndexDistance
        Distance callback used by diversity-aware seed modes.
    random_state : numpy.random.RandomState
        Random-state instance used for stochastic selection.

    Returns
    -------
    SeedSelectionState
        Updated seed-selection state with an activated batch.

    Raises
    ------
    ValueError
        Raised when ``seed_count`` is invalid or the bank is empty.
    """
    if seed_count <= 0:
        msg = "seed_count must be positive"
        raise ValueError(msg)

    if len(entries) == 0:
        msg = "cannot prepare a seed batch from an empty bank"
        raise ValueError(msg)

    validate_random_seed_mode(random_seed_mode)

    entry_count = len(entries)
    target_seed_count = min(seed_count, entry_count)
    unmasked_indices = tuple(
        index for index in range(entry_count) if index not in masked_seed_indices
    )
    used_entry_indices: frozenset[int] = frozenset(
        index for index in current_state.used_entry_indices if 0 <= index < entry_count
    )
    bank_status = current_state.resize_bank_status(entry_count=entry_count)

    if random_seed_mode == 2:
        selected_seed_indices = select_random_indices(
            candidate_indices=unmasked_indices,
            selection_count=target_seed_count,
            random_state=random_state,
        )
        return current_state.activate_seed_batch(
            selected_seed_indices=selected_seed_indices,
            bank_status=mark_selected(
                bank_status=bank_status, selected_indices=selected_seed_indices
            ),
            used_entry_indices=used_entry_indices.union(selected_seed_indices),
        )

    available_indices = tuple(
        index
        for index, is_used in enumerate(bank_status)
        if index not in masked_seed_indices and not is_used
    )
    if not available_indices:
        used_entry_indices = frozenset()
        bank_status = tuple(False for _ in range(entry_count))
        available_indices = unmasked_indices

    selected_indices, remaining_indices = select_initial_seed_pool(
        entries=entries,
        unmasked_indices=unmasked_indices,
        available_indices=available_indices,
        target_seed_count=target_seed_count,
        random_seed_mode=random_seed_mode,
        distance_between_indices=distance_between_indices,
        random_state=random_state,
    )
    if len(selected_indices) < target_seed_count:
        selected_indices = extend_seed_selection(
            entries=entries,
            selected_indices=selected_indices,
            remaining_indices=remaining_indices,
            target_seed_count=target_seed_count,
            random_seed_mode=random_seed_mode,
            distance_between_indices=distance_between_indices,
            random_state=random_state,
        )

    selected_seed_indices = tuple(selected_indices)
    return current_state.activate_seed_batch(
        selected_seed_indices=selected_seed_indices,
        bank_status=mark_selected(
            bank_status=bank_status, selected_indices=selected_seed_indices
        ),
        used_entry_indices=used_entry_indices.union(selected_seed_indices),
    )


def select_partner_indices(
    *,
    entries: tuple[BankEntry[CandidateT], ...],
    seed_index: int,
    partner_count: int,
    partner_mask: frozenset[int] = EMPTY_MASK,
    distance_between_indices: IndexDistance,
    weighted_partner_selection: bool,
    random_state: np.random.RandomState,
) -> tuple[int, ...]:
    """Select partner indices for one active seed.

    Parameters
    ----------
    entries : tuple[BankEntry[CandidateT], ...]
        Bank entries from which partners are selected.
    seed_index : int
        Active seed index whose partners are requested.
    partner_count : int
        Number of partners to select.
    partner_mask : frozenset[int], default=EMPTY_MASK
        Entry indices that must not be selected as partners.
    distance_between_indices : IndexDistance
        Distance callback used by weighted partner selection.
    weighted_partner_selection : bool
        Whether to bias partner selection by seed distance.
    random_state : numpy.random.RandomState
        Random-state instance used for stochastic selection.

    Returns
    -------
    tuple[int, ...]
        Selected partner indices.

    Raises
    ------
    ValueError
        Raised when the bank does not contain enough eligible partners.
    """
    if partner_count <= 0:
        return ()

    entry_count = len(entries)
    if (
        not weighted_partner_selection
        and not partner_mask
        and 0 <= seed_index < entry_count
    ):
        available_count = entry_count - 1
        if available_count < partner_count:
            msg = "bank does not contain enough partners for the requested arity"
            raise ValueError(msg)

        # Preserve the materialized-list RNG trajectory by drawing positions
        # from the same ``entry_count - 1`` population, then remapping over the
        # skipped seed index.
        selected_positions = random_state_choice_indices_without_replacement(
            random_state,
            available_count,
            partner_count,
        )
        return tuple(
            position if position < seed_index else position + 1
            for position in selected_positions
        )

    available_indices = [
        index
        for index in range(entry_count)
        if index != seed_index and index not in partner_mask
    ]
    if len(available_indices) < partner_count:
        msg = "bank does not contain enough partners for the requested arity"
        raise ValueError(msg)

    if not weighted_partner_selection:
        return select_random_indices(
            candidate_indices=tuple(available_indices),
            selection_count=partner_count,
            random_state=random_state,
        )

    selected_indices: list[int] = []
    remaining_indices = list(available_indices)
    while len(selected_indices) < partner_count:
        next_index = select_weighted_partner_index(
            seed_index=seed_index,
            remaining_indices=tuple(remaining_indices),
            distance_between_indices=distance_between_indices,
            random_state=random_state,
        )

        remaining_indices.remove(next_index)
        selected_indices.append(next_index)

    return tuple(selected_indices)


def select_initial_seed_pool(
    *,
    entries: tuple[BankEntry[CandidateT], ...],
    unmasked_indices: tuple[int, ...],
    available_indices: tuple[int, ...],
    target_seed_count: int,
    random_seed_mode: int,
    distance_between_indices: IndexDistance,
    random_state: np.random.RandomState,
) -> tuple[list[int], list[int]]:
    """Choose the initial seed pool before optional extension.

    Parameters
    ----------
    entries : tuple[BankEntry[CandidateT], ...]
        Bank entries from which seeds are selected.
    unmasked_indices : tuple[int, ...]
        All indices eligible for selection after masking.
    available_indices : tuple[int, ...]
        Currently unused indices preferred for selection.
    target_seed_count : int
        Requested number of seeds.
    random_seed_mode : int
        Legacy-compatible seed-selection mode identifier.
    distance_between_indices : IndexDistance
        Distance callback used by diversity-aware seed modes.
    random_state : numpy.random.RandomState
        Random-state instance used for stochastic selection.

    Returns
    -------
    tuple[list[int], list[int]]
        Selected indices together with remaining eligible indices.
    """
    selected_indices: list[int] = []
    remaining_indices = list(available_indices)

    if len(remaining_indices) <= target_seed_count:
        selected_indices.extend(remaining_indices)
        selected_set = set(selected_indices)
        remaining_indices = [
            index for index in unmasked_indices if index not in selected_set
        ]
        return selected_indices, remaining_indices

    first_seed_index = pick_next_seed_index(
        entries=entries,
        selected_indices=tuple(selected_indices),
        remaining_indices=tuple(remaining_indices),
        random_seed_mode=random_seed_mode,
        distance_between_indices=distance_between_indices,
        random_state=random_state,
    )
    remaining_indices.remove(first_seed_index)
    selected_indices.append(first_seed_index)
    return selected_indices, remaining_indices


def extend_seed_selection(
    *,
    entries: tuple[BankEntry[CandidateT], ...],
    selected_indices: list[int],
    remaining_indices: list[int],
    target_seed_count: int,
    random_seed_mode: int,
    distance_between_indices: IndexDistance,
    random_state: np.random.RandomState,
) -> list[int]:
    """Extend a partially selected seed set to the target size.

    Parameters
    ----------
    entries : tuple[BankEntry[CandidateT], ...]
        Bank entries from which seeds are selected.
    selected_indices : list[int]
        Current selected seed indices.
    remaining_indices : list[int]
        Remaining eligible indices.
    target_seed_count : int
        Requested number of seeds.
    random_seed_mode : int
        Legacy-compatible seed-selection mode identifier.
    distance_between_indices : IndexDistance
        Distance callback used by diversity-aware seed modes.
    random_state : numpy.random.RandomState
        Random-state instance used for stochastic selection.

    Returns
    -------
    list[int]
        Extended selected seed indices.
    """
    while remaining_indices and len(selected_indices) < target_seed_count:
        next_index = pick_next_seed_index(
            entries=entries,
            selected_indices=tuple(selected_indices),
            remaining_indices=tuple(remaining_indices),
            random_seed_mode=random_seed_mode,
            distance_between_indices=distance_between_indices,
            random_state=random_state,
        )
        remaining_indices.remove(next_index)
        selected_indices.append(next_index)

    return selected_indices


def pick_next_seed_index(
    *,
    entries: tuple[BankEntry[CandidateT], ...],
    selected_indices: tuple[int, ...],
    remaining_indices: tuple[int, ...],
    random_seed_mode: int,
    distance_between_indices: IndexDistance,
    random_state: np.random.RandomState,
) -> int:
    """Pick the next seed index under the configured seed mode.

    Parameters
    ----------
    entries : tuple[BankEntry[CandidateT], ...]
        Bank entries from which seeds are selected.
    selected_indices : tuple[int, ...]
        Seed indices already selected for the active batch.
    remaining_indices : tuple[int, ...]
        Candidate indices still eligible for selection.
    random_seed_mode : int
        Legacy-compatible seed-selection mode identifier.
    distance_between_indices : IndexDistance
        Distance callback used by diversity-aware seed modes.
    random_state : numpy.random.RandomState
        Random-state instance used for stochastic selection.

    Returns
    -------
    int
        Selected next seed index.
    """
    if random_seed_mode in {1, 2}:
        return select_random_indices(
            candidate_indices=remaining_indices,
            selection_count=1,
            random_state=random_state,
        )[0]

    if random_seed_mode == 3:
        return pick_lowest_value_index(
            entries=entries,
            candidate_indices=remaining_indices,
        )

    if not selected_indices:
        return select_random_indices(
            candidate_indices=remaining_indices,
            selection_count=1,
            random_state=random_state,
        )[0]

    return pick_diverse_low_value_seed(
        entries=entries,
        selected_indices=selected_indices,
        remaining_indices=remaining_indices,
        distance_between_indices=distance_between_indices,
    )


def select_random_indices(
    *,
    candidate_indices: tuple[int, ...],
    selection_count: int,
    random_state: np.random.RandomState,
) -> tuple[int, ...]:
    """Select random candidate indices without replacement.

    Parameters
    ----------
    candidate_indices : tuple[int, ...]
        Candidate indices eligible for selection.
    selection_count : int
        Number of indices to select.
    random_state : numpy.random.RandomState
        Random-state instance used for sampling.

    Returns
    -------
    tuple[int, ...]
        Selected candidate indices.
    """
    selected_positions = random_state_choice_indices_without_replacement(
        random_state,
        len(candidate_indices),
        selection_count,
    )
    return tuple(candidate_indices[position] for position in selected_positions)


def pick_lowest_value_index(
    *,
    entries: tuple[BankEntry[CandidateT], ...],
    candidate_indices: tuple[int, ...],
) -> int:
    """Pick the candidate index with the smallest objective value.

    Parameters
    ----------
    entries : tuple[BankEntry[CandidateT], ...]
        Bank entries whose values are compared.
    candidate_indices : tuple[int, ...]
        Candidate indices eligible for selection.

    Returns
    -------
    int
        Index with the lowest objective value.
    """
    return min(candidate_indices, key=lambda index: entries[index].value)


def pick_diverse_low_value_seed(
    *,
    entries: tuple[BankEntry[CandidateT], ...],
    selected_indices: tuple[int, ...],
    remaining_indices: tuple[int, ...],
    distance_between_indices: IndexDistance,
) -> int:
    """Pick a low-value seed among candidates that are sufficiently diverse.

    Parameters
    ----------
    entries : tuple[BankEntry[CandidateT], ...]
        Bank entries whose values are compared.
    selected_indices : tuple[int, ...]
        Seed indices already selected for the active batch.
    remaining_indices : tuple[int, ...]
        Candidate indices still eligible for selection.
    distance_between_indices : IndexDistance
        Distance callback used to score diversity from the selected set.

    Returns
    -------
    int
        Selected seed index.
    """
    scored_indices: list[tuple[int, float]] = []
    total_distance_sum = 0.0

    for index in remaining_indices:
        distance_sum = 0.0
        for selected_index in selected_indices:
            distance_sum += distance_between_indices(selected_index, index)

        scored_indices.append((index, distance_sum))
        total_distance_sum += distance_sum

    average_distance_sum = total_distance_sum / float(len(scored_indices))
    min_score_index = 0
    min_score = entries[remaining_indices[0]].value

    for offset, (index, distance_sum) in enumerate(scored_indices[1:], start=1):
        if distance_sum < average_distance_sum:
            continue

        score = entries[index].value
        if score < min_score:
            min_score_index = offset
            min_score = score

    return scored_indices[min_score_index][0]


def select_weighted_partner_index(
    *,
    seed_index: int,
    remaining_indices: tuple[int, ...],
    distance_between_indices: IndexDistance,
    random_state: np.random.RandomState,
) -> int:
    """Pick one partner index using distance-derived weights.

    Parameters
    ----------
    seed_index : int
        Active seed index whose partner is requested.
    remaining_indices : tuple[int, ...]
        Candidate partner indices still eligible for selection.
    distance_between_indices : IndexDistance
        Distance callback used to derive partner weights.
    random_state : numpy.random.RandomState
        Random-state instance used for stochastic selection.

    Returns
    -------
    int
        Selected partner index.
    """
    distances = [
        require_valid_distance(distance_between_indices(seed_index, index))
        for index in remaining_indices
    ]
    zero_distance_indices = tuple(
        index
        for index, distance in zip(remaining_indices, distances, strict=True)
        if distance == 0.0
    )
    if zero_distance_indices:
        return select_random_indices(
            candidate_indices=zero_distance_indices,
            selection_count=1,
            random_state=random_state,
        )[0]

    inverse_distances = [1.0 / distance for distance in distances]
    inverse_distance_sum = sum(inverse_distances)
    weights = [
        inverse_distance / inverse_distance_sum
        for inverse_distance in inverse_distances
    ]
    selected_position = random_state_choice_index(
        random_state,
        len(remaining_indices),
        weights=weights,
    )
    return remaining_indices[selected_position]


def mark_selected(
    *,
    bank_status: tuple[bool, ...],
    selected_indices: tuple[int, ...],
) -> tuple[bool, ...]:
    """Mark selected indices in a bank-status tuple.

    Parameters
    ----------
    bank_status : tuple[bool, ...]
        Current per-entry selection markers.
    selected_indices : tuple[int, ...]
        Indices that should be marked as selected.

    Returns
    -------
    tuple[bool, ...]
        Updated bank-status tuple.
    """
    next_bank_status = list(bank_status)
    for index in selected_indices:
        next_bank_status[index] = True
    return tuple(next_bank_status)
