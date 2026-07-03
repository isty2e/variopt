"""Kick and acceptance helpers for iterated structured local search."""

import numpy as np

from .....randomness import random_state_choice_indices_without_replacement
from .....spaces import CategoricalSpace, LeafPath, SpaceCandidateValue
from ..neighborhood import (
    BoundaryT,
    DiscreteLeafSpace,
    StructuredCandidateT,
    StructuredKickPolicy,
    discrete_leaf_neighbors,
)
from .prepared import PreparedStructuredLocalSearchRuntime
from .support import sample_neighbors_without_replacement


def sample_structured_kick_candidate(
    *,
    runtime: PreparedStructuredLocalSearchRuntime[BoundaryT, StructuredCandidateT],
    candidate: StructuredCandidateT,
    leaf_schedule: tuple[tuple[LeafPath, DiscreteLeafSpace], ...],
    kick_policy: StructuredKickPolicy,
    random_state: np.random.RandomState,
) -> StructuredCandidateT | None:
    """Sample one kicked candidate for iterated local search.

    Parameters
    ----------
    runtime : PreparedStructuredLocalSearchRuntime[BoundaryT, StructuredCandidateT]
        Prepared runtime exposing the structured search space.
    candidate : StructuredCandidateT
        Incumbent candidate to kick.
    leaf_schedule : tuple[tuple[LeafPath, DiscreteLeafSpace], ...]
        Ordered editable leaves eligible for kick moves.
    kick_policy : StructuredKickPolicy
        Kick configuration controlling how many leaves may change and how many
        categorical alternatives may be considered.
    random_state : np.random.RandomState
        Random state used for leaf and replacement sampling.

    Returns
    -------
    StructuredCandidateT | None
        Kicked candidate, or ``None`` when no admissible kick exists.
    """
    eligible_leaf_moves: list[tuple[LeafPath, tuple[SpaceCandidateValue, ...]]] = []
    space = runtime.neighborhood.space
    space.validate(candidate)
    for path, leaf_space in leaf_schedule:
        current_leaf_value = space.leaf_value_at_validated_path(
            candidate,
            path,
        )
        leaf_neighbors = discrete_leaf_neighbors(leaf_space, current_leaf_value)
        if (
            kick_policy.max_categorical_alternatives_per_leaf is not None
            and isinstance(leaf_space, CategoricalSpace)
        ):
            leaf_neighbors = sample_neighbors_without_replacement(
                neighbors=leaf_neighbors,
                random_state=random_state,
                max_samples=kick_policy.max_categorical_alternatives_per_leaf,
            )
        if len(leaf_neighbors) == 0:
            continue
        eligible_leaf_moves.append((path, leaf_neighbors))

    if len(eligible_leaf_moves) < kick_policy.kick_leaf_count:
        return None

    selected_indices = random_state_choice_indices_without_replacement(
        random_state,
        population_size=len(eligible_leaf_moves),
        count=kick_policy.kick_leaf_count,
    )
    replacements: dict[LeafPath, SpaceCandidateValue] = {}
    for index in selected_indices:
        path, leaf_neighbors = eligible_leaf_moves[index]
        replacement_index = random_state_choice_indices_without_replacement(
            random_state,
            population_size=len(leaf_neighbors),
            count=1,
        )[0]
        replacements[path] = leaf_neighbors[replacement_index]

    return space.replace_leaf_values_in_validated_candidate(candidate, replacements)


def accepts_strict_improvement(
    *,
    incumbent_score: float,
    candidate_score: float,
) -> bool:
    """Return whether a locally improved candidate strictly beats the incumbent.

    Parameters
    ----------
    incumbent_score : float
        Score of the incumbent candidate.
    candidate_score : float
        Score of the kicked or improved candidate.

    Returns
    -------
    bool
        Whether ``candidate_score`` is strictly better than ``incumbent_score``.
    """
    return candidate_score < incumbent_score
