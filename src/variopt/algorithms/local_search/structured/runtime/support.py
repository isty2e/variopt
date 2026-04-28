"""Support helpers shared by structured local-search runtime modules."""

import numpy as np

from .....randomness import random_state_choice_indices_without_replacement
from .....spaces import SpaceCandidateValue


def sample_neighbors_without_replacement(
    *,
    neighbors: tuple[SpaceCandidateValue, ...],
    random_state: np.random.RandomState,
    max_samples: int,
) -> tuple[SpaceCandidateValue, ...]:
    """Sample a bounded neighbor subset while preserving declaration order.

    Parameters
    ----------
    neighbors : tuple[SpaceCandidateValue, ...]
        Canonical neighbor candidates to subsample.
    random_state : np.random.RandomState
        Random state used for sampling.
    max_samples : int
        Maximum number of neighbors to keep.

    Returns
    -------
    tuple[SpaceCandidateValue, ...]
        Sampled neighbor subset in original declaration order.
    """
    if len(neighbors) <= max_samples:
        return neighbors

    selected_indices = random_state_choice_indices_without_replacement(
        random_state,
        population_size=len(neighbors),
        count=max_samples,
    )
    return tuple(neighbors[index] for index in sorted(selected_indices))
