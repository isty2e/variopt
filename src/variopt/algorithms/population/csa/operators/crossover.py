"""Structured crossover kernels for built-in CSA operators."""

from typing import TypeVar

import numpy as np

from .....spaces import StructuredSearchSpace
from .....spaces.types import SpaceCandidateValue
from .editing import (
    choose_paths_without_replacement,
    sample_exchange_count,
)

BoundaryT = TypeVar("BoundaryT")
CandidateT = TypeVar("CandidateT", bound=SpaceCandidateValue)


def uniform_crossover(
    *,
    space: StructuredSearchSpace[BoundaryT, CandidateT],
    primary_parent: CandidateT,
    partner_parent: CandidateT,
    max_exchange_fraction: float,
    random_state: np.random.RandomState,
) -> CandidateT:
    """Copy a sampled subset of partner leaves into the primary parent.

    Parameters
    ----------
    space : StructuredSearchSpace[BoundaryT, CandidateT]
        Structured search space that defines editable leaves.
    primary_parent : CandidateT
        Parent that contributes the base candidate structure.
    partner_parent : CandidateT
        Parent that donates replacement leaf values.
    max_exchange_fraction : float
        Maximum fraction of active leaves that may be copied from the partner.
    random_state : np.random.RandomState
        Random state used for leaf selection.

    Returns
    -------
    CandidateT
        Child candidate formed by replacing selected primary-parent leaves.

    Raises
    ------
    ValueError
        If the parents do not expose matching active leaf topology.
    """
    space.validate(primary_parent)
    space.validate(partner_parent)
    editable_paths = space.active_leaf_paths_for_validated_candidate(primary_parent)
    partner_editable_paths = space.active_leaf_paths_for_validated_candidate(
        partner_parent,
    )
    if editable_paths != partner_editable_paths:
        msg = "uniform crossover requires parents with matching active topology"
        raise ValueError(msg)

    exchange_count = sample_exchange_count(
        leaf_count=len(editable_paths),
        max_exchange_fraction=max_exchange_fraction,
        random_state=random_state,
    )
    selected_paths = choose_paths_without_replacement(
        editable_paths,
        exchange_count,
        random_state,
    )
    replacements = {
        path: space.leaf_value_at_validated_path(partner_parent, path)
        for path in selected_paths
    }
    return space.replace_leaf_values_in_validated_candidate(
        primary_parent,
        replacements,
    )
